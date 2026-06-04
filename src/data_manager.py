import time

import pandas as pd
from loguru import logger

from src.config import PreprocessingSettings
from src.constants import Columns


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


# Columns that must NOT be value-coerced by standardize_columns_and_values (audit #19):
# date columns and opaque identifier keys. Lowercasing/underscoring/categorizing these would
# mangle string dates (e.g. "2024-01-01 00:00:00" -> "2024-01-01_00:00:00") and case-sensitive
# IDs. On the current data these load as datetime64/numeric (not object), so excluding them is a
# no-op; the protection guards against future files delivering them as strings.
# NOTE: `se_decision_id` is intentionally NOT protected — it is a categorical filter ("ok"/"rv"/
# "ko") whose downstream comparisons require the lowercased form.
PROTECTED_FROM_COERCION = frozenset({"mis_date", "authorization_id"})


def load_data(df_path: str, encoding: str = "latin-1") -> pd.DataFrame:
    """Load data from SAS file (encoding configurable via settings.sas_encoding)."""
    df = pd.read_sas(df_path, format="sas7bdat", encoding=encoding)
    return df


def standardize_columns_and_values(data: pd.DataFrame) -> pd.DataFrame:
    """Lowercase/underscore column names and normalize categorical *values* in place.

    Column names are always normalized (lowercase + spaces→underscores). Object/string/category
    *values* are lowercased + underscored + cast to ``category`` — except for protected date and
    identifier columns (``PROTECTED_FROM_COERCION`` or any ``*_date`` name), which are left verbatim
    so dates and case-sensitive keys are never mangled (audit #19). Shared by both load paths
    (``load_and_prepare_data`` and ``run_batch.load_and_standardize_data``) so they normalize
    identically.
    """
    data.columns = data.columns.str.lower().str.replace(" ", "_")
    for col in data.select_dtypes(include=["object", "category", "string"]).columns:
        if col in PROTECTED_FROM_COERCION or col.endswith("_date"):
            continue
        data[col] = data[col].astype("string").str.lower().str.replace(" ", "_").astype("category")
    return data


def validate_data_columns(data: pd.DataFrame, required_columns: list[str], context: str = "data") -> list[str]:
    """
    Validate that required columns exist in the DataFrame.

    Args:
        data: DataFrame to validate
        required_columns: List of required column names
        context: Context for error messages

    Returns:
        List of missing columns (empty if all present)

    Raises:
        DataValidationError: If any required columns are missing
    """
    # Case-sensitive comparison — columns should already be standardized to
    # lowercase by load_and_prepare_data before this is called.
    data_columns = set(data.columns)
    missing = [col for col in required_columns if col not in data_columns]

    if missing:
        raise DataValidationError(f"Missing required columns in {context}: {missing}")

    return []


def validate_data_not_empty(data: pd.DataFrame, context: str = "data") -> None:
    """
    Validate that DataFrame is not empty.

    Args:
        data: DataFrame to validate
        context: Context for error messages

    Raises:
        DataValidationError: If DataFrame is empty
    """
    if data.empty:
        raise DataValidationError(f"{context} is empty")


def load_and_prepare_data(
    settings: PreprocessingSettings, preloaded_data: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, PreprocessingSettings]:
    """
    Load data from file or use preloaded data, standardize columns and validate.

    Args:
        settings: Configuration settings object
        preloaded_data: Optional pre-loaded and standardized DataFrame

    Returns:
        Prepared DataFrame

    Raises:
        DataValidationError: If data validation fails
        FileNotFoundError: If data file is not found
    """
    t0 = time.perf_counter()

    if preloaded_data is not None:
        data = preloaded_data.copy()
        logger.debug(f"Using pre-loaded data: {data.shape[0]:,} rows x {data.shape[1]} columns")
    else:
        data_path = settings.data_path
        data = load_data(data_path, encoding=getattr(settings, "sas_encoding", "latin-1"))
        validate_data_not_empty(data, "Input data")

        # Standardize column names and categorical values (only when loading fresh data).
        # Dates/IDs are protected from value coercion (audit #19).
        data = standardize_columns_and_values(data)
        logger.debug("Column names and categorical values standardized")

    # Validate required columns exist after standardization.
    # H3 columns are optional — downstream code already handles their absence
    # (e.g. mr_pipeline checks `"todu_30ever_h3_boo" in df.columns`).
    h3_optional = {Columns.TODU_30EVER_H3, Columns.TODU_AMT_PILE_H3}
    all_cols = list(dict.fromkeys(settings.keep_vars + settings.indicators))  # deduplicate, preserve order
    required_cols = [c for c in all_cols if c not in h3_optional]
    optional_cols = [c for c in all_cols if c in h3_optional]

    validate_data_columns(data, required_cols, "input data")

    missing_optional = [c for c in optional_cols if c not in data.columns]
    if missing_optional:
        logger.warning(f"Optional H3 columns not found in data (will be skipped): {missing_optional}")
        # Remove missing optional columns from settings so all downstream code
        # receives a clean list without columns that don't exist in the data.
        # Use new lists to avoid mutating the caller's settings object in-place.
        settings = settings.model_copy(
            update={
                "keep_vars": [c for c in settings.keep_vars if c not in missing_optional],
                "indicators": [c for c in settings.indicators if c not in missing_optional],
            }
        )

    # Schema validation: check types, value ranges, and categorical constraints
    from src.schema import validate_raw_data

    validate_raw_data(data, raise_on_error=True)

    elapsed = time.perf_counter() - t0
    source = "preloaded" if preloaded_data is not None else settings.data_path
    logger.info(f"Data ready | {data.shape[0]:,} rows x {data.shape[1]} cols | source={source} | {elapsed:.1f}s")

    return data, settings
