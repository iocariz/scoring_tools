import time

import pandas as pd
from loguru import logger

from src.config import PreprocessingSettings


class DataValidationError(Exception):
    """Raised when data validation fails."""

    pass


def load_data(df_path: str) -> pd.DataFrame:
    """Load data from SAS file."""
    df = pd.read_sas(df_path, format="sas7bdat", encoding="utf-8")
    return df


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
    # Normalize column names for comparison
    data_columns = set(data.columns.str.lower())
    missing = [col for col in required_columns if col.lower() not in data_columns]

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


def load_and_prepare_data(settings: PreprocessingSettings, preloaded_data: pd.DataFrame = None) -> pd.DataFrame:
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
        data = load_data(data_path)
        validate_data_not_empty(data, "Input data")

        # Standardize column names and categorical values (only when loading fresh data)
        data.columns = data.columns.str.lower().str.replace(" ", "_")
        for col in data.select_dtypes(include=["object", "category", "string"]).columns:
            data[col] = data[col].astype("string").str.lower().str.replace(" ", "_").astype("category")
        logger.debug("Column names and categorical values standardized")

    # Validate required columns exist after standardization
    required_cols = settings.keep_vars + settings.indicators
    validate_data_columns(data, required_cols, "input data")

    # Schema validation: check types, value ranges, and categorical constraints
    from src.schema import validate_raw_data

    validate_raw_data(data, raise_on_error=True)

    elapsed = time.perf_counter() - t0
    source = "preloaded" if preloaded_data is not None else settings.data_path
    logger.info(f"Data ready | {data.shape[0]:,} rows x {data.shape[1]} cols | source={source} | {elapsed:.1f}s")

    return data
