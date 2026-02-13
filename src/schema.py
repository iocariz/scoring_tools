"""
Data schema validation using Pandera.

Defines schemas for validating raw input data before pipeline processing.
Catches type mismatches, invalid values, and constraint violations early
to prevent silent data corruption.
"""

import pandera.pandas as pa
from loguru import logger

from src.constants import StatusName


def _valid_status_values() -> set[str]:
    """Return the set of valid status_name values."""
    return {s.value for s in StatusName}


# ---------------------------------------------------------------------------
# Core schema: validates columns that must always be present and correct
# ---------------------------------------------------------------------------
RAW_DATA_SCHEMA = pa.DataFrameSchema(
    columns={
        "status_name": pa.Column(
            dtype="category",
            checks=pa.Check.isin(_valid_status_values()),
            nullable=False,
        ),
        "segment_cut_off": pa.Column(
            nullable=False,
        ),
        "mis_date": pa.Column(
            nullable=False,
        ),
        # Risk indicators — nullable because demand/rejected records won't have them
        "todu_30ever_h6": pa.Column(
            dtype="float64",
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
            required=False,
        ),
        "todu_amt_pile_h6": pa.Column(
            dtype="float64",
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
            required=False,
        ),
        "oa_amt_h0": pa.Column(
            dtype="float64",
            checks=pa.Check.greater_than_or_equal_to(0),
            nullable=True,
            required=False,
        ),
    },
    # Don't reject columns that aren't in the schema
    strict=False,
    # Collect all errors instead of stopping at the first one
    ordered=False,
)


def validate_raw_data(data, *, raise_on_error: bool = True):
    """
    Validate a DataFrame against the raw data schema.

    Args:
        data: DataFrame to validate.
        raise_on_error: If True, raises DataValidationError on failure.
            If False, returns (is_valid, error_messages) tuple.

    Returns:
        The validated DataFrame if raise_on_error is True.
        Tuple of (bool, list[str]) if raise_on_error is False.

    Raises:
        DataValidationError: If validation fails and raise_on_error is True.
    """
    from src.data_manager import DataValidationError

    try:
        validated = RAW_DATA_SCHEMA.validate(data, lazy=True)
        logger.info("Schema validation passed")
        if raise_on_error:
            return validated
        return True, []
    except pa.errors.SchemaErrors as e:
        messages = []
        for _, row in e.failure_cases.iterrows():
            col = row.get("column", "unknown")
            check = row.get("check", "unknown")
            msg = f"Column '{col}': {check}"
            messages.append(msg)

        summary = f"Schema validation failed with {len(messages)} error(s): {'; '.join(messages[:5])}"
        if len(messages) > 5:
            summary += f" ... and {len(messages) - 5} more"

        logger.error(summary)

        if raise_on_error:
            raise DataValidationError(summary) from e
        return False, messages
