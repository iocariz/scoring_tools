"""
Utility functions for credit risk scoring and portfolio optimization.

This module provides core utility functions used throughout the scoring tools:
- Risk metric calculations (b2_ever_h6)
- Data optimization and memory management

"""

import os
from typing import overload

import numpy as np
import pandas as pd
from loguru import logger

from src.core.constants import DEFAULT_RISK_MULTIPLIER

# Cap parallel workers to avoid OOM on many-core servers.
# Override with SCORING_TOOLS_MAX_JOBS environment variable.
MAX_PARALLEL_JOBS = int(os.environ.get("SCORING_TOOLS_MAX_JOBS", min(4, os.cpu_count() or 4)))


# ---------------------------------------------------------------------------
# Supersegment resolution helpers
# ---------------------------------------------------------------------------


def resolve_modelling_supersegment(segment_config: dict) -> str | None:
    """Return the modelling supersegment for a segment.

    Resolution order: ``modelling_supersegment`` > ``supersegment`` > ``None``.
    """
    return segment_config.get("modelling_supersegment") or segment_config.get("supersegment")


def resolve_reporting_supersegment(segment_config: dict) -> str | None:
    """Return the reporting supersegment for a segment.

    Resolution order: ``reporting_supersegment`` > ``supersegment`` > ``None``.
    """
    return segment_config.get("reporting_supersegment") or segment_config.get("supersegment")


# Overloads narrow the return type for static checkers (todo #55).
# Runtime behaviour is unchanged — pandas broadcasting preserves Series
# when either input is a Series; scalar inputs return a float; ndarray
# inputs return an ndarray. The overloads let call sites that use
# ``.isna()`` / ``.fillna()`` type-check against a pd.Series result.
@overload
def calculate_b2_ever_h6(
    numerator: pd.Series,
    denominator: pd.Series | np.ndarray | float,
    multiplier: float = ...,
    as_percentage: bool = ...,
    decimals: int = ...,
) -> pd.Series: ...
@overload
def calculate_b2_ever_h6(
    numerator: np.ndarray | float,
    denominator: pd.Series,
    multiplier: float = ...,
    as_percentage: bool = ...,
    decimals: int = ...,
) -> pd.Series: ...
@overload
def calculate_b2_ever_h6(
    numerator: np.ndarray,
    denominator: np.ndarray | float,
    multiplier: float = ...,
    as_percentage: bool = ...,
    decimals: int = ...,
) -> np.ndarray: ...
@overload
def calculate_b2_ever_h6(
    numerator: float,
    denominator: float,
    multiplier: float = ...,
    as_percentage: bool = ...,
    decimals: int = ...,
) -> float: ...
def calculate_b2_ever_h6(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    as_percentage: bool = False,
    decimals: int = 6,
) -> pd.Series | np.ndarray | float:
    """
    Calculate the b2_ever_h6 risk metric.

    Formula: multiplier * numerator / denominator

    Args:
        numerator: todu_30ever_h6 values
        denominator: todu_amt_pile_h6 values
        multiplier: Risk multiplier (default: 7)
        as_percentage: If True, multiply result by 100
        decimals: Number of decimal places to round to (default 6 for
            intermediate precision; use 2 for final display values)

    Returns:
        Calculated b2_ever_h6 values, with division-by-zero handled as NaN.
        Callers that need 0 instead of NaN should apply np.nan_to_num() at the
        display/output boundary.

        Type contract (enforced by overloads and a small runtime wrap at
        the end of the body):
          - Any Series input → Series output (preserves the Series's index).
          - Both inputs scalar → float output (``.isna`` will NOT work;
            callers must use ``math.isnan`` / ``np.isnan``).
          - Otherwise (ndarray inputs) → ndarray output.
    """
    # Handle division by zero
    if isinstance(denominator, (pd.Series, np.ndarray)):
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
    else:
        safe_denominator = np.nan if denominator == 0 else denominator

    result = multiplier * numerator / safe_denominator

    if as_percentage:
        result = result * 100

    # Risk cannot be negative — clip to 0 (preserves NaN for missing cells)
    result = np.clip(result, 0, None)

    result = np.round(result, decimals)

    # Runtime wrap (todo #55 defense-in-depth): if either input is a Series
    # but the intermediate arithmetic degraded to an ndarray (e.g. via a
    # future change to np.where), restore the Series shape + index so
    # downstream ``.isna()`` / ``.fillna()`` calls stay valid.
    if isinstance(result, np.ndarray) and isinstance(numerator, pd.Series):
        result = pd.Series(result, index=numerator.index)
    elif isinstance(result, np.ndarray) and isinstance(denominator, pd.Series):
        result = pd.Series(result, index=denominator.index)

    return result


def calculate_todu_30ever_from_b2(
    b2_ever_h6: pd.Series | np.ndarray | float,
    todu_amt_pile_h6: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> pd.Series | np.ndarray | float:
    """
    Calculate todu_30ever_h6 from b2_ever_h6 and todu_amt_pile_h6.

    This is the inverse of calculate_b2_ever_h6:
        todu_30ever_h6 = b2_ever_h6 * todu_amt_pile_h6 / multiplier

    Args:
        b2_ever_h6: Risk metric values
        todu_amt_pile_h6: Exposure values
        multiplier: Risk multiplier (default: 7)

    Returns:
        Calculated todu_30ever_h6 values
    """
    return b2_ever_h6 * todu_amt_pile_h6 / multiplier


# Outcome pairs subject to the joint booked-completeness rule (audit F1, extends #41):
# a loan whose numerator OR denominator is unrealized (NaN) must contribute to NEITHER —
# NaN-skipping column sums otherwise admit its exposure with zero defaults, diluting risk.
# The HRI pairs (h_num_*/h_den_*) are deliberately NOT listed: their nulls are sparse by
# design (no default event / no accrual), not unrealized outcomes.
OUTCOME_COMPLETENESS_PAIRS: tuple[tuple[str, str], ...] = (
    ("todu_30ever_h6", "todu_amt_pile_h6"),
    ("todu_30ever_h3", "todu_amt_pile_h3"),
)


def mask_incomplete_outcome_pairs(
    df: pd.DataFrame,
    pairs: tuple[tuple[str, str], ...] = OUTCOME_COMPLETENESS_PAIRS,
) -> tuple[pd.DataFrame, int]:
    """Blank BOTH columns of each outcome pair where EITHER is NaN (joint completeness).

    One completeness rule for risk sums (the backtest's #41 rule, applied upstream):
    an incomplete outcome is excluded from that pair's numerator AND denominator, while
    production and every other column keep the loan. Pairs are masked independently
    (a partial H6 does not discard a complete H3). Returns ``(masked_copy, n_masked)``
    where ``n_masked`` counts rows blanked in at least one pair; when nothing needs
    masking the ORIGINAL frame is returned untouched (no copy).
    """
    masked_any = None
    out = df
    for num_col, den_col in pairs:
        if num_col not in df.columns or den_col not in df.columns:
            continue
        incomplete = df[num_col].isna() != df[den_col].isna()
        if not incomplete.any():
            continue
        if out is df:
            out = df.copy()
        out.loc[incomplete, [num_col, den_col]] = np.nan
        masked_any = incomplete if masked_any is None else (masked_any | incomplete)
    return out, int(masked_any.sum()) if masked_any is not None else 0


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate integer dtypes.

    Works on a copy to avoid mutating the input. Float64 columns are preserved
    to avoid precision loss on financial data.
    """
    df = df.copy()
    for col in df.columns:
        # Convert integer columns
        if df[col].dtype == "int64":
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= 0:
                if col_max <= 255:
                    df[col] = df[col].astype("uint8")
                elif col_max <= 65535:
                    df[col] = df[col].astype("uint16")
                elif col_max <= 4294967295:
                    df[col] = df[col].astype("uint32")
                # else: exceeds uint32 max — keep int64. Blindly casting to uint32
                # would silently WRAP large financial amounts (e.g. euro-cent sums
                # > 4.29e9) into tiny wrong values.
            else:
                if col_min >= -128 and col_max <= 127:
                    df[col] = df[col].astype("int8")
                elif col_min >= -32768 and col_max <= 32767:
                    df[col] = df[col].astype("int16")
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    df[col] = df[col].astype("int32")
                # else: exceeds int32 range — keep int64 (avoid silent overflow).

        # Float64 columns are intentionally preserved to avoid precision loss

    return df


def calculate_annual_coef(date_ini_book_obs: pd.Timestamp, date_fin_book_obs: pd.Timestamp) -> float:
    """Annualization coefficient for observation-window production: ``12 / n_month``.

    ``n_month`` is the number of monthly booking COHORTS in the window, counted from the
    calendar-month labels of the endpoints, inclusive: ``(Δyears·12 + Δmonths) + 1``. Production
    is aggregated per booking-month cohort, so this cohort count — NOT the elapsed day count — is
    the correct annualization basis: a window covering 12 month-labelled cohorts is one full year
    (coef 1.0) even though month-start to month-start spans only ~11 calendar months.

    This is correct only when both endpoints are MONTH-START markers (``day == 1``), i.e. the window
    is aligned to whole cohort months. A non-aligned window (mid-month endpoints, or a sub-month
    slice) is silently mis-annualized — e.g. a 20-day window ``day 1 → day 20`` still counts as 1
    cohort → coef 12, treating a partial month as a full one. We WARN on non-aligned endpoints
    rather than guess a fractional cohort count (the cohort basis is only defined for whole months);
    align both dates to a month start to silence it. Switching to elapsed days would be wrong — it
    would break the cohort basis and inflate production on the (correct) aligned windows.
    """
    if date_fin_book_obs < date_ini_book_obs:
        raise ValueError(
            f"date_fin_book_obs ({date_fin_book_obs.date()}) is before date_ini_book_obs ({date_ini_book_obs.date()})"
        )
    if date_ini_book_obs.day != 1 or date_fin_book_obs.day != 1:
        logger.warning(
            f"calculate_annual_coef: observation window {date_ini_book_obs.date()} -> {date_fin_book_obs.date()} "
            "is not month-start-aligned (an endpoint's day != 1). The coefficient counts whole monthly "
            "cohorts, so a non-aligned window mis-annualizes production — align both dates to a month "
            "start (day 1)."
        )
    n_month = (
        (date_fin_book_obs.year - date_ini_book_obs.year) * 12 + (date_fin_book_obs.month - date_ini_book_obs.month) + 1
    )
    annual_coef = 12 / n_month
    return annual_coef
