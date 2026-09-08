"""
Centralized constants for the scoring tools package.

This module defines all shared constants including:
- Column names used across the codebase
- Status values for filtering
- Risk calculation parameters
- Default configuration values

Using these constants instead of hardcoded strings improves:
- Maintainability: Change column names in one place
- Type safety: IDE autocompletion and typo detection
- Documentation: Clear overview of data schema expectations
"""

from enum import StrEnum
from typing import Final


# =============================================================================
# STATUS VALUES
# =============================================================================
class StatusName(StrEnum):
    """Status values for loan applications."""

    BOOKED = "booked"
    REJECTED = "rejected"
    CANCELED = "canceled"


class RejectReason(StrEnum):
    """Rejection reason codes."""

    OTHER = "08-other"
    SCORE = "09-score"


class SystemDecision(StrEnum):
    """Upstream scoring-system decision (``se_decision_id``): accept / review / reject."""

    OK = "ok"
    RV = "rv"
    KO = "ko"


# =============================================================================
# COLUMN NAMES - Core Identifiers
# =============================================================================
class Columns:
    """Standard column names used throughout the codebase."""

    # Status and filtering
    STATUS_NAME: Final[str] = "status_name"
    REJECT_REASON: Final[str] = "reject_reason"
    SE_DECISION_ID: Final[str] = "se_decision_id"

    # Date columns
    MIS_DATE: Final[str] = "mis_date"

    # Amount columns
    OA_AMT: Final[str] = "oa_amt"
    OA_AMT_H0: Final[str] = "oa_amt_h0"

    # Risk metric columns (H6 — 6-month horizon)
    TODU_30EVER_H6: Final[str] = "todu_30ever_h6"
    TODU_AMT_PILE_H6: Final[str] = "todu_amt_pile_h6"
    B2_EVER_H6: Final[str] = "b2_ever_h6"

    # Risk metric columns (H3 — 3-month horizon, complementary)
    TODU_30EVER_H3: Final[str] = "todu_30ever_h3"
    TODU_AMT_PILE_H3: Final[str] = "todu_amt_pile_h3"
    B2_EVER_H3: Final[str] = "b2_ever_h3"

    # Harmonized Risk Indicator (HRI) columns — hri = h_num / h_den, NO multiplier.
    # Source columns are optional (present in newer extracts only); everything HRI
    # degrades gracefully when they are absent.
    H_NUM_H6: Final[str] = "h_num_h6"
    H_DEN_H6: Final[str] = "h_den_h6"
    H_NUM_H3: Final[str] = "h_num_h3"
    H_DEN_H3: Final[str] = "h_den_h3"
    HRI_H6: Final[str] = "hri_h6"
    HRI_H3: Final[str] = "hri_h3"

    # Grouping variables (default names)
    OCTROI_BINNED: Final[str] = "octroi_binned"
    EFX_BINNED: Final[str] = "efx_binned"

    # Solution columns
    SOL_FAC: Final[str] = "sol_fac"
    N_OBSERVATIONS: Final[str] = "n_observations"
    GROUP: Final[str] = "group"


# Optional indicator source columns: present only in some data extracts. They are
# stripped from settings (with one warning) when absent — see data_manager.py — and
# every consumer degrades gracefully. Shared here so the set is defined ONCE
# (previously the H3 literal was duplicated across data_manager/data_quality/
# preprocess_improved).
H3_OPTIONAL_COLUMNS: Final[frozenset] = frozenset({Columns.TODU_30EVER_H3, Columns.TODU_AMT_PILE_H3})
HRI_OPTIONAL_COLUMNS: Final[frozenset] = frozenset(
    {Columns.H_NUM_H6, Columns.H_DEN_H6, Columns.H_NUM_H3, Columns.H_DEN_H3}
)
OPTIONAL_INDICATOR_COLUMNS: Final[frozenset] = H3_OPTIONAL_COLUMNS | HRI_OPTIONAL_COLUMNS


# Suffixes for aggregated data
class Suffixes:
    """Suffixes used for indicator columns in aggregated data."""

    BOOKED: Final[str] = "_boo"
    REPESCA: Final[str] = "_rep"
    CUT: Final[str] = "_cut"


# =============================================================================
# RISK CALCULATION CONSTANTS
# =============================================================================
# Default multiplier for b2_ever_h6 calculation: 7 * todu_30ever_h6 / todu_amt_pile_h6
DEFAULT_RISK_MULTIPLIER: Final[int] = 7

# Default multiplier for b2_ever_h3 calculation: 4 * todu_30ever_h3 / todu_amt_pile_h3
DEFAULT_RISK_MULTIPLIER_H3: Final[int] = 4

# Default Z-score threshold for outlier removal
DEFAULT_Z_THRESHOLD: Final[float] = 3.0

# Default test size for train/test splits
DEFAULT_TEST_SIZE: Final[float] = 0.4

# Random state for reproducibility
DEFAULT_RANDOM_STATE: Final[int] = 42

# Model and visualization constants
DEFAULT_N_POINTS_3D: Final[int] = 20
DEFAULT_ZERO_THRESHOLD: Final[float] = 1e-10

# PSI/CSI stability thresholds
PSI_STABLE_THRESHOLD: Final[float] = 0.1
PSI_UNSTABLE_THRESHOLD: Final[float] = 0.25

# PSI/CSI epsilon to prevent log(0) in divergence calculations
PSI_EPSILON: Final[float] = 0.0001

# Score scale: risk scores range from 0 to this value,
# where this value represents the best credit quality.
SCORE_SCALE_MAX: Final[int] = 9

# Decision tree default min samples per leaf
DEFAULT_MIN_SAMPLES_LEAF: Final[int] = 500

# Bootstrap resampling defaults
DEFAULT_N_BOOTSTRAPS: Final[int] = 1000

# Sensitivity analysis default perturbation levels (%)
DEFAULT_SENSITIVITY_LEVELS: Final[list[float]] = [-20, -10, -5, 5, 10, 20]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def get_suffixed_columns(base_columns: list, suffix: str) -> list:
    """
    Generate column names with a suffix.

    Args:
        base_columns: List of base column names.
        suffix: Suffix to append (e.g., '_boo', '_rep').

    Returns:
        List of column names with suffix appended.

    Example:
        >>> get_suffixed_columns(['oa_amt', 'todu_30ever_h6'], '_boo')
        ['oa_amt_boo', 'todu_30ever_h6_boo']
    """
    return [f"{col}{suffix}" for col in base_columns]


def get_risk_columns() -> list:
    """Return the standard risk-related column names."""
    return [
        Columns.TODU_30EVER_H6,
        Columns.TODU_AMT_PILE_H6,
        Columns.B2_EVER_H6,
    ]


def get_production_columns() -> list:
    """Return the standard production-related column names."""
    return [
        Columns.OA_AMT,
        Columns.OA_AMT_H0,
    ]
