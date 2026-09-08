"""Risk indicator registry — the metrics a portfolio's risk can be expressed in.

Two families exist:

- **b2** (classic): ``b2_ever_h6 = multiplier * todu_30ever_h6 / todu_amt_pile_h6``
  (H3 variant with ``multiplier_h3``). The multipliers are FIXED accounting
  constants (see the assumptions register).
- **HRI** (Harmonized Risk Indicator): ``hri_h6 = h_num_h6 / h_den_h6`` — a plain
  ratio, NO multiplier (H3 variant analogous). The ``h_*`` source columns are
  OPTIONAL (newer extracts only); use :func:`hri_available` /
  :func:`hri_h3_available` as the canonical fail-soft gate — never an inline
  ``"h_num_h6" in cols`` check.

Both indicators aggregate the same way: sum numerators and denominators, then
take the ratio (never average rates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.constants import Columns


@dataclass(frozen=True)
class RiskIndicator:
    """One risk metric: rate = multiplier * num_col / den_col (as %).

    ``multiplier_field`` names the ``PreprocessingSettings`` attribute holding the
    multiplier; ``None`` means an unmultiplied ratio (multiplier 1.0).
    """

    key: str
    num_col: str
    den_col: str
    output_col: str
    display_label: str
    multiplier_field: str | None = None
    h3_key: str | None = None

    def multiplier(self, settings: Any = None) -> float:
        if self.multiplier_field is None:
            return 1.0
        if settings is None:
            raise ValueError(f"indicator '{self.key}' needs settings to resolve its multiplier")
        return float(getattr(settings, self.multiplier_field))

    def compute(self, numerator, denominator, *, settings: Any = None, as_percentage: bool = True, decimals: int = 6):
        """Rate from raw sums via the shared ratio helper (NaN on zero denominator)."""
        from src.utils import calculate_b2_ever_h6

        return calculate_b2_ever_h6(
            numerator,
            denominator,
            multiplier=self.multiplier(settings),
            as_percentage=as_percentage,
            decimals=decimals,
        )


B2_H6 = RiskIndicator(
    key="b2_ever_h6",
    num_col=Columns.TODU_30EVER_H6,
    den_col=Columns.TODU_AMT_PILE_H6,
    output_col=Columns.B2_EVER_H6,
    display_label="Risk (%)",
    multiplier_field="multiplier",
    h3_key="b2_ever_h3",
)
B2_H3 = RiskIndicator(
    key="b2_ever_h3",
    num_col=Columns.TODU_30EVER_H3,
    den_col=Columns.TODU_AMT_PILE_H3,
    output_col=Columns.B2_EVER_H3,
    display_label="Risk H3 (%)",
    multiplier_field="multiplier_h3",
)
HRI_H6 = RiskIndicator(
    key="hri_h6",
    num_col=Columns.H_NUM_H6,
    den_col=Columns.H_DEN_H6,
    output_col=Columns.HRI_H6,
    display_label="HRI (%)",
    h3_key="hri_h3",
)
HRI_H3 = RiskIndicator(
    key="hri_h3",
    num_col=Columns.H_NUM_H3,
    den_col=Columns.H_DEN_H3,
    output_col=Columns.HRI_H3,
    display_label="HRI H3 (%)",
)

RISK_INDICATORS: dict[str, RiskIndicator] = {i.key: i for i in (B2_H6, B2_H3, HRI_H6, HRI_H3)}


def _columns_of(df_or_cols: Any) -> Any:
    return df_or_cols.columns if hasattr(df_or_cols, "columns") else df_or_cols


def hri_available(df_or_cols: Any, suffix: str = "") -> bool:
    """THE canonical fail-soft predicate: both H6 HRI source columns present."""
    cols = _columns_of(df_or_cols)
    return f"{Columns.H_NUM_H6}{suffix}" in cols and f"{Columns.H_DEN_H6}{suffix}" in cols


def hri_h3_available(df_or_cols: Any, suffix: str = "") -> bool:
    """Both H3 HRI source columns present (checked independently of the H6 pair)."""
    cols = _columns_of(df_or_cols)
    return f"{Columns.H_NUM_H3}{suffix}" in cols and f"{Columns.H_DEN_H3}{suffix}" in cols
