"""
Consolidation Module for Risk Production Tables

Aggregates risk_production_summary_table data across:
- Segments within a supersegment
- Total across all segments
- Main period and MR period
- Multiple scenarios

Produces portfolio-level views for executive reporting.
"""

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from .constants import DEFAULT_RISK_MULTIPLIER, DEFAULT_RISK_MULTIPLIER_H3
from .utils import calculate_b2_ever_h6


@dataclass
class ConsolidatedMetrics:
    """Consolidated metrics for a group of segments."""

    group_name: str
    period: str  # 'main' or 'mr'
    scenario: str
    segments: list[str]

    # Risk multipliers (from config; default to constants for backwards compatibility)
    multiplier: float = float(DEFAULT_RISK_MULTIPLIER)
    multiplier_h3: float = float(DEFAULT_RISK_MULTIPLIER_H3)

    # Aggregated metrics - Production (€)
    actual_production: float = 0.0
    optimum_production: float = 0.0
    swap_in_production: float = 0.0
    swap_out_production: float = 0.0

    # Raw risk components for proper aggregation
    # Risk = todu_30ever_h6 / todu_amt_pile_h6 * 7
    actual_todu_30ever_h6: float = 0.0
    actual_todu_amt_pile_h6: float = 0.0
    optimum_todu_30ever_h6: float = 0.0
    optimum_todu_amt_pile_h6: float = 0.0
    swap_in_todu_30ever_h6: float = 0.0
    swap_in_todu_amt_pile_h6: float = 0.0
    swap_out_todu_30ever_h6: float = 0.0
    swap_out_todu_amt_pile_h6: float = 0.0

    # Raw risk components for H3 (complementary 3-month horizon metric)
    actual_todu_30ever_h3: float = 0.0
    actual_todu_amt_pile_h3: float = 0.0
    optimum_todu_30ever_h3: float = 0.0
    optimum_todu_amt_pile_h3: float = 0.0
    swap_in_todu_30ever_h3: float = 0.0
    swap_in_todu_amt_pile_h3: float = 0.0
    swap_out_todu_30ever_h3: float = 0.0
    swap_out_todu_amt_pile_h3: float = 0.0

    # Total demand (booked + rejected, excluding canceled)
    total_demand: float = 0.0

    # Confidence Intervals (for optimum solution)
    optimum_production_ci_lower: float = 0.0
    optimum_production_ci_upper: float = 0.0
    optimum_risk_ci_lower: float = 0.0
    optimum_risk_ci_upper: float = 0.0

    # Calculated risk properties (as percentage, e.g. 1.5 means 1.5%)
    @cached_property
    def actual_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.actual_todu_30ever_h6, self.actual_todu_amt_pile_h6,
                    multiplier=self.multiplier, as_percentage=True, decimals=6,
                )
            )
        )

    @cached_property
    def optimum_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.optimum_todu_30ever_h6, self.optimum_todu_amt_pile_h6,
                    multiplier=self.multiplier, as_percentage=True, decimals=6,
                )
            )
        )

    @cached_property
    def swap_in_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_in_todu_30ever_h6, self.swap_in_todu_amt_pile_h6,
                    multiplier=self.multiplier, as_percentage=True, decimals=6,
                )
            )
        )

    @cached_property
    def swap_out_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_out_todu_30ever_h6, self.swap_out_todu_amt_pile_h6,
                    multiplier=self.multiplier, as_percentage=True, decimals=6,
                )
            )
        )

    # H3 risk properties (complementary 3-month horizon metric)
    @cached_property
    def actual_risk_h3(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.actual_todu_30ever_h3,
                    self.actual_todu_amt_pile_h3,
                    multiplier=self.multiplier_h3,
                    as_percentage=True,
                    decimals=6,
                )
            )
        )

    @cached_property
    def optimum_risk_h3(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.optimum_todu_30ever_h3,
                    self.optimum_todu_amt_pile_h3,
                    multiplier=self.multiplier_h3,
                    as_percentage=True,
                    decimals=6,
                )
            )
        )

    @cached_property
    def swap_in_risk_h3(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_in_todu_30ever_h3,
                    self.swap_in_todu_amt_pile_h3,
                    multiplier=self.multiplier_h3,
                    as_percentage=True,
                    decimals=6,
                )
            )
        )

    @cached_property
    def swap_out_risk_h3(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_out_todu_30ever_h3,
                    self.swap_out_todu_amt_pile_h3,
                    multiplier=self.multiplier_h3,
                    as_percentage=True,
                    decimals=6,
                )
            )
        )

    @property
    def actual_rejection_rate(self) -> float:
        """Rejection rate under the actual (current) policy: rejected / total_demand."""
        if self.total_demand <= 0:
            return 0.0
        return (1 - self.actual_production / self.total_demand) * 100

    @property
    def optimum_rejection_rate(self) -> float:
        """Rejection rate under the optimum policy: rejected / total_demand."""
        if self.total_demand <= 0:
            return 0.0
        return (1 - self.optimum_production / self.total_demand) * 100

    @property
    def production_delta(self) -> float:
        return self.optimum_production - self.actual_production

    @property
    def production_delta_pct(self) -> float:
        if self.actual_production == 0:
            return 0.0
        return (self.optimum_production - self.actual_production) / self.actual_production

    @property
    def risk_delta(self) -> float:
        return self.optimum_risk - self.actual_risk

    def to_dict(self) -> dict[str, Any]:
        # Risk properties already return percentage (e.g. 7.0 means 7%)
        d = {
            "group": self.group_name,
            "period": self.period,
            "scenario": self.scenario,
            "n_segments": len(self.segments),
            "segments": ", ".join(self.segments),
            "actual_production": self.actual_production,
            "actual_risk_pct": self.actual_risk,
            "actual_todu_30ever_h6": self.actual_todu_30ever_h6,
            "actual_todu_amt_pile_h6": self.actual_todu_amt_pile_h6,
            "optimum_production": self.optimum_production,
            "optimum_risk_pct": self.optimum_risk,
            "optimum_todu_30ever_h6": self.optimum_todu_30ever_h6,
            "optimum_todu_amt_pile_h6": self.optimum_todu_amt_pile_h6,
            "swap_in_production": self.swap_in_production,
            "swap_in_risk_pct": self.swap_in_risk,
            "swap_in_todu_30ever_h6": self.swap_in_todu_30ever_h6,
            "swap_in_todu_amt_pile_h6": self.swap_in_todu_amt_pile_h6,
            "swap_out_production": self.swap_out_production,
            "swap_out_risk_pct": self.swap_out_risk,
            "swap_out_todu_30ever_h6": self.swap_out_todu_30ever_h6,
            "swap_out_todu_amt_pile_h6": self.swap_out_todu_amt_pile_h6,
            "production_delta": self.production_delta,
            "production_delta_pct": self.production_delta_pct * 100,
            "risk_delta_pct": self.risk_delta,
            "production_ci_lower": self.optimum_production_ci_lower,
            "production_ci_upper": self.optimum_production_ci_upper,
            "risk_ci_lower": self.optimum_risk_ci_lower,
            "risk_ci_upper": self.optimum_risk_ci_upper,
            "total_demand": self.total_demand,
            "actual_rejection_rate_pct": self.actual_rejection_rate,
            "optimum_rejection_rate_pct": self.optimum_rejection_rate,
        }
        # H3 complementary risk metrics
        has_h3 = (
            self.actual_todu_amt_pile_h3 > 0
            or self.optimum_todu_amt_pile_h3 > 0
            or self.swap_in_todu_amt_pile_h3 > 0
            or self.swap_out_todu_amt_pile_h3 > 0
        )
        if has_h3:
            d.update(
                {
                    "actual_risk_h3_pct": self.actual_risk_h3,
                    "actual_todu_30ever_h3": self.actual_todu_30ever_h3,
                    "actual_todu_amt_pile_h3": self.actual_todu_amt_pile_h3,
                    "optimum_risk_h3_pct": self.optimum_risk_h3,
                    "optimum_todu_30ever_h3": self.optimum_todu_30ever_h3,
                    "optimum_todu_amt_pile_h3": self.optimum_todu_amt_pile_h3,
                    "swap_in_risk_h3_pct": self.swap_in_risk_h3,
                    "swap_in_todu_30ever_h3": self.swap_in_todu_30ever_h3,
                    "swap_in_todu_amt_pile_h3": self.swap_in_todu_amt_pile_h3,
                    "swap_out_risk_h3_pct": self.swap_out_risk_h3,
                    "swap_out_todu_30ever_h3": self.swap_out_todu_30ever_h3,
                    "swap_out_todu_amt_pile_h3": self.swap_out_todu_amt_pile_h3,
                }
            )
        return d


def find_scenario_suffix(filename: str) -> str:
    """Extract scenario suffix from filename.

    Handles both named scenarios (e.g., 'risk_production_summary_table_pessimistic.csv')
    and legacy numeric scenarios (e.g., 'risk_production_summary_table_1.1.csv').
    """
    # Remove extension
    name = Path(filename).stem

    # Named scenarios to detect
    named_scenarios = ["pessimistic", "base", "optimistic"]

    # Check for named scenario pattern first
    parts = name.split("_")
    for part in reversed(parts):
        if part.lower() in named_scenarios:
            return f"_{part.lower()}"

    # Check for legacy numeric pattern (e.g., _1.1, _0.9)
    for part in reversed(parts):
        try:
            float(part)
            return f"_{part}"
        except ValueError:
            continue

    return ""


def map_scenario_names(scenario_suffixes: list[str]) -> dict[str, str]:
    """
    Map scenario suffixes to meaningful names.

    Scenarios:
    - base: optimum risk threshold (middle value or no suffix)
    - pessimistic: optimum - step (lower value, more conservative)
    - optimistic: optimum + step (higher value, more aggressive)

    Args:
        scenario_suffixes: List of suffixes like ['_pessimistic', '_base', '_optimistic']
                          or legacy format ['', '_0.9', '_1.0', '_1.1']

    Returns:
        Dict mapping suffix to name, e.g., {'_base': 'base', '_pessimistic': 'pessimistic'}
    """
    mapping = {}

    # Check if we have named scenarios (new format)
    named_scenarios = {"pessimistic", "base", "optimistic"}

    for suffix in scenario_suffixes:
        clean_suffix = suffix.strip("_").lower()

        if clean_suffix in named_scenarios:
            # New format: _pessimistic, _base, _optimistic
            mapping[suffix] = clean_suffix
        elif suffix == "":
            # Empty suffix defaults to base
            mapping[suffix] = "base"
        else:
            # Legacy format: try to parse as numeric
            try:
                val = float(clean_suffix)
                # Will be mapped later based on relative values
                mapping[suffix] = None  # Placeholder
            except ValueError:
                # Unknown format, use as-is
                mapping[suffix] = clean_suffix

    # Handle legacy numeric format if any placeholders exist
    placeholders = [s for s, v in mapping.items() if v is None]
    if placeholders:
        # Extract numeric values and sort
        numeric_suffixes = []
        for suffix in placeholders:
            try:
                val = float(suffix.strip("_"))
                numeric_suffixes.append((suffix, val))
            except ValueError:
                mapping[suffix] = suffix.strip("_")

        if numeric_suffixes:
            numeric_suffixes.sort(key=lambda x: x[1])

            if len(numeric_suffixes) == 1:
                mapping[numeric_suffixes[0][0]] = "base"
            elif len(numeric_suffixes) == 2:
                mapping[numeric_suffixes[0][0]] = "pessimistic"
                mapping[numeric_suffixes[1][0]] = "optimistic"
            else:
                # Three or more: lowest is pessimistic, highest is optimistic, middle is base
                mapping[numeric_suffixes[0][0]] = "pessimistic"
                mapping[numeric_suffixes[-1][0]] = "optimistic"
                for suffix, _val in numeric_suffixes[1:-1]:
                    mapping[suffix] = "base"

    return mapping


def load_risk_production_table(
    segment_dir: Path, period: str = "main", scenario_suffix: str = ""
) -> pd.DataFrame | None:
    """
    Load risk_production_summary_table for a segment.

    Args:
        segment_dir: Path to segment output directory
        period: 'main' or 'mr'
        scenario_suffix: Scenario suffix like '_1.1' or ''

    Returns:
        DataFrame or None if file not found
    """
    if period == "mr":
        filename = f"risk_production_summary_table_mr{scenario_suffix}.csv"
    else:
        filename = f"risk_production_summary_table{scenario_suffix}.csv"

    filepath = segment_dir / "data" / filename

    if not filepath.exists():
        logger.debug(f"File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)
        return df
    except (pd.errors.ParserError, OSError, ValueError) as e:
        logger.warning(f"Error loading {filepath}: {e}")
        return None


def extract_metrics_from_table(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Extract key metrics from risk_production_summary_table.

    Expected columns:
        - Metric: 'Actual', 'Swap-in', 'Swap-out', 'Optimum selected'
        - Production (€): oa_amt_h0
        - Risk (%): b2_ever_h6 (calculated)
        - todu_30ever_h6: raw numerator for risk
        - todu_amt_pile_h6: raw denominator for risk

    Returns dict with production and raw todu values for proper aggregation.
    """
    metrics = {
        "actual": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "optimum": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "swap_in": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "swap_out": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
    }

    if df is None or df.empty:
        return metrics

    # Normalize column names
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Find metric column (could be 'metric', 'category', first column, etc.)
    metric_col = None
    for col in ["metric", "category", "type", "row"]:
        if col in df.columns:
            metric_col = col
            break
    if metric_col is None:
        metric_col = df.columns[0]

    # Find value columns
    prod_col = None
    todu_30_col = None
    todu_amt_col = None
    todu_30_h3_col = None
    todu_amt_h3_col = None
    rejection_rate_col = None

    for col in df.columns:
        col_lower = col.lower()
        if "production" in col_lower and "€" in col_lower:
            prod_col = col
        elif col_lower == "todu_30ever_h6":
            todu_30_col = col
        elif col_lower == "todu_amt_pile_h6":
            todu_amt_col = col
        elif col_lower == "todu_30ever_h3":
            todu_30_h3_col = col
        elif col_lower == "todu_amt_pile_h3":
            todu_amt_h3_col = col
        elif "rejection" in col_lower and "rate" in col_lower:
            rejection_rate_col = col

    # Fallback for production column
    if prod_col is None:
        for col in df.columns:
            if "prod" in col.lower() or "oa_amt" in col.lower():
                prod_col = col
                break

    logger.debug(f"Columns found - prod: {prod_col}, todu_30: {todu_30_col}, todu_amt: {todu_amt_col}")

    # Extract values for each row type
    for _, row in df.iterrows():
        row_type = str(row[metric_col]).lower().strip()

        # Map row type to our standard names
        if "actual" in row_type or "current" in row_type or "baseline" in row_type:
            key = "actual"
        elif "optim" in row_type or "target" in row_type:
            key = "optimum"
        elif "swap-in" in row_type or "swapin" in row_type or "swap_in" in row_type:
            key = "swap_in"
        elif "swap-out" in row_type or "swapout" in row_type or "swap_out" in row_type:
            key = "swap_out"
        else:
            continue

        # Extract production
        if prod_col and prod_col in row.index:
            try:
                metrics[key]["production"] = float(row[prod_col]) if pd.notna(row[prod_col]) else 0
            except (ValueError, TypeError):
                pass

        # Extract todu_30ever_h6
        if todu_30_col and todu_30_col in row.index:
            try:
                metrics[key]["todu_30ever_h6"] = float(row[todu_30_col]) if pd.notna(row[todu_30_col]) else 0
            except (ValueError, TypeError):
                pass

        # Extract todu_amt_pile_h6
        if todu_amt_col and todu_amt_col in row.index:
            try:
                metrics[key]["todu_amt_pile_h6"] = float(row[todu_amt_col]) if pd.notna(row[todu_amt_col]) else 0
            except (ValueError, TypeError):
                pass

        # Extract H3 columns (complementary metric)
        if todu_30_h3_col and todu_30_h3_col in row.index:
            try:
                metrics[key]["todu_30ever_h3"] = float(row[todu_30_h3_col]) if pd.notna(row[todu_30_h3_col]) else 0
            except (ValueError, TypeError):
                pass
        if todu_amt_h3_col and todu_amt_h3_col in row.index:
            try:
                metrics[key]["todu_amt_pile_h3"] = float(row[todu_amt_h3_col]) if pd.notna(row[todu_amt_h3_col]) else 0
            except (ValueError, TypeError):
                pass

        # Extract CI values (only for optimum)
        if key == "optimum":
            for col in df.columns:
                col_lower = col.lower()
                if "production_ci_lower" in col_lower:
                    metrics["optimum"]["production_ci_lower"] = float(row[col]) if pd.notna(row[col]) else 0
                elif "production_ci_upper" in col_lower:
                    metrics["optimum"]["production_ci_upper"] = float(row[col]) if pd.notna(row[col]) else 0
                elif "risk_ci_lower" in col_lower:
                    metrics["optimum"]["risk_ci_lower"] = float(row[col]) if pd.notna(row[col]) else 0
                elif "risk_ci_upper" in col_lower:
                    metrics["optimum"]["risk_ci_upper"] = float(row[col]) if pd.notna(row[col]) else 0

    # Read total_demand from explicit column, else derive from rejection rate
    actual_prod = metrics["actual"]["production"]
    total_demand_col = None
    for col in df.columns:
        if "total demand" in col.lower():
            total_demand_col = col
            break

    actual_row = df[df[metric_col].str.lower().str.strip().str.contains("actual|current|baseline", na=False)]
    if total_demand_col is not None and not actual_row.empty:
        td_val = actual_row.iloc[0].get(total_demand_col)
        if pd.notna(td_val) and float(td_val) > 0:
            metrics["_total_demand"] = float(td_val)
        else:
            metrics["_total_demand"] = actual_prod
    else:
        # Backward compat: derive from rejection rate
        actual_rej = 0.0
        if rejection_rate_col is not None and not actual_row.empty:
            rej_val = actual_row.iloc[0].get(rejection_rate_col)
            if pd.notna(rej_val):
                actual_rej = float(rej_val)
        if actual_rej < 100 and actual_prod > 0:
            metrics["_total_demand"] = actual_prod / (1 - actual_rej / 100)
        else:
            metrics["_total_demand"] = actual_prod

    return metrics


def aggregate_metrics(
    metrics_list: list[dict[str, dict[str, float]]],
    multiplier: float = float(DEFAULT_RISK_MULTIPLIER),
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics from multiple segments.

    Sums production and raw todu values. Risk is calculated from aggregated
    todu values: risk = sum(todu_30ever_h6) / sum(todu_amt_pile_h6) * multiplier
    """
    aggregated = {
        "actual": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "optimum": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "swap_in": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
        "swap_out": {
            "production": 0,
            "todu_30ever_h6": 0,
            "todu_amt_pile_h6": 0,
            "todu_30ever_h3": 0,
            "todu_amt_pile_h3": 0,
        },
    }

    total_demand_sum = 0.0
    for metrics in metrics_list:
        total_demand_sum += metrics.get("_total_demand", metrics["actual"]["production"])
        for key in aggregated:
            aggregated[key]["production"] += metrics[key]["production"]
            aggregated[key]["todu_30ever_h6"] += metrics[key]["todu_30ever_h6"]
            aggregated[key]["todu_amt_pile_h6"] += metrics[key]["todu_amt_pile_h6"]
            aggregated[key]["todu_30ever_h3"] += metrics[key].get("todu_30ever_h3", 0)
            aggregated[key]["todu_amt_pile_h3"] += metrics[key].get("todu_amt_pile_h3", 0)

            # Aggregate CIs for optimum using variance addition (assumes independence)
            if key == "optimum":
                if "_ci_segments" not in aggregated[key]:
                    aggregated[key]["_ci_segments"] = []

                aggregated[key]["_ci_segments"].append(
                    {
                        "prod_lower": metrics[key].get("production_ci_lower", 0),
                        "prod_upper": metrics[key].get("production_ci_upper", 0),
                        "risk_lower": metrics[key].get("risk_ci_lower", 0),
                        "risk_upper": metrics[key].get("risk_ci_upper", 0),
                    }
                )

    # Combine segment CIs using variance addition rule (assumes independence).
    # SE_i ≈ (upper - lower) / (2 * z_95); combined SE = sqrt(sum(SE_i²))
    import numpy as np

    z_95 = 1.96
    for key in aggregated:
        segments = aggregated[key].pop("_ci_segments", [])
        if not segments:
            aggregated[key].setdefault("production_ci_lower", 0)
            aggregated[key].setdefault("production_ci_upper", 0)
            continue

        prod_means, prod_ses = [], []
        risk_means, risk_ses = [], []
        for s in segments:
            p_lo, p_hi = s["prod_lower"], s["prod_upper"]
            r_lo, r_hi = s["risk_lower"], s["risk_upper"]
            prod_means.append((p_lo + p_hi) / 2)
            prod_ses.append((p_hi - p_lo) / (2 * z_95) if p_hi != p_lo else 0)
            risk_means.append((r_lo + r_hi) / 2)
            risk_ses.append((r_hi - r_lo) / (2 * z_95) if r_hi != r_lo else 0)

        combined_prod_mean = sum(prod_means)
        combined_prod_se = np.sqrt(sum(se**2 for se in prod_ses))
        aggregated[key]["production_ci_lower"] = combined_prod_mean - z_95 * combined_prod_se
        aggregated[key]["production_ci_upper"] = combined_prod_mean + z_95 * combined_prod_se

        # Risk CIs: compute the center from aggregated todu components (ratio metric,
        # not additive — cannot use sum of segment means).
        combined_risk_se = np.sqrt(sum(se**2 for se in risk_ses))
        if combined_risk_se > 0:
            agg_num = aggregated[key].get("todu_30ever_h6", 0)
            agg_den = aggregated[key].get("todu_amt_pile_h6", 0)
            agg_risk_point = float(calculate_b2_ever_h6(agg_num, agg_den, multiplier=multiplier, as_percentage=True)) if agg_den else 0.0
            aggregated[key]["risk_ci_lower"] = agg_risk_point - z_95 * combined_risk_se
            aggregated[key]["risk_ci_upper"] = agg_risk_point + z_95 * combined_risk_se
        else:
            aggregated[key]["risk_ci_lower"] = 0
            aggregated[key]["risk_ci_upper"] = 0

    aggregated["_total_demand"] = total_demand_sum

    return aggregated


def consolidate_segments(
    output_base: Path,
    segments: dict[str, dict[str, Any]],
    supersegments: dict[str, dict[str, Any]],
    scenarios: list[str] | None = None,
    multiplier: float = float(DEFAULT_RISK_MULTIPLIER),
    multiplier_h3: float = float(DEFAULT_RISK_MULTIPLIER_H3),
) -> pd.DataFrame:
    """
    Consolidate risk production tables across segments, supersegments, and scenarios.

    Args:
        output_base: Base output directory
        segments: Segment configurations
        supersegments: Supersegment configurations
        scenarios: List of scenario suffixes (e.g., ['', '_1.0', '_1.1'])
        multiplier: H6 risk multiplier (from config.toml)
        multiplier_h3: H3 risk multiplier (from config.toml)

    Returns:
        DataFrame with consolidated metrics
    """
    if scenarios is None:
        # Auto-detect scenarios from first segment
        scenarios = []
        for seg_name in segments:
            seg_dir = output_base / seg_name / "data"
            if seg_dir.exists():
                for f in seg_dir.glob("risk_production_summary_table*.csv"):
                    # Skip MR files for scenario detection
                    if "_mr" in f.name:
                        continue
                    suffix = find_scenario_suffix(f.name)
                    if suffix not in scenarios:
                        scenarios.append(suffix)
                if scenarios:
                    break

        # Ensure we have at least one scenario
        if not scenarios:
            scenarios = [""]

    # Map scenario suffixes to meaningful names (base, pessimistic, optimistic)
    scenario_name_map = map_scenario_names(scenarios)

    # Deduplicate scenarios that map to the same name (e.g., '' and '_base' both map to 'base')
    # Keep the more specific suffix (e.g., '_base' over '')
    seen_names = {}
    for suffix in scenarios:
        name = scenario_name_map.get(suffix, "base")
        if name not in seen_names:
            seen_names[name] = suffix

    # Rebuild scenarios list with deduplicated suffixes
    scenarios = list(seen_names.values())
    scenario_name_map = {s: map_scenario_names([s]).get(s, "base") for s in scenarios}

    logger.info(f"Consolidating data for scenarios: {scenario_name_map}")

    results = []

    # Build supersegment membership map
    segment_to_supersegment = {}
    for seg_name, seg_config in segments.items():
        ss = seg_config.get("supersegment")
        if ss:
            segment_to_supersegment[seg_name] = ss

    # Process each scenario
    for scenario_suffix in scenarios:
        scenario_name = scenario_name_map.get(scenario_suffix, "base")

        # Process each period (main and MR)
        for period in ["main", "mr"]:
            # Collect metrics by segment
            segment_metrics = {}
            for seg_name in segments:
                seg_dir = output_base / seg_name
                df = load_risk_production_table(seg_dir, period, scenario_suffix)
                if df is not None:
                    metrics = extract_metrics_from_table(df)
                    segment_metrics[seg_name] = metrics

            if not segment_metrics:
                logger.warning(f"No data found for scenario={scenario_name}, period={period}")
                continue

            # Aggregate by supersegment
            supersegment_data = {}
            for ss_name in supersegments:
                ss_segments = [
                    seg_name
                    for seg_name, ss in segment_to_supersegment.items()
                    if ss == ss_name and seg_name in segment_metrics
                ]
                if ss_segments:
                    ss_metrics_list = [segment_metrics[s] for s in ss_segments]
                    agg = aggregate_metrics(ss_metrics_list, multiplier=multiplier)

                    consolidated = ConsolidatedMetrics(
                        group_name=f"supersegment_{ss_name}",
                        period=period,
                        scenario=scenario_name,
                        segments=ss_segments,
                        multiplier=multiplier,
                        multiplier_h3=multiplier_h3,
                        actual_production=agg["actual"]["production"],
                        actual_todu_30ever_h6=agg["actual"]["todu_30ever_h6"],
                        actual_todu_amt_pile_h6=agg["actual"]["todu_amt_pile_h6"],
                        optimum_production=agg["optimum"]["production"],
                        optimum_todu_30ever_h6=agg["optimum"]["todu_30ever_h6"],
                        optimum_todu_amt_pile_h6=agg["optimum"]["todu_amt_pile_h6"],
                        swap_in_production=agg["swap_in"]["production"],
                        swap_in_todu_30ever_h6=agg["swap_in"]["todu_30ever_h6"],
                        swap_in_todu_amt_pile_h6=agg["swap_in"]["todu_amt_pile_h6"],
                        swap_out_production=agg["swap_out"]["production"],
                        swap_out_todu_30ever_h6=agg["swap_out"]["todu_30ever_h6"],
                        swap_out_todu_amt_pile_h6=agg["swap_out"]["todu_amt_pile_h6"],
                        # H3 complementary metrics
                        actual_todu_30ever_h3=agg["actual"].get("todu_30ever_h3", 0),
                        actual_todu_amt_pile_h3=agg["actual"].get("todu_amt_pile_h3", 0),
                        optimum_todu_30ever_h3=agg["optimum"].get("todu_30ever_h3", 0),
                        optimum_todu_amt_pile_h3=agg["optimum"].get("todu_amt_pile_h3", 0),
                        swap_in_todu_30ever_h3=agg["swap_in"].get("todu_30ever_h3", 0),
                        swap_in_todu_amt_pile_h3=agg["swap_in"].get("todu_amt_pile_h3", 0),
                        swap_out_todu_30ever_h3=agg["swap_out"].get("todu_30ever_h3", 0),
                        swap_out_todu_amt_pile_h3=agg["swap_out"].get("todu_amt_pile_h3", 0),
                        # Total demand for rejection rate
                        total_demand=agg.get("_total_demand", 0),
                        # Pass aggregated CIs
                        optimum_production_ci_lower=agg["optimum"].get("production_ci_lower", 0),
                        optimum_production_ci_upper=agg["optimum"].get("production_ci_upper", 0),
                        optimum_risk_ci_lower=agg["optimum"].get("risk_ci_lower", 0),
                        optimum_risk_ci_upper=agg["optimum"].get("risk_ci_upper", 0),
                    )
                    results.append(consolidated.to_dict())
                    supersegment_data[ss_name] = agg

            # Add all individual segments (including those in supersegments)
            for seg_name, metrics in segment_metrics.items():
                agg = aggregate_metrics([metrics], multiplier=multiplier)
                # Determine group name based on supersegment membership
                if seg_name in segment_to_supersegment:
                    ss_name = segment_to_supersegment[seg_name]
                    group_name = f"{ss_name}/{seg_name}"
                else:
                    group_name = f"segment_{seg_name}"

                consolidated = ConsolidatedMetrics(
                    group_name=group_name,
                    period=period,
                    scenario=scenario_name,
                    segments=[seg_name],
                    multiplier=multiplier,
                    multiplier_h3=multiplier_h3,
                    actual_production=agg["actual"]["production"],
                    actual_todu_30ever_h6=agg["actual"]["todu_30ever_h6"],
                    actual_todu_amt_pile_h6=agg["actual"]["todu_amt_pile_h6"],
                    optimum_production=agg["optimum"]["production"],
                    optimum_todu_30ever_h6=agg["optimum"]["todu_30ever_h6"],
                    optimum_todu_amt_pile_h6=agg["optimum"]["todu_amt_pile_h6"],
                    swap_in_production=agg["swap_in"]["production"],
                    swap_in_todu_30ever_h6=agg["swap_in"]["todu_30ever_h6"],
                    swap_in_todu_amt_pile_h6=agg["swap_in"]["todu_amt_pile_h6"],
                    swap_out_production=agg["swap_out"]["production"],
                    swap_out_todu_30ever_h6=agg["swap_out"]["todu_30ever_h6"],
                    swap_out_todu_amt_pile_h6=agg["swap_out"]["todu_amt_pile_h6"],
                    # H3 complementary metrics
                    actual_todu_30ever_h3=agg["actual"].get("todu_30ever_h3", 0),
                    actual_todu_amt_pile_h3=agg["actual"].get("todu_amt_pile_h3", 0),
                    optimum_todu_30ever_h3=agg["optimum"].get("todu_30ever_h3", 0),
                    optimum_todu_amt_pile_h3=agg["optimum"].get("todu_amt_pile_h3", 0),
                    swap_in_todu_30ever_h3=agg["swap_in"].get("todu_30ever_h3", 0),
                    swap_in_todu_amt_pile_h3=agg["swap_in"].get("todu_amt_pile_h3", 0),
                    swap_out_todu_30ever_h3=agg["swap_out"].get("todu_30ever_h3", 0),
                    swap_out_todu_amt_pile_h3=agg["swap_out"].get("todu_amt_pile_h3", 0),
                    # Total demand for rejection rate
                    total_demand=agg.get("_total_demand", 0),
                    # Pass segment-level CIs (fully available)
                    optimum_production_ci_lower=metrics["optimum"].get("production_ci_lower", 0),
                    optimum_production_ci_upper=metrics["optimum"].get("production_ci_upper", 0),
                    optimum_risk_ci_lower=metrics["optimum"].get("risk_ci_lower", 0),
                    optimum_risk_ci_upper=metrics["optimum"].get("risk_ci_upper", 0),
                )
                results.append(consolidated.to_dict())

            # Aggregate total across all segments
            all_metrics_list = list(segment_metrics.values())
            if all_metrics_list:
                total_agg = aggregate_metrics(all_metrics_list, multiplier=multiplier)
                total_consolidated = ConsolidatedMetrics(
                    group_name="TOTAL",
                    period=period,
                    scenario=scenario_name,
                    segments=list(segment_metrics.keys()),
                    multiplier=multiplier,
                    multiplier_h3=multiplier_h3,
                    actual_production=total_agg["actual"]["production"],
                    actual_todu_30ever_h6=total_agg["actual"]["todu_30ever_h6"],
                    actual_todu_amt_pile_h6=total_agg["actual"]["todu_amt_pile_h6"],
                    optimum_production=total_agg["optimum"]["production"],
                    optimum_todu_30ever_h6=total_agg["optimum"]["todu_30ever_h6"],
                    optimum_todu_amt_pile_h6=total_agg["optimum"]["todu_amt_pile_h6"],
                    swap_in_production=total_agg["swap_in"]["production"],
                    swap_in_todu_30ever_h6=total_agg["swap_in"]["todu_30ever_h6"],
                    swap_in_todu_amt_pile_h6=total_agg["swap_in"]["todu_amt_pile_h6"],
                    swap_out_production=total_agg["swap_out"]["production"],
                    swap_out_todu_30ever_h6=total_agg["swap_out"]["todu_30ever_h6"],
                    swap_out_todu_amt_pile_h6=total_agg["swap_out"]["todu_amt_pile_h6"],
                    # H3 complementary metrics
                    actual_todu_30ever_h3=total_agg["actual"].get("todu_30ever_h3", 0),
                    actual_todu_amt_pile_h3=total_agg["actual"].get("todu_amt_pile_h3", 0),
                    optimum_todu_30ever_h3=total_agg["optimum"].get("todu_30ever_h3", 0),
                    optimum_todu_amt_pile_h3=total_agg["optimum"].get("todu_amt_pile_h3", 0),
                    swap_in_todu_30ever_h3=total_agg["swap_in"].get("todu_30ever_h3", 0),
                    swap_in_todu_amt_pile_h3=total_agg["swap_in"].get("todu_amt_pile_h3", 0),
                    swap_out_todu_30ever_h3=total_agg["swap_out"].get("todu_30ever_h3", 0),
                    swap_out_todu_amt_pile_h3=total_agg["swap_out"].get("todu_amt_pile_h3", 0),
                    # Total demand for rejection rate
                    total_demand=total_agg.get("_total_demand", 0),
                    # Pass aggregated CIs
                    optimum_production_ci_lower=total_agg["optimum"].get("production_ci_lower", 0),
                    optimum_production_ci_upper=total_agg["optimum"].get("production_ci_upper", 0),
                    optimum_risk_ci_lower=total_agg["optimum"].get("risk_ci_lower", 0),
                    optimum_risk_ci_upper=total_agg["optimum"].get("risk_ci_upper", 0),
                )
                results.append(total_consolidated.to_dict())

    df = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = [
        "group",
        "period",
        "scenario",
        "n_segments",
        "segments",
        "actual_production",
        "actual_risk_pct",
        "actual_todu_30ever_h6",
        "actual_todu_amt_pile_h6",
        "optimum_production",
        "optimum_risk_pct",
        "optimum_todu_30ever_h6",
        "optimum_todu_amt_pile_h6",
        "production_delta",
        "production_delta_pct",
        "risk_delta_pct",
        "swap_in_production",
        "swap_in_risk_pct",
        "swap_in_todu_30ever_h6",
        "swap_in_todu_amt_pile_h6",
        "swap_out_production",
        "swap_out_risk_pct",
        "swap_out_todu_30ever_h6",
        "swap_out_todu_amt_pile_h6",
        "total_demand",
        "actual_rejection_rate_pct",
        "optimum_rejection_rate_pct",
        "production_ci_lower",
        "production_ci_upper",
        "risk_ci_lower",
        "risk_ci_upper",
    ]
    df = df[[c for c in column_order if c in df.columns]]

    return df


def create_consolidation_dashboard(df: pd.DataFrame, title: str = "Consolidated Risk Production Report") -> go.Figure:
    """
    Create an interactive dashboard for consolidated metrics.

    Args:
        df: Consolidated DataFrame from consolidate_segments()
        title: Dashboard title

    Returns:
        Plotly Figure
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig

    # Filter to TOTAL rows for overview
    total_df = df[df["group"] == "TOTAL"].copy()

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Production by Period & Scenario",
            "Risk by Period & Scenario",
            "Production Delta (Optimum vs Actual)",
            "Swap Analysis",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}], [{"type": "bar"}, {"type": "bar"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # Color scheme
    colors = {"main": "steelblue", "mr": "coral"}

    # 1. Production by Period & Scenario
    for period in ["main", "mr"]:
        period_data = total_df[total_df["period"] == period]
        fig.add_trace(
            go.Bar(
                name=f"{period.upper()} - Actual",
                x=period_data["scenario"],
                y=period_data["actual_production"],
                marker_color=colors[period],
                opacity=0.6,
                legendgroup=period,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                name=f"{period.upper()} - Optimum",
                x=period_data["scenario"],
                y=period_data["optimum_production"],
                marker_color=colors[period],
                opacity=1.0,
                legendgroup=period,
            ),
            row=1,
            col=1,
        )

    # 2. Risk by Period & Scenario
    for period in ["main", "mr"]:
        period_data = total_df[total_df["period"] == period]
        fig.add_trace(
            go.Bar(
                name=f"{period.upper()} Risk",
                x=period_data["scenario"],
                y=period_data["actual_risk_pct"],
                marker_color=colors[period],
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # 3. Production Delta
    for period in ["main", "mr"]:
        period_data = total_df[total_df["period"] == period]
        fig.add_trace(
            go.Bar(
                name=f"{period.upper()} Delta",
                x=period_data["scenario"],
                y=period_data["production_delta"],
                marker_color=colors[period],
                text=[f"{v:.1f}%" for v in period_data["production_delta_pct"]],
                textposition="outside",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # 4. Swap Analysis (stacked bar for latest scenario)
    latest_scenario = total_df["scenario"].iloc[-1] if not total_df.empty else "base"
    swap_data = total_df[total_df["scenario"] == latest_scenario]

    for period in ["main", "mr"]:
        period_swap = swap_data[swap_data["period"] == period]
        if not period_swap.empty:
            fig.add_trace(
                go.Bar(
                    name=f"{period.upper()} Swap-In",
                    x=[period.upper()],
                    y=period_swap["swap_in_production"].values,
                    marker_color="green",
                    showlegend=(period == "main"),
                ),
                row=2,
                col=2,
            )
            fig.add_trace(
                go.Bar(
                    name=f"{period.upper()} Swap-Out",
                    x=[period.upper()],
                    y=[-period_swap["swap_out_production"].values[0]],  # Negative for visual
                    marker_color="red",
                    showlegend=(period == "main"),
                ),
                row=2,
                col=2,
            )

    fig.update_layout(
        title=dict(text=title, x=0.5),
        height=800,
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    # Update axes labels
    fig.update_yaxes(title_text="Production (€)", row=1, col=1)
    fig.update_yaxes(title_text="Risk (%)", row=1, col=2)
    fig.update_yaxes(title_text="Delta (€)", row=2, col=1)
    fig.update_yaxes(title_text="Production (€)", row=2, col=2)

    return fig


def generate_consolidation_report(
    output_base: str,
    segments: dict[str, dict[str, Any]],
    supersegments: dict[str, dict[str, Any]],
    scenarios: list[str] | None = None,
    output_path: str | None = None,
    multiplier: float = float(DEFAULT_RISK_MULTIPLIER),
    multiplier_h3: float = float(DEFAULT_RISK_MULTIPLIER_H3),
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Generate complete consolidation report with CSV and HTML dashboard.

    Args:
        output_base: Base output directory
        segments: Segment configurations
        supersegments: Supersegment configurations
        scenarios: List of scenario suffixes
        output_path: Optional output path for files (defaults to output_base)
        multiplier: H6 risk multiplier (from config.toml)
        multiplier_h3: H3 risk multiplier (from config.toml)

    Returns:
        Tuple of (consolidated DataFrame, Plotly figure)
    """
    output_base = Path(output_base)
    output_path = Path(output_path) if output_path else output_base

    logger.info("Generating consolidated risk production report...")

    # Consolidate data
    df = consolidate_segments(output_base, segments, supersegments, scenarios, multiplier=multiplier, multiplier_h3=multiplier_h3)

    if df.empty:
        logger.warning("No data found to consolidate")
        return df, None

    # Save CSV
    csv_path = output_path / "consolidated_risk_production.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Consolidated CSV saved to {csv_path}")

    # Create dashboard
    fig = create_consolidation_dashboard(df)

    # Save HTML
    html_path = output_path / "consolidated_risk_production.html"
    fig.write_html(str(html_path))
    logger.info(f"Consolidated dashboard saved to {html_path}")

    # Export Excel workbook
    try:
        xlsx_path = export_consolidated_excel(df, output_base, segments)
        logger.info(f"Consolidated Excel saved to {xlsx_path}")
    except Exception as e:
        logger.warning(f"Excel export failed: {e}")

    # Print summary
    print_consolidation_summary(df)

    return df, fig


def export_consolidated_excel(
    consolidated_df: pd.DataFrame,
    output_base: str | Path,
    segments: dict[str, dict[str, Any]],
) -> Path:
    """Export consolidated report data to a management-ready Excel dashboard.

    Sheets:
        - Executive Summary: KPI cards (main + MR), curated tables, acceptance grids
        - Portfolio Summary: TOTAL + supersegment rows (all scenarios/periods)
        - Segment Detail: per-segment rows
        - Cutoff Comparison: concatenated cutoff data
        - Per-segment acceptance grids (one sheet each)
        - Per-segment risk production summaries

    Args:
        consolidated_df: DataFrame from consolidate_segments()
        output_base: Base output directory (where segment subdirectories live)
        segments: Segment configurations dict

    Returns:
        Path to the written .xlsx file.
    """
    from datetime import date

    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    output_base = Path(output_base)
    xlsx_path = output_base / "consolidated_risk_production.xlsx"

    # =====================================================================
    # Design tokens — modern, soft palette
    # =====================================================================
    # Core brand
    _CLR_PRIMARY = "1B2A4A"       # Deep navy — titles, header bg
    _CLR_PRIMARY_LIGHT = "34495E" # Lighter navy — secondary headers
    _CLR_ACCENT = "2980B9"        # Cerulean blue — KPI accents, links
    _CLR_ACCENT_LIGHT = "D6EAF8"  # Pale blue — KPI card bg
    _CLR_WHITE = "FFFFFF"

    # Semantic
    _CLR_GOOD = "1ABC9C"          # Teal-green — softer than pure green
    _CLR_GOOD_LIGHT = "D1F2EB"    # Pale teal — TOTAL row bg
    _CLR_GOOD_DARK = "0E6655"     # Dark teal — TOTAL row text
    _CLR_BAD = "E74C3C"           # Warm red — risk / negative deltas
    _CLR_BAD_LIGHT = "FDEDEC"     # Pale pink — reject cell tint (unused as fill on its own)
    _CLR_WARN = "F39C12"          # Amber — cutoff tab, warnings
    _CLR_MR_BG = "FEF5E7"         # Warm cream — MR KPI tint
    _CLR_MR_FG = "CA6F1E"         # Dark amber — MR labels

    # Neutral
    _CLR_NEUTRAL_LIGHT = "F8F9FA" # Near-white stripe
    _CLR_NEUTRAL = "DEE2E6"       # Soft grey — table borders
    _CLR_NEUTRAL_MID = "AEB6BF"   # Mid grey — subtle text
    _CLR_TEXT = "2C3E50"          # Dark grey — body text

    # Acceptance grid (softer, desaturated tones)
    _CLR_GRID_ACCEPT = "58D68D"   # Soft green
    _CLR_GRID_REJECT = "EC7063"   # Soft red-coral
    _CLR_GRID_NA = "D5DBDB"       # Light warm grey
    _CLR_GRID_HDR = "2C3E50"      # Dark header for contrast

    # Section
    _CLR_SECTION_BG = "EBF5FB"    # Pale blue — section header bg
    _CLR_SECTION_BAR = "2980B9"   # Accent bar

    # Sheet tab colours
    _CLR_TAB_EXEC = "2980B9"
    _CLR_TAB_PORTFOLIO = "1B2A4A"
    _CLR_TAB_SEGMENT = "AEB6BF"
    _CLR_TAB_CUTOFF = "F39C12"
    _CLR_TAB_GRID = "1ABC9C"

    # ----- Fonts -----
    _FN = "Calibri"
    _FONT_TITLE = Font(bold=True, color=_CLR_PRIMARY, size=20, name=_FN)
    _FONT_SUBTITLE = Font(bold=False, color=_CLR_NEUTRAL_MID, size=11, name=_FN)
    _FONT_SECTION = Font(bold=True, color=_CLR_PRIMARY, size=12, name=_FN)
    _FONT_KPI_VALUE = Font(bold=True, color=_CLR_PRIMARY, size=24, name=_FN)
    _FONT_KPI_LABEL = Font(bold=False, color=_CLR_NEUTRAL_MID, size=9, name=_FN)
    _FONT_KPI_DELTA_POS = Font(bold=True, color=_CLR_GOOD, size=12, name=_FN)
    _FONT_KPI_DELTA_NEG = Font(bold=True, color=_CLR_BAD, size=12, name=_FN)
    _FONT_HEADER = Font(bold=True, color=_CLR_WHITE, size=10, name=_FN)
    _FONT_DATA = Font(color=_CLR_TEXT, size=10, name=_FN)
    _FONT_TOTAL = Font(bold=True, color=_CLR_GOOD_DARK, size=10, name=_FN)
    _FONT_MR_LABEL = Font(bold=True, color=_CLR_MR_FG, size=11, name=_FN)
    _FONT_GRID_LABEL = Font(bold=True, color=_CLR_PRIMARY, size=9, name=_FN)
    _FONT_GRID_CELL = Font(bold=True, color=_CLR_WHITE, size=11, name=_FN)
    _FONT_GRID_HEADER = Font(bold=True, color=_CLR_WHITE, size=9, name=_FN)

    # ----- Fills -----
    _FILL_HEADER = PatternFill(start_color=_CLR_PRIMARY, end_color=_CLR_PRIMARY, fill_type="solid")
    _FILL_STRIPE = PatternFill(start_color=_CLR_NEUTRAL_LIGHT, end_color=_CLR_NEUTRAL_LIGHT, fill_type="solid")
    _FILL_TOTAL = PatternFill(start_color=_CLR_GOOD_LIGHT, end_color=_CLR_GOOD_LIGHT, fill_type="solid")
    _FILL_KPI = PatternFill(start_color=_CLR_ACCENT_LIGHT, end_color=_CLR_ACCENT_LIGHT, fill_type="solid")
    _FILL_MR = PatternFill(start_color=_CLR_MR_BG, end_color=_CLR_MR_BG, fill_type="solid")
    _FILL_SECTION = PatternFill(start_color=_CLR_SECTION_BG, end_color=_CLR_SECTION_BG, fill_type="solid")
    _FILL_ACCEPT = PatternFill(start_color=_CLR_GRID_ACCEPT, end_color=_CLR_GRID_ACCEPT, fill_type="solid")
    _FILL_REJECT = PatternFill(start_color=_CLR_GRID_REJECT, end_color=_CLR_GRID_REJECT, fill_type="solid")
    _FILL_NA = PatternFill(start_color=_CLR_GRID_NA, end_color=_CLR_GRID_NA, fill_type="solid")
    _FILL_GRID_HEADER = PatternFill(start_color=_CLR_GRID_HDR, end_color=_CLR_GRID_HDR, fill_type="solid")

    # ----- Borders -----
    _THIN = Side(style="thin", color=_CLR_NEUTRAL)
    _HAIR = Side(style="hair", color=_CLR_NEUTRAL)
    _BORDER_ALL = Border(top=_HAIR, bottom=_HAIR, left=_HAIR, right=_HAIR)
    _BORDER_HEADER = Border(
        top=Side(style="thin", color=_CLR_PRIMARY),
        bottom=Side(style="medium", color=_CLR_ACCENT),
        left=_HAIR, right=_HAIR,
    )
    _BORDER_BOTTOM = Border(bottom=Side(style="medium", color=_CLR_ACCENT))
    _ACCENT_LEFT = Border(
        left=Side(style="thick", color=_CLR_ACCENT),
        top=_HAIR, bottom=_HAIR, right=_HAIR,
    )
    _SECTION_LEFT = Border(left=Side(style="thick", color=_CLR_SECTION_BAR))
    _BORDER_GRID = Border(
        top=Side(style="medium", color=_CLR_WHITE),
        bottom=Side(style="medium", color=_CLR_WHITE),
        left=Side(style="medium", color=_CLR_WHITE),
        right=Side(style="medium", color=_CLR_WHITE),
    )

    # ----- Alignment -----
    _ALIGN_CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
    _ALIGN_LEFT = Alignment(horizontal="left", vertical="center")
    _ALIGN_RIGHT = Alignment(horizontal="right", vertical="center")

    # =====================================================================
    # Column classification
    # =====================================================================
    _CURRENCY_COLS = {
        "actual_production", "optimum_production", "swap_in_production", "swap_out_production",
        "production_delta", "production_ci_lower", "production_ci_upper", "total_demand",
        "Production (€)", "Total Demand (€)",
    }
    _PCT_COLS = {
        "actual_risk_pct", "optimum_risk_pct", "swap_in_risk_pct", "swap_out_risk_pct",
        "production_delta_pct", "risk_delta_pct", "risk_ci_lower", "risk_ci_upper",
        "actual_rejection_rate_pct", "optimum_rejection_rate_pct",
        "actual_risk_h3_pct", "optimum_risk_h3_pct", "swap_in_risk_h3_pct", "swap_out_risk_h3_pct",
        "Risk (%)", "Production (%)", "Rejection Rate (%)",
    }
    _INTEGER_COLS = {
        "n_segments",
        "actual_todu_30ever_h6", "actual_todu_amt_pile_h6",
        "optimum_todu_30ever_h6", "optimum_todu_amt_pile_h6",
        "swap_in_todu_30ever_h6", "swap_in_todu_amt_pile_h6",
        "swap_out_todu_30ever_h6", "swap_out_todu_amt_pile_h6",
        "actual_todu_30ever_h3", "actual_todu_amt_pile_h3",
        "optimum_todu_30ever_h3", "optimum_todu_amt_pile_h3",
        "swap_in_todu_30ever_h3", "swap_in_todu_amt_pile_h3",
        "swap_out_todu_30ever_h3", "swap_out_todu_amt_pile_h3",
    }
    _TEXT_COLS = {"group", "period", "scenario", "segments", "Metric", "segment"}
    _DELTA_COLS = {"production_delta", "production_delta_pct", "risk_delta_pct"}
    _CUTOFF_FIXED_COLS = frozenset({
        "accepted", "segment", "scenario", "risk_pct", "production",
        "production_ci_lower", "production_ci_upper", "risk_ci_lower", "risk_ci_upper",
    })

    _COLUMN_LABELS = {
        "group": "Group",
        "period": "Period",
        "scenario": "Scenario",
        "n_segments": "# Segments",
        "segments": "Segments",
        "actual_production": "Actual Production (€)",
        "actual_risk_pct": "Actual Risk (%)",
        "optimum_production": "Optimum Production (€)",
        "optimum_risk_pct": "Optimum Risk (%)",
        "production_delta": "Production Delta (€)",
        "production_delta_pct": "Production Delta (%)",
        "risk_delta_pct": "Risk Delta (%)",
        "swap_in_production": "Swap-In Production (€)",
        "swap_in_risk_pct": "Swap-In Risk (%)",
        "swap_out_production": "Swap-Out Production (€)",
        "swap_out_risk_pct": "Swap-Out Risk (%)",
        "total_demand": "Total Demand (€)",
        "actual_rejection_rate_pct": "Actual Rejection Rate (%)",
        "optimum_rejection_rate_pct": "Optimum Rejection Rate (%)",
        "production_ci_lower": "Production CI Lower (€)",
        "production_ci_upper": "Production CI Upper (€)",
        "risk_ci_lower": "Risk CI Lower (%)",
        "risk_ci_upper": "Risk CI Upper (%)",
    }

    # =====================================================================
    # Helpers
    # =====================================================================

    def _set_col_width(ws, col_idx, header_text, max_rows):
        letter = get_column_letter(col_idx)
        label_len = len(str(header_text))
        data_max = max(
            (len(str(ws.cell(row=r, column=col_idx).value or "")) for r in range(2, max_rows + 1)),
            default=0,
        )
        ws.column_dimensions[letter].width = min(max(label_len, data_max) + 3, 34)

    def _apply_number_format(cell, col_name):
        if col_name in _CURRENCY_COLS:
            cell.number_format = "#,##0"
        elif col_name in _PCT_COLS:
            cell.number_format = "0.00"
        elif col_name in _INTEGER_COLS:
            cell.number_format = "#,##0"

    def _style_table(ws, df_cols, *, header_row=1, highlight_total=True):
        """Apply full dashboard styling to a data table starting at header_row."""
        group_col_idx = None
        n_cols = len(df_cols)
        for col_idx, col_name in enumerate(df_cols, 1):
            if col_name == "group":
                group_col_idx = col_idx
            cell = ws.cell(row=header_row, column=col_idx)
            cell.value = _COLUMN_LABELS.get(col_name, col_name)
            cell.font = _FONT_HEADER
            cell.fill = _FILL_HEADER
            cell.alignment = _ALIGN_CENTER
            cell.border = _BORDER_HEADER
            _set_col_width(ws, col_idx, cell.value, ws.max_row)
            for r in range(header_row + 1, ws.max_row + 1):
                data_cell = ws.cell(row=r, column=col_idx)
                data_cell.font = _FONT_DATA
                data_cell.border = _BORDER_ALL
                data_cell.alignment = _ALIGN_LEFT if col_name in _TEXT_COLS else _ALIGN_RIGHT
                _apply_number_format(data_cell, col_name)

        ws.row_dimensions[header_row].height = 28
        delta_col_indices = {ci for ci, cn in enumerate(df_cols, 1) if cn in _DELTA_COLS}
        for r in range(header_row + 1, ws.max_row + 1):
            ws.row_dimensions[r].height = 20
            is_total = False
            if highlight_total and group_col_idx:
                is_total = str(ws.cell(row=r, column=group_col_idx).value or "") == "TOTAL"
            for col_idx in range(1, n_cols + 1):
                data_cell = ws.cell(row=r, column=col_idx)
                if is_total:
                    data_cell.fill = _FILL_TOTAL
                    data_cell.font = _FONT_TOTAL
                elif r % 2 == 0:
                    data_cell.fill = _FILL_STRIPE
                if col_idx in delta_col_indices:
                    val = data_cell.value
                    if isinstance(val, (int, float)):
                        col_name = list(df_cols)[col_idx - 1]
                        is_risk = col_name == "risk_delta_pct"
                        good = val <= 0 if is_risk else val >= 0
                        clr = _CLR_GOOD if good else _CLR_BAD
                        data_cell.font = Font(bold=is_total, color=clr, size=10, name=_FN)
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1).coordinate
        ws.auto_filter.ref = (
            f"{ws.cell(row=header_row, column=1).coordinate}"
            f":{ws.cell(row=ws.max_row, column=n_cols).coordinate}"
        )

    def _write_kpi_card(ws, row, col, label, value_str, delta_str=None, delta_positive=True):
        """Write a KPI card block: 2 rows x 2 columns."""
        val_cell = ws.cell(row=row, column=col)
        val_cell.value = value_str
        val_cell.font = _FONT_KPI_VALUE
        val_cell.fill = _FILL_KPI
        val_cell.alignment = Alignment(horizontal="center", vertical="bottom")
        val_cell.border = _ACCENT_LEFT
        if delta_str:
            dc = ws.cell(row=row, column=col + 1)
            dc.value = delta_str
            dc.font = _FONT_KPI_DELTA_POS if delta_positive else _FONT_KPI_DELTA_NEG
            dc.fill = _FILL_KPI
            dc.alignment = Alignment(horizontal="center", vertical="bottom")
            dc.border = _BORDER_ALL
        else:
            ws.cell(row=row, column=col + 1).fill = _FILL_KPI
            ws.cell(row=row, column=col + 1).border = _BORDER_ALL
        lc = ws.cell(row=row + 1, column=col)
        lc.value = label
        lc.font = _FONT_KPI_LABEL
        lc.fill = _FILL_KPI
        lc.alignment = Alignment(horizontal="center", vertical="top")
        lc.border = _ACCENT_LEFT
        ws.cell(row=row + 1, column=col + 1).fill = _FILL_KPI
        ws.cell(row=row + 1, column=col + 1).border = _BORDER_ALL

    def _write_exec_table(ws, df, start_row, section_title, *, n_table_cols=8):
        """Write a section header + styled table on the Executive Summary sheet.
        Returns the next free row below the table.
        """
        cols_list = list(df.columns)
        span = max(len(cols_list), n_table_cols)

        # Section header with left accent bar
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=span)
        hdr = ws.cell(row=start_row, column=1)
        hdr.value = f"  {section_title}"
        hdr.font = _FONT_SECTION
        hdr.fill = _FILL_SECTION
        hdr.alignment = _ALIGN_LEFT
        hdr.border = Border(left=Side(style="thick", color=_CLR_SECTION_BAR), bottom=_THIN)
        ws.row_dimensions[start_row].height = 28

        table_row = start_row + 1
        for ci, col_name in enumerate(cols_list, 1):
            c = ws.cell(row=table_row, column=ci)
            c.value = _COLUMN_LABELS.get(col_name, col_name)
            c.font = _FONT_HEADER
            c.fill = _FILL_HEADER
            c.alignment = _ALIGN_CENTER
            c.border = _BORDER_HEADER
        ws.row_dimensions[table_row].height = 28

        group_col_idx = cols_list.index("group") + 1 if "group" in cols_list else None
        for ri, (_, data_row) in enumerate(df.iterrows(), table_row + 1):
            ws.row_dimensions[ri].height = 20
            for ci, col_name in enumerate(cols_list, 1):
                cell = ws.cell(row=ri, column=ci)
                cell.value = data_row[col_name]
                cell.font = _FONT_DATA
                cell.border = _BORDER_ALL
                cell.alignment = _ALIGN_LEFT if col_name in _TEXT_COLS else _ALIGN_RIGHT
                _apply_number_format(cell, col_name)

            is_total = group_col_idx and str(ws.cell(row=ri, column=group_col_idx).value or "") == "TOTAL"
            for ci, col_name in enumerate(cols_list, 1):
                cell = ws.cell(row=ri, column=ci)
                if is_total:
                    cell.fill = _FILL_TOTAL
                    cell.font = _FONT_TOTAL
                elif ri % 2 == 0:
                    cell.fill = _FILL_STRIPE
                if col_name in _DELTA_COLS and isinstance(cell.value, (int, float)):
                    good = cell.value <= 0 if col_name == "risk_delta_pct" else cell.value >= 0
                    clr = _CLR_GOOD if good else _CLR_BAD
                    cell.font = Font(bold=bool(is_total), color=clr, size=10, name=_FN)

        for ci in range(1, len(cols_list) + 1):
            _set_col_width(ws, ci, ws.cell(row=table_row, column=ci).value, ws.max_row)
        return ws.max_row + 2

    def _write_single_pivot_grid(ws, pivot, col_var, row_var, start_row, col_offset=0):
        """Draw one pivot grid at (start_row, col_offset+1). Returns bottom row used."""
        c0 = col_offset + 1

        # Corner label
        corner = ws.cell(row=start_row, column=c0)
        corner.value = f"{row_var} \\ {col_var}"
        corner.font = _FONT_GRID_LABEL
        corner.fill = _FILL_GRID_HEADER
        corner.alignment = _ALIGN_CENTER
        corner.border = _BORDER_GRID
        ws.column_dimensions[get_column_letter(c0)].width = max(
            ws.column_dimensions[get_column_letter(c0)].width or 8, len(str(corner.value)) + 4
        )
        ws.row_dimensions[start_row].height = 26

        # Column headers
        for ci, col_val in enumerate(pivot.columns, c0 + 1):
            c = ws.cell(row=start_row, column=ci)
            c.value = col_val
            c.font = _FONT_GRID_HEADER
            c.fill = _FILL_GRID_HEADER
            c.alignment = _ALIGN_CENTER
            c.border = _BORDER_GRID
            ws.column_dimensions[get_column_letter(ci)].width = max(
                ws.column_dimensions[get_column_letter(ci)].width or 0, 7
            )

        # Data rows
        for ri, idx_val in enumerate(pivot.index, start_row + 1):
            rh = ws.cell(row=ri, column=c0)
            rh.value = idx_val
            rh.font = _FONT_GRID_HEADER
            rh.fill = _FILL_GRID_HEADER
            rh.alignment = _ALIGN_CENTER
            rh.border = _BORDER_GRID
            ws.row_dimensions[ri].height = 26

            for ci, col_val in enumerate(pivot.columns, c0 + 1):
                cell = ws.cell(row=ri, column=ci)
                val = pivot.loc[idx_val, col_val]
                if pd.isna(val):
                    cell.fill = _FILL_NA
                    cell.value = "—"
                    cell.font = Font(bold=False, color=_CLR_NEUTRAL_MID, size=10, name=_FN)
                elif val == 1:
                    cell.fill = _FILL_ACCEPT
                    cell.value = "A"
                    cell.font = _FONT_GRID_CELL
                else:
                    cell.fill = _FILL_REJECT
                    cell.value = "R"
                    cell.font = _FONT_GRID_CELL
                cell.alignment = _ALIGN_CENTER
                cell.border = _BORDER_GRID

        return start_row + len(pivot.index)

    def _sort_pivot(pivot):
        """Sort pivot index and columns numerically where possible."""
        try:
            pivot = pivot.sort_index(key=lambda x: pd.to_numeric(x, errors="coerce"))
        except (TypeError, ValueError):
            pass
        try:
            pivot = pivot[sorted(
                pivot.columns,
                key=lambda x: float(x) if str(x).replace(".", "").replace("-", "").isdigit() else x,
            )]
        except (TypeError, ValueError):
            pass
        return pivot

    def _write_acceptance_grid(ws, cutoff_df, seg_name, start_row, scenario="base"):
        """Draw coloured acceptance/rejection grids on *ws* starting at *start_row*.

        For N>2 variables (e.g. octroi_bin x efx_bin x income_bin), creates one
        sub-grid per slice of the 3rd+ variables, laid out side by side.
        Returns next free row.
        """
        variable_cols = [c for c in cutoff_df.columns if c not in _CUTOFF_FIXED_COLS]
        if "accepted" not in cutoff_df.columns or len(variable_cols) < 2:
            return start_row

        scen_df = cutoff_df
        if "scenario" in cutoff_df.columns:
            scen_df = cutoff_df[cutoff_df["scenario"] == scenario]
        if scen_df.empty:
            return start_row

        col_var = variable_cols[0]   # columns of each pivot
        row_var = variable_cols[1]   # rows of each pivot
        slice_vars = variable_cols[2:]  # additional dimensions (e.g. income_bin)

        # Section label with left accent bar
        slice_info = f"  —  sliced by {', '.join(slice_vars)}" if slice_vars else ""
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=12)
        lbl = ws.cell(row=start_row, column=1)
        lbl.value = f"  {seg_name}  |  {scenario.title()}{slice_info}"
        lbl.font = _FONT_SECTION
        lbl.fill = _FILL_SECTION
        lbl.alignment = _ALIGN_LEFT
        lbl.border = Border(left=Side(style="thick", color=_CLR_SECTION_BAR), bottom=_HAIR)
        ws.row_dimensions[start_row].height = 28
        start_row += 1

        # Build list of (label, slice_df) pairs
        if slice_vars:
            groups = scen_df.groupby(slice_vars, sort=True)
            group_items = list(groups)[:12]  # cap at 12 sub-grids
        else:
            group_items = [(None, scen_df)]

        # Lay out sub-grids side by side (horizontally)
        col_offset = 0
        grid_bottom = start_row
        for slice_key, slice_df in group_items:
            try:
                pivot = slice_df.pivot_table(index=row_var, columns=col_var, values="accepted", aggfunc="first")
                pivot = _sort_pivot(pivot)
            except (TypeError, ValueError, KeyError):
                continue
            if pivot.empty:
                continue

            # Slice label above the sub-grid
            if slice_key is not None:
                if isinstance(slice_key, tuple):
                    label = ", ".join(f"{v}={k}" for v, k in zip(slice_vars, slice_key))
                else:
                    label = f"{slice_vars[0]}={slice_key}"
                lbl_cell = ws.cell(row=start_row, column=col_offset + 1)
                lbl_cell.value = label
                lbl_cell.font = Font(bold=True, color=_CLR_ACCENT, size=9, name=_FN)
                lbl_cell.alignment = _ALIGN_LEFT
                grid_start = start_row + 1
            else:
                grid_start = start_row

            bottom = _write_single_pivot_grid(ws, pivot, col_var, row_var, grid_start, col_offset)
            grid_bottom = max(grid_bottom, bottom)

            # Advance column offset for next sub-grid (grid width + 1 gap column)
            col_offset += len(pivot.columns) + 2  # +1 for row header, +1 gap

        # Legend row
        legend_row = grid_bottom + 2
        legend_items = [
            ("  A  Accept  ", _FILL_ACCEPT, _CLR_WHITE),
            ("  R  Reject  ", _FILL_REJECT, _CLR_WHITE),
            ("  —  N/A  ", _FILL_NA, _CLR_TEXT),
        ]
        for ci, (label, fill, fg) in enumerate(legend_items, 1):
            c = ws.cell(row=legend_row, column=ci)
            c.value = label
            c.font = Font(bold=True, color=fg, size=9, name=_FN)
            c.fill = fill
            c.alignment = _ALIGN_CENTER
            c.border = _BORDER_GRID

        return legend_row + 2

    def _get_total_row(period: str, scenario: str = "base"):
        mask = (
            (consolidated_df["group"] == "TOTAL")
            & (consolidated_df["period"] == period)
            & (consolidated_df["scenario"] == scenario)
        )
        rows = consolidated_df[mask]
        return rows.iloc[0] if not rows.empty else None

    # =====================================================================
    # Build workbook
    # =====================================================================
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        if consolidated_df.empty:
            consolidated_df.to_excel(writer, sheet_name="Executive Summary", index=False)
            logger.debug("Empty consolidated DataFrame — wrote empty workbook")
            return xlsx_path

        wb = writer.book

        # =============================================================
        # Sheet 1: Executive Summary
        # =============================================================
        ws_exec = wb.create_sheet("Executive Summary", 0)
        ws_exec.sheet_properties.tabColor = _CLR_TAB_EXEC
        ws_exec.sheet_view.showGridLines = False

        # --- Title banner with coloured bar ---
        title_fill = PatternFill(start_color=_CLR_PRIMARY, end_color=_CLR_PRIMARY, fill_type="solid")
        ws_exec.merge_cells("A1:J1")
        title_cell = ws_exec["A1"]
        title_cell.value = "  Consolidated Risk & Production Dashboard"
        title_cell.font = Font(bold=True, color=_CLR_WHITE, size=20, name=_FN)
        title_cell.fill = title_fill
        title_cell.alignment = _ALIGN_LEFT
        ws_exec.row_dimensions[1].height = 44

        ws_exec.merge_cells("A2:J2")
        ws_exec["A2"].value = f"  Generated {date.today().strftime('%d %b %Y')}  |  {len(segments)} segment(s)"
        ws_exec["A2"].font = Font(bold=False, color=_CLR_ACCENT_LIGHT, size=11, name=_FN)
        ws_exec["A2"].fill = title_fill
        ws_exec["A2"].alignment = _ALIGN_LEFT
        ws_exec.row_dimensions[2].height = 22

        # Thin accent line under title bar
        for c in range(1, 11):
            ws_exec.cell(row=3, column=c).border = Border(top=Side(style="medium", color=_CLR_ACCENT))
        ws_exec.row_dimensions[3].height = 6

        # --- MAIN PERIOD KPI cards (rows 4-5) ---
        tr_main = _get_total_row("main")
        kpi_row = 4
        ws_exec.row_dimensions[kpi_row].height = 38
        ws_exec.row_dimensions[kpi_row + 1].height = 22

        if tr_main is not None:
            _write_kpi_card(ws_exec, kpi_row, 1, "Optimum Production",
                            f"€{tr_main.get('optimum_production', 0):,.0f}")
            pd_val = tr_main.get("production_delta", 0)
            _write_kpi_card(ws_exec, kpi_row, 3, "Production Delta",
                            f"€{pd_val:+,.0f}",
                            delta_str=f"{tr_main.get('production_delta_pct', 0):+.1f}%",
                            delta_positive=pd_val >= 0)
            rd = tr_main.get("risk_delta_pct", 0)
            _write_kpi_card(ws_exec, kpi_row, 5, "Optimum Risk",
                            f"{tr_main.get('optimum_risk_pct', 0):.2f}%",
                            delta_str=f"{rd:+.2f} pp", delta_positive=rd <= 0)
            _write_kpi_card(ws_exec, kpi_row, 7, "Rejection Rate",
                            f"{tr_main.get('optimum_rejection_rate_pct', 0):.1f}%")
            # Label
            ws_exec.cell(row=kpi_row, column=9).value = "MAIN PERIOD"
            ws_exec.cell(row=kpi_row, column=9).font = Font(bold=True, color=_CLR_ACCENT, size=9, name=_FN)
            ws_exec.cell(row=kpi_row, column=9).alignment = _ALIGN_LEFT
        else:
            ws_exec.cell(row=kpi_row, column=1).value = "No TOTAL/base/main data"
            ws_exec.cell(row=kpi_row, column=1).font = _FONT_SUBTITLE

        # --- MR PERIOD KPI cards (rows 6-7) ---
        tr_mr = _get_total_row("mr")
        mr_row = 6
        ws_exec.row_dimensions[mr_row].height = 38
        ws_exec.row_dimensions[mr_row + 1].height = 22

        if tr_mr is not None:
            _write_kpi_card(ws_exec, mr_row, 1, "MR Optimum Prod.",
                            f"€{tr_mr.get('optimum_production', 0):,.0f}")
            mr_pd = tr_mr.get("production_delta", 0)
            _write_kpi_card(ws_exec, mr_row, 3, "MR Prod. Delta",
                            f"€{mr_pd:+,.0f}",
                            delta_str=f"{tr_mr.get('production_delta_pct', 0):+.1f}%",
                            delta_positive=mr_pd >= 0)
            mr_rd = tr_mr.get("risk_delta_pct", 0)
            _write_kpi_card(ws_exec, mr_row, 5, "MR Optimum Risk",
                            f"{tr_mr.get('optimum_risk_pct', 0):.2f}%",
                            delta_str=f"{mr_rd:+.2f} pp", delta_positive=mr_rd <= 0)
            _write_kpi_card(ws_exec, mr_row, 7, "MR Rejection Rate",
                            f"{tr_mr.get('optimum_rejection_rate_pct', 0):.1f}%")
            ws_exec.cell(row=mr_row, column=9).value = "MR PERIOD"
            ws_exec.cell(row=mr_row, column=9).font = Font(bold=True, color=_CLR_MR_FG, size=9, name=_FN)
            ws_exec.cell(row=mr_row, column=9).alignment = _ALIGN_LEFT
            # Tint MR KPI row backgrounds
            for c in range(1, 9):
                for r in (mr_row, mr_row + 1):
                    cell = ws_exec.cell(row=r, column=c)
                    cell.fill = _FILL_MR
        else:
            ws_exec.cell(row=mr_row, column=1).value = "MR period data not available"
            ws_exec.cell(row=mr_row, column=1).font = Font(italic=True, color=_CLR_NEUTRAL_MID, size=10, name=_FN)

        # --- Spacer ---
        ws_exec.row_dimensions[8].height = 10
        next_row = 9

        # --- Main-period summary table ---
        exec_cols = [
            "group", "actual_production", "optimum_production", "production_delta",
            "production_delta_pct", "actual_risk_pct", "optimum_risk_pct", "risk_delta_pct",
        ]
        main_base_mask = (consolidated_df["period"] == "main") & (consolidated_df["scenario"] == "base")
        exec_main = consolidated_df.loc[main_base_mask, [c for c in exec_cols if c in consolidated_df.columns]].copy()
        if not exec_main.empty:
            next_row = _write_exec_table(ws_exec, exec_main, next_row, "Main Period — Base Scenario")

        # --- MR-period summary table ---
        mr_base_mask = (consolidated_df["period"] == "mr") & (consolidated_df["scenario"] == "base")
        exec_mr = consolidated_df.loc[mr_base_mask, [c for c in exec_cols if c in consolidated_df.columns]].copy()
        if not exec_mr.empty:
            next_row = _write_exec_table(ws_exec, exec_mr, next_row, "MR Period — Base Scenario")

        # --- Acceptance grids per segment on Executive Summary ---
        cutoff_data: dict[str, pd.DataFrame] = {}
        for seg_name in segments:
            csv_path = output_base / seg_name / "data" / "cutoff_summary_wide.csv"
            if not csv_path.exists():
                continue
            try:
                df_cut = pd.read_csv(csv_path)
            except (pd.errors.ParserError, OSError, ValueError):
                continue
            if df_cut.empty:
                continue
            if "segment" not in df_cut.columns:
                df_cut["segment"] = seg_name
            cutoff_data[seg_name] = df_cut

        if cutoff_data:
            for seg_name, df_cut in cutoff_data.items():
                next_row = _write_acceptance_grid(ws_exec, df_cut, seg_name, next_row)

        # Ensure KPI columns are wide enough
        for c in range(1, 11):
            cur = ws_exec.column_dimensions[get_column_letter(c)].width or 12
            ws_exec.column_dimensions[get_column_letter(c)].width = max(cur, 18)

        # =============================================================
        # Sheet 2: Portfolio Summary
        # =============================================================
        portfolio_mask = consolidated_df["group"].str.match(r"^(TOTAL|supersegment_)")
        portfolio_df = consolidated_df[portfolio_mask]
        if not portfolio_df.empty:
            portfolio_df.to_excel(writer, sheet_name="Portfolio Summary", index=False)
            _style_table(writer.sheets["Portfolio Summary"], portfolio_df.columns)
            writer.sheets["Portfolio Summary"].sheet_properties.tabColor = _CLR_TAB_PORTFOLIO

        # =============================================================
        # Sheet 3: Segment Detail
        # =============================================================
        segment_mask = ~consolidated_df["group"].str.match(r"^(TOTAL|supersegment_)")
        segment_df = consolidated_df[segment_mask]
        if not segment_df.empty:
            segment_df.to_excel(writer, sheet_name="Segment Detail", index=False)
            _style_table(writer.sheets["Segment Detail"], segment_df.columns)
            writer.sheets["Segment Detail"].sheet_properties.tabColor = _CLR_TAB_SEGMENT

        # =============================================================
        # Sheet 4: Cutoff Comparison (raw data table)
        # =============================================================
        if cutoff_data:
            all_cutoffs = pd.concat(cutoff_data.values(), ignore_index=True)
            all_cutoffs.to_excel(writer, sheet_name="Cutoff Comparison", index=False)
            _style_table(writer.sheets["Cutoff Comparison"], all_cutoffs.columns, highlight_total=False)
            writer.sheets["Cutoff Comparison"].sheet_properties.tabColor = _CLR_TAB_CUTOFF

        # =============================================================
        # Per-segment acceptance grid sheets
        # =============================================================
        for seg_name, df_cut in cutoff_data.items():
            sheet_name = f"Grid {seg_name}"[:31]
            ws_grid = wb.create_sheet(sheet_name)
            ws_grid.sheet_properties.tabColor = _CLR_TAB_GRID
            ws_grid.sheet_view.showGridLines = False

            # Title bar
            title_fill = PatternFill(start_color=_CLR_PRIMARY, end_color=_CLR_PRIMARY, fill_type="solid")
            ws_grid.merge_cells(start_row=1, start_column=1, end_row=1, end_column=14)
            t = ws_grid.cell(row=1, column=1)
            t.value = f"  Acceptance Grid — {seg_name}"
            t.font = Font(bold=True, color=_CLR_WHITE, size=16, name=_FN)
            t.fill = title_fill
            t.alignment = _ALIGN_LEFT
            ws_grid.row_dimensions[1].height = 36
            for gc in range(1, 15):
                ws_grid.cell(row=2, column=gc).border = Border(top=Side(style="medium", color=_CLR_ACCENT))
            ws_grid.row_dimensions[2].height = 4

            # Draw grids per scenario
            cur_row = 4
            for scen in ["pessimistic", "base", "optimistic"]:
                if "scenario" in df_cut.columns and scen in df_cut["scenario"].values:
                    cur_row = _write_acceptance_grid(ws_grid, df_cut, f"{scen.title()}", cur_row, scenario=scen)

        # =============================================================
        # Per-segment RP summary sheets
        # =============================================================
        for seg_name in segments:
            csv_path = output_base / seg_name / "data" / "risk_production_summary_table_base.csv"
            if not csv_path.exists():
                continue
            try:
                df_rp = pd.read_csv(csv_path)
            except (pd.errors.ParserError, OSError, ValueError):
                continue
            if df_rp.empty:
                continue
            sheet_name = f"RP {seg_name}"[:31]
            df_rp.to_excel(writer, sheet_name=sheet_name, index=False)
            _style_table(writer.sheets[sheet_name], df_rp.columns, highlight_total=False)
            writer.sheets[sheet_name].sheet_properties.tabColor = _CLR_TAB_SEGMENT

        # Remove default "Sheet" if auto-created
        if "Sheet" in wb.sheetnames and len(wb.sheetnames) > 1:
            del wb["Sheet"]

    logger.debug(f"Excel workbook written to {xlsx_path}")
    return xlsx_path


def print_consolidation_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary of consolidated metrics."""
    print("\n" + "=" * 80)
    print("CONSOLIDATED RISK PRODUCTION SUMMARY")
    print("=" * 80)

    # Get unique scenarios and periods
    scenarios = df["scenario"].unique()

    for scenario in scenarios:
        print(f"\n{'─' * 40}")
        print(f"SCENARIO: {scenario}")
        print(f"{'─' * 40}")

        scenario_df = df[df["scenario"] == scenario]

        # Show TOTAL rows
        total_df = scenario_df[scenario_df["group"] == "TOTAL"]

        for _, row in total_df.iterrows():
            period = row["period"].upper()
            print(f"\n  {period} Period:")
            print(f"    Actual Production:  €{row['actual_production']:,.0f}")
            print(f"    Optimum Production: €{row['optimum_production']:,.0f}")
            print(f"    Delta:              €{row['production_delta']:,.0f} ({row['production_delta_pct']:.1f}%)")
            print(f"    Risk:               {row['actual_risk_pct']:.2f}% → {row['optimum_risk_pct']:.2f}%")

        # Show supersegment breakdown
        ss_df = scenario_df[scenario_df["group"].str.startswith("supersegment_")]
        if not ss_df.empty:
            print("\n  By Supersegment (Main Period):")
            main_ss = ss_df[ss_df["period"] == "main"]
            for _, row in main_ss.iterrows():
                ss_name = row["group"].replace("supersegment_", "")
                print(
                    f"    {ss_name}: €{row['actual_production']:,.0f} → €{row['optimum_production']:,.0f} "
                    f"({row['production_delta_pct']:+.1f}%), Risk: {row['actual_risk_pct']:.2f}% → {row['optimum_risk_pct']:.2f}%"
                )

    print("\n" + "=" * 80)
