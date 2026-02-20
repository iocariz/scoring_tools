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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from plotly.subplots import make_subplots

from .utils import calculate_b2_ever_h6


@dataclass
class ConsolidatedMetrics:
    """Consolidated metrics for a group of segments."""

    group_name: str
    period: str  # 'main' or 'mr'
    scenario: str
    segments: list[str]

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

    # Confidence Intervals (for optimum solution)
    optimum_production_ci_lower: float = 0.0
    optimum_production_ci_upper: float = 0.0
    optimum_risk_ci_lower: float = 0.0
    optimum_risk_ci_upper: float = 0.0

    # Calculated risk properties (as percentage, e.g. 1.5 means 1.5%)
    @property
    def actual_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.actual_todu_30ever_h6, self.actual_todu_amt_pile_h6, as_percentage=True, decimals=6
                )
            )
        )

    @property
    def optimum_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.optimum_todu_30ever_h6, self.optimum_todu_amt_pile_h6, as_percentage=True, decimals=6
                )
            )
        )

    @property
    def swap_in_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_in_todu_30ever_h6, self.swap_in_todu_amt_pile_h6, as_percentage=True, decimals=6
                )
            )
        )

    @property
    def swap_out_risk(self) -> float:
        return float(
            np.nan_to_num(
                calculate_b2_ever_h6(
                    self.swap_out_todu_30ever_h6, self.swap_out_todu_amt_pile_h6, as_percentage=True, decimals=6
                )
            )
        )

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
        return {
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
        }


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
        "actual": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "optimum": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "swap_in": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "swap_out": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
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

    for col in df.columns:
        col_lower = col.lower()
        if "production" in col_lower and "€" in col_lower:
            prod_col = col
        elif col_lower == "todu_30ever_h6":
            todu_30_col = col
        elif col_lower == "todu_amt_pile_h6":
            todu_amt_col = col

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

    return metrics


def aggregate_metrics(metrics_list: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics from multiple segments.

    Sums production and raw todu values. Risk is calculated from aggregated
    todu values: risk = sum(todu_30ever_h6) / sum(todu_amt_pile_h6) * 7
    """
    aggregated = {
        "actual": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "optimum": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "swap_in": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
        "swap_out": {"production": 0, "todu_30ever_h6": 0, "todu_amt_pile_h6": 0},
    }

    for metrics in metrics_list:
        for key in aggregated:
            aggregated[key]["production"] += metrics[key]["production"]
            aggregated[key]["todu_30ever_h6"] += metrics[key]["todu_30ever_h6"]
            aggregated[key]["todu_amt_pile_h6"] += metrics[key]["todu_amt_pile_h6"]

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

        # Risk CIs: aggregate using exposure-weighted risk (already done for point estimate)
        combined_risk_se = np.sqrt(sum(se**2 for se in risk_ses))
        if combined_risk_se > 0:
            combined_risk_mean = sum(risk_means)
            aggregated[key]["risk_ci_lower"] = combined_risk_mean - z_95 * combined_risk_se
            aggregated[key]["risk_ci_upper"] = combined_risk_mean + z_95 * combined_risk_se
        else:
            aggregated[key]["risk_ci_lower"] = 0
            aggregated[key]["risk_ci_upper"] = 0

    return aggregated


def consolidate_segments(
    output_base: Path,
    segments: dict[str, dict[str, Any]],
    supersegments: dict[str, dict[str, Any]],
    scenarios: list[str] | None = None,
) -> pd.DataFrame:
    """
    Consolidate risk production tables across segments, supersegments, and scenarios.

    Args:
        output_base: Base output directory
        segments: Segment configurations
        supersegments: Supersegment configurations
        scenarios: List of scenario suffixes (e.g., ['', '_1.0', '_1.1'])

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
        if name not in seen_names or suffix:
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
                    agg = aggregate_metrics(ss_metrics_list)

                    consolidated = ConsolidatedMetrics(
                        group_name=f"supersegment_{ss_name}",
                        period=period,
                        scenario=scenario_name,
                        segments=ss_segments,
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
                        # Pass aggregated CIs (Production only)
                        optimum_production_ci_lower=agg["optimum"].get("production_ci_lower", 0),
                        optimum_production_ci_upper=agg["optimum"].get("production_ci_upper", 0),
                        # Risk CIs are 0 for aggregated
                    )
                    results.append(consolidated.to_dict())
                    supersegment_data[ss_name] = agg

            # Add all individual segments (including those in supersegments)
            for seg_name, metrics in segment_metrics.items():
                agg = aggregate_metrics([metrics])
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
                total_agg = aggregate_metrics(all_metrics_list)
                total_consolidated = ConsolidatedMetrics(
                    group_name="TOTAL",
                    period=period,
                    scenario=scenario_name,
                    segments=list(segment_metrics.keys()),
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
                    # Pass aggregated CIs (Production only)
                    optimum_production_ci_lower=total_agg["optimum"].get("production_ci_lower", 0),
                    optimum_production_ci_upper=total_agg["optimum"].get("production_ci_upper", 0),
                    # Risk CIs are 0 for aggregated
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
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Generate complete consolidation report with CSV and HTML dashboard.

    Args:
        output_base: Base output directory
        segments: Segment configurations
        supersegments: Supersegment configurations
        scenarios: List of scenario suffixes
        output_path: Optional output path for files (defaults to output_base)

    Returns:
        Tuple of (consolidated DataFrame, Plotly figure)
    """
    output_base = Path(output_base)
    output_path = Path(output_path) if output_path else output_base

    logger.info("Generating consolidated risk production report...")

    # Consolidate data
    df = consolidate_segments(output_base, segments, supersegments, scenarios)

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

    # Print summary
    print_consolidation_summary(df)

    return df, fig


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
