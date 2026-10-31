"""Cell-level acceptance grids (cutoff summaries) built from an optimal solution."""

import numpy as np
import pandas as pd
from loguru import logger

from src.optimize.optimization_utils import CellGrid


def generate_cutoff_summary(
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    segment_name: str,
    scenario_name: str = "base",
    risk_value: float | None = None,
    production_value: float | None = None,
    ci_data: dict[str, float] | None = None,
    mask: np.ndarray | None = None,
    grid: object | None = None,
) -> pd.DataFrame:
    """
    Generate a readable summary of cutoff points by segment.

    When ``mask``/``grid`` are provided (any number of variables): produces a
    cell-level table with one row per grid cell showing accepted/rejected —
    the format the consolidated acceptance-grid renderers consume.
    Without them: falls back to the classic 2-variable var0_bin → cutoff_value
    table derived from the optimal solution's bin columns.

    Args:
        optimal_solution_df: DataFrame containing the optimal solution with bin columns
        variables: List of variable names used in the optimization
        segment_name: Name of the segment being analyzed
        scenario_name: Name of the scenario (e.g., 'pessimistic', 'base', 'optimistic')
        risk_value: Optional risk (b2_ever_h6) value for this solution
        production_value: Optional production (oa_amt_h0) value for this solution
        ci_data: Optional confidence interval data dict
        mask: Optional binary acceptance mask (for N>2 cell-level summary)
        grid: Optional CellGrid (for N>2 cell-level summary)

    Returns:
        DataFrame with cutoff information.
    """
    if optimal_solution_df is None or optimal_solution_df.empty:
        logger.warning("No optimal solution provided for cutoff summary")
        return pd.DataFrame()

    # Cell-level summary from mask (any N — required by the consolidated
    # acceptance grids in the Excel/HTML reports, which need an `accepted`
    # column; the classic 2-var wide pivot has none, see audit on 2-var grids)
    if mask is not None and grid is not None:
        if isinstance(grid, CellGrid):
            cell_df = grid.cell_data[grid.variables].copy()
            cell_df["accepted"] = mask
            # Per-cell observed flag: False = phantom cell (no records in this
            # population). Mark its `accepted` as NA so the acceptance-grid
            # renderers show it as neutral "no data" rather than a risk-based
            # rejection (esp. on the smaller MR cohort / finer grids).
            observed = getattr(grid, "observed", None)
            if observed is not None and len(np.asarray(observed)) == len(cell_df):
                obs = np.asarray(observed, dtype=bool)
                cell_df["observed"] = obs.astype(int)
                cell_df.loc[~obs, "accepted"] = np.nan
            else:
                cell_df["observed"] = 1
            cell_df["segment"] = segment_name
            cell_df["scenario"] = scenario_name
            cell_df["risk_pct"] = risk_value
            cell_df["production"] = production_value
            if ci_data:
                cell_df["production_ci_lower"] = ci_data.get("production_ci_lower")
                cell_df["production_ci_upper"] = ci_data.get("production_ci_upper")
                cell_df["risk_ci_lower"] = ci_data.get("risk_ci_lower", 0)
                cell_df["risk_ci_upper"] = ci_data.get("risk_ci_upper", 0)
            logger.info(
                f"Generated cell-level cutoff summary for segment '{segment_name}', scenario '{scenario_name}' "
                f"({len(cell_df)} cells, {int(mask.sum())} accepted)"
            )
            return cell_df

    # 2-variable path: classic var0_bin → cutoff_value table
    var0_name = variables[0] if len(variables) > 0 else "var0"
    var1_name = variables[1] if len(variables) > 1 else "var1"

    # Get the first (selected) solution row
    opt_row = optimal_solution_df.iloc[0]

    # Find bin columns (numeric column names representing bins)
    bin_columns = []
    for col in optimal_solution_df.columns:
        try:
            bin_val = int(col) if isinstance(col, str) else col
            if isinstance(bin_val, (int, float)) and not pd.isna(bin_val):
                bin_columns.append((bin_val, col))
        except (ValueError, TypeError):
            continue

    # Sort bins numerically
    bin_columns.sort(key=lambda x: x[0])

    if not bin_columns:
        logger.warning("No bin columns found in optimal solution")
        return pd.DataFrame()

    # Build summary rows
    summary_rows = []
    for bin_val, col_name in bin_columns:
        cutoff = opt_row[col_name]

        if pd.notna(cutoff) and np.isfinite(cutoff):
            safe_cutoff = int(cutoff)
        elif pd.notna(cutoff):
            safe_cutoff = float(cutoff)  # Keep as inf or -inf
        else:
            safe_cutoff = None

        row_data = {
            "segment": segment_name,
            "scenario": scenario_name,
            f"{var0_name}_bin": int(bin_val),
            "var0_name": var0_name,
            "cutoff_value": safe_cutoff,
            "var1_name": var1_name,
            "risk_pct": risk_value,
            "production": production_value,
        }

        if ci_data:
            row_data["production_ci_lower"] = ci_data.get("production_ci_lower")
            row_data["production_ci_upper"] = ci_data.get("production_ci_upper")
            row_data["risk_ci_lower"] = ci_data.get("risk_ci_lower", 0)
            row_data["risk_ci_upper"] = ci_data.get("risk_ci_upper", 0)

        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)

    logger.info(f"Generated cutoff summary for segment '{segment_name}', scenario '{scenario_name}'")
    logger.info(
        f"  Bins: {len(bin_columns)}, Cutoff range: "
        f"[{summary_df['cutoff_value'].min()}, {summary_df['cutoff_value'].max()}]"
    )

    return summary_df


def format_cutoff_summary_table(
    cutoff_summary: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Format cutoff summary into a wide pivot table for easier reading.

    Args:
        cutoff_summary: DataFrame from generate_cutoff_summary
        variables: List of variable names [var0, var1]

    Returns:
        Pivoted DataFrame with segments/scenarios as rows and bins as columns
    """
    if cutoff_summary.empty:
        return pd.DataFrame()

    var0_name = variables[0] if len(variables) > 0 else "var0"
    bin_col = f"{var0_name}_bin"

    # Build pivot index — include CI columns if present so they survive pivot
    index_cols = ["segment", "scenario", "risk_pct", "production"]
    ci_cols = ["production_ci_lower", "production_ci_upper", "risk_ci_lower", "risk_ci_upper"]
    available_ci = [c for c in ci_cols if c in cutoff_summary.columns]
    index_cols.extend(available_ci)

    # Pivot to wide format
    pivot_df = cutoff_summary.pivot_table(
        index=index_cols, columns=bin_col, values="cutoff_value", aggfunc="first"
    ).reset_index()

    # Rename bin columns to be more readable
    pivot_df.columns = [f"bin_{col}" if isinstance(col, (int, float)) else col for col in pivot_df.columns]

    return pivot_df


def consolidate_cutoff_summaries(
    summaries: list[pd.DataFrame],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Consolidate multiple cutoff summaries into a single DataFrame.

    Args:
        summaries: List of DataFrames from generate_cutoff_summary
        output_path: Optional path to save the consolidated summary as CSV

    Returns:
        Consolidated DataFrame with all summaries
    """
    if not summaries:
        logger.warning("No summaries provided for consolidation")
        return pd.DataFrame()

    # Filter out empty DataFrames
    valid_summaries = [s for s in summaries if not s.empty]

    if not valid_summaries:
        logger.warning("All provided summaries are empty")
        return pd.DataFrame()

    consolidated = pd.concat(valid_summaries, ignore_index=True)

    if output_path:
        consolidated.to_csv(output_path, index=False)
        logger.info(f"Consolidated cutoff summary saved to {output_path}")

    return consolidated
