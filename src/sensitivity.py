"""
Sensitivity analysis and marginal impact for cutoff optimization.

Provides:
- Risk perturbation of summary data
- Sensitivity analysis: how cutoffs change under risk perturbations
- Cell-level flip thresholds
- Analytical marginal impact per cell (O(N))
"""

import numpy as np
import pandas as pd

from .optimization_utils import CellGrid, evaluate_solution, mask_to_cutoffs, milp_solve_cutoffs
from .utils import calculate_b2_ever_h6


def perturb_risk_summary(
    data_summary: pd.DataFrame,
    perturbation_pct: float,
) -> pd.DataFrame:
    """Scale todu_30ever_h6_rep by (1 + pct/100), recompute base column.

    Other indicators (_rep for oa_amt_h0, todu_amt_pile_h6) are unchanged.

    Args:
        data_summary: Aggregated summary with _boo and _rep columns.
        perturbation_pct: Percentage change (e.g. 10 means +10%).

    Returns:
        Copy of data_summary with perturbed risk columns.
    """
    df = data_summary.copy()
    factor = 1.0 + perturbation_pct / 100.0

    # Perturb the _rep risk column
    if "todu_30ever_h6_rep" in df.columns:
        df["todu_30ever_h6_rep"] = df["todu_30ever_h6_rep"] * factor

    # Recompute base = _boo + _rep
    if "todu_30ever_h6_boo" in df.columns and "todu_30ever_h6_rep" in df.columns:
        df["todu_30ever_h6"] = df["todu_30ever_h6_boo"] + df["todu_30ever_h6_rep"]

    return df


def run_sensitivity_analysis(
    data_summary: pd.DataFrame,
    variables: list[str],
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    baseline_mask: np.ndarray,
    risk_target: float,
    perturbation_levels: list[float] | None = None,
) -> pd.DataFrame:
    """For each perturbation level, perturb risk, re-solve MILP, compare mask to baseline.

    Args:
        data_summary: Aggregated summary data.
        variables: Variable names for grid.
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier.
        indicators: Indicator column names.
        baseline_mask: Binary mask from the baseline solution.
        risk_target: Risk target (%).
        perturbation_levels: List of perturbation percentages.

    Returns:
        DataFrame with columns: perturbation_pct, n_flipped, n_accept_to_reject,
        n_reject_to_accept, new_production, new_risk.
    """
    if perturbation_levels is None:
        perturbation_levels = [-20, -10, -5, 5, 10, 20]

    # Get var0 bin values for cutoff columns
    base_grid = CellGrid.from_summary(data_summary, variables)
    v0_bins = sorted(base_grid.values_per_var[variables[0]]) if len(variables) >= 2 else []

    results = []
    for pct in perturbation_levels:
        perturbed = perturb_risk_summary(data_summary, pct)
        grid = CellGrid.from_summary(perturbed, variables)
        new_mask = milp_solve_cutoffs(grid, risk_target, inv_vars, multiplier)

        if new_mask is None:
            row = {
                "perturbation_pct": pct,
                "n_flipped": None,
                "n_accept_to_reject": None,
                "n_reject_to_accept": None,
                "new_production": None,
                "new_risk": None,
            }
            for v0 in v0_bins:
                row[f"cutoff_{v0}"] = None
            results.append(row)
            continue

        flipped = baseline_mask != new_mask
        accept_to_reject = ((baseline_mask == 1) & (new_mask == 0)).sum()
        reject_to_accept = ((baseline_mask == 0) & (new_mask == 1)).sum()

        kpis = evaluate_solution(new_mask, grid, indicators, multiplier)

        # Recompute risk at higher precision from raw sums
        new_risk = float(
            calculate_b2_ever_h6(
                kpis.get("todu_30ever_h6", 0.0),
                kpis.get("todu_amt_pile_h6", 0.0),
                multiplier=multiplier,
                as_percentage=True,
                decimals=3,
            )
        )

        # Extract cutoff points from the new mask
        cutoffs = mask_to_cutoffs(new_mask, grid, inv_vars)
        cut_map = cutoffs.get(variables[0], {}) if len(variables) >= 2 else {}

        row = {
            "perturbation_pct": pct,
            "n_flipped": int(flipped.sum()),
            "n_accept_to_reject": int(accept_to_reject),
            "n_reject_to_accept": int(reject_to_accept),
            "new_production": kpis.get("oa_amt_h0", 0.0),
            "new_risk": new_risk,
        }
        for v0 in v0_bins:
            row[f"cutoff_{v0}"] = cut_map.get(float(v0))
        results.append(row)

    return pd.DataFrame(results)


def sensitivity_cell_detail(
    data_summary: pd.DataFrame,
    variables: list[str],
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    baseline_mask: np.ndarray,
    risk_target: float,
    perturbation_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Per-cell: minimum perturbation that flips its status.

    Args:
        data_summary: Aggregated summary data.
        variables: Variable names for grid.
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier.
        indicators: Indicator column names.
        baseline_mask: Binary mask from baseline solution.
        risk_target: Risk target (%).
        perturbation_levels: Sorted list of perturbation percentages to test.

    Returns:
        DataFrame with columns: var0, var1, ..., baseline_status, flip_threshold_pct,
        flip_direction.
    """
    if perturbation_levels is None:
        perturbation_levels = [-20, -10, -5, 5, 10, 20]

    # Sort levels by absolute value to find minimum flip threshold
    sorted_levels = sorted(perturbation_levels, key=abs)

    # Build baseline grid to get cell coordinates
    baseline_grid = CellGrid.from_summary(data_summary, variables)

    # Pre-compute masks for each perturbation level
    masks_by_level: dict[float, np.ndarray | None] = {}
    for pct in sorted_levels:
        perturbed = perturb_risk_summary(data_summary, pct)
        grid = CellGrid.from_summary(perturbed, variables)
        masks_by_level[pct] = milp_solve_cutoffs(grid, risk_target, inv_vars, multiplier)

    # Build per-cell results
    rows = []
    for combo, flat_idx in baseline_grid.cell_index.items():
        baseline_status = int(baseline_mask[flat_idx])
        flip_threshold = None
        flip_direction = None

        for pct in sorted_levels:
            new_mask = masks_by_level[pct]
            if new_mask is None:
                continue
            if new_mask[flat_idx] != baseline_status:
                flip_threshold = pct
                flip_direction = "accept_to_reject" if baseline_status == 1 else "reject_to_accept"
                break

        row = {variables[d]: combo[d] for d in range(len(variables))}
        row["baseline_status"] = baseline_status
        row["flip_threshold_pct"] = flip_threshold
        row["flip_direction"] = flip_direction
        rows.append(row)

    return pd.DataFrame(rows)


def compute_cell_marginal_impact(
    grid: CellGrid,
    baseline_mask: np.ndarray,
    indicators: list[str],
    multiplier: float,
) -> pd.DataFrame:
    """Marginal EUR/risk impact of flipping each cell.

    Uses analytical formula (O(N) not O(N^2)): compute baseline sums once,
    then per cell adjust the ratio by adding/subtracting the cell's contribution.

    Args:
        grid: CellGrid with per-cell data.
        baseline_mask: Binary acceptance mask.
        indicators: Indicator column names.
        multiplier: Risk multiplier.

    Returns:
        DataFrame with columns: var0, var1, ..., status, delta_production,
        delta_risk_pct, cell_production, cell_risk.
    """
    cell = grid.cell_data
    accepted = baseline_mask.astype(bool)

    # Baseline sums (accepted cells only)
    base_prod = float(cell.loc[accepted, "oa_amt_h0"].sum())
    base_t30 = float(cell.loc[accepted, "todu_30ever_h6"].sum())
    base_tamt = float(cell.loc[accepted, "todu_amt_pile_h6"].sum())
    base_risk = float(calculate_b2_ever_h6(base_t30, base_tamt, multiplier=multiplier, as_percentage=True))

    rows = []
    for combo, flat_idx in grid.cell_index.items():
        cell_row = cell.iloc[flat_idx]
        status = int(baseline_mask[flat_idx])
        cell_prod = float(cell_row["oa_amt_h0"])
        cell_t30 = float(cell_row["todu_30ever_h6"])
        cell_tamt = float(cell_row["todu_amt_pile_h6"])

        # Flip: if currently accepted, remove it; if rejected, add it
        if status == 1:
            # Remove cell from accepted set
            new_prod = base_prod - cell_prod
            new_t30 = base_t30 - cell_t30
            new_tamt = base_tamt - cell_tamt
        else:
            # Add cell to accepted set
            new_prod = base_prod + cell_prod
            new_t30 = base_t30 + cell_t30
            new_tamt = base_tamt + cell_tamt

        new_risk = float(calculate_b2_ever_h6(new_t30, new_tamt, multiplier=multiplier, as_percentage=True))

        delta_prod = new_prod - base_prod
        delta_risk = new_risk - base_risk if not (np.isnan(new_risk) or np.isnan(base_risk)) else 0.0

        # Per-cell risk (for information)
        cell_risk = float(calculate_b2_ever_h6(cell_t30, cell_tamt, multiplier=multiplier, as_percentage=True))

        row = {grid.variables[d]: combo[d] for d in range(len(grid.variables))}
        row.update(
            {
                "status": status,
                "delta_production": delta_prod,
                "delta_risk_pct": delta_risk,
                "cell_production": cell_prod,
                "cell_risk": cell_risk,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
