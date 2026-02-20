"""
Optimization utilities for credit risk scoring.

This module provides:
- MILP-based N-variable Pareto frontier optimization (via scipy.optimize.milp)
- Optional pymoo GA fallback for very large grids
- Fixed cutoff creation and validation
- CellGrid helper for N-dimensional bin grids
"""

from dataclasses import dataclass, field
from itertools import combinations_with_replacement, product

import numpy as np
import pandas as pd
from loguru import logger
from scipy import sparse
from scipy.optimize import LinearConstraint, milp

from .constants import DEFAULT_RISK_MULTIPLIER
from .utils import calculate_b2_ever_h6

# =============================================================================
# CellGrid — builds the N-dimensional grid from data_summary_desagregado
# =============================================================================


@dataclass
class CellGrid:
    """N-dimensional bin grid with per-cell KPI data.

    Attributes:
        variables: Ordered variable names.
        values_per_var: Dict mapping var name → sorted unique bin values.
        cell_index: Dict mapping cell coordinate tuple → flat index.
        cell_data: DataFrame with one row per cell, ordered by flat index.
        shape: Tuple of bin counts per variable.
    """

    variables: list[str]
    values_per_var: dict[str, list]
    cell_index: dict[tuple, int] = field(default_factory=dict, repr=False)
    cell_data: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    shape: tuple[int, ...] = ()

    @classmethod
    def from_summary(cls, data_summary_desagregado: pd.DataFrame, variables: list[str]) -> "CellGrid":
        """Build grid from aggregated summary data."""
        values_per_var = {var: sorted(data_summary_desagregado[var].unique()) for var in variables}
        shape = tuple(len(values_per_var[v]) for v in variables)

        # Build cell_index: tuple of bin values → flat int
        all_combos = list(product(*(values_per_var[v] for v in variables)))
        cell_index = {combo: idx for idx, combo in enumerate(all_combos)}

        # Build cell_data aligned to flat index
        # Left-join all combos with actual data to handle missing cells (fill 0)
        grid_df = pd.DataFrame(all_combos, columns=variables)
        cell_data = grid_df.merge(data_summary_desagregado, on=variables, how="left").fillna(0)

        return cls(
            variables=variables,
            values_per_var=values_per_var,
            cell_index=cell_index,
            cell_data=cell_data,
            shape=shape,
        )

    @property
    def n_cells(self) -> int:
        return len(self.cell_index)


# =============================================================================
# Monotonicity constraints
# =============================================================================


def _build_monotonicity_constraints(grid: CellGrid, inv_vars: list[str]) -> sparse.csc_matrix:
    """Build sparse A matrix for monotonicity: A @ x <= 0.

    For each dimension d and each pair of adjacent cells along d:
      x[riskier_cell] - x[safer_cell] <= 0

    This ensures riskier cells are rejected before safer ones: if a safer
    cell is rejected (x=0), all riskier cells must also be rejected.

    Direction convention:
      - Default (variable NOT in inv_vars): higher bin index = riskier.
      - Inverted (variable IN inv_vars): higher bin index = safer.

    For the legacy 2-variable setup:
      - var0 (internal score, BinConfig.invert=False): higher bin = safer → IN inv_vars
      - var1 (external score, BinConfig.invert=True): higher bin = riskier → NOT in inv_vars

    Returns:
        Sparse CSC matrix of shape (n_constraints, n_cells).
    """
    rows, cols, vals = [], [], []
    constraint_idx = 0

    for d, var in enumerate(grid.variables):
        n_vals = len(grid.values_per_var[var])
        inverted = var in inv_vars

        # Iterate over all cells, build constraint with neighbor along dim d
        for combo, flat_idx in grid.cell_index.items():
            pos_in_dim = grid.values_per_var[var].index(combo[d])
            if pos_in_dim + 1 >= n_vals:
                continue  # no neighbor

            # Build neighbor combo
            neighbor_list = list(combo)
            neighbor_list[d] = grid.values_per_var[var][pos_in_dim + 1]
            neighbor = tuple(neighbor_list)
            neighbor_idx = grid.cell_index.get(neighbor)
            if neighbor_idx is None:
                continue

            # Convention: higher index along dimension = riskier (unless inverted)
            # Constraint: x[riskier] <= x[safer]
            # i.e. x[riskier] - x[safer] <= 0
            if inverted:
                # Higher bin = safer → riskier is lower bin (flat_idx)
                riskier_idx = flat_idx
                safer_idx = neighbor_idx
            else:
                # Higher bin = riskier → riskier is neighbor
                riskier_idx = neighbor_idx
                safer_idx = flat_idx

            rows.extend([constraint_idx, constraint_idx])
            cols.extend([riskier_idx, safer_idx])
            vals.extend([1.0, -1.0])
            constraint_idx += 1

    if constraint_idx == 0:
        return sparse.csc_matrix((0, grid.n_cells))

    return sparse.csc_matrix(
        (vals, (rows, cols)),
        shape=(constraint_idx, grid.n_cells),
    )


# =============================================================================
# MILP solver
# =============================================================================


def milp_solve_cutoffs(
    grid: CellGrid,
    target_risk: float,
    inv_vars: list[str],
    multiplier: float,
) -> np.ndarray | None:
    """Solve single MILP: maximize production subject to risk budget + monotonicity.

    Args:
        grid: CellGrid with cell_data containing per-cell KPI columns.
        target_risk: Max allowed risk (percentage, e.g. 1.5 means 1.5%).
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier (typically 7).

    Returns:
        Binary mask array (1=accept, 0=reject) for each cell, or None if infeasible.
    """
    n = grid.n_cells
    cell = grid.cell_data

    # Objective: maximize production = minimize -production
    production = cell["oa_amt_h0"].values.astype(float)
    c = -production

    # Risk budget constraint (linearized ratio):
    #   multiplier * sum(todu_30ever_h6[i] * x[i]) / sum(todu_amt_pile_h6[i] * x[i]) <= target_risk/100
    # Linearized:
    #   sum((multiplier * todu_30ever_h6[i] - target_risk/100 * todu_amt_pile_h6[i]) * x[i]) <= 0
    todu_30 = cell["todu_30ever_h6"].values.astype(float)
    todu_amt = cell["todu_amt_pile_h6"].values.astype(float)
    risk_coeffs = multiplier * todu_30 - (target_risk / 100.0) * todu_amt

    # Monotonicity constraints: A_mono @ x <= 0
    A_mono = _build_monotonicity_constraints(grid, inv_vars)

    # Stack risk constraint row on top of monotonicity
    risk_row = sparse.csc_matrix(risk_coeffs.reshape(1, -1))
    if A_mono.shape[0] > 0:
        A = sparse.vstack([risk_row, A_mono], format="csc")
    else:
        A = risk_row

    n_constraints = A.shape[0]
    b_u = np.zeros(n_constraints)
    b_l = np.full(n_constraints, -np.inf)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(n)  # all binary

    from scipy.optimize import Bounds

    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": 30},
    )

    if not result.success:
        return None

    return np.round(result.x).astype(int)


# =============================================================================
# KPI evaluation
# =============================================================================


def evaluate_solution(
    mask: np.ndarray,
    grid: CellGrid,
    indicators: list[str],
    multiplier: float,
) -> dict:
    """Compute KPIs for a single acceptance mask.

    For accepted cells: sums base, _boo, and _rep columns.
    Computes _cut = total_boo - accepted_boo for each indicator.
    Computes b2_ever_h6 for each suffix group.
    """
    cell = grid.cell_data
    accepted = mask.astype(bool)

    result: dict = {}

    # Sum base, _boo, _rep columns for accepted cells
    for suffix in ["", "_boo", "_rep"]:
        for ind in indicators:
            col = f"{ind}{suffix}" if suffix else ind
            if col in cell.columns:
                result[col] = float(cell.loc[accepted, col].sum())

    # Compute _cut = total_boo - accepted_boo for each indicator
    for ind in indicators:
        boo_col = f"{ind}_boo"
        if boo_col in cell.columns:
            total_boo = float(cell[boo_col].sum())
            accepted_boo = result.get(boo_col, 0.0)
            result[f"{ind}_cut"] = max(total_boo - accepted_boo, 0.0)

    # Compute b2_ever_h6 for each suffix
    for suffix in ["", "_boo", "_rep", "_cut"]:
        t30 = f"todu_30ever_h6{suffix}"
        tamt = f"todu_amt_pile_h6{suffix}"
        if t30 in result and tamt in result:
            result[f"b2_ever_h6{suffix}"] = float(
                calculate_b2_ever_h6(result[t30], result[tamt], multiplier=multiplier, as_percentage=True)
            )

    return result


# =============================================================================
# Pareto frontier
# =============================================================================


def trace_pareto_frontier(
    data_summary_desagregado: pd.DataFrame,
    variables: list[str],
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    n_points: int = 50,
) -> tuple[pd.DataFrame, CellGrid, list[np.ndarray]]:
    """Sweep risk targets, solve MILP at each, filter to Pareto-optimal set.

    Args:
        data_summary_desagregado: Aggregated data by variable combinations.
        variables: List of variable names.
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier.
        indicators: Indicator column names.
        n_points: Number of risk targets to sweep.

    Returns:
        Tuple of (pareto_df, grid, masks) where pareto_df has KPI columns for each
        Pareto-optimal solution, grid is the CellGrid used, and masks is a list
        of binary acceptance masks (one per Pareto solution).
    """
    grid = CellGrid.from_summary(data_summary_desagregado, variables)

    # Determine risk sweep range
    # Max risk = all cells accepted
    all_t30 = grid.cell_data["todu_30ever_h6"].sum()
    all_tamt = grid.cell_data["todu_amt_pile_h6"].sum()
    max_risk = float(calculate_b2_ever_h6(all_t30, all_tamt, multiplier=multiplier, as_percentage=True))
    if np.isnan(max_risk) or max_risk <= 0:
        max_risk = 20.0  # fallback

    risk_targets = np.linspace(0.01, max_risk * 1.1, n_points)

    solutions = []
    all_masks: list[np.ndarray] = []
    seen_masks: set[tuple] = set()

    logger.info(f"MILP Pareto sweep: {n_points} risk targets in [0.01, {max_risk * 1.1:.2f}%]")

    for target in risk_targets:
        mask = milp_solve_cutoffs(grid, target, inv_vars, multiplier)
        if mask is None:
            continue

        mask_key = tuple(mask.tolist())
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)

        kpis = evaluate_solution(mask, grid, indicators, multiplier)
        solutions.append(kpis)
        all_masks.append(mask)

    if not solutions:
        logger.warning("No feasible MILP solutions found. Attempting GA fallback...")
        return _ga_pareto_fallback(grid, inv_vars, multiplier, indicators, n_points)

    df = pd.DataFrame(solutions)

    # Sort by risk, keep track of original indices for mask alignment
    sort_idx = df["b2_ever_h6"].argsort().values
    df = df.iloc[sort_idx].reset_index(drop=True)
    all_masks = [all_masks[i] for i in sort_idx]

    # Pareto filter: for ascending risk, production must be non-decreasing
    cummax = df["oa_amt_h0"].cummax()
    pareto_mask = df["oa_amt_h0"] >= cummax
    pareto_indices = pareto_mask[pareto_mask].index.tolist()

    df = df[pareto_mask].reset_index(drop=True)
    pareto_masks = [all_masks[i] for i in pareto_indices]

    logger.info(f"Pareto frontier: {len(df)} solutions (from {len(solutions)} unique MILP solves)")

    return df, grid, pareto_masks


# =============================================================================
# Mask → cutoff conversion (for backward-compat visualization / bootstrap)
# =============================================================================


def mask_to_cutoffs(
    mask: np.ndarray,
    grid: CellGrid,
    inv_vars: list[str],
) -> dict[str, dict]:
    """Convert binary mask to per-variable cutoff values.

    For 2-variable case, returns the classic var0-bin → var1-cutoff mapping.
    For N-variable case, returns per-dimension cutoff dicts.

    Returns:
        Dict mapping each variable name to a dict of bin_value → cutoff_limit.
    """
    if len(grid.variables) != 2:
        # N>2: for each dimension, find the max accepted bin for each
        # combination of the other dimensions. This is a simplification.
        logger.info("mask_to_cutoffs: N>2 variable case, returning per-dimension projections")

    result = {}
    # For 2-var case, build classic cut_map: var0_bin -> max var1 accepted
    if len(grid.variables) == 2:
        var0, var1 = grid.variables
        v0_vals = grid.values_per_var[var0]
        v1_vals = grid.values_per_var[var1]

        cut_map = {}
        for v0 in v0_vals:
            # Find max var1 value that is accepted for this var0
            max_v1 = None
            for v1 in v1_vals:
                combo = (v0, v1)
                idx = grid.cell_index.get(combo)
                if idx is not None and mask[idx] == 1:
                    if var1 in inv_vars:
                        # For inverted: find min accepted
                        if max_v1 is None or v1 < max_v1:
                            max_v1 = v1
                    else:
                        if max_v1 is None or v1 > max_v1:
                            max_v1 = v1

            if max_v1 is None:
                max_v1 = np.inf if var1 in inv_vars else -np.inf

            cut_map[float(v0)] = float(max_v1)

        result[var0] = cut_map
    else:
        # For each variable, project the mask
        for d, var in enumerate(grid.variables):
            cut_map = {}
            for val in grid.values_per_var[var]:
                # Check if any cell with this value is accepted
                accepted = False
                for combo, idx in grid.cell_index.items():
                    if combo[d] == val and mask[idx] == 1:
                        accepted = True
                        break
                cut_map[float(val)] = 1.0 if accepted else 0.0
            result[var] = cut_map

    return result


def mask_to_solution_df(
    mask: np.ndarray,
    grid: CellGrid,
    inv_vars: list[str],
    kpis: dict,
    sol_fac: int = 0,
) -> pd.DataFrame:
    """Convert a mask + KPIs into a solution DataFrame compatible with legacy format.

    For 2-var case: columns are [sol_fac, bin1, bin2, ..., KPI columns].
    """
    cutoffs = mask_to_cutoffs(mask, grid, inv_vars)

    if len(grid.variables) == 2:
        var0 = grid.variables[0]
        row_data = {"sol_fac": sol_fac}
        for bin_val, cutoff_val in cutoffs[var0].items():
            row_data[bin_val] = cutoff_val
        # Add KPI columns
        for k, v in kpis.items():
            if k not in ("_mask", "_mask_arr") and not isinstance(v, np.ndarray):
                row_data[k] = v
        return pd.DataFrame([row_data])
    else:
        # N-var: store acceptance mask info
        row_data = {"sol_fac": sol_fac}
        for k, v in kpis.items():
            if k not in ("_mask", "_mask_arr") and not isinstance(v, np.ndarray):
                row_data[k] = v
        return pd.DataFrame([row_data])


def add_bin_columns(
    pareto_df: pd.DataFrame,
    masks: list[np.ndarray],
    grid: CellGrid,
    inv_vars: list[str],
) -> pd.DataFrame:
    """Add per-var0-bin cutoff columns to Pareto DataFrame for 2-var backward compat.

    For 2-var case: adds a column per var0 bin value, where the value is the max
    accepted var1 bin for that var0 bin. Also adds sol_fac column.

    For N>2: adds sol_fac column only (bin columns not applicable).

    Args:
        pareto_df: DataFrame with KPI columns from trace_pareto_frontier.
        masks: List of binary acceptance masks, one per row.
        grid: CellGrid used during optimization.
        inv_vars: Variables with inverted risk ordering.

    Returns:
        DataFrame with bin columns and sol_fac added.
    """
    df = pareto_df.copy()

    if len(grid.variables) == 2:
        var0 = grid.variables[0]
        v0_vals = grid.values_per_var[var0]

        # Initialize bin columns
        for v0 in v0_vals:
            df[float(v0)] = 0.0

        for i, mask in enumerate(masks):
            cutoffs = mask_to_cutoffs(mask, grid, inv_vars)
            if var0 in cutoffs:
                for bin_val, cutoff_val in cutoffs[var0].items():
                    df.at[i, bin_val] = cutoff_val

    # Add sol_fac column at the beginning
    df.insert(0, "sol_fac", range(len(df)))

    return df


# =============================================================================
# GA fallback (pymoo)
# =============================================================================


def _ga_pareto_fallback(
    grid: CellGrid,
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    n_points: int,
) -> tuple[pd.DataFrame, CellGrid, list[np.ndarray]]:
    """Fallback GA solver using pymoo. Returns empty DataFrame if pymoo not available."""
    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize
        from pymoo.termination import get_termination
    except ImportError:
        logger.warning("pymoo not installed. Cannot use GA fallback. Returning empty Pareto set.")
        return pd.DataFrame(), grid, []

    logger.info("Running GA fallback for Pareto frontier...")

    cell = grid.cell_data
    todu_30 = cell["todu_30ever_h6"].values.astype(float)
    todu_amt = cell["todu_amt_pile_h6"].values.astype(float)
    production = cell["oa_amt_h0"].values.astype(float)
    A_mono = _build_monotonicity_constraints(grid, inv_vars)

    all_t30 = todu_30.sum()
    all_tamt = todu_amt.sum()
    max_risk = float(calculate_b2_ever_h6(all_t30, all_tamt, multiplier=multiplier, as_percentage=True))
    if np.isnan(max_risk) or max_risk <= 0:
        max_risk = 20.0

    solutions = []
    all_masks: list[np.ndarray] = []
    seen_masks: set[tuple] = set()

    for target in np.linspace(0.01, max_risk * 1.1, n_points):
        risk_coeffs = multiplier * todu_30 - (target / 100.0) * todu_amt

        class CutoffProblem(Problem):
            def __init__(self, _risk_coeffs=risk_coeffs):
                super().__init__(n_var=grid.n_cells, n_obj=1, n_ieq_constr=1 + A_mono.shape[0], xl=0, xu=1)
                self._risk_coeffs = _risk_coeffs

            def _evaluate(self, X, out, *args, **kwargs):
                X_round = np.round(X)
                out["F"] = -(X_round @ production)
                risk_g = X_round @ self._risk_coeffs
                mono_g = (A_mono @ X_round.T).T
                out["G"] = np.column_stack([risk_g.reshape(-1, 1), mono_g])

        problem = CutoffProblem()
        algorithm = GA(pop_size=100)
        termination = get_termination("n_gen", 50)

        res = minimize(problem, algorithm, termination, seed=42, verbose=False)

        if res.X is not None:
            mask = np.round(res.X).astype(int)
            mask_key = tuple(mask.tolist())
            if mask_key not in seen_masks:
                seen_masks.add(mask_key)
                kpis = evaluate_solution(mask, grid, indicators, multiplier)
                solutions.append(kpis)
                all_masks.append(mask)

    if not solutions:
        logger.warning("GA fallback also found no solutions.")
        return pd.DataFrame(), grid, []

    df = pd.DataFrame(solutions)

    sort_idx = df["b2_ever_h6"].argsort().values
    df = df.iloc[sort_idx].reset_index(drop=True)
    all_masks = [all_masks[i] for i in sort_idx]

    cummax = df["oa_amt_h0"].cummax()
    pareto_mask = df["oa_amt_h0"] >= cummax
    pareto_indices = pareto_mask[pareto_mask].index.tolist()

    df = df[pareto_mask].reset_index(drop=True)
    pareto_masks = [all_masks[i] for i in pareto_indices]

    logger.info(f"GA Pareto frontier: {len(df)} solutions")
    return df, grid, pareto_masks


# =============================================================================
# Fixed cutoff creation (kept from legacy, generalized)
# =============================================================================


def _build_cutoff_dataframe(
    var0_bins: list,
    var1_cutoffs: list,
    values_var0: list | np.ndarray,
) -> pd.DataFrame:
    """Construct single-row DataFrame with sol_fac + bin columns."""
    solution_data = {"sol_fac": [0]}
    bin_to_cutoff = {float(b): c for b, c in zip(var0_bins, var1_cutoffs)}

    for bin_val in values_var0:
        cutoff_val = bin_to_cutoff.get(float(bin_val))
        if cutoff_val is None:
            raise ValueError(
                f"No cutoff defined for bin {bin_val}. Fixed cutoffs must cover all data bins: {list(values_var0)}"
            )
        solution_data[bin_val] = [cutoff_val]

    return pd.DataFrame(solution_data)


def _validate_cutoff_dict_structure(
    fixed_cutoffs: dict[str, list[int | float]],
    variables: list[str],
) -> tuple[list, list]:
    """Validate fixed_cutoffs has correct keys and matching lengths."""
    var0_name = variables[0]
    var1_name = variables[1]

    if var0_name not in fixed_cutoffs or var1_name not in fixed_cutoffs:
        raise ValueError(f"fixed_cutoffs must contain both variables: {variables}. Got: {list(fixed_cutoffs.keys())}")

    var0_bins = fixed_cutoffs[var0_name]
    var1_cutoffs = fixed_cutoffs[var1_name]

    if len(var0_bins) != len(var1_cutoffs):
        raise ValueError(
            f"Length mismatch: {var0_name} has {len(var0_bins)} bins, "
            f"but {var1_name} has {len(var1_cutoffs)} cutoffs. They must match."
        )

    return var0_bins, var1_cutoffs


def _validate_bins_match_data(
    var0_bins: list,
    values_var0: list | np.ndarray,
    strict_validation: bool,
) -> None:
    """Verify config bins exist in data bins."""
    values_var0_set = {float(v) for v in values_var0}
    var0_bins_set = {float(v) for v in var0_bins}

    if var0_bins_set != values_var0_set:
        msg = f"Fixed cutoff bins {sorted(var0_bins_set)} don't exactly match data bins {sorted(values_var0_set)}. "
        if strict_validation:
            raise ValueError(msg + "Enable strict_validation=False to proceed anyway.")
        logger.warning(msg + "Proceeding with provided cutoffs.")


def _validate_cutoff_monotonicity(
    var0_bins: list,
    var1_cutoffs: list,
    inv_var1: bool,
    strict_validation: bool,
) -> None:
    """Check non-decreasing/non-increasing pattern of cutoffs."""
    sorted_pairs = sorted(zip(var0_bins, var1_cutoffs), key=lambda x: float(x[0]))
    monotonicity_issues = []
    for i in range(1, len(sorted_pairs)):
        prev_bin, prev_cutoff = sorted_pairs[i - 1]
        curr_bin, curr_cutoff = sorted_pairs[i]
        if inv_var1:
            if curr_cutoff > prev_cutoff:
                monotonicity_issues.append(
                    f"bin {prev_bin} (cutoff={prev_cutoff}) -> bin {curr_bin} (cutoff={curr_cutoff})"
                )
        else:
            if curr_cutoff < prev_cutoff:
                monotonicity_issues.append(
                    f"bin {prev_bin} (cutoff={prev_cutoff}) -> bin {curr_bin} (cutoff={curr_cutoff})"
                )

    if monotonicity_issues:
        direction = "non-increasing" if inv_var1 else "non-decreasing"
        msg = (
            f"Non-monotonic cutoffs detected. Expected {direction} cutoffs for ascending bins. "
            f"Issues: {monotonicity_issues}. This may indicate a configuration error."
        )
        if strict_validation:
            raise ValueError(msg)
        logger.warning(msg)


def _validate_cutoff_range(
    var1_cutoffs: list,
    values_var1: list | np.ndarray | None,
    strict_validation: bool,
) -> None:
    """Verify cutoff values are within data min/max."""
    if values_var1 is None or len(values_var1) == 0:
        return

    min_var1, max_var1 = min(values_var1), max(values_var1)
    out_of_range = [cutoff for cutoff in var1_cutoffs if cutoff < min_var1 or cutoff > max_var1]
    if out_of_range:
        msg = (
            f"Cutoff values {out_of_range} are outside data range [{min_var1}, {max_var1}]. "
            f"This may result in unexpected acceptance rates."
        )
        if strict_validation:
            raise ValueError(msg)
        logger.warning(msg)


def create_fixed_cutoff_solution(
    fixed_cutoffs: dict[str, list[int | float]],
    variables: list[str],
    values_var0: list | np.ndarray,
    values_var1: list | np.ndarray | None = None,
    strict_validation: bool = False,
    inv_var1: bool = False,
) -> pd.DataFrame:
    """
    Create a single-row solution DataFrame from predefined cutoffs.

    This function allows bypassing the optimization process by specifying
    exact cutoff values for each bin. Useful for applying known/validated
    cutoffs or for scenario analysis with specific cutoff configurations.

    Args:
        fixed_cutoffs: Dictionary mapping variable names to cutoff lists.
        variables: List of two variable names [var0, var1].
        values_var0: Array/list of bin values for the first variable.
        values_var1: Optional array/list of bin values for the second variable.
        strict_validation: If True, raise errors instead of warnings.
        inv_var1: If True, the var1 cutoffs use inverted logic.

    Returns:
        DataFrame with single row containing sol_fac + bin cutoff columns.
    """
    var0_bins, var1_cutoffs = _validate_cutoff_dict_structure(fixed_cutoffs, variables)
    _validate_bins_match_data(var0_bins, values_var0, strict_validation)
    _validate_cutoff_monotonicity(var0_bins, var1_cutoffs, inv_var1, strict_validation)
    _validate_cutoff_range(var1_cutoffs, values_var1, strict_validation)

    df = _build_cutoff_dataframe(var0_bins, var1_cutoffs, values_var0)
    logger.info(f"Created fixed cutoff solution: {dict(zip(var0_bins, var1_cutoffs))}")

    return df


# =============================================================================
# Legacy API — kept for backward compatibility during transition
# =============================================================================


def get_fact_sol(
    values_var0: list[float],
    values_var1: list[float],
    inv_var1: bool = False,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Generate all feasible solutions with monotonicity constraint (legacy 2-var)."""
    from .utils import optimize_dtypes

    n_bins = len(values_var0)
    cut_values = np.array(sorted(set([0] + list(values_var1))))

    logger.info("--Getting feasible solutions (legacy enumeration)")
    logger.info(f"Bins: {n_bins}, Cut values: {len(cut_values)}")

    n_values = len(cut_values)
    comb_indices = np.array(
        list(combinations_with_replacement(range(n_values), n_bins)),
        dtype=np.int16,
    )
    combinations_array = cut_values[comb_indices]

    logger.info(f"Number of feasible solutions: {len(combinations_array):,}")

    if inv_var1:
        combinations_array = combinations_array[:, ::-1]

    df_v = pd.DataFrame(combinations_array, columns=values_var0)
    df_v = optimize_dtypes(df_v)
    df_v.insert(0, "sol_fac", np.arange(len(df_v), dtype=np.int32))

    return df_v


def kpi_of_fact_sol(
    df_v: pd.DataFrame,
    values_var0: np.ndarray,
    data_sumary_desagregado: pd.DataFrame,
    variables: list[str],
    indicadores: list[str],
    inv_var1: bool = False,
    chunk_size: int = 1000,
) -> pd.DataFrame:
    """Calculate KPIs for all feasible solutions (legacy 2-var)."""
    import gc

    from joblib import Parallel, delayed
    from tqdm import tqdm

    from .utils import MAX_PARALLEL_JOBS

    def _process_kpi_chunk(chunk_data, vals_v0, data_sumary, vars_, inv_):
        chunk_melt = chunk_data.melt(
            id_vars=["sol_fac"], value_vars=vals_v0, var_name=vars_[0], value_name=f"{vars_[1]}_lim"
        )
        chunk_melt[vars_[0]] = chunk_melt[vars_[0]].astype(float)
        chunk_distinct = chunk_melt.drop_duplicates(subset=[vars_[0], f"{vars_[1]}_lim"])[[vars_[0], f"{vars_[1]}_lim"]]
        ds = chunk_distinct.merge(data_sumary, how="left", on=vars_[0])
        if inv_:
            ds = ds[ds[vars_[1]] > ds[f"{vars_[1]}_lim"]]
        else:
            ds = ds[ds[vars_[1]] <= ds[f"{vars_[1]}_lim"]]
        numeric_cols = ds.select_dtypes(include=[np.number]).columns
        agg_dict = {col: "sum" for col in numeric_cols if col not in [vars_[0], f"{vars_[1]}_lim"]}
        ds = ds.groupby([vars_[0], f"{vars_[1]}_lim"]).agg(agg_dict).reset_index()
        return (
            chunk_melt.merge(ds, how="left", on=[vars_[0], f"{vars_[1]}_lim"])
            .fillna(0)
            .groupby("sol_fac", observed=True)
            .agg(agg_dict)
            .reset_index()
        )

    logger.info("--Calculating KPIs for feasible solutions")
    chunks = [df_v.iloc[i : i + chunk_size] for i in range(0, len(df_v), chunk_size)]

    chunks_results = Parallel(n_jobs=MAX_PARALLEL_JOBS)(
        delayed(_process_kpi_chunk)(chunk, values_var0, data_sumary_desagregado, variables, inv_var1)
        for chunk in tqdm(chunks, desc="Processing chunks (Parallel)")
    )

    if not chunks_results:
        return pd.DataFrame()

    final_result = pd.concat(chunks_results, ignore_index=True)
    del chunks_results
    gc.collect()

    final_result = final_result.groupby("sol_fac", observed=True).sum().reset_index()

    for kpi in indicadores:
        raw_cut = data_sumary_desagregado[f"{kpi}_boo"].sum() - final_result[f"{kpi}_boo"]
        neg_count = (raw_cut < 0).sum()
        if neg_count > 0:
            logger.warning(f"{kpi}_cut has {neg_count} negative values. Clipping to 0.")
        final_result[f"{kpi}_cut"] = raw_cut.clip(lower=0)

    for metric in ["", "_cut", "_rep", "_boo"]:
        t30 = f"todu_30ever_h6{metric}"
        tamt = f"todu_amt_pile_h6{metric}"
        if t30 in final_result.columns and tamt in final_result.columns:
            final_result[f"b2_ever_h6{metric}"] = calculate_b2_ever_h6(
                final_result[t30].astype(float),
                final_result[tamt].replace(0, np.nan).astype(float),
                multiplier=DEFAULT_RISK_MULTIPLIER,
                as_percentage=True,
            ).fillna(0)

    return final_result.sort_values(["b2_ever_h6", "oa_amt_h0"])


def get_optimal_solutions(df_v: pd.DataFrame, data_sumary: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """Find Pareto-optimal solutions (legacy 2-var)."""
    import gc

    from joblib import Parallel, delayed
    from tqdm import tqdm

    from .utils import MAX_PARALLEL_JOBS, optimize_dtypes

    logger.info("--Getting optimal solutions")

    data_sumary = data_sumary.sort_values(by=["b2_ever_h6", "oa_amt_h0"])
    data_sumary = data_sumary.drop_duplicates(subset=["b2_ever_h6"], keep="last")

    cummax = data_sumary["oa_amt_h0"].cummax()
    pareto_mask = data_sumary["oa_amt_h0"] >= cummax
    data_sumary = data_sumary[pareto_mask]

    chunks = [df_v.iloc[i : i + chunk_size] for i in range(0, len(df_v), chunk_size)]

    def _process_chunk(df_chunk, ds):
        result = df_chunk.merge(ds, how="inner", on="sol_fac")
        return result if not result.empty else None

    chunks_results = Parallel(n_jobs=MAX_PARALLEL_JOBS)(
        delayed(_process_chunk)(chunk, data_sumary.reset_index())
        for chunk in tqdm(chunks, desc="Processing chunks (Parallel)")
    )

    chunks_results = [res for res in chunks_results if res is not None]
    final_result = pd.concat(chunks_results, ignore_index=True)
    del chunks_results
    gc.collect()

    final_result = optimize_dtypes(final_result)

    logger.info(f"Number of optimal solutions: {len(final_result):,}")
    return final_result.sort_values(by=["b2_ever_h6", "oa_amt_h0"])
