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

from .constants import DEFAULT_RANDOM_STATE, DEFAULT_RISK_MULTIPLIER, DEFAULT_RISK_MULTIPLIER_H3
from .utils import calculate_b2_ever_h6

# =============================================================================
# CellGrid — builds the N-dimensional grid from data_summary_desagregado
# =============================================================================


def _progress_iter(iterable, *, desc: str, enabled: bool = True, total: int | None = None):
    """Wrap an iterable with a tqdm progress bar when available and *enabled*.

    Returns the iterable unchanged when *enabled* is False or tqdm is not installed.
    """
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, total=total, leave=False)
    except Exception:
        return iterable


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
    observed: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool), repr=False)

    _MAX_GRID_CELLS = 10_000_000  # 10M cells — guard against OOM from high-cardinality grids

    @classmethod
    def from_summary(cls, data_summary_desagregado: pd.DataFrame, variables: list[str]) -> "CellGrid":
        """Build grid from aggregated summary data."""
        values_per_var = {var: sorted(data_summary_desagregado[var].unique()) for var in variables}
        shape = tuple(len(values_per_var[v]) for v in variables)

        # Guard against exponential grid explosion (e.g., 5 vars × 100 bins = 10^10 cells)
        n_total = 1
        for s in shape:
            n_total *= s
        if n_total > cls._MAX_GRID_CELLS:
            dims = " × ".join(f"{v}({s})" for v, s in zip(variables, shape))
            raise ValueError(
                f"Grid size {n_total:,} exceeds maximum {cls._MAX_GRID_CELLS:,} cells. "
                f"Dimensions: {dims}. Reduce bin counts or number of variables."
            )

        # Build cell_index: tuple of bin values → flat int
        all_combos = list(product(*(values_per_var[v] for v in variables)))
        cell_index = {combo: idx for idx, combo in enumerate(all_combos)}

        # Build cell_data aligned to flat index
        # Left-join all combos with actual data to handle missing cells (fill 0)
        grid_df = pd.DataFrame(all_combos, columns=variables)
        merged = grid_df.merge(data_summary_desagregado, on=variables, how="left", indicator=True)
        observed = (merged["_merge"] == "both").values
        cell_data = merged.drop(columns=["_merge"]).fillna(0)

        n_phantom = int((~observed).sum())
        if n_phantom > 0:
            logger.debug(
                f"CellGrid: {n_phantom}/{len(all_combos)} cells unobserved in data (will be excluded from optimization)"
            )

        return cls(
            variables=variables,
            values_per_var=values_per_var,
            cell_index=cell_index,
            cell_data=cell_data,
            shape=shape,
            observed=observed,
        )

    @property
    def n_cells(self) -> int:
        return len(self.cell_index)


# =============================================================================
# classify_by_mask — N-dimensional record classification
# =============================================================================


def classify_by_mask(data: pd.DataFrame, mask: np.ndarray, grid: CellGrid) -> pd.Series:
    """Classify records as accepted/rejected using a binary cell mask.

    Maps each record in *data* to its grid cell via the binned variable columns
    and looks up the mask value. Works for any number of dimensions.

    Args:
        data: DataFrame containing the binned variable columns (one per grid dimension).
        mask: Binary array (1=accept, 0=reject) aligned with ``grid.cell_index``.
        grid: CellGrid built from the same variables.

    Returns:
        Boolean Series aligned with *data* index (True = accepted).
    """
    # Build a lookup DataFrame from the grid with the mask column
    lookup = grid.cell_data[grid.variables].copy()
    lookup["_mask"] = mask

    lookup_series = lookup.set_index(grid.variables)["_mask"]
    if len(grid.variables) == 1:
        record_index = data[grid.variables[0]]
    else:
        record_index = pd.MultiIndex.from_frame(data[grid.variables])
    mask_values = lookup_series.reindex(record_index).fillna(0).to_numpy(dtype=bool)
    return pd.Series(mask_values, index=data.index)


# =============================================================================
# Monotonicity constraints
# =============================================================================


def _build_monotonicity_constraints(
    grid: CellGrid,
    inv_vars: list[str],
    *,
    relax_on_uncertainty: bool = False,
    uncertainty_min_exposure: float = 0.0,
    uncertainty_z_threshold: float = 0.0,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> sparse.csc_matrix:
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

    # Optional uncertainty-aware relaxation:
    # Skip adjacency constraints for pairs that are both sparse and statistically
    # ambiguous (difference smaller than z * pooled standard error), so monotonicity
    # is less brittle to noisy bins.
    risk_metric: np.ndarray | None = None
    risk_se: np.ndarray | None = None
    exposure: np.ndarray | None = None
    if (
        relax_on_uncertainty
        and (uncertainty_min_exposure > 0 or uncertainty_z_threshold > 0)
        and "todu_30ever_h6" in grid.cell_data.columns
        and "todu_amt_pile_h6" in grid.cell_data.columns
    ):
        t30 = grid.cell_data["todu_30ever_h6"].to_numpy(dtype=float)
        exposure = grid.cell_data["todu_amt_pile_h6"].to_numpy(dtype=float)
        safe_exp = np.where(exposure > 0, exposure, np.nan)

        # Base default rate q and risk metric r = multiplier * q
        q = np.clip(t30 / safe_exp, 0.0, 1.0)
        risk_metric = multiplier * q

        # Approximate standard error of q with n_eff ~= exposure.
        n_eff = np.where(exposure > 0, exposure, np.nan)
        q_se = np.sqrt(np.clip(q * (1.0 - q), 0.0, None) / n_eff)
        risk_se = multiplier * q_se

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

            if relax_on_uncertainty and risk_metric is not None and risk_se is not None and exposure is not None:
                sparse_pair = min(float(exposure[riskier_idx]), float(exposure[safer_idx])) < float(
                    uncertainty_min_exposure
                )
                ambiguous_pair = False
                if uncertainty_z_threshold > 0:
                    diff = abs(float(risk_metric[riskier_idx]) - float(risk_metric[safer_idx]))
                    pooled_se = float(np.sqrt(risk_se[riskier_idx] ** 2 + risk_se[safer_idx] ** 2))
                    ambiguous_pair = (
                        np.isfinite(pooled_se) and pooled_se > 0 and diff < float(uncertainty_z_threshold) * pooled_se
                    )
                if sparse_pair and ambiguous_pair:
                    continue

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
    fixed_cells: dict[int, int] | None = None,
    max_swapin_production_pct: float | None = None,
    max_swapin_risk: float | None = None,
    time_limit: float = 30.0,
    monotonicity_relaxation_enabled: bool = False,
    monotonicity_uncertainty_min_exposure: float = 0.0,
    monotonicity_uncertainty_z_threshold: float = 0.0,
) -> np.ndarray | None:
    """Solve single MILP: maximize production subject to risk budget + monotonicity.

    Args:
        grid: CellGrid with cell_data containing per-cell KPI columns.
        target_risk: Max allowed risk (percentage, e.g. 1.5 means 1.5%).
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier (typically 7).
        fixed_cells: Optional dict mapping flat cell index → value (0 or 1)
            to pin specific cells as accepted/rejected.

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

    # Guard: if total exposure is zero, the risk constraint is vacuous
    if todu_amt.sum() == 0:
        logger.warning("MILP: total todu_amt_pile_h6 is zero — risk constraint is vacuous.")
        return None

    risk_coeffs = multiplier * todu_30 - (target_risk / 100.0) * todu_amt

    # Denominator guard:
    # We linearize the ratio constraint, but we still need to prevent solutions
    # where the accepted population denominator is 0 (i.e. sum(todu_amt*x) = 0),
    # which would produce NaN risk downstream and can corrupt Pareto/scenario selection.
    # Expressed as: sum(todu_amt*x) >= exposure_denom_eps
    # Linear form used: -sum(todu_amt*x) <= -exposure_denom_eps
    # Use 0.1% of total exposure to prevent trivial near-zero-exposure solutions.
    exposure_denom_eps = max(todu_amt.sum() * 0.001, 1.0)
    denom_row = sparse.csc_matrix((-todu_amt).reshape(1, -1))

    # Monotonicity constraints: A_mono @ x <= 0
    A_mono = _build_monotonicity_constraints(
        grid,
        inv_vars,
        relax_on_uncertainty=monotonicity_relaxation_enabled,
        uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
        uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
        multiplier=multiplier,
    )

    # Stack risk constraint row on top of monotonicity
    risk_row = sparse.csc_matrix(risk_coeffs.reshape(1, -1))
    extra_rows: list[sparse.csc_matrix] = []

    # Optional swap-in production fraction constraint:
    #   sum(oa_amt_h0_rep[i] * x[i]) / sum(oa_amt_h0[i] * x[i]) <= pct/100
    # Linearized: sum((oa_amt_h0_rep[i] - pct/100 * oa_amt_h0[i]) * x[i]) <= 0
    if max_swapin_production_pct is not None and "oa_amt_h0_rep" in cell.columns:
        rep_prod = cell["oa_amt_h0_rep"].values.astype(float)
        swapin_prod_coeffs = rep_prod - (max_swapin_production_pct / 100.0) * production
        extra_rows.append(sparse.csc_matrix(swapin_prod_coeffs.reshape(1, -1)))

    # Optional swap-in risk constraint:
    #   multiplier * sum(todu_30ever_h6_rep[i] * x[i]) / sum(todu_amt_pile_h6_rep[i] * x[i]) <= max_risk/100
    # Linearized: sum((multiplier * todu_30ever_h6_rep[i] - max_risk/100 * todu_amt_pile_h6_rep[i]) * x[i]) <= 0
    if max_swapin_risk is not None and "todu_30ever_h6_rep" in cell.columns and "todu_amt_pile_h6_rep" in cell.columns:
        rep_t30 = cell["todu_30ever_h6_rep"].values.astype(float)
        rep_tamt = cell["todu_amt_pile_h6_rep"].values.astype(float)
        swapin_risk_coeffs = multiplier * rep_t30 - (max_swapin_risk / 100.0) * rep_tamt
        extra_rows.append(sparse.csc_matrix(swapin_risk_coeffs.reshape(1, -1)))

    rows: list[sparse.csc_matrix] = [risk_row, denom_row] + extra_rows

    # Upper bounds for each row block in `rows`.
    # risk constraint + swap-in constraints are <= 0,
    # denominator constraint is <= -exposure_denom_eps (due to sign flip).
    ub: list[float] = [0.0, -exposure_denom_eps] + [0.0] * len(extra_rows)

    if A_mono.shape[0] > 0:
        rows.append(A_mono)
        ub.extend([0.0] * A_mono.shape[0])

    A = sparse.vstack(rows, format="csc") if len(rows) > 1 else rows[0]

    b_u = np.array(ub, dtype=float)
    b_l = np.full_like(b_u, -np.inf, dtype=float)

    constraints = LinearConstraint(A, b_l, b_u)
    integrality = np.ones(n)  # all binary

    from scipy.optimize import Bounds

    bounds = Bounds(lb=np.zeros(n), ub=np.ones(n))

    # Exclude phantom (unobserved) cells: force x=0 so MILP cannot accept them
    if len(grid.observed) == n and not grid.observed.all():
        lb = bounds.lb.copy()
        ub = bounds.ub.copy()
        phantom_mask = ~grid.observed
        ub[phantom_mask] = 0
        bounds = Bounds(lb=lb, ub=ub)

    if fixed_cells:
        lb = bounds.lb.copy()
        ub = bounds.ub.copy()
        for idx, val in fixed_cells.items():
            lb[idx] = val
            ub[idx] = val
        bounds = Bounds(lb=lb, ub=ub)

    result = milp(
        c=c,
        constraints=constraints,
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit},
    )

    if not result.success:
        return None

    return np.round(result.x).astype(int)


def solve_with_fixed_cells(
    grid: CellGrid,
    target_risk: float,
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    fixed_accepts: list[tuple] | None = None,
    fixed_rejects: list[tuple] | None = None,
) -> tuple[np.ndarray | None, dict | None]:
    """Solve MILP with pinned cells. Returns (mask, kpis) or (None, None).

    Args:
        grid: CellGrid with per-cell data.
        target_risk: Max allowed risk (%).
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier.
        indicators: Indicator column names for KPI evaluation.
        fixed_accepts: List of cell coordinate tuples to pin as accepted (1).
        fixed_rejects: List of cell coordinate tuples to pin as rejected (0).

    Returns:
        Tuple of (binary mask, KPI dict) or (None, None) if infeasible.
    """
    fixed_cells: dict[int, int] = {}
    for coords in fixed_accepts or []:
        idx = grid.cell_index.get(coords)
        if idx is not None:
            fixed_cells[idx] = 1
    for coords in fixed_rejects or []:
        idx = grid.cell_index.get(coords)
        if idx is not None:
            if idx in fixed_cells and fixed_cells[idx] == 1:
                logger.warning(f"Cell {coords} pinned as both accept and reject — infeasible.")
                return None, None
            fixed_cells[idx] = 0

    mask = milp_solve_cutoffs(grid, target_risk, inv_vars, multiplier, fixed_cells=fixed_cells)
    if mask is None:
        return None, None

    kpis = evaluate_solution(mask, grid, indicators, multiplier)
    return mask, kpis


# =============================================================================
# KPI evaluation
# =============================================================================


def evaluate_solution(
    mask: np.ndarray,
    grid: CellGrid,
    indicators: list[str],
    multiplier: float,
    multiplier_h3: float | None = None,
) -> dict:
    """Compute KPIs for a single acceptance mask.

    For accepted cells: sums base, _boo, and _rep columns.
    Computes _cut = total_boo - accepted_boo for each indicator.
    Computes b2_ever_h6 for each suffix group.
    Optionally computes b2_ever_h3 when multiplier_h3 is provided.
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

    # Compute b2_ever_h3 for each suffix (complementary metric)
    if multiplier_h3 is not None:
        for suffix in ["", "_boo", "_rep", "_cut"]:
            t30_h3 = f"todu_30ever_h3{suffix}"
            tamt_h3 = f"todu_amt_pile_h3{suffix}"
            if t30_h3 in result and tamt_h3 in result:
                result[f"b2_ever_h3{suffix}"] = float(
                    calculate_b2_ever_h6(result[t30_h3], result[tamt_h3], multiplier=multiplier_h3, as_percentage=True)
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
    max_swapin_production_pct: float | None = None,
    max_swapin_risk: float | None = None,
    multiplier_h3: float | None = None,
    milp_time_limit: float = 30.0,
    show_progress: bool = True,
    monotonicity_relaxation_enabled: bool = False,
    monotonicity_uncertainty_min_exposure: float = 0.0,
    monotonicity_uncertainty_z_threshold: float = 0.0,
    fixed_cells: dict[int, int] | None = None,
) -> tuple[pd.DataFrame, CellGrid, list[np.ndarray]]:
    """Sweep risk targets, solve MILP at each, filter to Pareto-optimal set.

    Args:
        data_summary_desagregado: Aggregated data by variable combinations.
        variables: List of variable names.
        inv_vars: Variables with inverted risk ordering.
        multiplier: Risk multiplier.
        indicators: Indicator column names.
        n_points: Number of risk targets to sweep.
        fixed_cells: Optional dict mapping cell flat index to 0/1, pinning
            cells as rejected (0) or accepted (1). Used for sequential
            cutoff ordering constraints (floor masks).

    Returns:
        Tuple of (pareto_df, grid, masks) where pareto_df has KPI columns for each
        Pareto-optimal solution, grid is the CellGrid used, and masks is a list
        of binary acceptance masks (one per Pareto solution).
    """
    grid = CellGrid.from_summary(data_summary_desagregado, variables)

    # Validate required columns exist in cell_data
    required_cols = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    missing = [c for c in required_cols if c not in grid.cell_data.columns]
    if missing:
        logger.error(f"CellGrid missing required columns: {missing}. Cannot trace Pareto frontier.")
        return pd.DataFrame(), grid, []

    # Determine risk sweep range
    # Max risk = all cells accepted
    all_t30 = grid.cell_data["todu_30ever_h6"].sum()
    all_tamt = grid.cell_data["todu_amt_pile_h6"].sum()
    max_risk = float(calculate_b2_ever_h6(all_t30, all_tamt, multiplier=multiplier, as_percentage=True))
    if np.isnan(max_risk) or max_risk <= 0:
        max_risk = 20.0  # fallback

    # When floor constraints are active, compute the floor-only risk as the
    # sweep lower bound (targets below this are guaranteed infeasible).
    sweep_min = 0.01
    if fixed_cells:
        n_cells = len(grid.cell_data)
        floor_mask_tmp = np.zeros(n_cells, dtype=bool)
        for idx in fixed_cells:
            if fixed_cells[idx] == 1 and 0 <= idx < n_cells:
                floor_mask_tmp[idx] = True
        floor_t30 = grid.cell_data.loc[floor_mask_tmp, "todu_30ever_h6"].sum()
        floor_tamt = grid.cell_data.loc[floor_mask_tmp, "todu_amt_pile_h6"].sum()
        floor_risk = float(calculate_b2_ever_h6(floor_t30, floor_tamt, multiplier=multiplier, as_percentage=True))
        if np.isfinite(floor_risk) and floor_risk > sweep_min:
            sweep_min = floor_risk * 0.95  # start just below floor risk
            logger.info(f"Floor constraint minimum risk: {floor_risk:.2f}% — sweeping from {sweep_min:.2f}%")

    risk_targets = np.linspace(sweep_min, max_risk * 1.1, n_points)

    solutions = []
    all_masks: list[np.ndarray] = []
    seen_masks: set[tuple] = set()

    logger.info(f"MILP Pareto sweep: {n_points} risk targets in [{sweep_min:.2f}, {max_risk * 1.1:.2f}%]")

    for target in _progress_iter(
        risk_targets, desc="MILP Pareto sweep", enabled=show_progress, total=len(risk_targets)
    ):
        mask = milp_solve_cutoffs(
            grid,
            target,
            inv_vars,
            multiplier,
            fixed_cells=fixed_cells,
            max_swapin_production_pct=max_swapin_production_pct,
            max_swapin_risk=max_swapin_risk,
            time_limit=milp_time_limit,
            monotonicity_relaxation_enabled=monotonicity_relaxation_enabled,
            monotonicity_uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
            monotonicity_uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
        )
        if mask is None:
            continue

        mask_key = tuple(mask.tolist())
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)

        kpis = evaluate_solution(mask, grid, indicators, multiplier, multiplier_h3=multiplier_h3)
        solutions.append(kpis)
        all_masks.append(mask)

    if not solutions:
        if fixed_cells:
            # Floor constraint makes all risk targets infeasible. Rather than
            # falling back to GA (which ignores floor constraints), try a wider
            # sweep up to the accept-all risk.  If still infeasible, create a
            # single solution from the floor cells themselves.
            logger.warning("MILP sweep infeasible at all targets with floor constraint. Trying wider risk range...")
            wider_targets = np.linspace(max_risk * 0.5, max_risk * 2.0, n_points)
            for target in wider_targets:
                mask = milp_solve_cutoffs(
                    grid,
                    target,
                    inv_vars,
                    multiplier,
                    fixed_cells=fixed_cells,
                    max_swapin_production_pct=max_swapin_production_pct,
                    max_swapin_risk=max_swapin_risk,
                    time_limit=milp_time_limit,
                    monotonicity_relaxation_enabled=monotonicity_relaxation_enabled,
                    monotonicity_uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
                    monotonicity_uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
                )
                if mask is not None:
                    mask_key = tuple(mask.tolist())
                    if mask_key not in seen_masks:
                        seen_masks.add(mask_key)
                        kpis = evaluate_solution(mask, grid, indicators, multiplier, multiplier_h3=multiplier_h3)
                        solutions.append(kpis)
                        all_masks.append(mask)

            if not solutions:
                # Last resort: accept exactly the floor cells (+ all cells for
                # max-risk solution) so the constraint is guaranteed satisfied.
                logger.warning("Floor constraint still infeasible. Creating floor-only solution.")
                floor_mask = np.zeros(len(grid.cell_data), dtype=int)
                for idx, val in fixed_cells.items():
                    floor_mask[idx] = val
                kpis = evaluate_solution(floor_mask, grid, indicators, multiplier, multiplier_h3=multiplier_h3)
                solutions.append(kpis)
                all_masks.append(floor_mask)
                # Also add accept-all as upper bound
                all_mask = np.ones(len(grid.cell_data), dtype=int)
                kpis_all = evaluate_solution(all_mask, grid, indicators, multiplier, multiplier_h3=multiplier_h3)
                solutions.append(kpis_all)
                all_masks.append(all_mask)
        else:
            logger.warning("No feasible MILP solutions found. Attempting GA fallback...")
            return _ga_pareto_fallback(
                grid,
                inv_vars,
                multiplier,
                indicators,
                n_points,
                show_progress=show_progress,
                monotonicity_relaxation_enabled=monotonicity_relaxation_enabled,
                monotonicity_uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
                monotonicity_uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
            )

    df = pd.DataFrame(solutions)

    # Filter out any NaN/inf-risk solutions before Pareto dominance logic.
    # `evaluate_solution()` can yield NaN b2_ever_h6 when the accepted-cell
    # exposure denominator is zero; those should not participate in
    # sorting/dominance/scenario selection.
    if "b2_ever_h6" in df.columns:
        risk_arr = df["b2_ever_h6"].to_numpy(dtype=float, copy=False)
        prod_arr = df["oa_amt_h0"].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(risk_arr) & np.isfinite(prod_arr)
        if not valid.all():
            n_bad = int((~valid).sum())
            logger.warning(f"Filtering {n_bad} NaN/inf-risk solution(s) from Pareto sweep.")
        df = df.loc[valid].reset_index(drop=True)
        all_masks = [m for m, keep in zip(all_masks, valid) if keep]

    if df.empty:
        logger.warning("Pareto frontier: all MILP sweep solutions filtered out (NaN/inf risk).")
        return pd.DataFrame(), grid, []

    # Sort by risk, keep track of original indices for mask alignment
    sort_idx = df["b2_ever_h6"].argsort().values
    df = df.iloc[sort_idx].reset_index(drop=True)
    all_masks = [all_masks[i] for i in sort_idx]

    # Post-hoc Pareto dominance filter:
    # Sorted by ascending risk, keep solutions whose production is strictly > any previously seen.
    prev_max = float("-inf")
    pareto_keep = []
    for i, prod in enumerate(df["oa_amt_h0"].values):
        if prod > prev_max:
            pareto_keep.append(i)
            prev_max = prod

    if len(pareto_keep) < len(df):
        n_removed = len(df) - len(pareto_keep)
        df = df.iloc[pareto_keep].reset_index(drop=True)
        pareto_masks = [all_masks[i] for i in pareto_keep]
        logger.debug(f"Removed {n_removed} dominated solutions (non-increasing production)")
    else:
        pareto_masks = list(all_masks)

    # Stage 2 pairwise dominance check: skipped for the standard 2-objective
    # case (risk vs production) where the sort-and-sweep above is exact (#16).
    # The O(n²) check is only needed if >2 objectives are ever introduced.

    # ─────────────────────────────────────────────────────────────────
    # Local refinement: bisect between adjacent frontier points to
    # discover Pareto-optimal masks the linear grid missed (#26).
    # Each bisection solves one MILP at the midpoint risk; new unique
    # masks trigger further bisection on the new sub-intervals.
    # ─────────────────────────────────────────────────────────────────
    MAX_REFINE_ROUNDS = 3  # outer passes over the frontier
    MIN_RISK_GAP = 0.02  # skip intervals narrower than 0.02 pp

    n_refined = 0
    for _round in range(MAX_REFINE_ROUNDS):
        risk_vals = df["b2_ever_h6"].values
        if len(risk_vals) < 2:
            break
        midpoints = []
        for i in range(len(risk_vals) - 1):
            gap = risk_vals[i + 1] - risk_vals[i]
            if gap > MIN_RISK_GAP:
                midpoints.append((risk_vals[i] + risk_vals[i + 1]) / 2.0)
        if not midpoints:
            break

        found_new = False
        for mid_target in midpoints:
            mask = milp_solve_cutoffs(
                grid,
                mid_target,
                inv_vars,
                multiplier,
                fixed_cells=fixed_cells,
                max_swapin_production_pct=max_swapin_production_pct,
                max_swapin_risk=max_swapin_risk,
                time_limit=milp_time_limit,
                monotonicity_relaxation_enabled=monotonicity_relaxation_enabled,
                monotonicity_uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
                monotonicity_uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
            )
            if mask is None:
                continue
            mask_key = tuple(mask.tolist())
            if mask_key in seen_masks:
                continue
            seen_masks.add(mask_key)
            kpis = evaluate_solution(mask, grid, indicators, multiplier, multiplier_h3=multiplier_h3)
            risk_v = kpis.get("b2_ever_h6")
            prod_v = kpis.get("oa_amt_h0")
            if risk_v is None or prod_v is None or not np.isfinite(risk_v) or not np.isfinite(prod_v):
                continue
            # Insert into df + masks
            new_row = pd.DataFrame([kpis])
            df = pd.concat([df, new_row], ignore_index=True)
            pareto_masks.append(mask)
            found_new = True
            n_refined += 1

        if not found_new:
            break

        # Re-sort and re-filter after each refinement round
        sort_idx = df["b2_ever_h6"].argsort().values
        df = df.iloc[sort_idx].reset_index(drop=True)
        pareto_masks = [pareto_masks[i] for i in sort_idx]
        prev_max = float("-inf")
        keep = []
        for i, prod in enumerate(df["oa_amt_h0"].values):
            if prod > prev_max:
                keep.append(i)
                prev_max = prod
        if len(keep) < len(df):
            df = df.iloc[keep].reset_index(drop=True)
            pareto_masks = [pareto_masks[i] for i in keep]

    if n_refined:
        logger.info(f"Pareto refinement: {n_refined} new solution(s) discovered via bisection")

    logger.info(f"Pareto frontier: {len(df)} solutions (from {len(solutions) + n_refined} unique MILP solves)")

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
        logger.info("mask_to_cutoffs: N>2 variable case, returning cell-level + conditional cutoffs")

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
        # N>2: lossless cell-level dict + conditional cutoffs for last dimension
        # 1) Cell-level dict: {(v0, v1, v2, ...): 0_or_1, ...}
        cell_dict = {}
        for combo, idx in grid.cell_index.items():
            cell_dict[combo] = int(mask[idx])
        result["_cells"] = cell_dict

        # 2) Per-dimension marginals: is any cell with this value accepted?
        for d, var in enumerate(grid.variables):
            marginal = {}
            for val in grid.values_per_var[var]:
                accepted = any(mask[idx] == 1 for combo, idx in grid.cell_index.items() if combo[d] == val)
                marginal[float(val)] = 1.0 if accepted else 0.0
            result[f"_marginal_{var}"] = marginal

        # 3) Conditional cutoffs for last dimension: for each unique combo
        # of variables[:-1], find the extreme accepted value of variables[-1]
        last_var = grid.variables[-1]
        last_d = len(grid.variables) - 1
        last_inverted = last_var in inv_vars
        last_vals = grid.values_per_var[last_var]

        cond_cutoffs = {}
        # Get all unique combos of the first N-1 dimensions
        prefix_combos = set()
        for combo in grid.cell_index:
            prefix_combos.add(combo[:last_d])

        for prefix in sorted(prefix_combos):
            extreme_val = None
            for lv in last_vals:
                full_combo = prefix + (lv,)
                idx = grid.cell_index.get(full_combo)
                if idx is not None and mask[idx] == 1:
                    if last_inverted:
                        if extreme_val is None or lv < extreme_val:
                            extreme_val = lv
                    else:
                        if extreme_val is None or lv > extreme_val:
                            extreme_val = lv
            if extreme_val is None:
                extreme_val = np.inf if last_inverted else -np.inf
            cond_cutoffs[prefix] = float(extreme_val)

        result[last_var] = cond_cutoffs

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

    else:
        for i, mask in enumerate(masks):
            df.at[i, "acceptance_mask"] = ",".join(str(int(v)) for v in mask)

    # Add sol_fac column at the beginning (if not already present)
    if "sol_fac" in df.columns:
        df["sol_fac"] = range(len(df))
    else:
        df.insert(0, "sol_fac", range(len(df)))

    return df


def decode_mask(mask_str: str) -> np.ndarray:
    """Decode a comma-separated binary mask string back to numpy array."""
    return np.array([int(v) for v in mask_str.split(",")], dtype=int)


# =============================================================================
# GA fallback (pymoo)
# =============================================================================


def _ga_pareto_fallback(
    grid: CellGrid,
    inv_vars: list[str],
    multiplier: float,
    indicators: list[str],
    n_points: int,
    show_progress: bool = True,
    monotonicity_relaxation_enabled: bool = False,
    monotonicity_uncertainty_min_exposure: float = 0.0,
    monotonicity_uncertainty_z_threshold: float = 0.0,
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
    required_cols = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    missing = [c for c in required_cols if c not in cell.columns]
    if missing:
        logger.error(f"CellGrid missing required columns for GA fallback: {missing}.")
        return pd.DataFrame(), grid, []

    todu_30 = cell["todu_30ever_h6"].values.astype(float)
    todu_amt = cell["todu_amt_pile_h6"].values.astype(float)
    production = cell["oa_amt_h0"].values.astype(float)
    A_mono = _build_monotonicity_constraints(
        grid,
        inv_vars,
        relax_on_uncertainty=monotonicity_relaxation_enabled,
        uncertainty_min_exposure=monotonicity_uncertainty_min_exposure,
        uncertainty_z_threshold=monotonicity_uncertainty_z_threshold,
        multiplier=multiplier,
    )

    all_t30 = todu_30.sum()
    all_tamt = todu_amt.sum()
    max_risk = float(calculate_b2_ever_h6(all_t30, all_tamt, multiplier=multiplier, as_percentage=True))
    if np.isnan(max_risk) or max_risk <= 0:
        max_risk = 20.0

    solutions = []
    all_masks: list[np.ndarray] = []
    seen_masks: set[tuple] = set()

    targets = np.linspace(0.01, max_risk * 1.1, n_points)
    for target in _progress_iter(targets, desc="GA Pareto fallback", enabled=show_progress, total=len(targets)):
        risk_coeffs = multiplier * todu_30 - (target / 100.0) * todu_amt

        # Build upper bounds: phantom cells forced to 0
        xu = np.ones(grid.n_cells)
        if len(grid.observed) == grid.n_cells and not grid.observed.all():
            xu[~grid.observed] = 0

        class CutoffProblem(Problem):
            def __init__(self, _risk_coeffs=risk_coeffs, _xu=xu):
                super().__init__(n_var=grid.n_cells, n_obj=1, n_ieq_constr=1 + A_mono.shape[0], xl=0, xu=_xu)
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

        res = minimize(problem, algorithm, termination, seed=DEFAULT_RANDOM_STATE, verbose=False)

        if res.X is not None:
            mask = np.round(res.X).astype(int)
            # Post-hoc feasibility check: verify monotonicity constraints
            mono_violation = A_mono @ mask
            if np.any(mono_violation > 0):
                continue  # skip infeasible GA solutions
            # Verify risk constraint
            risk_check = mask @ risk_coeffs
            if risk_check > 1e-6:
                continue  # skip solutions exceeding risk target
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

    # Same safeguard as MILP: drop any NaN/inf-risk individuals before
    # frontier stage-1/2 filters.
    if "b2_ever_h6" in df.columns:
        risk_arr = df["b2_ever_h6"].to_numpy(dtype=float, copy=False)
        prod_arr = df["oa_amt_h0"].to_numpy(dtype=float, copy=False)
        valid = np.isfinite(risk_arr) & np.isfinite(prod_arr)
        if not valid.all():
            n_bad = int((~valid).sum())
            logger.warning(f"Filtering {n_bad} NaN/inf-risk solution(s) from GA fallback.")
        df = df.loc[valid].reset_index(drop=True)
        all_masks = [m for m, keep in zip(all_masks, valid) if keep]

    # Exposure floor: mirror the MILP denominator guard — reject solutions
    # where accepted exposure is below 0.1% of total (same epsilon as MILP).
    if "todu_amt_pile_h6" in df.columns:
        exposure_denom_eps = max(todu_amt.sum() * 0.001, 1.0)
        exposure_ok = df["todu_amt_pile_h6"].to_numpy(dtype=float, copy=False) >= exposure_denom_eps
        if not exposure_ok.all():
            n_low = int((~exposure_ok).sum())
            logger.warning(f"Filtering {n_low} near-zero-exposure solution(s) from GA fallback.")
            df = df.loc[exposure_ok].reset_index(drop=True)
            all_masks = [m for m, keep in zip(all_masks, exposure_ok) if keep]

    if df.empty:
        logger.warning("GA fallback: all solutions filtered out (NaN/inf risk).")
        return pd.DataFrame(), grid, []

    # Sort by risk ascending
    sort_idx = df["b2_ever_h6"].argsort().values
    df = df.iloc[sort_idx].reset_index(drop=True)
    all_masks = [all_masks[i] for i in sort_idx]

    # Stage 1: cumulative-max filter (fast)
    prev_max = float("-inf")
    pareto_keep = []
    for i, prod in enumerate(df["oa_amt_h0"].values):
        if prod > prev_max:
            pareto_keep.append(i)
            prev_max = prod

    if len(pareto_keep) < len(df):
        df = df.iloc[pareto_keep].reset_index(drop=True)
        all_masks = [all_masks[i] for i in pareto_keep]

    # Stage 2 pairwise dominance: skipped for 2-objective case — sort-and-sweep
    # above is exact for (risk, production) Pareto (#16).

    logger.info(f"GA Pareto frontier: {len(df)} solutions")
    return df, grid, all_masks


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
# N>2 Fixed cutoff mask creation
# =============================================================================


def _validate_nd_cutoff_structure(
    fixed_cutoffs: dict,
    variables: list[str],
) -> tuple[dict[str, list | dict[float, list]], str | None, str]:
    """Extract per-variable cutoff lists from fixed_cutoffs dict.

    Supports two formats for variables:
    - **Flat list**: paired cutoff per anchor bin (same length as anchor).
    - **Dict (matrix)**: cutoff schedule keyed by conditioning variable values.
      Each value list must have the same length as the anchor. The conditioning
      variable is the one variable absent from ``fixed_cutoffs`` whose
      accepted values are the union of dict keys.

    The anchor variable (flat list that defines the reference length) is
    auto-detected as the first variable with a flat list in ``fixed_cutoffs``,
    regardless of position in ``variables``.

    Skips meta keys like ``strict_validation`` and ``run_all_scenarios``
    (both at the top level and inside nested dicts).

    Args:
        fixed_cutoffs: Raw config dict with variable → cutoff-list or dict mappings.
        variables: Ordered variable names.

    Returns:
        Tuple of (accepted dict, conditioning variable name or None, anchor variable name).
        The dict maps each variable name to either a list (flat) or
        a dict[float, list] (matrix cutoffs).

    Raises:
        ValueError: On structural errors (missing vars, bad types, length mismatches).
    """
    meta_keys = {"strict_validation", "run_all_scenarios"}
    accepted: dict[str, list | dict[float, list]] = {}
    missing_vars: list[str] = []
    dict_vars: list[str] = []

    # Auto-detect anchor variable: first variable with a flat list
    anchor_var: str | None = None
    for var in variables:
        if var not in fixed_cutoffs:
            continue
        vals = fixed_cutoffs[var]
        if isinstance(vals, (list, tuple)):
            anchor_var = var
            break

    if anchor_var is None:
        raise ValueError(
            f"At least one variable in fixed_cutoffs must be a flat list to serve as the anchor. "
            f"Got: {[(v, type(fixed_cutoffs[v]).__name__) for v in variables if v in fixed_cutoffs and v not in meta_keys]}"
        )

    # Process anchor variable
    anchor_vals = fixed_cutoffs[anchor_var]
    accepted[anchor_var] = [float(v) for v in anchor_vals]
    anchor_len = len(accepted[anchor_var])

    # Process remaining variables
    for var in variables:
        if var == anchor_var:
            continue
        if var not in fixed_cutoffs:
            missing_vars.append(var)
            continue
        vals = fixed_cutoffs[var]
        if isinstance(vals, dict):
            # Matrix cutoffs: keys are conditioning var values, values are cutoff lists
            # Filter out meta keys that may be nested inside
            converted: dict[float, list] = {}
            for k, v in vals.items():
                if str(k) in meta_keys:
                    continue
                if not isinstance(v, (list, tuple)):
                    raise ValueError(f"fixed_cutoffs['{var}']['{k}'] must be a list, got {type(v).__name__}")
                key = float(k)
                row = [float(x) for x in v]
                if len(row) != anchor_len:
                    raise ValueError(
                        f"fixed_cutoffs['{var}']['{k}'] has length {len(row)}, "
                        f"expected {anchor_len} (same as anchor '{anchor_var}')."
                    )
                converted[key] = row
            accepted[var] = converted
            dict_vars.append(var)
        elif isinstance(vals, (list, tuple)):
            accepted[var] = [float(v) for v in vals]
        else:
            raise ValueError(f"fixed_cutoffs['{var}'] must be a list or dict, got {type(vals).__name__}")

    # Determine conditioning variable
    conditioning_var: str | None = None
    if dict_vars:
        if len(missing_vars) == 0:
            raise ValueError(
                f"Matrix cutoffs found for {dict_vars} but no conditioning variable is missing "
                f"from fixed_cutoffs. Exactly one variable must be absent to serve as the "
                f"conditioning variable."
            )
        if len(missing_vars) > 1:
            raise ValueError(
                f"Matrix cutoffs found for {dict_vars} but multiple variables are missing "
                f"from fixed_cutoffs: {missing_vars}. Exactly one variable must be absent "
                f"to serve as the conditioning variable."
            )
        conditioning_var = missing_vars[0]
        # Conditioning var's accepted values = union of all dict keys
        all_keys: set[float] = set()
        for dv in dict_vars:
            all_keys |= set(accepted[dv].keys())  # type: ignore[union-attr]
        accepted[conditioning_var] = sorted(all_keys)
    elif missing_vars:
        raise ValueError(
            f"fixed_cutoffs must contain all variables: {variables}. "
            f"Missing {missing_vars}. Got keys: {[k for k in fixed_cutoffs if k not in meta_keys]}"
        )

    # Enforce equal length for flat lists
    for var in variables:
        vals = accepted[var]
        if isinstance(vals, list) and var != conditioning_var and len(vals) != anchor_len:
            raise ValueError(
                f"fixed_cutoffs['{var}'] has length {len(vals)}, expected {anchor_len} "
                f"(same as anchor '{anchor_var}'). All flat cutoff lists must have equal length."
            )

    return accepted, conditioning_var, anchor_var


def create_fixed_cutoff_mask(
    fixed_cutoffs: dict,
    variables: list[str],
    data_summary_desagregado: pd.DataFrame,
    inv_vars: list[str] | None = None,
    strict_validation: bool = False,
) -> tuple["CellGrid", np.ndarray]:
    """Create a CellGrid and binary mask from N-variable fixed cutoffs.

    Supports two cutoff formats:

    **Flat (paired)**: var0 lists bin values, each secondary variable lists a
    cutoff threshold per var0 bin. A cell is accepted iff var0 matches AND
    cell.var_d <= cutoff_d[i] (or >= if inverted).

    **Matrix**: a secondary variable maps conditioning-variable values to
    per-var0-bin cutoff schedules. The conditioning variable is absent from
    ``fixed_cutoffs``; its accepted values are the dict keys.

    Args:
        fixed_cutoffs: Dict mapping variable names to cutoff lists or dicts.
            May also contain meta keys (``strict_validation``, ``run_all_scenarios``).
        variables: Ordered variable names (len >= 2).
        data_summary_desagregado: Aggregated data to build the grid from.
        inv_vars: Variables with inverted risk ordering (>= instead of <=).
        strict_validation: If True, raise errors instead of warnings.

    Returns:
        Tuple of (CellGrid, mask) where mask is a binary numpy array
        (1=accept, 0=reject) aligned with grid.cell_index.
    """
    if inv_vars is None:
        inv_vars = []

    # Parse and validate cutoff structure
    cutoffs, conditioning_var, anchor_var = _validate_nd_cutoff_structure(fixed_cutoffs, variables)

    # Build grid from data
    grid = CellGrid.from_summary(data_summary_desagregado, variables)

    var0 = anchor_var
    var0_bins = cutoffs[var0]  # always a flat list
    secondary_vars = [v for v in variables if v != var0]

    # Validate var0 bin values exist in data
    data_bins_var0 = {float(v) for v in grid.values_per_var[var0]}
    unknown_var0 = set(var0_bins) - data_bins_var0
    if unknown_var0:
        msg = (
            f"Fixed cutoff var0 bin values {sorted(unknown_var0)} for '{var0}' "
            f"don't exist in data bins {sorted(data_bins_var0)}."
        )
        if strict_validation:
            raise ValueError(msg)
        logger.warning(msg + " These entries will be ignored.")

    # Validate conditioning variable keys exist in data
    if conditioning_var is not None:
        data_cond_vals = {float(v) for v in grid.values_per_var[conditioning_var]}
        cond_keys = set(cutoffs[conditioning_var])  # type: ignore[arg-type]
        unknown_cond = cond_keys - data_cond_vals
        if unknown_cond:
            msg = (
                f"Conditioning variable '{conditioning_var}' dict keys {sorted(unknown_cond)} "
                f"don't exist in data bins {sorted(data_cond_vals)}."
            )
            if strict_validation:
                raise ValueError(msg)
            logger.warning(msg + " These entries will be ignored.")

    # Collect all cutoff values per variable for range validation
    def _all_cutoff_values(var: str) -> list[float]:
        entry = cutoffs[var]
        if isinstance(entry, dict):
            return [v for row in entry.values() for v in row]
        return entry

    # Validate cutoff range for secondary variables
    for var in secondary_vars:
        if var == conditioning_var:
            continue
        data_vals = grid.values_per_var[var]
        if len(data_vals) == 0:
            continue
        min_val, max_val = float(min(data_vals)), float(max(data_vals))
        all_vals = _all_cutoff_values(var)
        out_of_range = [c for c in all_vals if c < min_val or c > max_val]
        if out_of_range:
            msg = (
                f"Cutoff values {sorted(set(out_of_range))} for '{var}' are outside data range "
                f"[{min_val}, {max_val}]. This may result in unexpected acceptance rates."
            )
            if strict_validation:
                raise ValueError(msg)
            logger.warning(msg)

    # Validate monotonicity for secondary variables
    sorted_indices = sorted(range(len(var0_bins)), key=lambda i: var0_bins[i])

    def _check_monotonicity(var: str, var_cutoffs: list[float]) -> None:
        sorted_cutoffs = [var_cutoffs[i] for i in sorted_indices]
        inverted = var in inv_vars
        issues = []
        for j in range(1, len(sorted_cutoffs)):
            prev_c, curr_c = sorted_cutoffs[j - 1], sorted_cutoffs[j]
            prev_b = var0_bins[sorted_indices[j - 1]]
            curr_b = var0_bins[sorted_indices[j]]
            if inverted:
                if curr_c > prev_c:
                    issues.append(f"bin {prev_b} (cutoff={prev_c}) -> bin {curr_b} (cutoff={curr_c})")
            else:
                if curr_c < prev_c:
                    issues.append(f"bin {prev_b} (cutoff={prev_c}) -> bin {curr_b} (cutoff={curr_c})")
        if issues:
            direction = "non-increasing" if inverted else "non-decreasing"
            msg = (
                f"Non-monotonic cutoffs for '{var}'. Expected {direction} cutoffs "
                f"for ascending '{var0}' bins. Issues: {issues}."
            )
            if strict_validation:
                raise ValueError(msg)
            logger.warning(msg)

    for var in secondary_vars:
        if var == conditioning_var:
            continue
        entry = cutoffs[var]
        if isinstance(entry, dict):
            for _cond_key, row in entry.items():
                _check_monotonicity(var, row)
        else:
            _check_monotonicity(var, entry)

    # Build variable-to-dimension index for quick lookup
    var_dim = {v: d for d, v in enumerate(variables)}
    cond_dim = var_dim[conditioning_var] if conditioning_var is not None else None

    # Build mask
    anchor_dim = var_dim[var0]
    mask = np.zeros(grid.n_cells, dtype=int)
    for i, v0_bin in enumerate(var0_bins):
        if v0_bin not in data_bins_var0:
            continue
        for combo, idx in grid.cell_index.items():
            if float(combo[anchor_dim]) != v0_bin:
                continue
            accepted = True
            for var in secondary_vars:
                cell_val = float(combo[var_dim[var]])
                if var == conditioning_var:
                    # Conditioning variable: accept if value is in the dict keys
                    if cell_val not in set(cutoffs[conditioning_var]):  # type: ignore[arg-type]
                        accepted = False
                        break
                elif isinstance(cutoffs[var], dict):
                    # Matrix cutoff: look up schedule by conditioning var value
                    cond_val = float(combo[cond_dim])  # type: ignore[index]
                    schedule = cutoffs[var].get(cond_val)
                    if schedule is None:
                        accepted = False
                        break
                    cutoff = schedule[i]
                    if var in inv_vars:
                        if cell_val < cutoff:
                            accepted = False
                            break
                    else:
                        if cell_val > cutoff:
                            accepted = False
                            break
                else:
                    # Flat cutoff
                    cutoff = cutoffs[var][i]
                    if var in inv_vars:
                        if cell_val < cutoff:
                            accepted = False
                            break
                    else:
                        if cell_val > cutoff:
                            accepted = False
                            break
            if accepted:
                mask[idx] = 1

    n_accepted = int(mask.sum())
    logger.info(
        f"Fixed cutoff mask: {n_accepted}/{grid.n_cells} cells accepted ({n_accepted / grid.n_cells * 100:.1f}%)"
    )

    return grid, mask


# =============================================================================
# Legacy API — kept for backward compatibility during transition
# =============================================================================


_MAX_LEGACY_SOLUTIONS = 500_000


def get_fact_sol(
    values_var0: list[float],
    values_var1: list[float],
    inv_var1: bool = False,
    chunk_size: int = 10000,
) -> pd.DataFrame:
    """Generate all feasible solutions with monotonicity constraint (legacy 2-var)."""
    from math import comb

    from .utils import optimize_dtypes

    n_bins = len(values_var0)
    cut_values = np.array(sorted(set([0] + list(values_var1))))

    logger.info("--Getting feasible solutions (legacy enumeration)")
    logger.info(f"Bins: {n_bins}, Cut values: {len(cut_values)}")

    n_values = len(cut_values)

    # Pre-check: C(n_values + n_bins - 1, n_bins) combinations
    n_expected = comb(n_values + n_bins - 1, n_bins)
    if n_expected > _MAX_LEGACY_SOLUTIONS:
        raise RuntimeError(
            f"Legacy enumeration would generate {n_expected:,} solutions "
            f"(limit: {_MAX_LEGACY_SOLUTIONS:,}). "
            f"Reduce bin count or use MILP (the default solver)."
        )

    comb_indices = np.array(
        list(combinations_with_replacement(range(n_values), n_bins)),
        dtype=np.int32,
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
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    multiplier_h3: float | None = None,
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
            ds = ds[ds[vars_[1]] >= ds[f"{vars_[1]}_lim"]]
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
            b2_col = f"b2_ever_h6{metric}"
            raw_b2 = calculate_b2_ever_h6(
                final_result[t30].astype(float),
                final_result[tamt].astype(float),
                multiplier=multiplier,
                as_percentage=True,
            )
            n_nan = int(raw_b2.isna().sum())
            if n_nan > 0 and metric in ("", "_boo"):
                logger.warning(
                    f"kpi_of_fact_sol: {n_nan} solution(s) have NaN b2_ever_h6{metric} "
                    f"(zero exposure). These will be excluded from Pareto selection."
                )
            # Base b2_ever_h6 (no suffix): keep NaN so zero-exposure solutions
            # are excluded from ranking/selection instead of appearing as 0% risk.
            # Suffixed columns (_cut, _rep, _boo) are display-only: fill with 0.
            if metric == "":
                final_result[b2_col] = raw_b2
            else:
                final_result[b2_col] = raw_b2.fillna(0)

    # Compute b2_ever_h3 (complementary metric) when h3 columns are present
    effective_multiplier_h3 = multiplier_h3 if multiplier_h3 is not None else DEFAULT_RISK_MULTIPLIER_H3
    for metric in ["", "_cut", "_rep", "_boo"]:
        t30_h3 = f"todu_30ever_h3{metric}"
        tamt_h3 = f"todu_amt_pile_h3{metric}"
        if t30_h3 in final_result.columns and tamt_h3 in final_result.columns:
            raw_b2_h3 = calculate_b2_ever_h6(
                final_result[t30_h3].astype(float),
                final_result[tamt_h3].astype(float),
                multiplier=effective_multiplier_h3,
                as_percentage=True,
            )
            n_nan_h3 = int(raw_b2_h3.isna().sum())
            if n_nan_h3 > 0 and metric in ("", "_boo"):
                logger.warning(
                    f"kpi_of_fact_sol: {n_nan_h3} solution(s) have NaN b2_ever_h3{metric} "
                    f"(zero exposure). These will be excluded from Pareto selection."
                )
            if metric == "":
                final_result[f"b2_ever_h3{metric}"] = raw_b2_h3
            else:
                final_result[f"b2_ever_h3{metric}"] = raw_b2_h3.fillna(0)

    return final_result.sort_values(["b2_ever_h6", "oa_amt_h0"])


def get_optimal_solutions(df_v: pd.DataFrame, data_sumary: pd.DataFrame, chunk_size: int = 1000) -> pd.DataFrame:
    """Find Pareto-optimal solutions (legacy 2-var)."""
    import gc

    from joblib import Parallel, delayed
    from tqdm import tqdm

    from .utils import MAX_PARALLEL_JOBS, optimize_dtypes

    logger.info("--Getting optimal solutions")

    # Filter out solutions with NaN/inf risk (zero-exposure cells) before
    # Pareto ranking — these are "no data", not "zero risk" (#4).
    valid_risk = np.isfinite(data_sumary["b2_ever_h6"]) & np.isfinite(data_sumary["oa_amt_h0"])
    n_invalid = int((~valid_risk).sum())
    if n_invalid > 0:
        logger.warning(
            f"get_optimal_solutions: filtering {n_invalid} solution(s) with NaN/inf risk "
            f"(zero exposure) before Pareto ranking."
        )
        data_sumary = data_sumary[valid_risk].copy()

    data_sumary = data_sumary.sort_values(by=["b2_ever_h6", "oa_amt_h0"])
    # Dedup by exact (risk, production) pairs only — rounding dedup removed
    # because it merged distinct frontier points and discarded better
    # production for tight risk caps (#27).
    data_sumary = data_sumary.drop_duplicates(subset=["b2_ever_h6", "oa_amt_h0"], keep="last")

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
