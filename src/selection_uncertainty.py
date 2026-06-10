"""Selection-aware bootstrap for the optimized headline (audit #28, Phase B).

The scenario headline is the in-sample estimate of an *optimized* cutoff: the
selection picks the max-production Pareto point whose point-estimate risk sits
under the target, so noise at the binding boundary is selected INTO the answer
(winner's curse) and a bootstrap that holds the chosen mask fixed understates
the uncertainty of the procedure.

This module re-runs the *selection* (not the MILPs) inside every bootstrap
replicate, over the frozen frontier candidate set, at cell level:

- booked loans are resampled and re-aggregated to grid cells;
- the reject-inferred component is resampled via the rejected-population
  composition (same first-order cell-ratio approach as
  :class:`src.utils._RepescaBootstrapContext`);
- every candidate mask is then evaluated as a dot product over cells and the
  point-estimate selection rule (max production with risk ≤ threshold, else
  min risk) is re-applied.

Outputs: a **selection-aware CI** for the optimized headline, an Efron-style
**optimism estimate** (how much the reported in-sample risk understates,
in pp), the share of replicates that re-select a different frontier point, and
a per-candidate risk CI (consumed by the optional ``ci_upper`` selection rule,
Phase C).

Limitation (documented, by design): the candidate set is the frozen frontier —
re-solving the MILPs per replicate is not captured. The estimate is therefore
conditional on the candidates, a lower bound on full procedure uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RISK_MULTIPLIER

if TYPE_CHECKING:
    from src.optimization_utils import CellGrid


@dataclass
class SelectionBootstrapResult:
    """Selection-aware uncertainty of the optimized headline."""

    risk_ci_sel_lower: float
    risk_ci_sel_upper: float
    production_ci_sel_lower: float
    production_ci_sel_upper: float
    #: Estimated winner's-curse optimism in percentage points: the average gap
    #: between a replicate-winner's risk evaluated on the ORIGINAL data and its
    #: risk on the replicate that selected it. Positive ⇒ the reported
    #: in-sample risk understates by roughly this much.
    selection_optimism_pp: float
    #: Share (%) of replicates whose re-selection picked a different frontier
    #: point than the original selection.
    reselect_pct: float
    #: Per-candidate risk CI upper bound (%), aligned with the pareto_masks
    #: order — input to the optional ``ci_upper`` selection rule.
    candidate_risk_ci_upper: np.ndarray
    #: Per-candidate original-data risk (%) and production, same alignment.
    candidate_risk_orig: np.ndarray
    candidate_production_orig: np.ndarray
    n_candidates: int
    original_selected_index: int


def _loans_to_cells(df: pd.DataFrame, variables: list[str], cell_index: dict) -> np.ndarray:
    """Map each loan row to its grid flat index (-1 when its cell is absent)."""
    normalized = {tuple(float(v) for v in k): i for k, i in cell_index.items()}
    return np.array(
        [normalized.get(tuple(float(v) for v in row), -1) for row in df[variables].values],
        dtype=np.int64,
    )


def _select(risk: np.ndarray, production: np.ndarray, threshold: float) -> int:
    """The point-estimate selection rule, mirroring the scenario selection:
    max production among finite-risk candidates with risk <= threshold; if none
    qualify, the minimum-risk candidate."""
    finite = np.isfinite(risk)
    eligible = finite & (risk <= threshold)
    if eligible.any():
        idx = np.flatnonzero(eligible)
        return int(idx[np.argmax(production[idx])])
    if finite.any():
        idx = np.flatnonzero(finite)
        return int(idx[np.argmin(risk[idx])])
    return 0


def selection_aware_bootstrap(
    data_booked: pd.DataFrame,
    grid: CellGrid,
    pareto_masks: list[np.ndarray],
    variables: list[str],
    threshold: float,
    multiplier: float = float(DEFAULT_RISK_MULTIPLIER),
    annual_coef: float = 1.0,
    rejected_loans: pd.DataFrame | None = None,
    rep_cells: pd.DataFrame | None = None,
    n_bootstraps: int = DEFAULT_N_BOOTSTRAPS,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
) -> SelectionBootstrapResult | None:
    """Bootstrap the optimized headline INCLUDING the selection step.

    Returns ``None`` when the inputs cannot support the analysis (no masks, no
    booked rows mapping into the grid) — callers degrade gracefully.
    """
    if not pareto_masks or data_booked.empty:
        return None

    n_cells = grid.n_cells
    masks = np.asarray(pareto_masks, dtype=float)
    if masks.ndim != 2 or masks.shape[1] != n_cells:
        logger.warning("selection_aware_bootstrap: masks/grid shape mismatch — skipping.")
        return None
    n_candidates = masks.shape[0]

    # --- booked loans → cells ---
    loan_cell = _loans_to_cells(data_booked, variables, grid.cell_index)
    in_grid = loan_cell >= 0
    if not in_grid.any():
        logger.warning("selection_aware_bootstrap: no booked loans map into the grid — skipping.")
        return None
    if (~in_grid).sum():
        logger.debug(f"selection_aware_bootstrap: {(~in_grid).sum()} booked loan(s) outside the grid (excluded).")
    loan_cell = loan_cell[in_grid]
    prod_col = "oa_amt_h0" if "oa_amt_h0" in data_booked.columns else "oa_amt"
    # NaN→0 to match pandas .sum() semantics (loan-level todu_* carry NaNs;
    # np.bincount would otherwise poison every cell sum).
    loan_oa = np.nan_to_num(data_booked[prod_col].to_numpy(float)[in_grid])
    loan_num = np.nan_to_num(data_booked["todu_30ever_h6"].to_numpy(float)[in_grid])
    loan_den = np.nan_to_num(data_booked["todu_amt_pile_h6"].to_numpy(float)[in_grid])
    n_loans = len(loan_cell)

    # --- repesca component over ALL cells (selection varies per replicate) ---
    rep_oa = np.zeros(n_cells)
    rep_num = np.zeros(n_cells)
    rep_den = np.zeros(n_cells)
    rej_cell = np.empty(0, dtype=np.int64)
    rej_amt = np.empty(0)
    base_amt = np.ones(n_cells)
    scalable = np.zeros(n_cells, dtype=bool)
    has_rep = (
        rejected_loans is not None
        and not rejected_loans.empty
        and rep_cells is not None
        and not rep_cells.empty
        and "oa_amt_h0_rep" in rep_cells.columns
    )
    if has_rep:
        cell_pos_all = _loans_to_cells(rep_cells, variables, grid.cell_index)
        for arr, col in ((rep_oa, "oa_amt_h0_rep"), (rep_num, "todu_30ever_h6_rep"), (rep_den, "todu_amt_pile_h6_rep")):
            if col in rep_cells.columns:
                vals = np.nan_to_num(rep_cells[col].to_numpy(float))
                ok = cell_pos_all >= 0
                np.add.at(arr, cell_pos_all[ok], vals[ok])
        rej_cell = _loans_to_cells(rejected_loans, variables, grid.cell_index)
        amt_col = "oa_amt_h0" if "oa_amt_h0" in rejected_loans.columns else "oa_amt"
        rej_amt = np.nan_to_num(rejected_loans[amt_col].to_numpy(float))
        ok = rej_cell >= 0
        base = np.bincount(rej_cell[ok], weights=rej_amt[ok], minlength=n_cells)
        scalable = base > 0
        base_amt = np.where(scalable, base, 1.0)

    def _cell_sums(idx: np.ndarray | None, rng: np.random.RandomState | None) -> tuple[np.ndarray, ...]:
        """Cell-level (production, risk numerator, risk denominator) vectors —
        original data when idx is None, else the resampled replicate."""
        if idx is None:
            b_oa = np.bincount(loan_cell, weights=loan_oa, minlength=n_cells)
            b_num = np.bincount(loan_cell, weights=loan_num, minlength=n_cells)
            b_den = np.bincount(loan_cell, weights=loan_den, minlength=n_cells)
            ratio = np.ones(n_cells)
        else:
            c = loan_cell[idx]
            b_oa = np.bincount(c, weights=loan_oa[idx], minlength=n_cells)
            b_num = np.bincount(c, weights=loan_num[idx], minlength=n_cells)
            b_den = np.bincount(c, weights=loan_den[idx], minlength=n_cells)
            if has_rep and len(rej_amt):
                ridx = rng.randint(0, len(rej_amt), len(rej_amt))
                rc = rej_cell[ridx]
                ok_r = rc >= 0
                resampled = np.bincount(rc[ok_r], weights=rej_amt[ridx][ok_r], minlength=n_cells)
                ratio = np.where(scalable, resampled / base_amt, 1.0)
            else:
                ratio = np.ones(n_cells)
        prod_vec = annual_coef * b_oa + rep_oa * ratio
        num_vec = annual_coef * b_num + rep_num * ratio
        den_vec = annual_coef * b_den + rep_den * ratio
        return prod_vec, num_vec, den_vec

    def _evaluate(prod_vec, num_vec, den_vec) -> tuple[np.ndarray, np.ndarray]:
        prod = masks @ prod_vec
        num = masks @ num_vec
        den = masks @ den_vec
        with np.errstate(divide="ignore", invalid="ignore"):
            risk = np.where(den > 0, multiplier * num / den * 100.0, np.nan)
        return risk, prod

    # --- original-data candidate values + original selection ---
    risk_orig, prod_orig = _evaluate(*_cell_sums(None, None))
    j0 = _select(risk_orig, prod_orig, threshold)

    rng = np.random.RandomState(random_state)
    sel_risk = np.empty(n_bootstraps)
    sel_prod = np.empty(n_bootstraps)
    optimism = np.empty(n_bootstraps)
    reselected = np.empty(n_bootstraps, dtype=bool)
    cand_risks = np.empty((n_bootstraps, n_candidates))

    for r in range(n_bootstraps):
        idx = rng.randint(0, n_loans, n_loans)
        risk_b, prod_b = _evaluate(*_cell_sums(idx, rng))
        cand_risks[r] = risk_b
        j = _select(risk_b, prod_b, threshold)
        sel_risk[r] = risk_b[j]
        sel_prod[r] = prod_b[j]
        # Winner's curse: the replicate-winner's risk on the replicate that
        # crowned it vs the same mask's risk on the original data. For a FIXED
        # mask this gap is mean-zero; selection makes it systematically
        # negative — its negation is the optimism of the reported number.
        optimism[r] = risk_orig[j] - risk_b[j]
        reselected[r] = j != j0

    alpha = (1 - confidence_level) / 2
    lo, hi = alpha * 100, (1 - alpha) * 100

    return SelectionBootstrapResult(
        risk_ci_sel_lower=float(np.nanpercentile(sel_risk, lo)),
        risk_ci_sel_upper=float(np.nanpercentile(sel_risk, hi)),
        production_ci_sel_lower=float(np.nanpercentile(sel_prod, lo)),
        production_ci_sel_upper=float(np.nanpercentile(sel_prod, hi)),
        selection_optimism_pp=float(np.nanmean(optimism)),
        reselect_pct=float(reselected.mean() * 100.0),
        candidate_risk_ci_upper=np.nanpercentile(cand_risks, hi, axis=0),
        candidate_risk_orig=risk_orig,
        candidate_production_orig=prod_orig,
        n_candidates=n_candidates,
        original_selected_index=j0,
    )
