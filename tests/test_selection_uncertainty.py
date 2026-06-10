"""Tests for the selection-aware bootstrap (audit #28, Phase B)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.optimization_utils import CellGrid
from src.selection_uncertainty import selection_aware_bootstrap


def _grid_2x2():
    """2×2 grid; cell (2,2) is the risky boundary cell."""
    rows = []
    for v0 in (1, 2):
        for v1 in (1, 2):
            rows.append(
                {
                    "v0": v0,
                    "v1": v1,
                    "oa_amt_h0": 1000.0,
                    "todu_30ever_h6": 1.0,
                    "todu_amt_pile_h6": 700.0,
                }
            )
    return CellGrid.from_summary(pd.DataFrame(rows), ["v0", "v1"])


def _booked_loans(n_risky=40, risky_rate=0.10, seed=0):
    """Safe cells have near-zero risk; cell (2,2) carries noisy risk around
    7*rate (few loans → high sampling variance at the boundary)."""
    rng = np.random.RandomState(seed)
    rows = []
    for v0, v1 in [(1, 1), (1, 2), (2, 1)]:
        for _ in range(200):
            rows.append({"v0": v0, "v1": v1, "oa_amt_h0": 100.0, "todu_30ever_h6": 0.0, "todu_amt_pile_h6": 100.0})
    for _ in range(n_risky):
        bad = rng.binomial(1, risky_rate)
        rows.append({"v0": 2, "v1": 2, "oa_amt_h0": 100.0, "todu_30ever_h6": float(bad), "todu_amt_pile_h6": 7.0})
    return pd.DataFrame(rows)


MASK_SAFE = np.array([1, 1, 1, 0])  # rejects the risky boundary cell
MASK_ALL = np.array([1, 1, 1, 1])  # accepts it (more production, near-threshold risk)


def test_returns_none_without_masks():
    grid = _grid_2x2()
    assert selection_aware_bootstrap(_booked_loans(), grid, [], ["v0", "v1"], threshold=1.0) is None


def test_single_candidate_no_reselection_and_negligible_optimism():
    grid = _grid_2x2()
    res = selection_aware_bootstrap(
        _booked_loans(),
        grid,
        [MASK_ALL],
        ["v0", "v1"],
        threshold=50.0,
        n_bootstraps=200,
        random_state=42,
    )
    assert res is not None
    assert res.reselect_pct == 0.0
    # For a FIXED mask the orig-vs-replicate gap is mean-zero — no curse.
    assert abs(res.selection_optimism_pp) < 0.5
    assert res.risk_ci_sel_lower <= res.risk_ci_sel_upper


def test_boundary_noise_produces_reselection_and_positive_optimism():
    """Two candidates: rejecting vs accepting a noisy boundary cell whose true
    risk sits AT the threshold. Replicates where the cell looks lucky select
    the bigger mask — the classic winner's curse — so re-selection happens and
    the optimism estimate is positive."""
    grid = _grid_2x2()
    booked = _booked_loans(n_risky=40, risky_rate=0.10)
    # true risky-cell rate ~ 7*4/(40*7) → portfolio risk of MASK_ALL near the threshold
    full_risk = 7.0 * booked["todu_30ever_h6"].sum() / booked["todu_amt_pile_h6"].sum() * 100
    res = selection_aware_bootstrap(
        booked,
        grid,
        [MASK_SAFE, MASK_ALL],
        ["v0", "v1"],
        threshold=float(full_risk),  # boundary: noise decides which mask wins
        n_bootstraps=400,
        random_state=42,
    )
    assert res is not None
    assert res.reselect_pct > 5.0, "boundary noise must flip the selection in some replicates"
    assert res.selection_optimism_pp > 0.0, "winner's curse: reported risk understates"
    # per-candidate CI uppers exist and are ordered sensibly (risky mask higher)
    assert res.candidate_risk_ci_upper.shape == (2,)
    assert res.candidate_risk_ci_upper[1] > res.candidate_risk_ci_upper[0]


def test_deterministic_under_seed():
    grid = _grid_2x2()
    booked = _booked_loans()
    kwargs = dict(
        data_booked=booked,
        grid=grid,
        pareto_masks=[MASK_SAFE, MASK_ALL],
        variables=["v0", "v1"],
        threshold=1.0,
        n_bootstraps=50,
        random_state=7,
    )
    a = selection_aware_bootstrap(**kwargs)
    b = selection_aware_bootstrap(**kwargs)
    assert a.risk_ci_sel_lower == b.risk_ci_sel_lower
    assert a.selection_optimism_pp == b.selection_optimism_pp


def test_repesca_component_shifts_candidate_risk():
    """With a high-risk rep component on the boundary cell, the accepting
    candidate's original risk must exceed its booked-only value."""
    grid = _grid_2x2()
    booked = _booked_loans()
    rep_cells = pd.DataFrame(
        {
            "v0": [2],
            "v1": [2],
            "oa_amt_h0_rep": [5000.0],
            "todu_30ever_h6_rep": [50.0],
            "todu_amt_pile_h6_rep": [700.0],
        }
    )
    rejected = pd.DataFrame({"v0": [2] * 30, "v1": [2] * 30, "oa_amt_h0": np.linspace(50, 500, 30)})

    res_no_rep = selection_aware_bootstrap(
        booked, grid, [MASK_ALL], ["v0", "v1"], threshold=50.0, n_bootstraps=30, random_state=1
    )
    res_rep = selection_aware_bootstrap(
        booked,
        grid,
        [MASK_ALL],
        ["v0", "v1"],
        threshold=50.0,
        n_bootstraps=30,
        random_state=1,
        rejected_loans=rejected,
        rep_cells=rep_cells,
    )
    assert res_rep.candidate_risk_orig[0] > res_no_rep.candidate_risk_orig[0]
    assert res_rep.production_ci_sel_upper > res_no_rep.production_ci_sel_upper


def test_nan_loan_values_do_not_poison_cell_sums():
    """Real-data regression: loan-level todu_* columns carry NaNs; np.bincount
    propagates them (unlike pandas .sum()), which NaN'd every candidate's risk
    and returned an all-NaN selection-aware CI on the first real run."""
    grid = _grid_2x2()
    booked = _booked_loans()
    booked.loc[booked.index[:5], "todu_30ever_h6"] = np.nan
    booked.loc[booked.index[5:8], "todu_amt_pile_h6"] = np.nan

    res = selection_aware_bootstrap(
        booked, grid, [MASK_ALL], ["v0", "v1"], threshold=50.0, n_bootstraps=30, random_state=3
    )
    assert res is not None
    assert np.isfinite(res.risk_ci_sel_lower)
    assert np.isfinite(res.risk_ci_sel_upper)
    assert np.isfinite(res.selection_optimism_pp)


def test_select_with_ci_margin():
    """Phase C rule: max production among candidates whose CI-UPPER clears the
    threshold; None when no candidate qualifies (caller falls back loudly)."""
    from src.selection_uncertainty import SelectionBootstrapResult, select_with_ci_margin

    res = SelectionBootstrapResult(
        risk_ci_sel_lower=0.0,
        risk_ci_sel_upper=0.0,
        production_ci_sel_lower=0.0,
        production_ci_sel_upper=0.0,
        selection_optimism_pp=0.0,
        reselect_pct=0.0,
        candidate_risk_ci_upper=np.array([0.8, 1.05, 1.4]),
        candidate_risk_orig=np.array([0.7, 0.95, 1.1]),
        candidate_production_orig=np.array([100.0, 200.0, 300.0]),
        n_candidates=3,
        original_selected_index=2,
    )
    # threshold 1.1: candidates 0 and 1 qualify by CI-upper; pick the bigger one.
    # (the point rule would pick candidate 2 — risk 1.1 <= 1.1 — illustrating
    # the conservatism of the noise-margin rule)
    assert select_with_ci_margin(res, 1.1) == 1
    # tight threshold: only candidate 0 qualifies
    assert select_with_ci_margin(res, 0.9) == 0
    # unattainable margin
    assert select_with_ci_margin(res, 0.5) is None


def test_config_selection_risk_basis_validation():
    from src.config import PreprocessingSettings

    base = dict(
        keep_vars=["mis_date"],
        indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        segment_filter="seg",
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-12-31",
        variables=["a", "b"],
        octroi_bins=[0.0, 1.0],
        efx_bins=[0.0, 1.0],
    )
    assert PreprocessingSettings(**base).selection_risk_basis == "point"  # default OFF
    assert PreprocessingSettings(**base, selection_risk_basis="ci_upper").selection_risk_basis == "ci_upper"
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PreprocessingSettings(**base, selection_risk_basis="bogus")
