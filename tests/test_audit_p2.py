"""Regression tests for the HRI and partial-order statistical errors (F10/F11)."""

from itertools import permutations

import numpy as np
import pandas as pd
import pytest

from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.pipeline.sensitivity import run_sensitivity_phase
from src.reject_inference import _enforce_multiplier_monotonicity, apply_parceling_adjustment
from src.risk_indicators import HRI_H6
from src.sensitivity import perturb_risk_summary, run_sensitivity_analysis


@pytest.mark.parametrize("frozen", [True, False])
def test_hri_sensitivity_uses_selected_target_for_all_outputs(tmp_path, frozen):
    """Zero perturbation preserves HRI policy; a large increase tightens it."""
    cfg = PreprocessingSettings(
        variables=["a"],
        inference_variables=["a"],
        segment_filter="hri_test",
        keep_vars=["status_name", "reject_reason", "mis_date", "score"],
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={"a": BinConfig(source_col="score", output_col="a", bin_edges=[-np.inf, 50, np.inf])},
        indicators=["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"],
        risk_indicator="hri_h6",
        optimum_hri=1.0,
        optimum_risk=2.0,
        run_sensitivity=True,
        sensitivity_levels=[0.0, 100.0],
    )
    cells = pd.DataFrame(
        {
            "a": [1, 2],
            "oa_amt_h0": [100.0, 100.0],
            "todu_30ever_h6": [0.1, 0.1],
            "todu_amt_pile_h6": [700.0, 700.0],
            "h_num_h6": [0.4, 1.2],
            "h_den_h6": [100.0, 100.0],
        }
    )
    output = OutputPaths(base_dir=tmp_path)
    output.data_dir.mkdir(parents=True)
    if frozen:
        pd.DataFrame({"acceptance_mask": ["1,1"]}).to_csv(output.optimal_solution_csv("_base"), index=False)
    run_sensitivity_phase(cells, pd.DataFrame(), cfg, output)
    sensitivity = pd.read_csv(output.sensitivity_analysis_csv("_base"))
    assert sensitivity.risk_indicator.tolist() == ["hri_h6", "hri_h6"]
    assert sensitivity.n_flipped.tolist() == [0, 1]
    assert sensitivity.new_production.tolist() == [200.0, 100.0]
    np.testing.assert_allclose(sensitivity.new_risk, [0.8, 0.8])
    detail = pd.read_csv(output.sensitivity_analysis_csv("_cell_detail")).set_index("a")
    assert pd.isna(detail.loc[1, "flip_threshold_pct"])
    assert detail.loc[2, "flip_threshold_pct"] == 100.0
    assert detail.loc[2, "flip_direction"] == "accept_to_reject"
    marginal = pd.read_csv(output.cell_marginal_impact_csv("_base")).set_index("a")
    np.testing.assert_allclose(marginal.cell_risk, [0.4, 1.2])
    np.testing.assert_allclose(marginal.delta_risk_pct, [0.4, -0.4])


def test_hri_perturbation_scales_its_components_and_preserves_b2():
    cells = pd.DataFrame(
        {
            "h_num_h6": [3.0],
            "h_num_h6_boo": [1.0],
            "h_num_h6_rep": [2.0],
            "h_den_h6": [100.0],
            "todu_30ever_h6": [7.0],
            "todu_30ever_h6_rep": [4.0],
        }
    )
    changed = perturb_risk_summary(cells, 50.0, risk_indicator=HRI_H6)
    np.testing.assert_allclose(changed[["h_num_h6", "h_num_h6_boo", "h_num_h6_rep"]], [[4.5, 1.5, 3.0]])
    for col in ["h_den_h6", "todu_30ever_h6", "todu_30ever_h6_rep"]:
        pd.testing.assert_series_equal(changed[col], cells[col])


def test_hri_sensitivity_keeps_swapin_cap_on_b2_basis():
    cells = pd.DataFrame(
        {
            "a": [1, 2],
            "oa_amt_h0": [100.0, 100.0],
            "oa_amt_h0_rep": [0.0, 100.0],
            "todu_30ever_h6": [0.1, 8.0],
            "todu_amt_pile_h6": [700.0, 700.0],
            "todu_30ever_h6_rep": [0.0, 8.0],
            "todu_amt_pile_h6_rep": [0.0, 700.0],
            "h_num_h6": [0.1, 0.1],
            "h_den_h6": [100.0, 100.0],
        }
    )
    result = run_sensitivity_analysis(
        cells,
        ["a"],
        [],
        7.0,
        list(cells.columns[1:]),
        np.array([1, 0]),
        1.0,
        perturbation_levels=[0.0],
        max_swapin_risk=2.0,
        risk_indicator=HRI_H6,
    )
    assert result.iloc[0].n_flipped == 0
    assert result.iloc[0].new_production == 100.0
    assert result.iloc[0].new_risk == pytest.approx(0.1)


@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_branching_isotonic_public_parceling_is_least_squares_and_order_invariant(order):
    rates = pd.DataFrame(
        {
            "a": [0, 1, 2],
            "b": [0, 2, 1],
            "acceptance_rate": [0.01, 0.25, 0.99],
            "n_booked": [10, 250, 990],
            "n_score_rejected": [990, 750, 10],
        }
    )
    rep = rates[["a", "b"]].assign(todu_30ever_h6=1.0).iloc[list(order)]
    result = apply_parceling_adjustment(
        rep,
        rates,
        ["a", "b"],
        reject_uplift_factor=2.0,
        max_risk_multiplier=3.0,
        enforce_monotonicity=True,
        quiet=True,
    ).sort_values("a")
    np.testing.assert_allclose(result.reject_risk_multiplier, [2.0, 2.5, 2.0], atol=1e-8)
    np.testing.assert_allclose(result.todu_30ever_h6, [2.0, 2.5, 2.0], atol=1e-8)


def _partitions(items):
    """Enumerate every possible fitted level-set partition for a tiny reference."""
    if not items:
        yield []
        return
    first, *rest = items
    for partition in _partitions(rest):
        yield [[first], *partition]
        for i in range(len(partition)):
            yield [*partition[:i], [first, *partition[i]], *partition[i + 1 :]]


@pytest.mark.parametrize("seed", range(8))
def test_weighted_isotonic_matches_exhaustive_level_set_reference(seed):
    """Independent reference: enumerate ALL partitions, retaining only feasible fits."""
    rng = np.random.RandomState(seed)
    coords = np.array([[0, 0], [0, 1], [1, 0], [2, 1], [1, 2]])
    y = rng.uniform(1, 3, 5)
    w = rng.uniform(1, 100, 5)
    dominates = np.all(coords[:, None] >= coords[None, :], axis=2)
    best_sse = np.inf
    for partition in _partitions(list(range(5))):
        fitted = np.empty(5)
        for block in partition:
            fitted[block] = np.average(y[block], weights=w[block])
        if np.all((fitted[:, None] >= fitted[None, :] - 1e-10) | ~dominates):
            sse = np.dot(w, (fitted - y) ** 2)
            if sse < best_sse:
                best_sse, best_fit = sse, fitted.copy()
    cells = pd.DataFrame(coords, columns=["a", "b"]).assign(reject_risk_multiplier=y, weight=w)
    result = _enforce_multiplier_monotonicity(cells, ["a", "b"], weight_col="weight", quiet=True)
    np.testing.assert_allclose(result.reject_risk_multiplier, best_fit, atol=2e-6)
    assert np.dot(w, (result.reject_risk_multiplier - y) ** 2) == pytest.approx(best_sse, abs=1e-8)


def test_isotonic_solver_failure_does_not_silently_return_an_approximation():
    cells = pd.DataFrame({"a": [0, 1, 2], "b": [0, 2, 1], "reject_risk_multiplier": [2.98, 2.5, 1.02]})
    with pytest.raises(RuntimeError, match="projection failed"):
        _enforce_multiplier_monotonicity(cells, ["a", "b"], max_iterations=1, quiet=True)
