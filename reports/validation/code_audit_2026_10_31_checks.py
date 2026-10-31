"""Reproduce the 2026-10-31 second-verifier audit findings.

Run: uv run python reports/validation/code_audit_2026_10_31_checks.py

All six findings are fixed; every check asserts the CORRECTED behavior (F3-F6 in PR #264, F1/F2 after it).
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.mr_extrapolation import extrapolate_h3_to_h6
from src.analysis.mr_pipeline import _assign_tiered_risk
from src.analysis.selection_uncertainty import SelectionBootstrapResult, select_with_ci_margin
from src.core.config import PreprocessingSettings
from src.model.inference_optimized import _prepare_pipeline_data
from src.model.training import _get_regression_weights, process_dataset
from src.optimize.reject_inference import apply_parceling_adjustment
from src.optimize.reject_inference_optimizer import OptimizerInputs, evaluate_ri_params
from src.pipeline.scenarios import _resolve_selection
from src.pipeline.segment_runner import _config_fingerprint, _config_hash, _config_snapshot
from src.report.plots import RiskProductionVisualizer

_HRI = {
    "risk_indicator": "hri_h6",
    "optimum_hri": 2.0,
}


def _base_settings() -> dict:
    return {
        "keep_vars": ["mis_date"],
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"],
        "segment_filter": "seg_a",
        "date_ini_book_obs": "2024-01-01",
        "date_fin_book_obs": "2024-12-31",
        "variables": ["sc_octroi_new_clus", "new_efx_clus"],
        "octroi_bins": [0.0, 1.0, 2.0],
        "efx_bins": [0.0, 1.0, 2.0],
    }


def check_mr_nonlinear_aggregation() -> dict:
    n = 100
    data = pd.DataFrame(
        {
            "x": [0] * n,
            "y": [0] * n,
            "status_name": ["booked"] * n,
            "mis_date": pd.to_datetime(["2025-06-01"] * n),
            "todu_30ever_h3": [2.5] + [0.0] * (n - 1),
            "todu_amt_pile_h3": [100.0] * n,
            "b2_ever_h6_tmp": [0.002] * n,
            "oa_amt": [100.0] * n,
        }
    )
    comparison = pd.DataFrame(
        {"x": [0], "y": [0], "h6_h3_ratio": [2.0], "b2_main_h3": [0.001], "b2_mr_h3": [0.001], "n_obs_mr_h3": [n]}
    )
    results = {}
    for alpha in (0.5, 1.0, 2.0):
        out = _assign_tiered_risk(
            data.copy(),
            comparison,
            comparison,
            ["x", "y"],
            multiplier=7.0,
            multiplier_h3=4.0,
            mr_extrapolation_method="power",
            mr_extrapolation_curvature=alpha,
            maturity_reference_date=pd.Timestamp("2025-10-01"),
        )
        assert (out["_mr_tier"] == 2).all()
        expected = float(extrapolate_h3_to_h6(0.001, 2.0, "power", alpha, 0.001))
        actual = float(out["b2_ever_h6_tmp"].mean())
        results[str(alpha)] = {
            "bin_curve_risk_pct": expected * 100,
            "loan_reconstruction_risk_pct": actual * 100,
            "ratio": actual / expected,
        }
    # Fixed (F1): the curve is evaluated once on the bin H3, so the reconstruction matches it at every curvature.
    assert all(np.isclose(r["ratio"], 1.0) for r in results.values())
    return results


def check_auxiliary_missingness() -> dict:
    data = pd.DataFrame(
        {
            "x": [0, 0, 1],
            "y": [0, 0, 0],
            "status_name": ["booked"] * 3,
            "oa_amt_h0": [100.0] * 3,
            "todu_30ever_h6": [0.0, 10.0, 0.0],
            "todu_amt_pile_h6": [100.0] * 3,
            "todu_30ever_h3": [0.0, np.nan, 0.0],
            "todu_amt_pile_h3": [100.0, np.nan, 100.0],
        }
    )
    required = ["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]
    a, _, _ = _prepare_pipeline_data(data, (), ["x", "y"], required, "b2_ever_h6", 1.0)
    b, _, _ = _prepare_pipeline_data(
        data, (), ["x", "y"], required + ["todu_30ever_h3", "todu_amt_pile_h3"], "b2_ever_h6", 1.0
    )
    # Fixed (F2): an auxiliary H3 pair no longer drops loans with complete H6 outcomes.
    assert len(a) == 3 and len(b) == 3 and b["todu_30ever_h6"].sum() == 10.0
    return {
        "h6_valid_rows": len(a),
        "rows_after_enabling_h3_columns": len(b),
        "h6_default_numerator_before": float(a["todu_30ever_h6"].sum()),
        "h6_default_numerator_after": float(b["todu_30ever_h6"].sum()),
    }


def check_hri_tuning_weights() -> dict:
    import src.model.optuna_tuning as tuning

    data = pd.DataFrame(
        [
            {
                "x": x,
                "y": y,
                "todu_30ever_h6": float(i % 2),
                "todu_amt_pile_h6": 100.0 if x == 0 else 1.0,
                "h_num_h6": 0.1 + x,
                "h_den_h6": 1.0 if x == 0 else 100.0,
            }
            for x in range(3)
            for y in range(2)
            for i in range(12)
        ]
    )
    indicators = [c for c in data.columns if c not in ("x", "y")]
    seen = []

    def observe(frame, target_var="b2_ever_h6"):
        seen.append(target_var)
        return _get_regression_weights(frame, target_var)

    with (
        patch.object(tuning, "_get_regression_weights", side_effect=observe),
        patch.object(tuning, "XGBRegressor", side_effect=lambda **kw: DummyRegressor()),
        patch.object(tuning, "LGBMRegressor", side_effect=lambda **kw: DummyRegressor()),
    ):
        tuning.tune_tree_models(data, (), ["x", "y"], indicators, "hri_h6", 1.0, 0.0, cv_folds=2, n_trials=1)
    assert seen and set(seen) == {"hri_h6"}  # fixed: tuning weights follow the HRI denominator
    agg = process_dataset(data, (), ["x", "y"], indicators, "hri_h6", 1.0, ["x", "y"], 0)
    return {
        "weighting_targets_observed_during_hri_tree_tuning": sorted(set(seen)),
        "b2_weights": _get_regression_weights(agg).round(4).tolist(),
        "hri_weights": _get_regression_weights(agg, "hri_h6").round(4).tolist(),
    }


def check_ri_hri_mismatch() -> dict:
    indicators = ["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"]
    rep = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "oa_amt_h0": [1000.0],
            "todu_30ever_h6": [10.0],
            "todu_amt_pile_h6": [1000.0],
            "h_num_h6": [50.0],
            "h_den_h6": [1000.0],
        }
    )
    booked = rep.rename(columns={c: c + "_boo" for c in indicators})
    rates = pd.DataFrame({"x": [0], "y": [0], "acceptance_rate": [0.5], "n_booked": [100], "n_score_rejected": [100]})
    inputs = OptimizerInputs(booked, rep, rates, 1.0, ["x", "y"], indicators, multiplier=1.0)
    captured = []

    def capture(grid, *args, **kwargs):
        captured.append(grid.cell_data.copy())
        return np.ones(grid.n_cells, dtype=int)

    with patch("src.optimize.reject_inference_optimizer.milp_solve_cutoffs", side_effect=capture):
        evaluate_ri_params(inputs, 1.0, 5.0, 2.0)
    runtime = apply_parceling_adjustment(
        rep.copy(),
        rates,
        ["x", "y"],
        reject_uplift_factor=1.0,
        max_risk_multiplier=5.0,
        apply_hri_multiplier=True,
        quiet=True,
    )
    trial_hnum = float(captured[0]["h_num_h6_rep"].iloc[0])
    deployed_hnum = float(runtime["h_num_h6"].iloc[0])
    assert deployed_hnum > trial_hnum  # the trial path itself is unchanged...
    try:  # ...so the combination is now refused at config validation (fix for F4)
        PreprocessingSettings(**_base_settings(), **_HRI, run_ri_optimizer=True)
        refused = False
    except ValueError:
        refused = True
    assert refused
    return {
        "trial_rejected_hri_numerator": trial_hnum,
        "runtime_rejected_hri_numerator": deployed_hnum,
        "hri_with_ri_optimizer_refused": refused,
    }


def check_resimulation_identity() -> dict:
    original = SimpleNamespace(
        variables=["x", "y"],
        multiplier=7.0,
        data_path="snapshot.sas7bdat",
        bins={v: SimpleNamespace(bin_edges=[-np.inf, 0.0, np.inf], source_col=v) for v in ("x", "y")},
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-12-01",
        reject_uplift_factor=1.0,
        directions={"x": -1, "y": -1},
    )
    variants = {}
    for field, value in (
        ("date_fin_book_obs", "2025-12-01"),
        ("reject_uplift_factor", 5.0),
        ("directions", {"x": 1, "y": -1}),
    ):
        changed = deepcopy(original)
        setattr(changed, field, value)
        variants[field] = _config_hash(changed) == _config_hash(original)
    assert all(variants.values())  # the legacy 4-field hash is still blind (kept for pre-fix runs)...
    base = _config_fingerprint(_config_snapshot(PreprocessingSettings(**_base_settings())))
    detected = {}
    for field, value in (
        ("date_fin_book_obs", "2025-06-30"),
        ("reject_uplift_factor", 5.0),
        ("directions", {"sc_octroi_new_clus": 1, "new_efx_clus": 1}),
    ):
        changed = PreprocessingSettings(**{**_base_settings(), field: value})
        detected[field] = _config_fingerprint(_config_snapshot(changed)) != base
    assert all(detected.values())  # ...the full fingerprint (fix for F5) detects every change
    return {"legacy_hash_unchanged": variants, "fingerprint_detects_change": detected}


def check_ci_fallback() -> dict:
    result = SelectionBootstrapResult(
        0.7, 1.0, 100.0, 200.0, 0.01, 10.0, np.array([1.1, 1.3]), np.array([0.8, 0.95]), np.array([100.0, 200.0]), 2, 1
    )
    margin_selected = select_with_ci_margin(result, 1.0)
    assert margin_selected is None
    # scenarios.py:205-220 leaves target_sol_fac=None on this branch; the visualizer
    # consequently selects by point risk and calls the result feasible.
    vis = RiskProductionVisualizer.__new__(RiskProductionVisualizer)
    vis.target_sol_fac = None
    vis.risk_col = "b2_ever_h6"
    vis.optimum_risk = 1.0
    vis.data_summary = pd.DataFrame({"sol_fac": [0, 1], "b2_ever_h6": [0.8, 0.95], "oa_amt_h0": [100.0, 200.0]})
    selected = vis._get_selected_solution_row()
    assert int(selected.iloc[0]["sol_fac"]) == 1 and vis._selection_feasible
    # Fix for F6: the summary row now records that the requested CI margin was not applied.
    _, basis_applied, ci_feasible = _resolve_selection("ci_upper", result, 1.0, "audit")
    assert (basis_applied, ci_feasible) == ("point", False)
    return {
        "selection_basis_applied": basis_applied,
        "ci_upper_feasible": ci_feasible,
        "ci_eligible_candidates": 0,
        "fallback_selected_sol_fac": 1,
        "selected_point_risk_pct": 0.95,
        "selected_risk_ci_upper_pct": 1.3,
        "target_pct": 1.0,
        "reported_feasible": bool(vis._selection_feasible),
    }


def main() -> None:
    logger.remove()
    checks = {
        "F1_mr_nonlinear_aggregation": check_mr_nonlinear_aggregation(),
        "F2_auxiliary_missingness": check_auxiliary_missingness(),
        "F3_hri_tuning_weights": check_hri_tuning_weights(),
        "F4_hri_ri_trial_runtime_mismatch": check_ri_hri_mismatch(),
        "F5_resimulation_identity": check_resimulation_identity(),
        "F6_ci_upper_fallback": check_ci_fallback(),
    }
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
