"""Regression tests for the 2026-10-31 second-verifier audit (findings F3–F6).

F1 (MR per-loan nonlinear extrapolation) and F2 (auxiliary-indicator listwise deletion) change
numbers and are fixed separately with real-data validation.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.analysis.selection_uncertainty import SelectionBootstrapResult
from src.core.config import OutputPaths, PreprocessingSettings
from src.model.training import _get_regression_weights
from src.pipeline.scenarios import _resolve_selection
from src.pipeline.segment_runner import (
    ResimulationError,
    _config_fingerprint,
    _config_snapshot,
    _save_resimulation_artifacts,
    run_resimulation,
)


def _settings(**overrides):
    values = {
        "keep_vars": ["mis_date"],
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        "segment_filter": "seg_a",
        "date_ini_book_obs": "2024-01-01",
        "date_fin_book_obs": "2024-12-31",
        "variables": ["sc_octroi_new_clus", "new_efx_clus"],
        "octroi_bins": [0.0, 1.0, 2.0],
        "efx_bins": [0.0, 1.0, 2.0],
    }
    values.update(overrides)
    return PreprocessingSettings(**values)


# ---------------------------------------------------------------- F3: tuner exposure basis


def test_hri_tree_tuning_weights_use_hri_denominator():
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
    assert seen and set(seen) == {"hri_h6"}


# ---------------------------------------------------------------- F4: HRI + RI optimizer refused


def test_hri_target_with_ri_optimizer_is_refused():
    hri = {
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"],
        "risk_indicator": "hri_h6",
        "optimum_hri": 2.0,
    }
    assert _settings(**hri).run_ri_optimizer is False  # still valid without the optimizer
    with pytest.raises(ValueError, match="run_ri_optimizer"):
        _settings(**hri, run_ri_optimizer=True)


# ---------------------------------------------------------------- F5: resimulation identity


def test_fingerprint_ignores_targets_but_catches_population_and_risk_inputs():
    base = _config_fingerprint(_config_snapshot(_settings()))
    assert _config_fingerprint(_config_snapshot(_settings(optimum_risk=3.3, risk_step=0.2))) == base
    for change in (
        {"date_fin_book_obs": "2025-06-30"},
        {"reject_uplift_factor": 4.0},
        {"directions": {"sc_octroi_new_clus": 1, "new_efx_clus": 1}},
    ):
        assert _config_fingerprint(_config_snapshot(_settings(**change))) != base, change


def _resim_fixture(tmp_path, saved_settings):
    output = OutputPaths(base_dir=tmp_path)
    output.ensure_dirs()
    frame = pd.DataFrame({"a": [1]})
    _save_resimulation_artifacts(
        output, frame, frame, 1.0, 1.0, 1.0, saved_settings, config_snapshot=_config_snapshot(saved_settings)
    )
    for path in (output.pareto_solutions_csv, output.data_summary_desagregado_csv("_base")):
        frame.to_csv(path, index=False)
    return output


def test_resimulation_refuses_changed_config(monkeypatch, tmp_path):
    output = _resim_fixture(tmp_path, _settings())
    changed = _settings(reject_uplift_factor=4.0)
    monkeypatch.setattr("src.pipeline.segment_runner.load_and_validate_config", lambda p: (changed, None, None, 1.0))
    with pytest.raises(ResimulationError, match="reject_uplift_factor"):
        run_resimulation("cfg.toml", [1.0], output=output)


def test_effective_cli_overrides_enter_the_fingerprint(monkeypatch, tmp_path):
    """Follow-up: the snapshot is taken AFTER --baseline is applied, so a baseline run's artifacts
    cannot pass as an ordinary run's; the DQ escape hatch stays an allowed difference."""
    import src.pipeline.segment_runner as runner

    captured = {}

    def capture(settings):
        captured["baseline_mode"] = settings.baseline_mode
        raise RuntimeError("stop after snapshot")

    monkeypatch.setattr(runner, "load_and_validate_config", lambda p: (_settings(), None, None, 1.0))
    monkeypatch.setattr(runner, "_config_snapshot", capture)
    with pytest.raises(RuntimeError, match="stop after snapshot"):
        runner.run_pipeline("cfg.toml", baseline_mode=True, output=OutputPaths(base_dir=tmp_path))
    assert captured["baseline_mode"] is True

    base = _config_fingerprint(_config_snapshot(_settings()))
    assert _config_fingerprint(_config_snapshot(_settings(baseline_mode=True))) != base
    assert _config_fingerprint(_config_snapshot(_settings(dq_allow_warnings=True))) == base


def test_resimulation_meta_records_snapshot(tmp_path):
    output = _resim_fixture(tmp_path, _settings())
    meta = json.loads(Path(output.resimulation_meta_json).read_text())
    assert meta["config_fingerprint"] == _config_fingerprint(meta["config_snapshot"])
    assert "optimum_risk" not in meta["config_snapshot"]


# ---------------------------------------------------------------- F6: ci_upper fallback is recorded


def _boot(ci_upper):
    return SelectionBootstrapResult(
        0.7, 1.0, 100.0, 200.0, 0.01, 10.0, np.array(ci_upper), np.array([0.8, 0.95]), np.array([100.0, 200.0]), 2, 1
    )


def test_point_basis_is_recorded_as_point():
    assert _resolve_selection("point", None, 1.0, "s") == (None, "point", None)


def test_ci_upper_fallback_records_unmet_margin():
    assert _resolve_selection("ci_upper", _boot([1.1, 1.3]), 1.0, "s") == (None, "point", False)
    assert _resolve_selection("ci_upper", None, 1.0, "s") == (None, "point", False)


def test_ci_upper_success_records_ci_basis():
    sol, basis, feasible = _resolve_selection("ci_upper", _boot([0.9, 1.3]), 1.0, "s")
    assert (basis, feasible) == ("ci_upper", True) and sol is not None


# ---------------------------------------------------------------- F1: MR H3→H6 at the bin level


def _tier2_frame(h3_num, pile_h3, oa):
    n = len(h3_num)
    return pd.DataFrame(
        {
            "x": [0] * n,
            "y": [0] * n,
            "status_name": ["booked"] * n,
            "mis_date": pd.to_datetime(["2025-06-01"] * n),  # 4 months before the reference: tier 2
            "todu_30ever_h3": h3_num,
            "todu_amt_pile_h3": pile_h3,
            "b2_ever_h6_tmp": [0.01] * n,
            "oa_amt": oa,
        }
    )


def _assign(frame, method, alpha, ratio=2.0, main_h3=0.001):
    from src.analysis.mr_pipeline import _assign_tiered_risk

    comparison = pd.DataFrame(
        {
            "x": [0],
            "y": [0],
            "h6_h3_ratio": [ratio],
            "b2_main_h3": [main_h3],
            "b2_mr_h3": [main_h3],
            "n_obs_mr_h3": [1000],
        }
    )
    return _assign_tiered_risk(
        frame.copy(),
        comparison,
        comparison,
        ["x", "y"],
        multiplier=7.0,
        multiplier_h3=4.0,
        mr_extrapolation_method=method,
        mr_extrapolation_curvature=alpha,
        maturity_reference_date=pd.Timestamp("2025-10-01"),
    )


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_power_curve_is_evaluated_once_on_the_bin(alpha):
    from src.analysis.mr_extrapolation import extrapolate_h3_to_h6

    rng = np.random.RandomState(0)
    n = 200
    pile = rng.uniform(50, 500, n)
    num = np.where(rng.rand(n) < 0.03, pile * rng.uniform(0.01, 0.05, n), 0.0)  # sparse H3 defaults (~0.3% bin H3)
    frame = _tier2_frame(num, pile, pile * 1.1)  # oa proportional to the H3 pile
    out = _assign(frame, "power", alpha, main_h3=0.004)
    assert (out["_mr_tier"] == 2).all()
    bin_h3 = 4.0 * num.sum() / pile.sum()  # deviation from main != 1
    expected = float(extrapolate_h3_to_h6(bin_h3, 2.0, "power", alpha, 0.004))
    weighted = float(np.average(out["b2_ever_h6_tmp"], weights=frame["oa_amt"]))
    assert weighted == pytest.approx(max(expected, bin_h3), rel=1e-4)


def test_linear_method_unchanged_when_oa_tracks_h3_exposure():
    num = np.array([0.0, 30.0, 0.0, 5.0])
    pile = np.array([100.0, 200.0, 50.0, 100.0])
    out = _assign(_tier2_frame(num, pile, pile), "linear", 1.0, ratio=2.5)
    assert np.allclose(out["b2_ever_h6_tmp"].values, 4.0 * num / pile * 2.5, atol=1e-5)


@pytest.mark.parametrize("method,alpha", [("power", 2.0), ("power", 0.5), ("linear", 1.0)])
@pytest.mark.parametrize("defaulter_oa", [10.0, 1000.0])
def test_reconstruction_matches_bin_forecast_with_unequal_exposures(method, alpha, defaulter_oa):
    """Follow-up: the H6 exposure is pro-rated by oa_amt, so the oa-weighted bin rate must equal the
    bin forecast even when a defaulter's oa differs from its H3 pile (was x9.2 / x0.10 off)."""
    from src.analysis.mr_extrapolation import extrapolate_h3_to_h6

    num = np.array([0.025] + [0.0] * 99)
    pile = np.full(100, 100.0)
    oa = np.array([defaulter_oa] + [100.0] * 99)
    out = _assign(_tier2_frame(num, pile, oa), method, alpha)
    bin_h3 = 4.0 * num.sum() / pile.sum()
    target = max(float(extrapolate_h3_to_h6(bin_h3, 2.0, method, alpha, 0.001)), bin_h3)
    assert float(np.average(out["b2_ever_h6_tmp"], weights=oa)) == pytest.approx(target, rel=1e-4)


def test_h3_floor_applies_to_the_bin_multiplier():
    num = np.array([0.0, 20.0, 0.0, 0.0])
    pile = np.array([100.0] * 4)
    out = _assign(_tier2_frame(num, pile, pile), "linear", 1.0, ratio=0.5)  # ratio < 1: floor -> m = 1
    assert np.allclose(out["b2_ever_h6_tmp"].values, np.round(4.0 * num / pile, 6))


# ---------------------------------------------------------------- F2: training completeness


def _f2_frame():
    return pd.DataFrame(
        {
            "x": [0, 0, 1],
            "y": [0, 0, 0],
            "status_name": ["booked"] * 3,
            "oa_amt_h0": [100.0, np.nan, 100.0],
            "todu_30ever_h6": [0.0, 10.0, np.nan],
            "todu_amt_pile_h6": [100.0, 100.0, 100.0],
            "todu_30ever_h3": [0.0, np.nan, 0.0],
            "todu_amt_pile_h3": [100.0, np.nan, 100.0],
            "h_num_h6": [np.nan, 1.0, np.nan],
            "h_den_h6": [10.0, 10.0, np.nan],
        }
    )


def test_b2_training_requires_only_features_and_h6_pair():
    from src.model.inference_optimized import _prepare_pipeline_data

    indicators = [c for c in _f2_frame().columns if c not in ("x", "y", "status_name")]
    booked, _, _ = _prepare_pipeline_data(_f2_frame(), (), ["x", "y"], indicators, "b2_ever_h6", 7.0)
    # row 1 (missing H3 + oa_amt_h0) is kept; row 2 (missing the H6 numerator) is dropped
    assert len(booked) == 2 and booked["todu_30ever_h6"].sum() == 10.0


def test_hri_training_requires_only_features():
    from src.model.inference_optimized import _prepare_pipeline_data

    indicators = [c for c in _f2_frame().columns if c not in ("x", "y", "status_name")]
    booked, _, _ = _prepare_pipeline_data(_f2_frame(), (), ["x", "y"], indicators, "hri_h6", 1.0)
    assert len(booked) == 3


# ---------------------------------------------------------------- F6 follow-up: surfaced in reports


def _summary_csv(tmp_path, requested, applied, feasible):
    path = tmp_path / "risk_production_summary_table_base.csv"
    pd.DataFrame(
        {
            "Metric": ["Actual", "Optimum selected"],
            "Feasible": [True, True],
            "selection_basis_requested": [None, requested],
            "selection_basis_applied": [None, applied],
            "ci_upper_feasible": [None, feasible],
        }
    ).to_csv(path, index=False)
    return path


def test_segment_banner_flags_unmet_ci_margin(tmp_path):
    from src.report.reporting import _feasibility_banner, _selection_trail_hidden

    unmet = _summary_csv(tmp_path, "ci_upper", "point", False)
    assert "CI-upper selection NOT applied" in _feasibility_banner(unmet, "base")
    assert _selection_trail_hidden(unmet) == []  # columns shown when ci_upper was requested

    point = _summary_csv(tmp_path, "point", "point", None)
    assert _feasibility_banner(point, "base") == ""
    assert _selection_trail_hidden(point)  # hidden for ordinary point-basis runs


def test_consolidated_lists_segments_with_unmet_ci_margin(tmp_path):
    from src.report.consolidation import _ci_margin_unmet_segments

    for seg, feasible in (("seg_unmet", False), ("seg_ok", True)):
        data = tmp_path / seg / "data"
        data.mkdir(parents=True)
        _summary_csv(data, "ci_upper", "point" if not feasible else "ci_upper", feasible)
    assert _ci_margin_unmet_segments(tmp_path, {"seg_unmet": {}, "seg_ok": {}}, "_base") == ["seg_unmet"]
