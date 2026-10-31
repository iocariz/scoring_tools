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
