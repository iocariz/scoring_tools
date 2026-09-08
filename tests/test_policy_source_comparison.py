"""F9: compare frozen score mappings, including every optimization axis."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import run_policy_registry as runner
from src.config import BinConfig, PreprocessingSettings
from src.policy_registry import build_policy_entry, register_policy
from src.preprocess_improved import _apply_binning_from_config


@pytest.fixture
def policy_run(tmp_path):
    settings = PreprocessingSettings(
        variables=["a", "b"],
        inference_variables=["a"],
        segment_filter="source_test",
        keep_vars=["status_name", "reject_reason", "mis_date", "old_score", "b_score"],
        indicators=["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={
            "a": BinConfig(source_col="old_score", output_col="a", bin_edges=[-np.inf, 50, np.inf]),
            "b": BinConfig(source_col="b_score", output_col="b", bin_edges=[-np.inf, 50, np.inf]),
        },
    )
    run = tmp_path / "run"
    (run / "data").mkdir(parents=True)
    pd.DataFrame({"a": [1], "b": [1]}).to_csv(run / "data/accepted_cells_base.csv", index=False)
    pd.DataFrame({"b2_ever_h6": [1.0], "oa_amt_h0": [2000.0]}).to_csv(
        run / "data/optimal_solution_base.csv", index=False
    )
    entry = build_policy_entry(run, settings)
    return settings, run, entry, tmp_path / "registry"


def _unexpected_preprocessing(*args, **kwargs):
    pytest.fail("Incompatible mappings must be refused before binning/scoring the cohort")


@pytest.mark.parametrize("axis", ["a", "b"])
def test_changed_source_is_refused_even_outside_inference_variables(policy_run, monkeypatch, axis):
    settings, run, entry, registry = policy_run
    register_policy(entry, registry_dir=registry)
    settings.bins[axis].source_col = "new_score"
    monkeypatch.setattr(runner, "_run_data_transformations", _unexpected_preprocessing)
    result = runner.compare_segment(pd.DataFrame(), settings, run, registry_dir=registry)
    assert not result.sufficient
    assert "score sources changed" in result.message
    assert result.challenger_policy_id == build_policy_entry(run, settings).policy_id
    assert result.challenger_policy_id != entry.policy_id
    assert axis in result.message
    assert result.champion_policy_id == entry.policy_id
    assert not result.champion and not result.challenger


@pytest.mark.parametrize("sources", [{}, {"a": "old_score"}, {"a": "old_score", "b": ""}])
def test_missing_frozen_sources_fail_closed(policy_run, monkeypatch, sources):
    settings, run, entry, registry = policy_run
    register_policy(replace(entry, bin_sources=sources), registry_dir=registry)
    monkeypatch.setattr(runner, "_run_data_transformations", _unexpected_preprocessing)
    result = runner.compare_segment(pd.DataFrame(), settings, run, registry_dir=registry)
    assert not result.sufficient
    assert "missing frozen score sources" in result.message


def test_missing_current_source_fails_closed(policy_run, monkeypatch):
    settings, run, entry, registry = policy_run
    register_policy(entry, registry_dir=registry)
    settings.bins["b"].source_col = ""
    monkeypatch.setattr(runner, "_run_data_transformations", _unexpected_preprocessing)
    result = runner.compare_segment(pd.DataFrame(), settings, run, registry_dir=registry)
    assert not result.sufficient
    assert "missing frozen score sources" in result.message


def test_matching_sources_evaluate_the_frozen_champion(policy_run, monkeypatch):
    settings, run, entry, registry = policy_run
    register_policy(entry, registry_dir=registry)
    data = pd.DataFrame(
        {
            "old_score": [10.0] * 20 + [90.0] * 20 + [10.0],
            "new_score": [90.0] * 20 + [10.0] * 20 + [90.0],
            "b_score": [10.0] * 41,
            "mis_date": pd.to_datetime(["2024-01-01"] * 40 + ["2024-08-01"]),
            "status_name": ["booked"] * 41,
            "oa_amt_h0": [100.0] * 20 + [200.0] * 20 + [100.0],
            "todu_30ever_h6": [1.0] * 20 + [2.0] * 20 + [1.0],
            "todu_amt_pile_h6": [700.0] * 41,
        }
    )
    monkeypatch.setattr(runner, "_run_data_transformations", lambda df, cfg: _apply_binning_from_config(df, cfg.bins))
    result = runner.compare_segment(data, settings, run, registry_dir=registry)
    assert result.sufficient and not result.message
    assert result.champion["risk"] == pytest.approx(1.0)
    assert result.champion["production"] == pytest.approx(2000.0)
    assert result.challenger["risk"] == result.champion["risk"]
    assert result.challenger_policy_id == entry.policy_id
