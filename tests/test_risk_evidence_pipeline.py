"""F8 regression through aggregation, fixed cutoffs and orchestration fallbacks."""

import numpy as np
import pandas as pd
import pytest

from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.inference_optimized import run_optimization_pipeline
from src.optimization_utils import CellGrid, evaluate_solution, milp_solve_cutoffs
from src.pipeline import cutoff_optimization as phase

INDICATORS = ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]


@pytest.fixture
def booked():
    return pd.DataFrame(
        {
            "a": [1, 2],
            "b": [1, 1],
            "status_name": ["booked", "booked"],
            "reject_reason": ["", ""],
            "oa_amt": [100.0, 10000.0],
            "oa_amt_h0": [100.0, 10000.0],
            "todu_30ever_h6": [1.0, np.nan],
            "todu_amt_pile_h6": [700.0, np.nan],
        }
    )


@pytest.mark.parametrize("missing_den", [np.nan, 900.0])
def test_all_missing_or_jointly_excluded_booked_cell_is_not_accepted(booked, monkeypatch, missing_den):
    booked.loc[1, "todu_amt_pile_h6"] = missing_den
    monkeypatch.setattr("src.models.calculate_risk_values", lambda df, *args, **kwargs: df)
    summary = run_optimization_pipeline(
        booked,
        booked,
        {"best_model_info": {"model": None}, "features": ["a"]},
        None,
        1.0,
        1.0,
        indicators=INDICATORS,
        variables=["a"],
        annual_coef=1.0,
    )
    grid = CellGrid.from_summary(summary, ["a"])
    mask = milp_solve_cutoffs(grid, 2.0, [], 7.0)
    kpis = evaluate_solution(mask, grid, INDICATORS, 7.0)
    assert mask.tolist() == [1, 0]
    assert kpis["oa_amt_h0"] == 100.0
    assert kpis["b2_ever_h6"] == pytest.approx(1.0)
    assert summary.oa_amt_h0.sum() == 10100.0


def _run_phase(booked, monkeypatch, tmp_path, variables, **overrides):
    settings = PreprocessingSettings(
        variables=variables,
        inference_variables=variables,
        indicators=INDICATORS,
        keep_vars=["status_name", "reject_reason"],
        segment_filter="evidence",
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={v: BinConfig(source_col=f"{v}_score", output_col=v, bin_edges=[0, 50, 100]) for v in variables},
        **overrides,
    )
    output = OutputPaths(base_dir=tmp_path)
    output.ensure_dirs()
    # Isolate phase routing; aggregation and joint masking are exercised above.
    summary = booked[variables + INDICATORS].copy().fillna(0.0)
    for col in INDICATORS:
        summary[f"{col}_boo"] = summary[col]
        summary[f"{col}_rep"] = 0.0
    monkeypatch.setattr(phase, "run_optimization_pipeline", lambda **kwargs: summary.copy())
    return phase.run_optimization_phase(booked, booked, {}, None, 1.0, 1.0, settings, 1.0, output)


@pytest.mark.parametrize("variables", [["a"], ["a", "b"]])
def test_fixed_cutoffs_refuse_accepted_cells_with_unknown_risk(booked, monkeypatch, tmp_path, variables):
    cutoffs = {v: [1, 2] if v == "a" else [1, 1] for v in variables}
    with pytest.raises(ValueError, match="Fixed cutoffs accept.*without usable risk evidence"):
        _run_phase(booked, monkeypatch, tmp_path, variables, fixed_cutoffs=cutoffs)


def test_fixed_cutoffs_can_reject_the_unmeasured_cell(booked, monkeypatch, tmp_path):
    result = _run_phase(booked, monkeypatch, tmp_path, ["a"], fixed_cutoffs={"a": [1]})
    assert result.pareto_masks[0].tolist() == [1, 0]
    assert result.data_summary.iloc[0].oa_amt_h0 == 100.0


def test_baseline_keeps_actual_booked_production(booked, monkeypatch, tmp_path):
    result = _run_phase(booked, monkeypatch, tmp_path, ["a"], baseline_mode=True)
    assert result.data_summary.iloc[0].oa_amt_h0 == 10100.0


def test_empty_frontier_cannot_reenter_unrestricted_legacy_enumeration(booked, monkeypatch, tmp_path):
    monkeypatch.setattr(
        phase,
        "trace_pareto_frontier",
        lambda data_summary_desagregado, variables, **kwargs: (
            pd.DataFrame(),
            CellGrid.from_summary(data_summary_desagregado, variables),
            [],
        ),
    )

    def unsafe_legacy(*args, **kwargs):
        pytest.fail("Legacy enumeration must not bypass risk-evidence exclusions")

    monkeypatch.setattr(phase, "get_fact_sol", unsafe_legacy)
    monkeypatch.setattr("src.optimization_utils._ga_pareto_fallback", lambda grid, *a, **kw: (pd.DataFrame(), grid, []))
    with pytest.raises(RuntimeError, match="Both MILP and GA produced no solutions"):
        _run_phase(booked, monkeypatch, tmp_path, ["a", "b"])
