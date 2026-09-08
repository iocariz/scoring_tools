"""Synthetic reproductions for the 2026-09-08 code audit.

Run from the repository root with uv run python reports/validation/code_audit_2026_09_08_checks.py.
These checks assert CURRENT DEFECTS; invert/update them when implementing fixes.
All temporary artifacts are isolated and removed at exit; no real loan data is read.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression

from src.backtest import _realized_metrics
from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.estimators import HurdleRegressor
from src.inference_optimized import compute_cell_level_ci, compute_pre_reject_inference_data
from src.optimization_utils import CellGrid, evaluate_solution, milp_solve_cutoffs
from src.persistence import validate_reused_model_config
from src.pipeline.inference import run_inference_phase
from src.policy_registry import build_policy_entry, compare_policies, get_champion, register_policy
from src.preprocess_improved import _apply_binning_from_config

logger.remove()
results = {}


def settings(source="old_score", edge=50):
    return PreprocessingSettings(
        variables=["a"],
        inference_variables=["a"],
        segment_filter="audit_segment",
        keep_vars=["status_name", "reject_reason", "mis_date", source],
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={"a": BinConfig(source_col=source, output_col="a", bin_edges=[-np.inf, edge, np.inf])},
        indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
    )


def missing_booked_h6():
    data = pd.DataFrame(
        {
            "a": [0, 0],
            "status_name": ["booked"] * 2,
            "reject_reason": ["", ""],
            "oa_amt": [100.0, 900.0],
            "oa_amt_h0": [100.0, 900.0],
            "todu_30ever_h6": [1.0, np.nan],
            "todu_amt_pile_h6": [100.0, 900.0],
        }
    )
    inds = ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]
    risk_model = {"best_model_info": {"model": object()}, "features": ["a"]}
    with patch("src.models.calculate_risk_values", side_effect=lambda df, *a, **k: df):
        booked, _ = compute_pre_reject_inference_data(
            data,
            data,
            risk_model,
            None,
            1.0,
            indicators=inds,
            variables=["a"],
            annual_coef=1,
        )
    cell = booked.copy()
    for col in inds:
        cell[col] = cell[col + "_boo"]
        cell[col + "_rep"] = 0.0
    grid = CellGrid.from_summary(cell, ["a"])
    mask = milp_solve_cutoffs(grid, 1.0, [], 7.0)
    kpis = evaluate_solution(np.array([1]), grid, inds, 7.0)
    realized = _realized_metrics(data, pd.Series([True, True]), 7)
    results["incomplete_booked"] = {
        "solver_target_pct": 1.0,
        "mask_at_1pct_target": None if mask is None else int(mask.sum()),
        "accept_all_production": kpis["oa_amt_h0"],
        "accept_all_risk_pct": kpis["b2_ever_h6"],
        "complete_outcome_risk_pct": realized["risk"],
        "incomplete_h6_rows": realized["n_incomplete_h6"],
    }
    # FIXED (F1): the joint completeness rule excludes the partial-H6 loan from BOTH
    # risk sums, so the cell's aggregated risk equals the complete-outcome 7.0% and the
    # MILP finds NO feasible acceptance under a 1% cap (previously it accepted the cell
    # at a diluted 0.7%). Production still counts the incomplete loan.
    assert mask is None
    assert np.isclose(kpis["b2_ever_h6"], 7.0) and np.isclose(realized["risk"], 7.0)
    assert np.isclose(kpis["oa_amt_h0"], 1000.0)


def registry_grid_collision(root):
    run = root / "run"
    (run / "data").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(run / "data/accepted_cells_base.csv", index=False)
    pd.DataFrame({"b2_ever_h6": [1.0], "oa_amt_h0": [100.0]}).to_csv(
        run / "data/optimal_solution_base.csv",
        index=False,
    )
    first = build_policy_entry(run, settings(edge=50))
    second = build_policy_entry(run, settings(edge=80))
    register_policy(first, registry_dir=root / "registry")
    reg = register_policy(second, make_champion=True, registry_dir=root / "registry")
    champion = get_champion("audit_segment", registry_dir=root / "registry")
    results["registry_collision"] = {
        "ids_equal": first.policy_id == second.policy_id,
        "policy_count_after_register_and_promote": len(reg["policies"]),
        "requested_edge": second.bin_edges["a"][1],
        "stored_champion_edge": champion.bin_edges["a"][1],
    }
    # FIXED (F4): the policy id fingerprints the grid (axes, sources, cutpoints) with the
    # accepted cells — changed edges give a NEW id, both entries register, and promotion
    # lands on the requested grid (previously: same id, 1 entry, stale champion edge=50).
    assert first.bin_edges != second.bin_edges and first.policy_id != second.policy_id
    assert len(reg["policies"]) == 2
    assert champion.bin_edges["a"][1] == 80


def source_mapping_guard():
    metadata = {
        "multiplier": 7,
        "model_variables": ["a"],
        "bin_edges": {"a": [-np.inf, 50, np.inf]},
        "bin_sources": {"a": "old_score"},
    }
    original, changed = settings(), settings(source="new_score")
    loan = pd.DataFrame({"old_score": [10.0], "new_score": [90.0]})
    old_bin = _apply_binning_from_config(loan, original.bins).a.iloc[0]
    new_bin = _apply_binning_from_config(loan, changed.bins).a.iloc[0]
    # FIXED (F5): the model metadata now records the raw source column per bin variable
    # and validate_reused_model_config refuses a changed source (same output name + same
    # edges over a DIFFERENT raw score maps the same loan to a different bin: 1 -> 2).
    validate_reused_model_config(metadata, original)  # matching source still passes
    try:
        validate_reused_model_config(metadata, changed)
    except ValueError as exc:
        source_error = str(exc)
    else:
        raise AssertionError("Expected the changed bin source to be refused")
    assert "bin source" in source_error
    # Legacy metadata WITHOUT bin_sources keeps the warn-and-proceed posture (same as
    # models predating the bin_edges pin).
    legacy = {k: v for k, v in metadata.items() if k != "bin_sources"}
    validate_reused_model_config(legacy, changed)
    results["source_mapping_guard"] = {
        "changed_source_refused": True,
        "legacy_metadata_warns_and_passes": True,
        "old_bin": int(old_bin),
        "new_bin": int(new_bin),
    }


def model_pairing(root):
    output = OutputPaths(base_dir=root / "model_output")
    old_dir = root / "model_output/models/model_20240101_000000"
    new_dir = root / "model_output/models/model_20240202_000000"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    companion = old_dir.parent / "todu_model.joblib"
    companion.touch()
    new_exposure = LinearRegression(fit_intercept=False).fit(pd.DataFrame({"oa_amt": [100.0, 200.0]}), [200, 400])
    metadata = {"multiplier": 7, "model_variables": ["a"], "bin_edges": {"a": [-np.inf, 50, np.inf]}}
    patches = (
        patch("src.pipeline.inference.load_model_for_prediction", return_value=(object(), metadata, ["a"])),
        patch("src.pipeline.inference.safe_joblib_load", return_value=new_exposure),
    )
    # FIXED (F3a): selecting the OLDER dir with only a root companion now fails loudly —
    # the root copy was overwritten by the newer run, so the pairing is known-mismatched.
    with patches[0], patches[1]:
        try:
            run_inference_phase(pd.DataFrame(), settings(), str(old_dir), output)
        except RuntimeError as exc:
            older_dir_error = str(exc)
        else:
            raise AssertionError("Expected the mismatched exposure pairing to be refused")
    assert "pairing cannot be verified" in older_dir_error
    # FIXED (F3b): the NEWEST dir may use the root companion (that run wrote it).
    with patches[0], patches[1] as loader:
        run_inference_phase(pd.DataFrame(), settings(), str(new_dir), output)
    assert loader.call_args.args[0] == companion
    # FIXED (F3c): an IN-DIR companion is the verified pair and loads even for older dirs.
    (old_dir / "todu_model.joblib").touch()
    with patches[0], patches[1] as loader:
        run_inference_phase(pd.DataFrame(), settings(), str(old_dir), output)
    assert loader.call_args.args[0] == old_dir / "todu_model.joblib"
    results["model_pairing"] = {
        "older_dir_with_root_companion": "refused (RuntimeError)",
        "newest_dir_with_root_companion": "allowed (legacy, warned)",
        "in_dir_companion": "loaded as the verified pair",
    }


def hurdle_ci():
    # Every cell contains real zero and nonzero loan outcomes; every aggregated
    # cell has a positive mean. This is precisely the per-loan hurdle use case.
    n = 600
    frame = pd.DataFrame(
        {
            "a": np.repeat([0, 1, 2], 200),
            "oa_amt_h0": 100.0,
            "todu_amt_pile_h6": 700.0,
            "todu_30ever_h6": np.tile([0.0, 1.0], n // 2),
        }
    )
    frame["_hurdle_r"] = 7 * frame.todu_30ever_h6 / frame.todu_amt_pile_h6
    frame["_hurdle_w"] = frame.todu_amt_pile_h6
    HurdleRegressor().fit(frame[["a"]], frame._hurdle_r, sample_weight=frame._hurdle_w)
    ci_df = compute_cell_level_ci(
        frame,
        ([-np.inf, 0.5, 1.5, np.inf],),
        ["a"],
        ["oa_amt_h0", "todu_amt_pile_h6", "todu_30ever_h6"],
        "b2_ever_h6",
        7,
        0,
        ["a"],
        HurdleRegressor(),
        cv_folds=3,
    )
    # FIXED (F6): the CI folds now use the SAME per-loan hurdle training branch as model
    # selection and the final fit (previously the bin-aggregated refit raised "at least
    # 2 classes" and the non-blocking catch silently dropped the CI).
    assert not ci_df.empty and ci_df["pred_mean"].notna().all()
    results["hurdle_ci"] = {
        "per_loan_fit_succeeds": True,
        "cell_ci_rows": int(len(ci_df)),
        "ci_computed_for_hurdle_winner": True,
    }


def zero_exposure_observability():
    common = pd.DataFrame(
        {
            "a": [0] * 100,
            "oa_amt_h0": [100.0] * 100,
            "todu_30ever_h6": [1.0] * 20 + [0.0] * 80,
            "todu_amt_pile_h6": [700.0] * 100,
        }
    )
    removed = pd.DataFrame(
        {"a": [1] * 100, "oa_amt_h0": [100.0] * 100, "todu_30ever_h6": [100.0] * 100, "todu_amt_pile_h6": [700.0] * 100}
    )
    zero = pd.DataFrame({"a": [2], "oa_amt_h0": [100.0], "todu_30ever_h6": [0.0], "todu_amt_pile_h6": [0.0]})
    booked = pd.concat([common, removed, zero], ignore_index=True)
    demand = pd.concat([booked, pd.DataFrame({"a": [2] * 1000, "oa_amt_h0": [100.0] * 1000})], ignore_index=True)
    comparison = compare_policies({(0.0,), (1.0,)}, {(0.0,), (2.0,)}, booked, ["a"], 7, cohort_demand=demand)
    added_share = float(
        demand.loc[demand.a.eq(2), "oa_amt_h0"].sum() / demand.loc[demand.a.isin([0, 2]), "oa_amt_h0"].sum()
    )
    results["zero_exposure_observability"] = {
        "verdict": comparison.verdict,
        "reported_unobserved_added_cells": comparison.n_added_cells_unobserved,
        "reported_unobservable_share": comparison.unobservable_added_share,
        "added_cell_positive_h6_exposure": 0,
        "added_demand_share": added_share,
    }
    # FIXED (F2): a 0/0 booked row no longer counts as an observed outcome — the added
    # cell is unobservable, its ~91% demand share exceeds the survivorship threshold,
    # and BETTER is blocked (previously verdict=BETTER with unobservable_share=0).
    assert comparison.verdict == "INCONCLUSIVE"
    assert comparison.n_added_cells_unobserved == 1
    assert np.isclose(comparison.unobservable_added_share, added_share)


with tempfile.TemporaryDirectory(prefix="scoring-audit-") as temp:
    missing_booked_h6()
    registry_grid_collision(Path(temp))
    source_mapping_guard()
    model_pairing(Path(temp))
    hurdle_ci()
    zero_exposure_observability()
print(json.dumps(results, indent=2))
