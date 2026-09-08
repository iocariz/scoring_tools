"""Second-pass audit reproductions; run with uv run python <this file> from the repo root.

All F7–F11 assertions verify corrected behavior. Synthetic loans and
temporary artifacts only. No model fitting, real data, or production output changes.
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

from run_policy_registry import compare_segment
from src.backtest import _realized_metrics, apply_policy
from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.inference_optimized import run_optimization_pipeline
from src.optimization_utils import CellGrid, evaluate_solution, milp_solve_cutoffs
from src.persistence import save_model_with_metadata
from src.pipeline.inference import run_inference_phase
from src.pipeline.sensitivity import run_sensitivity_phase
from src.policy_registry import build_policy_entry, register_policy
from src.preprocess_improved import _apply_binning_from_config
from src.reject_inference import apply_parceling_adjustment

logger.remove()
INDICATORS = ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]


def settings(source="old_score", **kwargs):
    return PreprocessingSettings(
        variables=["a"],
        inference_variables=["a"],
        segment_filter="audit_pass2",
        keep_vars=["status_name", "reject_reason", "mis_date", source],
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={"a": BinConfig(source_col=source, output_col="a", bin_edges=[-np.inf, 50, np.inf])},
        indicators=kwargs.pop("indicators", INDICATORS),
        **kwargs,
    )


def model_pair_save(root):
    output = OutputPaths(base_dir=root / "pairing")
    model = LinearRegression()
    metadata = {
        "multiplier": 7,
        "model_variables": ["a"],
        "bin_edges": {"a": [-np.inf, 50, np.inf]},
        "bin_sources": {"a": "old_score"},
    }
    model_dir = Path(save_model_with_metadata(model, ["a"], metadata, output.model_base_path))
    inference = {
        "model_path": str(model_dir),
        "features": ["a"],
        "best_model_info": {"model": model, "name": "LinearRegression"},
    }
    # Only bypass training: exercise the actual save return value, orchestration,
    # joblib write and integrity sidecar generation.
    with (
        patch("src.pipeline.inference.inference_pipeline", return_value=inference),
        patch("src.pipeline.inference.todu_average_inference", return_value=(None, model, None)),
    ):
        run_inference_phase(pd.DataFrame(), settings(), output=output)
    in_dir_exists = (model_dir / "todu_model.joblib").exists()
    root_exists = (model_dir.parent / "todu_model.joblib").exists()
    # A later save makes the first model older, exercising the actual load guard.
    save_model_with_metadata(model, ["a"], metadata, output.model_base_path)
    with (
        patch("src.pipeline.inference.load_model_for_prediction", return_value=(model, metadata, ["a"])),
        patch("src.pipeline.inference.safe_joblib_load", return_value=model) as loader,
    ):
        try:
            run_inference_phase(pd.DataFrame(), settings(), str(model_dir), output)
        except RuntimeError as exc:
            refused = "pairing cannot be verified" in str(exc)
        else:
            refused = False
    assert in_dir_exists and not root_exists and not refused
    assert loader.call_args.args[0] == model_dir / "todu_model.joblib"
    return {
        "save_returns_directory": model_dir.is_dir(),
        "versioned_exposure_file_exists": in_dir_exists,
        "unversioned_exposure_file_exists": root_exists,
        "earlier_new_model_refused_after_second_save": refused,
    }


def unobserved_booked_cell():
    data = pd.DataFrame(
        {
            "a": [1, 2],
            "status_name": ["booked", "booked"],
            "reject_reason": ["", ""],
            "oa_amt": [100.0, 10000.0],
            "oa_amt_h0": [100.0, 10000.0],
            "todu_30ever_h6": [1.0, np.nan],
            "todu_amt_pile_h6": [700.0, np.nan],
        }
    )
    risk_model = {"best_model_info": {"model": object()}, "features": ["a"]}
    # No rejected loans; bypass the unused model prediction on that empty frame.
    with patch("src.models.calculate_risk_values", side_effect=lambda df, *args, **kw: df):
        booked = run_optimization_pipeline(
            data, data, risk_model, None, 1.0, 1.0, indicators=INDICATORS, variables=["a"], annual_coef=1
        )
    grid = CellGrid.from_summary(booked, ["a"])
    mask = milp_solve_cutoffs(grid, 2.0, [], 7.0)
    kpis = evaluate_solution(mask, grid, INDICATORS, 7.0)
    assert mask.tolist() == [1, 0]
    assert kpis["oa_amt_h0"] == 100 and np.isclose(kpis["b2_ever_h6"], 1.0)
    assert booked.oa_amt_h0.sum() == 10100.0  # production accounting retained
    return {
        "target_pct": 2.0,
        "mask": mask.tolist(),
        "observed_cell_flags": grid.observed.tolist(),
        "reported_risk_pct": kpis["b2_ever_h6"],
        "accepted_production": kpis["oa_amt_h0"],
        "production_in_cell_with_no_usable_outcomes": 10000.0,
    }


def policy_source_comparison(root):
    run = root / "registry_run"
    (run / "data").mkdir(parents=True)
    pd.DataFrame({"a": [1]}).to_csv(run / "data/accepted_cells_base.csv", index=False)
    pd.DataFrame({"b2_ever_h6": [1.0], "oa_amt_h0": [100.0]}).to_csv(
        run / "data/optimal_solution_base.csv", index=False
    )
    champion = build_policy_entry(run, settings())
    register_policy(champion, registry_dir=root / "registry")
    changed_settings = settings(source="new_score")
    challenger_entry = build_policy_entry(run, changed_settings)
    data = pd.DataFrame(
        {
            "old_score": [10.0] * 20 + [90.0] * 20 + [10.0],
            "new_score": [90.0] * 20 + [10.0] * 20 + [90.0],
            "mis_date": pd.to_datetime(["2024-01-01"] * 40 + ["2024-08-01"]),
            "status_name": ["booked"] * 41,
            "reject_reason": [""] * 41,
            "oa_amt": [100.0] * 20 + [200.0] * 20 + [100.0],
            "oa_amt_h0": [100.0] * 20 + [200.0] * 20 + [100.0],
            "todu_30ever_h6": [1.0] * 20 + [2.0] * 20 + [1.0],
            "todu_amt_pile_h6": [700.0] * 41,
        }
    )
    # Bypass unrelated segment filtering; use the real source-to-bin mapping.
    # All comparison guards, holdout slicing, metrics and bootstraps are real.
    with patch(
        "run_policy_registry._run_data_transformations",
        side_effect=lambda df, cfg: _apply_binning_from_config(df, cfg.bins),
    ):
        cmp = compare_segment(data, changed_settings, run, registry_dir=root / "registry")
    correct_cohort = _apply_binning_from_config(data.iloc[:40], settings().bins)
    correct = _realized_metrics(correct_cohort, apply_policy(correct_cohort, ["a"], champion.accepted_set()), 7.0)
    assert not cmp.sufficient and "score sources changed" in cmp.message
    assert not cmp.champion and not cmp.challenger
    assert np.isclose(correct["risk"], 1.0) and correct["production"] == 2000.0
    assert cmp.challenger_policy_id == challenger_entry.policy_id
    return {
        "champion_recorded_source": champion.bin_sources,
        "current_source": "new_score",
        "comparison_proceeds": cmp.sufficient,
        "refusal_message": cmp.message,
        "reported_champion_risk_pct": cmp.champion.get("risk"),
        "frozen_champion_risk_pct": correct["risk"],
        "reported_champion_production": cmp.champion.get("production"),
        "frozen_champion_production": correct["production"],
        "comparison_challenger_id": cmp.challenger_policy_id,
        "registry_challenger_id": challenger_entry.policy_id,
    }


def hri_sensitivity(root):
    output = OutputPaths(base_dir=root / "hri")
    output.data_dir.mkdir(parents=True)
    cfg = settings(
        risk_indicator="hri_h6",
        optimum_hri=1.0,
        optimum_risk=2.0,
        run_sensitivity=True,
        sensitivity_levels=[0.0],
        indicators=INDICATORS + ["h_num_h6", "h_den_h6"],
    )
    cells = pd.DataFrame(
        {
            "a": [1, 2],
            "oa_amt": [100.0, 100.0],
            "oa_amt_h0": [100.0, 100.0],
            "todu_30ever_h6": [0.1, 0.1],
            "todu_amt_pile_h6": [700.0, 700.0],
            "h_num_h6": [0.5, 8.0],
            "h_den_h6": [100.0, 100.0],
        }
    )
    grid = CellGrid.from_summary(cells, cfg.variables)
    baseline = milp_solve_cutoffs(
        grid, cfg.selected_target, cfg.inv_vars, 1.0, risk_num_col="h_num_h6", risk_den_col="h_den_h6"
    )
    assert baseline.tolist() == [1, 0]
    pd.DataFrame({"acceptance_mask": [",".join(str(int(v)) for v in baseline)]}).to_csv(
        output.optimal_solution_csv("_base"), index=False
    )
    run_sensitivity_phase(cells, pd.DataFrame(), cfg, output)
    row = pd.read_csv(output.sensitivity_analysis_csv("_base")).iloc[0]
    assert row["perturbation_pct"] == 0 and row["n_flipped"] == 0
    assert row["new_production"] == 100 and np.isclose(row["new_risk"], 0.5)
    assert row["risk_indicator"] == "hri_h6"
    return {
        "selected_target": cfg.risk_indicator,
        "hri_target_pct": cfg.selected_target,
        "frozen_hri_mask": baseline.tolist(),
        "zero_perturbation_result": row.to_dict(),
        "actual_hri_of_sensitivity_policy_pct": 0.5,
    }


def isotonic_projection():
    # (0,0) is safer than both (1,2) and (2,1); those two are incomparable.
    # Unique coordinates also ensure axis-wise warm-starts cannot alter this case.
    rates = pd.DataFrame(
        {
            "a": [0, 1, 2],
            "b": [0, 2, 1],
            "acceptance_rate": [0.01, 0.25, 0.99],
            "n_booked": [10, 250, 990],
            "n_score_rejected": [990, 750, 10],
        }
    )
    rep = rates[["a", "b"]].copy()
    rep["todu_30ever_h6"] = 1.0
    kwargs = {"reject_uplift_factor": 2.0, "max_risk_multiplier": 3.0, "quiet": True}
    raw = apply_parceling_adjustment(rep, rates, ["a", "b"], **kwargs)
    adjusted = apply_parceling_adjustment(rep, rates, ["a", "b"], enforce_monotonicity=True, **kwargs)
    permuted = apply_parceling_adjustment(
        rep.iloc[[0, 2, 1]], rates, ["a", "b"], enforce_monotonicity=True, **kwargs
    ).sort_values("a")
    y = raw.reject_risk_multiplier.to_numpy()
    fitted = adjusted.reject_risk_multiplier.to_numpy()
    other_order = permuted.reject_risk_multiplier.to_numpy()
    better = np.array([2.0, 2.5, 2.0])
    # Both candidates satisfy EVERY comparable pair, with identical uniform evidence.
    # Pooling just root + lowest child gives less squared error than pooling all three.
    assert fitted[0] <= fitted[1] + 1e-9 and fitted[0] <= fitted[2] + 1e-9
    assert better[0] <= better[1] and better[0] <= better[2]
    current_sse, better_sse = float(np.square(fitted - y).sum()), float(np.square(better - y).sum())
    assert np.isclose(current_sse, better_sse)
    assert np.allclose(fitted, other_order)
    assert np.allclose(other_order, better)
    return {
        "cell_coordinates": rep[["a", "b"]].values.tolist(),
        "raw_multipliers": y.tolist(),
        "fitted_multipliers": fitted.tolist(),
        "feasible_lower_error_multipliers": better.tolist(),
        "same_rows_reordered_multipliers": other_order.tolist(),
        "current_squared_error": current_sse,
        "feasible_lower_squared_error": better_sse,
        "adjusted_default_amounts": adjusted.todu_30ever_h6.tolist(),
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="scoring-audit-pass2-") as tmp:
        root = Path(tmp)
        results = {
            "F7_model_pair_save": model_pair_save(root),
            "F8_unobserved_booked_cell": unobserved_booked_cell(),
            "F9_policy_source_comparison": policy_source_comparison(root),
            "F10_hri_sensitivity": hri_sensitivity(root),
            "F11_isotonic_projection": isotonic_projection(),
        }
    print(json.dumps(results, indent=2))
