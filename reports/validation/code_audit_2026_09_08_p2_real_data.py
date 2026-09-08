"""Read-only F11 impact check: frozen model predictions, fresh SAS acceptance rates.

Rebuilds pre-projection RI multipliers, verifies the old projection against the
saved run, and substitutes only the corrected projection into the saved surface.
Compares frozen-policy risk and same-target MILP decisions (persisted predecessor
floors held fixed). This is not a full model/frontier/MR/batch reproduction.
Run from the repository root; prints only aggregate diagnostics, never loan rows.
"""

import ast
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))

import numpy as np
import pandas as pd
from loguru import logger

import src.reject_inference as ri
from src.config import PreprocessingSettings
from src.data_manager import load_data, standardize_columns_and_values
from src.lineage import _sha256
from src.optimization_utils import CellGrid, milp_solve_cutoffs
from src.preprocess_improved import (
    _filter_booked_for_period,
    _filter_demand_for_period,
    _infer_monotonicity,
    _run_data_transformations,
)

BEFORE_COMMIT = "c4fc4f9"


def input_fingerprints(segments):
    paths = set()
    for segment in segments:
        if "skipped" in segment:
            continue
        root = Path("output") / segment["segment"]
        config = root / "config_segment.toml"
        cfg = PreprocessingSettings.from_toml(str(config))
        paths.update([config, root / "data/data_summary_desagregado.csv", root / "data/accepted_cells_base.csv"])
        paths.add(Path(cfg.data_path))
        if cfg.cutoff_floor_segment:
            paths.add(Path("output") / cfg.cutoff_floor_segment / "data/accepted_cells_base.csv")
    return {str(path): _sha256(path) for path in sorted(paths)}


def old_projection():
    source = subprocess.check_output(["git", "show", f"{BEFORE_COMMIT}:src/reject_inference.py"], text=True)
    names = {"_slice_weights", "_enforce_multiplier_monotonicity", "_fix_partial_order_violations"}
    selected = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"np": np, "pd": pd, "logger": logger}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<before-projection>", "exec"), namespace)
    return namespace["_enforce_multiplier_monotonicity"]


def headline(grid, mask, multiplier):
    selected = grid.cell_data.loc[mask.astype(bool)]
    denominator = selected.todu_amt_pile_h6.sum()
    return {
        "production": float(selected.oa_amt_h0.sum()),
        "risk_pct": float(100 * multiplier * selected.todu_30ever_h6.sum() / denominator),
        "accepted_cells": int(mask.sum()),
    }


def run():
    logger.remove()
    before = old_projection()
    loaded = {}
    results = []
    for path in sorted(Path("output").glob("*/data/data_summary_desagregado.csv")):
        run_dir = path.parent.parent
        cfg = PreprocessingSettings.from_toml(str(run_dir / "config_segment.toml"))
        if cfg.baseline_mode or cfg.reject_inference_method != "parceling" or not cfg.reject_enforce_monotonicity:
            results.append({"segment": run_dir.name, "skipped": "baseline or RI monotonicity disabled"})
            continue
        if cfg.data_path not in loaded:
            loaded[cfg.data_path] = standardize_columns_and_values(load_data(cfg.data_path, cfg.sas_encoding))
        clean = _run_data_transformations(loaded[cfg.data_path].copy(), cfg)
        booked = _filter_booked_for_period(clean, cfg.date_ini_book_obs, cfg.date_fin_book_obs)
        _infer_monotonicity(booked, cfg)
        demand = _filter_demand_for_period(clean, cfg.date_ini_book_obs, cfg.date_fin_book_obs)
        summary = pd.read_csv(path)
        rep = summary.loc[summary.ri_multiplier_rep.notna(), cfg.variables].assign(todu_30ever_h6=1.0)
        rates = ri.compute_acceptance_rates(
            demand,
            cfg.variables,
            bayesian_smoothing=cfg.reject_bayesian_smoothing,
            bayesian_prior_strength=cfg.reject_bayesian_prior_strength,
            include_all_rejections=cfg.reject_include_all_rejections,
            recent_months=cfg.reject_acceptance_recent_months,
            decay_half_life_months=cfg.reject_acceptance_decay_half_life_months,
            date_col=cfg.reject_acceptance_date_col,
        )
        kwargs = {
            "reject_uplift_factor": cfg.reject_uplift_factor,
            "max_risk_multiplier": cfg.reject_max_risk_multiplier,
            "method": cfg.reject_parceling_method,
            "inv_vars": cfg.inv_vars,
            "no_demand_anchor_percentile": cfg.reject_no_demand_anchor_percentile,
            "confidence_scale": cfg.reject_confidence_scale,
            "quiet": True,
        }
        raw = ri.apply_parceling_adjustment(rep, rates, cfg.variables, **kwargs)
        with patch.object(ri, "_enforce_multiplier_monotonicity", before):
            old = ri.apply_parceling_adjustment(rep, rates, cfg.variables, enforce_monotonicity=True, **kwargs)
        t0 = time.perf_counter()
        new = ri.apply_parceling_adjustment(rep, rates, cfg.variables, enforce_monotonicity=True, **kwargs)
        elapsed = time.perf_counter() - t0
        order = cfg.variables
        saved = summary.loc[summary.ri_multiplier_rep.notna()].sort_values(order).ri_multiplier_rep.to_numpy()
        old_values = old.sort_values(order).reject_risk_multiplier.to_numpy()
        new_values = new.sort_values(order).reject_risk_multiplier.to_numpy()
        np.testing.assert_allclose(old_values, saved, atol=1e-7, rtol=1e-7)
        permuted = ri.apply_parceling_adjustment(
            rep.sample(frac=1, random_state=42), rates, cfg.variables, enforce_monotonicity=True, **kwargs
        )
        np.testing.assert_allclose(permuted.sort_values(order).reject_risk_multiplier, new_values, atol=1e-8)
        evidence = rates[order].copy()
        evidence["weight"] = (rates.n_booked + rates.n_score_rejected).clip(lower=1)
        weights = raw[order].merge(evidence, on=order, how="left").weight.fillna(1).to_numpy()
        y = raw.reject_risk_multiplier.to_numpy()
        old_sse = float(np.dot(weights, (old.reject_risk_multiplier.to_numpy() - y) ** 2))
        new_sse = float(np.dot(weights, (new.reject_risk_multiplier.to_numpy() - y) ** 2))
        assert new_sse <= old_sse + 1e-7
        changed = summary.merge(new[order + ["reject_risk_multiplier"]], on=order, how="left", validate="one_to_one")
        ratio = (changed.reject_risk_multiplier / changed.ri_multiplier_rep).fillna(1)
        changed["todu_30ever_h6_rep"] *= ratio
        changed["todu_30ever_h6"] = changed.todu_30ever_h6_boo + changed.todu_30ever_h6_rep
        old_grid = CellGrid.from_summary(summary, order)
        new_grid = CellGrid.from_summary(changed, order)
        accepted = pd.read_csv(path.parent / "accepted_cells_base.csv")
        accepted_set = set(map(tuple, accepted[order].to_numpy()))
        frozen_mask = np.array([int(tuple(row) in accepted_set) for row in old_grid.cell_data[order].to_numpy()])
        record = {
            "segment": run_dir.name,
            "rep_cells": len(rep),
            "demand_rows": len(demand),
            "old_projection_matches_saved": True,
            "permutation_invariant": True,
            "new_projection_seconds": elapsed,
            "old_weighted_sse": old_sse,
            "new_weighted_sse": new_sse,
            "changed_multipliers": int(np.count_nonzero(np.abs(new_values - old_values) > 1e-6)),
            "max_multiplier_change": float(np.max(np.abs(new_values - old_values))),
            "frozen_before": headline(old_grid, frozen_mask, cfg.multiplier),
            "frozen_after": headline(new_grid, frozen_mask, cfg.multiplier),
        }
        if not cfg.fixed_cutoffs:
            fixed = {}
            if cfg.cutoff_floor_segment:
                predecessor = pd.read_csv(Path("output") / cfg.cutoff_floor_segment / "data/accepted_cells_base.csv")
                predecessor_set = set(map(tuple, predecessor[order].to_numpy()))
                with (run_dir / "config_segment.toml").open("rb") as config_file:
                    assert tomllib.load(config_file)["preprocessing"].get("cutoff_ordering_mode") == "bottom_up"
                fixed = {idx: 1 for cell, idx in old_grid.cell_index.items() if cell in predecessor_set}
            solve_kwargs = {
                "fixed_cells": fixed,
                "max_swapin_production_pct": cfg.max_swapin_production_pct,
                "max_swapin_risk": cfg.max_swapin_risk,
                "time_limit": cfg.milp_time_limit,
                "monotonicity_relaxation_enabled": cfg.monotonicity_relaxation_enabled,
                "monotonicity_uncertainty_min_exposure": cfg.monotonicity_uncertainty_min_exposure,
                "monotonicity_uncertainty_z_threshold": cfg.monotonicity_uncertainty_z_threshold,
            }
            assert not cfg.min_accepted_bin_by_variable
            assert cfg.risk_indicator == "b2_ever_h6"
            old_mask = milp_solve_cutoffs(old_grid, cfg.selected_target, cfg.inv_vars, cfg.multiplier, **solve_kwargs)
            new_mask = milp_solve_cutoffs(new_grid, cfg.selected_target, cfg.inv_vars, cfg.multiplier, **solve_kwargs)
            record.update(
                {
                    "target_pct": cfg.selected_target,
                    "persisted_floor_cells": len(fixed),
                    "same_target_before": None if old_mask is None else headline(old_grid, old_mask, cfg.multiplier),
                    "same_target_after": None if new_mask is None else headline(new_grid, new_mask, cfg.multiplier),
                    "same_target_cells_flipped": (
                        None if old_mask is None or new_mask is None else int(np.count_nonzero(old_mask != new_mask))
                    ),
                }
            )
            if old_mask is None or new_mask is None:
                record["same_target_limit"] = "No feasible solution returned with persisted predecessor floor"
        results.append(record)
        print(json.dumps(record), file=sys.stderr, flush=True)
    return {"before_commit": BEFORE_COMMIT, "basis": __doc__, "segments": results, "inputs": input_fingerprints(results)}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
