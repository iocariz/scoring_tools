"""Config-sensitivity harness (M3 hardening milestone).

One-at-a-time (OAT) sensitivity sweep over pipeline configuration knobs. For each
knob it perturbs the base ``config.toml`` across a small set of plausible values,
runs the single-segment pipeline, and records how much the *headline outputs*
move — total accepted production, portfolio risk, and the chosen acceptance mask.
The result is a ranked table telling you which knobs actually drive the answer
(govern these) versus which are effectively noise (safe to default-and-forget).

This is a *scaffold*: the knob registry (``KNOBS``) is intentionally a starting
subset of the most plausible drivers. Start here, read the ranking, then add or
narrow knobs. It deliberately does not try to sweep all ~40 fields at once.

Usage
-----
    uv run python config_sensitivity.py --list                 # show knob registry
    uv run python config_sensitivity.py --dry-run              # show planned runs
    uv run python config_sensitivity.py                        # full sweep (trains per run)
    uv run python config_sensitivity.py --reuse-model          # train once, reuse (faster)
    uv run python config_sensitivity.py --knobs multiplier,ri_calibration_gamma
    uv run python config_sensitivity.py -c config.toml -o sensitivity_report.csv

Notes
-----
* Needs a config that runs the pipeline end-to-end (data + a segment that yields
  rows). The default ``config.toml`` may stop in preprocessing if its
  ``segment_filter`` matches no records; failures are reported per run, not
  silently dropped.
* ``--reuse-model`` trains a model once on the base config and reuses it for
  post-training knobs (RI, MR, optimization). Training-affecting knobs
  (``affects_training=True``) are skipped in that mode — sweep them with a
  separate full run.
* Runs force ``base_scenario_only`` so each run produces one comparable scenario.
"""

from __future__ import annotations

import argparse
import copy
import glob
import shutil
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import tomli_w
from loguru import logger

from src.config import OutputPaths


@dataclass
class Knob:
    """A single sweepable configuration field under ``[preprocessing]``."""

    name: str  # registry name (CLI selector) — usually equals the config key
    key: str  # field name inside the [preprocessing] table
    values: list  # alternative values to test (baseline value supplies an extra point)
    affects_training: bool = False  # True → masked by --reuse-model, skipped there
    note: str = ""


# Starting registry: the most plausible drivers. Extend/trim after the first run.
KNOBS: list[Knob] = [
    # --- risk scaling ---
    Knob("multiplier", "multiplier", [6, 8], note="H6 annualization factor"),
    Knob("multiplier_h3", "multiplier_h3", [3, 5], note="H3 annualization factor"),
    # --- reject inference ---
    Knob("reject_uplift_factor", "reject_uplift_factor", [1.0, 2.5]),
    Knob("reject_max_risk_multiplier", "reject_max_risk_multiplier", [2.0, 5.0]),
    Knob("reject_parceling_method", "reject_parceling_method", ["power", "sigmoid"]),
    Knob("ri_calibration_gamma", "ri_calibration_gamma", [0.5, 1.0]),
    Knob("reject_bayesian_prior_strength", "reject_bayesian_prior_strength", [5.0, 50.0]),
    Knob("reject_decay_half_life", "reject_acceptance_decay_half_life_months", [3.0, 12.0]),
    Knob("run_ri_optimizer", "run_ri_optimizer", [False], note="toggle the RI optimizer off"),
    # --- MR / extrapolation ---
    Knob("mr_min_obs_per_bin", "mr_min_obs_per_bin", [5, 30]),
    Knob("mr_maturity_months", "mr_maturity_months", [3, 9]),
    Knob("mr_extrapolation_method", "mr_extrapolation_method", ["linear", "power"]),
    Knob("mr_extrapolation_risk_multiplier", "mr_extrapolation_risk_multiplier", [2.0, 5.0]),
    Knob("mr_extrapolation_hard_cap", "mr_extrapolation_hard_cap", [10.0, 20.0]),
    Knob("use_mr_outcomes", "use_mr_outcomes", [False], note="toggle MR outcomes off"),
    # --- stress / transformation ---
    Knob("stress_mode", "stress_mode", ["global", "per_bin"]),
    Knob("per_bin_tasa_fin", "per_bin_tasa_fin", [False]),
    # --- monotonicity / optimization ---
    Knob("monotonicity_relaxation", "monotonicity_relaxation_enabled", [False]),
    Knob("mono_z_threshold", "monotonicity_uncertainty_z_threshold", [0.5, 2.0]),
    Knob("pareto_n_points", "pareto_n_points", [50, 200], note="frontier resolution"),
    # --- training-affecting (masked by --reuse-model) ---
    Knob("cv_folds", "cv_folds", [3, 5], affects_training=True),
]


@dataclass
class RunResult:
    label: str
    knob: str
    value: object
    ok: bool
    production: float = float("nan")
    risk: float = float("nan")
    n_accepted: int = -1
    cutoff_diff: int = -1  # accepted-cell symmetric difference vs baseline
    error: str = ""
    fingerprint: frozenset = field(default_factory=frozenset, repr=False)


def load_base_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def write_config(cfg: dict, dest: Path) -> None:
    with open(dest, "wb") as f:
        tomli_w.dump(cfg, f)


def _accepted_fingerprint(accepted_cells_csv: str) -> frozenset:
    """Frozenset of accepted-cell coordinate tuples (the chosen-cutoff fingerprint)."""
    p = Path(accepted_cells_csv)
    if not p.exists():
        return frozenset()
    df = pd.read_csv(p)
    return frozenset(tuple(round(float(v), 6) for v in row) for row in df.itertuples(index=False))


def extract_metrics(out: OutputPaths) -> tuple[float, float, frozenset]:
    """Read headline metrics for the base scenario from a run's output dir."""
    opt_path = Path(out.optimal_solution_csv("_base"))
    if not opt_path.exists():
        raise FileNotFoundError(f"missing {opt_path} (run did not reach scenario analysis)")
    opt = pd.read_csv(opt_path)
    if opt.empty:
        raise ValueError(f"empty optimal solution: {opt_path}")
    row0 = opt.iloc[0]
    production = float(row0.get("oa_amt_h0", float("nan")))
    risk = float(row0.get("b2_ever_h6", float("nan")))
    fingerprint = _accepted_fingerprint(out.accepted_cells_csv("_base"))
    return production, risk, fingerprint


def run_pipeline(cfg: dict, run_dir: Path, model_path: str | None, preloaded=None) -> tuple[float, float, frozenset]:
    """Write the config, run the pipeline into run_dir, and extract headline metrics."""
    from main import main as run_main

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.toml"
    write_config(cfg, cfg_path)
    out = OutputPaths(base_dir=run_dir)
    run_main(
        config_path=str(cfg_path),
        output=out,
        model_path=model_path,
        base_scenario_only=True,
        skip_dq_checks=True,
        preloaded_data=preloaded.copy() if preloaded is not None else None,
    )
    return extract_metrics(out)


def preload_data(data_path: str):
    """Load + standardize the SAS file once (run_batch's pattern) for reuse across runs."""
    from run_batch import load_and_standardize_data

    data = load_and_standardize_data(data_path)
    if data is None:
        raise RuntimeError(f"Failed to load/standardize data from {data_path}")
    logger.info(f"Preloaded data: {data.shape[0]:,} rows x {data.shape[1]} cols")
    return data


def train_shared_model(cfg: dict, workdir: Path, preloaded=None) -> str | None:
    """Train once on the base config; return the saved model dir for reuse."""
    from main import main as run_main

    train_dir = workdir / "_shared_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = train_dir / "config.toml"
    write_config(cfg, cfg_path)
    run_main(
        config_path=str(cfg_path),
        output=OutputPaths(base_dir=train_dir),
        training_only=True,
        preloaded_data=preloaded.copy() if preloaded is not None else None,
    )
    candidates = sorted(glob.glob(str(train_dir / "models" / "model_*")))
    if not candidates:
        logger.warning("train_shared_model: no model_* dir produced; falling back to per-run training")
        return None
    logger.info(f"Reusing shared model: {candidates[-1]}")
    return candidates[-1]


def sweep(base_cfg: dict, knobs: list[Knob], workdir: Path, model_path: str | None, preloaded=None) -> list[RunResult]:
    results: list[RunResult] = []

    # Baseline run (unperturbed) — supplies each knob's base-value datapoint.
    logger.info("Running baseline ...")
    try:
        prod, risk, fp = run_pipeline(copy.deepcopy(base_cfg), workdir / "baseline", model_path, preloaded)
        baseline = RunResult("baseline", "(baseline)", None, True, prod, risk, len(fp), 0, fingerprint=fp)
    except Exception as e:
        logger.error(f"Baseline run failed: {e}")
        return [RunResult("baseline", "(baseline)", None, False, error=str(e))]
    results.append(baseline)
    logger.info(f"Baseline: production={prod:,.0f}  risk={risk:.4f}  accepted_cells={len(fp)}")

    for knob in knobs:
        if model_path is not None and knob.affects_training:
            logger.info(f"Skipping '{knob.name}' (affects training; masked by --reuse-model)")
            continue
        for value in knob.values:
            label = f"{knob.name}={value}"
            run_dir = workdir / label.replace("/", "_").replace(" ", "")
            logger.info(f"Running {label} ...")
            cfg = copy.deepcopy(base_cfg)
            cfg.setdefault("preprocessing", {})[knob.key] = value
            try:
                prod, risk, fp = run_pipeline(cfg, run_dir, model_path, preloaded)
                diff = len(fp.symmetric_difference(baseline.fingerprint))
                results.append(RunResult(label, knob.name, value, True, prod, risk, len(fp), diff, fingerprint=fp))
            except Exception as e:
                logger.error(f"  {label} failed: {e}")
                results.append(RunResult(label, knob.name, value, False, error=str(e)[:200]))
    return results


def summarize(results: list[RunResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detailed long-form df, ranked-by-knob summary df)."""
    baseline = next((r for r in results if r.knob == "(baseline)" and r.ok), None)
    detail = pd.DataFrame(
        [
            {
                "label": r.label,
                "knob": r.knob,
                "value": r.value,
                "ok": r.ok,
                "production": r.production,
                "risk": r.risk,
                "n_accepted": r.n_accepted,
                "cutoff_diff_vs_base": r.cutoff_diff,
                "error": r.error,
            }
            for r in results
        ]
    )

    if baseline is None:
        return detail, pd.DataFrame()

    base_prod, base_risk = baseline.production, baseline.risk
    rows = []
    for knob_name in [k for k in detail["knob"].unique() if k != "(baseline)"]:
        runs = [r for r in results if r.knob == knob_name and r.ok]
        prods = [r.production for r in runs] + [base_prod]
        risks = [r.risk for r in runs] + [base_risk]
        max_cutoff_diff = max([r.cutoff_diff for r in runs], default=0)
        n_failed = sum(1 for r in results if r.knob == knob_name and not r.ok)
        prod_spread_pct = (max(prods) - min(prods)) / base_prod * 100 if base_prod else float("nan")
        risk_spread_pp = max(risks) - min(risks)  # b2_ever_h6 is already a percentage
        risk_spread_pct = risk_spread_pp / base_risk * 100 if base_risk else float("nan")
        rows.append(
            {
                "knob": knob_name,
                "production_spread_pct": round(prod_spread_pct, 3),
                "risk_spread_pp": round(risk_spread_pp, 4),
                "risk_spread_pct": round(risk_spread_pct, 3),
                "max_cutoff_cells_changed": max_cutoff_diff,
                "n_values_tested": len(runs),
                "n_failed": n_failed,
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        # Composite rank: how much this knob moves the headline answer.
        summary["impact_score"] = summary[["production_spread_pct", "risk_spread_pct"]].abs().max(axis=1)
        summary = summary.sort_values("impact_score", ascending=False).reset_index(drop=True)
    return detail, summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-sensitivity sweep (M3).")
    p.add_argument("-c", "--config", default="config.toml", help="Base config TOML")
    p.add_argument("-o", "--output", default="config_sensitivity_report.csv", help="Ranked summary CSV")
    p.add_argument("--knobs", default="all", help="Comma-separated knob names, or 'all'")
    p.add_argument("--workdir", default="output/_sensitivity", help="Scratch dir for per-run outputs")
    p.add_argument("--reuse-model", action="store_true", help="Train once and reuse for non-training knobs")
    p.add_argument(
        "--preload", action="store_true", help="Load+standardize the SAS file once and reuse (avoids per-run reload)"
    )
    p.add_argument("--keep-runs", action="store_true", help="Do not delete per-run output dirs afterwards")
    p.add_argument("--list", action="store_true", help="List the knob registry and exit")
    p.add_argument("--dry-run", action="store_true", help="Show planned runs without executing")
    return p.parse_args(argv)


def select_knobs(spec: str) -> list[Knob]:
    if spec == "all":
        return KNOBS
    wanted = {s.strip() for s in spec.split(",") if s.strip()}
    chosen = [k for k in KNOBS if k.name in wanted]
    unknown = wanted - {k.name for k in chosen}
    if unknown:
        raise SystemExit(f"Unknown knob(s): {sorted(unknown)}. Use --list to see available knobs.")
    return chosen


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.list:
        print(f"{'knob':<34}{'config key':<42}{'values':<22}train?")
        for k in KNOBS:
            print(f"{k.name:<34}{k.key:<42}{str(k.values):<22}{'Y' if k.affects_training else ''}  {k.note}")
        return 0

    knobs = select_knobs(args.knobs)
    n_runs = 1 + sum(len(k.values) for k in knobs if not (args.reuse_model and k.affects_training))
    if args.dry_run:
        print(f"Base config: {args.config}")
        print(f"Planned runs: {n_runs} (1 baseline + perturbations); reuse_model={args.reuse_model}")
        for k in knobs:
            masked = args.reuse_model and k.affects_training
            print(f"  {k.name}: {'SKIP (training)' if masked else k.values}")
        return 0

    base_cfg = load_base_config(args.config)
    workdir = Path(args.workdir)
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    data_path = base_cfg.get("preprocessing", {}).get("data_path")
    preloaded = preload_data(data_path) if args.preload and data_path else None

    model_path = train_shared_model(base_cfg, workdir, preloaded) if args.reuse_model else None

    logger.info(f"Sweeping {len(knobs)} knob(s) over ~{n_runs} runs into {workdir}")
    results = sweep(base_cfg, knobs, workdir, model_path, preloaded)
    detail, summary = summarize(results)

    detail_path = Path(args.output).with_suffix("").as_posix() + "_detail.csv"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(args.output, index=False)

    print("\n=== Config sensitivity (ranked by headline impact) ===")
    if summary.empty:
        print("No ranking produced (baseline failed?). See detail CSV.")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 160):
            print(summary.to_string(index=False))
    print(f"\nRanked summary : {args.output}")
    print(f"Per-run detail : {detail_path}")

    if not args.keep_runs:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
