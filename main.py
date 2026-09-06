import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from src.config import OutputPaths
from src.data_manager import DataValidationError, load_and_prepare_data
from src.lineage import build_lineage, log_run_banner, write_lineage
from src.pipeline.config_loader import load_and_validate_config
from src.pipeline.inference import run_inference_phase
from src.pipeline.optimization import (
    build_scenario_list,
    compute_mr_annual_coef,
    run_optimization_phase,
    run_ri_optimizer_phase,
    run_scenario_analysis,
    save_cutoff_summaries,
)
from src.pipeline.preprocessing import run_preprocessing_phase


class PipelineExecutionError(RuntimeError):
    """Base exception for pipeline execution failures."""


class ConfigLoadError(PipelineExecutionError):
    """Raised when config load/validation fails."""


class DataLoadError(PipelineExecutionError):
    """Raised when data load/validation fails."""


class InferencePhaseError(PipelineExecutionError):
    """Raised when inference phase fails."""


class ResimulationError(PipelineExecutionError):
    """Raised when resimulation cannot run (missing artifacts / model)."""


def _config_hash(settings) -> str:
    """Compute a short hash of key config fields for staleness detection.

    Includes the resolved per-variable bin_edges (#40): a bin-edge change alone (same variables /
    multiplier / data_path) redefines every grid cell, so the old 3-field hash missed it and a
    resimulation would reuse a Pareto frontier built on a different grid.
    """
    edges = ""
    if getattr(settings, "bins", None):
        edges = ";".join(f"{v}={list(settings.bins[v].bin_edges)}" for v in sorted(settings.bins))
    key_fields = f"{settings.variables}|{settings.multiplier}|{settings.data_path}|{edges}"
    return hashlib.md5(key_fields.encode()).hexdigest()[:12]


def _save_resimulation_artifacts(
    output: OutputPaths,
    data_clean: pd.DataFrame,
    data_booked: pd.DataFrame,
    stress_factor: float,
    tasa_fin: float,
    annual_coef: float,
    settings,
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
) -> None:
    """Save artifacts needed for resimulation mode."""
    data_clean.to_parquet(output.data_clean_parquet, index=False)
    data_booked.to_parquet(output.data_booked_parquet, index=False)
    meta = {
        "stress_factor": stress_factor,
        "tasa_fin": tasa_fin,
        "annual_coef": annual_coef,
        "config_hash": _config_hash(settings),
    }
    Path(output.resimulation_meta_json).write_text(json.dumps(meta, indent=2))
    if per_bin_stress is not None and not per_bin_stress.empty:
        per_bin_stress.to_csv(output.per_bin_stress_csv, index=False)
    if per_bin_tasa_fin is not None and not per_bin_tasa_fin.empty:
        per_bin_tasa_fin.to_csv(output.per_bin_tasa_fin_csv, index=False)
    logger.debug(f"Resimulation artifacts saved to {output.data_dir}")


def _build_resimulation_scenarios(risk_values: list[float]) -> list[tuple[float, str]]:
    """Build scenario list from user-provided risk values."""
    risk_values = sorted(risk_values)
    if len(risk_values) == 1:
        return [(risk_values[0], "base")]
    if len(risk_values) == 3:
        return [
            (risk_values[0], "pessimistic"),
            (risk_values[1], "base"),
            (risk_values[2], "optimistic"),
        ]
    return [(r, f"target_{r:.2f}") for r in risk_values]


def _scenario_artifacts(output: OutputPaths, name: str) -> list[str]:
    """Every per-scenario artifact path for scenario ``name`` (main + MR + audit).

    Used to purge a stale scenario so a run that produces a different set of
    scenarios than the previous one cannot leave orphaned files behind for
    scenario auto-detection or the consolidated report to pick up (#48).
    """
    suffix = f"_{name}"
    data_dir = Path(output.data_dir)
    return [
        output.risk_production_summary_csv(suffix),
        output.data_summary_desagregado_csv(suffix),
        output.optimal_solution_csv(suffix),
        output.efficient_frontier_csv(suffix),
        output.risk_production_visualizer_html(suffix),
        output.acceptance_grid_html(suffix),
        output.accepted_cells_csv(suffix),
        output.sensitivity_analysis_csv(suffix),
        output.cell_marginal_impact_csv(suffix),
        output.ri_optimizer_csv(suffix),
        output.stability_report_html(suffix),
        output.stability_psi_csv(suffix),
        output.drift_alerts_json(suffix),
        # MR variants
        output.mr_summary_csv(suffix),
        output.mr_b2_visualization_html(suffix),
        output.mr_cutoff_drift_html(suffix),
        output.mr_cutoff_summary_wide_csv(suffix),
        output.mr_optimal_solution_csv(suffix),
        output.mr_risk_production_summary_csv(suffix),
        output.mr_risk_comparison_csv(suffix),
        # audit tables (written directly by src/audit.py — no OutputPaths helper)
        str(data_dir / f"audit_{name}.csv"),
        str(data_dir / f"audit_{name}_mr.csv"),
    ]


def _discover_scenario_names(output: OutputPaths) -> set[str]:
    """Scenario names present on disk, from ``risk_production_summary_table_*.csv`` markers.

    Mirrors the scenario auto-detection in ``src.pipeline.reporting._detect_scenarios``
    (skips the ``_mr`` variants) so cleanup and the report agree on what a scenario is.
    """
    data_dir = Path(output.data_dir)
    if not data_dir.exists():
        return set()
    prefix = "risk_production_summary_table_"
    names: set[str] = set()
    for f in data_dir.glob(f"{prefix}*.csv"):
        if "_mr_" in f.name or f.name.endswith("_mr.csv"):
            continue  # MR variant, not a distinct scenario marker
        name = f.stem[len(prefix) :]
        if name:
            names.add(name)
    return names


def cleanup_stale_scenarios(
    output: OutputPaths,
    active_names,
    segment: str = "",
    protect=frozenset(),
) -> list[str]:
    """Remove artifacts for scenarios present on disk but not being (re)generated.

    A run — full or resimulation — that produces a different set of scenarios
    than a previous run would otherwise leave the old scenario files behind,
    and scenario auto-detection / the consolidated report would present a mixed
    family (a fresh base target next to a stale pessimistic/optimistic, or an
    orphaned ``target_X.XX`` from a prior resimulation). ``protect`` names are
    never removed — resimulation protects ``"base"`` because the base
    desagregado grid is its optimization input, not just a report output (#48).

    Returns the list of scenario names purged.
    """
    active = {str(n) for n in active_names} | {str(n) for n in protect}
    stale = sorted(_discover_scenario_names(output) - active)
    for name in stale:
        for path in _scenario_artifacts(output, name):
            p = Path(path)
            if p.exists():
                p.unlink()
                logger.debug(f"[{segment}] Removed stale scenario artifact: {p.name}")
    if stale:
        logger.info(f"[{segment}] Purged stale scenario(s) not regenerated by this run: {', '.join(stale)}")
    return stale


def _resolve_resimulation_model_path(output: OutputPaths, config_path: str, segment: str = "") -> str | None:
    """Find the model directory to reuse for resimulation.

    Order: the segment's OWN model dir, else THIS segment's modelling supersegment
    (`output/_supersegment_<ms>/models/`, resolved from the segment config — #40),
    else, as a loud last resort, any `_supersegment_*` model. The last resort was
    the only prior behaviour, and a bare lexicographic glob over ALL supersegments
    picks the wrong model the moment a second modelling supersegment exists.
    Returns the model-dir path, or ``None`` when nothing is found.
    """
    import tomllib

    from src.utils import resolve_modelling_supersegment

    output_root = output.base_dir.parent
    model_dirs = sorted(Path(output.models_dir).glob("model_*"), reverse=True)
    if model_dirs:
        return str(model_dirs[0])

    try:
        raw = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
        seg_cfg = raw.get("preprocessing", raw)
    except (OSError, ValueError, tomllib.TOMLDecodeError):
        seg_cfg = {}
    modelling_ss = resolve_modelling_supersegment(seg_cfg)

    if modelling_ss:
        ss_dirs = sorted((output_root / f"_supersegment_{modelling_ss}" / "models").glob("model_*"), reverse=True)
        if ss_dirs:
            logger.info(f"[{segment}] Resimulation: using modelling supersegment '{modelling_ss}' model.")
            return str(ss_dirs[0])

    fallback = sorted(output_root.glob("_supersegment_*/models/model_*"), reverse=True)
    if fallback:
        logger.warning(
            f"[{segment}] Resimulation: no model for modelling_supersegment '{modelling_ss or '(none)'}'; "
            f"falling back to {fallback[0]} — verify it is the intended model."
        )
        return str(fallback[0])
    return None


def run_resimulation(
    config_path: str,
    resimulate_risk: list[float],
    output: OutputPaths | None = None,
) -> None:
    """Re-run scenario analysis with different risk targets using saved artifacts.

    Skips data loading, preprocessing, training, and optimization — loads all
    required artifacts from a previous full run.
    """
    t0 = time.perf_counter()

    if output is None:
        # Infer output directory from config path: if it's config_segment.toml
        # inside an output dir, use that dir as base.
        config_dir = Path(config_path).resolve().parent
        if (config_dir / "data").exists() and (config_dir / "models").exists():
            output = OutputPaths(base_dir=config_dir)
        else:
            output = OutputPaths()

    # 1. Load config
    settings, date_ini, date_fin, annual_coef = load_and_validate_config(config_path)
    segment = settings.segment_filter

    # 2. Validate required artifacts
    required_files = {
        "data_clean.parquet": output.data_clean_parquet,
        "data_booked.parquet": output.data_booked_parquet,
        "resimulation_meta.json": output.resimulation_meta_json,
        "pareto_optimal_solutions.csv": output.pareto_solutions_csv,
        "data_summary_desagregado_base.csv": output.data_summary_desagregado_csv("_base"),
    }
    missing = [name for name, path in required_files.items() if not Path(path).exists()]
    if missing:
        # #48: raise (do not return) so callers — run_batch's per-segment
        # try/except and the CLI exit code — see the failure instead of the run
        # silently consolidating the previous run's artifacts as "resimulated".
        raise ResimulationError(
            f"[{segment}] Resimulation requires a previous full run. Missing: {', '.join(missing)}. "
            "Run the full pipeline first."
        )

    # 3. Load saved scalars
    meta = json.loads(Path(output.resimulation_meta_json).read_text())
    stress_factor = meta["stress_factor"]
    tasa_fin = meta["tasa_fin"]
    annual_coef = meta.get("annual_coef", annual_coef)

    # Staleness check
    current_hash = _config_hash(settings)
    if meta.get("config_hash") and meta["config_hash"] != current_hash:
        logger.warning(
            f"[{segment}] Config has changed since the last full run (hash mismatch). "
            "Resimulation uses the saved Pareto frontier — results may not reflect config changes. "
            "Run a full pipeline to regenerate artifacts."
        )

    # 4. Load data artifacts
    logger.info(f"[{segment}] Resimulation: loading artifacts...")
    data_clean = pd.read_parquet(output.data_clean_parquet)
    data_booked = pd.read_parquet(output.data_booked_parquet)
    data_summary = pd.read_csv(output.pareto_solutions_csv)
    data_summary_desagregado = pd.read_csv(output.data_summary_desagregado_csv("_base"))

    per_bin_stress = None
    if Path(output.per_bin_stress_csv).exists():
        per_bin_stress = pd.read_csv(output.per_bin_stress_csv)
    per_bin_tasa_fin = None
    if Path(output.per_bin_tasa_fin_csv).exists():
        per_bin_tasa_fin = pd.read_csv(output.per_bin_tasa_fin_csv)

    # 5. Reconstruct grid + masks
    from src.optimization_utils import CellGrid, decode_mask

    grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)
    pareto_masks = []
    if "acceptance_mask" in data_summary.columns:
        for _, row in data_summary.iterrows():
            mask_str = row.get("acceptance_mask")
            if pd.notna(mask_str):
                pareto_masks.append(decode_mask(str(mask_str)))

    values_per_var = {var: sorted(data_summary_desagregado[var].unique()) for var in settings.variables}

    # 6. Load model (fast — just loads from disk).
    model_path = _resolve_resimulation_model_path(output, config_path, segment)
    if model_path is None:
        # #48: raise (do not return) — see the missing-artifacts branch above.
        raise ResimulationError(f"[{segment}] No model directory found in {output.models_dir} or supersegment dirs")
    risk_inference, reg_todu_amt_pile = run_inference_phase(data_clean, settings, model_path, output=output)

    # 7. Compute total_demand from data_clean
    from src.preprocess_improved import filter_by_date

    data_demand_period = filter_by_date(data_clean, "mis_date", settings.date_ini_book_obs, settings.date_fin_book_obs)
    if "status_name" in data_demand_period.columns and "oa_amt_h0" in data_demand_period.columns:
        total_demand = data_demand_period.loc[data_demand_period["status_name"] != "canceled", "oa_amt_h0"].sum()
    else:
        total_demand = data_demand_period["oa_amt_h0"].sum() if "oa_amt_h0" in data_demand_period.columns else 0.0

    # 8. Build scenarios and run
    scenarios = _build_resimulation_scenarios(resimulate_risk)

    # #48: purge scenarios from the previous run that this resimulation does not
    # regenerate (e.g. --resimulate 1.0 → only "base"; the old pessimistic/
    # optimistic would otherwise be consolidated as a mixed family). "base" is
    # protected: its desagregado grid is the optimization INPUT read above, so a
    # target_X.XX-only resimulation must not delete it.
    cleanup_stale_scenarios(output, {name for _, name in scenarios}, segment, protect={"base"})

    annual_coef_mr = compute_mr_annual_coef(settings)
    data_summary_sample_no_opt = pd.DataFrame(columns=["oa_amt_h0", "b2_ever_h6"])

    logger.info(
        f"[{segment}] Resimulation: {len(scenarios)} scenario(s) — "
        + ", ".join(f"{name}={risk:.2f}%" for risk, name in scenarios)
    )

    cutoff_summaries = []
    for scenario_risk, scenario_name in scenarios:
        summary = run_scenario_analysis(
            scenario_risk,
            scenario_name,
            data_summary=data_summary,
            data_summary_desagregado=data_summary_desagregado,
            data_summary_sample_no_opt=data_summary_sample_no_opt,
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stress_factor=stress_factor,
            tasa_fin=tasa_fin,
            annual_coef_mr=annual_coef_mr,
            values_per_var=values_per_var,
            grid=grid,
            pareto_masks=pareto_masks,
            output=output,
            total_demand=total_demand,
            per_bin_stress=per_bin_stress,
            per_bin_tasa_fin=per_bin_tasa_fin,
        )
        cutoff_summaries.append(summary)

    save_cutoff_summaries(cutoff_summaries, settings, output=output)

    # Generate HTML report
    try:
        from src.pipeline.reporting import generate_segment_report

        report_path = generate_segment_report(
            settings=settings, output=output, scenarios=[name for _, name in scenarios]
        )
        if report_path:
            logger.info(f"[{segment}] Report generated: {report_path}")
    except Exception as e:
        logger.warning(f"[{segment}] Report generation failed (non-blocking): {e}")

    elapsed = time.perf_counter() - t0
    logger.info(f"[{segment}] Resimulation complete | {len(scenarios)} scenarios | {elapsed:.1f}s total")


def main(
    config_path: str = "config.toml",
    model_path: str | None = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    skip_dq_checks: bool = False,
    allow_dq_warnings: bool = False,
    preloaded_data: pd.DataFrame | None = None,
    output: OutputPaths | None = None,
    floor_cells_path: str | None = None,
    floor_cells_mode: str = "floor",
    resimulate_risk: list[float] | None = None,
    run_id: str | None = None,
    run_ts_iso: str | None = None,
):
    """
    Run the full single-segment scoring pipeline end-to-end.

    Orchestrates the phases documented in ``CLAUDE.md``: config load,
    data loading, preprocessing, inference, optimization, scenario
    analysis, sensitivity, RI optimizer, trend analysis, and HTML report.

    Args:
        config_path: Path to the configuration TOML file (default: ``config.toml``).
        model_path: Optional path to a pre-trained model directory. When set,
            the inference phase skips training and loads this artefact; paths
            are subject to the ``safe_joblib_load`` trusted-root allowlist
            (see ``src/persistence.py`` and ``SCORING_TRUSTED_MODEL_ROOTS``).
        training_only: If True, run only the preprocessing + inference phases
            (skip optimization, scenario analysis, sensitivity, RI optimizer,
            and report generation).
        baseline_mode: If True, force ``settings.baseline_mode = True`` — show
            the current booked portfolio as-is with no cutoff optimization
            (Optimum = Actual, zero swap-in/swap-out). MR inference still runs.
            Only the base scenario is generated; sensitivity and RI optimizer
            are skipped. Equivalent to the ``--baseline`` CLI flag.
        base_scenario_only: If True, force ``settings.base_scenario_only =
            True`` — generate only the base scenario, skip pessimistic /
            optimistic scenarios. Config-only flag; no CLI equivalent.
        skip_dq_checks: If True, skip data quality checks.
        allow_dq_warnings: If True, force ``settings.dq_allow_warnings = True`` —
            proceed past soft DQ warnings (coverage gaps, outliers, small
            segments) instead of halting. DQ is fail-closed by default (M2);
            this is the analyst escape hatch. Equivalent to the
            ``--allow-dq-warnings`` CLI flag. The flag can only *relax* DQ; when
            absent the resolved ``dq_allow_warnings`` config value is used.
        preloaded_data: Optional pre-loaded and standardized DataFrame. When
            provided, bypasses the SAS read in the data-loading phase.
            ``run_batch.py`` uses this to share a loaded DataFrame across
            segments.
        output: ``OutputPaths`` instance controlling where all artifacts are
            written. Defaults to paths rooted at the current working
            directory.
        floor_cells_path: Path to a CSV of "floor cells" (bins that must be
            accepted regardless of optimization). Used for sequential cutoff
            ordering across segments — see ``cutoff_floor_segment`` /
            ``cutoff_ordering_mode`` in ``CLAUDE.md``.
        floor_cells_mode: ``"floor"`` (must-accept) or ``"ceiling"``
            (must-reject) interpretation of ``floor_cells_path``. Paired with
            the bottom-up vs top-down ordering chosen in batch mode.
        resimulate_risk: Optional list of risk targets (in %). When provided,
            the pipeline skips training and optimization, reloads cached
            optimization artefacts, and re-runs scenario analysis at the
            supplied targets. See the ``--resimulate`` CLI flag and the
            Resimulation section of ``CLAUDE.md``.

    Returns:
        On success: a result tuple from ``run_optimization_phase`` /
        ``run_scenario_analysis`` (full and ``training_only`` runs), or ``None``
        in ``resimulate_risk`` mode (no downstream result to return).
        On failure: ``False`` — a ``PipelineExecutionError`` is caught, logged,
        and converted to this sentinel so the CLI can exit non-zero (``None``
        alone cannot distinguish resimulation success from failure).
    """
    if output is None:
        output = OutputPaths()
    output.ensure_dirs()

    t0_total = time.perf_counter()

    try:
        # Step 1: Load and validate configuration
        try:
            settings, date_ini, date_fin, annual_coef = load_and_validate_config(config_path)
        except ValidationError as e:
            raise ConfigLoadError(f"Configuration validation failed for '{config_path}'") from e
        except Exception as e:
            raise ConfigLoadError(f"Error loading configuration from '{config_path}'") from e

        segment = settings.segment_filter
        if baseline_mode:
            settings.baseline_mode = True
        if base_scenario_only:
            settings.base_scenario_only = True
        if allow_dq_warnings:
            # Analyst escape hatch (M2): relax the fail-closed DQ gate for this run.
            settings.dq_allow_warnings = True

        # Resimulation mode: skip phases 2-5, load artifacts, run scenarios only
        if resimulate_risk:
            # Let run_resimulation infer output dir from config path unless
            # an explicit output was provided by the caller (e.g., run_batch.py).
            import pathlib

            resim_output = output if output.base_dir != pathlib.Path(".") else None
            run_resimulation(config_path, resimulate_risk, output=resim_output)
            return None

        # Step 2: Load and prepare data
        try:
            data, settings = load_and_prepare_data(settings, preloaded_data)
        except (DataValidationError, FileNotFoundError) as e:
            raise DataLoadError(f"[{segment}] Data error") from e
        except Exception as e:
            raise DataLoadError(f"[{segment}] Unexpected data loading error") from e

        # Capture per-run data lineage / provenance (M2). Best-effort: a lineage
        # failure must never abort the pipeline (it is a diagnostic, not decisioning).
        try:
            lineage = build_lineage(
                settings,
                config_path=config_path,
                n_rows=len(data),
                run_id=run_id,
                run_ts_iso=run_ts_iso,
            )
            log_run_banner(lineage)
            write_lineage(output, lineage)
        except Exception:
            logger.warning(f"[{segment}] Could not capture run lineage", exc_info=True)

        # Step 3: Preprocessing (DQ checks, binning, stress factor, transformation rate)
        prep = run_preprocessing_phase(data, settings, skip_dq_checks, output=output)
        if prep is None:
            return None
        data_clean = prep.data_clean
        data_booked = prep.data_booked
        data_demand = prep.data_demand
        stress_factor = prep.stress_factor
        tasa_fin = prep.tasa_fin
        per_bin_stress = prep.per_bin_stress
        per_bin_tasa_fin = prep.per_bin_tasa_fin

        # Save resimulation artifacts (parquet + scalars) for future --resimulate runs
        try:
            _save_resimulation_artifacts(
                output,
                data_clean,
                data_booked,
                stress_factor,
                tasa_fin,
                annual_coef,
                settings,
                per_bin_stress,
                per_bin_tasa_fin,
            )
        except Exception as e:
            logger.warning(f"[{segment}] Failed to save resimulation artifacts (non-blocking): {e}")

        # Step 4: Risk inference (model training or loading)
        try:
            risk_inference, reg_todu_amt_pile = run_inference_phase(data_clean, settings, model_path, output=output)
        except Exception as e:
            raise InferencePhaseError(f"[{segment}] Inference phase failed") from e

        # Early return for training_only mode (supersegment training)
        if training_only:
            elapsed = time.perf_counter() - t0_total
            logger.info(
                f"[{segment}] Training only complete | "
                f"model={risk_inference.get('model_path', 'models/')} | {elapsed:.1f}s total"
            )
            return data_clean, data_booked, data_demand, risk_inference, reg_todu_amt_pile

        # Step 5: Optimization (MILP Pareto frontier or fixed cutoffs)
        opt = run_optimization_phase(
            data_booked,
            data_demand,
            risk_inference,
            reg_todu_amt_pile,
            stress_factor,
            tasa_fin,
            settings,
            annual_coef,
            output=output,
            per_bin_stress=per_bin_stress,
            per_bin_tasa_fin=per_bin_tasa_fin,
            floor_cells_path=floor_cells_path,
            floor_cells_mode=floor_cells_mode,
        )
        data_summary_desagregado = opt.data_summary_desagregado
        data_summary = opt.data_summary
        data_summary_sample_no_opt = opt.data_summary_sample_no_opt
        values_per_var = opt.values_per_var
        grid = opt.grid
        pareto_masks = opt.pareto_masks
        floor_fixed_cells = opt.floor_fixed_cells

        # Compute total demand (booked + rejected, excluding canceled)
        if "status_name" in data_demand.columns and "oa_amt_h0" in data_demand.columns:
            total_demand = data_demand.loc[data_demand["status_name"] != "canceled", "oa_amt_h0"].sum()
        else:
            total_demand = data_demand["oa_amt_h0"].sum() if "oa_amt_h0" in data_demand.columns else 0.0

        # Step 6: Scenario analysis loop
        use_fixed_cutoffs = settings.fixed_cutoffs is not None and len(settings.fixed_cutoffs) > 0
        scenarios = build_scenario_list(settings, use_fixed_cutoffs)
        annual_coef_mr = compute_mr_annual_coef(settings)

        # Clean up stale scenario files that won't be regenerated (e.g. baseline
        # mode only produces "base", but old pessimistic/optimistic files — or an
        # orphaned target_X.XX from a prior resimulation — would confuse the
        # consolidated report). Discovers on-disk scenarios so it also purges
        # non-canonical target_X.XX names and covers accepted_cells/audit (#48).
        cleanup_stale_scenarios(output, {name for _, name in scenarios}, segment)

        cutoff_summaries = []
        for scenario_risk, scenario_name in scenarios:
            summary = run_scenario_analysis(
                scenario_risk,
                scenario_name,
                data_summary=data_summary,
                data_summary_desagregado=data_summary_desagregado,
                data_summary_sample_no_opt=data_summary_sample_no_opt,
                data_clean=data_clean,
                data_booked=data_booked,
                settings=settings,
                risk_inference=risk_inference,
                reg_todu_amt_pile=reg_todu_amt_pile,
                stress_factor=stress_factor,
                tasa_fin=tasa_fin,
                annual_coef_mr=annual_coef_mr,
                values_per_var=values_per_var,
                grid=grid,
                pareto_masks=pareto_masks,
                output=output,
                total_demand=total_demand,
                per_bin_stress=per_bin_stress,
                per_bin_tasa_fin=per_bin_tasa_fin,
            )
            cutoff_summaries.append(summary)

        save_cutoff_summaries(cutoff_summaries, settings, output=output)

        # Step 6c: Reject inference parameter optimization (optional, non-blocking, skipped in baseline mode)
        best_ri_params = None
        if not settings.baseline_mode:
            best_ri_params = run_ri_optimizer_phase(
                data_booked=data_booked,
                data_demand=data_demand,
                risk_inference=risk_inference,
                reg_todu_amt_pile=reg_todu_amt_pile,
                stress_factor=stress_factor,
                tasa_fin=tasa_fin,
                settings=settings,
                annual_coef=annual_coef,
                output=output,
                per_bin_stress=per_bin_stress,
                per_bin_tasa_fin=per_bin_tasa_fin,
            )

        # Step 6d: Re-run optimization with tuned RI params if they changed
        if best_ri_params:
            old_uplift = settings.reject_uplift_factor
            old_max_mult = settings.reject_max_risk_multiplier
            new_uplift = best_ri_params["uplift_factor"]
            new_max_mult = best_ri_params["max_risk_multiplier"]

            if abs(old_uplift - new_uplift) > 0.01 or abs(old_max_mult - new_max_mult) > 0.01:
                logger.info(
                    f"[{segment}] RI optimizer found better params: "
                    f"uplift {old_uplift:.2f} -> {new_uplift:.2f}, "
                    f"max_mult {old_max_mult:.2f} -> {new_max_mult:.2f}. Re-running optimization."
                )
                settings.reject_uplift_factor = new_uplift
                settings.reject_max_risk_multiplier = new_max_mult

                opt = run_optimization_phase(
                    data_booked,
                    data_demand,
                    risk_inference,
                    reg_todu_amt_pile,
                    stress_factor,
                    tasa_fin,
                    settings,
                    annual_coef,
                    output=output,
                    per_bin_stress=per_bin_stress,
                    per_bin_tasa_fin=per_bin_tasa_fin,
                    floor_cells_path=floor_cells_path,
                    floor_cells_mode=floor_cells_mode,
                )
                data_summary_desagregado = opt.data_summary_desagregado
                data_summary = opt.data_summary
                data_summary_sample_no_opt = opt.data_summary_sample_no_opt
                values_per_var = opt.values_per_var
                grid = opt.grid
                pareto_masks = opt.pareto_masks
                floor_fixed_cells = opt.floor_fixed_cells

                cutoff_summaries = []
                for scenario_risk, scenario_name in scenarios:
                    summary = run_scenario_analysis(
                        scenario_risk,
                        scenario_name,
                        data_summary=data_summary,
                        data_summary_desagregado=data_summary_desagregado,
                        data_summary_sample_no_opt=data_summary_sample_no_opt,
                        data_clean=data_clean,
                        data_booked=data_booked,
                        settings=settings,
                        risk_inference=risk_inference,
                        reg_todu_amt_pile=reg_todu_amt_pile,
                        stress_factor=stress_factor,
                        tasa_fin=tasa_fin,
                        annual_coef_mr=annual_coef_mr,
                        values_per_var=values_per_var,
                        grid=grid,
                        pareto_masks=pareto_masks,
                        output=output,
                        total_demand=total_demand,
                        per_bin_stress=per_bin_stress,
                        per_bin_tasa_fin=per_bin_tasa_fin,
                    )
                    cutoff_summaries.append(summary)

                save_cutoff_summaries(cutoff_summaries, settings, output=output)
            else:
                logger.info(f"[{segment}] RI optimizer confirmed current params are optimal")

        # Step 6e: Sensitivity analysis (optional, non-blocking, skipped in baseline mode).
        # #55: runs AFTER the RI optimizer's re-optimization (6d) so it describes the
        # FINAL surface + shipped base mask, not a pre-tuning policy that may have
        # been superseded.
        if not settings.baseline_mode:
            from src.pipeline.optimization import run_sensitivity_phase

            run_sensitivity_phase(
                data_summary_desagregado=data_summary_desagregado,
                data_summary=data_summary,
                settings=settings,
                output=output,
                fixed_cells=floor_fixed_cells,
            )

        # Step 7: Temporal trend analysis (non-blocking)
        try:
            from src.trends import compute_monthly_metrics, detect_trend_changes, plot_metric_trends

            monthly = compute_monthly_metrics(
                data_clean,
                date_column="mis_date",
                segment_filter=segment,
                maturity_months=settings.mr_maturity_months,
            )
            if not monthly.empty:
                # Save monthly metrics
                monthly.to_csv(output.monthly_metrics_csv(segment))

                # Plot key metrics
                plot_metric_trends(
                    monthly,
                    ["approval_rate", "total_records", "mean_production"],
                    output_path=output.metric_trends_html(segment),
                )

                # Detect anomalies in approval rate (window=6: the SPC default; >=5 moving ranges
                # per estimate — window=3 gave only 2, an unstable scale; audit #10)
                anomalies = detect_trend_changes(monthly, "approval_rate", window=6)
                anomaly_months = anomalies[anomalies["is_anomaly"]]
                if not anomaly_months.empty:
                    anomalies.to_csv(output.trend_anomalies_csv(segment))
                    logger.warning(f"[{segment}] Trend anomalies detected in {len(anomaly_months)} month(s)")
                else:
                    logger.info(f"[{segment}] No trend anomalies detected")
            else:
                logger.info(f"[{segment}] Insufficient data for trend analysis")
        except Exception as e:
            logger.warning(f"[{segment}] Trend analysis failed (non-blocking): {e}")

        # Step 8: Generate HTML report (non-blocking)
        try:
            from src.pipeline.reporting import generate_segment_report

            report_path = generate_segment_report(
                settings=settings, output=output, scenarios=[name for _, name in scenarios]
            )
            if report_path:
                logger.info(f"[{segment}] Report generated: {report_path}")
        except Exception as e:
            logger.warning(f"[{segment}] Report generation failed (non-blocking): {e}")

        elapsed_total = time.perf_counter() - t0_total
        logger.info(f"[{segment}] Pipeline complete | {len(scenarios)} scenarios | {elapsed_total:.1f}s total")

        return data_clean, data_booked, data_demand, data_summary_desagregado, data_summary
    except PipelineExecutionError as e:
        logger.exception(str(e))
        # #61: return an explicit failure sentinel (not None) so the CLI can set
        # a non-zero exit code — resimulation-mode SUCCESS also returns None, so
        # None alone cannot distinguish success from failure.
        return False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring and Portfolio Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run python main.py

  # Run with custom config file
  uv run python main.py --config path/to/config.toml

  # Skip data quality checks (faster, not recommended for production)
  uv run python main.py --skip-dq-checks

  # Training only mode (skip optimization and scenario analysis)
  uv run python main.py --training-only

  # Use pre-trained model for optimization
  uv run python main.py --model-path models/model_20240101_120000

Output files:
  data/optimal_solution_*.csv          - Optimal cutoff solutions per scenario
  data/risk_production_summary_*.csv   - Risk/production metrics
  data/cutoff_summary_by_segment.csv   - Cutoff points summary (long format)
  data/cutoff_summary_wide.csv         - Cutoff points summary (wide format)
  images/risk_production_*.html        - Interactive visualizations
  data/monthly_metrics_*.csv           - Monthly aggregated metrics
  data/trend_anomalies_*.csv           - Detected trend anomalies
  images/metric_trends_*.html          - Monthly metric trend charts
        """,
    )

    parser.add_argument(
        "--config", "-c", type=str, default="config.toml", help="Path to configuration TOML file (default: config.toml)"
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=None,
        help="Path to pre-trained model directory. Skips training and uses existing model.",
    )

    parser.add_argument(
        "--training-only",
        "-t",
        action="store_true",
        help="Run only data preprocessing and model training (skips optimization). Useful for supersegment training.",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: show current portfolio as-is (no cutoff optimization, no swap-in/swap-out).",
    )

    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run only the base scenario (skip pessimistic and optimistic).",
    )

    parser.add_argument(
        "--skip-dq-checks",
        action="store_true",
        help="Skip data quality checks (use with caution).",
    )

    parser.add_argument(
        "--allow-dq-warnings",
        action="store_true",
        help=(
            "Analyst escape hatch: proceed past non-critical DQ warnings (coverage gaps, outliers, "
            "small segments) instead of halting. DQ is fail-closed by default; FAILED-severity checks "
            "(negative counts/amounts, unparseable dates) still halt — use --skip-dq-checks to skip DQ entirely."
        ),
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to write all log output to a file (in addition to console).",
    )

    parser.add_argument(
        "--resimulate",
        type=float,
        nargs="+",
        default=None,
        metavar="RISK",
        help=(
            "Resimulation mode: skip data loading/preprocessing/training/optimization, "
            "load artifacts from a previous run, and re-run scenario analysis with the "
            "specified optimum_risk value(s). "
            "1 value → base only; 3 values → pessimistic/base/optimistic; "
            "N values → target_X.XX. Example: --resimulate 0.8 0.96 1.2"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = parse_args()

    sink_id = None
    if args.log_file:
        sink_id = logger.add(args.log_file, rotation="50 MB", level="DEBUG")

    try:
        result = main(
            config_path=args.config,
            model_path=args.model_path,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            skip_dq_checks=args.skip_dq_checks,
            allow_dq_warnings=args.allow_dq_warnings,
            resimulate_risk=args.resimulate,
        )
    finally:
        if sink_id is not None:
            logger.remove(sink_id)

    # #61: exit non-zero on pipeline failure so CI / scripts can detect it.
    # main() returns the False sentinel on PipelineExecutionError; success
    # returns a tuple (full/training) or None (resimulation).
    if result is False:
        sys.exit(1)
