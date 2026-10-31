"""
Batch runner for processing multiple segments.

This script runs the main pipeline for each segment defined in segments.toml,
creating separate output directories for each segment.

Usage:
    python run_batch.py                    # Run all segments
    python run_batch.py --segments a b c   # Run specific segments
    python run_batch.py --list             # List available segments
    python run_batch.py --parallel         # Run segments in parallel

Output structure:
    output/
    ├── {segment_name}/
    │   ├── images/
    │   ├── models/
    │   ├── data/
    │   └── logs/
    └── ...
"""

import argparse
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from src.core.config import OutputPaths
from src.core.utils import resolve_modelling_supersegment, resolve_reporting_supersegment
from src.data.bin_learning import learn_global_bin_edges, learn_supersegment_bin_edges
from src.data.data_manager import load_configured_data
from src.governance.lineage import git_provenance
from src.pipeline.batch import (
    _safe_remove_sink,
    clean_output_directories,
    run_segments_parallel,
    run_segments_sequential,
)
from src.pipeline.batch_config import (
    load_base_config,
    load_reporting_supersegments_config,
    load_segments_config,
    load_supersegments_config,
    validate_supersegment_edge_consistency,
)
from src.report.consolidation import generate_consolidation_report

# Removed in R2 (todo #67): _working_directory context manager.
# It used os.chdir to isolate each segment's output under dirs["root"].
# All pipeline phases now accept an explicit `output: OutputPaths`
# parameter that threads through every write, so process-global cwd
# manipulation is no longer needed. Unblocks ThreadPoolExecutor usage.


def print_summary(results: dict[str, bool]) -> None:
    """Print a summary of all segment runs."""
    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)

    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]

    print(f"\nTotal segments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSuccessful segments:")
        for name in successful:
            print(f"  - {name}")

    if failed:
        print("\nFailed segments:")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 80)


def list_segments(segments_path: str = "segments.toml") -> None:
    """List all available segments and supersegments."""
    segments = load_segments_config(segments_path)
    supersegments = load_supersegments_config(segments_path)

    # Show supersegments first
    if supersegments:
        print("\nSupersegments (shared model training):")
        print("-" * 60)
        for name, config in supersegments.items():
            filters = config.get("segment_filters", [])
            print(f"  {name}:")
            print("    segment_filters:")
            for sf in filters:
                print(f"      - {sf}")
            print()

    print("\nAvailable segments:")
    print("-" * 60)

    for name, config in segments.items():
        filter_val = config.get("segment_filter", "N/A")
        risk = config.get("optimum_risk", "default")
        modelling_ss = resolve_modelling_supersegment(config)
        reporting_ss = resolve_reporting_supersegment(config)
        print(f"  {name}:")
        print(f"    segment_filter: {filter_val}")
        print(f"    optimum_risk: {risk}")
        if modelling_ss:
            print(f"    modelling_supersegment: {modelling_ss} (shared model)")
        if reporting_ss:
            print(f"    reporting_supersegment: {reporting_ss} (reporting group)")
        print()


def _run_resimulation_mode(args, base_config, segments, all_supersegments, used_supersegments) -> int:
    """``--resimulate``: re-run scenarios at new risk targets on completed segment runs (no retraining)."""
    from src.pipeline.segment_runner import run_resimulation

    # Parse risk targets: either a TOML file or inline float values
    # Result: per_segment_risks = {seg_name: [risk1, risk2, ...]} or None for global
    per_segment_risks: dict[str, list[float]] | None = None
    global_risks: list[float] | None = None

    if len(args.resimulate) == 1 and args.resimulate[0].endswith(".toml"):
        # TOML file with per-segment targets
        toml_path = args.resimulate[0]
        if not Path(toml_path).exists():
            print(f"Error: resimulation TOML file not found: {toml_path}")
            return 1
        resim_config = tomllib.loads(Path(toml_path).read_text(encoding="utf-8"))
        per_segment_risks = {}
        for seg_name, val in resim_config.items():
            if isinstance(val, (int, float)):
                per_segment_risks[seg_name] = [float(val)]
            elif isinstance(val, list):
                per_segment_risks[seg_name] = [float(v) for v in val]
            else:
                logger.warning(f"Skipping invalid entry in resimulation TOML: {seg_name} = {val}")
        print(f"\n{'=' * 60}")
        print(f"Resimulation Mode — per-segment targets from {toml_path}")
        for sn, rv in per_segment_risks.items():
            print(f"  {sn}: {rv}")
        print(f"{'=' * 60}")
    elif len(args.resimulate) == 0:
        print("Error: --resimulate requires risk values or a TOML file path")
        return 1
    else:
        # Inline float values applied to all segments
        try:
            global_risks = [float(v) for v in args.resimulate]
        except ValueError:
            print(f"Error: --resimulate values must be numbers or a .toml file path, got: {args.resimulate}")
            return 1
        print(f"\n{'=' * 60}")
        print(f"Resimulation Mode — risk targets: {global_risks}")
        print(f"{'=' * 60}")

    failed_segments = []
    segments_to_run = list(per_segment_risks.keys()) if per_segment_risks else list(segments.keys())
    for seg_name in segments_to_run:
        if seg_name not in segments:
            logger.warning(f"[{seg_name}] Not in segments.toml, skipping")
            failed_segments.append(seg_name)
            continue
        seg_output_dir = Path(args.output) / seg_name
        if not seg_output_dir.exists():
            logger.warning(f"[{seg_name}] No output directory found, skipping resimulation")
            failed_segments.append(seg_name)
            continue
        seg_config_path = seg_output_dir / "config_segment.toml"
        if not seg_config_path.exists():
            logger.warning(f"[{seg_name}] No config_segment.toml found, skipping")
            failed_segments.append(seg_name)
            continue

        risk_values = per_segment_risks[seg_name] if per_segment_risks else global_risks

        # Skip segments whose requested risk matches the original optimum_risk
        # (no point re-running if nothing changed — avoids MR drift from code updates).
        if per_segment_risks:
            try:
                _seg_cfg = tomllib.loads(seg_config_path.read_text(encoding="utf-8"))
                _orig_risk = _seg_cfg.get("preprocessing", _seg_cfg).get("optimum_risk")
                if _orig_risk is not None and risk_values == [float(_orig_risk)]:
                    logger.info(f"[{seg_name}] Risk unchanged ({risk_values[0]}), skipping resimulation")
                    continue
            except Exception:
                pass  # proceed with resimulation if config can't be read

        try:
            seg_output = OutputPaths(base_dir=seg_output_dir)
            run_resimulation(
                config_path=str(seg_config_path),
                resimulate_risk=risk_values,
                output=seg_output,
            )
        except Exception as e:
            logger.error(f"[{seg_name}] Resimulation failed: {e}")
            failed_segments.append(seg_name)

    # Generate consolidated report — exclude segments whose resimulation
    # failed (#48/#61): their on-disk artifacts are the PREVIOUS run's, so
    # consolidating them would present stale targets as "resimulated".
    consolidation_segments = {n: c for n, c in segments.items() if n not in failed_segments}
    try:
        consolidated_df, _ = generate_consolidation_report(
            output_base=args.output,
            segments=consolidation_segments,
            supersegments=all_supersegments,
            output_path=args.output,
            multiplier=base_config.get("multiplier", 7),
            multiplier_h3=base_config.get("multiplier_h3", 4),
        )
    except Exception as e:
        logger.error(f"Consolidation failed: {e}")

    if failed_segments:
        print(f"\nFailed segments (excluded from consolidation): {failed_segments}")
        return 1
    print(f"\nResimulation complete for {len(segments_to_run)} segment(s)")
    return 0


def _post_run_outputs(args, base_config, segments, all_supersegments, preloaded_data, results) -> None:
    """After the segment runs: OOT backtest, consolidated report, HTML reports, score discriminance."""
    # Out-of-time backtest (M4) — runs before consolidation so the report's OOT sheet is populated.
    # Reuses the already-loaded SAS data; skipped under --training-only / --no-backtest / no data.
    if not args.no_backtest and not args.training_only and preloaded_data is not None:
        successful_segments = [name for name, ok in results.items() if ok]
        if successful_segments:
            print(f"\n{'=' * 60}")
            print("Out-of-time backtest (frozen cutoffs vs held-out cohort)")
            print(f"{'=' * 60}")
            try:
                from src.governance.backtest_runner import backtest_all

                backtest_all(
                    preloaded_data,
                    data_dir=args.output,
                    out_dir=str(Path(args.output) / "backtest"),
                    scenario="base",
                    segments=successful_segments,
                )
            except Exception:
                logger.exception("Out-of-time backtest step failed (non-fatal; consolidated report continues)")

    # Generate consolidated report
    if not args.no_consolidation and not args.training_only:
        successful_segments = {name: config for name, config in segments.items() if results.get(name, False)}
        if successful_segments:
            print(f"\n{'=' * 60}")
            print("Generating Consolidated Report")
            print(f"{'=' * 60}")
            try:
                consolidated_df, _ = generate_consolidation_report(
                    output_base=args.output,
                    segments=successful_segments,
                    supersegments=all_supersegments,
                    output_path=args.output,
                    multiplier=base_config.get("multiplier", 7),
                    multiplier_h3=base_config.get("multiplier_h3", 4),
                )
                print("\nConsolidated report saved to:")
                print(f"  - {args.output}/consolidated_risk_production.csv")
                print(f"  - {args.output}/consolidated_risk_production.html")
            except Exception as e:
                logger.error(f"Error generating consolidated report: {e}")
                logger.exception("Full traceback:")
        else:
            print("\nNo successful segments to consolidate.")

    # Generate HTML reports
    if not args.no_consolidation and not args.training_only and not args.no_report:
        print(f"\n{'=' * 60}")
        print("Generating HTML Reports")
        print(f"{'=' * 60}")
        try:
            from src.pipeline.reporting import generate_batch_reports

            report_paths = generate_batch_reports(
                output_base=args.output,
                segments=segments,
                supersegments=all_supersegments,
                segment_results=results,
            )
            if report_paths:
                print(f"\nHTML reports generated ({len(report_paths)}):")
                for name, path in report_paths.items():
                    print(f"  - {name}: {path}")
            else:
                print("\nNo HTML reports generated (no successful segments or artifacts).")
        except Exception as e:
            logger.error(f"Report generation failed: {e}")

    # Generate score discriminance report
    if not args.no_consolidation and not args.training_only and preloaded_data is not None:
        print(f"\n{'=' * 60}")
        print("Generating Score Discriminance Report")
        print(f"{'=' * 60}")
        try:
            from src.report.score_metrics import generate_score_discriminance_report

            disc_df = generate_score_discriminance_report(
                preloaded_data=preloaded_data,
                segments=segments,
                supersegments=all_supersegments,
                base_config=base_config,
                output_path=args.output,
            )
            if not disc_df.empty:
                print(f"\nScore discriminance report saved to: {args.output}/score_discriminance.csv")
            else:
                print("\nNo score discriminance metrics computed.")
        except Exception as e:
            logger.error(f"Error generating score discriminance report: {e}")
            logger.exception("Full traceback:")


def main():
    parser = argparse.ArgumentParser(
        description="Run scoring pipeline for multiple segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--segments", "-s", nargs="+", help="Specific segments to run (default: all)")
    parser.add_argument("--list", "-l", action="store_true", help="List available segments and exit")
    parser.add_argument("--parallel", "-p", action="store_true", help="Run segments in parallel")
    parser.add_argument(
        "--workers", "-w", type=int, default=None, help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument("--output", "-o", default="output", help="Base output directory (default: output)")
    parser.add_argument("--config", "-c", default="config.toml", help="Path to base config file (default: config.toml)")
    parser.add_argument(
        "--segments-config", default="segments.toml", help="Path to segments config file (default: segments.toml)"
    )
    parser.add_argument(
        "--reuse-models", action="store_true", help="Reuse existing supersegment models if available (skip retraining)"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Remove output directories for selected segments before running"
    )
    parser.add_argument("--clean-only", action="store_true", help="Only clean output directories (don't run pipeline)")
    parser.add_argument(
        "--skip-dq-checks", action="store_true", help="Skip data quality checks (not recommended for production)"
    )
    parser.add_argument(
        "--allow-dq-warnings",
        action="store_true",
        help=(
            "Analyst escape hatch: proceed past non-critical DQ warnings instead of halting. DQ is "
            "fail-closed by default; FAILED-severity checks still halt (use --skip-dq-checks to skip DQ entirely)."
        ),
    )
    parser.add_argument(
        "--no-consolidation", action="store_true", help="Skip generating consolidated report at the end"
    )
    parser.add_argument(
        "--consolidate-only", action="store_true", help="Only generate consolidated report (skip running segments)"
    )
    parser.add_argument("--training-only", action="store_true", help="Only run data quality and training")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: show current portfolio as-is (no cutoff optimization).",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run only the base scenario (skip pessimistic and optimistic).",
    )
    parser.add_argument(
        "--cutoff-ordering-mode",
        choices=["bottom_up", "top_down"],
        default=None,
        help="Cutoff ordering direction: bottom_up (tightest first, floor constraints) "
        "or top_down (least restrictive first, ceiling constraints). "
        "Overrides cutoff_ordering_mode in config.toml.",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip generating HTML reports")
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip the out-of-time backtest step (M4) that runs after segments and feeds the consolidated report.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Path to write all log output to a file (in addition to console)"
    )
    parser.add_argument(
        "--resimulate",
        type=str,
        nargs="*",
        default=None,
        metavar="RISK_OR_FILE",
        help=(
            "Resimulation mode: skip data loading/preprocessing/training/optimization, "
            "load artifacts from a previous full run, and re-run scenario analysis with "
            "new risk targets. Accepts either:\n"
            "  (a) One or more risk values applied to all segments: --resimulate 0.8 1.2 1.6\n"
            "  (b) A TOML file with per-segment targets: --resimulate scenarios.toml\n"
            "TOML format:  no_premium_cd = [0.8, 1.2, 1.6]  or  no_premium_cd = 1.4"
        ),
    )

    args = parser.parse_args()

    # Setup global log file if requested
    global_sink_id = None
    if args.log_file:
        global_sink_id = logger.add(args.log_file, rotation="50 MB", level="DEBUG")

    # List segments if requested
    if args.list:
        list_segments(args.segments_config)
        return 0

    # Load configurations
    try:
        base_config = load_base_config(args.config)
        all_segments = load_segments_config(args.segments_config)
        all_supersegments = load_supersegments_config(args.segments_config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Filter segments if specific ones requested
    if args.segments:
        segments = {name: config for name, config in all_segments.items() if name in args.segments}

        # Check for unknown segments
        unknown = set(args.segments) - set(all_segments.keys())
        if unknown:
            print(f"Warning: Unknown segments will be skipped: {unknown}")

        if not segments:
            print("Error: No valid segments to process")
            return 1
    else:
        segments = all_segments

    # Identify and validate supersegments used by selected segments
    used_supersegments = set()
    for seg_name, seg_config in segments.items():
        modelling_ss = resolve_modelling_supersegment(seg_config)
        if modelling_ss:
            if modelling_ss not in all_supersegments:
                print(f"Warning: Segment '{seg_name}' references unknown modelling supersegment '{modelling_ss}'")
                print(f"  Available supersegments: {list(all_supersegments.keys())}")
                print("  Segment will train its own model instead.")
            else:
                used_supersegments.add(modelling_ss)
        reporting_ss = resolve_reporting_supersegment(seg_config)
        if reporting_ss and reporting_ss not in all_supersegments:
            print(f"Warning: Segment '{seg_name}' references unknown reporting supersegment '{reporting_ss}'")
            print(f"  Available supersegments: {list(all_supersegments.keys())}")

    # Handle clean operations
    if args.clean or args.clean_only:
        print(f"\nCleaning output directories for {len(segments)} segment(s)...")
        clean_results = clean_output_directories(
            segments=segments, supersegments=all_supersegments, output_base=args.output
        )
        failed_cleans = [name for name, success in clean_results.items() if not success]
        if failed_cleans:
            print(f"Warning: Failed to clean: {failed_cleans}")

        if args.clean_only:
            print("\nClean complete.")
            return 0 if not failed_cleans else 1

        print()  # Blank line after clean output

    # Handle consolidate-only mode
    if args.consolidate_only:
        print(f"\n{'=' * 60}")
        print("Consolidate-Only Mode")
        print(f"{'=' * 60}")
        try:
            consolidated_df, _ = generate_consolidation_report(
                output_base=args.output,
                segments=segments,
                supersegments=all_supersegments,
                output_path=args.output,
                multiplier=base_config.get("multiplier", 7),
                multiplier_h3=base_config.get("multiplier_h3", 4),
            )
            print("\nConsolidated report saved to:")
            print(f"  - {args.output}/consolidated_risk_production.csv")
            print(f"  - {args.output}/consolidated_risk_production.html")
            return 0
        except Exception as e:
            logger.error(f"Error generating consolidated report: {e}")
            logger.exception("Full traceback:")
            return 1

    # Handle resimulation mode
    if args.resimulate is not None:
        return _run_resimulation_mode(args, base_config, segments, all_supersegments, used_supersegments)

    print(f"\nProcessing {len(segments)} segment(s): {list(segments.keys())}")
    if used_supersegments:
        print(f"Supersegments to train: {list(used_supersegments)}")
        if args.reuse_models:
            print("Reuse models: enabled (will skip training if model exists)")
    if args.skip_dq_checks:
        print("Data quality checks: DISABLED (--skip-dq-checks)")
    print(f"Output directory: {args.output}")
    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")
    print()

    # Try to load data once for all segments (optimization)
    print(f"{'=' * 60}")
    print("Attempting to preload data (optimization)")
    print(f"{'=' * 60}")
    preloaded_data = load_configured_data(base_config)

    if preloaded_data is not None:
        print(f"Data preloaded: {preloaded_data.shape[0]:,} rows × {preloaded_data.shape[1]} columns")
        print("All segments will use preloaded data.\n")
    else:
        print("Data preloading skipped (file not accessible locally).")
        print("Each segment will load data individually from configured path.\n")

    # Learn global bin edges ONCE on the full dataset so all segments share
    # identical edges (instead of each segment learning from its own filtered
    # subpopulation).
    global_bin_edges: dict[str, list[float]] = {}
    supersegment_bin_edges: dict[str, dict[str, list[float]]] = {}
    reporting_ss = load_reporting_supersegments_config(args.segments_config)
    if preloaded_data is not None:
        global_bin_edges = learn_global_bin_edges(preloaded_data, base_config)

        # Learn per-reporting-supersegment bin edges so each supersegment can
        # have splits tuned to its own population (e.g., different income_bin
        # edges for "others" vs "known").
        if reporting_ss:
            supersegment_bin_edges = learn_supersegment_bin_edges(preloaded_data, base_config, reporting_ss)
    else:
        # Preload failed. Fixed supersegment edges are CONFIGURATION, not
        # data-derived — extract them regardless so a transient load failure
        # can never silently swap the configured grid for per-segment
        # quantile-learned bins (audit #27).
        if reporting_ss:
            supersegment_bin_edges = learn_supersegment_bin_edges(None, base_config, reporting_ss)

        # Fail closed: if any bins were supposed to be LEARNED on the shared
        # population (globally or per supersegment), proceeding would let each
        # segment learn its own edges — different grids, different cutoffs,
        # silently. Halt instead of degrading methodology.
        bins_config = base_config.get("bins", {})
        learnable_global = [
            var
            for var, bc in bins_config.items()
            if not (bc.get("bin_edges") and len(bc["bin_edges"]) >= 2) and bc.get("max_bins") is not None
        ]
        unresolved_global = [var for var in learnable_global if var not in global_bin_edges]
        learn_ss = [
            ss for ss, cfg in reporting_ss.items() if cfg.get("learn_own_bin_edges", False) and not cfg.get("bin_edges")
        ]
        if unresolved_global or learn_ss:
            logger.error(
                "Data preloading failed but shared bin learning was requested "
                f"(global learnable bins: {unresolved_global or 'none'}; "
                f"supersegments with learn_own_bin_edges: {learn_ss or 'none'}). "
                "Proceeding would silently fall back to per-segment bin learning — "
                "different grids and cutoffs. Fix the data path/storage and rerun "
                "(audit #27: fail-closed)."
            )
            print("\nERROR: data preload failed while shared bin learning is configured — aborting (see log).")
            sys.exit(1)

    # Batch-start pre-flight (#40c): fail FAST (before training/optimizing any segment) if a segment
    # that reuses a modelling-supersegment's shared model would run on a different INFERENCE grid than
    # the shared model trains on. The #40 a/b reuse validation also catches this per member, but here
    # we surface it up front for the whole batch.
    validate_supersegment_edge_consistency(base_config, segments, global_bin_edges, supersegment_bin_edges)

    # Resolve cutoff ordering mode: CLI flag overrides config.toml
    cutoff_ordering_mode = args.cutoff_ordering_mode or base_config.get("cutoff_ordering_mode", "bottom_up")

    # One canonical run-id for the whole batch (M2 lineage); every segment's
    # run_lineage.json shares it so a batch is traceable end-to-end.
    run_ts = datetime.now(UTC)
    run_ts_iso = run_ts.isoformat()
    git = git_provenance()
    run_id = f"{run_ts.strftime('%Y%m%dT%H%M%SZ')}_{git.get('short') or 'nogit'}"
    dirty = " (dirty)" if git.get("dirty") else ""
    logger.info(
        f"==== BATCH RUN {run_id} | {len(segments)} segment(s) | config={args.config} | "
        f"git={git.get('short') or 'n/a'}{dirty} ===="
    )

    # Run segments
    if args.parallel:
        results = run_segments_parallel(
            segments,
            base_config,
            args.output,
            args.workers,
            supersegments=all_supersegments,
            reuse_models=args.reuse_models,
            skip_dq_checks=args.skip_dq_checks,
            allow_dq_warnings=args.allow_dq_warnings,
            preloaded_data=preloaded_data,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            global_bin_edges=global_bin_edges,
            supersegment_bin_edges=supersegment_bin_edges,
            cutoff_ordering_mode=cutoff_ordering_mode,
            run_id=run_id,
            run_ts_iso=run_ts_iso,
        )
    else:
        results = run_segments_sequential(
            segments,
            base_config,
            args.output,
            supersegments=all_supersegments,
            reuse_models=args.reuse_models,
            skip_dq_checks=args.skip_dq_checks,
            allow_dq_warnings=args.allow_dq_warnings,
            preloaded_data=preloaded_data,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            global_bin_edges=global_bin_edges,
            supersegment_bin_edges=supersegment_bin_edges,
            cutoff_ordering_mode=cutoff_ordering_mode,
            run_id=run_id,
            run_ts_iso=run_ts_iso,
        )

    # Print summary
    print_summary(results)

    _post_run_outputs(args, base_config, segments, all_supersegments, preloaded_data, results)

    # Cleanup global log sink
    if global_sink_id is not None:
        _safe_remove_sink(global_sink_id)

    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
