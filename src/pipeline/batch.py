"""Multi-segment execution: supersegment model training, per-segment pipeline runs (sequential or parallel,
honouring cutoff-ordering dependencies) and output-directory cleanup. CLI lives in ``run_batch.py``.
"""

import copy
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.core.config import OutputPaths
from src.core.utils import resolve_modelling_supersegment
from src.pipeline.batch_config import (
    _inject_learned_bin_edges,
    _topological_sort_segments,
    create_output_directories,
    merge_configs,
    validate_segment_name,
    write_segment_config,
)
from src.pipeline.segment_runner import run_pipeline


class SegmentPipelineError(RuntimeError):
    """Raised when a segment pipeline execution fails."""


class SupersegmentTrainingError(RuntimeError):
    """Raised when supersegment model training fails."""


def _safe_remove_sink(sink_id: int) -> None:
    """Remove a loguru sink if it still exists."""
    try:
        logger.remove(sink_id)
    except ValueError:
        # Sink may already be removed by nested logger configuration.
        pass


def run_segment_pipeline(
    segment_name: str,
    segment_config: dict[str, Any],
    base_config: dict[str, Any],
    output_base: str = "output",
    model_path: str | None = None,
    skip_dq_checks: bool = False,
    allow_dq_warnings: bool = False,
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    floor_cells_path: str | None = None,
    floor_cells_mode: str = "floor",
    run_id: str | None = None,
    run_ts_iso: str | None = None,
) -> bool:
    """
    Run the pipeline for a single segment.

    Args:
        segment_name: Name of the segment (used for output directory)
        segment_config: Segment-specific configuration overrides
        base_config: Base configuration from config.toml
        output_base: Base directory for all outputs
        model_path: Optional path to a pre-trained model. If provided, skips
                   inference training and loads the existing model (used for
                   supersegment workflows).
        skip_dq_checks: If True, skip data quality checks.
        preloaded_data: Optional pre-loaded and standardized DataFrame. If provided,
                       skips loading data from file for faster batch processing.
        training_only: Optional parameter. If True, skipping optimization and scenario generation.
        baseline_mode: If True, show current portfolio as-is (no cutoff optimization).
        global_bin_edges: Optional pre-learned bin edges from the full dataset.
                         When provided, these are injected into the merged config so
                         that all segments share consistent bin edges.
        supersegment_bin_edges: Optional per-reporting-supersegment bin edges.
                               When a segment belongs to a reporting supersegment
                               that has its own learned edges, those override the
                               global edges for the matching variables.
        floor_cells_path: Optional path to accepted cells CSV from a previous
                         segment (sequential cutoff ordering).
        floor_cells_mode: ``"floor"`` (bottom-up) or ``"ceiling"`` (top-down).

    Returns:
        True if successful, False otherwise
    """
    validate_segment_name(segment_name)

    # Create output directories
    output_dir = Path(output_base) / segment_name
    dirs = create_output_directories(output_dir)

    # Setup logging for this segment
    log_file = dirs["logs"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sink_id = logger.add(log_file, rotation="10 MB", level="DEBUG")

    logger.info("=" * 80)
    logger.info(f"PROCESSING SEGMENT: {segment_name}")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Merge configurations
    merged_config = merge_configs(base_config, segment_config)

    # Inject learned bin edges: global first, then supersegment overrides — but a segment that
    # explicitly pins bin_edges keeps them (#60). See _inject_learned_bin_edges for precedence.
    merged_config = _inject_learned_bin_edges(
        merged_config, segment_config, segment_name, global_bin_edges, supersegment_bin_edges
    )

    logger.info(f"Segment filter: {merged_config.get('segment_filter')}")
    logger.info(f"Optimum risk: {merged_config.get('optimum_risk')}")
    if model_path:
        logger.info(f"Using pre-trained model from: {model_path}")

    # Pass an explicit OutputPaths rooted at the segment's output directory
    # instead of os.chdir'ing the whole process (todo #67). All pipeline
    # phases thread `output` through, so every write lands under dirs["root"]
    # without touching process-global cwd. Safe for ThreadPoolExecutor too.
    try:
        temp_config = write_segment_config(merged_config, dirs["root"]).resolve()
        resolved_model_path = str(Path(model_path).resolve()) if model_path else None
        segment_output = OutputPaths(base_dir=dirs["root"].resolve())

        result = run_pipeline(
            config_path=str(temp_config),
            model_path=resolved_model_path,
            skip_dq_checks=skip_dq_checks,
            allow_dq_warnings=allow_dq_warnings,
            preloaded_data=preloaded_data,
            training_only=training_only,
            baseline_mode=baseline_mode,
            base_scenario_only=base_scenario_only,
            floor_cells_path=floor_cells_path,
            floor_cells_mode=floor_cells_mode,
            output=segment_output,
            run_id=run_id,
            run_ts_iso=run_ts_iso,
        )

        if result is None:
            raise SegmentPipelineError(f"Pipeline returned no result for segment: {segment_name}")

        logger.info(f"Pipeline completed successfully for segment: {segment_name}")
        return True

    except SegmentPipelineError:
        logger.exception(f"Error processing segment {segment_name}")
        return False
    except Exception as e:
        error = SegmentPipelineError(f"Unexpected segment error: {segment_name}")
        logger.error(f"{error}: {e}")
        logger.exception("Full traceback:")
        return False
    finally:
        _safe_remove_sink(sink_id)


def run_supersegment_training(
    supersegment_name: str,
    supersegment_config: dict[str, Any],
    base_config: dict[str, Any],
    output_base: str = "output",
    skip_dq_checks: bool = False,
    allow_dq_warnings: bool = False,
    preloaded_data: pd.DataFrame = None,
    global_bin_edges: dict[str, list[float]] | None = None,
    run_id: str | None = None,
    run_ts_iso: str | None = None,
) -> str | None:
    """
    Train a model on combined supersegment data (multiple segment_filters).

    This function:
    1. Creates a combined segment_filter using regex OR pattern
    2. Trains the inference model on the combined population
    3. Saves the model for use by individual segment optimizations

    Args:
        supersegment_name: Name of the supersegment
        supersegment_config: Config containing list of segment_filters to combine
        base_config: Base configuration from config.toml
        output_base: Base directory for all outputs
        skip_dq_checks: If True, skip data quality checks.
        preloaded_data: Optional pre-loaded DataFrame.
        global_bin_edges: Optional pre-learned bin edges from the full dataset.

    Returns:
        Path to the trained model directory, or None if training failed
    """
    validate_segment_name(supersegment_name)

    # Create output directories for supersegment
    output_dir = Path(output_base) / f"_supersegment_{supersegment_name}"
    dirs = create_output_directories(output_dir)

    # Setup logging
    log_file = dirs["logs"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    sink_id = logger.add(log_file, rotation="10 MB", level="DEBUG")

    logger.info("=" * 80)
    logger.info(f"TRAINING SUPERSEGMENT MODEL: {supersegment_name}")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Get the list of segment_filters to combine
    segment_filters = supersegment_config.get("segment_filters", [])
    if not segment_filters:
        logger.error(f"No segment_filters defined for supersegment: {supersegment_name}")
        return None

    logger.info(f"Combining {len(segment_filters)} segment filters:")
    for sf in segment_filters:
        logger.info(f"  - {sf}")

    # Create combined segment_filter using regex OR pattern
    # Escape special regex characters in segment filter values
    escaped_filters = [re.escape(sf) for sf in segment_filters]
    combined_filter = "|".join(f"({sf})" for sf in escaped_filters)

    # Merge config with combined filter (deep copy to avoid mutating base_config)
    merged_config = copy.deepcopy(base_config)
    merged_config["segment_filter"] = combined_filter

    # Inject globally-learned bin edges so supersegment uses the same edges as segments.
    if global_bin_edges:
        bins_section = merged_config.setdefault("bins", {})
        for var_name, edges in global_bin_edges.items():
            if var_name in bins_section:
                bins_section[var_name]["bin_edges"] = edges

    try:
        temp_config = write_segment_config(merged_config, dirs["root"]).resolve()
        # Explicit OutputPaths instead of os.chdir (todo #67).
        supersegment_output = OutputPaths(base_dir=dirs["root"].resolve())
        result = run_pipeline(
            config_path=str(temp_config),
            training_only=True,
            skip_dq_checks=skip_dq_checks,
            allow_dq_warnings=allow_dq_warnings,
            preloaded_data=preloaded_data,
            output=supersegment_output,
            run_id=run_id,
            run_ts_iso=run_ts_iso,
        )

        if result is None:
            raise SupersegmentTrainingError(f"Supersegment training returned no result: {supersegment_name}")

        # Find the most recent model directory
        models_dir = dirs["models"]
        model_dirs = sorted(models_dir.glob("model_*"), reverse=True)

        if not model_dirs:
            raise SupersegmentTrainingError(f"No model directory found after training: {supersegment_name}")

        model_path = str(model_dirs[0].resolve())
        logger.info(f"Supersegment model trained successfully: {model_path}")
        return model_path

    except SupersegmentTrainingError:
        logger.exception(f"Error training supersegment {supersegment_name}")
        return None
    except Exception as e:
        error = SupersegmentTrainingError(f"Unexpected supersegment training error: {supersegment_name}")
        logger.error(f"{error}: {e}")
        logger.exception("Full traceback:")
        return None
    finally:
        _safe_remove_sink(sink_id)


def run_segments_sequential(
    segments: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
    output_base: str = "output",
    supersegments: dict[str, dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False,
    allow_dq_warnings: bool = False,
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    cutoff_ordering_mode: str = "bottom_up",
    run_id: str | None = None,
    run_ts_iso: str | None = None,
) -> dict[str, bool]:
    """
    Run all segments sequentially, with supersegment support.

    If segments reference a supersegment, the supersegment model is trained first
    and then reused for all segments in that supersegment.

    Args:
        segments: Segment configurations
        base_config: Base configuration
        output_base: Output directory base
        supersegments: Optional supersegment configurations
        reuse_models: If True, reuse existing supersegment models instead of retraining
        skip_dq_checks: If True, skip data quality checks.
        preloaded_data: Optional pre-loaded DataFrame to avoid reloading for each segment.
        global_bin_edges: Optional pre-learned bin edges from the full dataset.

    Returns:
        Dictionary of segment names to success status
    """
    results = {}
    supersegment_models = {}  # Cache: supersegment_name -> model_path

    # Phase 1: Train supersegment models (or reuse existing)
    if supersegments:
        # Find which modelling supersegments are actually used by the selected segments
        used_supersegments = []
        for segment_config in segments.values():
            ss = resolve_modelling_supersegment(segment_config)
            if ss and ss in supersegments and ss not in used_supersegments:
                used_supersegments.append(ss)

        # Train or reuse each supersegment with progress bar
        if used_supersegments:
            print(f"\n{'=' * 60}")
            print("PHASE 1: Training Supersegment Models")
            print(f"{'=' * 60}")

            ss_progress = tqdm(
                used_supersegments,
                desc="Supersegments",
                unit="model",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            for ss_name in ss_progress:
                ss_progress.set_postfix_str(ss_name)

                # Check for existing model if reuse_models is enabled
                if reuse_models:
                    ss_output_dir = Path(output_base) / f"_supersegment_{ss_name}" / "models"
                    if ss_output_dir.exists():
                        model_dirs = sorted(ss_output_dir.glob("model_*"), reverse=True)
                        if model_dirs:
                            model_path = str(model_dirs[0])
                            logger.info(f"Reusing existing supersegment model: {model_path}")
                            supersegment_models[ss_name] = model_path
                            continue

                # Train new model
                logger.info(f"Training supersegment model: {ss_name}")
                model_path = run_supersegment_training(
                    supersegment_name=ss_name,
                    supersegment_config=supersegments[ss_name],
                    base_config=base_config,
                    output_base=output_base,
                    skip_dq_checks=skip_dq_checks,
                    allow_dq_warnings=allow_dq_warnings,
                    preloaded_data=preloaded_data,
                    global_bin_edges=global_bin_edges,
                    run_id=run_id,
                    run_ts_iso=run_ts_iso,
                )

                if model_path:
                    supersegment_models[ss_name] = model_path
                    logger.info(f"Supersegment {ss_name} model ready: {model_path}")
                else:
                    logger.error(f"Failed to train supersegment {ss_name}")
                    # Mark all segments using this modelling supersegment as failed
                    for seg_name, seg_config in segments.items():
                        if resolve_modelling_supersegment(seg_config) == ss_name:
                            results[seg_name] = False
                            logger.error(f"Segment {seg_name} marked as failed (supersegment training failed)")

    # Phase 2: Run individual segment optimizations
    # Filter segments that haven't already failed
    failed_segments = set(results.keys())
    segments_to_run_dict = {name: config for name, config in segments.items() if name not in failed_segments}

    # Sort by cutoff_floor_segment dependencies (topological order)
    try:
        ordered_names = _topological_sort_segments(segments_to_run_dict)
    except ValueError as e:
        logger.error(f"Segment ordering failed: {e}")
        for name in segments_to_run_dict:
            results[name] = False
        return results

    # For top-down mode: reverse order and build reverse dependency map
    is_top_down = cutoff_ordering_mode == "top_down"
    if is_top_down:
        ordered_names = list(reversed(ordered_names))
        # Reverse deps: for each segment, find all segments that depend on it (are less restrictive).
        # Multiple segments can share the same cutoff_floor_segment, so map to a list.
        reverse_deps: dict[str, list[str]] = {}
        for seg_name, seg_config in segments_to_run_dict.items():
            floor_seg = seg_config.get("cutoff_floor_segment")
            if floor_seg:
                reverse_deps.setdefault(floor_seg, []).append(seg_name)

    segments_to_run = [(name, segments_to_run_dict[name]) for name in ordered_names]
    floor_cells_mode = "ceiling" if is_top_down else "floor"

    if segments_to_run:
        print(f"\n{'=' * 60}")
        direction = "top-down" if is_top_down else "bottom-up"
        print(f"PHASE 2: Running Segment Optimizations ({direction})")
        print(f"{'=' * 60}")

        seg_progress = tqdm(
            segments_to_run,
            desc="Segments",
            unit="segment",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for segment_name, segment_config in seg_progress:
            seg_progress.set_postfix_str(segment_name)

            # Check if this segment uses a modelling supersegment model
            modelling_ss = resolve_modelling_supersegment(segment_config)
            model_path = None
            if modelling_ss and modelling_ss in supersegment_models:
                model_path = supersegment_models[modelling_ss]
                logger.info(f"Using supersegment model: {modelling_ss}")

            # Resolve floor_cells_path based on ordering mode
            floor_cells_path = None
            if is_top_down:
                # Top-down: constraint comes from the LESS restrictive segment
                # (the segment that lists us as its cutoff_floor_segment).
                # If multiple segments depend on us, use the first one in
                # execution order (already processed, most restrictive ceiling).
                dep_list = reverse_deps.get(segment_name, [])
                if len(dep_list) > 1:
                    logger.warning(
                        f"[{segment_name}] Multiple segments depend on this segment "
                        f"as cutoff_floor_segment: {dep_list}. Using '{dep_list[0]}' as "
                        f"ceiling constraint source."
                    )
                constraint_source = dep_list[0] if dep_list else None
            else:
                # Bottom-up: constraint comes from the MORE restrictive segment
                constraint_source = segment_config.get("cutoff_floor_segment")

            if constraint_source:
                # Skip segment if its constraint source failed
                if constraint_source in results and not results[constraint_source]:
                    logger.error(f"[{segment_name}] Skipping: constraint source '{constraint_source}' failed")
                    results[segment_name] = False
                    seg_progress.set_postfix_str(f"{segment_name} ✗ (dep failed)", refresh=True)
                    continue

                # #61: the source did not run in this batch — its accepted_cells
                # are from a PRIOR run and may be incoherent with the current
                # data/config/code, silently breaking the nested-mask guarantee.
                # Warn loudly but proceed (don't block legitimate incremental
                # re-runs); the file-missing case still hard-fails below (#36).
                if constraint_source not in results:
                    logger.warning(
                        f"[{segment_name}] Constraint source '{constraint_source}' did not run in this batch — "
                        f"using its accepted_cells from a PRIOR run (potential staleness). Re-run "
                        f"'{constraint_source}' in the same batch to guarantee a consistent nested mask."
                    )

                floor_path = Path(output_base) / constraint_source / "data" / "accepted_cells_base.csv"
                if floor_path.exists():
                    floor_cells_path = str(floor_path.resolve())
                    logger.info(f"[{segment_name}] Using {direction} constraint from segment '{constraint_source}'")
                else:
                    # Audit #36: running unconstrained would silently break the
                    # nested-mask guarantee (mask_dependent ⊆ mask_source) —
                    # fail the segment instead of degrading.
                    logger.error(
                        f"[{segment_name}] Constraint source '{constraint_source}' accepted cells "
                        f"file not found: {floor_path}. Refusing to run UNCONSTRAINED — the nested "
                        "cutoff ordering would be silently violated. Re-run the source segment first."
                    )
                    results[segment_name] = False
                    seg_progress.set_postfix_str(f"{segment_name} ✗ (no constraint artifact)", refresh=True)
                    continue

            success = run_segment_pipeline(
                segment_name,
                segment_config,
                base_config,
                output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks,
                allow_dq_warnings=allow_dq_warnings,
                preloaded_data=preloaded_data,
                training_only=training_only,
                baseline_mode=baseline_mode,
                base_scenario_only=base_scenario_only,
                global_bin_edges=global_bin_edges,
                supersegment_bin_edges=supersegment_bin_edges,
                floor_cells_path=floor_cells_path,
                floor_cells_mode=floor_cells_mode,
                run_id=run_id,
                run_ts_iso=run_ts_iso,
            )
            results[segment_name] = success

            # Update progress bar color based on result
            if success:
                seg_progress.set_postfix_str(f"{segment_name} ✓", refresh=True)
            else:
                seg_progress.set_postfix_str(f"{segment_name} ✗", refresh=True)

    return results


def run_segments_parallel(
    segments: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
    output_base: str = "output",
    max_workers: int = None,
    supersegments: dict[str, dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False,
    allow_dq_warnings: bool = False,
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    cutoff_ordering_mode: str = "bottom_up",
    run_id: str | None = None,
    run_ts_iso: str | None = None,
) -> dict[str, bool]:
    """
    Run all segments in parallel, with supersegment support.

    Note: Supersegment models are trained sequentially first, then
    individual segment optimizations run in parallel.

    Note: When using preloaded_data with parallel execution, each worker
    receives a copy of the data. For very large datasets, sequential
    execution may be more memory-efficient.

    Args:
        segments: Segment configurations
        base_config: Base configuration
        output_base: Output directory base
        max_workers: Maximum parallel workers
        supersegments: Optional supersegment configurations
        reuse_models: If True, reuse existing supersegment models instead of retraining
        skip_dq_checks: If True, skip data quality checks.
        preloaded_data: Optional pre-loaded DataFrame to avoid reloading for each segment.
        global_bin_edges: Optional pre-learned bin edges from the full dataset.

    Returns:
        Dictionary of segment names to success status
    """
    results = {}
    supersegment_models = {}

    # Phase 1: Train supersegment models SEQUENTIALLY (cannot parallelize training)
    if supersegments:
        used_supersegments = []
        for segment_config in segments.values():
            ss = resolve_modelling_supersegment(segment_config)
            if ss and ss in supersegments and ss not in used_supersegments:
                used_supersegments.append(ss)

        if used_supersegments:
            print(f"\n{'=' * 60}")
            print("PHASE 1: Training Supersegment Models (sequential)")
            print(f"{'=' * 60}")

            ss_progress = tqdm(
                used_supersegments,
                desc="Supersegments",
                unit="model",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            for ss_name in ss_progress:
                ss_progress.set_postfix_str(ss_name)

                # Check for existing model if reuse_models is enabled
                if reuse_models:
                    ss_output_dir = Path(output_base) / f"_supersegment_{ss_name}" / "models"
                    if ss_output_dir.exists():
                        model_dirs = sorted(ss_output_dir.glob("model_*"), reverse=True)
                        if model_dirs:
                            model_path = str(model_dirs[0])
                            logger.info(f"Reusing existing supersegment model: {model_path}")
                            supersegment_models[ss_name] = model_path
                            continue

                logger.info(f"Training supersegment model: {ss_name}")
                model_path = run_supersegment_training(
                    supersegment_name=ss_name,
                    supersegment_config=supersegments[ss_name],
                    base_config=base_config,
                    output_base=output_base,
                    skip_dq_checks=skip_dq_checks,
                    allow_dq_warnings=allow_dq_warnings,
                    preloaded_data=preloaded_data,
                    global_bin_edges=global_bin_edges,
                    run_id=run_id,
                    run_ts_iso=run_ts_iso,
                )
                if model_path:
                    supersegment_models[ss_name] = model_path
                else:
                    for seg_name, seg_config in segments.items():
                        if resolve_modelling_supersegment(seg_config) == ss_name:
                            results[seg_name] = False

    # Phase 2: Run individual segment optimizations IN PARALLEL
    segments_to_run = {name: config for name, config in segments.items() if name not in results}

    # Identify all segments involved in cutoff ordering chains
    # (both those with cutoff_floor_segment AND those referenced by it)
    in_chain: set[str] = set()
    for n, c in segments_to_run.items():
        floor_seg = c.get("cutoff_floor_segment")
        if floor_seg:
            in_chain.add(n)
            if floor_seg in segments_to_run:
                in_chain.add(floor_seg)
    unconstrained = {n: c for n, c in segments_to_run.items() if n not in in_chain}
    constrained = {n: c for n, c in segments_to_run.items() if n in in_chain}

    if constrained:
        logger.info(
            f"Segments with cutoff_floor_segment will run sequentially after parallel batch: {list(constrained.keys())}"
        )

    if unconstrained:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: Running Segment Optimizations (parallel, {max_workers or 'auto'} workers)")
        print(f"{'=' * 60}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for segment_name, segment_config in unconstrained.items():
                modelling_ss = resolve_modelling_supersegment(segment_config)
                model_path = supersegment_models.get(modelling_ss) if modelling_ss else None

                # Keyword args: robust to signature changes. (Previously positional,
                # which silently mis-mapped base_scenario_only/global_bin_edges/
                # floor_cells_* after base_scenario_only was inserted into the
                # signature — fixed here.)
                future = executor.submit(
                    run_segment_pipeline,
                    segment_name,
                    segment_config,
                    base_config,
                    output_base,
                    model_path=model_path,
                    skip_dq_checks=skip_dq_checks,
                    allow_dq_warnings=allow_dq_warnings,
                    preloaded_data=preloaded_data,
                    training_only=training_only,
                    baseline_mode=baseline_mode,
                    base_scenario_only=base_scenario_only,
                    global_bin_edges=global_bin_edges,
                    supersegment_bin_edges=supersegment_bin_edges,
                    floor_cells_path=None,  # unconstrained
                    floor_cells_mode="floor",
                    run_id=run_id,
                    run_ts_iso=run_ts_iso,
                )
                futures[future] = segment_name

            # Progress bar for parallel execution
            seg_progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Segments",
                unit="segment",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )

            for future in seg_progress:
                segment_name = futures[future]
                try:
                    success = future.result()
                    results[segment_name] = success
                    status = "✓" if success else "✗"
                    seg_progress.set_postfix_str(f"{segment_name} {status}", refresh=True)
                except Exception as e:
                    results[segment_name] = False
                    seg_progress.set_postfix_str(f"{segment_name} ✗", refresh=True)
                    logger.error(f"Segment {segment_name} raised exception: {e}")

    # Phase 2b: Run constrained segments sequentially (cutoff_floor_segment)
    if constrained:
        try:
            ordered_constrained = _topological_sort_segments(constrained)
        except ValueError as e:
            logger.error(f"Constrained segment ordering failed: {e}")
            for name in constrained:
                results[name] = False
            return results

        is_top_down = cutoff_ordering_mode == "top_down"
        if is_top_down:
            ordered_constrained = list(reversed(ordered_constrained))
            # dict[str, list[str]] like the sequential path — the old dict[str, str]
            # silently kept only the LAST dependent per floor segment, so the
            # ceiling source could differ between --parallel and sequential runs
            # of the same config (audit #36).
            reverse_deps: dict[str, list[str]] = {}
            for seg_name, seg_config in constrained.items():
                floor_seg = seg_config.get("cutoff_floor_segment")
                if floor_seg:
                    reverse_deps.setdefault(floor_seg, []).append(seg_name)

        direction = "top-down" if is_top_down else "bottom-up"
        floor_cells_mode = "ceiling" if is_top_down else "floor"

        print(f"\n{'=' * 60}")
        print(f"PHASE 2b: Running Constrained Segments (sequential, {direction})")
        print(f"{'=' * 60}")

        for segment_name in ordered_constrained:
            segment_config = constrained[segment_name]
            modelling_ss = resolve_modelling_supersegment(segment_config)
            model_path = supersegment_models.get(modelling_ss) if modelling_ss else None

            floor_cells_path = None
            if is_top_down:
                dep_list = reverse_deps.get(segment_name, [])
                if len(dep_list) > 1:
                    logger.warning(
                        f"[{segment_name}] Multiple segments depend on this segment "
                        f"as cutoff_floor_segment: {dep_list}. Using '{dep_list[0]}' as "
                        f"ceiling constraint source."
                    )
                constraint_source = dep_list[0] if dep_list else None
            else:
                constraint_source = segment_config.get("cutoff_floor_segment")

            if constraint_source:
                # Skip segment if its constraint source failed
                if constraint_source in results and not results[constraint_source]:
                    logger.error(f"[{segment_name}] Skipping: constraint source '{constraint_source}' failed")
                    results[segment_name] = False
                    continue

                # #61: source not run this batch → its accepted_cells are from a
                # PRIOR run (potential staleness). Warn but proceed; missing-file
                # still hard-fails below (#36). See the sequential path for detail.
                if constraint_source not in results:
                    logger.warning(
                        f"[{segment_name}] Constraint source '{constraint_source}' did not run in this batch — "
                        f"using its accepted_cells from a PRIOR run (potential staleness). Re-run "
                        f"'{constraint_source}' in the same batch to guarantee a consistent nested mask."
                    )

                floor_path = Path(output_base) / constraint_source / "data" / "accepted_cells_base.csv"
                if floor_path.exists():
                    floor_cells_path = str(floor_path.resolve())
                    logger.info(f"[{segment_name}] Using {direction} constraint from segment '{constraint_source}'")
                else:
                    # Audit #36: running unconstrained would silently break the
                    # nested-mask guarantee — fail the segment instead.
                    logger.error(
                        f"[{segment_name}] Constraint source '{constraint_source}' accepted cells "
                        f"file not found: {floor_path}. Refusing to run UNCONSTRAINED — the nested "
                        "cutoff ordering would be silently violated. Re-run the source segment first."
                    )
                    results[segment_name] = False
                    continue

            success = run_segment_pipeline(
                segment_name,
                segment_config,
                base_config,
                output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks,
                allow_dq_warnings=allow_dq_warnings,
                preloaded_data=preloaded_data,
                training_only=training_only,
                baseline_mode=baseline_mode,
                base_scenario_only=base_scenario_only,
                global_bin_edges=global_bin_edges,
                supersegment_bin_edges=supersegment_bin_edges,
                floor_cells_path=floor_cells_path,
                floor_cells_mode=floor_cells_mode,
                run_id=run_id,
                run_ts_iso=run_ts_iso,
            )
            results[segment_name] = success

    return results


def clean_output_directories(
    segments: dict[str, dict[str, Any]], supersegments: dict[str, dict[str, Any]], output_base: str = "output"
) -> dict[str, bool]:
    """
    Remove output directories for specified segments and their supersegments.

    Args:
        segments: Segment configurations to clean
        supersegments: All supersegment configurations
        output_base: Base output directory

    Returns:
        Dictionary of directory names to removal success status
    """
    results = {}
    output_path = Path(output_base)

    # Find modelling supersegments used by these segments
    used_supersegments = set()
    for seg_config in segments.values():
        ss = resolve_modelling_supersegment(seg_config)
        if ss and ss in supersegments:
            used_supersegments.add(ss)

    # Clean supersegment directories
    for ss_name in used_supersegments:
        ss_dir = output_path / f"_supersegment_{ss_name}"
        if ss_dir.exists():
            try:
                shutil.rmtree(ss_dir)
                print(f"  Removed: {ss_dir}")
                results[f"_supersegment_{ss_name}"] = True
            except Exception as e:
                print(f"  Failed to remove {ss_dir}: {e}")
                results[f"_supersegment_{ss_name}"] = False
        else:
            results[f"_supersegment_{ss_name}"] = True  # Already clean

    # Clean segment directories
    for seg_name in segments:
        seg_dir = output_path / seg_name
        if seg_dir.exists():
            try:
                shutil.rmtree(seg_dir)
                print(f"  Removed: {seg_dir}")
                results[seg_name] = True
            except Exception as e:
                print(f"  Failed to remove {seg_dir}: {e}")
                results[seg_name] = False
        else:
            results[seg_name] = True  # Already clean

    return results
