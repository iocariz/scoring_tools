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
import copy
import re
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.config import OutputPaths
from src.consolidation import generate_consolidation_report
from src.utils import resolve_modelling_supersegment, resolve_reporting_supersegment


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


def load_and_standardize_data(data_path: str) -> pd.DataFrame | None:
    """
    Load data from SAS file and standardize column names and categorical values.

    This function is called once at the batch level to avoid reloading the same
    data file for each segment. If loading fails (e.g., remote storage like MinIO),
    returns None and the pipeline will fall back to per-segment loading.

    Args:
        data_path: Path to the SAS data file.

    Returns:
        Standardized DataFrame ready for processing, or None if loading fails.
    """
    try:
        logger.info(f"Attempting to preload data from {data_path}...")
        data = pd.read_sas(data_path, format="sas7bdat", encoding="utf-8")
        logger.info(f"Data loaded: {data.shape[0]:,} rows × {data.shape[1]} columns")

        # Standardize column names
        logger.info("Standardizing column names...")
        data.columns = data.columns.str.lower().str.replace(" ", "_")

        # Standardize categorical values
        logger.info("Standardizing categorical values...")
        for col in data.select_dtypes(include=["object", "category", "string"]).columns:
            data[col] = data[col].astype("string").str.lower().str.replace(" ", "_").astype("category")

        logger.info("Data standardization complete.")
        return data

    except FileNotFoundError:
        logger.warning(f"Data file not found at {data_path}. Each segment will load data individually.")
        return None
    except Exception as e:
        logger.warning(f"Could not preload data: {e}. Each segment will load data individually.")
        return None


def learn_global_bin_edges(data: pd.DataFrame, base_config: dict[str, Any]) -> dict[str, list[float]]:
    """Learn bin edges on the full dataset (all segments) for consistency.

    When bins use ``max_bins`` instead of fixed ``bin_edges``, edges are learned
    (unsupervised quantiles) from data.  In batch mode we want a single set of
    edges shared across all segments, so we learn them here on the full demand
    population (all statuses, date-filtered to the observation window) before
    any segment run.  ``method = "optimization"`` is deprecated and falls back
    to quantile (see :func:`src.preprocess_improved.learn_optimization_bins`).

    Parameters
    ----------
    data : pd.DataFrame
        Full pre-loaded and standardized dataset (all segments).
    base_config : dict
        Base ``[preprocessing]`` configuration dictionary.

    Returns
    -------
    dict[str, list[float]]
        Mapping of variable name → learned bin edges.  Empty if no bins
        require learning.
    """
    from src.preprocess_improved import filter_by_date, learn_quantile_bins

    bins_config = base_config.get("bins", {})
    if not bins_config:
        return {}

    # Basic quality filters (NO segment filter).  Learn on the DEMAND population
    # (all statuses) the bins are applied to — not booked-only, which is a
    # selected, risk-truncated subset.  Date-filter to the observation window to
    # avoid look-ahead bias.
    mask = (
        (data["fuera_norma"] == "n")
        & (data["fraud_flag"] == "n")
        & (data["nature_holder"] != "legal")
    )
    data_demand = data[mask]
    date_ini = base_config.get("date_ini_book_obs")
    date_fin = base_config.get("date_fin_book_obs")
    if date_ini and date_fin:
        data_demand = filter_by_date(data_demand, "mis_date", date_ini, date_fin)
    logger.info(f"Global bin learning: {len(data_demand):,} demand records across all segments")

    global_edges: dict[str, list[float]] = {}
    for var_name, bc_raw in bins_config.items():
        has_edges = bc_raw.get("bin_edges") and len(bc_raw["bin_edges"]) >= 2
        if has_edges or bc_raw.get("max_bins") is None:
            continue  # Fixed edges or no learning needed

        source_col = bc_raw["source_col"]
        method = bc_raw.get("method", "quantile")
        max_bins = bc_raw["max_bins"]

        if method == "optimization":
            logger.warning(
                f"Bin method 'optimization' for '{var_name}' is deprecated (target leakage); "
                f"falling back to quantile splits."
            )
        edges = learn_quantile_bins(data_demand, source_col=source_col, max_bins=max_bins)

        global_edges[var_name] = edges
        logger.info(f"Global bin edges for '{var_name}': {edges}")

    return global_edges


def learn_supersegment_bin_edges(
    data: pd.DataFrame,
    base_config: dict[str, Any],
    supersegments: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Learn bin edges per reporting supersegment population.

    For each reporting supersegment, filters the demand data (all statuses,
    date-filtered to the observation window) to only the segments in that
    supersegment and learns bin edges from that subpopulation.  This allows
    different supersegments to have bin splits tuned to their own data
    distribution.

    Parameters
    ----------
    data : pd.DataFrame
        Full pre-loaded and standardized dataset (all segments).
    base_config : dict
        Base ``[preprocessing]`` configuration dictionary.
    supersegments : dict
        Mapping of supersegment name → config with ``segment_filters``.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Mapping of supersegment name → {variable name → learned bin edges}.
    """
    from src.preprocess_improved import filter_by_date, learn_quantile_bins

    bins_config = base_config.get("bins", {})
    if not bins_config:
        return {}

    # Only learn for bins that need it (no fixed bin_edges, have max_bins)
    learnable = {
        var_name: bc_raw
        for var_name, bc_raw in bins_config.items()
        if not (bc_raw.get("bin_edges") and len(bc_raw["bin_edges"]) >= 2) and bc_raw.get("max_bins") is not None
    }
    if not learnable:
        return {}

    # Basic quality filters (no segment filter yet).  Learn on the DEMAND
    # population (all statuses) the bins are applied to — not booked-only.
    quality_mask = (
        (data["fuera_norma"] == "n")
        & (data["fraud_flag"] == "n")
        & (data["nature_holder"] != "legal")
    )
    date_ini = base_config.get("date_ini_book_obs")
    date_fin = base_config.get("date_fin_book_obs")

    result: dict[str, dict[str, list[float]]] = {}
    for ss_name, ss_config in supersegments.items():
        # Check for fixed bin_edges first — these take priority over learning.
        fixed_edges = ss_config.get("bin_edges", {})
        if fixed_edges:
            ss_edges: dict[str, list[float]] = {}
            for var_name, edges in fixed_edges.items():
                if isinstance(edges, list) and len(edges) >= 2:
                    ss_edges[var_name] = edges
                    logger.info(f"Supersegment '{ss_name}' using fixed bin edges for '{var_name}': {edges}")
            if ss_edges:
                result[ss_name] = ss_edges
            continue  # Fixed edges provided — skip learning for this supersegment

        if not ss_config.get("learn_own_bin_edges", False):
            continue  # Use global edges (default)

        segment_filters = ss_config.get("segment_filters", [])
        if not segment_filters:
            continue

        # Filter to this supersegment's population
        pattern = "|".join(re.escape(sf) for sf in segment_filters)
        ss_mask = quality_mask & data["segment_cut_off"].str.contains(pattern, regex=True, na=False)
        ss_demand = data[ss_mask]
        if date_ini and date_fin:
            ss_demand = filter_by_date(ss_demand, "mis_date", date_ini, date_fin)

        if len(ss_demand) == 0:
            logger.warning(f"Supersegment '{ss_name}': no demand records for bin learning")
            continue

        ss_edges = {}
        for var_name, bc_raw in learnable.items():
            source_col = bc_raw["source_col"]
            method = bc_raw.get("method", "quantile")
            max_bins = bc_raw["max_bins"]

            try:
                if method == "optimization":
                    logger.warning(
                        f"Bin method 'optimization' for '{var_name}' is deprecated (target leakage); "
                        f"falling back to quantile splits."
                    )
                edges = learn_quantile_bins(ss_demand, source_col=source_col, max_bins=max_bins)
                ss_edges[var_name] = edges
                logger.info(f"Supersegment '{ss_name}' bin edges for '{var_name}': {edges}")
            except (ValueError, KeyError) as e:
                logger.warning(f"Supersegment '{ss_name}' bin learning failed for '{var_name}': {e}")

        if ss_edges:
            result[ss_name] = ss_edges

    return result


def load_base_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load the base configuration from config.toml."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("preprocessing", {})


_VALID_SEGMENT_NAME = re.compile(r"^[a-zA-Z0-9_.\-]+$")


def validate_segment_name(name: str) -> None:
    """Validate segment/supersegment name to prevent path traversal."""
    if not _VALID_SEGMENT_NAME.match(name):
        raise ValueError(
            f"Invalid segment name {name!r}: must match [a-zA-Z0-9_.-]. "
            f"Path separators and special characters are not allowed."
        )


def load_segments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load segment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("segments", {})


def load_supersegments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load supersegment configurations from segments.toml.

    Merges names from ``supersegments``, ``modelling_supersegments``, and
    ``reporting_supersegments`` so that validation sees all defined supersegments.
    """
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    merged: dict[str, dict[str, Any]] = {}
    for key in ("supersegments", "modelling_supersegments", "reporting_supersegments"):
        merged.update(config.get(key, {}))
    return merged


def load_reporting_supersegments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load only reporting supersegment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("reporting_supersegments", {})


def create_output_directories(base_output_dir: Path) -> dict[str, Path]:
    """Create output directory structure for a segment."""
    dirs = {
        "root": base_output_dir,
        "images": base_output_dir / "images",
        "models": base_output_dir / "models",
        "data": base_output_dir / "data",
        "logs": base_output_dir / "logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def merge_configs(base_config: dict[str, Any], segment_config: dict[str, Any]) -> dict[str, Any]:
    """Merge base config with segment-specific overrides (recursive for nested dicts)."""
    merged = copy.deepcopy(base_config)
    for key, value in segment_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def run_segment_pipeline(
    segment_name: str,
    segment_config: dict[str, Any],
    base_config: dict[str, Any],
    output_base: str = "output",
    model_path: str | None = None,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    floor_cells_path: str | None = None,
    floor_cells_mode: str = "floor",
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

    # Inject learned bin edges: start with global, then override with
    # supersegment-specific edges if this segment belongs to one.
    if global_bin_edges:
        bins_section = merged_config.setdefault("bins", {})
        for var_name, edges in global_bin_edges.items():
            if var_name in bins_section:
                bins_section[var_name]["bin_edges"] = edges

    if supersegment_bin_edges:
        reporting_ss = resolve_reporting_supersegment(segment_config)
        if reporting_ss and reporting_ss in supersegment_bin_edges:
            ss_edges = supersegment_bin_edges[reporting_ss]
            bins_section = merged_config.setdefault("bins", {})
            for var_name, edges in ss_edges.items():
                if var_name in bins_section:
                    bins_section[var_name]["bin_edges"] = edges
                    logger.info(f"Using supersegment '{reporting_ss}' bin edges for '{var_name}' (overriding global)")

    logger.info(f"Segment filter: {merged_config.get('segment_filter')}")
    logger.info(f"Optimum risk: {merged_config.get('optimum_risk')}")
    if model_path:
        logger.info(f"Using pre-trained model from: {model_path}")

    # Pass an explicit OutputPaths rooted at the segment's output directory
    # instead of os.chdir'ing the whole process (todo #67). All pipeline
    # phases thread `output` through, so every write lands under dirs["root"]
    # without touching process-global cwd. Safe for ThreadPoolExecutor too.
    try:
        from main import main as run_main_pipeline

        temp_config = write_temp_config(merged_config, dirs["root"]).resolve()
        resolved_model_path = str(Path(model_path).resolve()) if model_path else None
        segment_output = OutputPaths(base_dir=dirs["root"].resolve())

        result = run_main_pipeline(
            config_path=str(temp_config),
            model_path=resolved_model_path,
            skip_dq_checks=skip_dq_checks,
            preloaded_data=preloaded_data,
            training_only=training_only,
            baseline_mode=baseline_mode,
            base_scenario_only=base_scenario_only,
            floor_cells_path=floor_cells_path,
            floor_cells_mode=floor_cells_mode,
            output=segment_output,
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
    preloaded_data: pd.DataFrame = None,
    global_bin_edges: dict[str, list[float]] | None = None,
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
        from main import main as run_main_pipeline

        temp_config = write_temp_config(merged_config, dirs["root"]).resolve()
        # Explicit OutputPaths instead of os.chdir (todo #67).
        supersegment_output = OutputPaths(base_dir=dirs["root"].resolve())
        result = run_main_pipeline(
            config_path=str(temp_config),
            training_only=True,
            skip_dq_checks=skip_dq_checks,
            preloaded_data=preloaded_data,
            output=supersegment_output,
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


def write_temp_config(config: dict[str, Any], output_dir: Path) -> Path:
    """Write a temporary config file for this segment run."""
    import tomli_w

    config_path = output_dir / "config_segment.toml"

    # Keep data source stable if process working directory changes.
    config_for_dump = config.copy()
    data_path = config_for_dump.get("data_path")
    if isinstance(data_path, str):
        data_path_obj = Path(data_path)
        if not data_path_obj.is_absolute():
            config_for_dump["data_path"] = str((Path.cwd() / data_path_obj).resolve())

    # Wrap config in preprocessing section
    full_config = {"preprocessing": config_for_dump}

    with open(config_path, "wb") as f:
        tomli_w.dump(full_config, f)

    return config_path


# Removed in R2 (todo #67): _working_directory context manager.
# It used os.chdir to isolate each segment's output under dirs["root"].
# All pipeline phases now accept an explicit `output: OutputPaths`
# parameter that threads through every write, so process-global cwd
# manipulation is no longer needed. Unblocks ThreadPoolExecutor usage.


def _topological_sort_segments(
    segments: dict[str, dict[str, Any]],
) -> list[str]:
    """Sort segment names respecting cutoff_floor_segment dependencies.

    Segments with no dependency come first. Segments that depend on another
    come after their dependency. Raises ValueError on circular dependencies.

    Args:
        segments: Segment configurations keyed by segment name.

    Returns:
        Ordered list of segment names.
    """
    # Build dependency map: seg_name -> floor_segment_name or None
    deps: dict[str, str | None] = {}
    for seg_name, seg_config in segments.items():
        deps[seg_name] = seg_config.get("cutoff_floor_segment")

    ordered: list[str] = []
    resolved: set[str] = set()
    visiting: set[str] = set()

    def _visit(name: str) -> None:
        if name in resolved:
            return
        if name in visiting:
            raise ValueError(f"Circular cutoff_floor_segment dependency involving '{name}'")
        if name not in deps:
            # Dependency references a segment not in the current batch — treat as resolved
            resolved.add(name)
            return
        visiting.add(name)
        dep = deps[name]
        if dep is not None:
            _visit(dep)
        visiting.discard(name)
        resolved.add(name)
        ordered.append(name)

    for seg_name in segments:
        _visit(seg_name)

    return ordered


def run_segments_sequential(
    segments: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
    output_base: str = "output",
    supersegments: dict[str, dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    cutoff_ordering_mode: str = "bottom_up",
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
                    preloaded_data=preloaded_data,
                    global_bin_edges=global_bin_edges,
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

                floor_path = Path(output_base) / constraint_source / "data" / "accepted_cells_base.csv"
                if floor_path.exists():
                    floor_cells_path = str(floor_path.resolve())
                    logger.info(f"[{segment_name}] Using {direction} constraint from segment '{constraint_source}'")
                else:
                    logger.warning(
                        f"[{segment_name}] Constraint source '{constraint_source}' "
                        f"accepted cells file not found: {floor_path}"
                    )

            success = run_segment_pipeline(
                segment_name,
                segment_config,
                base_config,
                output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks,
                preloaded_data=preloaded_data,
                training_only=training_only,
                baseline_mode=baseline_mode,
                base_scenario_only=base_scenario_only,
                global_bin_edges=global_bin_edges,
                supersegment_bin_edges=supersegment_bin_edges,
                floor_cells_path=floor_cells_path,
                floor_cells_mode=floor_cells_mode,
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
    preloaded_data: pd.DataFrame = None,
    training_only: bool = False,
    baseline_mode: bool = False,
    base_scenario_only: bool = False,
    global_bin_edges: dict[str, list[float]] | None = None,
    supersegment_bin_edges: dict[str, dict[str, list[float]]] | None = None,
    cutoff_ordering_mode: str = "bottom_up",
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
                    preloaded_data=preloaded_data,
                    global_bin_edges=global_bin_edges,
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

                future = executor.submit(
                    run_segment_pipeline,
                    segment_name,
                    segment_config,
                    base_config,
                    output_base,
                    model_path,
                    skip_dq_checks,
                    preloaded_data,
                    training_only,
                    baseline_mode,
                    global_bin_edges,
                    supersegment_bin_edges,
                    None,  # floor_cells_path (unconstrained)
                    "floor",  # floor_cells_mode
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
            reverse_deps: dict[str, str] = {}
            for seg_name, seg_config in constrained.items():
                floor_seg = seg_config.get("cutoff_floor_segment")
                if floor_seg:
                    reverse_deps[floor_seg] = seg_name

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
                constraint_source = reverse_deps.get(segment_name)
            else:
                constraint_source = segment_config.get("cutoff_floor_segment")

            if constraint_source:
                # Skip segment if its constraint source failed
                if constraint_source in results and not results[constraint_source]:
                    logger.error(f"[{segment_name}] Skipping: constraint source '{constraint_source}' failed")
                    results[segment_name] = False
                    continue

                floor_path = Path(output_base) / constraint_source / "data" / "accepted_cells_base.csv"
                if floor_path.exists():
                    floor_cells_path = str(floor_path.resolve())
                    logger.info(f"[{segment_name}] Using {direction} constraint from segment '{constraint_source}'")
                else:
                    logger.warning(
                        f"[{segment_name}] Constraint source '{constraint_source}' "
                        f"accepted cells file not found: {floor_path}"
                    )

            success = run_segment_pipeline(
                segment_name,
                segment_config,
                base_config,
                output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks,
                preloaded_data=preloaded_data,
                training_only=training_only,
                baseline_mode=baseline_mode,
                base_scenario_only=base_scenario_only,
                global_bin_edges=global_bin_edges,
                supersegment_bin_edges=supersegment_bin_edges,
                floor_cells_path=floor_cells_path,
                floor_cells_mode=floor_cells_mode,
            )
            results[segment_name] = success

    return results


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
        from main import run_resimulation

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

        # Generate consolidated report
        try:
            consolidated_df, _ = generate_consolidation_report(
                output_base=args.output,
                segments=segments,
                supersegments=all_supersegments,
                output_path=args.output,
                multiplier=base_config.get("multiplier", 7),
                multiplier_h3=base_config.get("multiplier_h3", 4),
            )
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

        if failed_segments:
            print(f"\nFailed segments: {failed_segments}")
            return 1
        print(f"\nResimulation complete for {len(segments_to_run)} segment(s)")
        return 0

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
    data_path = base_config.get("data_path", "data/demanda_direct_out.sas7bdat")
    preloaded_data = load_and_standardize_data(data_path)

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
    if preloaded_data is not None:
        global_bin_edges = learn_global_bin_edges(preloaded_data, base_config)

        # Learn per-reporting-supersegment bin edges so each supersegment can
        # have splits tuned to its own population (e.g., different income_bin
        # edges for "others" vs "known").
        reporting_ss = load_reporting_supersegments_config(args.segments_config)
        if reporting_ss:
            supersegment_bin_edges = learn_supersegment_bin_edges(preloaded_data, base_config, reporting_ss)

    # Resolve cutoff ordering mode: CLI flag overrides config.toml
    cutoff_ordering_mode = args.cutoff_ordering_mode or base_config.get("cutoff_ordering_mode", "bottom_up")

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
            preloaded_data=preloaded_data,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            global_bin_edges=global_bin_edges,
            supersegment_bin_edges=supersegment_bin_edges,
            cutoff_ordering_mode=cutoff_ordering_mode,
        )
    else:
        results = run_segments_sequential(
            segments,
            base_config,
            args.output,
            supersegments=all_supersegments,
            reuse_models=args.reuse_models,
            skip_dq_checks=args.skip_dq_checks,
            preloaded_data=preloaded_data,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            global_bin_edges=global_bin_edges,
            supersegment_bin_edges=supersegment_bin_edges,
            cutoff_ordering_mode=cutoff_ordering_mode,
        )

    # Print summary
    print_summary(results)

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
            from run_score_metrics import generate_score_discriminance_report

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

    # Cleanup global log sink
    if global_sink_id is not None:
        _safe_remove_sink(global_sink_id)

    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
