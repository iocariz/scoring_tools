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
import os
import re
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from tqdm import tqdm

from src.consolidation import generate_consolidation_report


class SegmentPipelineError(RuntimeError):
    """Raised when a segment pipeline execution fails."""


class SupersegmentTrainingError(RuntimeError):
    """Raised when supersegment model training fails."""


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


def load_base_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load the base configuration from config.toml."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("preprocessing", {})


def load_segments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load segment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("segments", {})


def load_supersegments_config(segments_path: str = "segments.toml") -> dict[str, dict[str, Any]]:
    """Load supersegment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("supersegments", {})


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
    """Merge base config with segment-specific overrides."""
    merged = base_config.copy()
    merged.update(segment_config)
    return merged


def run_segment_pipeline(
    segment_name: str,
    segment_config: dict[str, Any],
    base_config: dict[str, Any],
    output_base: str = "output",
    model_path: str | None = None,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
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

    Returns:
        True if successful, False otherwise
    """
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
    logger.info(f"Segment filter: {merged_config.get('segment_filter')}")
    logger.info(f"Optimum risk: {merged_config.get('optimum_risk')}")
    if model_path:
        logger.info(f"Using pre-trained model from: {model_path}")

    # Run with segment output directory as cwd so all relative outputs are isolated.
    try:
        from main import main as run_main_pipeline

        temp_config = write_temp_config(merged_config, dirs["root"])
        resolved_model_path = str(Path(model_path).resolve()) if model_path else None

        with _working_directory(dirs["root"]):
            result = run_main_pipeline(
                config_path=str(temp_config),
                model_path=resolved_model_path,
                skip_dq_checks=skip_dq_checks,
                preloaded_data=preloaded_data,
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
        logger.remove(sink_id)


def run_supersegment_training(
    supersegment_name: str,
    supersegment_config: dict[str, Any],
    base_config: dict[str, Any],
    output_base: str = "output",
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
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

    Returns:
        Path to the trained model directory, or None if training failed
    """
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

    # Merge config with combined filter
    merged_config = base_config.copy()
    merged_config["segment_filter"] = combined_filter

    try:
        from main import main as run_main_pipeline

        temp_config = write_temp_config(merged_config, dirs["root"])
        with _working_directory(dirs["root"]):
            result = run_main_pipeline(
                config_path=str(temp_config),
                training_only=True,
                skip_dq_checks=skip_dq_checks,
                preloaded_data=preloaded_data,
            )

        if result is None:
            raise SupersegmentTrainingError(
                f"Supersegment training returned no result: {supersegment_name}"
            )

        # Find the most recent model directory
        models_dir = dirs["models"]
        model_dirs = sorted(models_dir.glob("model_*"), reverse=True)

        if not model_dirs:
            raise SupersegmentTrainingError(
                f"No model directory found after training: {supersegment_name}"
            )

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
        logger.remove(sink_id)


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


@contextmanager
def _working_directory(path: Path):
    """Temporarily switch process cwd."""
    original_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_cwd)


def run_segments_sequential(
    segments: dict[str, dict[str, Any]],
    base_config: dict[str, Any],
    output_base: str = "output",
    supersegments: dict[str, dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
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

    Returns:
        Dictionary of segment names to success status
    """
    results = {}
    supersegment_models = {}  # Cache: supersegment_name -> model_path

    # Phase 1: Train supersegment models (or reuse existing)
    if supersegments:
        # Find which supersegments are actually used by the selected segments
        used_supersegments = []
        for segment_config in segments.values():
            ss = segment_config.get("supersegment")
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
                )

                if model_path:
                    supersegment_models[ss_name] = model_path
                    logger.info(f"Supersegment {ss_name} model ready: {model_path}")
                else:
                    logger.error(f"Failed to train supersegment {ss_name}")
                    # Mark all segments using this supersegment as failed
                    for seg_name, seg_config in segments.items():
                        if seg_config.get("supersegment") == ss_name:
                            results[seg_name] = False
                            logger.error(f"Segment {seg_name} marked as failed (supersegment training failed)")

    # Phase 2: Run individual segment optimizations
    # Filter segments that haven't already failed
    segments_to_run = [(name, config) for name, config in segments.items() if name not in results]

    if segments_to_run:
        print(f"\n{'=' * 60}")
        print("PHASE 2: Running Segment Optimizations")
        print(f"{'=' * 60}")

        seg_progress = tqdm(
            segments_to_run,
            desc="Segments",
            unit="segment",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

        for segment_name, segment_config in seg_progress:
            seg_progress.set_postfix_str(segment_name)

            # Check if this segment uses a supersegment model
            supersegment = segment_config.get("supersegment")
            model_path = None
            if supersegment and supersegment in supersegment_models:
                model_path = supersegment_models[supersegment]
                logger.info(f"Using supersegment model: {supersegment}")

            success = run_segment_pipeline(
                segment_name,
                segment_config,
                base_config,
                output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks,
                preloaded_data=preloaded_data,
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

    Returns:
        Dictionary of segment names to success status
    """
    results = {}
    supersegment_models = {}

    # Phase 1: Train supersegment models SEQUENTIALLY (cannot parallelize training)
    if supersegments:
        used_supersegments = []
        for segment_config in segments.values():
            ss = segment_config.get("supersegment")
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
                )
                if model_path:
                    supersegment_models[ss_name] = model_path
                else:
                    for seg_name, seg_config in segments.items():
                        if seg_config.get("supersegment") == ss_name:
                            results[seg_name] = False

    # Phase 2: Run individual segment optimizations IN PARALLEL
    segments_to_run = {name: config for name, config in segments.items() if name not in results}

    if segments_to_run:
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: Running Segment Optimizations (parallel, {max_workers or 'auto'} workers)")
        print(f"{'=' * 60}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for segment_name, segment_config in segments_to_run.items():
                supersegment = segment_config.get("supersegment")
                model_path = supersegment_models.get(supersegment) if supersegment else None

                future = executor.submit(
                    run_segment_pipeline,
                    segment_name,
                    segment_config,
                    base_config,
                    output_base,
                    model_path,
                    skip_dq_checks,
                    preloaded_data,
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

    # Find supersegments used by these segments
    used_supersegments = set()
    for seg_config in segments.values():
        ss = seg_config.get("supersegment")
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
        supersegment = config.get("supersegment", None)
        print(f"  {name}:")
        print(f"    segment_filter: {filter_val}")
        print(f"    optimum_risk: {risk}")
        if supersegment:
            print(f"    supersegment: {supersegment} (uses shared model)")
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

    args = parser.parse_args()

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
        ss = seg_config.get("supersegment")
        if ss:
            if ss not in all_supersegments:
                print(f"Warning: Segment '{seg_name}' references unknown supersegment '{ss}'")
                print(f"  Available supersegments: {list(all_supersegments.keys())}")
                print("  Segment will train its own model instead.")
            else:
                used_supersegments.add(ss)

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
                output_base=args.output, segments=segments, supersegments=all_supersegments, output_path=args.output
            )
            print("\nConsolidated report saved to:")
            print(f"  - {args.output}/consolidated_risk_production.csv")
            print(f"  - {args.output}/consolidated_risk_production.html")
            return 0
        except Exception as e:
            logger.error(f"Error generating consolidated report: {e}")
            logger.exception("Full traceback:")
            return 1

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
        )

    # Print summary
    print_summary(results)

    # Generate consolidated report
    if not args.no_consolidation:
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
                )
                print("\nConsolidated report saved to:")
                print(f"  - {args.output}/consolidated_risk_production.csv")
                print(f"  - {args.output}/consolidated_risk_production.html")
            except Exception as e:
                logger.error(f"Error generating consolidated report: {e}")
                logger.exception("Full traceback:")
        else:
            print("\nNo successful segments to consolidate.")

    # Generate score discriminance report
    if not args.no_consolidation and preloaded_data is not None:
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

    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
