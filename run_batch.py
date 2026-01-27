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
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from loguru import logger
from tqdm import tqdm


def load_base_config(config_path: str = "config.toml") -> Dict[str, Any]:
    """Load the base configuration from config.toml."""
    with open(config_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("preprocessing", {})


def load_segments_config(segments_path: str = "segments.toml") -> Dict[str, Dict[str, Any]]:
    """Load segment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("segments", {})


def load_supersegments_config(segments_path: str = "segments.toml") -> Dict[str, Dict[str, Any]]:
    """Load supersegment configurations from segments.toml."""
    with open(segments_path, "rb") as f:
        config = tomllib.load(f)
    return config.get("supersegments", {})


def create_output_directories(base_output_dir: Path) -> Dict[str, Path]:
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


def merge_configs(base_config: Dict[str, Any], segment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge base config with segment-specific overrides."""
    merged = base_config.copy()
    merged.update(segment_config)
    return merged


def run_segment_pipeline(
    segment_name: str,
    segment_config: Dict[str, Any],
    base_config: Dict[str, Any],
    output_base: str = "output",
    model_path: Optional[str] = None,
    skip_dq_checks: bool = False
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

    Returns:
        True if successful, False otherwise
    """
    # Create output directories
    output_dir = Path(output_base) / segment_name
    dirs = create_output_directories(output_dir)

    # Setup logging for this segment
    log_file = dirs["logs"] / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(log_file, rotation="10 MB", level="DEBUG")

    logger.info(f"=" * 80)
    logger.info(f"PROCESSING SEGMENT: {segment_name}")
    logger.info(f"=" * 80)
    logger.info(f"Output directory: {output_dir}")

    # Merge configurations
    merged_config = merge_configs(base_config, segment_config)
    logger.info(f"Segment filter: {merged_config.get('segment_filter')}")
    logger.info(f"Optimum risk: {merged_config.get('optimum_risk')}")
    if model_path:
        logger.info(f"Using pre-trained model from: {model_path}")

    # Change to output directory context and run
    original_cwd = os.getcwd()

    try:
        # Import main module
        from main import main as run_main_pipeline

        # Temporarily modify paths by changing working directory
        # and creating symlinks/copies as needed

        # Create a temporary config with this segment's settings
        temp_config = write_temp_config(merged_config, dirs["root"])

        # Store original image/model/data directories
        original_dirs = {
            "images": Path("images"),
            "models": Path("models"),
            "data": Path("data"),
        }

        # Create symbolic links or rename directories
        # Copy input data file to segment directory before redirection
        data_file = merged_config.get("data_path", "data/demanda_direct_out.sas7bdat")
        setup_output_redirection(dirs, data_file=data_file)

        try:
            # Run the main pipeline
            result = run_main_pipeline(
                config_path=str(temp_config),
                model_path=model_path,
                skip_dq_checks=skip_dq_checks
            )

            if result is None:
                logger.error(f"Pipeline failed for segment: {segment_name}")
                return False

            logger.info(f"Pipeline completed successfully for segment: {segment_name}")
            return True

        finally:
            # Restore original directory structure
            cleanup_output_redirection(dirs)

    except Exception as e:
        logger.error(f"Error processing segment {segment_name}: {e}")
        logger.exception("Full traceback:")
        return False


def run_supersegment_training(
    supersegment_name: str,
    supersegment_config: Dict[str, Any],
    base_config: Dict[str, Any],
    output_base: str = "output",
    skip_dq_checks: bool = False
) -> Optional[str]:
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
    logger.add(log_file, rotation="10 MB", level="DEBUG")

    logger.info(f"=" * 80)
    logger.info(f"TRAINING SUPERSEGMENT MODEL: {supersegment_name}")
    logger.info(f"=" * 80)
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

    original_cwd = os.getcwd()

    try:
        from main import main as run_main_pipeline

        # Create temporary config
        temp_config = write_temp_config(merged_config, dirs["root"])

        # Setup output redirection
        data_file = merged_config.get("data_path", "data/demanda_direct_out.sas7bdat")
        setup_output_redirection(dirs, data_file=data_file)

        try:
            # Run the main pipeline in training-only mode (skip optimization)
            result = run_main_pipeline(
                config_path=str(temp_config),
                training_only=True,
                skip_dq_checks=skip_dq_checks
            )

            if result is None:
                logger.error(f"Supersegment training failed: {supersegment_name}")
                return None

            # Find the most recent model directory
            models_dir = dirs["models"]
            model_dirs = sorted(models_dir.glob("model_*"), reverse=True)

            if not model_dirs:
                logger.error(f"No model directory found after training")
                return None

            model_path = str(model_dirs[0])
            logger.info(f"Supersegment model trained successfully: {model_path}")

            return model_path

        finally:
            cleanup_output_redirection(dirs)

    except Exception as e:
        logger.error(f"Error training supersegment {supersegment_name}: {e}")
        logger.exception("Full traceback:")
        return None

    finally:
        os.chdir(original_cwd)


def write_temp_config(config: Dict[str, Any], output_dir: Path) -> Path:
    """Write a temporary config file for this segment run."""
    import tomli_w

    config_path = output_dir / "config_segment.toml"

    # Wrap config in preprocessing section
    full_config = {"preprocessing": config}

    with open(config_path, "wb") as f:
        tomli_w.dump(full_config, f)

    return config_path


def setup_output_redirection(dirs: Dict[str, Path], data_file: str = None) -> None:
    """
    Setup output redirection by renaming existing directories
    and creating symlinks to segment directories.

    Args:
        dirs: Dictionary of output directories
        data_file: Optional path to the input data file to copy to segment data dir
    """
    # If data file is specified, copy it to the segment's data directory first
    if data_file:
        source_data = Path(data_file)
        if source_data.exists():
            dest_data = dirs["data"] / source_data.name
            if not dest_data.exists():
                shutil.copy2(source_data, dest_data)
                logger.info(f"Copied data file to {dest_data}")

    for name in ["images", "models", "data"]:
        original = Path(name)
        backup = Path(f"{name}_backup")
        segment_dir = dirs[name]

        # Backup original if exists
        if original.exists() and not original.is_symlink():
            if backup.exists():
                shutil.rmtree(backup)
            original.rename(backup)
        elif original.is_symlink():
            original.unlink()

        # Create symlink to segment directory
        original.symlink_to(segment_dir.absolute())


def cleanup_output_redirection(dirs: Dict[str, Path]) -> None:
    """Restore original directory structure after processing."""
    for name in ["images", "models", "data"]:
        original = Path(name)
        backup = Path(f"{name}_backup")

        # Remove symlink
        if original.is_symlink():
            original.unlink()

        # Restore backup
        if backup.exists():
            backup.rename(original)


def run_segments_sequential(
    segments: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any],
    output_base: str = "output",
    supersegments: Dict[str, Dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False
) -> Dict[str, bool]:
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
            print(f"\n{'='*60}")
            print("PHASE 1: Training Supersegment Models")
            print(f"{'='*60}")

            ss_progress = tqdm(
                used_supersegments,
                desc="Supersegments",
                unit="model",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
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
                    skip_dq_checks=skip_dq_checks
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
    segments_to_run = [
        (name, config) for name, config in segments.items()
        if name not in results
    ]

    if segments_to_run:
        print(f"\n{'='*60}")
        print("PHASE 2: Running Segment Optimizations")
        print(f"{'='*60}")

        seg_progress = tqdm(
            segments_to_run,
            desc="Segments",
            unit="segment",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
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
                segment_name, segment_config, base_config, output_base,
                model_path=model_path,
                skip_dq_checks=skip_dq_checks
            )
            results[segment_name] = success

            # Update progress bar color based on result
            if success:
                seg_progress.set_postfix_str(f"{segment_name} ✓", refresh=True)
            else:
                seg_progress.set_postfix_str(f"{segment_name} ✗", refresh=True)

    return results


def run_segments_parallel(
    segments: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any],
    output_base: str = "output",
    max_workers: int = None,
    supersegments: Dict[str, Dict[str, Any]] = None,
    reuse_models: bool = False,
    skip_dq_checks: bool = False
) -> Dict[str, bool]:
    """
    Run all segments in parallel, with supersegment support.

    Note: Supersegment models are trained sequentially first, then
    individual segment optimizations run in parallel.

    Args:
        segments: Segment configurations
        base_config: Base configuration
        output_base: Output directory base
        max_workers: Maximum parallel workers
        supersegments: Optional supersegment configurations
        reuse_models: If True, reuse existing supersegment models instead of retraining

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
            print(f"\n{'='*60}")
            print("PHASE 1: Training Supersegment Models (sequential)")
            print(f"{'='*60}")

            ss_progress = tqdm(
                used_supersegments,
                desc="Supersegments",
                unit="model",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
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
                    skip_dq_checks=skip_dq_checks
                )
                if model_path:
                    supersegment_models[ss_name] = model_path
                else:
                    for seg_name, seg_config in segments.items():
                        if seg_config.get("supersegment") == ss_name:
                            results[seg_name] = False

    # Phase 2: Run individual segment optimizations IN PARALLEL
    segments_to_run = {
        name: config for name, config in segments.items()
        if name not in results
    }

    if segments_to_run:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Running Segment Optimizations (parallel, {max_workers or 'auto'} workers)")
        print(f"{'='*60}")

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
                    skip_dq_checks
                )
                futures[future] = segment_name

            # Progress bar for parallel execution
            seg_progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Segments",
                unit="segment",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
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


def print_summary(results: Dict[str, bool]) -> None:
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
        print(f"\nSuccessful segments:")
        for name in successful:
            print(f"  - {name}")

    if failed:
        print(f"\nFailed segments:")
        for name in failed:
            print(f"  - {name}")

    print("\n" + "=" * 80)


def clean_output_directories(
    segments: Dict[str, Dict[str, Any]],
    supersegments: Dict[str, Dict[str, Any]],
    output_base: str = "output"
) -> Dict[str, bool]:
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
    for seg_name in segments.keys():
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
            print(f"    segment_filters:")
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
        epilog=__doc__
    )

    parser.add_argument(
        "--segments", "-s",
        nargs="+",
        help="Specific segments to run (default: all)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available segments and exit"
    )
    parser.add_argument(
        "--parallel", "-p",
        action="store_true",
        help="Run segments in parallel"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Base output directory (default: output)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.toml",
        help="Path to base config file (default: config.toml)"
    )
    parser.add_argument(
        "--segments-config",
        default="segments.toml",
        help="Path to segments config file (default: segments.toml)"
    )
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="Reuse existing supersegment models if available (skip retraining)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove output directories for selected segments before running"
    )
    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="Only clean output directories (don't run pipeline)"
    )
    parser.add_argument(
        "--skip-dq-checks",
        action="store_true",
        help="Skip data quality checks (not recommended for production)"
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
        segments = {
            name: config
            for name, config in all_segments.items()
            if name in args.segments
        }

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
                print(f"  Segment will train its own model instead.")
            else:
                used_supersegments.add(ss)

    # Handle clean operations
    if args.clean or args.clean_only:
        print(f"\nCleaning output directories for {len(segments)} segment(s)...")
        clean_results = clean_output_directories(
            segments=segments,
            supersegments=all_supersegments,
            output_base=args.output
        )
        failed_cleans = [name for name, success in clean_results.items() if not success]
        if failed_cleans:
            print(f"Warning: Failed to clean: {failed_cleans}")

        if args.clean_only:
            print("\nClean complete.")
            return 0 if not failed_cleans else 1

        print()  # Blank line after clean output

    print(f"\nProcessing {len(segments)} segment(s): {list(segments.keys())}")
    if used_supersegments:
        print(f"Supersegments to train: {list(used_supersegments)}")
        if args.reuse_models:
            print(f"Reuse models: enabled (will skip training if model exists)")
    if args.skip_dq_checks:
        print(f"Data quality checks: DISABLED (--skip-dq-checks)")
    print(f"Output directory: {args.output}")
    print(f"Mode: {'parallel' if args.parallel else 'sequential'}")
    print()

    # Run segments
    if args.parallel:
        results = run_segments_parallel(
            segments, base_config, args.output, args.workers,
            supersegments=all_supersegments,
            reuse_models=args.reuse_models,
            skip_dq_checks=args.skip_dq_checks
        )
    else:
        results = run_segments_sequential(
            segments, base_config, args.output,
            supersegments=all_supersegments,
            reuse_models=args.reuse_models,
            skip_dq_checks=args.skip_dq_checks
        )

    # Print summary
    print_summary(results)

    # Return exit code based on results
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
