"""
Global Allocation CLI Script

This script runs the global portfolio optimization process.
It loads efficient frontiers from 'data/efficient_frontier_*.csv' (or specified paths),
and allocates risk targets to maximize global production.

Per-segment risk constraints are loaded from segments.toml (min_risk / max_risk).

Usage:
    uv run python run_allocation.py --target 1.0
    uv run python run_allocation.py --target 1.0 --method greedy
"""

import argparse
import tomllib
import pandas as pd
from pathlib import Path
from loguru import logger
import sys
from typing import Dict, Tuple

# Ensure src is in path
sys.path.append(".")

from src.global_optimizer import GlobalAllocator


def load_risk_constraints(segments_path: str = "segments.toml") -> Dict[str, Tuple[float, float]]:
    """Load per-segment risk constraints (min_risk, max_risk) from segments.toml."""
    path = Path(segments_path)
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        config = tomllib.load(f)

    constraints = {}
    for seg_name, seg_cfg in config.get("segments", {}).items():
        min_r = seg_cfg.get("min_risk")
        max_r = seg_cfg.get("max_risk")
        if min_r is not None and max_r is not None:
            constraints[seg_name] = (float(min_r), float(max_r))
        elif min_r is not None:
            constraints[seg_name] = (float(min_r), float("inf"))
        elif max_r is not None:
            constraints[seg_name] = (0.0, float(max_r))

    return constraints


def main():
    parser = argparse.ArgumentParser(description="Global Portfolio Risk Allocation")
    parser.add_argument("--target", type=float, required=True, help="Global risk target (%%)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing frontier CSVs")
    parser.add_argument("--output", type=str, default="allocation_results.csv", help="Output file path")
    parser.add_argument("--scenario", type=str, default="base", help="Scenario to filter files by (default: base)")
    parser.add_argument("--method", type=str, choices=["exact", "greedy"], default="exact",
                        help="Optimization method: exact (MILP) or greedy (default: exact)")
    parser.add_argument("--segments-config", type=str, default="segments.toml",
                        help="Path to segments config file (default: segments.toml)")

    args = parser.parse_args()

    allocator = GlobalAllocator()

    output_base = Path("output")
    if not output_base.exists():
        logger.error("Output directory 'output/' does not exist. Run run_batch.py first.")
        return

    segments_found = []

    for segment_dir in output_base.iterdir():
        if segment_dir.is_dir():
            frontier_path = segment_dir / "data" / f"efficient_frontier_{args.scenario}.csv"
            if frontier_path.exists():
                logger.info(f"Loading frontier for {segment_dir.name} from {frontier_path}")
                try:
                    df = pd.read_csv(frontier_path)
                    allocator.load_frontier(segment_dir.name, df)
                    segments_found.append(segment_dir.name)
                except Exception as e:
                    logger.error(f"Failed to load {frontier_path}: {e}")

    if not segments_found:
        logger.error(f"No efficient frontiers found for scenario '{args.scenario}' in output/*/")
        return

    logger.info(f"Starting optimization for global target: {args.target}% across {len(segments_found)} segments")

    constraints = load_risk_constraints(args.segments_config)
    if constraints:
        logger.info(f"Risk constraints loaded for: {list(constraints.keys())}")

    result = allocator.optimize(args.target, constraints, method=args.method)

    print(result)

    result.to_full_dataframe().to_csv(args.output, index=False)
    logger.info(f"Allocation saved to {args.output}")

if __name__ == "__main__":
    main()
