"""
Global Allocation CLI Script

This script runs the global portfolio optimization process.
It loads efficient frontiers from 'data/efficient_frontier_*.csv' (or specified paths),
and allocates risk targets to maximize global production.

The global risk target is PRODUCTION-weighted (Sum prod*risk / Sum prod) -- the figure the solver
constrains. The exposure-weighted portfolio bad-rate (Sum bad / Sum exposure) is reported alongside
it for reference only (audit #14).

Per-segment constraints are loaded from segments.toml (min_risk, max_risk, min_production).
Segments can be locked to a specific sol_fac via --lock.

Portfolio-owner outputs (written next to --output):
  - ``<stem>_policy_cutoff_table.csv`` — long-form cutoffs / mask summary per segment
  - ``<stem>_allocation_narrative.md`` — plain-language constraints (single target)
  - ``<stem>_what_if.csv`` — when multiple targets: comparison table
  - ``<stem>_allocation_narratives.md`` — when multiple targets: one narrative section per target

Usage:
    uv run python run_allocation.py --target 1.0
    uv run python run_allocation.py --what-if 2.0,2.5,3.0
    uv run python run_allocation.py --target 2.5 --what-if 2.0,3.0
    uv run python run_allocation.py --target 1.0 --method greedy
    uv run python run_allocation.py --target 1.0 --production-floor 50000
    uv run python run_allocation.py --target 1.0 --lock seg_a:3 --lock seg_b:1
"""

import argparse
import sys
import tomllib
from pathlib import Path

import pandas as pd
from loguru import logger

# Ensure src is in path
sys.path.append(".")

from src.global_optimizer import GlobalAllocator, SegmentConstraints
from src.portfolio_owner import (
    allocation_constraint_narrative,
    policy_cutoff_table_from_allocation,
    what_if_dataframe_row,
    write_portfolio_owner_pack,
)


def load_segment_constraints(segments_path: str = "segments.toml") -> dict[str, SegmentConstraints]:
    """Load per-segment constraints from segments.toml."""
    path = Path(segments_path)
    if not path.exists():
        return {}

    with open(path, "rb") as f:
        config = tomllib.load(f)

    constraints = {}
    for seg_name, seg_cfg in config.get("segments", {}).items():
        min_r = seg_cfg.get("min_risk")
        max_r = seg_cfg.get("max_risk")
        min_p = seg_cfg.get("min_production")
        if min_r is not None or max_r is not None or min_p is not None:
            constraints[seg_name] = SegmentConstraints(
                min_risk=float(min_r) if min_r is not None else None,
                max_risk=float(max_r) if max_r is not None else None,
                min_production=float(min_p) if min_p is not None else None,
            )

    return constraints


def _parse_target_list(target: float | None, what_if: str | None) -> list[float]:
    """Build ordered unique list of risk targets (%%)."""
    targets: list[float] = []
    if target is not None:
        targets.append(float(target))
    if what_if:
        for part in what_if.split(","):
            part = part.strip()
            if not part:
                continue
            val = float(part)
            if val not in targets:
                targets.append(val)
    return targets


def _discover_segment_frontiers(output_base: Path, scenario: str) -> list[tuple[str, Path]]:
    """Resolve which efficient-frontier CSVs to load, as an ordered ``(name, path)`` list.

    Prefers the multi-segment (batch) layout over the single-segment layout so a directory that
    has BOTH (a top-level ``data/`` AND per-segment subdirs) doesn't double-count the root as an
    extra 'segment' — the old code ran both blocks unconditionally and summed the root frontier on
    top of the real segments.

    - Multi-segment (batch): ``<output_base>/<segment>/data/efficient_frontier_<scenario>.csv``
    - Single-segment (fallback only): ``<output_base>/data/efficient_frontier_<scenario>.csv``
    """
    found: list[tuple[str, Path]] = []
    for segment_dir in sorted(p for p in output_base.iterdir() if p.is_dir()):
        fp = segment_dir / "data" / f"efficient_frontier_{scenario}.csv"
        if fp.exists():
            found.append((segment_dir.name, fp))
    if found:
        return found
    single = output_base / "data" / f"efficient_frontier_{scenario}.csv"
    if single.exists():
        return [(output_base.name, single)]
    return []


def main():
    parser = argparse.ArgumentParser(description="Global Portfolio Risk Allocation")
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Global risk target (%%). Optional if --what-if is provided.",
    )
    parser.add_argument(
        "--what-if",
        dest="what_if",
        type=str,
        default=None,
        help="Comma-separated extra risk targets (%%) for a comparison pack, e.g. 2.0,2.5,3.0",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="output",
        help=(
            "Base directory for segment outputs/frontiers. "
            "Expected layout: <data-dir>/<segment>/data/efficient_frontier_<scenario>.csv "
            "(default: output)"
        ),
    )
    parser.add_argument("--output", type=str, default="allocation_results.csv", help="Output file path")
    parser.add_argument("--scenario", type=str, default="base", help="Scenario to filter files by (default: base)")
    parser.add_argument(
        "--method",
        type=str,
        choices=["exact", "greedy"],
        default="exact",
        help="Optimization method: exact (MILP) or greedy (default: exact)",
    )
    parser.add_argument(
        "--segments-config",
        type=str,
        default="segments.toml",
        help="Path to segments config file (default: segments.toml)",
    )
    parser.add_argument(
        "--production-floor",
        type=float,
        default=None,
        help="Global minimum production floor",
    )
    parser.add_argument(
        "--lock",
        action="append",
        default=[],
        metavar="SEGMENT:SOL_FAC",
        help="Lock a segment to a specific sol_fac (repeatable, e.g. --lock seg_a:3)",
    )

    args = parser.parse_args()

    targets = _parse_target_list(args.target, args.what_if)
    if not targets:
        parser.error("Provide at least one risk target via --target and/or --what-if")

    allocator = GlobalAllocator()

    output_base = Path(args.data_dir)
    if not output_base.exists():
        logger.error(f"Data directory '{output_base}' does not exist. Run run_batch.py first or pass --data-dir.")
        return

    segments_found = []

    # Multi-segment (batch) layout preferred; single-segment as a fallback — never both, so a
    # directory with both layouts isn't double-counted.
    for name, frontier_path in _discover_segment_frontiers(output_base, args.scenario):
        logger.info(f"Loading frontier for {name} from {frontier_path}")
        try:
            df = pd.read_csv(frontier_path)
            allocator.load_frontier(name, df)
            segments_found.append(name)
        except Exception as e:
            logger.error(f"Failed to load {frontier_path}: {e}")

    if not segments_found:
        logger.error(
            f"No efficient frontiers found for scenario '{args.scenario}' under '{output_base}'. "
            "Expected either <data-dir>/data/... (single segment) or <data-dir>/<segment>/data/... (batch)."
        )
        return

    constraints = load_segment_constraints(args.segments_config)

    # Apply --lock overrides
    for lock_spec in args.lock:
        if ":" not in lock_spec:
            logger.error(f"Invalid --lock format '{lock_spec}'. Expected SEGMENT:SOL_FAC")
            return
        seg_name, sol_fac_str = lock_spec.split(":", 1)
        try:
            sol_fac = int(sol_fac_str)
        except ValueError:
            logger.error(f"Invalid sol_fac in --lock '{lock_spec}'. Must be an integer.")
            return
        if seg_name in constraints:
            constraints[seg_name].locked_sol_fac = sol_fac
        else:
            constraints[seg_name] = SegmentConstraints(locked_sol_fac=sol_fac)
        logger.info(f"Locked segment '{seg_name}' to sol_fac={sol_fac}")

    if constraints:
        logger.info(f"Segment constraints loaded for: {list(constraints.keys())}")

    out_path = Path(args.output)
    stem = out_path.stem
    out_dir = out_path.parent

    results: list[tuple[float, object]] = []
    for t in targets:
        logger.info(f"Optimizing for global target: {t}% across {len(segments_found)} segments")
        result = allocator.optimize(
            t,
            constraints=constraints if constraints else None,
            global_production_floor=args.production_floor,
            method=args.method,
        )
        if not getattr(result, "feasible", True):
            logger.warning(
                f"INFEASIBLE allocation at target {t}%: {result.message or 'constraints not satisfied'}. "
                "Reported figures are the closest feasible-effort result, not an optimum."
            )
        pbr = getattr(result, "portfolio_bad_rate", None)
        if pbr is not None:
            logger.info(
                f"Target {t}%: production-weighted risk {result.global_risk:.4f}% (the optimized ceiling) | "
                f"exposure-weighted portfolio bad-rate {pbr:.4f}% (reference)."
            )
        results.append((t, result))

    # Primary CSV = first target (same as first requested)
    primary = results[0][1]
    primary.to_full_dataframe().to_csv(out_path, index=False)
    logger.info(f"Primary allocation ({results[0][0]}%%) saved to {out_path}")

    print(primary)
    if primary.binding_constraints:
        print(f"\nBinding constraints: {', '.join(primary.binding_constraints)}")

    if len(results) == 1:
        _, narrative_path = write_portfolio_owner_pack(
            primary,
            constraints if constraints else None,
            output_csv=out_path,
            scenario=args.scenario,
            global_production_floor=args.production_floor,
        )
        logger.info(f"Policy cutoff table and constraint narrative saved beside output ({narrative_path})")
    else:
        policy_tables: list[pd.DataFrame] = []
        narrative_blocks: list[str] = []
        for t, result in results:
            policy_tables.append(
                policy_cutoff_table_from_allocation(
                    result.segment_details,
                    scenario=args.scenario,
                    target_risk_pct=t,
                )
            )
            narrative_blocks.append(f"### Target {t:.4f}%\n\n")
            narrative_blocks.append(
                allocation_constraint_narrative(
                    result,
                    constraints if constraints else None,
                    global_production_floor=args.production_floor,
                )
            )
            narrative_blocks.append("\n")

        policy_path = out_dir / f"{stem}_policy_cutoff_table.csv"
        pd.concat(policy_tables, ignore_index=True).to_csv(policy_path, index=False)
        logger.info(f"Policy / cutoff table (all targets) saved to {policy_path}")

        multi_narr_path = out_dir / f"{stem}_allocation_narratives.md"
        multi_narr_path.write_text(
            "# Allocation what-if — constraint narratives\n\n" + "".join(narrative_blocks), encoding="utf-8"
        )
        logger.info(f"Multi-target narratives saved to {multi_narr_path}")

        what_if_path = out_dir / f"{stem}_what_if.csv"
        pd.DataFrame([what_if_dataframe_row(r, t) for t, r in results]).to_csv(what_if_path, index=False)
        logger.info(f"What-if comparison table saved to {what_if_path}")


if __name__ == "__main__":
    main()
