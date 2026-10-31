"""3D risk-surface plots: b2_ever_h6 over the two score bins, split by audit category.

For every segment (and reporting supersegment) with a completed run, renders an interactive
3D figure of ``b2_ever_h6`` over the two score bins, with one coloured surface per audit
category — **keep** / **swap_out** (booked) and **swap_in** / **rejected** (repesca). When a
third grid variable (e.g. ``income_bin``) is present, the figure is faceted side-by-side, one
3D scene per value. Read-only: it consumes a completed run's per-cell summary
(``data_summary_desagregado_*.csv``) + accepted-cell mask (``accepted_cells_*.csv``).

Supersegments are built by aggregating their member segments' classified cells (each cell's
category comes from that member's own mask), so a reporting group shows the combined surfaces.

Usage
-----
    uv run python plot_risk_surface.py                          # all segments + reporting supersegments (main period)
    uv run python plot_risk_surface.py --period mr              # the proposed cutoff applied to the out-of-time (MR) cohort
    uv run python plot_risk_surface.py -s no_premium_cd premium # specific segments
    uv run python plot_risk_surface.py --scenario base --no-supersegments
    uv run python plot_risk_surface.py --data-dir output --output output/risk_surfaces
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.report.risk_surface_batch import generate_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="3D risk surfaces (b2_ever_h6 by score bin, split by audit category).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default="output", help="Dir holding the per-segment runs.")
    parser.add_argument("--output", "-o", default="output/risk_surfaces", help="Output dir for HTML.")
    parser.add_argument("--segments-config", default="segments.toml", help="For supersegment grouping.")
    parser.add_argument("--scenario", default="base", help="Scenario suffix (default: base).")
    parser.add_argument("--segments", "-s", nargs="+", default=None, help="Subset of segments (default: all).")
    parser.add_argument("--no-supersegments", action="store_true", help="Skip reporting-supersegment figures.")
    parser.add_argument(
        "--period",
        choices=["main", "mr"],
        default="main",
        help="Period: 'main' (default) or 'mr' (the proposed cutoff applied to the out-of-time cohort).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not Path(args.data_dir).exists():
        logger.error(f"--data-dir not found: {args.data_dir}")
        return 1

    written = generate_all(
        data_dir=args.data_dir,
        out_dir=args.output,
        segments_config=args.segments_config,
        scenario=args.scenario,
        segments=args.segments,
        include_supersegments=not args.no_supersegments,
        period=args.period,
    )
    if not written:
        logger.error("No risk-surface figures were generated.")
        return 1
    idx = "index_mr.html" if args.period == "mr" else "index.html"
    logger.info(f"Wrote {len(written)} figure(s) → {args.output} (open {idx})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
