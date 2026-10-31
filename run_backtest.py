"""Out-of-time backtest of frozen cutoffs (M4).

Applies each segment's **already-chosen** acceptance policy (from a completed run's
``accepted_cells*.csv``) as-is to a held-out, out-of-time cohort and compares realized
vs predicted risk/production. This is a pure, repeatable backtest — it never re-fits
bins, the model, or the mask (unlike the inline MR check, which re-optimizes).

The held-out window is auto-derived as the cohorts booked *after* the training window
that are old enough to have a realized H6 outcome:
``(date_fin_book_obs, max(mis_date) − maturity_months]``. Risk is compared directly
(a rate); production is reported as realized acceptance-rate + realized € (absolute €
is not comparable across periods of different size).

Usage
-----
    uv run python run_backtest.py                                  # all frozen segments
    uv run python run_backtest.py -s no_premium_cd premium         # specific segments
    uv run python run_backtest.py --data-dir output --output output/backtest
    uv run python run_backtest.py --scenario base --maturity-months 6
    uv run python run_backtest.py -s no_premium_cd --holdout-start 2025-06-01 --holdout-end 2025-08-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from src.governance.backtest_runner import backtest_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Out-of-time backtest of frozen cutoffs (M4).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", default="config.toml", help="Base config (for data_path).")
    parser.add_argument("--data-dir", default="output", help="Dir holding the frozen per-segment runs.")
    parser.add_argument("--output", "-o", default="output/backtest", help="Output dir for backtest artifacts.")
    parser.add_argument("--scenario", default="base", help="Scenario suffix to backtest (default: base).")
    parser.add_argument("--segments", "-s", nargs="+", default=None, help="Subset of segments (default: all frozen).")
    parser.add_argument(
        "--maturity-months",
        type=int,
        default=None,
        help="Months of seasoning for H6 maturity (default: each segment's own mr_maturity_months).",
    )
    parser.add_argument("--holdout-start", default=None, help="Override: held-out window start (YYYY-MM-DD).")
    parser.add_argument("--holdout-end", default=None, help="Override: held-out window end (YYYY-MM-DD).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not Path(args.data_dir).exists():
        logger.error(f"--data-dir not found: {args.data_dir}")
        return 1

    # Load + standardize the SAS once, shared across all segments.
    from src.data.data_manager import load_configured_data
    from src.pipeline.batch_config import load_base_config

    base_cfg = load_base_config(args.config)
    full_data = load_configured_data(base_cfg)  # configured path + sas_encoding (#64)
    if full_data is None:
        logger.error(f"Could not load/standardize data from {base_cfg.get('data_path')}")
        return 1

    results = backtest_all(
        full_data,
        data_dir=args.data_dir,
        out_dir=args.output,
        scenario=args.scenario,
        maturity_months=args.maturity_months,
        segments=args.segments,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
    )
    if not results:
        logger.error("No segments produced a backtest result.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
