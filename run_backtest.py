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
import tomllib
from pathlib import Path

from loguru import logger

from src.backtest import (
    BacktestError,
    backtest_segment,
    write_backtest_report,
    write_consolidated_report,
)
from src.config import PreprocessingSettings


def _scenario_suffix(scenario: str) -> str:
    return f"_{scenario}" if scenario else ""


def _discover_segments(data_dir: Path, suffix: str, wanted: list[str] | None) -> list[Path]:
    """Frozen-run segment dirs: have config_segment.toml + accepted_cells{suffix}.csv. Skip _internal dirs."""
    found = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "config_segment.toml").exists() and (d / "data" / f"accepted_cells{suffix}.csv").exists():
            if wanted is None or d.name in wanted:
                found.append(d)
    return found


def backtest_all(
    full_data,
    data_dir: str | Path = "output",
    out_dir: str | Path = "output/backtest",
    scenario: str = "base",
    maturity_months: int = 6,
    segments: list[str] | None = None,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> list:
    """Backtest every frozen segment under *data_dir* against a held-out cohort.

    Writes per-segment + consolidated reports under *out_dir* and returns the list of
    ``BacktestResult``. Shared by the CLI and ``run_batch`` (which passes its already-loaded
    ``full_data`` so the SAS file isn't re-read).
    """
    suffix = _scenario_suffix(scenario)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    seg_dirs = _discover_segments(data_dir, suffix, segments)
    if not seg_dirs:
        logger.warning(f"No frozen segments with accepted_cells{suffix}.csv under {data_dir} — skipping backtest.")
        return []

    logger.info(f"Backtesting {len(seg_dirs)} segment(s) on scenario '{scenario}' → {out_dir}")
    results = []
    for seg_dir in seg_dirs:
        try:
            settings = PreprocessingSettings.from_toml(str(seg_dir / "config_segment.toml"))
            result = backtest_segment(
                full_data,
                settings,
                seg_dir,
                suffix=suffix,
                maturity_months=maturity_months,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )
        except BacktestError as e:
            logger.error(f"[{seg_dir.name}] backtest failed: {e}")
            continue
        except Exception:
            logger.exception(f"[{seg_dir.name}] unexpected backtest error")
            continue

        write_backtest_report(result, out_dir, suffix)
        results.append(result)
        if result.sufficient:
            logger.info(
                f"[{result.segment}] {result.drift_flag()} | window "
                f"{result.window['start'].date()}→{result.window['end'].date()} | "
                f"defaults={result.out_of_time.get('n_defaults')} | "
                f"in-sample {result.in_sample.get('risk')}% → OOT {result.out_of_time.get('risk')}%"
            )
        else:
            logger.warning(f"[{result.segment}] {result.message}")

    if results:
        paths = write_consolidated_report(results, out_dir, suffix)
        logger.info(f"Consolidated backtest: {paths['consolidated']}")
    return results


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
    parser.add_argument("--maturity-months", type=int, default=6, help="Months of seasoning for H6 maturity.")
    parser.add_argument("--holdout-start", default=None, help="Override: held-out window start (YYYY-MM-DD).")
    parser.add_argument("--holdout-end", default=None, help="Override: held-out window end (YYYY-MM-DD).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not Path(args.data_dir).exists():
        logger.error(f"--data-dir not found: {args.data_dir}")
        return 1

    # Load + standardize the SAS once, shared across all segments.
    from run_batch import load_and_standardize_data

    with open(args.config, "rb") as f:
        base_cfg = tomllib.load(f).get("preprocessing", {})
    data_path = base_cfg.get("data_path", "data/demanda_direct_out.sas7bdat")
    full_data = load_and_standardize_data(data_path)
    if full_data is None:
        logger.error(f"Could not load/standardize data from {data_path}")
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
