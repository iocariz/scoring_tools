"""Out-of-time backtest (M4) across every frozen segment of a completed run: discovery, per-segment
evaluation and consolidated outputs. Shared by ``run_backtest.py`` and ``run_batch.py``.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from src.core.config import PreprocessingSettings
from src.governance.backtest import (
    BacktestError,
    backtest_segment,
    write_backtest_report,
    write_consolidated_report,
)


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
    maturity_months: int | None = None,
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
            # #64: default to each segment's own mr_maturity_months (its H6
            # maturity definition) rather than a hard-coded 6; an explicit
            # --maturity-months override still applies to every segment.
            seg_maturity = maturity_months if maturity_months is not None else settings.mr_maturity_months
            result = backtest_segment(
                full_data,
                settings,
                seg_dir,
                suffix=suffix,
                maturity_months=seg_maturity,
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
