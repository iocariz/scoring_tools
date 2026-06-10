"""Policy registry + champion/challenger comparison (phase 1).

Maintains a committed, per-segment registry of deployed cutoff policies
(``reports/policy_registry/<segment>.json``) and compares a *challenger* (the latest run's
base policy) against the registered *champion* on a common, matured out-of-time cohort —
reusing the M4 backtest machinery and the M5 headline/provenance. Reporting-only and
read-only with respect to the pipeline: it never re-fits bins, the model, or the mask.

Actions (pick one):
    --register   Freeze each segment's current base policy into the registry. The first
                 policy for a segment becomes champion automatically; add --make-champion to
                 promote a later one. (No data load needed.)
    --list       Print each segment's champion + policy history.
    --compare    Score the current base policy (challenger) against the champion on the
                 auto-derived matured holdout cohort. Writes per-segment + consolidated
                 reports under --output. (Loads the SAS once.)

Usage
-----
    uv run python run_policy_registry.py --register                       # all frozen segments
    uv run python run_policy_registry.py --register -s no_premium_cd --make-champion
    uv run python run_policy_registry.py --list -s no_premium_cd
    uv run python run_policy_registry.py --compare                        # challenger vs champion, all segments
    uv run python run_policy_registry.py --compare -s no_premium_cd \
        --holdout-start 2025-06-01 --holdout-end 2025-08-01               # pinned eval window
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from loguru import logger

from run_backtest import _discover_segments, _scenario_suffix
from src.backtest import BacktestError, derive_holdout_window, load_frozen_policy
from src.config import PreprocessingSettings
from src.constants import StatusName
from src.policy_registry import (
    REGISTRY_DIR,
    PolicyComparison,
    bin_edges_match,
    build_policy_entry,
    compare_policies,
    get_champion,
    load_registry,
    register_policy,
    settings_bin_edges,
    write_comparison_consolidated,
    write_comparison_report,
)
from src.preprocess_improved import _run_data_transformations
from src.reproducibility import _accepted_set_hash

# --------------------------------------------------------------------------- #
# Register / list
# --------------------------------------------------------------------------- #


def register_all(
    data_dir: str | Path = "output",
    scenario: str = "base",
    segments: list[str] | None = None,
    make_champion: bool = False,
    registry_dir: Path = REGISTRY_DIR,
    created: str | None = None,
) -> int:
    """Register every frozen segment's current base policy. Returns the count registered."""
    suffix = _scenario_suffix(scenario)
    seg_dirs = _discover_segments(Path(data_dir), suffix, segments)
    if not seg_dirs:
        logger.warning(f"No frozen segments with accepted_cells{suffix}.csv under {data_dir}.")
        return 0
    n = 0
    for seg_dir in seg_dirs:
        try:
            settings = PreprocessingSettings.from_toml(str(seg_dir / "config_segment.toml"))
            entry = build_policy_entry(seg_dir, settings, suffix, created=created)
        except Exception as e:
            logger.error(f"[{seg_dir.name}] could not build policy entry: {e}")
            continue
        reg = register_policy(entry, make_champion=make_champion, registry_dir=registry_dir)
        role = "champion" if reg["champion_policy_id"] == entry.policy_id else "challenger"
        logger.info(f"[{entry.segment}] registered {entry.policy_id} ({role}) → {registry_dir}/{entry.segment}.json")
        n += 1
    return n


def list_all(
    data_dir: str | Path = "output", segments: list[str] | None = None, registry_dir: Path = REGISTRY_DIR
) -> int:
    """Print each requested segment's registry (champion + history). Returns segment count shown."""
    # Segment names come from the filesystem registry, optionally filtered by --segments.
    reg_dir = Path(registry_dir)
    names = sorted(p.stem for p in reg_dir.glob("*.json")) if reg_dir.exists() else []
    if segments:
        names = [n for n in names if n in segments]
    if not names:
        logger.warning(f"No registries under {registry_dir}" + (f" for {segments}" if segments else ""))
        return 0
    for name in names:
        reg = load_registry(name, registry_dir)
        champ = reg.get("champion_policy_id")
        logger.info(f"=== {name} === champion: {champ or '(none)'} | {len(reg['policies'])} policy(ies)")
        for p in reg["policies"]:
            h = p.get("headline", {})
            marker = "★" if p["policy_id"] == champ else " "
            logger.info(
                f"  {marker} {p['policy_id']}  cells={h.get('n_accepted_cells')} "
                f"risk={h.get('risk_pct')}% prod=€{h.get('production_eur')} "
                f"created={p.get('provenance', {}).get('created')}"
            )
    return len(names)


# --------------------------------------------------------------------------- #
# Compare (challenger vs champion)
# --------------------------------------------------------------------------- #


def compare_segment(
    full_data: pd.DataFrame,
    settings: PreprocessingSettings,
    seg_output_dir: Path,
    registry_dir: Path = REGISTRY_DIR,
    suffix: str = "_base",
    maturity_months: int = 6,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
) -> PolicyComparison:
    """Compare the current base policy (challenger) vs the registered champion on a holdout cohort."""
    segment = settings.segment_filter
    variables = list(settings.variables)
    multiplier = float(settings.multiplier)
    seg_data_dir = Path(seg_output_dir) / "data"

    champion = get_champion(segment, registry_dir)
    if champion is None:
        return PolicyComparison(
            segment=segment,
            basis="realized_oot",
            sufficient=False,
            message="no champion registered — run --register --make-champion first",
        )

    challenger_set, _ = load_frozen_policy(seg_data_dir, variables, suffix)
    challenger_id = f"{segment}-{_accepted_set_hash(challenger_set)[:8]}"
    champion_set = champion.accepted_set()

    # Audit #31: the champion's cell indices are only meaningful on the bin
    # edges it was registered with — refuse to score it on different edges
    # (the comparison would silently evaluate a different policy).
    current_edges = settings_bin_edges(settings)
    if champion.bin_edges and not bin_edges_match(champion.bin_edges, current_edges):
        logger.error(
            f"[{segment}] champion bin edges differ from the current config "
            f"(champion: {champion.bin_edges} | current: {current_edges}) — refusing to compare."
        )
        return PolicyComparison(
            segment=segment,
            basis="realized_oot",
            sufficient=False,
            message="bin edges changed since champion registration — re-register the champion "
            "(or restore the edges); comparing would score a different policy",
            champion_policy_id=champion.policy_id,
            challenger_policy_id=challenger_id,
        )
    if list(champion.variables) != variables:
        logger.error(
            f"[{segment}] champion variables {list(champion.variables)} != current {variables} — refusing to compare."
        )
        return PolicyComparison(
            segment=segment,
            basis="realized_oot",
            sufficient=False,
            message="grid variables changed since champion registration — re-register the champion",
            champion_policy_id=champion.policy_id,
            challenger_policy_id=challenger_id,
        )

    # Bin the full population with the frozen edges, then slice the matured out-of-time cohort.
    data_clean = _run_data_transformations(full_data, settings)
    if holdout_start and holdout_end:
        window = {
            "start": pd.to_datetime(holdout_start),
            "end": pd.to_datetime(holdout_end),
            "reference_date": pd.to_datetime(data_clean["mis_date"]).max(),
            "maturity_months": maturity_months,
            "sufficient": True,
        }
    else:
        window = derive_holdout_window(data_clean, settings, maturity_months)

    if not window["sufficient"]:
        return PolicyComparison(
            segment=segment,
            basis="realized_oot",
            sufficient=False,
            message="no mature out-of-time cohort to compare on",
            champion_policy_id=champion.policy_id,
            challenger_policy_id=challenger_id,
            window=window,
        )

    booked = data_clean[data_clean["status_name"] == StatusName.BOOKED.value].copy()
    booked_mis = pd.to_datetime(booked["mis_date"])
    booked_oot = booked[(booked_mis > window["start"]) & (booked_mis <= window["end"])]

    # Full demand of the same window — sizes the unobservable exposure of the
    # challenger's added cells for the survivorship guard (audit #31).
    all_mis = pd.to_datetime(data_clean["mis_date"])
    demand_oot = data_clean[(all_mis > window["start"]) & (all_mis <= window["end"])]

    cmp = compare_policies(
        champion_set,
        challenger_set,
        booked_oot,
        variables,
        multiplier,
        segment=segment,
        champion_id=champion.policy_id,
        challenger_id=challenger_id,
        cohort_demand=demand_oot,
    )
    cmp.window = window
    return cmp


def compare_all(
    full_data: pd.DataFrame,
    data_dir: str | Path = "output",
    out_dir: str | Path = "output/policy_registry",
    scenario: str = "base",
    maturity_months: int = 6,
    segments: list[str] | None = None,
    holdout_start: str | None = None,
    holdout_end: str | None = None,
    registry_dir: Path = REGISTRY_DIR,
) -> list[PolicyComparison]:
    """Compare every frozen segment's challenger vs its champion; write reports under *out_dir*."""
    suffix = _scenario_suffix(scenario)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    seg_dirs = _discover_segments(data_dir, suffix, segments)
    if not seg_dirs:
        logger.warning(f"No frozen segments with accepted_cells{suffix}.csv under {data_dir}.")
        return []

    results: list[PolicyComparison] = []
    for seg_dir in seg_dirs:
        try:
            settings = PreprocessingSettings.from_toml(str(seg_dir / "config_segment.toml"))
            cmp = compare_segment(
                full_data,
                settings,
                seg_dir,
                registry_dir=registry_dir,
                suffix=suffix,
                maturity_months=maturity_months,
                holdout_start=holdout_start,
                holdout_end=holdout_end,
            )
        except BacktestError as e:
            logger.error(f"[{seg_dir.name}] compare failed: {e}")
            continue
        except Exception:
            logger.exception(f"[{seg_dir.name}] unexpected compare error")
            continue

        write_comparison_report(cmp, out_dir, suffix)
        results.append(cmp)
        if cmp.sufficient:
            logger.info(
                f"[{cmp.segment}] {cmp.verdict} | +{len(cmp.added_cells)}/−{len(cmp.removed_cells)} cells | "
                f"risk {cmp.champion.get('risk')}% → {cmp.challenger.get('risk')}% | "
                f"production Δ {cmp.production_delta_pct}%"
            )
        else:
            logger.warning(f"[{cmp.segment}] {cmp.message}")

    if results:
        paths = write_comparison_consolidated(results, out_dir, suffix)
        logger.info(f"Consolidated comparison: {paths['consolidated']}")
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Policy registry + champion/challenger comparison (phase 1).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--register", action="store_true", help="Register each segment's current base policy.")
    action.add_argument("--list", action="store_true", help="Print each segment's champion + policy history.")
    action.add_argument("--compare", action="store_true", help="Compare challenger vs champion on a holdout cohort.")

    parser.add_argument("--config", "-c", default="config.toml", help="Base config (for data_path).")
    parser.add_argument("--data-dir", default="output", help="Dir holding the frozen per-segment runs.")
    parser.add_argument("--output", "-o", default="output/policy_registry", help="Output dir for comparison reports.")
    parser.add_argument("--registry-dir", default=str(REGISTRY_DIR), help="Committed registry dir.")
    parser.add_argument("--scenario", default="base", help="Scenario suffix (default: base).")
    parser.add_argument("--segments", "-s", nargs="+", default=None, help="Subset of segments (default: all).")
    parser.add_argument("--make-champion", action="store_true", help="With --register: promote to champion.")
    parser.add_argument("--maturity-months", type=int, default=6, help="Months of seasoning for H6 maturity.")
    parser.add_argument("--holdout-start", default=None, help="Override: held-out window start (YYYY-MM-DD).")
    parser.add_argument("--holdout-end", default=None, help="Override: held-out window end (YYYY-MM-DD).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    registry_dir = Path(args.registry_dir)

    if args.list:
        list_all(args.data_dir, args.segments, registry_dir)
        return 0

    if not Path(args.data_dir).exists():
        logger.error(f"--data-dir not found: {args.data_dir}")
        return 1

    if args.register:
        created = datetime.now(UTC).isoformat()
        n = register_all(
            data_dir=args.data_dir,
            scenario=args.scenario,
            segments=args.segments,
            make_champion=args.make_champion,
            registry_dir=registry_dir,
            created=created,
        )
        if n == 0:
            logger.error("No policies registered.")
            return 1
        return 0

    # --compare: load the SAS once, shared across segments.
    from run_batch import load_and_standardize_data

    with open(args.config, "rb") as f:
        base_cfg = tomllib.load(f).get("preprocessing", {})
    data_path = base_cfg.get("data_path", "data/demanda_direct_out.sas7bdat")
    full_data = load_and_standardize_data(data_path)
    if full_data is None:
        logger.error(f"Could not load/standardize data from {data_path}")
        return 1

    results = compare_all(
        full_data,
        data_dir=args.data_dir,
        out_dir=args.output,
        scenario=args.scenario,
        maturity_months=args.maturity_months,
        segments=args.segments,
        holdout_start=args.holdout_start,
        holdout_end=args.holdout_end,
        registry_dir=registry_dir,
    )
    if not results:
        logger.error("No segments produced a comparison.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
