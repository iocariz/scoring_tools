"""Golden-numbers reproducibility check (M5).

Re-runs a single segment end-to-end and compares its headline numbers (chosen optimum:
risk %, production €, accepted-cell count + accepted-set hash) to a **committed reference**
within tolerance. Confirms an independent party can reproduce the same answer from the
pinned (data snapshot, code, config); flags loudly when the data SHA-256 has changed.

The reference is a self-consistent **standalone** run (`config.toml` `[preprocessing]` +
`segment_filter`) — it trains the segment's own model, so the headline differs from the
production pooled-`total` numbers (those are validated via the M4 backtest + the #7
multi-segment validation). The point here is *deterministic reproducibility of the pipeline*.

Usage
-----
    uv run python run_reproducibility.py --update-reference -s no_premium_cd   # establish the reference
    uv run python run_reproducibility.py -s no_premium_cd                      # check against the reference
    uv run python run_reproducibility.py -s no_premium_cd --model-path output/_repro/no_premium_cd/models/model_*  # fast
    uv run python run_reproducibility.py -s no_premium_cd --risk-tol-pp 0.02 --prod-tol-pct 0.5
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tomllib
from pathlib import Path

import tomli_w
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.reproducibility import (
    REFERENCE_DIR,
    compare_headline,
    extract_headline,
    load_reference,
    render_report,
    write_reference,
)


def _build_standalone_config(base_config_path: str, segment_filter: str, dest: Path) -> Path:
    """Write a standalone config = base [preprocessing] with *segment_filter* injected."""
    with open(base_config_path, "rb") as f:
        cfg = copy.deepcopy(tomllib.load(f))
    cfg.setdefault("preprocessing", {})["segment_filter"] = segment_filter
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        tomli_w.dump(cfg, f)
    return dest


def _resolve_segment_filter(segment: str, segments_config_path: str) -> str:
    """Resolve the segment's REAL ``segment_filter`` from segments.toml (#47).

    The ``-s`` argument is a segment NAME; the actual filter (e.g. ``direct/consolidation/known/premium``)
    lives in segments.toml — it is NOT the name (``direct-conso-known-premium``). Injecting the raw name
    as the filter matched no ``segment_cut_off`` rows (empty/wrong population). Fall back to the raw
    value — with a warning — when the segment isn't found, so a directly-passed filter still works.
    """
    try:
        from run_batch import load_segments_config

        segs = load_segments_config(segments_config_path)
    except Exception as e:  # noqa: BLE001 — missing/unreadable config → fall back, don't crash the gate
        logger.warning(f"Could not load segments config '{segments_config_path}': {e}; using '{segment}' verbatim.")
        return segment
    cfg = segs.get(segment) if isinstance(segs, dict) else None
    if cfg and cfg.get("segment_filter"):
        logger.info(
            f"Resolved segment '{segment}' → segment_filter='{cfg['segment_filter']}' ({segments_config_path})."
        )
        return cfg["segment_filter"]
    logger.warning(
        f"Segment '{segment}' not found in '{segments_config_path}' (or it has no segment_filter); "
        f"treating '{segment}' as the literal segment_filter."
    )
    return segment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Golden-numbers reproducibility check (M5).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", "-c", default="config.toml", help="Base config.")
    parser.add_argument("--segment", "-s", required=True, help="Segment to reproduce (name from segments.toml).")
    parser.add_argument(
        "--segments-config", default="segments.toml", help="Segments config to resolve the segment's filter."
    )
    parser.add_argument(
        "--reference", default=None, help="Reference JSON (default reports/validation/reference/<seg>...)."
    )
    parser.add_argument("--output", "-o", default="output/_repro", help="Run/report output dir.")
    parser.add_argument("--scenario", default="base", help="Scenario suffix (default: base).")
    parser.add_argument("--risk-tol-pp", type=float, default=0.01, help="Risk tolerance (percentage points).")
    parser.add_argument("--prod-tol-pct", type=float, default=0.1, help="Production tolerance (percent).")
    parser.add_argument("--model-path", default=None, help="Reuse a trained model dir (skip retraining; faster).")
    parser.add_argument(
        "--update-reference", action="store_true", help="(Re)write the committed reference instead of checking."
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    suffix = f"_{args.scenario}" if args.scenario else ""
    segment = args.segment
    ref_path = Path(args.reference) if args.reference else REFERENCE_DIR / f"{segment}_headline.json"
    run_dir = Path(args.output) / segment
    # Clean the run dir first (#47): otherwise a re-run that fails partway leaves the PRIOR run's
    # artifacts in place, and extract_headline below would read those → a spurious PASS (or freeze
    # stale numbers via --update-reference). Rebuilt by _build_standalone_config.
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Re-run the segment end-to-end (base scenario only) into run_dir.
    from main import main as run_main_pipeline

    # Resolve the segment's REAL filter from segments.toml (the -s value is a name, not the filter).
    segment_filter = _resolve_segment_filter(segment, args.segments_config)
    cfg_path = _build_standalone_config(args.config, segment_filter, run_dir / "config.toml")
    settings = PreprocessingSettings.from_toml(str(cfg_path))
    logger.info(
        f"Reproducing headline for '{segment}' (model_path={'reuse' if args.model_path else 'full retrain'}) ..."
    )
    result_pipeline = run_main_pipeline(
        config_path=str(cfg_path),
        output=OutputPaths(base_dir=run_dir),
        model_path=args.model_path,
        base_scenario_only=True,
        # Reproduce under the same DQ posture the headline was established with: the known,
        # accepted benign outlier warnings (M2 is fail-closed by default).
        allow_dq_warnings=True,
    )
    # Fail closed (#47): the pipeline returns None on failure. Refuse to read whatever is in run_dir
    # (partial/absent) — that is how the gate used to pass spuriously or freeze stale references.
    if result_pipeline is None:
        logger.error(
            f"[{segment}] pipeline returned no result — reproduction FAILED (refusing to read stale/"
            "partial artifacts). Fix the run before checking or re-pinning the reference."
        )
        return 1

    headline = extract_headline(run_dir, settings, suffix=suffix)
    logger.info(
        f"[{segment}] reproduced: risk={headline.risk_pct} prod={headline.production_eur} "
        f"cells={headline.n_accepted_cells} data_sha={str(headline.data_sha256)[:12]}"
    )

    if args.update_reference:
        path = write_reference(ref_path, headline)
        logger.info(f"Reference written: {path}")
        return 0

    if not ref_path.exists():
        logger.error(f"No reference at {ref_path} — run with --update-reference first.")
        return 1

    reference = load_reference(ref_path)
    result = compare_headline(headline, reference, risk_tol_pp=args.risk_tol_pp, prod_tol_pct=args.prod_tol_pct)

    report_md = run_dir / f"reproducibility_{segment}{suffix}.md"
    report_md.write_text(render_report(result, segment), encoding="utf-8")
    import json

    (run_dir / f"reproducibility_{segment}{suffix}.json").write_text(json.dumps(result, indent=2, default=str))

    status = "PASS" if result["passed"] else "FAIL"
    (logger.success if result["passed"] else logger.error)(
        f"[{segment}] reproducibility {status} | risk Δ {result['risk_delta_pp']} pp | "
        f"prod Δ {result['prod_delta_pct']} % | snapshot_match={result['snapshot_match']}"
    )
    for reason in result["reasons"]:
        logger.warning(f"  - {reason}")
    logger.info(f"Report: {report_md}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
