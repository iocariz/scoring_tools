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
import tomllib
from pathlib import Path

import pandas as pd
from loguru import logger

from src.risk_surface import aggregate_to_cells, build_risk_surface_figure, classify_cells


def _scenario_suffix(scenario: str) -> str:
    return f"_{scenario}" if scenario else ""


def _read_cell_key(accepted_path: Path) -> list[str]:
    """The grid-cell key = the columns of accepted_cells (authoritative ordered coords)."""
    return list(pd.read_csv(accepted_path, nrows=0).columns)


def _summary_path(data: Path, suffix: str, period: str) -> Path:
    """Per-cell summary for the chosen period (the MR file carries an extra ``_mr`` infix)."""
    infix = "_mr" if period == "mr" else ""
    return data / f"data_summary_desagregado{infix}{suffix}.csv"


def _load_unit(seg_dir: Path, suffix: str, period: str = "main") -> tuple[pd.DataFrame, set, list[str], float] | None:
    """Load one segment's (summary, accepted_set, variables, multiplier), or None if incomplete.

    The cutoff mask (``accepted_cells``) is the same policy for both periods; only the per-cell
    summary differs (main vs MR), so the MR surface shows the proposed cutoff applied to the
    out-of-time cohort.
    """
    data = seg_dir / "data"
    summary_path = _summary_path(data, suffix, period)
    accepted_path = data / f"accepted_cells{suffix}.csv"
    if not summary_path.exists() or not accepted_path.exists():
        return None
    variables = _read_cell_key(accepted_path)
    summary = pd.read_csv(summary_path)
    if not set(variables).issubset(summary.columns):
        logger.warning(f"[{seg_dir.name}] summary missing grid columns {variables}; skipping.")
        return None
    accepted_df = pd.read_csv(accepted_path).dropna(subset=variables)
    accepted_set = {tuple(float(r[v]) for v in variables) for _, r in accepted_df.iterrows()}
    multiplier = _read_multiplier(seg_dir)
    return summary, accepted_set, variables, multiplier


def _read_multiplier(seg_dir: Path) -> float:
    """Read the segment's multiplier from config_segment.toml; default 7.0."""
    cfg = seg_dir / "config_segment.toml"
    if cfg.exists():
        try:
            return float(tomllib.loads(cfg.read_text()).get("preprocessing", {}).get("multiplier", 7.0))
        except Exception:
            pass
    return 7.0


def _discover_segments(data_dir: Path, suffix: str, wanted: list[str] | None) -> list[Path]:
    found = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if (d / "data" / f"accepted_cells{suffix}.csv").exists():
            if wanted is None or d.name in wanted:
                found.append(d)
    return found


def _reporting_supersegments(segments_config: Path, available: set[str]) -> dict[str, list[str]]:
    """Map each reporting supersegment to its member segment dir-names that are available."""
    if not segments_config.exists():
        return {}
    try:
        cfg = tomllib.loads(segments_config.read_text())
    except Exception:
        logger.warning(f"Could not parse {segments_config}; skipping supersegments.")
        return {}
    # filter value -> segment key (output dir name)
    filter_to_key = {s.get("segment_filter", key): key for key, s in (cfg.get("segments", {}) or {}).items()}
    groups: dict[str, list[str]] = {}
    tables = {**(cfg.get("reporting_supersegments", {}) or {}), **(cfg.get("supersegments", {}) or {})}
    for name, body in tables.items():
        members = []
        for filt in body.get("segment_filters", []) or []:
            key = filter_to_key.get(filt, filt)
            if key in available:
                members.append(key)
        if members:
            groups[name] = members
    return groups


def _figure_for_unit(units: list[tuple], title: str):
    """Build the figure for one unit from a list of (summary, accepted_set, variables, multiplier)."""
    variables = units[0][2]
    multiplier = units[0][3]
    long_parts = [classify_cells(summary, acc, variables) for summary, acc, _vars, _mult in units]
    cells = aggregate_to_cells(pd.concat(long_parts, ignore_index=True), variables, multiplier)
    if cells.empty:
        return None
    score_vars = variables[:2]
    facet_var = variables[2] if len(variables) > 2 else None
    return build_risk_surface_figure(cells, score_vars, facet_var, title)


def generate_all(
    data_dir: str | Path = "output",
    out_dir: str | Path = "output/risk_surfaces",
    segments_config: str | Path = "segments.toml",
    scenario: str = "base",
    segments: list[str] | None = None,
    include_supersegments: bool = True,
    period: str = "main",
) -> list[Path]:
    """Write a 3D risk-surface HTML per segment (and reporting supersegment). Returns the paths."""
    suffix = _scenario_suffix(scenario)
    ptag = "_mr" if period == "mr" else ""  # distinguishes MR files/index from main
    plabel = "MR period" if period == "mr" else "main period"
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_dirs = _discover_segments(data_dir, suffix, segments)
    if not seg_dirs:
        logger.warning(f"No segments with accepted_cells{suffix}.csv under {data_dir}.")
        return []

    loaded: dict[str, tuple] = {}
    written: list[Path] = []
    for seg_dir in seg_dirs:
        unit = _load_unit(seg_dir, suffix, period)
        if unit is None:
            continue
        loaded[seg_dir.name] = unit
        fig = _figure_for_unit([unit], f"Risk surface — {seg_dir.name} ({scenario}, {plabel})")
        if fig is None:
            logger.warning(f"[{seg_dir.name}] no classifiable cells; skipping.")
            continue
        path = out_dir / f"risk_surface_{seg_dir.name}{ptag}{suffix}.html"
        fig.write_html(str(path), include_plotlyjs="cdn")
        written.append(path)
        logger.info(f"[{seg_dir.name}] wrote {path}")

    if include_supersegments and not segments:
        groups = _reporting_supersegments(Path(segments_config), set(loaded))
        for name, members in groups.items():
            units = [loaded[m] for m in members]
            # All members must share the same cell key to aggregate.
            if len({tuple(u[2]) for u in units}) != 1:
                logger.warning(f"[{name}] members have differing grid variables; skipping supersegment.")
                continue
            fig = _figure_for_unit(units, f"Risk surface — {name} [{'+'.join(members)}] ({scenario}, {plabel})")
            if fig is None:
                continue
            path = out_dir / f"risk_surface_supersegment_{name}{ptag}{suffix}.html"
            fig.write_html(str(path), include_plotlyjs="cdn")
            written.append(path)
            logger.info(f"[{name}] (supersegment: {', '.join(members)}) wrote {path}")

    if written:
        _write_index(out_dir, written, scenario, plabel, ptag)
    return written


def _write_index(out_dir: Path, paths: list[Path], scenario: str, plabel: str = "main period", ptag: str = "") -> Path:
    """A small index.html linking every generated surface."""
    items = "\n".join(f'  <li><a href="{p.name}">{p.stem}</a></li>' for p in sorted(paths))
    html = (
        "<!doctype html><meta charset='utf-8'>"
        f"<title>Risk surfaces ({scenario}, {plabel})</title>"
        "<style>body{font-family:Arial,sans-serif;color:#2C3E50;margin:40px}"
        "h1{font-size:20px}li{margin:4px 0}</style>"
        f"<h1>Risk surfaces — b2_ever_h6 by score bin & audit category ({scenario}, {plabel})</h1>"
        f"<ul>\n{items}\n</ul>"
    )
    idx = out_dir / f"index{ptag}.html"
    idx.write_text(html, encoding="utf-8")
    return idx


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
