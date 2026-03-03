"""
HTML Report Generation

Produces self-contained HTML reports that aggregate pipeline output artifacts
(CSVs, interactive Plotly charts, JSON alerts) into a single document per segment,
plus a consolidated multi-segment report.

Reports read saved artifacts from disk, decoupling report generation from pipeline
execution and enabling regeneration without re-running the pipeline.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from loguru import logger

from .config import OutputPaths, PreprocessingSettings

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class ReportSection:
    """A single section of the HTML report."""

    id: str
    title: str
    charts: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ReportContext:
    """Full context passed to a Jinja2 report template."""

    title: str
    segment_name: str
    generation_date: str
    date_range: str
    sections: list[ReportSection] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

_PLOTLY_DIV_RE = re.compile(
    r'(<div\s+id="[^"]*"[^>]*>.*?</div>\s*<script\s+type="text/javascript">.*?</script>)',
    re.DOTALL,
)


def extract_plotly_div(html_path: str | Path) -> str | None:
    """Extract the Plotly ``<div>`` + ``<script>`` snippet from a full-page Plotly HTML.

    Strips the bundled ``plotly.js`` so the report can load it once via CDN.
    Returns *None* when the file is missing, empty, or unparseable.
    """
    path = Path(html_path)
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None

    match = _PLOTLY_DIV_RE.search(text)
    return match.group(1) if match else None


def csv_to_html_table(
    csv_path: str | Path,
    max_rows: int = 50,
    float_format: str = ",.2f",
    css_class: str = "report-table",
    exclude_cols: list[str] | None = None,
) -> str | None:
    """Read a CSV and render it as a styled ``<table>`` element.

    Returns *None* when the file is missing or unreadable.
    """
    path = Path(csv_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, OSError, ValueError):
        return None
    if df.empty:
        return None

    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns])

    if len(df) > max_rows:
        df = df.head(max_rows)

    # Apply float formatting
    formatters = {}
    for col in df.select_dtypes(include="number").columns:
        formatters[col] = lambda x, fmt=float_format: format(x, fmt) if pd.notna(x) else ""

    return df.to_html(index=False, classes=css_class, formatters=formatters, border=0, na_rep="")


# ---------------------------------------------------------------------------
# Cutoff comparison helpers (consolidated report)
# ---------------------------------------------------------------------------

_FIXED_COLS = frozenset(
    {
        "accepted",
        "segment",
        "scenario",
        "risk_pct",
        "production",
        "production_ci_lower",
        "production_ci_upper",
        "risk_ci_lower",
        "risk_ci_upper",
    }
)


def _detect_variable_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered list of grid variable columns (everything not a fixed KPI/meta column)."""
    return [c for c in df.columns if c not in _FIXED_COLS]


def _read_cutoff_data(output_base: Path, segments: dict) -> pd.DataFrame | None:
    """Read and concatenate ``cutoff_summary_wide.csv`` from each segment directory.

    Returns *None* when no valid data is found.
    """
    frames: list[pd.DataFrame] = []
    for seg_name in segments:
        seg_output = OutputPaths(base_dir=output_base / seg_name)
        csv_path = Path(seg_output.cutoff_summary_wide_csv)
        if not csv_path.exists():
            logger.warning(f"Cutoff CSV not found for segment {seg_name}: {csv_path}")
            continue
        try:
            df = pd.read_csv(csv_path)
        except (pd.errors.ParserError, OSError, ValueError):
            logger.warning(f"Could not read cutoff CSV for segment {seg_name}: {csv_path}")
            continue
        if df.empty:
            continue
        if "segment" not in df.columns:
            df["segment"] = seg_name
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


_SCENARIO_ORDER = {"pessimistic": 0, "base": 1, "optimistic": 2}


def _build_scenario_kpi_table(df: pd.DataFrame, variable_cols: list[str]) -> str:
    """Build an HTML summary table with one row per (segment, scenario)."""
    rows: list[dict] = []
    group_cols = ["segment", "scenario"]
    for (seg, scen), grp in df.groupby(group_cols, sort=False):
        first = grp.iloc[0]

        risk = first.get("risk_pct", float("nan"))
        production = first.get("production", float("nan"))
        risk_ci_lo = first.get("risk_ci_lower", float("nan"))
        risk_ci_hi = first.get("risk_ci_upper", float("nan"))
        prod_ci_lo = first.get("production_ci_lower", float("nan"))
        prod_ci_hi = first.get("production_ci_upper", float("nan"))

        def _fmt_ci_pair(lo, hi, scale=1.0, dp=2):
            if pd.isna(lo) or pd.isna(hi) or (lo == 0 and hi == 0):
                return "\u2014"
            return f"[{lo / scale:.{dp}f}, {hi / scale:.{dp}f}]"

        rows.append(
            {
                "Segment": seg,
                "Scenario": scen,
                "Risk (%)": f"{risk:.2f}" if pd.notna(risk) else "\u2014",
                "Production (M)": f"{production / 1e6:.2f}" if pd.notna(production) else "\u2014",
                "Risk CI": _fmt_ci_pair(risk_ci_lo, risk_ci_hi),
                "Prod CI (M)": _fmt_ci_pair(prod_ci_lo, prod_ci_hi, scale=1e6),
            }
        )

    summary = pd.DataFrame(rows)
    # Sort by segment then scenario order
    summary["_sort"] = summary["Scenario"].map(_SCENARIO_ORDER).fillna(99)
    summary = summary.sort_values(["Segment", "_sort"]).drop(columns="_sort")

    # Build semantic HTML table
    columns = list(summary.columns)
    numeric_cols = {"Risk (%)", "Production (M)", "Risk CI", "Prod CI (M)"}
    ci_cols = {"Risk CI", "Prod CI (M)"}

    lines: list[str] = ['<table class="report-table" border="0">']
    lines.append("<thead><tr>")
    for col in columns:
        cls = ' class="num"' if col in numeric_cols else ""
        lines.append(f"<th{cls}>{col}</th>")
    lines.append("</tr></thead>")

    lines.append("<tbody>")
    for _, row in summary.iterrows():
        lines.append("<tr>")
        for col in columns:
            val = row[col]
            if col == "Scenario":
                badge_cls = f"badge-{val}" if val in _SCENARIO_ORDER else ""
                lines.append(f'<td><span class="badge {badge_cls}">{val}</span></td>')
            elif col in ci_cols:
                lines.append(f'<td class="num ci">{val}</td>')
            elif col in numeric_cols:
                lines.append(f'<td class="num">{val}</td>')
            else:
                lines.append(f"<td>{val}</td>")
        lines.append("</tr>")
    lines.append("</tbody></table>")

    return "\n".join(lines)


def _build_acceptance_matrices(df: pd.DataFrame, variable_cols: list[str], scenario: str = "base") -> str | None:
    """Build colored HTML pivot grids showing accept/reject patterns for a scenario."""
    if "accepted" not in df.columns:
        return None
    if "scenario" not in df.columns:
        return None

    scen_df = df[df["scenario"] == scenario]
    if scen_df.empty:
        return None

    if len(variable_cols) < 2:
        return None

    col_var = variable_cols[0]  # columns of the pivot
    row_var = variable_cols[1]  # rows of the pivot
    slice_vars = variable_cols[2:]  # additional dimensions → sub-tables

    parts: list[str] = []

    for seg_name, seg_df in scen_df.groupby("segment", sort=True):
        parts.append(f"<h4>{seg_name}</h4>")

        if slice_vars:
            # Group by the slice variables for sub-tables
            slice_groups = seg_df.groupby(slice_vars, sort=True)
            # Cap at 12 sub-tables
            group_items = list(slice_groups)[:12]
        else:
            group_items = [(None, seg_df)]

        parts.append('<div style="display: flex; gap: 1.5rem; flex-wrap: wrap;">')
        for slice_key, slice_df in group_items:
            if slice_key is not None:
                if isinstance(slice_key, tuple):
                    label = ", ".join(f"{v}={k}" for v, k in zip(slice_vars, slice_key))
                else:
                    label = f"{slice_vars[0]}={slice_key}"
                parts.append(f"<div><strong>{label}</strong>")
            else:
                parts.append("<div>")

            pivot = slice_df.pivot_table(index=row_var, columns=col_var, values="accepted", aggfunc="first")
            # Sort index and columns numerically if possible
            try:
                pivot = pivot.sort_index(key=lambda x: pd.to_numeric(x, errors="coerce"))
                pivot = pivot[sorted(pivot.columns, key=lambda x: pd.to_numeric(x, errors="coerce"))]
            except (TypeError, ValueError):
                pass

            parts.append(_render_acceptance_pivot(pivot, col_var, row_var))
            parts.append("</div>")
        parts.append("</div>")

    return "\n".join(parts) if parts else None


def _render_acceptance_pivot(pivot: pd.DataFrame, col_label: str, row_label: str) -> str:
    """Render a single acceptance pivot table as inline-styled HTML."""
    lines: list[str] = []
    lines.append('<table class="report-table" style="font-size: 0.85em; margin: 0.5rem 0;">')
    # Header row
    lines.append("<thead><tr>")
    lines.append(f"<th>{row_label} \\ {col_label}</th>")
    for col in pivot.columns:
        lines.append(f"<th>{col}</th>")
    lines.append("</tr></thead>")

    # Data rows
    lines.append("<tbody>")
    for idx in pivot.index:
        lines.append("<tr>")
        lines.append(f"<th>{idx}</th>")
        for col in pivot.columns:
            val = pivot.loc[idx, col]
            if pd.isna(val):
                cls = "cell-missing"
                text = "\u2014"
            elif val == 1:
                cls = "cell-accept"
                text = "&#10003;"
            else:
                cls = "cell-reject"
                text = "&#10007;"
            lines.append(f'<td class="{cls}">{text}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


_PORTFOLIO_METRIC_ROWS = [
    ("Actual", "actual"),
    ("Optimum", "optimum"),
    ("Swap-in", "swap_in"),
    ("Swap-out", "swap_out"),
]


def _fmt_num(val, fmt=",.2f"):
    """Format a numeric value, returning '\u2014' for NaN."""
    if pd.isna(val):
        return "\u2014"
    return format(val, fmt)


def _fmt_ci(lo, hi, fmt=",.2f"):
    """Format a CI pair, returning '\u2014' when both bounds are zero (not computed) or NaN."""
    if (pd.isna(lo) and pd.isna(hi)) or (lo == 0 and hi == 0):
        return "\u2014"
    return _fmt_num(lo, fmt), _fmt_num(hi, fmt)


def _build_portfolio_metric_table(row: pd.Series, *, show_ci: bool = True) -> str:
    """Render one consolidated-CSV row as a 4-row metric breakdown table.

    Unpivots the wide columns (actual_*, optimum_*, swap_in_*, swap_out_*)
    into four rows with the user-requested columns.
    """
    columns = [
        ("Metric", False),
        ("Risk (%)", True),
        ("Production (\u20ac)", True),
        ("Production (%)", True),
        ("Rejection Rate (%)", True),
    ]
    if show_ci:
        columns += [
            ("Production CI Lower", True),
            ("Production CI Upper", True),
            ("Risk CI Lower", True),
            ("Risk CI Upper", True),
        ]

    lines: list[str] = ['<table class="report-table" border="0">']
    lines.append("<thead><tr>")
    for col_name, is_num in columns:
        cls = ' class="num"' if is_num else ""
        lines.append(f"<th{cls}>{col_name}</th>")
    lines.append("</tr></thead>")

    lines.append("<tbody>")
    for label, prefix in _PORTFOLIO_METRIC_ROWS:
        risk = row.get(f"{prefix}_risk_pct", float("nan"))
        prod = row.get(f"{prefix}_production", float("nan"))

        # Production delta (%) and CIs only for Optimum
        if prefix == "optimum":
            prod_pct = row.get("production_delta_pct", float("nan"))
            prod_ci_lo = row.get("production_ci_lower", float("nan"))
            prod_ci_hi = row.get("production_ci_upper", float("nan"))
            risk_ci_lo = row.get("risk_ci_lower", float("nan"))
            risk_ci_hi = row.get("risk_ci_upper", float("nan"))
        else:
            prod_pct = prod_ci_lo = prod_ci_hi = risk_ci_lo = risk_ci_hi = float("nan")

        # Rejection rate: only for Actual and Optimum
        if prefix == "actual":
            rej_rate = row.get("actual_rejection_rate_pct", float("nan"))
        elif prefix == "optimum":
            rej_rate = row.get("optimum_rejection_rate_pct", float("nan"))
        else:
            rej_rate = float("nan")

        # Format production delta with sign
        if pd.notna(prod_pct):
            sign = "+" if prod_pct > 0 else ""
            prod_pct_str = f"{sign}{prod_pct:.1f}"
        else:
            prod_pct_str = "\u2014"

        # Format CI pairs (show — when both bounds are 0 = not computed)
        prod_ci = _fmt_ci(prod_ci_lo, prod_ci_hi, ",.0f")
        risk_ci = _fmt_ci(risk_ci_lo, risk_ci_hi)
        prod_ci_lo_str = prod_ci if isinstance(prod_ci, str) else prod_ci[0]
        prod_ci_hi_str = "\u2014" if isinstance(prod_ci, str) else prod_ci[1]
        risk_ci_lo_str = risk_ci if isinstance(risk_ci, str) else risk_ci[0]
        risk_ci_hi_str = "\u2014" if isinstance(risk_ci, str) else risk_ci[1]

        # Row CSS class for targeted styling
        if prefix == "actual":
            row_cls = "row-actual"
        elif prefix == "optimum":
            row_cls = "row-optimum"
        else:
            row_cls = "row-swap"

        # KPI highlight on key risk/production cells for Actual and Optimum
        kpi = ' kpi-highlight' if prefix in ("actual", "optimum") else ""

        lines.append(f'<tr class="{row_cls}">')
        lines.append(f"<td><strong>{label}</strong></td>")
        lines.append(f'<td class="num{kpi}">{_fmt_num(risk)}</td>')
        lines.append(f'<td class="num{kpi}">{_fmt_num(prod, ",.0f")}</td>')
        lines.append(f'<td class="num">{prod_pct_str}</td>')
        lines.append(f'<td class="num">{_fmt_num(rej_rate)}</td>')
        if show_ci:
            lines.append(f'<td class="num ci">{prod_ci_lo_str}</td>')
            lines.append(f'<td class="num ci">{prod_ci_hi_str}</td>')
            lines.append(f'<td class="num ci">{risk_ci_lo_str}</td>')
            lines.append(f'<td class="num ci">{risk_ci_hi_str}</td>')
        lines.append("</tr>")

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _portfolio_group_list(period_df: pd.DataFrame) -> list[tuple[str, str]]:
    """Return an ordered list of ``(group_col_value, display_name)`` for portfolio sections.

    Supersegment aggregates come first, then standalone segments, then TOTAL last.
    """
    all_groups = period_df["group"].unique()
    supersegment_names = sorted(g[len("supersegment_") :] for g in all_groups if g.startswith("supersegment_"))

    group_list: list[tuple[str, str]] = []
    if supersegment_names:
        for ss in supersegment_names:
            group_list.append((f"supersegment_{ss}", ss))
    else:
        for g in sorted(all_groups):
            if g != "TOTAL":
                group_list.append((g, g.replace("segment_", "")))
    if "TOTAL" in all_groups:
        group_list.append(("TOTAL", "Total"))
    return group_list


def _build_portfolio_summary_sections(csv_path: Path) -> list[ReportSection]:
    """Build portfolio summary: one section per supersegment group, each with per-scenario tables.

    Main period sections are emitted first.  If MR-period data exists, a parallel
    set of sections (titled "MR Validation") is appended afterwards.
    """
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, OSError, ValueError):
        return []
    if df.empty or "group" not in df.columns:
        return []

    has_period = "period" in df.columns
    scenario_order = ["pessimistic", "base", "optimistic"]

    # Determine which periods to render
    if has_period:
        periods = [("main", "Portfolio Summary", "portfolio"), ("mr", "MR Validation", "mr-portfolio")]
    else:
        periods = [(None, "Portfolio Summary", "portfolio")]

    all_sections: list[ReportSection] = []

    for period_value, section_prefix, id_prefix in periods:
        if period_value is not None:
            period_df = df[df["period"] == period_value]
        else:
            period_df = df
        if period_df.empty:
            continue

        group_list = _portfolio_group_list(period_df)

        for group_value, display_name in group_list:
            safe_id = display_name.lower().replace(" ", "-").replace("/", "-")
            section = ReportSection(
                id=f"{id_prefix}-{safe_id}",
                title=f"{section_prefix} \u2014 {display_name}",
            )

            group_df = period_df[period_df["group"] == group_value]
            if "scenario" in group_df.columns:
                available = [s for s in scenario_order if s in group_df["scenario"].values]
                if not available:
                    available = list(group_df["scenario"].unique())
            else:
                available = [None]

            for scenario in available:
                scen_df = group_df[group_df["scenario"] == scenario] if scenario else group_df
                if scen_df.empty:
                    continue

                badge_cls = f"badge-{scenario}" if scenario and scenario in _SCENARIO_ORDER else ""
                label = scenario.title() if scenario else "Base"
                heading = f'<h4><span class="badge {badge_cls}">{label}</span></h4>'
                section.tables.append(
                    heading + _build_portfolio_metric_table(scen_df.iloc[0], show_ci=period_value != "mr")
                )

            if section.tables:
                all_sections.append(section)

    return all_sections


def _build_cutoff_comparison_section(output_base: Path, segments: dict) -> ReportSection | None:
    """Build the redesigned cutoff comparison section for the consolidated report."""
    df = _read_cutoff_data(output_base, segments)
    if df is None:
        return None

    variable_cols = _detect_variable_cols(df)
    section = ReportSection(id="cutoff-comparison", title="Cutoff Comparison")

    # KPI summary table
    if "scenario" in df.columns and "segment" in df.columns:
        kpi_html = _build_scenario_kpi_table(df, variable_cols)
        section.tables.append("<h4>Scenario KPI Summary</h4>" + kpi_html)

    # Acceptance matrices (base scenario only)
    if "accepted" in df.columns:
        matrices_html = _build_acceptance_matrices(df, variable_cols)
        if matrices_html:
            section.tables.append("<h4>Acceptance Matrices (Base Scenario)</h4>" + matrices_html)

    if not section.tables:
        return None
    return section


# ---------------------------------------------------------------------------
# Segment report builder
# ---------------------------------------------------------------------------


def build_segment_report(
    output_paths: OutputPaths,
    settings: PreprocessingSettings,
    scenarios: list[str],
) -> ReportContext:
    """Build a :class:`ReportContext` for a single segment by reading saved artifacts."""
    segment = settings.segment_filter
    date_range = f"{settings.date_ini_book_obs} to {settings.date_fin_book_obs}"

    sections: list[ReportSection] = []

    # --- Executive Summary ---
    exec_section = ReportSection(id="executive-summary", title="Executive Summary")
    config_notes = [
        f"Segment: {segment}",
        f"Date range: {date_range}",
        f"Variables: {', '.join(settings.variables)}",
        f"Optimum risk: {settings.optimum_risk}",
        f"Risk step: {settings.risk_step}",
        f"Multiplier: {settings.multiplier}",
    ]
    if settings.reject_inference_method != "none":
        config_notes.append(f"Reject inference: {settings.reject_inference_method}")
    exec_section.notes = config_notes

    # Add per-scenario summary tables
    _exclude_todu = ["todu_30ever_h6", "todu_amt_pile_h6"]
    for scenario in scenarios:
        suffix = f"_{scenario}" if scenario else ""
        tbl = csv_to_html_table(output_paths.risk_production_summary_csv(suffix), exclude_cols=_exclude_todu)
        if tbl:
            exec_section.tables.append(f"<h4>Scenario: {scenario or 'base'}</h4>{tbl}")
    sections.append(exec_section)

    # --- Preprocessing ---
    prep_section = ReportSection(id="preprocessing", title="Preprocessing")
    for html_path in [output_paths.risk_vs_production_html, output_paths.transformation_rate_html]:
        div = extract_plotly_div(html_path)
        if div:
            prep_section.charts.append(div)
    if prep_section.charts:
        sections.append(prep_section)

    # --- Model Inference ---
    inf_section = ReportSection(id="model-inference", title="Model Inference")
    div = extract_plotly_div(output_paths.b2_visualization_html)
    if div:
        inf_section.charts.append(div)
    if inf_section.charts:
        sections.append(inf_section)

    # --- Per-scenario sections ---
    for scenario in scenarios:
        suffix = f"_{scenario}" if scenario else ""
        label = scenario or "base"

        # Scenario Analysis
        sa_section = ReportSection(id=f"scenario-{label}", title=f"Scenario Analysis — {label}")
        div = extract_plotly_div(output_paths.risk_production_visualizer_html(suffix))
        if div:
            sa_section.charts.append(div)
        tbl = csv_to_html_table(output_paths.risk_production_summary_csv(suffix), exclude_cols=_exclude_todu)
        if tbl:
            sa_section.tables.append(tbl)
        opt_tbl = csv_to_html_table(output_paths.optimal_solution_csv(suffix))
        if opt_tbl:
            sa_section.tables.append(f"<h4>Optimal Cutoffs</h4>{opt_tbl}")
        # N>2: acceptance grid per income bin
        acc_div = extract_plotly_div(output_paths.acceptance_grid_html(suffix))
        if acc_div:
            sa_section.charts.append(acc_div)
        if sa_section.charts or sa_section.tables:
            sections.append(sa_section)

        # MR Validation
        mr_section = ReportSection(id=f"mr-{label}", title=f"MR Validation — {label}")
        mr_tbl = csv_to_html_table(output_paths.mr_risk_production_summary_csv(suffix), exclude_cols=_exclude_todu)
        if mr_tbl:
            mr_section.tables.append(mr_tbl)
        if mr_section.tables:
            sections.append(mr_section)

        # Stability
        stab_section = ReportSection(id=f"stability-{label}", title=f"Stability — {label}")
        div = extract_plotly_div(output_paths.stability_report_html(suffix))
        if div:
            stab_section.charts.append(div)
        psi_tbl = csv_to_html_table(output_paths.stability_psi_csv(suffix))
        if psi_tbl:
            stab_section.tables.append(psi_tbl)
        if stab_section.charts or stab_section.tables:
            sections.append(stab_section)

    # --- Trends ---
    trends_section = ReportSection(id="trends", title="Trends")
    div = extract_plotly_div(output_paths.metric_trends_html(segment))
    if div:
        trends_section.charts.append(div)
    anom_tbl = csv_to_html_table(output_paths.trend_anomalies_csv(segment))
    if anom_tbl:
        trends_section.tables.append(f"<h4>Anomalies</h4>{anom_tbl}")
    if trends_section.charts or trends_section.tables:
        sections.append(trends_section)

    # --- Cutoff Reference ---
    cutoff_section = ReportSection(id="cutoff-reference", title="Cutoff Reference")
    tbl = csv_to_html_table(output_paths.cutoff_summary_wide_csv, max_rows=200)
    if tbl:
        cutoff_section.tables.append(tbl)
    if cutoff_section.tables:
        sections.append(cutoff_section)

    return ReportContext(
        title=f"Segment Report — {segment}",
        segment_name=segment,
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        date_range=date_range,
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Consolidated report builder
# ---------------------------------------------------------------------------


def build_consolidated_report(
    output_base: str | Path,
    segments: dict,
    supersegments: dict,
) -> ReportContext:
    """Build a :class:`ReportContext` for the multi-segment consolidated report."""
    output_base = Path(output_base)
    sections: list[ReportSection] = []

    # --- Portfolio Summary (per supersegment + total) ---
    consol_csv = output_base / "consolidated_risk_production.csv"
    sections.extend(_build_portfolio_summary_sections(consol_csv))

    # --- Consolidated Dashboard ---
    dash_section = ReportSection(id="consolidated-dashboard", title="Consolidated Dashboard")
    div = extract_plotly_div(output_base / "consolidated_risk_production.html")
    if div:
        dash_section.charts.append(div)
    if dash_section.charts:
        sections.append(dash_section)

    # --- Segment Comparison ---
    _exclude_todu = ["todu_30ever_h6", "todu_amt_pile_h6"]
    comparison_section = ReportSection(id="segment-comparison", title="Segment Comparison")
    for seg_name in segments:
        seg_dir = output_base / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)
        # Use base scenario summary
        for suffix in ["_base", ""]:
            tbl = csv_to_html_table(seg_output.risk_production_summary_csv(suffix), exclude_cols=_exclude_todu)
            if tbl:
                comparison_section.tables.append(f"<h4>{seg_name}</h4>{tbl}")
                break
    if comparison_section.tables:
        sections.append(comparison_section)

    # --- Cutoff Comparison ---
    cutoff_section = _build_cutoff_comparison_section(output_base, segments)
    if cutoff_section is not None:
        sections.append(cutoff_section)

    segment_list = ", ".join(segments.keys()) if segments else "none"
    return ReportContext(
        title="Consolidated Report",
        segment_name="all",
        generation_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        date_range=f"Segments: {segment_list}",
        sections=sections,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_report(
    context: ReportContext,
    output_path: str | Path,
    template_name: str = "segment_report.html",
) -> Path:
    """Render a Jinja2 template with the given *context* and write it to *output_path*.

    Returns the path of the written file.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # HTML content already escaped / trusted
    )
    template = env.get_template(template_name)

    html = template.render(
        title=context.title,
        segment_name=context.segment_name,
        generation_date=context.generation_date,
        date_range=context.date_range,
        sections=context.sections,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info(f"Report written to {output_path}")
    return output_path
