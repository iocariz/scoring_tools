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

    if len(df) > max_rows:
        df = df.head(max_rows)

    # Apply float formatting
    formatters = {}
    for col in df.select_dtypes(include="number").columns:
        formatters[col] = lambda x, fmt=float_format: format(x, fmt) if pd.notna(x) else ""

    return df.to_html(index=False, classes=css_class, formatters=formatters, border=0, na_rep="")


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
    for scenario in scenarios:
        suffix = f"_{scenario}" if scenario else ""
        tbl = csv_to_html_table(output_paths.risk_production_summary_csv(suffix))
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
        tbl = csv_to_html_table(output_paths.risk_production_summary_csv(suffix))
        if tbl:
            sa_section.tables.append(tbl)
        opt_tbl = csv_to_html_table(output_paths.optimal_solution_csv(suffix))
        if opt_tbl:
            sa_section.tables.append(f"<h4>Optimal Cutoffs</h4>{opt_tbl}")
        if sa_section.charts or sa_section.tables:
            sections.append(sa_section)

        # MR Validation
        mr_section = ReportSection(id=f"mr-{label}", title=f"MR Validation — {label}")
        mr_tbl = csv_to_html_table(output_paths.mr_risk_production_summary_csv(suffix))
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

    # --- Portfolio Summary ---
    portfolio_section = ReportSection(id="portfolio-summary", title="Portfolio Summary")
    consol_csv = output_base / "consolidated_risk_production.csv"
    tbl = csv_to_html_table(consol_csv, max_rows=200)
    if tbl:
        portfolio_section.tables.append(tbl)
    if portfolio_section.tables:
        sections.append(portfolio_section)

    # --- Consolidated Dashboard ---
    dash_section = ReportSection(id="consolidated-dashboard", title="Consolidated Dashboard")
    div = extract_plotly_div(output_base / "consolidated_risk_production.html")
    if div:
        dash_section.charts.append(div)
    if dash_section.charts:
        sections.append(dash_section)

    # --- Segment Comparison ---
    comparison_section = ReportSection(id="segment-comparison", title="Segment Comparison")
    for seg_name in segments:
        seg_dir = output_base / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)
        # Use base scenario summary
        for suffix in ["_base", ""]:
            tbl = csv_to_html_table(seg_output.risk_production_summary_csv(suffix))
            if tbl:
                comparison_section.tables.append(f"<h4>{seg_name}</h4>{tbl}")
                break
    if comparison_section.tables:
        sections.append(comparison_section)

    # --- Cutoff Comparison ---
    cutoff_section = ReportSection(id="cutoff-comparison", title="Cutoff Comparison")
    for seg_name in segments:
        seg_dir = output_base / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)
        tbl = csv_to_html_table(seg_output.cutoff_summary_wide_csv, max_rows=200)
        if tbl:
            cutoff_section.tables.append(f"<h4>{seg_name}</h4>{tbl}")
    if cutoff_section.tables:
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
