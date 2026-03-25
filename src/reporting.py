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
from html import escape
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
# Cutoff reference table (segment report)
# ---------------------------------------------------------------------------

# Columns that hold KPI / metadata (not bin cutoff values)
_CUTOFF_META_COLS = {"segment", "scenario", "risk_pct", "production"}
_CUTOFF_CI_COLS = {"production_ci_lower", "production_ci_upper", "risk_ci_lower", "risk_ci_upper"}
_CUTOFF_CELL_META = {"accepted"}  # N>2 cell-level summary column

_SCENARIO_BADGE = {
    "pessimistic": "badge-pessimistic",
    "base": "badge-base",
    "optimistic": "badge-optimistic",
}


def _build_cutoff_reference_table(csv_path: str | Path, max_rows: int = 200) -> str | None:
    """Render the cutoff reference CSV as a readable, scenario-grouped HTML table.

    Improvements over raw ``csv_to_html_table``:
    - Scenario group headers with colored badges
    - Bin/cutoff columns visually separated from KPI columns
    - Numeric formatting and risk color-coding
    - CI columns collapsed into a single column per metric
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

    has_scenario = "scenario" in df.columns
    has_accepted = "accepted" in df.columns

    # Identify bin / cutoff columns (everything that's not a known meta/CI column)
    all_meta = _CUTOFF_META_COLS | _CUTOFF_CI_COLS | _CUTOFF_CELL_META
    bin_cols = [c for c in df.columns if c not in all_meta]

    # Determine which CI columns are present
    ci_pairs = []
    if "production_ci_lower" in df.columns and "production_ci_upper" in df.columns:
        ci_pairs.append(("production_ci_lower", "production_ci_upper", "Production CI"))
    if "risk_ci_lower" in df.columns and "risk_ci_upper" in df.columns:
        ci_pairs.append(("risk_ci_lower", "risk_ci_upper", "Risk CI"))

    # Build display column order: bin cols, then KPIs
    kpi_display = []
    if "risk_pct" in df.columns:
        kpi_display.append(("risk_pct", "Risk (%)"))
    if "production" in df.columns:
        kpi_display.append(("production", "Production"))
    for _lo_col, _hi_col, label in ci_pairs:
        kpi_display.append((f"_ci_{label}", label))
    if has_accepted:
        kpi_display.append(("accepted", "Accepted"))

    # Pretty-print bin column headers
    bin_labels = []
    for c in bin_cols:
        label = c.replace("_", " ").replace("bin ", "Bin ")
        if label[0].islower():
            label = label[0].upper() + label[1:]
        bin_labels.append(label)

    lines: list[str] = []
    lines.append('<table class="report-table cutoff-ref-table" border="0">')

    # ── Header ──
    lines.append("<thead><tr>")
    for label in bin_labels:
        lines.append(f'<th class="col-bin">{label}</th>')
    for _col_key, label in kpi_display:
        lines.append(f'<th class="num">{label}</th>')
    lines.append("</tr></thead>")

    # ── Body (optionally grouped by scenario) ──
    lines.append("<tbody>")
    if has_scenario:
        scenarios = df["scenario"].unique()
        for scenario in scenarios:
            badge_cls = _SCENARIO_BADGE.get(scenario, "")
            lines.append(
                f'<tr class="scenario-header"><td colspan="{len(bin_cols) + len(kpi_display)}">'
                f'<span class="badge {badge_cls}">{scenario.title()}</span></td></tr>'
            )
            sub = df[df["scenario"] == scenario]
            for _, row in sub.iterrows():
                lines.append(_cutoff_ref_row(row, bin_cols, kpi_display, ci_pairs, has_accepted))
    else:
        for _, row in df.iterrows():
            lines.append(_cutoff_ref_row(row, bin_cols, kpi_display, ci_pairs, has_accepted))

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _cutoff_ref_row(
    row: pd.Series,
    bin_cols: list[str],
    kpi_display: list[tuple[str, str]],
    ci_pairs: list[tuple[str, str, str]],
    has_accepted: bool,
) -> str:
    """Render a single row of the cutoff reference table."""
    cells: list[str] = []

    # Bin / cutoff value cells
    for c in bin_cols:
        val = row.get(c, float("nan"))
        if pd.isna(val):
            cells.append('<td class="col-bin">\u2014</td>')
        elif isinstance(val, float) and val == int(val):
            cells.append(f'<td class="col-bin">{int(val)}</td>')
        elif isinstance(val, float):
            cells.append(f'<td class="col-bin">{val:,.2f}</td>')
        else:
            cells.append(f'<td class="col-bin">{escape(str(val))}</td>')

    # KPI cells
    ci_lookup = {label: (lo, hi) for lo, hi, label in ci_pairs}
    for col_key, label in kpi_display:
        if col_key == "risk_pct":
            val = row.get("risk_pct", float("nan"))
            if pd.notna(val):
                cls = "num risk-high" if val > 5 else "num risk-med" if val > 3 else "num risk-low"
                cells.append(f'<td class="{cls}">{val:,.2f}%</td>')
            else:
                cells.append('<td class="num">\u2014</td>')
        elif col_key == "production":
            val = row.get("production", float("nan"))
            cells.append(f'<td class="num">{format(val, ",.0f") if pd.notna(val) else "\u2014"}</td>')
        elif col_key.startswith("_ci_"):
            lo_col, hi_col = ci_lookup[label]
            lo = row.get(lo_col, float("nan"))
            hi = row.get(hi_col, float("nan"))
            if pd.notna(lo) and pd.notna(hi) and not (lo == 0 and hi == 0):
                fmt = ",.0f" if "production" in label.lower() else ",.2f"
                cells.append(f'<td class="num ci">[{format(lo, fmt)}, {format(hi, fmt)}]</td>')
            else:
                cells.append('<td class="num ci">\u2014</td>')
        elif col_key == "accepted":
            val = row.get("accepted", float("nan"))
            if pd.notna(val):
                cls = "cell-accept" if val == 1 else "cell-reject"
                text = "&#10003;" if val == 1 else "&#10007;"
                cells.append(f'<td class="{cls}">{text}</td>')
            else:
                cells.append('<td class="cell-missing">\u2014</td>')
        else:
            val = row.get(col_key, float("nan"))
            if pd.notna(val):
                try:
                    rendered = format(float(val), ",.2f")
                except (TypeError, ValueError):
                    rendered = escape(str(val))
                cells.append(f'<td class="num">{rendered}</td>')
            else:
                cells.append('<td class="num">\u2014</td>')

    return f"<tr>{''.join(cells)}</tr>"


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

    if len(variable_cols) < 1:
        return None

    # 1D: horizontal acceptance strip
    if len(variable_cols) == 1:
        return _build_acceptance_strip_1d(scen_df, variable_cols[0])

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

        parts.append('<div class="matrix-grid">')
        for slice_key, slice_df in group_items:
            if slice_key is not None:
                if isinstance(slice_key, tuple):
                    label = ", ".join(f"{v}={k}" for v, k in zip(slice_vars, slice_key))
                else:
                    label = f"{slice_vars[0]}={slice_key}"
                parts.append(f'<div class="matrix-cell"><div class="matrix-label">{label}</div>')
            else:
                parts.append('<div class="matrix-cell">')

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


def _build_acceptance_strip_1d(df: pd.DataFrame, var_col: str) -> str:
    """Render a horizontal acceptance strip for 1D optimization."""
    parts: list[str] = []

    for seg_name, seg_df in df.groupby("segment", sort=True):
        parts.append(f"<h4>{seg_name}</h4>")

        # Sort by bin value
        seg_sorted = seg_df.sort_values(var_col)
        bins = seg_sorted[var_col].values
        accepted = seg_sorted["accepted"].values

        n_accepted = int(accepted.sum())
        n_total = len(accepted)

        # Summary line
        parts.append(
            f'<div class="strip-summary">{n_accepted}/{n_total} bins accepted '
            f'({100 * n_accepted / n_total:.0f}%)</div>'
        )

        # Horizontal strip
        parts.append('<div class="acceptance-strip">')
        parts.append(f'<div class="strip-label">{var_col}</div>')
        parts.append('<div class="strip-cells">')
        for bv, acc in zip(bins, accepted):
            cls = "m-ok" if acc == 1 else "m-no"
            label = int(bv) if isinstance(bv, float) and bv == int(bv) else bv
            parts.append(f'<div class="strip-cell {cls}"><span class="strip-bin">{label}</span></div>')
        parts.append("</div></div>")

    return "\n".join(parts) if parts else None


def _build_optimal_cutoff_1d(
    csv_path: Path, var_name: str, settings: PreprocessingSettings
) -> str | None:
    """Render the optimal solution CSV for 1D as a table with decoded acceptance mask.

    Instead of showing the raw comma-separated mask, displays a clean table with
    the selected Pareto point metrics and a visual acceptance strip.
    """
    try:
        df = pd.read_csv(csv_path)
    except (pd.errors.ParserError, OSError, ValueError):
        return None
    if df.empty:
        return None

    # Get bin values from settings
    bin_config = settings.bins.get(var_name)
    if bin_config and bin_config.bin_edges:
        n_bins = len(bin_config.bin_edges) - 1
        bin_labels = list(range(1, n_bins + 1))
    else:
        # Infer from mask length
        if "acceptance_mask" in df.columns:
            first_mask = str(df["acceptance_mask"].iloc[0])
            n_bins = len(first_mask.split(","))
            bin_labels = list(range(1, n_bins + 1))
        else:
            return None

    # Show the selected (last qualifying) row's KPIs in a summary table
    kpi_cols = ["sol_fac", "b2_ever_h6", "oa_amt_h0", "b2_ever_h6_cut", "oa_amt_h0_cut"]
    available_kpis = [c for c in kpi_cols if c in df.columns]

    lines: list[str] = []

    # KPI table for the selected solution
    if available_kpis:
        lines.append('<table class="report-table" border="0">')
        lines.append("<thead><tr>")
        kpi_labels = {
            "sol_fac": "Solution",
            "b2_ever_h6": "Risk (%)",
            "oa_amt_h0": "Production",
            "b2_ever_h6_cut": "Risk Cut (%)",
            "oa_amt_h0_cut": "Production Cut",
        }
        for col in available_kpis:
            lines.append(f'<th class="num">{kpi_labels.get(col, col)}</th>')
        lines.append("<th>Acceptance Pattern</th></tr></thead>")
        lines.append("<tbody>")

        # Show up to 10 Pareto solutions
        for _, row in df.head(10).iterrows():
            lines.append("<tr>")
            for col in available_kpis:
                val = row.get(col, float("nan"))
                if pd.notna(val):
                    if col in ("b2_ever_h6", "b2_ever_h6_cut"):
                        cls = "num risk-high" if val > 5 else "num risk-med" if val > 3 else "num risk-low"
                        lines.append(f'<td class="{cls}">{val:,.2f}%</td>')
                    elif col in ("oa_amt_h0", "oa_amt_h0_cut"):
                        lines.append(f'<td class="num">{val:,.0f}</td>')
                    else:
                        lines.append(f'<td class="num">{val}</td>')
                else:
                    lines.append('<td class="num">&mdash;</td>')

            # Inline acceptance strip
            mask_str = str(row.get("acceptance_mask", ""))
            if mask_str:
                bits = mask_str.split(",")
                strip = '<td><div class="strip-cells" style="display:inline-flex">'
                for i, bit in enumerate(bits):
                    cls = "m-ok" if bit.strip() == "1" else "m-no"
                    bl = bin_labels[i] if i < len(bin_labels) else i + 1
                    strip += f'<div class="strip-cell {cls}" style="min-width:24px;height:24px">'
                    strip += f'<span class="strip-bin" style="font-size:0.6rem">{bl}</span></div>'
                strip += "</div></td>"
                lines.append(strip)
            else:
                lines.append("<td>&mdash;</td>")
            lines.append("</tr>")

        lines.append("</tbody></table>")

    return "\n".join(lines) if lines else None


def _render_acceptance_pivot(pivot: pd.DataFrame, col_label: str, row_label: str) -> str:
    """Render a single acceptance pivot as a compact color-coded heatmap grid."""
    lines: list[str] = []
    lines.append('<table class="matrix-table">')
    # Axis labels as a caption-like first row
    lines.append(f'<tr><td class="matrix-corner">{row_label}\u2009\\\u2009{col_label}</td>')
    for col in pivot.columns:
        lines.append(f'<td class="matrix-col-hdr">{col}</td>')
    lines.append("</tr>")

    # Data rows
    for idx in pivot.index:
        lines.append("<tr>")
        lines.append(f'<td class="matrix-row-hdr">{idx}</td>')
        for col in pivot.columns:
            val = pivot.loc[idx, col]
            if pd.isna(val):
                cls = "m-na"
            elif val == 1:
                cls = "m-ok"
            else:
                cls = "m-no"
            lines.append(f'<td class="{cls}"></td>')
        lines.append("</tr>")

    lines.append("</table>")
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
            legend = (
                '<div class="matrix-legend">'
                '<span><i style="background: var(--accent-green)"></i> Accept</span>'
                '<span><i style="background: var(--accent-red); opacity: 0.7"></i> Reject</span>'
                '<span><i style="background: var(--border)"></i> N/A</span>'
                "</div>"
            )
            section.tables.append("<h4>Acceptance Matrices (Base Scenario)</h4>" + legend + matrices_html)

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
    _exclude_todu = ["todu_30ever_h6", "todu_amt_pile_h6", "Total Demand (€)"]
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
    is_1d = len(settings.variables) == 1
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
        # Optimal cutoffs: for 1D/N-d, decode acceptance_mask into a visual strip
        opt_csv = Path(output_paths.optimal_solution_csv(suffix))
        if is_1d and opt_csv.exists():
            opt_html = _build_optimal_cutoff_1d(opt_csv, settings.variables[0], settings)
            if opt_html:
                sa_section.tables.append(f"<h4>Optimal Cutoffs</h4>{opt_html}")
        else:
            opt_tbl = csv_to_html_table(opt_csv)
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
    tbl = _build_cutoff_reference_table(output_paths.cutoff_summary_wide_csv)
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
    _exclude_todu = ["todu_30ever_h6", "todu_amt_pile_h6", "Total Demand (€)"]
    _exclude_todu_mr = _exclude_todu + ["todu_30ever_h3", "todu_amt_pile_h3"]

    # Main period
    comparison_section = ReportSection(id="segment-comparison", title="Segment Comparison — Main Period")
    for seg_name in segments:
        seg_dir = output_base / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)
        for suffix in ["_base", ""]:
            tbl = csv_to_html_table(seg_output.risk_production_summary_csv(suffix), exclude_cols=_exclude_todu)
            if tbl:
                comparison_section.tables.append(f"<h4>{seg_name}</h4>{tbl}")
                break
    if comparison_section.tables:
        sections.append(comparison_section)

    # MR period
    mr_comparison_section = ReportSection(id="segment-comparison-mr", title="Segment Comparison — MR Period")
    for seg_name in segments:
        seg_dir = output_base / seg_name
        seg_output = OutputPaths(base_dir=seg_dir)
        for suffix in ["_base", ""]:
            tbl = csv_to_html_table(seg_output.mr_risk_production_summary_csv(suffix), exclude_cols=_exclude_todu_mr)
            if tbl:
                mr_comparison_section.tables.append(f"<h4>{seg_name}</h4>{tbl}")
                break
    if mr_comparison_section.tables:
        sections.append(mr_comparison_section)

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
