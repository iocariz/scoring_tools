"""
Dashboard for Risk Scoring Optimization results visualization.

Provides an interactive Dash application to explore:
- Scenario comparisons (pessimistic, base, optimistic)
- Main period results and visualizations
- MR (Recent Monitoring) period results
- Cutoff Explorer for interactive what-if analysis

Supports both single-run output (data/, images/) and batch output (output/{segment}/).

Usage:
    python dashboard.py                    # Auto-detect output location
    python dashboard.py --output output    # Use batch output directory
    python dashboard.py --segment seg1     # View specific segment
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from dash.dependencies import MATCH
from flask import send_from_directory
from loguru import logger

from src.styles import (
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_NEUTRAL,
    COLOR_PRIMARY,
    COLOR_PRODUCTION,
    COLOR_RISK,
    apply_plotly_style,
)

# --- Constants ---
SCENARIO_ORDER: dict[str, int] = {"pessimistic": 0, "base": 1, "optimistic": 2}

IFRAME_STYLE: dict[str, Any] = {"width": "100%", "height": "800px", "border": "none", "marginBottom": "20px"}

TABLE_STYLE: dict[str, Any] = {
    "fontSize": "0.9rem",
}

# --- Global State ---
# These will be set at startup based on CLI args or auto-detection
OUTPUT_BASE: Path = Path(".")
CURRENT_SEGMENT: str | None = None


def get_data_dir(segment: str | None = None) -> Path:
    """Get the data directory for a segment."""
    if segment:
        return OUTPUT_BASE / segment / "data"
    # Check if we're in batch mode (output/ structure) or single mode
    if (OUTPUT_BASE / "data").exists():
        return OUTPUT_BASE / "data"
    return Path("data")


def get_images_dir(segment: str | None = None) -> Path:
    """Get the images directory for a segment."""
    if segment:
        return OUTPUT_BASE / segment / "images"
    if (OUTPUT_BASE / "images").exists():
        return OUTPUT_BASE / "images"
    return Path("images")


def get_available_segments() -> list[str]:
    """Get list of available segments from output directory."""
    segments = []

    if not OUTPUT_BASE.exists():
        return segments

    for item in OUTPUT_BASE.iterdir():
        if item.is_dir() and not item.name.startswith(("_", ".")):
            # Check if it has data subdirectory (valid segment output)
            if (item / "data").exists():
                segments.append(item.name)

    return sorted(segments)


def get_scenarios(segment: str | None = None) -> list[str]:
    """
    Parse available scenarios from data directory.

    Scenarios are named: pessimistic, base, optimistic
    - pessimistic: optimum - step (more conservative)
    - base: optimum risk threshold
    - optimistic: optimum + step (more aggressive)

    Returns:
        List of scenario names sorted by order (pessimistic, base, optimistic).
    """
    data_dir = get_data_dir(segment)

    if not data_dir.exists():
        return []

    files = list(data_dir.glob("risk_production_summary_table_*.csv"))
    scenarios: list[str] = []

    for f in files:
        filename = f.name

        # Skip MR files for this discovery (they are secondary)
        if "_mr" in filename:
            continue

        # Match new named format: risk_production_summary_table_pessimistic.csv
        match = re.search(r"risk_production_summary_table_(pessimistic|base|optimistic)\.csv", filename)
        if match:
            scenarios.append(match.group(1))
            continue

        # Match legacy numeric format: risk_production_summary_table_1.0.csv
        match = re.search(r"risk_production_summary_table_(\d+\.\d+)\.csv", filename)
        if match:
            scenarios.append(match.group(1))

    # Also check for base file without suffix
    if (data_dir / "risk_production_summary_table.csv").exists() and "base" not in scenarios:
        scenarios.append("base")

    # Sort by scenario order (pessimistic, base, optimistic) then alphabetically for others
    return sorted(scenarios, key=lambda x: (SCENARIO_ORDER.get(x, 99), x))


def get_scenario_paths(scenario: str, segment: str | None = None) -> dict[str, Any]:
    """
    Get all file paths for a given scenario with fallbacks.

    Args:
        scenario: Scenario name (e.g., 'base', 'pessimistic', 'optimistic').
        segment: Optional segment name for batch output.

    Returns:
        Dictionary with paths for main_csv, main_viz, mr_csv, mr_viz.
    """
    data_dir = get_data_dir(segment)
    suffix = f"_{scenario}"

    # Build static URL path for serving
    if segment:
        static_prefix = f"/static/{segment}"
    else:
        static_prefix = "/static"

    paths = {
        "main_csv": data_dir / f"risk_production_summary_table{suffix}.csv",
        "main_viz": f"{static_prefix}/risk_production_visualizer{suffix}.html",
        "mr_csv": data_dir / f"risk_production_summary_table_mr{suffix}.csv",
        "mr_viz": f"{static_prefix}/b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html",
        "stability": f"{static_prefix}/stability_report{suffix}.html",
        "cutoffs": data_dir / f"optimal_solution{suffix}.csv",
        "summary_data": data_dir / f"data_summary_desagregado{suffix}.csv",
    }

    # Fallback to unsuffixed files for base scenario
    if scenario == "base":
        fallbacks = {
            "main_csv": data_dir / "risk_production_summary_table.csv",
            "main_viz": f"{static_prefix}/risk_production_visualizer.html",
            "mr_csv": data_dir / "risk_production_summary_table_mr.csv",
            "mr_viz": f"{static_prefix}/b2_ever_h6_vs_octroi_and_risk_score_mr.html",
            "stability": f"{static_prefix}/stability_report.html",
            "cutoffs": data_dir / "optimal_solution.csv",
            "summary_data": data_dir / "data_summary_desagregado.csv",
        }
        for key in ["main_csv", "mr_csv", "cutoffs", "summary_data"]:
            if not paths[key].exists() and fallbacks[key].exists():
                paths[key] = fallbacks[key]

        # For viz paths, check if the file exists in images dir
        images_dir = get_images_dir(segment)
        for key in ["main_viz", "mr_viz", "stability"]:
            current_file = images_dir / paths[key].split("/")[-1]
            fallback_file = images_dir / fallbacks[key].split("/")[-1]
            if not current_file.exists() and fallback_file.exists():
                paths[key] = fallbacks[key]

    return paths


def file_exists_for_viz(viz_path: str, segment: str | None = None) -> bool:
    """Check if a visualization file exists."""
    filename = viz_path.split("/")[-1]
    images_dir = get_images_dir(segment)
    return (images_dir / filename).exists()


def load_table(csv_path: Path) -> html.Div:
    """
    Load CSV and return a Dash-compatible table.

    Args:
        csv_path: Path to CSV file.

    Returns:
        Dash component with table or error message.
    """
    if not csv_path.exists():
        return html.Div(f"Data file not found: {csv_path.name}", className="text-warning p-3 bg-light rounded")

    try:
        df = pd.read_csv(csv_path)

        # Format numeric columns for better display
        for col in df.select_dtypes(include=["float64"]).columns:
            if "Risk" in col or "%" in col:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and abs(x) < 1 else f"{x:.2f}%")
            elif "Production" in col and "€" in col:
                df[col] = df[col].apply(lambda x: f"€{x:,.0f}" if pd.notna(x) else "")

        return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm", style=TABLE_STYLE)
    except Exception as e:
        logger.error(f"Error loading table from {csv_path}: {e}")
        return html.Div(f"Error loading data: {e}", className="text-danger p-3")


def load_iframe(
    viz_path: str, segment: str | None = None, fallback_message: str = "Visualization not available"
) -> html.Div:
    """
    Load an HTML file in an iframe with fallback.

    Args:
        viz_path: URL path to visualization (e.g., '/static/file.html').
        segment: Optional segment name.
        fallback_message: Message to show if file not found.

    Returns:
        Dash component with iframe or fallback message.
    """
    if not file_exists_for_viz(viz_path, segment):
        return html.Div(fallback_message, className="text-muted p-3 text-center bg-light rounded")

    return html.Iframe(src=viz_path, style=IFRAME_STYLE)


def get_comparison_data(segment: str | None = None) -> pd.DataFrame:
    """
    Aggregate metrics across all scenarios for comparison.

    Returns:
        DataFrame with scenario comparison metrics.
    """
    scenarios = get_scenarios(segment)
    data: list[dict[str, Any]] = []

    for scenario in scenarios:
        paths = get_scenario_paths(scenario, segment)
        row: dict[str, Any] = {"Scenario": scenario.capitalize(), "Scenario_Order": SCENARIO_ORDER.get(scenario, 99)}

        try:
            # Main Period Metrics
            if paths["main_csv"].exists():
                df_main = pd.read_csv(paths["main_csv"])
                opt_row = df_main[df_main["Metric"] == "Optimum selected"]
                if not opt_row.empty:
                    row["Main_Risk"] = opt_row["Risk (%)"].values[0]
                    row["Main_Prod_Pct"] = opt_row["Production (%)"].values[0]

            # MR Period Metrics
            if paths["mr_csv"].exists():
                df_mr = pd.read_csv(paths["mr_csv"])
                opt_row_mr = df_mr[df_mr["Metric"] == "Optimum selected"]
                if not opt_row_mr.empty:
                    row["MR_Risk"] = opt_row_mr["Risk (%)"].values[0]
                    row["MR_Prod_Pct"] = opt_row_mr["Production (%)"].values[0]

            data.append(row)

        except Exception as e:
            logger.warning(f"Error processing scenario {scenario}: {e}")

    return pd.DataFrame(data)


def create_comparison_charts(df_comp: pd.DataFrame) -> html.Div:
    """
    Create comparison charts for scenario analysis.

    Args:
        df_comp: Comparison DataFrame with scenario metrics.

    Returns:
        Dash component with charts.
    """
    if df_comp.empty:
        return html.Div(
            [
                dbc.Alert(
                    [
                        html.H4("No Data Available", className="alert-heading"),
                        html.P("No optimization results found. Please run the pipeline first:"),
                        html.Code("uv run python main.py", className="d-block p-2 bg-light mt-2"),
                        html.P("Or for batch processing:", className="mt-2"),
                        html.Code("uv run python run_batch.py", className="d-block p-2 bg-light mt-2"),
                    ],
                    color="info",
                    className="mt-3",
                )
            ]
        )

    # Sort by scenario order
    df_comp = df_comp.sort_values("Scenario_Order")

    # Compute delta columns (Main vs MR)
    has_mr_risk = "MR_Risk" in df_comp.columns and df_comp["MR_Risk"].notna().any()
    has_mr_prod = "MR_Prod_Pct" in df_comp.columns and df_comp["MR_Prod_Pct"].notna().any()

    if has_mr_risk:
        df_comp["Risk_Delta"] = df_comp["Main_Risk"] - df_comp["MR_Risk"]
    if has_mr_prod:
        df_comp["Prod_Delta"] = df_comp["Main_Prod_Pct"] - df_comp["MR_Prod_Pct"]

    # Risk vs Production scatter
    fig_risk_prod = px.scatter(df_comp, x="Main_Risk", y="Main_Prod_Pct", text="Scenario", size_max=15)
    fig_risk_prod.update_traces(textposition="top center", marker=dict(size=14, color=COLOR_PRIMARY))
    apply_plotly_style(fig_risk_prod, title="Main Period: Risk vs Production Trade-off", height=400)
    fig_risk_prod.update_xaxes(title="Risk (%)")
    fig_risk_prod.update_yaxes(title="Production (%)")

    # Risk comparison bar chart with delta annotations
    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(
        go.Bar(name="Main Period", x=df_comp["Scenario"], y=df_comp["Main_Risk"], marker_color=COLOR_PRIMARY)
    )

    if has_mr_risk:
        fig_sensitivity.add_trace(
            go.Bar(name="MR Period", x=df_comp["Scenario"], y=df_comp["MR_Risk"], marker_color=COLOR_RISK)
        )
        # Add delta annotations
        for _i, row in df_comp.iterrows():
            if pd.notna(row.get("Risk_Delta")):
                delta = row["Risk_Delta"]
                color = COLOR_GOOD if delta <= 0 else COLOR_BAD
                y_pos = max(row["Main_Risk"], row["MR_Risk"])
                fig_sensitivity.add_annotation(
                    x=row["Scenario"],
                    y=y_pos,
                    text=f"\u0394 {delta:+.2%}",
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color=color, weight="bold"),
                )

    fig_sensitivity.update_layout(barmode="group")
    apply_plotly_style(fig_sensitivity, title="Risk by Scenario (Main & MR Periods)", height=400)
    fig_sensitivity.update_xaxes(title="Scenario")
    fig_sensitivity.update_yaxes(title="Risk (%)")

    # Production comparison with delta annotations
    fig_production = go.Figure()

    fig_production.add_trace(
        go.Bar(name="Main Period", x=df_comp["Scenario"], y=df_comp["Main_Prod_Pct"], marker_color=COLOR_PRODUCTION)
    )

    if has_mr_prod:
        fig_production.add_trace(
            go.Bar(name="MR Period", x=df_comp["Scenario"], y=df_comp["MR_Prod_Pct"], marker_color=COLOR_PRIMARY)
        )
        # Add delta annotations
        for _i, row in df_comp.iterrows():
            if pd.notna(row.get("Prod_Delta")):
                delta = row["Prod_Delta"]
                color = COLOR_GOOD if delta >= 0 else COLOR_BAD
                y_pos = max(row["Main_Prod_Pct"], row["MR_Prod_Pct"])
                fig_production.add_annotation(
                    x=row["Scenario"],
                    y=y_pos,
                    text=f"\u0394 {delta:+.1%}",
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color=color, weight="bold"),
                )

    fig_production.update_layout(barmode="group")
    apply_plotly_style(fig_production, title="Production by Scenario (Main & MR Periods)", height=400)
    fig_production.update_xaxes(title="Scenario")
    fig_production.update_yaxes(title="Production (%)")

    # Format display DataFrame
    display_df = df_comp.drop(columns=["Scenario_Order"], errors="ignore").copy()
    for col in ["Main_Risk", "MR_Risk"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    for col in ["Main_Prod_Pct", "MR_Prod_Pct"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for col in ["Risk_Delta", "Prod_Delta"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:+.2%}" if pd.notna(x) else "-")

    return html.Div(
        [
            html.H4("Scenario Comparison", className="my-3"),
            dbc.Row([dbc.Col(dcc.Graph(figure=fig_risk_prod), md=6), dbc.Col(dcc.Graph(figure=fig_sensitivity), md=6)]),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_production), md=6),
                ],
                className="mt-3",
            ),
            html.Hr(),
            html.H5("Summary Metrics", className="mt-4"),
            dbc.Table.from_dataframe(display_df, striped=True, bordered=True, hover=True, size="sm"),
        ]
    )


def create_empty_state() -> html.Div:
    """Create an empty state message when no data is available."""
    return dbc.Alert(
        [
            html.H4("No Data Available", className="alert-heading"),
            html.P("No optimization results found. Please run the pipeline first:"),
            html.Code("uv run python main.py", className="d-block p-2 bg-light mt-2"),
            html.P("Or for batch processing:", className="mt-2"),
            html.Code("uv run python run_batch.py", className="d-block p-2 bg-light mt-2"),
            html.Hr(),
            html.P(
                [
                    "This dashboard displays results from the credit risk scoring optimization pipeline. ",
                    "Once you run the pipeline, the following will be generated:",
                ],
                className="mb-2",
            ),
            html.Ul(
                [
                    html.Li("Scenario comparison (pessimistic, base, optimistic)"),
                    html.Li("Risk production summary tables"),
                    html.Li("Interactive visualizations"),
                    html.Li("Stability analysis reports"),
                ]
            ),
        ],
        color="info",
        className="mt-3",
    )


# --- KPI & UI Helper Functions ---


def create_kpi_card(
    title: str, value: str, delta: str = "", fmt: str = "", color_border: str = COLOR_PRIMARY
) -> dbc.Col:
    """Create a single KPI card with label, value, and colored delta."""
    delta_class = "neutral"
    if delta:
        if delta.startswith("+") or delta.startswith("-"):
            # For risk: negative is good. For production: positive is good.
            if "risk" in title.lower():
                delta_class = "positive" if delta.startswith("-") else "negative"
            else:
                delta_class = "positive" if delta.startswith("+") else "negative"

    return dbc.Col(
        html.Div(
            [
                html.Div(title, className="kpi-label"),
                html.Div(value, className="kpi-value"),
                html.Div(delta, className=f"kpi-delta {delta_class}") if delta else html.Div(),
            ],
            className="kpi-card",
            style={"borderLeftColor": color_border},
        ),
        md=3,
        className="mb-3",
    )


def create_kpi_row(main_csv_path: Path, mr_csv_path: Path) -> dbc.Row:
    """Create a row of 4 KPI cards from main and MR CSV data."""
    main_risk, main_prod_pct, main_prod_eur = None, None, None
    mr_risk, mr_prod_pct = None, None

    try:
        if main_csv_path.exists():
            df = pd.read_csv(main_csv_path)
            opt = df[df["Metric"] == "Optimum selected"]
            if not opt.empty:
                main_risk = opt["Risk (%)"].values[0]
                main_prod_pct = opt["Production (%)"].values[0]
                if "Production (€)" in df.columns:
                    main_prod_eur = opt["Production (€)"].values[0]

        if mr_csv_path.exists():
            df_mr = pd.read_csv(mr_csv_path)
            opt_mr = df_mr[df_mr["Metric"] == "Optimum selected"]
            if not opt_mr.empty:
                mr_risk = opt_mr["Risk (%)"].values[0]
                mr_prod_pct = opt_mr["Production (%)"].values[0]
    except Exception as e:
        logger.warning(f"Error loading KPI data: {e}")

    # Build cards
    risk_val = f"{main_risk:.2%}" if main_risk is not None else "N/A"
    risk_delta = f"{main_risk - mr_risk:+.2%} vs MR" if main_risk is not None and mr_risk is not None else ""

    prod_pct_val = f"{main_prod_pct:.1%}" if main_prod_pct is not None else "N/A"
    prod_pct_delta = (
        f"{main_prod_pct - mr_prod_pct:+.1%} vs MR" if main_prod_pct is not None and mr_prod_pct is not None else ""
    )

    prod_eur_val = f"EUR {main_prod_eur:,.0f}" if main_prod_eur is not None else "N/A"

    risk_delta_val = f"{main_risk - mr_risk:+.2%}" if main_risk is not None and mr_risk is not None else "N/A"
    risk_delta_suffix = "vs MR" if main_risk is not None and mr_risk is not None else ""

    return dbc.Row(
        [
            create_kpi_card("Risk", risk_val, risk_delta, color_border=COLOR_RISK),
            create_kpi_card("Production (%)", prod_pct_val, prod_pct_delta, color_border=COLOR_PRODUCTION),
            create_kpi_card("Production (EUR)", prod_eur_val, color_border=COLOR_PRIMARY),
            create_kpi_card("Risk Delta", risk_delta_val, risk_delta_suffix, color_border=COLOR_NEUTRAL),
        ],
        className="mb-4",
    )


def load_table_with_formatting(csv_path: Path) -> html.Div:
    """Enhanced table with per-cell CSS: red for high risk, green for low."""
    if not csv_path.exists():
        return html.Div(f"Data file not found: {csv_path.name}", className="text-warning p-3 bg-light rounded")

    try:
        df = pd.read_csv(csv_path)

        # Identify risk columns for conditional formatting
        style_data_conditional = []
        for col in df.columns:
            if "Risk" in col or "%" in col:
                # Values > 2% get red highlight
                style_data_conditional.append(
                    {
                        "if": {"column_id": col, "filter_query": f"{{{col}}} > 0.02"},
                        "className": "cell-high-risk",
                    }
                )
                # Values < 1% get green highlight
                style_data_conditional.append(
                    {
                        "if": {"column_id": col, "filter_query": f"{{{col}}} < 0.01"},
                        "className": "cell-low-risk",
                    }
                )

        # Format display values
        columns = [{"name": c, "id": c, "type": "numeric"} for c in df.columns]
        data = df.to_dict("records")

        return dash_table.DataTable(
            columns=columns,
            data=data,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "right", "padding": "8px", "fontSize": "0.85rem"},
            style_header={"backgroundColor": "#f8f9fa", "fontWeight": "600", "borderBottom": "2px solid #dee2e6"},
            style_data_conditional=style_data_conditional,
            page_size=20,
        )
    except Exception as e:
        logger.error(f"Error loading table from {csv_path}: {e}")
        return html.Div(f"Error loading data: {e}", className="text-danger p-3")


def create_download_button(button_id: str, label: str = "Download CSV") -> html.Div:
    """Create a dbc.Button + dcc.Download pair."""
    return html.Div(
        [
            dbc.Button(
                [html.I(className="fas fa-download me-1"), label],
                id={"type": "download-btn", "index": button_id},
                color="secondary",
                size="sm",
                outline=True,
                className="mb-3",
            ),
            dcc.Download(id={"type": "download-sink", "index": button_id}),
        ],
        className="d-inline-block",
    )


def create_collapsible_section(header: str, children: list, section_id: str, is_open: bool = True) -> html.Div:
    """Pattern-matching collapsible wrapper."""
    return html.Div(
        [
            html.Div(
                dbc.Button(
                    [
                        html.Span(header, className="fw-bold"),
                        html.I(className="fas fa-chevron-down ms-2"),
                    ],
                    id={"type": "collapse-btn", "index": section_id},
                    color="link",
                    className="collapse-header text-decoration-none text-dark p-2 w-100 text-start",
                ),
            ),
            dbc.Collapse(
                html.Div(children, className="p-2"),
                id={"type": "collapse-section", "index": section_id},
                is_open=is_open,
            ),
        ],
        className="border rounded mb-3",
    )


# --- Audit Trail Functions ---


def get_audit_paths(scenario: str, segment: str | None = None) -> dict[str, Path]:
    """Return paths for audit CSVs (main and MR)."""
    data_dir = get_data_dir(segment)
    return {
        "main": data_dir / f"audit_{scenario}.csv",
        "mr": data_dir / f"audit_{scenario}_mr.csv",
    }


def load_audit_summary(path: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load audit CSV, group by classification, return (df, summary_df)."""
    if not path.exists():
        return None, None

    try:
        df = pd.read_csv(path)

        # Group by classification for summary
        classifications = ["keep", "swap_in", "swap_out", "rejected", "rejected_other"]
        summary_rows = []
        for cls in classifications:
            subset = df[df["classification"] == cls]
            count = len(subset)
            total_amt = subset["oa_amt"].sum() if "oa_amt" in subset.columns else 0
            total_adj = subset["oa_amt_adjusted"].sum() if "oa_amt_adjusted" in subset.columns else 0
            summary_rows.append(
                {"classification": cls, "count": count, "total_oa_amt": total_amt, "total_oa_amt_adjusted": total_adj}
            )

        summary_df = pd.DataFrame(summary_rows)
        return df, summary_df
    except Exception as e:
        logger.error(f"Error loading audit from {path}: {e}")
        return None, None


def create_audit_tab_content(scenario: str, segment: str | None = None, period: str = "main") -> html.Div:
    """Full Audit Trail tab: KPI cards + donut chart + filterable DataTable."""
    paths = get_audit_paths(scenario, segment)
    path = paths.get(period, paths["main"])
    df, summary_df = load_audit_summary(path)

    if df is None:
        return dbc.Alert(
            [
                html.H4("Audit Trail Not Available", className="alert-heading"),
                html.P(f"No audit data found for scenario '{scenario}' ({period} period)."),
                html.P(f"Expected file: {path.name}"),
            ],
            color="warning",
        )

    scenario_display = scenario.capitalize()
    segment_display = f" - {segment}" if segment else ""
    period_display = "Main" if period == "main" else "MR"

    # KPI cards from summary
    classification_colors = {
        "keep": COLOR_PRODUCTION,
        "swap_in": "#27AE60",
        "swap_out": COLOR_RISK,
        "rejected": "#E67E22",
        "rejected_other": COLOR_NEUTRAL,
    }

    kpi_cards = []
    for _, row in summary_df.iterrows():
        cls = row["classification"]
        kpi_cards.append(
            create_kpi_card(
                title=cls.replace("_", " ").title(),
                value=f"{row['count']:,}",
                delta=f"EUR {row['total_oa_amt_adjusted']:,.0f}",
                color_border=classification_colors.get(cls, COLOR_PRIMARY),
            )
        )

    # Donut chart
    fig_donut = px.pie(
        summary_df,
        values="count",
        names="classification",
        hole=0.4,
        color="classification",
        color_discrete_map=classification_colors,
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    apply_plotly_style(fig_donut, title="Classification Breakdown", height=350)

    # Filterable DataTable
    display_cols = ["authorization_id", "status_name", "classification", "oa_amt", "oa_amt_adjusted", "cut_limit"]
    available_cols = [c for c in display_cols if c in df.columns]
    table_df = df[available_cols].copy()

    # Conditional row styling
    style_data_conditional = [
        {"if": {"filter_query": '{classification} = "swap_in"'}, "backgroundColor": "rgba(46,204,113,0.08)"},
        {"if": {"filter_query": '{classification} = "swap_out"'}, "backgroundColor": "rgba(231,76,60,0.08)"},
    ]

    audit_table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in available_cols],
        data=table_df.to_dict("records"),
        page_size=20,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "right", "padding": "6px 10px", "fontSize": "0.83rem"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "600", "borderBottom": "2px solid #dee2e6"},
        style_data_conditional=style_data_conditional,
    )

    return html.Div(
        [
            html.H4(f"Audit Trail ({scenario_display}{segment_display} - {period_display})", className="mb-4"),
            # Period toggle
            dbc.RadioItems(
                id="audit-period-toggle",
                options=[
                    {"label": "Main Period", "value": "main"},
                    {"label": "MR Period", "value": "mr"},
                ],
                value=period,
                inline=True,
                className="mb-3",
            ),
            # KPI row (up to 5 cards)
            dbc.Row(kpi_cards[:4], className="mb-3"),
            dbc.Row(kpi_cards[4:], className="mb-3") if len(kpi_cards) > 4 else html.Div(),
            # Charts + Table
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_donut), md=5),
                    dbc.Col(
                        [
                            html.H5("Detail Records", className="mb-2"),
                            audit_table,
                        ],
                        md=7,
                    ),
                ]
            ),
        ]
    )


# --- Model Details Functions ---


def get_models_dir(segment: str | None = None) -> Path:
    """Resolve model directory (handles supersegment fallback)."""
    if segment:
        direct = OUTPUT_BASE / segment / "models"
        if direct.exists() and any(direct.iterdir()):
            return direct
        # Try supersegment fallback (e.g., _supersegment_no_premium for no_premium_*)
        if "_" in segment:
            prefix = segment.rsplit("_", 1)[0]
        else:
            prefix = segment
        for item in OUTPUT_BASE.iterdir():
            if item.is_dir() and item.name.startswith("_supersegment") and prefix in item.name:
                models_dir = item / "models"
                if models_dir.exists():
                    return models_dir
        return direct
    # Single-run mode
    if (OUTPUT_BASE / "models").exists():
        return OUTPUT_BASE / "models"
    return Path("models")


def get_latest_model(segment: str | None = None) -> dict | None:
    """Find newest model_*/metadata.json, return parsed dict."""
    models_dir = get_models_dir(segment)
    if not models_dir.exists():
        return None

    model_dirs = sorted(models_dir.glob("model_*"), reverse=True)
    for md in model_dirs:
        meta_path = md / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                metadata["_model_dir"] = str(md)
                return metadata
            except Exception as e:
                logger.warning(f"Error reading {meta_path}: {e}")
    return None


def parse_coefficients(model_dir: str) -> list[dict[str, Any]]:
    """Parse model_summary.txt for feature coefficients."""
    summary_path = Path(model_dir) / "model_summary.txt"
    if not summary_path.exists():
        return []

    coefficients = []
    in_coef_section = False
    try:
        text = summary_path.read_text()
        for line in text.splitlines():
            line = line.strip()
            if "MODEL COEFFICIENTS" in line:
                in_coef_section = True
                continue
            if in_coef_section:
                if line.startswith("---") or not line:
                    continue
                # New section starts
                if line.isupper() and ":" not in line:
                    break
                # Parse "feature_name: value"
                if ":" in line:
                    parts = line.rsplit(":", 1)
                    if len(parts) == 2:
                        try:
                            name = parts[0].strip()
                            value = float(parts[1].strip())
                            coefficients.append({"feature": name, "coefficient": value})
                        except ValueError:
                            continue
    except Exception as e:
        logger.warning(f"Error parsing coefficients from {summary_path}: {e}")

    return coefficients


def create_model_details_content(segment: str | None = None) -> html.Div:
    """Full Model Details tab: info card + R2 progress bars + coef chart + params table."""
    metadata = get_latest_model(segment)

    if metadata is None:
        return dbc.Alert(
            [
                html.H4("Model Details Not Available", className="alert-heading"),
                html.P("No trained model found. Run the pipeline first to generate a model."),
            ],
            color="warning",
        )

    model_dir = metadata.get("_model_dir", "")
    segment_display = f" - {segment}" if segment else ""

    # Model info card (3-column layout)
    info_items = [
        ("Model Type", metadata.get("model_type", "N/A")),
        ("Features", str(metadata.get("num_features", "N/A"))),
        ("Date", metadata.get("timestamp", "N/A")),
        (
            "Samples",
            f"{metadata.get('total_samples', 'N/A'):,}" if isinstance(metadata.get("total_samples"), int) else "N/A",
        ),
        ("CV Folds", str(metadata.get("cv_folds", "N/A"))),
        ("Is Hurdle", str(metadata.get("is_hurdle", "N/A"))),
        (
            "Zero Proportion",
            f"{metadata.get('zero_proportion', 0):.1%}" if metadata.get("zero_proportion") is not None else "N/A",
        ),
        ("Weighted", str(metadata.get("weighted_regression", "N/A"))),
    ]

    info_cols = []
    for label, value in info_items:
        info_cols.append(
            dbc.Col(
                [html.Small(label, className="text-muted d-block"), html.Strong(value)],
                md=3,
                className="mb-2",
            )
        )

    info_card = dbc.Card(
        [
            dbc.CardHeader("Model Information"),
            dbc.CardBody(dbc.Row(info_cols)),
        ],
        className="mb-4",
    )

    # Performance: R2 progress bars
    r2_metrics = [
        ("CV Mean R\u00b2", metadata.get("cv_mean_r2")),
        ("Train R² (in-sample)", metadata.get("train_r2", metadata.get("full_r2"))),
        ("Step 1 CV R\u00b2", metadata.get("step1_cv_r2")),
        ("Step 2 CV R\u00b2", metadata.get("step2_cv_r2")),
    ]

    progress_items = []
    for label, val in r2_metrics:
        if val is not None:
            pct = max(0, min(100, val * 100))
            color = "success" if pct >= 50 else "warning" if pct >= 20 else "danger"
            progress_items.append(
                html.Div(
                    [
                        html.Div(
                            [html.Span(label, className="small"), html.Span(f"{val:.4f}", className="small float-end")],
                            className="mb-1",
                        ),
                        dbc.Progress(value=pct, color=color, style={"height": "8px"}, className="mb-3"),
                    ]
                )
            )

    performance_card = dbc.Card(
        [
            dbc.CardHeader("Performance Metrics"),
            dbc.CardBody(progress_items if progress_items else "No metrics available"),
        ],
        className="mb-4",
    )

    # Coefficient chart
    coefficients = parse_coefficients(model_dir)
    coef_chart = html.Div()
    if coefficients:
        coef_df = pd.DataFrame(coefficients)
        coef_df = coef_df.sort_values("coefficient")
        colors = [COLOR_PRODUCTION if v >= 0 else COLOR_RISK for v in coef_df["coefficient"]]

        fig_coef = go.Figure(
            go.Bar(
                x=coef_df["coefficient"],
                y=coef_df["feature"],
                orientation="h",
                marker_color=colors,
            )
        )
        apply_plotly_style(fig_coef, title="Feature Coefficients", height=max(250, len(coefficients) * 40))
        fig_coef.update_layout(margin=dict(l=200))

        coef_chart = dbc.Card(
            [dbc.CardHeader("Feature Coefficients"), dbc.CardBody(dcc.Graph(figure=fig_coef))],
            className="mb-4",
        )

    # Parameters table
    model_params = metadata.get("model_params", {})
    params_table = html.Div()
    if model_params:
        params_rows = [html.Tr([html.Td(k, className="fw-bold"), html.Td(str(v))]) for k, v in model_params.items()]
        params_table = dbc.Card(
            [
                dbc.CardHeader("Model Parameters"),
                dbc.CardBody(
                    dbc.Table(
                        [html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value")])), html.Tbody(params_rows)],
                        striped=True,
                        bordered=True,
                        hover=True,
                        size="sm",
                    )
                ),
            ],
            className="mb-4",
        )

    return html.Div(
        [
            html.H4(f"Model Details{segment_display}", className="mb-4"),
            info_card,
            dbc.Row(
                [
                    dbc.Col(performance_card, md=5),
                    dbc.Col(coef_chart, md=7),
                ]
            ),
            params_table,
        ]
    )


# --- Cutoff Explorer Functions ---


def load_cutoff_data(
    scenario: str, segment: str | None = None
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """
    Load data needed for cutoff explorer.

    Returns:
        Tuple of (summary_data, optimal_solution, pareto_solutions, variables)
    """
    paths = get_scenario_paths(scenario, segment)
    data_dir = get_data_dir(segment)

    summary_data = None
    optimal_solution = None
    pareto_solutions = None
    variables = []

    # Load summary data
    if paths["summary_data"].exists():
        summary_data = pd.read_csv(paths["summary_data"])
        # Detect variables (first two columns that look like cluster columns)
        for col in summary_data.columns:
            if "clus" in col.lower() or col in ["sc_octroi_new_clus", "new_efx_clus"]:
                variables.append(col)
            if len(variables) == 2:
                break

    # Load optimal solution
    if paths["cutoffs"].exists():
        optimal_solution = pd.read_csv(paths["cutoffs"])

    # Load all Pareto-optimal solutions (for risk slider)
    pareto_path = data_dir / "pareto_optimal_solutions.csv"
    if pareto_path.exists():
        pareto_solutions = pd.read_csv(pareto_path)

    return summary_data, optimal_solution, pareto_solutions, variables


def calculate_metrics_from_custom_cuts(
    data: pd.DataFrame, cut_map: dict[int, int], var0_col: str, var1_col: str
) -> dict[str, Any]:
    """
    Calculate metrics for custom cutoff configuration.

    Args:
        data: Summary data DataFrame
        cut_map: Dictionary mapping var0 bins to var1 cutoff values
        var0_col: First variable column name
        var1_col: Second variable column name

    Returns:
        Dictionary with calculated metrics
    """
    df = data.copy()

    # Apply cuts
    df["cut_limit"] = df[var0_col].map(cut_map)
    df["passes_cut"] = df[var1_col] <= df["cut_limit"]

    def calc_metrics(subset, suffix):
        prod_col = f"oa_amt_h0{suffix}"
        risk_num_col = f"todu_30ever_h6{suffix}"
        risk_den_col = f"todu_amt_pile_h6{suffix}"

        if prod_col not in subset.columns:
            return 0, 0, 0, 0

        prod = subset[prod_col].sum()
        risk_num = subset[risk_num_col].sum() if risk_num_col in subset.columns else 0
        risk_den = subset[risk_den_col].sum() if risk_den_col in subset.columns else 0
        b2_ever = (risk_num / risk_den * 7) if risk_den > 0 else 0.0
        return prod, b2_ever, risk_num, risk_den

    # Actual (All Booked)
    actual_prod, actual_risk, actual_rn, actual_rd = calc_metrics(df, "_boo")

    # Swap-in (Repesca that passes)
    swap_in_df = df[df["passes_cut"]]
    si_prod, si_risk, si_rn, si_rd = calc_metrics(swap_in_df, "_rep")

    # Swap-out (Booked that fails)
    swap_out_df = df[~df["passes_cut"]]
    so_prod, so_risk, so_rn, so_rd = calc_metrics(swap_out_df, "_boo")

    # Optimum
    opt_prod = (actual_prod - so_prod) + si_prod
    opt_rn = (actual_rn - so_rn) + si_rn
    opt_rd = (actual_rd - so_rd) + si_rd
    opt_risk = (opt_rn / opt_rd * 7) if opt_rd > 0 else 0.0

    return {
        "actual_prod": actual_prod,
        "actual_risk": actual_risk,
        "swap_in_prod": si_prod,
        "swap_in_risk": si_risk,
        "swap_out_prod": so_prod,
        "swap_out_risk": so_risk,
        "opt_prod": opt_prod,
        "opt_risk": opt_risk,
        "opt_prod_pct": opt_prod / actual_prod if actual_prod > 0 else 0,
    }


def create_cutoff_explorer_layout(scenario: str, segment: str | None = None) -> html.Div:
    """Create the cutoff explorer tab layout."""
    summary_data, optimal_solution, pareto_solutions, variables = load_cutoff_data(scenario, segment)

    if summary_data is None or len(variables) < 2:
        return dbc.Alert(
            [
                html.H4("Cutoff Explorer Not Available", className="alert-heading"),
                html.P("Summary data not found for this scenario. Run the pipeline first."),
            ],
            color="warning",
        )

    var0_col, var1_col = variables[0], variables[1]

    # Get unique bins
    bins = sorted(summary_data[var0_col].unique())
    var1_max = int(summary_data[var1_col].max())

    # Get optimal cutoffs
    optimal_cuts = {}
    if optimal_solution is not None and not optimal_solution.empty:
        opt_row = optimal_solution.iloc[0]
        for bin_val in bins:
            if bin_val in optimal_solution.columns:
                optimal_cuts[bin_val] = int(opt_row[bin_val])
            elif str(bin_val) in optimal_solution.columns:
                optimal_cuts[bin_val] = int(opt_row[str(bin_val)])
            else:
                optimal_cuts[bin_val] = var1_max

    # Get risk range from Pareto solutions
    risk_min, risk_max, risk_current = 0.0, 2.0, 1.0
    pareto_data = []
    if pareto_solutions is not None and not pareto_solutions.empty:
        risk_min = float(pareto_solutions["b2_ever_h6"].min())
        risk_max = float(pareto_solutions["b2_ever_h6"].max())
        # Get current risk from optimal solution if available
        if optimal_solution is not None and "b2_ever_h6" in optimal_solution.columns:
            risk_current = float(optimal_solution["b2_ever_h6"].iloc[0])
        else:
            risk_current = (risk_min + risk_max) / 2
        # Convert Pareto solutions to list of dicts for storage
        pareto_data = pareto_solutions.to_dict("records")

    # Create sliders for each bin
    sliders = []
    for bin_val in bins:
        initial_value = optimal_cuts.get(bin_val, var1_max)
        sliders.append(
            dbc.Col(
                [
                    html.Label(f"Bin {int(bin_val)}", className="fw-bold text-center d-block"),
                    dcc.Slider(
                        id={"type": "cutoff-slider", "index": int(bin_val)},
                        min=1,
                        max=var1_max,
                        step=1,
                        value=initial_value,
                        marks={i: str(i) for i in range(1, var1_max + 1, max(1, var1_max // 5))},
                        vertical=True,
                        verticalHeight=200,
                        tooltip={"placement": "right", "always_visible": True},
                    ),
                    html.Div(
                        f"Opt: {initial_value}",
                        className="text-muted text-center small mt-1",
                        id={"type": "optimal-label", "index": int(bin_val)},
                    ),
                ],
                className="text-center",
                style={"minWidth": "80px"},
            )
        )

    # Store for data
    scenario_display = scenario.capitalize()
    segment_display = f" - {segment}" if segment else ""

    # Create risk slider marks
    risk_step = (risk_max - risk_min) / 4 if risk_max > risk_min else 0.25
    risk_marks = {round(risk_min + i * risk_step, 2): f"{round(risk_min + i * risk_step, 2):.2f}%" for i in range(5)}

    return html.Div(
        [
            html.H4(f"Cutoff Explorer ({scenario_display}{segment_display})", className="mb-4"),
            # Store data for callbacks
            dcc.Store(
                id="cutoff-data-store",
                data={
                    "scenario": scenario,
                    "segment": segment,
                    "var0_col": var0_col,
                    "var1_col": var1_col,
                    "optimal_cuts": {str(k): v for k, v in optimal_cuts.items()},
                    "bins": [int(b) for b in bins],
                    "pareto_solutions": pareto_data,
                },
            ),
            # Risk Level Selector
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span("Target Risk Level", className="fw-bold"),
                            html.Span(
                                " - Select a risk level to find the closest optimal solution",
                                className="text-muted small ms-2",
                            ),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Label("Risk Level (b2_ever_h6 %)", className="small"),
                                            dcc.Slider(
                                                id="risk-level-slider",
                                                min=round(risk_min, 2),
                                                max=round(risk_max, 2),
                                                step=0.01,
                                                value=round(risk_current, 2),
                                                marks=risk_marks,
                                                tooltip={"placement": "bottom", "always_visible": True},
                                            ),
                                        ],
                                        md=10,
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Apply",
                                                id="apply-risk-btn",
                                                color="primary",
                                                size="sm",
                                                className="mt-4",
                                            ),
                                        ],
                                        md=2,
                                        className="d-flex align-items-center",
                                    ),
                                ]
                            ),
                            html.Div(id="risk-selection-info", className="mt-2 small text-muted"),
                        ]
                    ),
                ],
                className="mb-3",
            ),
            dbc.Row(
                [
                    # Sliders panel
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        [
                                            html.Span("Adjust Cutoffs by Score Bin", className="fw-bold"),
                                            dbc.Button(
                                                "Reset to Optimal",
                                                id="reset-cutoffs-btn",
                                                color="secondary",
                                                size="sm",
                                                className="float-end",
                                            ),
                                        ]
                                    ),
                                    dbc.CardBody(
                                        [
                                            html.P(
                                                [
                                                    f"Variable: {var0_col} (bins) → {var1_col} (max cutoff)",
                                                ],
                                                className="text-muted small",
                                            ),
                                            html.Div(
                                                [dbc.Row(sliders, className="g-2 justify-content-center")],
                                                style={"overflowX": "auto"},
                                            ),
                                        ]
                                    ),
                                ],
                                className="mb-3",
                            ),
                        ],
                        md=5,
                    ),
                    # Results panel
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardHeader("Impact Analysis"), dbc.CardBody(id="cutoff-results")], className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Risk vs Production"),
                                    dbc.CardBody([dcc.Graph(id="cutoff-chart", style={"height": "300px"})]),
                                ]
                            ),
                        ],
                        md=7,
                    ),
                ]
            ),
            # Heatmap
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader("Cutoff Visualization (cells passing cut are highlighted)"),
                                    dbc.CardBody([dcc.Graph(id="cutoff-heatmap", style={"height": "400px"})]),
                                ]
                            )
                        ]
                    )
                ],
                className="mt-3",
            ),
        ]
    )


# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Risk Scoring Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server


# Serve static files from images directory (supports both single and batch modes)
@server.route("/static/<path:filename>")
def serve_static_root(filename):
    """Serve static HTML files from root images directory."""
    images_dir = get_images_dir(None)
    return send_from_directory(str(images_dir), filename)


@server.route("/static/<segment>/<path:filename>")
def serve_static_segment(segment, filename):
    """Serve static HTML files from segment images directory."""
    images_dir = get_images_dir(segment)
    return send_from_directory(str(images_dir), filename)


# --- Layout ---
def create_layout():
    """Create the dashboard layout."""
    segments = get_available_segments()
    has_segments = len(segments) > 0

    # Determine initial segment
    initial_segment = CURRENT_SEGMENT
    if not initial_segment and has_segments:
        initial_segment = segments[0]

    scenarios = get_scenarios(initial_segment)
    has_data = len(scenarios) > 0

    return dbc.Container(
        [
            # Header
            html.Div(
                [
                    html.H1("Risk Scoring Optimization Dashboard", className="text-center mb-0"),
                    html.P(
                        "Explore optimization results across scenarios and time periods",
                        className="text-muted text-center",
                    ),
                ],
                className="my-4",
            ),
            # Segment and Scenario selectors
            dbc.Row(
                [
                    # Segment selector (only show if batch output exists)
                    dbc.Col(
                        [
                            html.Label("Select Segment:", className="fw-bold"),
                            dcc.Dropdown(
                                id="segment-selector",
                                options=[{"label": s, "value": s} for s in segments],
                                value=initial_segment,
                                clearable=False,
                                disabled=not has_segments,
                                className="mb-3",
                                placeholder="No segments available" if not has_segments else "Select segment...",
                            ),
                        ],
                        md=4,
                        style={"display": "block" if has_segments else "none"},
                    ),
                    # Scenario selector
                    dbc.Col(
                        [
                            html.Label("Select Scenario:", className="fw-bold"),
                            dcc.Dropdown(
                                id="scenario-selector",
                                options=[{"label": s.capitalize(), "value": s} for s in scenarios],
                                value=scenarios[0] if scenarios else None,
                                clearable=False,
                                disabled=not has_data,
                                className="mb-3",
                                placeholder="No scenarios available" if not has_data else "Select scenario...",
                            ),
                        ],
                        md=4,
                    ),
                ]
            ),
            # Tabs
            dbc.Tabs(
                [
                    dbc.Tab(label="Scenario Comparison", tab_id="tab-comp"),
                    dbc.Tab(label="Main Period", tab_id="tab-main", disabled=not has_data),
                    dbc.Tab(label="MR Period (Recent)", tab_id="tab-mr", disabled=not has_data),
                    dbc.Tab(label="Cutoff Explorer", tab_id="tab-cutoff", disabled=not has_data),
                    dbc.Tab(label="Stability Analysis", tab_id="tab-stability", disabled=not has_data),
                    dbc.Tab(label="Audit Trail", tab_id="tab-audit", disabled=not has_data),
                    dbc.Tab(label="Model Details", tab_id="tab-model", disabled=not has_data),
                ],
                id="tabs",
                active_tab="tab-comp",
                className="mb-3",
            ),
            # Content area with loading spinner
            dcc.Loading(
                html.Div(id="tab-content", className="mt-3"),
                type="default",
                className="dash-loading",
            ),
            # Footer
            html.Hr(className="mt-5"),
            html.Footer(
                [html.Small(f"Risk Scoring Optimization Tools | Output: {OUTPUT_BASE}", className="text-muted")],
                className="text-center mb-3",
            ),
        ],
        fluid=True,
    )


app.layout = create_layout


# --- Callbacks ---


@app.callback(
    [
        Output("scenario-selector", "options"),
        Output("scenario-selector", "value"),
        Output("scenario-selector", "disabled"),
        Output("tabs", "children"),
    ],
    [Input("segment-selector", "value")],
)
def update_scenarios_on_segment_change(segment: str | None):
    """Update scenario dropdown when segment changes."""
    scenarios = get_scenarios(segment)
    has_data = len(scenarios) > 0

    options = [{"label": s.capitalize(), "value": s} for s in scenarios]
    value = scenarios[0] if scenarios else None

    # Rebuild tabs with correct disabled state
    tabs = [
        dbc.Tab(label="Scenario Comparison", tab_id="tab-comp"),
        dbc.Tab(label="Main Period", tab_id="tab-main", disabled=not has_data),
        dbc.Tab(label="MR Period (Recent)", tab_id="tab-mr", disabled=not has_data),
        dbc.Tab(label="Cutoff Explorer", tab_id="tab-cutoff", disabled=not has_data),
        dbc.Tab(label="Stability Analysis", tab_id="tab-stability", disabled=not has_data),
        dbc.Tab(label="Audit Trail", tab_id="tab-audit", disabled=not has_data),
        dbc.Tab(label="Model Details", tab_id="tab-model", disabled=not has_data),
    ]

    return options, value, not has_data, tabs


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"), Input("scenario-selector", "value"), Input("segment-selector", "value")],
)
def render_content(active_tab: str, scenario: str | None, segment: str | None) -> html.Div:
    """
    Render tab content based on selected tab and scenario.

    Args:
        active_tab: Currently active tab ID.
        scenario: Selected scenario name.
        segment: Selected segment name.

    Returns:
        Dash component for the tab content.
    """
    # Comparison tab doesn't need a scenario
    if active_tab == "tab-comp":
        df_comp = get_comparison_data(segment)
        return create_comparison_charts(df_comp)

    # Other tabs require a scenario
    if not scenario:
        return create_empty_state()

    paths = get_scenario_paths(scenario, segment)
    scenario_display = scenario.capitalize()
    segment_display = f" - {segment}" if segment else ""

    if active_tab == "tab-main":
        return html.Div(
            [
                html.H4(f"Main Period Results ({scenario_display}{segment_display})", className="mb-4"),
                create_kpi_row(paths["main_csv"], paths["mr_csv"]),
                create_download_button("main", "Download Main CSV"),
                dbc.Card(
                    [dbc.CardHeader("Summary Metrics"), dbc.CardBody(load_table(paths["main_csv"]))], className="mb-4"
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Risk Production Visualizer"),
                        dbc.CardBody(
                            load_iframe(
                                paths["main_viz"],
                                segment,
                                "Main period visualization not available. Run the pipeline to generate.",
                            )
                        ),
                    ]
                ),
            ]
        )

    elif active_tab == "tab-mr":
        return html.Div(
            [
                html.H4(f"MR Period Results ({scenario_display}{segment_display})", className="mb-4"),
                create_kpi_row(paths["mr_csv"], paths["main_csv"]),
                create_download_button("mr", "Download MR CSV"),
                dbc.Card(
                    [dbc.CardHeader("Summary Metrics"), dbc.CardBody(load_table(paths["mr_csv"]))], className="mb-4"
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Risk Surface (3D)"),
                        dbc.CardBody(
                            load_iframe(
                                paths["mr_viz"],
                                segment,
                                "MR period visualization not available. Run the pipeline to generate.",
                            )
                        ),
                    ]
                ),
            ]
        )

    elif active_tab == "tab-cutoff":
        return create_cutoff_explorer_layout(scenario, segment)

    elif active_tab == "tab-stability":
        return html.Div(
            [
                html.H4(f"Stability Analysis ({scenario_display}{segment_display})", className="mb-4"),
                dbc.Card(
                    [
                        dbc.CardHeader("PSI/CSI Stability Report"),
                        dbc.CardBody(
                            load_iframe(
                                paths["stability"],
                                segment,
                                "Stability report not available. Run MR period processing to generate.",
                            )
                        ),
                    ]
                ),
            ]
        )

    elif active_tab == "tab-audit":
        return html.Div(id="audit-content", children=create_audit_tab_content(scenario, segment, "main"))

    elif active_tab == "tab-model":
        return create_model_details_content(segment)

    return html.Div("Select a tab", className="text-muted")


# Cutoff Explorer callbacks
@app.callback(
    [Output("cutoff-results", "children"), Output("cutoff-chart", "figure"), Output("cutoff-heatmap", "figure")],
    [Input({"type": "cutoff-slider", "index": dash.ALL}, "value"), Input("cutoff-data-store", "data")],
    prevent_initial_call=False,
)
def update_cutoff_analysis(slider_values, store_data):
    """Update cutoff analysis when sliders change."""
    if not store_data or not slider_values:
        empty_fig = go.Figure()
        empty_fig.update_layout(annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False)])
        return html.Div("No data available"), empty_fig, empty_fig

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    var0_col = store_data["var0_col"]
    var1_col = store_data["var1_col"]
    optimal_cuts = {int(k): v for k, v in store_data["optimal_cuts"].items()}
    bins = store_data["bins"]

    # Load data
    summary_data, _, _, _ = load_cutoff_data(scenario, segment)
    if summary_data is None:
        empty_fig = go.Figure()
        return html.Div("Data not available"), empty_fig, empty_fig

    # Build current cut map from sliders
    current_cuts = {bins[i]: slider_values[i] for i in range(len(bins))}

    # Calculate metrics for current and optimal
    current_metrics = calculate_metrics_from_custom_cuts(summary_data, current_cuts, var0_col, var1_col)
    optimal_metrics = calculate_metrics_from_custom_cuts(summary_data, optimal_cuts, var0_col, var1_col)

    # Create results display
    risk_delta = current_metrics["opt_risk"] - optimal_metrics["opt_risk"]
    prod_delta = current_metrics["opt_prod_pct"] - optimal_metrics["opt_prod_pct"]

    risk_color = COLOR_GOOD if risk_delta <= 0 else COLOR_BAD
    prod_color = COLOR_GOOD if prod_delta >= 0 else COLOR_BAD

    results = dbc.Row(
        [
            dbc.Col(
                [
                    html.H6("Current Configuration", className="text-primary"),
                    html.P(
                        [
                            html.Strong("Risk: "),
                            f"{current_metrics['opt_risk']:.2%}",
                            html.Span(f" ({risk_delta:+.2%})", style={"color": risk_color}, className="ms-1"),
                        ]
                    ),
                    html.P(
                        [
                            html.Strong("Production: "),
                            f"{current_metrics['opt_prod_pct']:.1%}",
                            html.Span(f" ({prod_delta:+.1%})", style={"color": prod_color}, className="ms-1"),
                        ]
                    ),
                ],
                md=6,
            ),
            dbc.Col(
                [
                    html.H6("Optimal Configuration", className="text-muted"),
                    html.P([html.Strong("Risk: "), f"{optimal_metrics['opt_risk']:.2%}"]),
                    html.P([html.Strong("Production: "), f"{optimal_metrics['opt_prod_pct']:.1%}"]),
                ],
                md=6,
            ),
        ]
    )

    # Create comparison chart
    fig_chart = go.Figure()

    categories = ["Risk (%)", "Production (%)"]
    fig_chart.add_trace(
        go.Bar(
            name="Current",
            x=categories,
            y=[current_metrics["opt_risk"] * 100, current_metrics["opt_prod_pct"] * 100],
            marker_color=COLOR_PRIMARY,
        )
    )
    fig_chart.add_trace(
        go.Bar(
            name="Optimal",
            x=categories,
            y=[optimal_metrics["opt_risk"] * 100, optimal_metrics["opt_prod_pct"] * 100],
            marker_color=COLOR_PRODUCTION,
        )
    )

    fig_chart.update_layout(
        barmode="group",
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis_title="Value (%)",
    )
    apply_plotly_style(fig_chart, height=280)

    # Create heatmap showing which cells pass/fail
    df = summary_data.copy()
    df["cut_limit"] = df[var0_col].map(current_cuts)
    df["passes"] = (df[var1_col] <= df["cut_limit"]).astype(int)

    # Pivot for heatmap
    pivot = df.pivot_table(index=var1_col, columns=var0_col, values="passes", aggfunc="first")
    pivot = pivot.sort_index(ascending=False)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(int(c)) for c in pivot.columns],
            y=[str(int(r)) for r in pivot.index],
            colorscale=[[0, "#ffcccc"], [1, "#ccffcc"]],
            showscale=False,
            hovertemplate=f"{var0_col}: %{{x}}<br>{var1_col}: %{{y}}<br>Status: %{{customdata}}<extra></extra>",
            customdata=[["Pass" if v == 1 else "Reject" for v in row] for row in pivot.values],
        )
    )

    # Add cutoff line
    for i, bin_val in enumerate(pivot.columns):
        cut_val = current_cuts.get(int(bin_val), 9)
        fig_heatmap.add_shape(
            type="line",
            x0=i - 0.5,
            x1=i + 0.5,
            y0=len(pivot.index) - cut_val - 0.5,
            y1=len(pivot.index) - cut_val - 0.5,
            line=dict(color="red", width=3),
        )

    fig_heatmap.update_layout(
        xaxis_title=var0_col,
        yaxis_title=var1_col,
        margin=dict(l=60, r=20, t=30, b=60),
    )
    apply_plotly_style(fig_heatmap, height=380)

    return results, fig_chart, fig_heatmap


# Reset button callback
@app.callback(
    Output({"type": "cutoff-slider", "index": dash.ALL}, "value"),
    [Input("reset-cutoffs-btn", "n_clicks"), Input("apply-risk-btn", "n_clicks")],
    [
        State("cutoff-data-store", "data"),
        State("risk-level-slider", "value"),
        State({"type": "cutoff-slider", "index": dash.ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def update_cutoffs_from_buttons(reset_clicks, apply_clicks, store_data, risk_value, current_values):
    """Update sliders from reset button or risk level selection."""
    if not store_data:
        return current_values

    ctx = callback_context
    if not ctx.triggered:
        return current_values

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    bins = store_data["bins"]

    if trigger_id == "reset-cutoffs-btn":
        # Reset to original optimal values
        optimal_cuts = {int(k): v for k, v in store_data["optimal_cuts"].items()}
        return [optimal_cuts.get(bin_val, 9) for bin_val in bins]

    elif trigger_id == "apply-risk-btn":
        # Find closest Pareto-optimal solution to selected risk level
        pareto_solutions = store_data.get("pareto_solutions", [])
        if not pareto_solutions or risk_value is None:
            return current_values

        # Find the solution with risk closest to the selected value
        closest_solution = None
        min_diff = float("inf")
        for sol in pareto_solutions:
            diff = abs(sol.get("b2_ever_h6", 0) - risk_value)
            if diff < min_diff:
                min_diff = diff
                closest_solution = sol

        if closest_solution is None:
            return current_values

        # Extract cutoffs from the closest solution
        new_values = []
        for bin_val in bins:
            # Try both integer and string keys
            cutoff = (
                closest_solution.get(bin_val)
                or closest_solution.get(str(bin_val))
                or closest_solution.get(float(bin_val))
            )
            if cutoff is not None:
                new_values.append(int(cutoff))
            else:
                # Fallback to current value
                idx = bins.index(bin_val)
                new_values.append(current_values[idx] if idx < len(current_values) else 9)

        return new_values

    return current_values


# Risk selection info callback
@app.callback(
    Output("risk-selection-info", "children"),
    [Input("risk-level-slider", "value")],
    [State("cutoff-data-store", "data")],
    prevent_initial_call=False,
)
def update_risk_info(risk_value, store_data):
    """Show info about the closest optimal solution to selected risk."""
    if not store_data or risk_value is None:
        return ""

    pareto_solutions = store_data.get("pareto_solutions", [])
    if not pareto_solutions:
        return html.Span("No Pareto solutions available. Run the pipeline first.", className="text-warning")

    # Find closest solution
    closest_solution = None
    min_diff = float("inf")
    for sol in pareto_solutions:
        diff = abs(sol.get("b2_ever_h6", 0) - risk_value)
        if diff < min_diff:
            min_diff = diff
            closest_solution = sol

    if closest_solution is None:
        return ""

    actual_risk = closest_solution.get("b2_ever_h6", 0)
    production = closest_solution.get("oa_amt_h0", 0)

    return html.Div(
        [
            html.Span("Closest optimal solution: ", className="fw-bold"),
            html.Span(f"Risk = {actual_risk:.2f}%, Production = €{production:,.0f}"),
            html.Br(),
            html.Span(f"({len(pareto_solutions)} Pareto-optimal solutions available)", className="text-muted"),
        ]
    )


# --- New Callbacks: Collapsible, Download, Audit Period ---


@app.callback(
    Output({"type": "collapse-section", "index": MATCH}, "is_open"),
    [Input({"type": "collapse-btn", "index": MATCH}, "n_clicks")],
    [State({"type": "collapse-section", "index": MATCH}, "is_open")],
    prevent_initial_call=True,
)
def toggle_collapse(n_clicks, is_open):
    """Toggle collapsible section open/closed."""
    if n_clicks:
        return not is_open
    return is_open


@app.callback(
    Output({"type": "download-sink", "index": MATCH}, "data"),
    [Input({"type": "download-btn", "index": MATCH}, "n_clicks")],
    [State("scenario-selector", "value"), State("segment-selector", "value")],
    prevent_initial_call=True,
)
def download_table(n_clicks, scenario, segment):
    """Send CSV download for the matching table."""
    if not n_clicks or not scenario:
        return dash.no_update

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update

    # Extract index from trigger to determine which file
    trigger = ctx.triggered_id
    index = trigger["index"] if isinstance(trigger, dict) else "main"

    paths = get_scenario_paths(scenario, segment)
    file_map = {
        "main": paths["main_csv"],
        "mr": paths["mr_csv"],
    }

    # Also handle audit downloads
    if index in ("audit_main", "audit_mr"):
        audit_paths = get_audit_paths(scenario, segment)
        period = "main" if index == "audit_main" else "mr"
        file_map[index] = audit_paths[period]

    csv_path = file_map.get(index)
    if csv_path is None or not csv_path.exists():
        return dash.no_update

    df = pd.read_csv(csv_path)
    return dcc.send_data_frame(df.to_csv, csv_path.name, index=False)


@app.callback(
    Output("audit-content", "children"),
    [Input("audit-period-toggle", "value")],
    [State("scenario-selector", "value"), State("segment-selector", "value")],
    prevent_initial_call=True,
)
def update_audit_period(period, scenario, segment):
    """Re-render audit content for selected period."""
    if not scenario:
        return html.Div("No scenario selected", className="text-muted")
    return create_audit_tab_content(scenario, segment, period)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Risk Scoring Dashboard", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", "-o", default="output", help="Base output directory for batch results (default: output)"
    )
    parser.add_argument("--segment", "-s", default=None, help="Initial segment to display")
    parser.add_argument("--port", "-p", type=int, default=8050, help="Port to run dashboard on (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set global output base
    OUTPUT_BASE = Path(args.output)
    CURRENT_SEGMENT = args.segment

    # Auto-detect: if output dir doesn't exist, check root data/images
    if not OUTPUT_BASE.exists():
        if Path("data").exists() or Path("images").exists():
            OUTPUT_BASE = Path(".")
            logger.info("Using root directory for output (single-run mode)")
        else:
            logger.warning(f"Output directory '{args.output}' does not exist")

    logger.info("Starting Risk Scoring Dashboard...")
    logger.info(f"Output base: {OUTPUT_BASE}")

    segments = get_available_segments()
    if segments:
        logger.info(f"Found segments: {segments}")
    else:
        scenarios = get_scenarios(None)
        if scenarios:
            logger.info(f"Found scenarios (single-run mode): {scenarios}")
        else:
            logger.warning("No data found. Run 'uv run python main.py' or 'uv run python run_batch.py' first.")

    app.run(debug=args.debug, port=args.port)
