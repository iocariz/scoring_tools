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
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Symbol
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
from src.utils import calculate_b2_ever_h6

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

    # SHAP resolution (might be in supersegment)
    shap_prefix = get_shap_static_prefix(segment)
    paths["shap_summary"] = f"{shap_prefix}/shap_summary.html"

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
    # Check if this is a SHAP visualization request routing to a supersegment
    if "_supersegment_" in viz_path:
        # Extract supersegment name from path components
        parts = viz_path.split("/")
        superseg_folder = next((p for p in parts if p.startswith("_supersegment_")), None)
        if superseg_folder:
            return (OUTPUT_BASE / superseg_folder / "images" / filename).exists()

    images_dir = get_images_dir(segment)
    return (images_dir / filename).exists()


def _resolve_supersegment(segment: str | None) -> str | None:
    """Look up the supersegment name for *segment* from ``segments.toml``."""
    if not segment:
        return None
    try:
        import tomllib

        with open("segments.toml", "rb") as f:
            seg_config = tomllib.load(f).get("segments", {})
            cfg = seg_config.get(segment, {})
            return cfg.get("modelling_supersegment") or cfg.get("supersegment")
    except Exception:
        return None


def get_shap_images_dir(segment: str | None = None) -> Path:
    """Resolve the images directory for SHAP specifically (handles supersegments)."""
    superseg = _resolve_supersegment(segment)
    if superseg:
        return OUTPUT_BASE / f"_supersegment_{superseg}" / "images"
    return get_images_dir(segment)


def get_shap_static_prefix(segment: str | None = None) -> str:
    """Resolve the static prefix for SHAP specifically (handles supersegments)."""
    superseg = _resolve_supersegment(segment)
    if superseg:
        return f"/static/_supersegment_{superseg}"
    return f"/static/{segment}" if segment else "/static"


def get_shap_dependence_features(segment: str | None = None) -> list[str]:
    """Get list of features that have SHAP dependence plots available."""
    images_dir = get_shap_images_dir(segment)
    if not images_dir.exists():
        return []

    features = []
    for f in images_dir.glob("shap_dependence_*.html"):
        # Extrapolate feature name from 'shap_dependence_feature_name.html'
        feature_name = f.stem.replace("shap_dependence_", "")
        features.append(feature_name)
    return sorted(features)


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

        # Format numeric columns for better display.
        # Risk (%) is already in percentage form (1.5 means 1.5%).
        # Production (%) is a raw ratio (0.95 means 95%).
        for col in df.select_dtypes(include=["float64"]).columns:
            if "Risk" in col and "%" in col:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
            elif "Production" in col and "%" in col:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
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
            # Risk (%) is already in percentage form (1.5 means 1.5%) — keep as-is.
            # Production (%) is a raw ratio (0.95 means 95%) — keep as-is.
            if paths["main_csv"].exists():
                df_main = pd.read_csv(paths["main_csv"])
                opt_row = df_main[df_main["Metric"] == "Optimum selected"]
                if not opt_row.empty:
                    row["Main_Risk"] = (
                        opt_row["Risk (%)"].values[0] if not pd.isna(opt_row["Risk (%)"].values[0]) else None
                    )
                    row["Main_Prod_Pct"] = (
                        opt_row["Production (%)"].values[0]
                        if not pd.isna(opt_row["Production (%)"].values[0])
                        else None
                    )

            # MR Period Metrics
            if paths["mr_csv"].exists():
                df_mr = pd.read_csv(paths["mr_csv"])
                opt_row_mr = df_mr[df_mr["Metric"] == "Optimum selected"]
                if not opt_row_mr.empty:
                    row["MR_Risk"] = (
                        opt_row_mr["Risk (%)"].values[0] if not pd.isna(opt_row_mr["Risk (%)"].values[0]) else None
                    )
                    row["MR_Prod_Pct"] = (
                        opt_row_mr["Production (%)"].values[0]
                        if not pd.isna(opt_row_mr["Production (%)"].values[0])
                        else None
                    )

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
                    text=f"\u0394 {delta:+.2f}pp",
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
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
    for col in ["Main_Prod_Pct", "MR_Prod_Pct"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
    for col in ["Risk_Delta"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:+.2f}pp" if pd.notna(x) else "-")
    for col in ["Prod_Delta"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:+.1%}" if pd.notna(x) else "-")

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


def create_kpi_card(title: str, value: str, delta: str = "", color_border: str = COLOR_PRIMARY) -> dbc.Col:
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


def create_kpi_row(main_csv_path: Path, mr_csv_path: Path, comparison_label: str = "vs MR") -> dbc.Row:
    """Create a row of 4 KPI cards from main and MR CSV data.

    Args:
        main_csv_path: Primary CSV (displayed as the main value).
        mr_csv_path: Comparison CSV (used for delta calculation).
        comparison_label: Label for the delta (e.g. "vs MR" or "vs Main").
    """
    main_risk, main_prod_pct, main_prod_eur = None, None, None
    mr_risk, mr_prod_pct = None, None

    try:
        if main_csv_path.exists():
            df = pd.read_csv(main_csv_path)
            opt = df[df["Metric"] == "Optimum selected"]
            if not opt.empty:
                # Risk (%) is already percentage form (1.5 = 1.5%) — keep as-is.
                # Production (%) is a raw ratio (0.95 = 95%) — keep as-is.
                main_risk = opt["Risk (%)"].values[0] if not pd.isna(opt["Risk (%)"].values[0]) else None
                main_prod_pct = (
                    opt["Production (%)"].values[0] if not pd.isna(opt["Production (%)"].values[0]) else None
                )
                if "Production (€)" in df.columns:
                    main_prod_eur = opt["Production (€)"].values[0]

        if mr_csv_path.exists():
            df_mr = pd.read_csv(mr_csv_path)
            opt_mr = df_mr[df_mr["Metric"] == "Optimum selected"]
            if not opt_mr.empty:
                mr_risk = opt_mr["Risk (%)"].values[0] if not pd.isna(opt_mr["Risk (%)"].values[0]) else None
                mr_prod_pct = (
                    opt_mr["Production (%)"].values[0] if not pd.isna(opt_mr["Production (%)"].values[0]) else None
                )
    except Exception as e:
        logger.warning(f"Error loading KPI data: {e}")

    # Build cards — risk is already in % form (1.5 = 1.5%), production is ratio (0.95 = 95%)
    risk_val = f"{main_risk:.2f}%" if main_risk is not None else "N/A"
    risk_delta = (
        f"{main_risk - mr_risk:+.2f}pp {comparison_label}" if main_risk is not None and mr_risk is not None else ""
    )

    prod_pct_val = f"{main_prod_pct:.1%}" if main_prod_pct is not None else "N/A"
    prod_pct_delta = (
        f"{main_prod_pct - mr_prod_pct:+.1%} {comparison_label}"
        if main_prod_pct is not None and mr_prod_pct is not None
        else ""
    )

    prod_eur_val = f"EUR {main_prod_eur:,.0f}" if main_prod_eur is not None else "N/A"

    risk_delta_val = f"{main_risk - mr_risk:+.2f}pp" if main_risk is not None and mr_risk is not None else "N/A"
    risk_delta_suffix = comparison_label if main_risk is not None and mr_risk is not None else ""

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

        # Identify risk columns for conditional formatting.
        # Risk (%) values are in percentage form (1.5 means 1.5%).
        style_data_conditional = []
        for col in df.columns:
            if "Risk" in col and "%" in col:
                # Values > 2% get red highlight
                style_data_conditional.append(
                    {
                        "if": {"column_id": col, "filter_query": f"{{{col}}} > 2.0"},
                        "className": "cell-high-risk",
                    }
                )
                # Values < 1% get green highlight
                style_data_conditional.append(
                    {
                        "if": {"column_id": col, "filter_query": f"{{{col}}} < 1.0"},
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
        style_cell={"textAlign": "right", "padding": "6px", "fontSize": "0.83rem"},
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
        # Try supersegment fallback via segments.toml
        superseg = _resolve_supersegment(segment)
        if superseg:
            models_dir = OUTPUT_BASE / f"_supersegment_{superseg}" / "models"
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

    coefficients = parse_coefficients(model_dir)
    intercept_val = next((c["coefficient"] for c in coefficients if c["feature"].lower() == "intercept"), "N/A")
    if intercept_val != "N/A":
        info_items.append(("Intercept", f"{intercept_val:.6f}"))

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

    # Performance metrics
    perf_metrics = [
        ("CV Mean RMSE", metadata.get("cv_mean_rmse")),
        ("Train R² (in-sample)", metadata.get("train_r2", metadata.get("full_r2"))),
        ("LOO-CV R²", metadata.get("loo_r2")),
        ("Step 1 CV RMSE", metadata.get("step1_cv_rmse")),
        ("Step 2 CV RMSE", metadata.get("step2_cv_rmse")),
    ]

    metric_items = []
    for label, val in perf_metrics:
        if val is not None:
            metric_items.append(
                html.Div(
                    [html.Span(label, className="small fw-bold"), html.Span(f"{val:.4f}", className="small float-end")],
                    className="mb-2 border-bottom pb-1",
                )
            )

    performance_card = dbc.Card(
        [
            dbc.CardHeader("Performance Metrics"),
            dbc.CardBody(metric_items if metric_items else "No metrics available"),
        ],
        className="mb-4",
    )

    coef_chart = html.Div()
    if coefficients:
        # Exclude intercept from the main bar chart to avoid skewing the scale (it's often much larger)
        coef_display = [c for c in coefficients if c["feature"].lower() != "intercept"]
        if coef_display:
            coef_df = pd.DataFrame(coef_display)
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
            apply_plotly_style(fig_coef, title="Feature Coefficients", height=max(250, len(coef_display) * 40))
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
) -> tuple[
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame | None,
    list[str],
    list[str],
    float,
    pd.DataFrame | None,
    list[int] | None,
]:
    """
    Load data needed for cutoff explorer.

    Returns:
        Tuple of (summary_data, optimal_solution, pareto_solutions, variables,
                  inv_vars, multiplier, cell_ci, optimal_mask)
        optimal_mask is a decoded acceptance mask list for N>2, or None for 2-var.
    """
    from src.config import PreprocessingSettings

    # Extract variables, inv_vars and multiplier from configuration, preferring the segment's own config.
    # inv_vars is populated at runtime by preprocessing (_infer_monotonicity) and is never
    # written to the TOML config.  We must derive it from the ``directions`` dict instead.
    inv_vars = []
    config_variables: list[str] = []
    multiplier = 7.0
    try:
        config_candidates = []
        if segment:
            config_candidates.append(OUTPUT_BASE / segment / "config_segment.toml")
        config_candidates.append(Path("config.toml"))
        for config_path in config_candidates:
            if config_path.exists():
                settings = PreprocessingSettings.from_toml(str(config_path))
                multiplier = settings.multiplier
                # Derive inv_vars from directions: direction == -1 means higher bin = safer
                inv_vars = [var for var, d in settings.directions.items() if d == -1]
                config_variables = list(settings.variables)
                break
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
    paths = get_scenario_paths(scenario, segment)
    data_dir = get_data_dir(segment)

    summary_data = None
    optimal_solution = None
    pareto_solutions = None
    variables: list[str] = []

    # Load summary data
    if paths["summary_data"].exists():
        summary_data = pd.read_csv(paths["summary_data"])

        # Use config-derived variables if available (supports N variables)
        if config_variables:
            # Resolve actual column names: config names may differ from CSV column names.
            # Look up bin output columns from preprocessing.bins config or match directly.
            variables = [v for v in config_variables if v in summary_data.columns]

        # Fallback: pattern-match cluster columns (legacy detection, any N)
        if not variables:
            for col in summary_data.columns:
                if "clus" in col.lower() or col in ["sc_octroi_new_clus", "new_efx_clus"]:
                    variables.append(col)

    # Load optimal solution
    if paths["cutoffs"].exists():
        optimal_solution = pd.read_csv(paths["cutoffs"])

    # Load all Pareto-optimal solutions (for risk slider)
    pareto_path = data_dir / "pareto_optimal_solutions.csv"
    if pareto_path.exists():
        pareto_solutions = pd.read_csv(pareto_path)

    # Load cell-level CI if available
    cell_ci = None
    models_dir = get_models_dir(segment)
    if models_dir.exists():
        # Search in model subdirectories for cell_level_ci.csv
        ci_candidates = list(models_dir.glob("model_*/cell_level_ci.csv"))
        if ci_candidates:
            # Use the most recent one
            ci_path = sorted(ci_candidates, reverse=True)[0]
            try:
                cell_ci = pd.read_csv(ci_path)
            except Exception as e:
                logger.warning(f"Could not load cell CI from {ci_path}: {e}")

    # Decode optimal acceptance mask for 1D and N>2 (both use mask-based logic)
    optimal_mask: list[int] | None = None
    if (
        len(variables) != 2
        and optimal_solution is not None
        and not optimal_solution.empty
        and "acceptance_mask" in optimal_solution.columns
    ):
        from src.optimization_utils import decode_mask

        mask_str = str(optimal_solution["acceptance_mask"].iloc[0])
        optimal_mask = decode_mask(mask_str).tolist()

    return summary_data, optimal_solution, pareto_solutions, variables, inv_vars, multiplier, cell_ci, optimal_mask


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _format_currency(value: Any, compact: bool = False) -> str:
    amount = _coerce_float(value)
    sign = "-" if amount < 0 else ""
    abs_amount = abs(amount)
    if compact:
        if abs_amount >= 1_000_000_000:
            return f"{sign}€{abs_amount / 1_000_000_000:.2f}B"
        if abs_amount >= 1_000_000:
            return f"{sign}€{abs_amount / 1_000_000:.1f}M"
        if abs_amount >= 1_000:
            return f"{sign}€{abs_amount / 1_000:.0f}k"
    return f"{sign}€{abs_amount:,.0f}"


def _build_risk_marks(risk_min: float, risk_max: float) -> dict[float, str]:
    if risk_max <= risk_min:
        rounded = round(risk_min, 2)
        return {rounded: f"{rounded:.2f}%"}
    marks = {}
    for value in np.linspace(risk_min, risk_max, 5):
        rounded = round(float(value), 2)
        marks[rounded] = f"{rounded:.2f}%"
    return dict(sorted(marks.items()))


def _build_cutoff_stat_card(title: str, value: str, subtitle: str, accent: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted text-uppercase small fw-semibold mb-1"),
                html.Div(value, className="fw-bold mb-1", style={"fontSize": "1.1rem"}),
                html.Div(subtitle, className="text-muted small"),
            ]
        ),
        className="h-100 shadow-sm border-0",
        style={"borderTop": f"4px solid {accent}"},
    )


def _build_cutoff_results_content(
    current_metrics: dict[str, Any], reference_metrics: dict[str, Any], reference_label: str
) -> html.Div:
    actual_prod = _coerce_float(current_metrics.get("actual_prod"))
    current_prod = _coerce_float(current_metrics.get("opt_prod"))
    current_risk = _coerce_float(current_metrics.get("opt_risk"))
    current_prod_pct = _coerce_float(current_metrics.get("opt_prod_pct")) * 100
    reference_prod = _coerce_float(reference_metrics.get("opt_prod"))
    reference_risk = _coerce_float(reference_metrics.get("opt_risk"))
    reference_prod_pct = _coerce_float(reference_metrics.get("opt_prod_pct")) * 100
    risk_gap = current_risk - reference_risk
    prod_gap = current_prod - reference_prod
    prod_gap_pct = current_prod_pct - reference_prod_pct
    risk_accent = COLOR_GOOD if risk_gap <= 0 else COLOR_BAD
    gap_accent = COLOR_GOOD if prod_gap >= 0 else COLOR_BAD

    summary_cards = dbc.Row(
        [
            dbc.Col(
                _build_cutoff_stat_card(
                    "Actual booked",
                    _format_currency(actual_prod, compact=True),
                    "Baseline production before cut changes",
                    COLOR_NEUTRAL,
                ),
                md=6,
                xl=3,
                className="mb-3",
            ),
            dbc.Col(
                _build_cutoff_stat_card(
                    "Current production",
                    _format_currency(current_prod, compact=True),
                    f"{current_prod_pct:.1f}% of actual booked production",
                    COLOR_PRODUCTION,
                ),
                md=6,
                xl=3,
                className="mb-3",
            ),
            dbc.Col(
                _build_cutoff_stat_card(
                    "Current risk",
                    f"{current_risk:.2f}%",
                    f"{risk_gap:+.2f}pp vs {reference_label.lower()}",
                    risk_accent,
                ),
                md=6,
                xl=3,
                className="mb-3",
            ),
            dbc.Col(
                _build_cutoff_stat_card(
                    f"Gap vs {reference_label}",
                    _format_currency(prod_gap, compact=True),
                    f"{prod_gap_pct:+.1f} pts production share vs {reference_label.lower()}",
                    gap_accent,
                ),
                md=6,
                xl=3,
                className="mb-3",
            ),
        ],
        className="g-3",
    )

    comparison_table = dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Configuration"),
                        html.Th("Risk"),
                        html.Th("Production"),
                        html.Th("Share of Actual"),
                        html.Th("Swap-in"),
                        html.Th("Swap-out"),
                    ]
                )
            ),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(html.Strong("Current")),
                            html.Td(f"{current_risk:.2f}%"),
                            html.Td(_format_currency(current_prod)),
                            html.Td(f"{current_prod_pct:.1f}%"),
                            html.Td(_format_currency(current_metrics.get("swap_in_prod"))),
                            html.Td(_format_currency(current_metrics.get("swap_out_prod"))),
                        ]
                    ),
                    html.Tr(
                        [
                            html.Td(reference_label),
                            html.Td(f"{reference_risk:.2f}%"),
                            html.Td(_format_currency(reference_prod)),
                            html.Td(f"{reference_prod_pct:.1f}%"),
                            html.Td(_format_currency(reference_metrics.get("swap_in_prod"))),
                            html.Td(_format_currency(reference_metrics.get("swap_out_prod"))),
                        ]
                    ),
                ]
            ),
        ],
        size="sm",
        hover=True,
        responsive=True,
        className="mb-0 align-middle",
    )

    return html.Div([summary_cards, comparison_table])


def _build_pareto_frontier_figure(
    pareto_solutions: list[dict[str, Any]],
    current_metrics: dict[str, Any],
    reference_metrics: dict[str, Any],
    reference_label: str,
) -> go.Figure:
    fig = go.Figure()
    frontier_df = pd.DataFrame(pareto_solutions or [])
    if not frontier_df.empty and {"b2_ever_h6", "oa_amt_h0"}.issubset(frontier_df.columns):
        frontier_df = frontier_df.copy()
        frontier_df["b2_ever_h6"] = pd.to_numeric(frontier_df["b2_ever_h6"], errors="coerce")
        frontier_df["oa_amt_h0"] = pd.to_numeric(frontier_df["oa_amt_h0"], errors="coerce")
        frontier_df = frontier_df.dropna(subset=["b2_ever_h6", "oa_amt_h0"]).sort_values("b2_ever_h6")

    if not frontier_df.empty:
        fig.add_trace(
            go.Scatter(
                x=frontier_df["b2_ever_h6"],
                y=frontier_df["oa_amt_h0"],
                mode="lines+markers",
                name="Pareto frontier",
                line=dict(color=COLOR_NEUTRAL, width=2),
                marker=dict(size=7, color=COLOR_NEUTRAL, opacity=0.8),
                hovertemplate="Pareto frontier<br>Risk %{x:.2f}%<br>Production €%{y:,.0f}<extra></extra>",
            )
        )

    actual_prod = _coerce_float(current_metrics.get("actual_prod"))
    if actual_prod > 0:
        fig.add_hline(
            y=actual_prod,
            line=dict(color=COLOR_NEUTRAL, dash="dot", width=1.5),
            annotation_text="Actual booked baseline",
            annotation_position="top left",
        )

    fig.add_trace(
        go.Scatter(
            x=[_coerce_float(current_metrics.get("opt_risk"))],
            y=[_coerce_float(current_metrics.get("opt_prod"))],
            mode="markers",
            name="Current configuration",
            marker=dict(color=COLOR_PRIMARY, size=14, symbol="diamond", line=dict(color="white", width=1.5)),
            hovertemplate="Current configuration<br>Risk %{x:.2f}%<br>Production €%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[_coerce_float(reference_metrics.get("opt_risk"))],
            y=[_coerce_float(reference_metrics.get("opt_prod"))],
            mode="markers",
            name=reference_label,
            marker=dict(color=COLOR_PRODUCTION, size=12, symbol="circle", line=dict(color="white", width=1.5)),
            hovertemplate=f"{reference_label}<br>Risk %{{x:.2f}}%<br>Production €%{{y:,.0f}}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(l=50, r=20, t=30, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis_title="Risk (%)",
        yaxis_title="Production (€)",
        hovermode="closest",
    )
    apply_plotly_style(fig, height=320)
    return fig


def _pin_counts(pinned_cells: dict[str, int] | None) -> tuple[int, int, int]:
    pins = pinned_cells or {}
    n_pins = len(pins)
    n_accept = sum(1 for value in pins.values() if value == 1)
    n_reject = sum(1 for value in pins.values() if value == 0)
    return n_pins, n_accept, n_reject


def _format_pin_state(pin_value: int | None) -> str:
    if pin_value == 1:
        return "Pinned Accept"
    if pin_value == 0:
        return "Pinned Reject"
    return "Unpinned"


def _build_nd_acceptance_table_columns(variables: list[str]) -> list[dict[str, Any]]:
    currency_format = Format(precision=0, scheme=Scheme.fixed, group=Group.yes, symbol=Symbol.yes, symbol_prefix="€")
    pct_format = Format(precision=2, scheme=Scheme.fixed)
    columns = [{"name": var, "id": var, "type": "numeric"} for var in variables]
    columns.extend(
        [
            {"name": "Current", "id": "current_state"},
            {"name": "Reference", "id": "reference_state"},
            {"name": "Pin", "id": "pin_state"},
            {"name": "Delta", "id": "decision_delta"},
            {"name": "Booked €", "id": "booked_prod", "type": "numeric", "format": currency_format},
            {"name": "Repesca €", "id": "rep_prod", "type": "numeric", "format": currency_format},
            {"name": "Booked Risk %", "id": "booked_risk_pct", "type": "numeric", "format": pct_format},
        ]
    )
    return columns


def _build_nd_acceptance_table_rows(
    summary_data: pd.DataFrame,
    variables: list[str],
    current_mask: list[int] | np.ndarray | None,
    optimal_mask: list[int] | np.ndarray | None,
    pinned_cells: dict[str, int] | None,
    multiplier: float,
) -> list[dict[str, Any]]:
    from src.optimization_utils import CellGrid

    grid = CellGrid.from_summary(summary_data, variables)
    pins = pinned_cells or {}
    fallback_mask = np.zeros(grid.n_cells, dtype=int)
    if optimal_mask is not None:
        fallback_mask = np.asarray(optimal_mask, dtype=int)
    current_mask_arr = np.asarray(current_mask, dtype=int) if current_mask is not None else fallback_mask
    optimal_mask_arr = np.asarray(optimal_mask, dtype=int) if optimal_mask is not None else current_mask_arr
    rows: list[dict[str, Any]] = []

    for combo, idx in grid.cell_index.items():
        cell_key = ",".join(str(int(value)) for value in combo)
        cell_row = grid.cell_data.iloc[idx]
        booked_prod = _coerce_float(cell_row.get("oa_amt_h0_boo"))
        rep_prod = _coerce_float(cell_row.get("oa_amt_h0_rep"))
        risk_num = _coerce_float(cell_row.get("todu_30ever_h6_boo"))
        risk_den = _coerce_float(cell_row.get("todu_amt_pile_h6_boo"))
        booked_risk_raw = calculate_b2_ever_h6(
            risk_num,
            risk_den,
            multiplier=multiplier,
            as_percentage=True,
            decimals=2,
        )
        current_state = "Accept" if current_mask_arr[idx] == 1 else "Reject"
        reference_state = "Accept" if optimal_mask_arr[idx] == 1 else "Reject"
        row = {
            "id": cell_key,
            "current_state": current_state,
            "reference_state": reference_state,
            "pin_state": _format_pin_state(pins.get(cell_key)),
            "decision_delta": "Changed" if current_state != reference_state else "Reference",
            "booked_prod": booked_prod,
            "rep_prod": rep_prod,
            "booked_risk_pct": _coerce_float(booked_risk_raw),
        }
        for dim, variable in enumerate(variables):
            row[variable] = int(combo[dim])
        rows.append(row)

    return rows


def _build_pin_status_children(pin_mode: bool, pinned_cells: dict[str, int] | None) -> html.Div:
    n_pins, n_accept, n_reject = _pin_counts(pinned_cells)
    badges: list[dbc.Badge] = [
        dbc.Badge("Pin Mode On" if pin_mode else "Pin Mode Off", color="warning" if pin_mode else "secondary"),
        dbc.Badge(f"{n_pins} pinned", color="dark"),
    ]
    if n_pins:
        badges.extend(
            [
                dbc.Badge(f"{n_accept} accept", color="success"),
                dbc.Badge(f"{n_reject} reject", color="danger"),
            ]
        )
    helper_text = (
        "Click a table row or heatmap cell to cycle unpinned → accept → reject while Pin Mode is enabled."
        if pin_mode
        else "Enable Pin Mode to add or update pinned constraints."
    )
    return html.Div(
        [
            html.Div(badges, className="d-flex flex-wrap gap-2 mb-1"),
            html.Div(helper_text, className="text-muted small"),
        ]
    )


def _build_pin_feedback_alert(feedback: dict[str, Any] | None) -> html.Div | dbc.Alert:
    if not feedback or not feedback.get("message"):
        return html.Div()
    tone = str(feedback.get("tone", "secondary"))
    color_map = {
        "success": "success",
        "warning": "warning",
        "danger": "danger",
        "info": "info",
        "secondary": "secondary",
    }
    return dbc.Alert(feedback["message"], color=color_map.get(tone, "secondary"), className="py-2 mb-0")


def _format_bin_label(value: Any) -> str:
    numeric = _coerce_float(value, np.nan)
    if pd.notna(numeric):
        if float(numeric).is_integer():
            return str(int(numeric))
        return f"{numeric:g}"
    return str(value)


def _format_nd_prefix_label(prefix_values: tuple[Any, ...], prefix_vars: list[str]) -> str:
    if not prefix_vars:
        return "All cells"
    return " | ".join(
        f"{var}={_format_bin_label(value)}" for var, value in zip(prefix_vars, prefix_values, strict=False)
    )


def _build_nd_slice_grid_figure(
    summary_data: pd.DataFrame,
    variables: list[str],
    fixed_vars: dict[str, Any],
    grid_x_var: str,
    grid_y_var: str,
    current_mask: list[int] | np.ndarray | None,
    optimal_mask: list[int] | np.ndarray | None,
    pinned_cells: dict[str, int] | None,
    multiplier: float,
) -> go.Figure:
    """Build a 2D heatmap slice for any N>=3, fixing *fixed_vars* and using
    *grid_x_var* / *grid_y_var* as the two displayed axes."""
    from src.optimization_utils import CellGrid

    if len(variables) < 3:
        return go.Figure()

    grid = CellGrid.from_summary(summary_data, variables)
    x_values = list(grid.values_per_var[grid_x_var])
    y_values = list(grid.values_per_var[grid_y_var])
    x_labels = [_format_bin_label(v) for v in x_values]
    y_labels = [_format_bin_label(v) for v in y_values]
    fixed_label = " | ".join(f"{k}={_format_bin_label(v)}" for k, v in fixed_vars.items())
    pins = pinned_cells or {}
    fallback_mask = np.zeros(grid.n_cells, dtype=int)
    if optimal_mask is not None:
        fallback_mask = np.asarray(optimal_mask, dtype=int)
    current_mask_arr = np.asarray(current_mask, dtype=int) if current_mask is not None else fallback_mask
    optimal_mask_arr = np.asarray(optimal_mask, dtype=int) if optimal_mask is not None else current_mask_arr

    z_values: list[list[int]] = []
    customdata: list[list[list[Any]]] = []
    changed_x: list[str] = []
    changed_y: list[str] = []
    changed_keys: list[str] = []
    pinned_accept_x: list[str] = []
    pinned_accept_y: list[str] = []
    pinned_accept_keys: list[str] = []
    pinned_reject_x: list[str] = []
    pinned_reject_y: list[str] = []
    pinned_reject_keys: list[str] = []

    for y_value, y_label in zip(y_values, y_labels, strict=False):
        row_values: list[int] = []
        row_customdata: list[list[Any]] = []
        for x_value, x_label in zip(x_values, x_labels, strict=False):
            # Build full combo tuple in variable order
            combo = tuple(
                x_value if v == grid_x_var else y_value if v == grid_y_var else fixed_vars[v]
                for v in variables
            )
            idx = grid.cell_index.get(combo)
            if idx is None:
                row_values.append(0)
                row_customdata.append(["?", "N/A", "N/A", "N/A", "N/A", 0, 0, 0])
                continue
            cell_key = ",".join(str(int(value)) for value in combo)
            cell_row = grid.cell_data.iloc[idx]
            booked_prod = _coerce_float(cell_row.get("oa_amt_h0_boo"))
            rep_prod = _coerce_float(cell_row.get("oa_amt_h0_rep"))
            risk_num = _coerce_float(cell_row.get("todu_30ever_h6_boo"))
            risk_den = _coerce_float(cell_row.get("todu_amt_pile_h6_boo"))
            booked_risk_raw = calculate_b2_ever_h6(
                risk_num,
                risk_den,
                multiplier=multiplier,
                as_percentage=True,
                decimals=2,
            )
            current_state = "Accept" if current_mask_arr[idx] == 1 else "Reject"
            reference_state = "Accept" if optimal_mask_arr[idx] == 1 else "Reject"
            decision_delta = "Changed" if current_state != reference_state else "Reference"
            pin_state = _format_pin_state(pins.get(cell_key))

            row_values.append(int(current_mask_arr[idx]))
            row_customdata.append(
                [
                    cell_key,
                    current_state,
                    reference_state,
                    decision_delta,
                    pin_state,
                    booked_prod,
                    rep_prod,
                    _coerce_float(booked_risk_raw),
                ]
            )

            if decision_delta == "Changed":
                changed_x.append(x_label)
                changed_y.append(y_label)
                changed_keys.append(cell_key)
            if pin_state == "Pinned Accept":
                pinned_accept_x.append(x_label)
                pinned_accept_y.append(y_label)
                pinned_accept_keys.append(cell_key)
            elif pin_state == "Pinned Reject":
                pinned_reject_x.append(x_label)
                pinned_reject_y.append(y_label)
                pinned_reject_keys.append(cell_key)

        z_values.append(row_values)
        customdata.append(row_customdata)

    hover_fixed = f"<br>{fixed_label}" if fixed_label else ""
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#ffcccc"], [1, "#ccffcc"]],
            showscale=False,
            xgap=1,
            ygap=1,
            customdata=customdata,
            hovertemplate=(
                f"{grid_x_var}: %{{x}}<br>{grid_y_var}: %{{y}}{hover_fixed}"
                "<br>Current: %{customdata[1]}"
                "<br>Reference: %{customdata[2]}"
                "<br>Delta: %{customdata[3]}"
                "<br>Pin: %{customdata[4]}"
                "<br>Booked production: \u20ac%{customdata[5]:,.0f}"
                "<br>Repesca production: \u20ac%{customdata[6]:,.0f}"
                "<br>Booked risk: %{customdata[7]:.2f}%<extra></extra>"
            ),
        )
    )

    if changed_x:
        fig.add_trace(
            go.Scatter(
                x=changed_x,
                y=changed_y,
                mode="markers",
                customdata=changed_keys,
                marker=dict(symbol="diamond-open", size=10, color="#f1c40f", line=dict(width=2, color="#c8a500")),
                hovertemplate="Changed vs reference<extra></extra>",
                showlegend=False,
            )
        )
    if pinned_accept_x:
        fig.add_trace(
            go.Scatter(
                x=pinned_accept_x,
                y=pinned_accept_y,
                mode="markers",
                customdata=pinned_accept_keys,
                marker=dict(symbol="star", size=12, color="#0b7a30", line=dict(width=1, color="white")),
                hovertemplate="Pinned accept<extra></extra>",
                showlegend=False,
            )
        )
    if pinned_reject_x:
        fig.add_trace(
            go.Scatter(
                x=pinned_reject_x,
                y=pinned_reject_y,
                mode="markers",
                customdata=pinned_reject_keys,
                marker=dict(symbol="x", size=11, color="#8b0000", line=dict(width=2, color="#8b0000")),
                hovertemplate="Pinned reject<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        xaxis_title=grid_x_var,
        yaxis_title=grid_y_var,
        margin=dict(l=45, r=10, t=10, b=40),
        clickmode="event+select",
        showlegend=False,
    )
    apply_plotly_style(fig, height=250)
    return fig


def _build_nd_slice_grid_panel(
    summary_data: pd.DataFrame,
    variables: list[str],
    current_mask: list[int] | np.ndarray | None,
    optimal_mask: list[int] | np.ndarray | None,
    pinned_cells: dict[str, int] | None,
    multiplier: float,
) -> list[Any]:
    """Build slice grid cards for N>=3.  Uses variables[0:2] as the displayed
    grid axes and iterates over all unique combinations of variables[2:]."""
    from itertools import product as iterproduct

    from src.optimization_utils import CellGrid

    if len(variables) < 3:
        return []

    grid = CellGrid.from_summary(summary_data, variables)
    grid_x_var, grid_y_var = variables[0], variables[1]
    slice_vars = variables[2:]

    fallback_mask = np.zeros(grid.n_cells, dtype=int)
    if optimal_mask is not None:
        fallback_mask = np.asarray(optimal_mask, dtype=int)
    current_mask_arr = np.asarray(current_mask, dtype=int) if current_mask is not None else fallback_mask

    # Build all slice combos; cap at 12 to avoid UI overload
    slice_val_lists = [list(grid.values_per_var[sv]) for sv in slice_vars]
    slice_combos = list(iterproduct(*slice_val_lists))
    max_slices = 12
    truncated = len(slice_combos) > max_slices
    slice_combos = slice_combos[:max_slices]

    columns: list[dbc.Col] = []
    for s_combo in slice_combos:
        fixed_vars = dict(zip(slice_vars, s_combo, strict=False))
        # Count accepted cells for this slice
        var_positions = {v: variables.index(v) for v in slice_vars}
        slice_indices = [
            idx for combo, idx in grid.cell_index.items()
            if all(combo[var_positions[sv]] == fv for sv, fv in fixed_vars.items())
        ]
        accepted_count = int(current_mask_arr[slice_indices].sum()) if slice_indices else 0
        total_cells = len(slice_indices)

        if len(slice_vars) == 1:
            header_text = f"{slice_vars[0]} = {_format_bin_label(s_combo[0])}"
        else:
            header_text = " | ".join(f"{sv}={_format_bin_label(v)}" for sv, v in fixed_vars.items())
        # Use a serializable string index for the graph id
        slice_key = ",".join(_format_bin_label(v) for v in s_combo)

        columns.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Span(header_text, className="fw-bold"),
                                dbc.Badge(
                                    f"{accepted_count}/{total_cells} accepted",
                                    color="light",
                                    text_color="dark",
                                    className="float-end",
                                ),
                            ]
                        ),
                        dbc.CardBody(
                            dcc.Graph(
                                id={"type": "nd-slice-grid", "index": slice_key},
                                figure=_build_nd_slice_grid_figure(
                                    summary_data,
                                    variables,
                                    fixed_vars,
                                    grid_x_var,
                                    grid_y_var,
                                    current_mask,
                                    optimal_mask,
                                    pinned_cells,
                                    multiplier,
                                ),
                                config={"displayModeBar": False},
                                style={"height": "250px"},
                            ),
                            className="p-2",
                        ),
                    ],
                    className="h-100",
                ),
                md=6,
                className="mb-3",
            )
        )

    children: list[Any] = [dbc.Row(columns, className="g-3")]
    if truncated:
        children.append(
            dbc.Alert(
                f"Showing first {max_slices} of {len(list(iterproduct(*slice_val_lists)))} slice combinations. "
                "Use the DataTable below to inspect all cells.",
                color="info",
                className="mt-2",
            )
        )
    return children


def calculate_metrics_from_custom_cuts(
    data: pd.DataFrame,
    cut_map: dict,
    var0_col: str,
    var1_col: str,
    inv_vars: list[str] | None = None,
    multiplier: float = 7.0,
) -> dict[str, Any]:
    """
    Calculate metrics for custom cutoff configuration.

    Args:
        data: Summary data DataFrame
        cut_map: Dictionary mapping var0 bins to var1 cutoff values
        var0_col: First variable column name
        var1_col: Second variable column name
        inv_vars: List of variables with inverted risk ordering
        multiplier: Risk multiplier (default 7.0)

    Returns:
        Dictionary with calculated metrics
    """
    if inv_vars is None:
        inv_vars = []

    df = data.copy()

    # Normalize cut_map keys to float to match data column types
    cut_map_norm = {float(k): v for k, v in cut_map.items()}

    # Apply cuts
    df["cut_limit"] = df[var0_col].astype(float).map(cut_map_norm)

    # Respect the optimization direction (RMSE/error is typically an inv_var where higher score = lower risk)
    if var1_col in inv_vars:
        df["passes_cut"] = df[var1_col] >= df["cut_limit"]
    else:
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
        b2_ever = (risk_num / risk_den * multiplier) if risk_den > 0 else 0.0
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
    opt_risk = (opt_rn / opt_rd * multiplier) if opt_rd > 0 else 0.0

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


def calculate_metrics_from_mask(
    data: pd.DataFrame,
    mask: list[int] | np.ndarray,
    variables: list[str],
    multiplier: float = 7.0,
) -> dict[str, Any]:
    """Calculate metrics from an N-dimensional acceptance mask.

    Thin wrapper around CellGrid + evaluate_solution, returning the same dict
    shape as calculate_metrics_from_custom_cuts().
    """
    from src.optimization_utils import CellGrid, evaluate_solution

    mask_arr = np.asarray(mask, dtype=int)
    grid = CellGrid.from_summary(data, variables)
    kpis = evaluate_solution(mask_arr, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier)

    actual_prod = kpis.get("oa_amt_h0_boo", 0.0) + kpis.get("oa_amt_h0_cut", 0.0)
    opt_prod = kpis.get("oa_amt_h0", 0.0)

    return {
        "actual_prod": actual_prod,
        "actual_risk": kpis.get("b2_ever_h6_boo", 0.0),
        "swap_in_prod": kpis.get("oa_amt_h0_rep", 0.0),
        "swap_in_risk": kpis.get("b2_ever_h6_rep", 0.0),
        "swap_out_prod": kpis.get("oa_amt_h0_cut", 0.0),
        "swap_out_risk": kpis.get("b2_ever_h6_cut", 0.0),
        "opt_prod": opt_prod,
        "opt_risk": kpis.get("b2_ever_h6", 0.0),
        "opt_prod_pct": opt_prod / actual_prod if actual_prod > 0 else 0,
    }


def create_cutoff_explorer_layout(scenario: str, segment: str | None = None) -> html.Div:
    """Create the cutoff explorer tab layout."""
    summary_data, optimal_solution, pareto_solutions, variables, inv_vars, multiplier, cell_ci, optimal_mask = (
        load_cutoff_data(scenario, segment)
    )

    if summary_data is None or len(variables) < 1:
        return dbc.Alert(
            [
                html.H4("Cutoff Explorer Not Available", className="alert-heading"),
                html.P("Summary data not found for this scenario. Run the pipeline first."),
            ],
            color="warning",
        )

    is_1d = len(variables) == 1
    is_nd = len(variables) > 2

    # Get risk range from Pareto solutions
    risk_min, risk_max, risk_current = 0.0, 2.0, 1.0
    pareto_data = []
    if pareto_solutions is not None and not pareto_solutions.empty:
        risk_min = float(pareto_solutions["b2_ever_h6"].min())
        risk_max = float(pareto_solutions["b2_ever_h6"].max())
        if optimal_solution is not None and "b2_ever_h6" in optimal_solution.columns:
            risk_current = float(optimal_solution["b2_ever_h6"].iloc[0])
        else:
            risk_current = (risk_min + risk_max) / 2
        pareto_data = pareto_solutions.to_dict("records")

    # Serialize cell_ci for storage
    cell_ci_data = cell_ci.to_dict("records") if cell_ci is not None and not cell_ci.empty else []
    has_ci = len(cell_ci_data) > 0

    scenario_display = scenario.capitalize()
    segment_display = f" - {segment}" if segment else ""
    if is_1d:
        explorer_mode_label = "1D bin mask"
    elif is_nd:
        explorer_mode_label = f"{len(variables)}D cell mask"
    else:
        explorer_mode_label = "2D cutoff map"

    if is_1d:
        n_bins = len(summary_data[variables[0]].unique())
        acceptance_rule_label = f"Per-bin accept / reject ({variables[0]})"
        acceptance_rule_detail = f"{n_bins} bins, {sum(optimal_mask) if optimal_mask else 0} accepted"
    elif is_nd:
        acceptance_rule_label = "Cell-level accept / reject mask"
        acceptance_rule_detail = f"{len(variables)} variables, {len(optimal_mask) if optimal_mask else 0} cells"
    else:
        acceptance_rule_label = f"{variables[1]} {'≥' if variables[1] in inv_vars else '≤'} cutoff"
        acceptance_rule_detail = f"{len(summary_data[variables[0]].unique())} {variables[0]} bins"
    frontier_label = f"{len(pareto_data)} points" if pareto_data else "Not loaded"
    frontier_detail = f"{risk_min:.2f}% to {risk_max:.2f}% risk"
    uncertainty_label = "Available" if has_ci else "Not loaded"
    uncertainty_detail = "Toggle uncertainty overlay to inspect cell CI width" if has_ci else "No cell_level_ci.csv found"
    if is_1d:
        guide_message = (
            "Use the risk selector to jump to the closest Pareto solution. "
            "Toggle individual bins in the acceptance table to run what-if analysis."
        )
    elif is_nd:
        if len(variables) == 3:
            guide_message = (
                "Use the risk selector to jump to a frontier solution, then click the acceptance "
                "grids to toggle cells. Pin Mode lets you lock cells before re-optimizing."
            )
        else:
            guide_message = (
                "Use the risk selector to jump to a frontier solution, then enable Pin Mode "
                "to lock cells before re-optimizing."
            )
    else:
        guide_message = (
            "Use the risk selector to jump to the closest Pareto solution. "
            "Drag individual cutoffs to run what-if analysis."
        )

    # Create risk slider marks
    risk_marks = _build_risk_marks(risk_min, risk_max)

    # ------------------------------------------------------------------
    # 2-var path: sliders + per-bin cutoffs (unchanged)
    # ------------------------------------------------------------------
    if not is_nd and not is_1d:
        var0_col, var1_col = variables[0], variables[1]

        bins = sorted(summary_data[var0_col].unique())
        var1_bins = sorted(summary_data[var1_col].unique())
        var1_max = int(max(var1_bins))

        optimal_cuts = {}
        if optimal_solution is not None and not optimal_solution.empty:
            opt_row = optimal_solution.iloc[0]
            col_float_map = {}
            for col in optimal_solution.columns:
                try:
                    col_float_map[float(col)] = col
                except (ValueError, TypeError):
                    pass
            for bin_val in bins:
                col_name = col_float_map.get(float(bin_val))
                if col_name is not None:
                    optimal_cuts[bin_val] = int(opt_row[col_name])
                else:
                    optimal_cuts[bin_val] = var1_max

        slider_marks = {-i: str(i) for i in range(1, var1_max + 1, max(1, var1_max // 5))}
        slider_marks[-1] = "1"
        slider_marks[-var1_max] = str(var1_max)

        sliders = []
        for bin_val in bins:
            initial_value = optimal_cuts.get(bin_val, var1_max)
            sliders.append(
                dbc.Col(
                    [
                        html.Label(f"Bin {int(bin_val)}", className="fw-bold text-center d-block"),
                        dcc.Slider(
                            id={"type": "cutoff-slider", "index": int(bin_val)},
                            min=-var1_max,
                            max=-1,
                            step=1,
                            value=-initial_value,
                            marks=slider_marks,
                            vertical=True,
                            verticalHeight=200,
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

        store_data = {
            "scenario": scenario,
            "segment": segment,
            "is_nd": False,
            "variables": variables,
            "var0_col": var0_col,
            "var1_col": var1_col,
            "optimal_cuts": {str(float(k)): v for k, v in optimal_cuts.items()},
            "inv_vars": inv_vars,
            "bins": [float(b) for b in bins],
            "var1_bins": [float(b) for b in var1_bins],
            "multiplier": multiplier,
            "pareto_solutions": pareto_data,
            "cell_ci": cell_ci_data,
        }

        slider_panel = dbc.Col(
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
                                        f"Variable: {var0_col} (bins) \u2192 {var1_col} "
                                        f"({'min accepted' if var1_col in inv_vars else 'max accepted'})",
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
        )

        heatmap_header = (
            f"Cutoff Visualization \u2014 green = accepted "
            f"({var1_col} {'\u2265' if var1_col in inv_vars else '\u2264'} cutoff)"
        )

    # ------------------------------------------------------------------
    # 1D and N>2 path: acceptance DataTable + mask store
    # ------------------------------------------------------------------
    else:
        from src.optimization_utils import CellGrid

        grid = CellGrid.from_summary(summary_data, variables)

        if is_1d:
            editor_help_text = (
                "Toggle bins in the table below to accept or reject them. "
                "The acceptance strip and KPIs update automatically."
            )
        elif len(variables) == 3:
            editor_help_text = (
                f"Click a cell to toggle Accept \u2192 Reject. One grid is shown per {variables[2]} value."
            )
        else:
            editor_help_text = "Click a row in Pin Mode to cycle: unpinned \u2192 accept \u2192 reject \u2192 unpinned"

        grid_children = (
            _build_nd_slice_grid_panel(summary_data, variables, optimal_mask, optimal_mask, {}, multiplier)
            if len(variables) >= 3
            else []
        )

        table_rows = _build_nd_acceptance_table_rows(
            summary_data,
            variables,
            optimal_mask,
            optimal_mask,
            {},
            multiplier,
        )
        nd_table_columns = _build_nd_acceptance_table_columns(variables)

        store_data = {
            "scenario": scenario,
            "segment": segment,
            "is_1d": is_1d,
            "is_nd": is_nd or is_1d,  # 1D also uses mask-based logic
            "variables": variables,
            "optimal_mask": optimal_mask,
            "inv_vars": inv_vars,
            "multiplier": multiplier,
            "pareto_solutions": pareto_data,
            "cell_ci": cell_ci_data,
        }

        slider_panel = dbc.Col(
            [
                dbc.Card(
                    [
                        dbc.CardHeader(
                            [
                                html.Span(
                                    f"Bin Acceptance ({grid.n_cells} bins)" if is_1d
                                    else f"Cell Acceptance ({len(variables)} variables, {grid.n_cells} cells)",
                                    className="fw-bold",
                                ),
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
                                    editor_help_text,
                                    className="text-muted small",
                                ),
                                html.Div(
                                    [
                                        dbc.Badge("Accept", color="success", className="me-2"),
                                        dbc.Badge("Reject", color="danger", className="me-2"),
                                        dbc.Badge("Changed vs reference", color="warning", text_color="dark", className="me-2"),
                                        dbc.Badge("Pinned Accept", color="success", className="me-2", style={"opacity": 0.8}),
                                        dbc.Badge("Pinned Reject", color="danger", style={"opacity": 0.8}),
                                        *([dbc.Badge("Click to toggle", color="primary")] if len(variables) >= 3 else []),
                                    ],
                                    className="mb-3 d-flex flex-wrap gap-2",
                                ),
                                html.Div(
                                    id="nd-acceptance-grid-panel",
                                    children=grid_children,
                                    style={"display": "block" if len(variables) >= 3 else "none"},
                                ),
                                html.Div(
                                    dash_table.DataTable(
                                        id="nd-acceptance-table",
                                        columns=nd_table_columns,
                                        data=table_rows,
                                        page_size=min(16, max(grid.n_cells, 1)),
                                        fixed_rows={"headers": True},
                                        filter_action="native",
                                        sort_action="native",
                                        sort_mode="multi",
                                        hidden_columns=["id"],
                                        row_selectable=False,
                                        style_table={
                                            "overflowX": "auto",
                                            "maxHeight": "460px",
                                            "overflowY": "auto",
                                            "border": "1px solid #dee2e6",
                                            "borderRadius": "0.5rem",
                                        },
                                        style_cell={
                                            "textAlign": "center",
                                            "padding": "7px 10px",
                                            "fontSize": "0.85rem",
                                            "border": "none",
                                        },
                                        style_cell_conditional=[
                                            {"if": {"column_id": variable}, "fontWeight": "600"} for variable in variables
                                        ]
                                        + [
                                            {"if": {"column_id": "booked_prod"}, "textAlign": "right", "minWidth": "110px"},
                                            {"if": {"column_id": "rep_prod"}, "textAlign": "right", "minWidth": "110px"},
                                            {"if": {"column_id": "booked_risk_pct"}, "minWidth": "110px"},
                                        ],
                                        style_header={
                                            "backgroundColor": "#f8f9fa",
                                            "fontWeight": "600",
                                            "borderBottom": "2px solid #dee2e6",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"filter_query": '{current_state} = "Accept"'},
                                                "backgroundColor": "rgba(46,204,113,0.08)",
                                            },
                                            {
                                                "if": {"filter_query": '{current_state} = "Reject"'},
                                                "backgroundColor": "rgba(231,76,60,0.08)",
                                            },
                                            {
                                                "if": {"filter_query": '{decision_delta} = "Changed"'},
                                                "boxShadow": "inset 3px 0 0 #f1c40f",
                                            },
                                            {
                                                "if": {"filter_query": '{pin_state} = "Pinned Accept"'},
                                                "backgroundColor": "rgba(46,204,113,0.18)",
                                                "borderLeft": "4px solid #2ecc71",
                                                "fontWeight": "600",
                                            },
                                            {
                                                "if": {"filter_query": '{pin_state} = "Pinned Reject"'},
                                                "backgroundColor": "rgba(231,76,60,0.18)",
                                                "borderLeft": "4px solid #e74c3c",
                                                "fontWeight": "600",
                                            },
                                            {
                                                "if": {"state": "active"},
                                                "backgroundColor": "rgba(52,152,219,0.10)",
                                                "border": "1px solid #3498db",
                                            },
                                        ],
                                    ),
                                    style={"display": "block"},
                                ),
                            ]
                        ),
                    ],
                    className="mb-3",
                ),
            ],
            md=5,
        )

        if is_1d:
            heatmap_header = f"Acceptance Strip \u2014 {variables[0]} bins (green = accepted)"
        else:
            heatmap_header = (
                f"Acceptance Matrix \u2014 rows = {', '.join(variables[:-1])}, columns = {variables[-1]} "
                "(green = accepted)"
            )

    # ------------------------------------------------------------------
    # Common layout (both paths share the same component IDs)
    # ------------------------------------------------------------------
    return html.Div(
        [
            html.H4(f"Cutoff Explorer ({scenario_display}{segment_display})", className="mb-4"),
            dcc.Store(id="cutoff-data-store", data=store_data),
            dcc.Store(id="pinned-cells-store", data={}),
            dcc.Store(id="current-mask-store", data=optimal_mask),
            dcc.Store(id="pin-feedback-store", data={}),
            dbc.Row(
                [
                    dbc.Col(
                        _build_cutoff_stat_card(
                            "Explorer mode",
                            explorer_mode_label,
                            ", ".join(variables),
                            COLOR_PRIMARY,
                        ),
                        md=6,
                        xl=3,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _build_cutoff_stat_card(
                            "Acceptance rule",
                            acceptance_rule_label,
                            acceptance_rule_detail,
                            COLOR_NEUTRAL,
                        ),
                        md=6,
                        xl=3,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _build_cutoff_stat_card(
                            "Pareto frontier",
                            frontier_label,
                            frontier_detail,
                            COLOR_PRODUCTION,
                        ),
                        md=6,
                        xl=3,
                        className="mb-3",
                    ),
                    dbc.Col(
                        _build_cutoff_stat_card(
                            "Cell uncertainty",
                            uncertainty_label,
                            uncertainty_detail,
                            COLOR_RISK if has_ci else COLOR_NEUTRAL,
                        ),
                        md=6,
                        xl=3,
                        className="mb-3",
                    ),
                ],
                className="g-3",
            ),
            dbc.Alert(guide_message, color="light", className="border mb-3"),
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
                    slider_panel,
                    # Results panel
                    dbc.Col(
                        [
                            dbc.Card(
                                [dbc.CardHeader("Configuration Summary"), dbc.CardBody(id="cutoff-results")], className="mb-3"
                            ),
                            dbc.Card(
                                [
                                    dbc.CardHeader("Pareto Frontier vs Current Configuration"),
                                    dbc.CardBody([dcc.Graph(id="cutoff-chart", style={"height": "300px"})]),
                                ]
                            ),
                        ],
                        md=7,
                    ),
                ]
            ),
            # Heatmap controls row
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Switch(
                                id="show-uncertainty-toggle",
                                label="Show Risk Uncertainty",
                                value=False,
                                className="mb-2",
                            ),
                        ],
                        md=3,
                        style={"display": "block" if has_ci else "none"},
                    ),
                    dbc.Col(
                        [
                            dbc.Switch(id="pin-mode-toggle", label="Pin Mode", value=False, className="mb-2"),
                        ],
                        md=3,
                    ),
                    dbc.Col(
                        [
                            html.Div(
                                [
                                    dbc.Button(
                                        "Re-optimize with Pins",
                                        id="reoptimize-pin-btn",
                                        color="success",
                                        size="sm",
                                        className="me-2 mb-2",
                                    ),
                                    dbc.Button(
                                        "Clear Pins",
                                        id="clear-pins-btn",
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                        className="mb-2",
                                    ),
                                ],
                                className="d-flex flex-wrap align-items-center",
                            ),
                            html.Div(id="pin-status", className="small text-muted mb-2"),
                            html.Div(id="pin-feedback"),
                        ],
                        md=6,
                    ),
                ],
                className="mt-3",
            ),
            # Heatmap
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(heatmap_header),
                                    dbc.CardBody([dcc.Graph(id="cutoff-heatmap", style={"height": "400px"})]),
                                ]
                            )
                        ]
                    )
                ],
                className="mt-3",
            ),
            # Marginal Impact Heatmap
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Marginal Impact \u2014 production change if cell is flipped "
                                        "(green = gain, red = loss)"
                                    ),
                                    dbc.CardBody([dcc.Graph(id="marginal-impact-heatmap", style={"height": "400px"})]),
                                ]
                            )
                        ]
                    )
                ],
                className="mt-3",
            ),
            # Per-variable marginal impact summary (N>2 only)
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Card(
                                [
                                    dbc.CardHeader(
                                        "Per-Variable Marginal Impact \u2014 avg production change per bin value"
                                    ),
                                    dbc.CardBody([dcc.Graph(id="variable-marginal-impact", style={"height": "400px"})]),
                                ]
                            )
                        ]
                    )
                ],
                className="mt-3",
                style={"display": "block" if (is_nd and not is_1d) else "none"},
                id="variable-marginal-row",
            ),
        ]
    )


def create_allocations_tab_content(scenario: str, segment: str | None) -> html.Div:
    """Create the segment allocations tab layout."""
    # The allocations make the most sense in a batch/global context,
    # so we load the main consolidated report.
    output_dir = Path(OUTPUT_BASE)
    report_file = output_dir / "consolidated_risk_production.csv"

    if not report_file.exists():
        return dbc.Alert(
            [
                html.H4("Consolidated Report Not Found", className="alert-heading"),
                html.P("This view requires batch processing results. Please run `run_batch.py`."),
            ],
            color="warning",
        )

    try:
        df = pd.read_csv(report_file)

        # Filter for the selected scenario and main period
        df_filtered = df[(df["scenario"] == scenario) & (df["period"] == "main")].copy()

        if df_filtered.empty:
            return html.Div(f"No consolidated data available for scenario '{scenario}' in 'main' period.")

        # Get only the base segments for the allocations treemap and bar chart
        df_segs = df_filtered[df_filtered["n_segments"] == 1].copy()

        # Extract supersegment and segment from group (format: "supersegment/segment")
        # Handle cases where group is just a segment name
        df_segs["supersegment"] = df_segs["group"].apply(lambda x: x.split("/")[0] if "/" in x else x)
        df_segs["segment"] = df_segs["group"].apply(lambda x: x.split("/")[-1])

        # Plotly Treemap raises an error if the root node (supersegment) is identical
        # to the leaf node (segment) across all rows. We build the path dynamically.
        if (df_segs["supersegment"] == df_segs["segment"]).all():
            tree_path = ["segment"]
        else:
            tree_path = ["supersegment", "segment"]

        if df_segs.empty:
            return html.Div("No segment-level data available for allocations view.")

        # Create Treemap for Optimum Production Allocation
        fig_treemap = px.treemap(
            df_segs,
            path=tree_path,
            values="optimum_production",
            color="optimum_production",
            color_continuous_scale="Viridis",
            title=f"Optimum Production Allocation ({scenario.capitalize()})",
        )
        apply_plotly_style(fig_treemap, height=500)

        # Create Comparative Bar Chart
        fig_bar = go.Figure()

        # Sort by actual production descending for cleaner look
        df_sorted = df_segs.sort_values("actual_production", ascending=False)
        segments = df_sorted["segment"]

        fig_bar.add_trace(
            go.Bar(name="Actual Production", x=segments, y=df_sorted["actual_production"], marker_color=COLOR_PRIMARY)
        )

        fig_bar.add_trace(
            go.Bar(
                name="Optimum Production", x=segments, y=df_sorted["optimum_production"], marker_color=COLOR_PRODUCTION
            )
        )

        fig_bar.update_layout(
            barmode="group",
            title="Production Comparison by Segment",
            xaxis_title="Segment",
            yaxis_title="Production (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        apply_plotly_style(fig_bar, height=400)

        return html.Div(
            [
                html.H4(f"Segment Allocations ({scenario.capitalize()})", className="mb-4"),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Card([dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_treemap), type="default"))]),
                            md=12,
                            className="mb-4",
                        ),
                        dbc.Col(
                            dbc.Card([dbc.CardBody(dcc.Loading(dcc.Graph(figure=fig_bar), type="default"))]), md=12
                        ),
                    ]
                ),
            ]
        )

    except Exception as e:
        logger.error(f"Failed to generate allocations view: {e}")
        return html.Div(f"Error generating allocations view: {str(e)}")


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
                    dbc.Tab(label="Explainability (SHAP)", tab_id="tab-shap", disabled=not has_data),
                    dbc.Tab(label="Segment Allocations", tab_id="tab-allocations", disabled=not has_data),
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
        dbc.Tab(label="Explainability (SHAP)", tab_id="tab-shap", disabled=not has_data),
        dbc.Tab(label="Segment Allocations", tab_id="tab-allocations", disabled=not has_data),
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
                create_kpi_row(paths["mr_csv"], paths["main_csv"], comparison_label="vs Main"),
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

    elif active_tab == "tab-shap":
        dependence_features = get_shap_dependence_features(segment)
        dropdown_options = [{"label": f, "value": f} for f in dependence_features]
        default_feature = dependence_features[0] if dependence_features else None

        return html.Div(
            [
                html.H4(f"Explainability (SHAP) - {scenario_display}{segment_display}", className="mb-4"),
                dbc.Card(
                    [
                        dbc.CardHeader("Global Feature Importance (Summary)"),
                        dbc.CardBody(
                            dcc.Loading(
                                load_iframe(
                                    paths["shap_summary"],
                                    segment,
                                    "SHAP summary plot not available. Ensure models were trained with SHAP generation enabled.",
                                ),
                                type="default",
                            )
                        ),
                    ],
                    className="mb-4",
                ),
                dbc.Card(
                    [
                        dbc.CardHeader("Feature Dependence Plots"),
                        dbc.CardBody(
                            [
                                html.P("Select a feature to view its SHAP dependence plot:"),
                                dcc.Dropdown(
                                    id="shap-feature-dropdown",
                                    options=dropdown_options,
                                    value=default_feature,
                                    clearable=False,
                                    className="mb-3",
                                ),
                                dcc.Loading(html.Div(id="shap-dependence-container"), type="default"),
                            ]
                        ),
                    ]
                )
                if dependence_features
                else html.Div("No dependence plots available.", className="text-muted p-3"),
            ]
        )

    elif active_tab == "tab-allocations":
        return create_allocations_tab_content(scenario, segment)

    return html.Div("Select a tab", className="text-muted")


# Cutoff Explorer callbacks
@app.callback(
    [
        Output("cutoff-results", "children"),
        Output("cutoff-chart", "figure"),
        Output("cutoff-heatmap", "figure"),
        Output("marginal-impact-heatmap", "figure"),
        Output("variable-marginal-impact", "figure"),
    ],
    [
        Input({"type": "cutoff-slider", "index": dash.ALL}, "value"),
        Input("cutoff-data-store", "data"),
        Input("show-uncertainty-toggle", "value"),
        Input("pinned-cells-store", "data"),
        Input("current-mask-store", "data"),
    ],
    prevent_initial_call=False,
)
def update_cutoff_analysis(slider_values, store_data, show_uncertainty, pinned_cells, current_mask_data):
    """Update cutoff analysis when sliders or mask change."""
    empty_fig = go.Figure()
    empty_fig.update_layout(annotations=[dict(text="No data available", x=0.5, y=0.5, showarrow=False)])
    empty_5 = (html.Div("No data available"), empty_fig, empty_fig, empty_fig, empty_fig)

    if not store_data:
        return empty_5

    is_nd = store_data.get("is_nd", False)

    # For 2-var, require sliders; for 1D/N>2, require mask
    if not is_nd and not slider_values:
        return empty_5

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    variables = store_data.get("variables", [])
    inv_vars = store_data.get("inv_vars", [])
    multiplier = store_data.get("multiplier", 7.0)

    # Load summary data
    paths = get_scenario_paths(scenario, segment)
    summary_data = None
    if paths["summary_data"].exists():
        try:
            summary_data = pd.read_csv(paths["summary_data"])
        except Exception as e:
            logger.warning(f"Failed to load summary data from {paths['summary_data']}: {e}")
    if summary_data is None:
        return empty_5

    # ==================================================================
    # 1D and N>2 path (mask-based)
    # ==================================================================
    if is_nd:
        optimal_mask = store_data.get("optimal_mask")
        current_mask = current_mask_data if current_mask_data else optimal_mask

        if not current_mask:
            return empty_5

        # Compute metrics
        current_metrics = calculate_metrics_from_mask(summary_data, current_mask, variables, multiplier)
        optimal_metrics = (
            calculate_metrics_from_mask(summary_data, optimal_mask, variables, multiplier)
            if optimal_mask
            else current_metrics
        )

        # Results display
        results = _build_cutoff_results_content(current_metrics, optimal_metrics, "Reference optimum")

        # Bar chart
        fig_chart = _build_pareto_frontier_figure(
            store_data.get("pareto_solutions", []), current_metrics, optimal_metrics, "Reference optimum"
        )

        # Heatmap / acceptance strip
        from src.optimization_utils import CellGrid

        grid = CellGrid.from_summary(summary_data, variables)
        mask_arr = np.asarray(current_mask, dtype=int)
        optimal_mask_arr = np.asarray(optimal_mask, dtype=int) if optimal_mask else mask_arr
        pins = pinned_cells or {}
        is_1d = store_data.get("is_1d", False)

        # ------ 1D: horizontal bar chart with accept/reject coloring ------
        if is_1d:
            var0 = variables[0]
            bin_values = list(grid.values_per_var[var0])
            bin_labels = [_format_bin_label(v) for v in bin_values]
            cell_prod = []
            cell_risk = []
            bar_colors = []
            for bv in bin_values:
                idx = grid.cell_index[(bv,)]
                row = grid.cell_data.iloc[idx]
                prod = _coerce_float(row.get("oa_amt_h0_boo"))
                rn = _coerce_float(row.get("todu_30ever_h6_boo"))
                rd = _coerce_float(row.get("todu_amt_pile_h6_boo"))
                risk = float(calculate_b2_ever_h6(rn, rd, multiplier=multiplier, as_percentage=True, decimals=2))
                cell_prod.append(prod)
                cell_risk.append(risk)
                bar_colors.append("#2ECC71" if mask_arr[idx] == 1 else "#E74C3C")

            fig_heatmap = go.Figure()
            fig_heatmap.add_trace(
                go.Bar(
                    x=bin_labels,
                    y=cell_prod,
                    marker_color=bar_colors,
                    hovertemplate=(
                        f"{var0}: %{{x}}<br>Production: \u20ac%{{y:,.0f}}"
                        "<br>Risk: %{customdata:.2f}%<extra></extra>"
                    ),
                    customdata=cell_risk,
                )
            )
            fig_heatmap.update_layout(
                xaxis_title=var0,
                yaxis_title="Booked Production (\u20ac)",
                margin=dict(l=60, r=20, t=30, b=60),
                bargap=0.15,
            )
            apply_plotly_style(fig_heatmap, height=350)

            # Marginal impact: simple bar chart per bin
            fig_marginal = go.Figure()
            try:
                from src.sensitivity import compute_cell_marginal_impact

                marginal_df = compute_cell_marginal_impact(
                    grid, mask_arr, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier
                )
                marginal_df = marginal_df.sort_values(var0)
                m_labels = [_format_bin_label(v) for v in marginal_df[var0].values]
                m_colors = [COLOR_PRODUCTION if v >= 0 else COLOR_RISK for v in marginal_df["delta_production"]]
                fig_marginal = go.Figure(
                    go.Bar(
                        x=m_labels,
                        y=marginal_df["delta_production"].values,
                        marker_color=m_colors,
                        hovertemplate=f"{var0}: %{{x}}<br>Delta Production: \u20ac%{{y:,.0f}}<extra></extra>",
                    )
                )
                fig_marginal.update_layout(
                    xaxis_title=var0,
                    yaxis_title="Delta Production (\u20ac)",
                    margin=dict(l=60, r=20, t=30, b=60),
                    bargap=0.15,
                )
                apply_plotly_style(fig_marginal, height=350)
            except Exception as e:
                logger.warning(f"1D marginal impact failed: {e}")
                fig_marginal.update_layout(
                    annotations=[dict(text="Marginal impact not available", x=0.5, y=0.5, showarrow=False)]
                )

            return results, fig_chart, fig_heatmap, fig_marginal, go.Figure()

        # ------ N>2: acceptance matrix (prefix vars x last var) ------
        last_var = variables[-1]
        prefix_vars = variables[:-1]
        last_values = list(grid.values_per_var[last_var])
        last_labels = [_format_bin_label(value) for value in last_values]
        prefix_combos = sorted({combo[:-1] for combo in grid.cell_index})
        prefix_labels = [_format_nd_prefix_label(prefix, prefix_vars) for prefix in prefix_combos]

        z_values = []
        hover_customdata = []
        changed_x, changed_y, changed_keys = [], [], []
        pinned_accept_x, pinned_accept_y, pinned_accept_keys = [], [], []
        pinned_reject_x, pinned_reject_y, pinned_reject_keys = [], [], []

        for prefix_combo, prefix_label in zip(prefix_combos, prefix_labels, strict=False):
            row_values = []
            row_customdata = []
            for last_value, last_label in zip(last_values, last_labels, strict=False):
                combo = prefix_combo + (last_value,)
                idx = grid.cell_index[combo]
                cell_key = ",".join(str(int(value)) for value in combo)
                cell_row = grid.cell_data.iloc[idx]
                booked_prod = _coerce_float(cell_row.get("oa_amt_h0_boo"))
                rep_prod = _coerce_float(cell_row.get("oa_amt_h0_rep"))
                risk_num = _coerce_float(cell_row.get("todu_30ever_h6_boo"))
                risk_den = _coerce_float(cell_row.get("todu_amt_pile_h6_boo"))
                booked_risk_raw = calculate_b2_ever_h6(
                    risk_num,
                    risk_den,
                    multiplier=multiplier,
                    as_percentage=True,
                    decimals=2,
                )
                current_state = "Accept" if mask_arr[idx] == 1 else "Reject"
                reference_state = "Accept" if optimal_mask_arr[idx] == 1 else "Reject"
                decision_delta = "Changed" if current_state != reference_state else "Reference"
                pin_state = _format_pin_state(pins.get(cell_key))
                combo_label = "<br>".join(
                    f"{var}: {_format_bin_label(value)}" for var, value in zip(variables, combo, strict=False)
                )

                row_values.append(int(mask_arr[idx]))
                row_customdata.append(
                    [
                        combo_label,
                        current_state,
                        reference_state,
                        decision_delta,
                        pin_state,
                        booked_prod,
                        rep_prod,
                        _coerce_float(booked_risk_raw),
                        cell_key,
                    ]
                )

                if decision_delta == "Changed":
                    changed_x.append(last_label)
                    changed_y.append(prefix_label)
                    changed_keys.append(cell_key)
                if pin_state == "Pinned Accept":
                    pinned_accept_x.append(last_label)
                    pinned_accept_y.append(prefix_label)
                    pinned_accept_keys.append(cell_key)
                elif pin_state == "Pinned Reject":
                    pinned_reject_x.append(last_label)
                    pinned_reject_y.append(prefix_label)
                    pinned_reject_keys.append(cell_key)

            z_values.append(row_values)
            hover_customdata.append(row_customdata)

        fig_heatmap = go.Figure(
            data=go.Heatmap(
                z=z_values,
                x=last_labels,
                y=prefix_labels,
                zmin=0,
                zmax=1,
                colorscale=[[0, "#ffcccc"], [1, "#ccffcc"]],
                showscale=False,
                xgap=1,
                ygap=1,
                customdata=hover_customdata,
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>Current: %{customdata[1]}"
                    "<br>Reference: %{customdata[2]}"
                    "<br>Delta: %{customdata[3]}"
                    "<br>Pin: %{customdata[4]}"
                    "<br>Booked production: €%{customdata[5]:,.0f}"
                    "<br>Repesca production: €%{customdata[6]:,.0f}"
                    "<br>Booked risk: %{customdata[7]:.2f}%<extra></extra>"
                ),
            )
        )

        if changed_x:
            fig_heatmap.add_trace(
                go.Scatter(
                    x=changed_x,
                    y=changed_y,
                    mode="markers",
                    name="Changed vs reference",
                    customdata=changed_keys,
                    marker=dict(symbol="diamond-open", size=10, color="#f1c40f", line=dict(width=2, color="#c8a500")),
                    hovertemplate="Changed vs reference<extra></extra>",
                )
            )
        if pinned_accept_x:
            fig_heatmap.add_trace(
                go.Scatter(
                    x=pinned_accept_x,
                    y=pinned_accept_y,
                    mode="markers",
                    name="Pinned accept",
                    customdata=pinned_accept_keys,
                    marker=dict(symbol="star", size=12, color="#0b7a30", line=dict(width=1, color="white")),
                    hovertemplate="Pinned accept<extra></extra>",
                )
            )
        if pinned_reject_x:
            fig_heatmap.add_trace(
                go.Scatter(
                    x=pinned_reject_x,
                    y=pinned_reject_y,
                    mode="markers",
                    name="Pinned reject",
                    customdata=pinned_reject_keys,
                    marker=dict(symbol="x", size=11, color="#8b0000", line=dict(width=2, color="#8b0000")),
                    hovertemplate="Pinned reject<extra></extra>",
                )
            )

        fig_heatmap.update_layout(
            xaxis_title=last_var,
            yaxis_title=" | ".join(prefix_vars) if prefix_vars else "Cell group",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=80, r=20, t=30, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            clickmode="event+select",
        )
        apply_plotly_style(fig_heatmap, height=min(900, max(380, len(prefix_labels) * 28)))

        # Marginal impact: horizontal bar chart sorted by magnitude
        fig_marginal = go.Figure()
        try:
            from src.sensitivity import compute_cell_marginal_impact

            marginal_df = compute_cell_marginal_impact(
                grid, mask_arr, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier
            )
            marginal_df = marginal_df.sort_values("delta_production", key=abs, ascending=True)

            # Build cell labels
            labels = [",".join(str(int(row[v])) for v in variables) for _, row in marginal_df[variables].iterrows()]
            colors = [COLOR_PRODUCTION if v >= 0 else COLOR_RISK for v in marginal_df["delta_production"]]

            fig_marginal = go.Figure(
                go.Bar(
                    x=marginal_df["delta_production"].values,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="Cell %{y}<br>Delta Production: \u20ac%{x:,.0f}<extra></extra>",
                )
            )
            fig_marginal.update_layout(
                xaxis_title="Delta Production (\u20ac)",
                yaxis_title="Cell",
                margin=dict(l=80, r=20, t=30, b=40),
            )
            apply_plotly_style(fig_marginal, height=max(380, len(labels) * 20))
        except Exception as e:
            logger.warning(f"Marginal impact computation failed: {e}")
            fig_marginal.update_layout(
                annotations=[dict(text="Marginal impact not available", x=0.5, y=0.5, showarrow=False)]
            )

        # Per-variable marginal impact summary (N>2 only)
        fig_var_marginal = go.Figure()
        try:
            from plotly.subplots import make_subplots

            from src.sensitivity import compute_cell_marginal_impact as _compute_marginal

            marginal_df = _compute_marginal(
                grid, mask_arr, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier
            )
            n_vars = len(variables)
            fig_var_marginal = make_subplots(
                rows=1,
                cols=n_vars,
                subplot_titles=[f"{v}" for v in variables],
                shared_yaxes=False,
                horizontal_spacing=0.06,
            )
            for vi, var in enumerate(variables, 1):
                grouped = marginal_df.groupby(var)["delta_production"].mean().sort_index()
                var_labels = [_format_bin_label(v) for v in grouped.index]
                var_colors = [COLOR_PRODUCTION if v >= 0 else COLOR_RISK for v in grouped.values]
                fig_var_marginal.add_trace(
                    go.Bar(
                        x=var_labels,
                        y=grouped.values,
                        marker_color=var_colors,
                        hovertemplate=f"{var}: %{{x}}<br>Avg \u0394 Prod: \u20ac%{{y:,.0f}}<extra></extra>",
                        showlegend=False,
                    ),
                    row=1,
                    col=vi,
                )
                fig_var_marginal.update_xaxes(title_text=var, row=1, col=vi)
            fig_var_marginal.update_yaxes(title_text="\u0394 Production (\u20ac)", row=1, col=1)
            fig_var_marginal.update_layout(margin=dict(l=60, r=20, t=40, b=60))
            apply_plotly_style(fig_var_marginal, height=350)
        except Exception as e:
            logger.warning(f"Per-variable marginal impact failed: {e}")
            fig_var_marginal.update_layout(
                annotations=[dict(text="Per-variable marginal not available", x=0.5, y=0.5, showarrow=False)]
            )

        return results, fig_chart, fig_heatmap, fig_marginal, fig_var_marginal

    # ==================================================================
    # 2-var path (unchanged)
    # ==================================================================
    var0_col = store_data["var0_col"]
    var1_col = store_data["var1_col"]
    optimal_cuts = {float(k): v for k, v in store_data["optimal_cuts"].items()}
    bins = store_data["bins"]

    # Build current cut map from sliders (keys as float to match data column types).
    # Slider values are negated (for reversed direction), so negate back.
    current_cuts = {float(bins[i]): -slider_values[i] for i in range(len(bins))}

    # Calculate metrics for current and optimal
    current_metrics = calculate_metrics_from_custom_cuts(
        summary_data, current_cuts, var0_col, var1_col, inv_vars, multiplier
    )
    optimal_metrics = calculate_metrics_from_custom_cuts(
        summary_data, optimal_cuts, var0_col, var1_col, inv_vars, multiplier
    )

    # Create results display
    # Risk values are already in percentage form (e.g. 1.5 = 1.5%)
    results = _build_cutoff_results_content(current_metrics, optimal_metrics, "Reference optimum")

    # Create comparison chart — risk is already in %, production ratio needs *100
    fig_chart = _build_pareto_frontier_figure(
        store_data.get("pareto_solutions", []), current_metrics, optimal_metrics, "Reference optimum"
    )

    # Create heatmap showing which cells pass/fail
    df = summary_data.copy()
    # Normalize keys to float for .map()
    cuts_float = {float(k): v for k, v in current_cuts.items()}
    df["cut_limit"] = df[var0_col].astype(float).map(cuts_float)

    # Respect the optimization direction
    if var1_col in inv_vars:
        df["passes"] = (df[var1_col] >= df["cut_limit"]).astype(int)
    else:
        df["passes"] = (df[var1_col] <= df["cut_limit"]).astype(int)

    # Pivot for heatmap — ascending y-axis so origin is (1,1) at bottom-left
    pivot = df.pivot_table(index=var1_col, columns=var0_col, values="passes", aggfunc="first")
    pivot = pivot.sort_index(ascending=True)

    cell_lookup = {(float(row[var1_col]), float(row[var0_col])): row for _, row in df.iterrows()}
    heatmap_customdata = []
    for y_val in pivot.index:
        row_customdata = []
        for x_val in pivot.columns:
            cell_row = cell_lookup.get((float(y_val), float(x_val)))
            booked_prod = _coerce_float(cell_row.get("oa_amt_h0_boo")) if cell_row is not None else 0.0
            rep_prod = _coerce_float(cell_row.get("oa_amt_h0_rep")) if cell_row is not None else 0.0
            risk_num = _coerce_float(cell_row.get("todu_30ever_h6_boo")) if cell_row is not None else 0.0
            risk_den = _coerce_float(cell_row.get("todu_amt_pile_h6_boo")) if cell_row is not None else 0.0
            booked_risk = (risk_num / risk_den * multiplier) if risk_den > 0 else 0.0
            pin_state = "Unpinned"
            if pinned_cells:
                pin_value = pinned_cells.get(f"{int(x_val)},{int(y_val)}")
                if pin_value == 1:
                    pin_state = "Pinned accept"
                elif pin_value == 0:
                    pin_state = "Pinned reject"
            row_customdata.append(
                [
                    "Accepted" if pivot.loc[y_val, x_val] == 1 else "Rejected",
                    _coerce_float(cuts_float.get(float(x_val)), np.nan),
                    booked_prod,
                    rep_prod,
                    booked_risk,
                    pin_state,
                    f"{int(x_val)},{int(y_val)}",
                ]
            )
        heatmap_customdata.append(row_customdata)

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(int(c)) for c in pivot.columns],
            y=[str(int(r)) for r in pivot.index],
            colorscale=[[0, "#ffcccc"], [1, "#ccffcc"]],
            showscale=False,
            xgap=1,
            ygap=1,
            hovertemplate=(
                f"{var0_col}: %{{x}}<br>{var1_col}: %{{y}}"
                "<br>Status: %{customdata[0]}"
                "<br>Cutoff: %{customdata[1]:.0f}"
                "<br>Booked production: €%{customdata[2]:,.0f}"
                "<br>Repesca production: €%{customdata[3]:,.0f}"
                "<br>Booked risk: %{customdata[4]:.2f}%"
                "<br>%{customdata[5]}<extra></extra>"
            ),
            customdata=heatmap_customdata,
        )
    )

    # Add cutoff line — y-axis is ascending (1 at bottom, max at top)
    y_vals_asc = list(pivot.index)  # ascending order
    for i, bin_val in enumerate(pivot.columns):
        cut_val = cuts_float.get(float(bin_val))
        if cut_val is None:
            continue
        if var1_col in inv_vars:
            # passes when val >= cut_val → line below cut_val (between rejected below and accepted above)
            pos = next((j for j, v in enumerate(y_vals_asc) if v >= cut_val), len(y_vals_asc))
        else:
            # passes when val <= cut_val → line above cut_val (between accepted below and rejected above)
            pos = next((j for j, v in enumerate(y_vals_asc) if v > cut_val), len(y_vals_asc))
        y_line = pos - 0.5
        fig_heatmap.add_shape(
            type="line",
            x0=i - 0.5,
            x1=i + 0.5,
            y0=y_line,
            y1=y_line,
            line=dict(color="red", width=3),
        )

    # Add uncertainty annotations if toggle is on and CI data is available
    cell_ci_data = store_data.get("cell_ci", [])
    if show_uncertainty and cell_ci_data:
        ci_df = pd.DataFrame(cell_ci_data)
        if var0_col in ci_df.columns and var1_col in ci_df.columns and "ci_upper" in ci_df.columns:
            ci_df["ci_width"] = ci_df["ci_upper"] - ci_df["ci_lower"]
            for _, ci_row in ci_df.iterrows():
                v0 = ci_row[var0_col]
                v1 = ci_row[var1_col]
                width = ci_row["ci_width"]
                if pd.notna(width):
                    x_idx = list(pivot.columns).index(v0) if v0 in list(pivot.columns) else None
                    y_idx = list(pivot.index).index(v1) if v1 in list(pivot.index) else None
                    if x_idx is not None and y_idx is not None:
                        fig_heatmap.add_annotation(
                            x=str(int(v0)),
                            y=str(int(v1)),
                            text=f"\u00b1{width:.2f}",
                            showarrow=False,
                            font=dict(size=9, color="#333"),
                        )

    # Annotate pinned cells on heatmap
    if pinned_cells:
        for cell_key, pin_val in pinned_cells.items():
            parts = cell_key.split(",")
            if len(parts) == 2:
                v0_str, v1_str = parts
                marker = "\u2605" if pin_val == 1 else "\u2717"  # star for accept, X for reject
                color = "#006400" if pin_val == 1 else "#8B0000"
                fig_heatmap.add_annotation(
                    x=v0_str.strip(),
                    y=v1_str.strip(),
                    text=marker,
                    showarrow=False,
                    font=dict(size=16, color=color),
                    yshift=8,
                )

    fig_heatmap.update_layout(
        xaxis_title=var0_col,
        yaxis_title=var1_col,
        margin=dict(l=60, r=20, t=30, b=60),
    )
    apply_plotly_style(fig_heatmap, height=380)

    # Marginal impact heatmap (analytical O(N))
    fig_marginal = go.Figure()
    try:
        from src.optimization_utils import CellGrid
        from src.sensitivity import compute_cell_marginal_impact

        grid = CellGrid.from_summary(summary_data, [var0_col, var1_col])

        # Build mask from current cuts
        mask = np.zeros(grid.n_cells, dtype=int)
        for combo, idx in grid.cell_index.items():
            v0_val, v1_val = combo
            cut = cuts_float.get(float(v0_val))
            if cut is not None:
                if var1_col in inv_vars:
                    mask[idx] = 1 if v1_val >= cut else 0
                else:
                    mask[idx] = 1 if v1_val <= cut else 0

        marginal_df = compute_cell_marginal_impact(
            grid, mask, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier
        )

        # Pivot for heatmap
        marginal_pivot = marginal_df.pivot_table(
            index=var1_col, columns=var0_col, values="delta_production", aggfunc="first"
        )
        marginal_pivot = marginal_pivot.sort_index(ascending=True)

        # Compute symmetric color range for diverging scale
        abs_max = max(abs(marginal_pivot.values.min()), abs(marginal_pivot.values.max()), 1)

        fig_marginal = go.Figure(
            data=go.Heatmap(
                z=marginal_pivot.values,
                x=[str(int(c)) for c in marginal_pivot.columns],
                y=[str(int(r)) for r in marginal_pivot.index],
                colorscale=[[0, "#d32f2f"], [0.5, "#ffffff"], [1, "#388e3c"]],
                zmin=-abs_max,
                zmax=abs_max,
                showscale=True,
                colorbar=dict(title="EUR"),
                hovertemplate=(
                    f"{var0_col}: %{{x}}<br>{var1_col}: %{{y}}<br>Delta Production: \u20ac%{{z:,.0f}}<extra></extra>"
                ),
            )
        )
        fig_marginal.update_layout(
            xaxis_title=var0_col,
            yaxis_title=var1_col,
            margin=dict(l=60, r=20, t=30, b=60),
        )
        apply_plotly_style(fig_marginal, height=380)
    except Exception as e:
        logger.warning(f"Marginal impact computation failed: {e}")
        fig_marginal.update_layout(
            annotations=[dict(text="Marginal impact not available", x=0.5, y=0.5, showarrow=False)]
        )

    return results, fig_chart, fig_heatmap, fig_marginal, go.Figure()


@app.callback(
    Output("nd-acceptance-table", "data"),
    [Input("current-mask-store", "data"), Input("pinned-cells-store", "data"), Input("cutoff-data-store", "data")],
    prevent_initial_call=False,
)
def update_nd_acceptance_table(current_mask, pinned_cells, store_data):
    if not store_data or not store_data.get("is_nd", False):
        return dash.no_update

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    variables = store_data.get("variables", [])
    multiplier = store_data.get("multiplier", 7.0)
    optimal_mask = store_data.get("optimal_mask")
    paths = get_scenario_paths(scenario, segment)
    if not paths["summary_data"].exists():
        return []

    summary_data = pd.read_csv(paths["summary_data"])
    active_mask = current_mask if current_mask is not None else optimal_mask
    return _build_nd_acceptance_table_rows(
        summary_data,
        variables,
        active_mask,
        optimal_mask,
        pinned_cells,
        multiplier,
    )


@app.callback(
    Output("nd-acceptance-grid-panel", "children"),
    [Input("current-mask-store", "data"), Input("pinned-cells-store", "data"), Input("cutoff-data-store", "data")],
    prevent_initial_call=False,
)
def render_nd_acceptance_grid_panel(current_mask, pinned_cells, store_data):
    if not store_data or not store_data.get("is_nd", False):
        return dash.no_update

    variables = store_data.get("variables", [])
    if len(variables) < 3:
        return []

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    multiplier = store_data.get("multiplier", 7.0)
    optimal_mask = store_data.get("optimal_mask")
    paths = get_scenario_paths(scenario, segment)
    if not paths["summary_data"].exists():
        return []

    summary_data = pd.read_csv(paths["summary_data"])
    active_mask = current_mask if current_mask is not None else optimal_mask
    return _build_nd_slice_grid_panel(
        summary_data,
        variables,
        active_mask,
        optimal_mask,
        pinned_cells,
        multiplier,
    )


@app.callback(
    [
        Output("current-mask-store", "data", allow_duplicate=True),
        Output("pinned-cells-store", "data", allow_duplicate=True),
    ],
    [Input({"type": "nd-slice-grid", "index": dash.ALL}, "clickData")],
    [
        State("cutoff-data-store", "data"),
        State("current-mask-store", "data"),
        State("pinned-cells-store", "data"),
        State({"type": "nd-slice-grid", "index": dash.ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def toggle_nd_slice_grid_cell(click_data_list, store_data, current_mask, pinned_cells, grid_ids):
    if not store_data or not store_data.get("is_nd", False):
        return dash.no_update, dash.no_update

    variables = store_data.get("variables", [])
    if len(variables) < 3:
        return dash.no_update, dash.no_update

    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    triggered_id = getattr(ctx, "triggered_id", None)
    click_data = None
    if isinstance(triggered_id, dict) and grid_ids and click_data_list:
        for grid_id, candidate in zip(grid_ids, click_data_list, strict=False):
            if grid_id == triggered_id:
                click_data = candidate
                break
    if click_data is None and click_data_list:
        click_data = next(
            (
                candidate
                for candidate in reversed(click_data_list)
                if isinstance(candidate, dict) and isinstance(candidate.get("points"), list) and candidate.get("points")
            ),
            None,
        )
    if not click_data or "points" not in click_data:
        return dash.no_update, dash.no_update

    point = click_data["points"][0]
    customdata = point.get("customdata")
    cell_key = None
    if isinstance(customdata, str) and "," in customdata:
        cell_key = customdata
    elif isinstance(customdata, (list, tuple)):
        for candidate in reversed(customdata):
            if isinstance(candidate, str) and "," in candidate:
                cell_key = candidate
                break
    if not cell_key:
        return dash.no_update, dash.no_update

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    paths = get_scenario_paths(scenario, segment)
    if not paths["summary_data"].exists():
        return dash.no_update, dash.no_update

    from src.optimization_utils import CellGrid

    summary_data = pd.read_csv(paths["summary_data"])
    grid = CellGrid.from_summary(summary_data, variables)
    base_mask = current_mask if current_mask is not None else store_data.get("optimal_mask")
    if base_mask is None:
        return dash.no_update, dash.no_update

    mask_arr = np.asarray(base_mask, dtype=int).copy()
    combo = tuple(int(part.strip()) for part in cell_key.split(","))
    idx = grid.cell_index.get(combo)
    if idx is None:
        return dash.no_update, dash.no_update

    new_value = 0 if mask_arr[idx] == 1 else 1
    mask_arr[idx] = new_value
    pinned_cells_updated = dict(pinned_cells or {})
    pinned_cells_updated[cell_key] = int(new_value)
    return mask_arr.tolist(), pinned_cells_updated


@app.callback(
    Output("pin-status", "children"),
    [Input("pin-mode-toggle", "value"), Input("pinned-cells-store", "data")],
    prevent_initial_call=False,
)
def render_pin_status(pin_mode, pinned_cells):
    return _build_pin_status_children(bool(pin_mode), pinned_cells)


@app.callback(Output("pin-feedback", "children"), [Input("pin-feedback-store", "data")], prevent_initial_call=False)
def render_pin_feedback(feedback):
    return _build_pin_feedback_alert(feedback)


@app.callback(
    [Output("reoptimize-pin-btn", "disabled"), Output("clear-pins-btn", "disabled")],
    [Input("pinned-cells-store", "data")],
    prevent_initial_call=False,
)
def update_pin_action_buttons(pinned_cells):
    has_pins = bool(pinned_cells)
    return not has_pins, not has_pins


@app.callback(
    [
        Output("pinned-cells-store", "data", allow_duplicate=True),
        Output("pin-feedback-store", "data", allow_duplicate=True),
    ],
    [Input("clear-pins-btn", "n_clicks"), Input("reset-cutoffs-btn", "n_clicks")],
    [State("pinned-cells-store", "data")],
    prevent_initial_call=True,
)
def clear_pins(clear_clicks, reset_clicks, pinned_cells):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    n_pins, _, _ = _pin_counts(pinned_cells)
    if trigger_id == "clear-pins-btn":
        if not n_pins:
            return dash.no_update, {"tone": "info", "message": "No pins to clear."}
        return {}, {"tone": "secondary", "message": f"Cleared {n_pins} pinned cells."}

    if trigger_id == "reset-cutoffs-btn":
        if n_pins:
            return {}, {"tone": "secondary", "message": f"Reset to the reference optimum and cleared {n_pins} pinned cells."}
        return {}, {"tone": "secondary", "message": "Reset to the reference optimum."}

    return dash.no_update, dash.no_update


# Reset button callback
@app.callback(
    [
        Output({"type": "cutoff-slider", "index": dash.ALL}, "value"),
        Output("current-mask-store", "data"),
    ],
    [Input("reset-cutoffs-btn", "n_clicks"), Input("apply-risk-btn", "n_clicks")],
    [
        State("cutoff-data-store", "data"),
        State("risk-level-slider", "value"),
        State({"type": "cutoff-slider", "index": dash.ALL}, "value"),
        State("current-mask-store", "data"),
    ],
    prevent_initial_call=True,
)
def update_cutoffs_from_buttons(reset_clicks, apply_clicks, store_data, risk_value, current_values, current_mask):
    """Update sliders/mask from reset button or risk level selection."""
    if not store_data:
        return current_values, dash.no_update

    ctx = callback_context
    if not ctx.triggered:
        return current_values, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    is_nd = store_data.get("is_nd", False)

    # ------------------------------------------------------------------
    # N>2 path: update mask store only, sliders are empty
    # ------------------------------------------------------------------
    if is_nd:
        if trigger_id == "reset-cutoffs-btn":
            return current_values, store_data.get("optimal_mask")

        if trigger_id == "apply-risk-btn":
            pareto_solutions = store_data.get("pareto_solutions", [])
            if not pareto_solutions or risk_value is None:
                return current_values, dash.no_update

            closest_solution = min(pareto_solutions, key=lambda s: abs(s.get("b2_ever_h6", 0) - risk_value))
            mask_str = closest_solution.get("acceptance_mask")
            if mask_str:
                from src.optimization_utils import decode_mask

                return current_values, decode_mask(str(mask_str)).tolist()
            return current_values, dash.no_update

        return current_values, dash.no_update

    # ------------------------------------------------------------------
    # 2-var path (unchanged, mask store untouched)
    # ------------------------------------------------------------------
    bins = store_data["bins"]

    if trigger_id == "reset-cutoffs-btn":
        optimal_cuts = {float(k): v for k, v in store_data["optimal_cuts"].items()}
        return [-optimal_cuts.get(float(bin_val), 9) for bin_val in bins], dash.no_update

    elif trigger_id == "apply-risk-btn":
        pareto_solutions = store_data.get("pareto_solutions", [])
        if not pareto_solutions or risk_value is None:
            return current_values, dash.no_update

        closest_solution = None
        min_diff = float("inf")
        for sol in pareto_solutions:
            diff = abs(sol.get("b2_ever_h6", 0) - risk_value)
            if diff < min_diff:
                min_diff = diff
                closest_solution = sol

        if closest_solution is None:
            return current_values, dash.no_update

        new_values = []
        for bin_val in bins:
            cutoff = None
            for key in [str(float(bin_val)), float(bin_val), str(int(bin_val)), int(bin_val), bin_val]:
                val = closest_solution.get(key)
                if val is not None:
                    cutoff = val
                    break
            if cutoff is not None:
                new_values.append(-int(cutoff))
            else:
                idx = bins.index(bin_val)
                new_values.append(current_values[idx] if idx < len(current_values) else -9)

        return new_values, dash.no_update

    return current_values, dash.no_update


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
    risk_gap = actual_risk - risk_value

    return html.Div(
        [
            dbc.Badge(f"Target {risk_value:.2f}%", color="light", text_color="dark", className="me-2"),
            dbc.Badge(
                f"Closest Pareto {actual_risk:.2f}% ({risk_gap:+.2f}pp)",
                color="primary",
                className="me-2",
            ),
            dbc.Badge(f"Production {_format_currency(production, compact=True)}", color="secondary", className="me-2"),
            html.Div(f"{len(pareto_solutions)} Pareto-optimal solutions available", className="text-muted mt-2"),
        ]
    )


# --- Pin Mode Callbacks ---


@app.callback(
    [
        Output("pinned-cells-store", "data"),
        Output("current-mask-store", "data", allow_duplicate=True),
    ],
    [Input("cutoff-heatmap", "clickData")],
    [
        State("pin-mode-toggle", "value"),
        State("pinned-cells-store", "data"),
        State("cutoff-data-store", "data"),
        State("current-mask-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_cell_pin(click_data, pin_mode, pinned_cells, store_data, current_mask):
    """On heatmap click: pin mode cycles pin state; otherwise toggle accept/reject for N>2."""
    if not click_data or not store_data:
        return dash.no_update, dash.no_update

    is_nd = store_data.get("is_nd", False)

    point = click_data["points"][0]
    customdata = point.get("customdata")
    cell_key = None
    if isinstance(customdata, str) and "," in customdata:
        cell_key = customdata
    elif isinstance(customdata, (list, tuple)):
        for candidate in reversed(customdata):
            if isinstance(candidate, str) and "," in candidate:
                cell_key = candidate
                break

    if not cell_key:
        v0_str = str(point.get("x", ""))
        v1_str = str(point.get("y", ""))
        cell_key = f"{v0_str},{v1_str}"

    pinned_cells = dict(pinned_cells or {})

    # Pin mode: cycle unpinned → accept → reject → unpinned
    if pin_mode:
        current = pinned_cells.get(cell_key)
        if current is None:
            pinned_cells[cell_key] = 1  # pin as accept
        elif current == 1:
            pinned_cells[cell_key] = 0  # pin as reject
        else:
            del pinned_cells[cell_key]  # unpin
        return pinned_cells, dash.no_update

    # Direct toggle for N>2 (without pin mode)
    if is_nd:
        variables = store_data.get("variables", [])
        optimal_mask = store_data.get("optimal_mask")
        base_mask = current_mask if current_mask is not None else optimal_mask
        if not base_mask:
            return dash.no_update, dash.no_update

        from src.optimization_utils import CellGrid

        scenario = store_data["scenario"]
        segment = store_data.get("segment")
        paths = get_scenario_paths(scenario, segment)
        if not paths["summary_data"].exists():
            return dash.no_update, dash.no_update

        summary_data = pd.read_csv(paths["summary_data"])
        grid = CellGrid.from_summary(summary_data, variables)

        mask_arr = np.asarray(base_mask, dtype=int).copy()
        combo = tuple(int(part.strip()) for part in cell_key.split(","))
        idx = grid.cell_index.get(combo)
        if idx is None:
            return dash.no_update, dash.no_update

        new_value = 0 if mask_arr[idx] == 1 else 1
        mask_arr[idx] = new_value
        pinned_cells[cell_key] = int(new_value)
        return pinned_cells, mask_arr.tolist()

    # 2-var case without pin mode: no action (controlled by sliders)
    return dash.no_update, dash.no_update


@app.callback(
    [
        Output({"type": "cutoff-slider", "index": dash.ALL}, "value", allow_duplicate=True),
        Output("current-mask-store", "data", allow_duplicate=True),
        Output("pin-feedback-store", "data", allow_duplicate=True),
    ],
    [Input("reoptimize-pin-btn", "n_clicks")],
    [
        State("cutoff-data-store", "data"),
        State("pinned-cells-store", "data"),
        State("risk-level-slider", "value"),
        State({"type": "cutoff-slider", "index": dash.ALL}, "value"),
        State("current-mask-store", "data"),
    ],
    prevent_initial_call=True,
)
def reoptimize_with_pins(n_clicks, store_data, pinned_cells, risk_value, current_values, current_mask):
    """Re-optimize MILP with pinned cell constraints and update sliders/mask."""
    if not n_clicks or not store_data:
        return current_values, dash.no_update, dash.no_update
    if not pinned_cells:
        return current_values, dash.no_update, {"tone": "info", "message": "Pin at least one cell before re-optimizing."}

    try:
        from src.optimization_utils import CellGrid, mask_to_cutoffs, solve_with_fixed_cells

        scenario = store_data["scenario"]
        segment = store_data.get("segment")
        variables = store_data.get("variables", [])
        inv_vars = store_data.get("inv_vars", [])
        multiplier = store_data.get("multiplier", 7.0)
        is_nd = store_data.get("is_nd", False)

        # Load summary data
        paths = get_scenario_paths(scenario, segment)
        if not paths["summary_data"].exists():
            return current_values, dash.no_update, {"tone": "danger", "message": "Summary data not found for re-optimization."}
        summary_data = pd.read_csv(paths["summary_data"])

        grid = CellGrid.from_summary(summary_data, variables)
        n_pins, n_accept, n_reject = _pin_counts(pinned_cells)

        # Build fixed_accepts and fixed_rejects from pinned cells (N-tuple keys)
        fixed_accepts = []
        fixed_rejects = []
        for cell_key, pin_val in pinned_cells.items():
            parts = cell_key.split(",")
            combo = tuple(int(p.strip()) for p in parts)
            if len(combo) == len(variables):
                if pin_val == 1:
                    fixed_accepts.append(combo)
                else:
                    fixed_rejects.append(combo)

        target_risk = risk_value if risk_value else 1.0
        mask, kpis = solve_with_fixed_cells(
            grid,
            target_risk,
            inv_vars,
            multiplier,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            fixed_accepts=fixed_accepts,
            fixed_rejects=fixed_rejects,
        )

        if mask is None:
            logger.warning("Re-optimization with pins: infeasible")
            return current_values, dash.no_update, {
                "tone": "warning",
                "message": (
                    f"No feasible solution found for {n_pins} pins ({n_accept} accept, {n_reject} reject) "
                    f"at target risk {target_risk:.2f}%. Try fewer accept pins or a higher risk target."
                ),
            }

        success_message = (
            f"Applied {n_pins} pins ({n_accept} accept, {n_reject} reject) at target {target_risk:.2f}% — "
            f"solution risk {kpis.get('b2_ever_h6', 0):.2f}% and production {_format_currency(kpis.get('oa_amt_h0', 0), compact=True)}."
        )

        # N>2: return mask directly
        if is_nd:
            return current_values, mask.tolist(), {"tone": "success", "message": success_message}

        # 2-var: convert mask → cutoffs → slider values
        var0_col = store_data["var0_col"]
        bins = store_data["bins"]
        cutoffs = mask_to_cutoffs(mask, grid, inv_vars)
        if var0_col not in cutoffs:
            return current_values, dash.no_update, {
                "tone": "warning",
                "message": "Pinned solution was feasible, but cutoffs could not be reconstructed from the mask.",
            }

        new_values = []
        for bin_val in bins:
            cut = cutoffs[var0_col].get(float(bin_val))
            if cut is not None and not (cut == float("inf") or cut == float("-inf")):
                new_values.append(-int(cut))
            else:
                idx = bins.index(bin_val)
                new_values.append(current_values[idx] if idx < len(current_values) else -9)

        return new_values, dash.no_update, {"tone": "success", "message": success_message}

    except Exception as e:
        logger.error(f"Re-optimize with pins failed: {e}")
        return current_values, dash.no_update, {
            "tone": "danger",
            "message": "Re-optimization with pins failed. Check the logs and try a simpler set of pins.",
        }


# --- N>2 Cell Pin Callback ---


@app.callback(
    [
        Output("pinned-cells-store", "data", allow_duplicate=True),
        Output("current-mask-store", "data", allow_duplicate=True),
    ],
    [Input("nd-acceptance-table", "active_cell")],
    [
        State("nd-acceptance-table", "derived_viewport_data"),
        State("pin-mode-toggle", "value"),
        State("pinned-cells-store", "data"),
        State("cutoff-data-store", "data"),
        State("current-mask-store", "data"),
    ],
    prevent_initial_call=True,
)
def handle_nd_cell_pin(active_cell, table_rows, pin_mode, pinned_cells, store_data, current_mask):
    """On N>2 DataTable cell click: pin mode cycles pin state; otherwise toggle accept/reject."""
    if not active_cell or not store_data or not store_data.get("is_nd", False):
        return dash.no_update, dash.no_update

    pinned_cells = dict(pinned_cells or {})

    cell_key = active_cell.get("row_id")
    if cell_key is None and table_rows is not None:
        row_index = int(active_cell.get("row", -1))
        if 0 <= row_index < len(table_rows):
            cell_key = table_rows[row_index].get("id")
    if not cell_key:
        return dash.no_update, dash.no_update

    # Pin mode: cycle unpinned → accept → reject → unpinned
    if pin_mode:
        current = pinned_cells.get(cell_key)
        if current is None:
            pinned_cells[cell_key] = 1  # pin as accept
        elif current == 1:
            pinned_cells[cell_key] = 0  # pin as reject
        else:
            del pinned_cells[cell_key]  # unpin
        return pinned_cells, dash.no_update

    # Direct toggle: flip accept ↔ reject in the mask
    variables = store_data.get("variables", [])
    optimal_mask = store_data.get("optimal_mask")
    base_mask = current_mask if current_mask is not None else optimal_mask
    if not base_mask:
        return dash.no_update, dash.no_update

    from src.optimization_utils import CellGrid

    scenario = store_data["scenario"]
    segment = store_data.get("segment")
    paths = get_scenario_paths(scenario, segment)
    if not paths["summary_data"].exists():
        return dash.no_update, dash.no_update

    summary_data = pd.read_csv(paths["summary_data"])
    grid = CellGrid.from_summary(summary_data, variables)

    mask_arr = np.asarray(base_mask, dtype=int).copy()
    combo = tuple(int(part.strip()) for part in cell_key.split(","))
    idx = grid.cell_index.get(combo)
    if idx is None:
        return dash.no_update, dash.no_update

    new_value = 0 if mask_arr[idx] == 1 else 1
    mask_arr[idx] = new_value
    pinned_cells[cell_key] = int(new_value)
    return pinned_cells, mask_arr.tolist()


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


@app.callback(
    Output("shap-dependence-container", "children"),
    [Input("shap-feature-dropdown", "value")],
    [State("segment-selector", "value")],
    prevent_initial_call=False,
)
def update_shap_dependence(feature: str | None, segment: str | None) -> html.Div:
    """Render the selected SHAP dependence plot."""
    if not feature:
        return html.Div("Select a feature to view its dependence plot.", className="text-muted")

    static_prefix = get_shap_static_prefix(segment)
    viz_path = f"{static_prefix}/shap_dependence_{feature}.html"

    # Set segment=None in load_iframe so file_exists_for_viz routes through the _supersegment path correctly.
    # Otherwise, file_exists_for_viz might try to check the raw segment images dir, though our recent fix there covers it.
    return load_iframe(
        viz_path,
        segment,
        f"SHAP dependence plot for '{feature}' not available.",
    )


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
