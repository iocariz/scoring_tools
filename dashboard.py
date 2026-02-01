"""
Dashboard for Risk Scoring Optimization results visualization.

Provides an interactive Dash application to explore:
- Scenario comparisons (pessimistic, base, optimistic)
- Main period results and visualizations
- MR (Recent Monitoring) period results

Supports both single-run output (data/, images/) and batch output (output/{segment}/).

Usage:
    python dashboard.py                    # Auto-detect output location
    python dashboard.py --output output    # Use batch output directory
    python dashboard.py --segment seg1     # View specific segment
"""

import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import send_from_directory
from loguru import logger

from src.styles import (
    COLOR_PRIMARY,
    COLOR_RISK,
    COLOR_PRODUCTION,
    apply_plotly_style,
)

# --- Constants ---
SCENARIO_ORDER: Dict[str, int] = {
    'pessimistic': 0,
    'base': 1,
    'optimistic': 2
}

IFRAME_STYLE: Dict[str, Any] = {
    "width": "100%",
    "height": "800px",
    "border": "none",
    "marginBottom": "20px"
}

TABLE_STYLE: Dict[str, Any] = {
    "fontSize": "0.9rem",
}


# --- Global State ---
# These will be set at startup based on CLI args or auto-detection
OUTPUT_BASE: Path = Path(".")
CURRENT_SEGMENT: Optional[str] = None


def get_data_dir(segment: Optional[str] = None) -> Path:
    """Get the data directory for a segment."""
    if segment:
        return OUTPUT_BASE / segment / "data"
    # Check if we're in batch mode (output/ structure) or single mode
    if (OUTPUT_BASE / "data").exists():
        return OUTPUT_BASE / "data"
    return Path("data")


def get_images_dir(segment: Optional[str] = None) -> Path:
    """Get the images directory for a segment."""
    if segment:
        return OUTPUT_BASE / segment / "images"
    if (OUTPUT_BASE / "images").exists():
        return OUTPUT_BASE / "images"
    return Path("images")


def get_available_segments() -> List[str]:
    """Get list of available segments from output directory."""
    segments = []

    if not OUTPUT_BASE.exists():
        return segments

    for item in OUTPUT_BASE.iterdir():
        if item.is_dir() and not item.name.startswith(('_', '.')):
            # Check if it has data subdirectory (valid segment output)
            if (item / "data").exists():
                segments.append(item.name)

    return sorted(segments)


def get_scenarios(segment: Optional[str] = None) -> List[str]:
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
    scenarios: List[str] = []

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
    if (data_dir / "risk_production_summary_table.csv").exists() and 'base' not in scenarios:
        scenarios.append("base")

    # Sort by scenario order (pessimistic, base, optimistic) then alphabetically for others
    return sorted(scenarios, key=lambda x: (SCENARIO_ORDER.get(x, 99), x))


def get_scenario_paths(scenario: str, segment: Optional[str] = None) -> Dict[str, Any]:
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
        'main_csv': data_dir / f"risk_production_summary_table{suffix}.csv",
        'main_viz': f"{static_prefix}/risk_production_visualizer{suffix}.html",
        'mr_csv': data_dir / f"risk_production_summary_table_mr{suffix}.csv",
        'mr_viz': f"{static_prefix}/b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html",
        'stability': f"{static_prefix}/stability_report{suffix}.html",
        'cutoffs': data_dir / f"optimal_solution{suffix}.csv",
    }

    # Fallback to unsuffixed files for base scenario
    if scenario == "base":
        fallbacks = {
            'main_csv': data_dir / "risk_production_summary_table.csv",
            'main_viz': f"{static_prefix}/risk_production_visualizer.html",
            'mr_csv': data_dir / "risk_production_summary_table_mr.csv",
            'mr_viz': f"{static_prefix}/b2_ever_h6_vs_octroi_and_risk_score_mr.html",
            'stability': f"{static_prefix}/stability_report.html",
            'cutoffs': data_dir / "optimal_solution.csv",
        }
        for key in ['main_csv', 'mr_csv', 'cutoffs']:
            if not paths[key].exists() and fallbacks[key].exists():
                paths[key] = fallbacks[key]

        # For viz paths, check if the file exists in images dir
        images_dir = get_images_dir(segment)
        for key in ['main_viz', 'mr_viz', 'stability']:
            current_file = images_dir / paths[key].split('/')[-1]
            fallback_file = images_dir / fallbacks[key].split('/')[-1]
            if not current_file.exists() and fallback_file.exists():
                paths[key] = fallbacks[key]

    return paths


def file_exists_for_viz(viz_path: str, segment: Optional[str] = None) -> bool:
    """Check if a visualization file exists."""
    filename = viz_path.split('/')[-1]
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
        return html.Div(
            f"Data file not found: {csv_path.name}",
            className="text-warning p-3 bg-light rounded"
        )

    try:
        df = pd.read_csv(csv_path)

        # Format numeric columns for better display
        for col in df.select_dtypes(include=['float64']).columns:
            if 'Risk' in col or '%' in col:
                df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) and abs(x) < 1 else f"{x:.2f}%")
            elif 'Production' in col and '€' in col:
                df[col] = df[col].apply(lambda x: f"€{x:,.0f}" if pd.notna(x) else "")

        return dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            size="sm",
            style=TABLE_STYLE
        )
    except Exception as e:
        logger.error(f"Error loading table from {csv_path}: {e}")
        return html.Div(f"Error loading data: {e}", className="text-danger p-3")


def load_iframe(viz_path: str, segment: Optional[str] = None,
                fallback_message: str = "Visualization not available") -> html.Div:
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
        return html.Div(
            fallback_message,
            className="text-muted p-3 text-center bg-light rounded"
        )

    return html.Iframe(src=viz_path, style=IFRAME_STYLE)


def get_comparison_data(segment: Optional[str] = None) -> pd.DataFrame:
    """
    Aggregate metrics across all scenarios for comparison.

    Returns:
        DataFrame with scenario comparison metrics.
    """
    scenarios = get_scenarios(segment)
    data: List[Dict[str, Any]] = []

    for scenario in scenarios:
        paths = get_scenario_paths(scenario, segment)
        row: Dict[str, Any] = {
            'Scenario': scenario.capitalize(),
            'Scenario_Order': SCENARIO_ORDER.get(scenario, 99)
        }

        try:
            # Main Period Metrics
            if paths['main_csv'].exists():
                df_main = pd.read_csv(paths['main_csv'])
                opt_row = df_main[df_main['Metric'] == 'Optimum selected']
                if not opt_row.empty:
                    row['Main_Risk'] = opt_row['Risk (%)'].values[0]
                    row['Main_Prod_Pct'] = opt_row['Production (%)'].values[0]

            # MR Period Metrics
            if paths['mr_csv'].exists():
                df_mr = pd.read_csv(paths['mr_csv'])
                opt_row_mr = df_mr[df_mr['Metric'] == 'Optimum selected']
                if not opt_row_mr.empty:
                    row['MR_Risk'] = opt_row_mr['Risk (%)'].values[0]
                    row['MR_Prod_Pct'] = opt_row_mr['Production (%)'].values[0]

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
        return html.Div([
            dbc.Alert([
                html.H4("No Data Available", className="alert-heading"),
                html.P("No optimization results found. Please run the pipeline first:"),
                html.Code("uv run python main.py", className="d-block p-2 bg-light mt-2"),
                html.P("Or for batch processing:", className="mt-2"),
                html.Code("uv run python run_batch.py", className="d-block p-2 bg-light mt-2"),
            ], color="info", className="mt-3")
        ])

    # Sort by scenario order
    df_comp = df_comp.sort_values("Scenario_Order")

    # Risk vs Production scatter
    fig_risk_prod = px.scatter(
        df_comp,
        x="Main_Risk",
        y="Main_Prod_Pct",
        text="Scenario",
        size_max=15
    )
    fig_risk_prod.update_traces(
        textposition='top center',
        marker=dict(size=14, color=COLOR_PRIMARY)
    )
    apply_plotly_style(
        fig_risk_prod,
        title="Main Period: Risk vs Production Trade-off",
        height=400
    )
    fig_risk_prod.update_xaxes(title="Risk (%)")
    fig_risk_prod.update_yaxes(title="Production (%)")

    # Risk comparison bar chart
    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(go.Bar(
        name='Main Period',
        x=df_comp['Scenario'],
        y=df_comp['Main_Risk'],
        marker_color=COLOR_PRIMARY
    ))

    if 'MR_Risk' in df_comp.columns and df_comp['MR_Risk'].notna().any():
        fig_sensitivity.add_trace(go.Bar(
            name='MR Period',
            x=df_comp['Scenario'],
            y=df_comp['MR_Risk'],
            marker_color=COLOR_RISK
        ))

    fig_sensitivity.update_layout(barmode='group')
    apply_plotly_style(
        fig_sensitivity,
        title="Risk by Scenario (Main & MR Periods)",
        height=400
    )
    fig_sensitivity.update_xaxes(title="Scenario")
    fig_sensitivity.update_yaxes(title="Risk (%)")

    # Production comparison
    fig_production = go.Figure()

    fig_production.add_trace(go.Bar(
        name='Main Period',
        x=df_comp['Scenario'],
        y=df_comp['Main_Prod_Pct'],
        marker_color=COLOR_PRODUCTION
    ))

    if 'MR_Prod_Pct' in df_comp.columns and df_comp['MR_Prod_Pct'].notna().any():
        fig_production.add_trace(go.Bar(
            name='MR Period',
            x=df_comp['Scenario'],
            y=df_comp['MR_Prod_Pct'],
            marker_color=COLOR_PRIMARY
        ))

    fig_production.update_layout(barmode='group')
    apply_plotly_style(
        fig_production,
        title="Production by Scenario (Main & MR Periods)",
        height=400
    )
    fig_production.update_xaxes(title="Scenario")
    fig_production.update_yaxes(title="Production (%)")

    # Format display DataFrame
    display_df = df_comp.drop(columns=['Scenario_Order'], errors='ignore').copy()
    for col in ['Main_Risk', 'MR_Risk']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    for col in ['Main_Prod_Pct', 'MR_Prod_Pct']:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")

    return html.Div([
        html.H4("Scenario Comparison", className="my-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_risk_prod), md=6),
            dbc.Col(dcc.Graph(figure=fig_sensitivity), md=6)
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_production), md=6),
        ], className="mt-3"),
        html.Hr(),
        html.H5("Summary Metrics", className="mt-4"),
        dbc.Table.from_dataframe(
            display_df,
            striped=True,
            bordered=True,
            hover=True,
            size="sm"
        )
    ])


def create_empty_state() -> html.Div:
    """Create an empty state message when no data is available."""
    return dbc.Alert([
        html.H4("No Data Available", className="alert-heading"),
        html.P("No optimization results found. Please run the pipeline first:"),
        html.Code("uv run python main.py", className="d-block p-2 bg-light mt-2"),
        html.P("Or for batch processing:", className="mt-2"),
        html.Code("uv run python run_batch.py", className="d-block p-2 bg-light mt-2"),
        html.Hr(),
        html.P([
            "This dashboard displays results from the credit risk scoring optimization pipeline. ",
            "Once you run the pipeline, the following will be generated:"
        ], className="mb-2"),
        html.Ul([
            html.Li("Scenario comparison (pessimistic, base, optimistic)"),
            html.Li("Risk production summary tables"),
            html.Li("Interactive visualizations"),
            html.Li("Stability analysis reports"),
        ])
    ], color="info", className="mt-3")


# --- Initialize Dash App ---
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Risk Scoring Dashboard",
    suppress_callback_exceptions=True
)
server = app.server


# Serve static files from images directory (supports both single and batch modes)
@server.route('/static/<path:filename>')
def serve_static_root(filename):
    """Serve static HTML files from root images directory."""
    images_dir = get_images_dir(None)
    return send_from_directory(str(images_dir), filename)


@server.route('/static/<segment>/<path:filename>')
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

    return dbc.Container([
        # Header
        html.Div([
            html.H1("Risk Scoring Optimization Dashboard", className="text-center mb-0"),
            html.P(
                "Explore optimization results across scenarios and time periods",
                className="text-muted text-center"
            )
        ], className="my-4"),

        # Segment and Scenario selectors
        dbc.Row([
            # Segment selector (only show if batch output exists)
            dbc.Col([
                html.Label("Select Segment:", className="fw-bold"),
                dcc.Dropdown(
                    id="segment-selector",
                    options=[{'label': s, 'value': s} for s in segments],
                    value=initial_segment,
                    clearable=False,
                    disabled=not has_segments,
                    className="mb-3",
                    placeholder="No segments available" if not has_segments else "Select segment..."
                )
            ], md=4, style={'display': 'block' if has_segments else 'none'}),

            # Scenario selector
            dbc.Col([
                html.Label("Select Scenario:", className="fw-bold"),
                dcc.Dropdown(
                    id="scenario-selector",
                    options=[{'label': s.capitalize(), 'value': s} for s in scenarios],
                    value=scenarios[0] if scenarios else None,
                    clearable=False,
                    disabled=not has_data,
                    className="mb-3",
                    placeholder="No scenarios available" if not has_data else "Select scenario..."
                )
            ], md=4)
        ]),

        # Tabs
        dbc.Tabs([
            dbc.Tab(label="Scenario Comparison", tab_id="tab-comp"),
            dbc.Tab(label="Main Period", tab_id="tab-main", disabled=not has_data),
            dbc.Tab(label="MR Period (Recent)", tab_id="tab-mr", disabled=not has_data),
            dbc.Tab(label="Stability Analysis", tab_id="tab-stability", disabled=not has_data),
        ], id="tabs", active_tab="tab-comp", className="mb-3"),

        # Content area
        html.Div(id="tab-content", className="mt-3"),

        # Footer
        html.Hr(className="mt-5"),
        html.Footer([
            html.Small(
                f"Risk Scoring Optimization Tools | Output: {OUTPUT_BASE}",
                className="text-muted"
            )
        ], className="text-center mb-3")

    ], fluid=True)


app.layout = create_layout


# --- Callbacks ---

@app.callback(
    [Output("scenario-selector", "options"),
     Output("scenario-selector", "value"),
     Output("scenario-selector", "disabled"),
     Output("tabs", "children")],
    [Input("segment-selector", "value")]
)
def update_scenarios_on_segment_change(segment: Optional[str]):
    """Update scenario dropdown when segment changes."""
    scenarios = get_scenarios(segment)
    has_data = len(scenarios) > 0

    options = [{'label': s.capitalize(), 'value': s} for s in scenarios]
    value = scenarios[0] if scenarios else None

    # Rebuild tabs with correct disabled state
    tabs = [
        dbc.Tab(label="Scenario Comparison", tab_id="tab-comp"),
        dbc.Tab(label="Main Period", tab_id="tab-main", disabled=not has_data),
        dbc.Tab(label="MR Period (Recent)", tab_id="tab-mr", disabled=not has_data),
        dbc.Tab(label="Stability Analysis", tab_id="tab-stability", disabled=not has_data),
    ]

    return options, value, not has_data, tabs


@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("scenario-selector", "value"),
     Input("segment-selector", "value")]
)
def render_content(active_tab: str, scenario: Optional[str], segment: Optional[str]) -> html.Div:
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
        return html.Div([
            html.H4(f"Main Period Results ({scenario_display}{segment_display})", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Summary Metrics"),
                dbc.CardBody(load_table(paths['main_csv']))
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Risk Production Visualizer"),
                dbc.CardBody(load_iframe(
                    paths['main_viz'],
                    segment,
                    "Main period visualization not available. Run the pipeline to generate."
                ))
            ])
        ])

    elif active_tab == "tab-mr":
        return html.Div([
            html.H4(f"MR Period Results ({scenario_display}{segment_display})", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Summary Metrics"),
                dbc.CardBody(load_table(paths['mr_csv']))
            ], className="mb-4"),
            dbc.Card([
                dbc.CardHeader("Risk Surface (3D)"),
                dbc.CardBody(load_iframe(
                    paths['mr_viz'],
                    segment,
                    "MR period visualization not available. Run the pipeline to generate."
                ))
            ])
        ])

    elif active_tab == "tab-stability":
        return html.Div([
            html.H4(f"Stability Analysis ({scenario_display}{segment_display})", className="mb-4"),
            dbc.Card([
                dbc.CardHeader("PSI/CSI Stability Report"),
                dbc.CardBody(load_iframe(
                    paths['stability'],
                    segment,
                    "Stability report not available. Run MR period processing to generate."
                ))
            ])
        ])

    return html.Div("Select a tab", className="text-muted")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Risk Scoring Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Base output directory for batch results (default: output)"
    )
    parser.add_argument(
        "--segment", "-s",
        default=None,
        help="Initial segment to display"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8050,
        help="Port to run dashboard on (default: 8050)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode"
    )
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
