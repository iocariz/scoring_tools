"""
Interactive Global Allocator Dashboard

A Dash web app for interactive global portfolio optimization.
Loads efficient frontiers from output/*/data/ and allocates risk targets
to maximize global production.

Usage:
    uv run python interactive_allocator.py
    uv run python interactive_allocator.py --port 8051 --debug
"""

import argparse
import os
import tomllib
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dash_table, dcc, html
from loguru import logger

from src.global_optimizer import GlobalAllocator, SegmentConstraints
from src.styles import COLOR_RISK, apply_plotly_style

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# --- Helper Functions ---


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Global Allocator Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", "-p", type=int, default=8051, help="Port to run dashboard on (default: 8051)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    return parser.parse_args()


def get_available_scenarios(output_base: Path) -> list[str]:
    """Scan the output directory to find available scenarios."""
    scenarios = set()
    if not output_base.exists():
        return []
    for segment_dir in output_base.iterdir():
        if segment_dir.is_dir() and not segment_dir.name.startswith(("_", ".")):
            data_dir = segment_dir / "data"
            if data_dir.exists():
                for f in data_dir.glob("efficient_frontier_*.csv"):
                    scenario = f.stem.replace("efficient_frontier_", "")
                    scenarios.add(scenario)
    return sorted(scenarios)


def get_segments_for_scenario(output_base: Path, scenario: str) -> list[str]:
    """Return segment names that have a frontier file for the given scenario."""
    segments = []
    if not output_base.exists():
        return segments
    for segment_dir in sorted(output_base.iterdir()):
        if segment_dir.is_dir() and not segment_dir.name.startswith(("_", ".")):
            frontier_path = segment_dir / "data" / f"efficient_frontier_{scenario}.csv"
            if frontier_path.exists():
                segments.append(segment_dir.name)
    return segments


def load_segments_config(segments_path: str = "segments.toml") -> dict[str, dict]:
    """Read segments.toml and return a dict of segment config values."""
    path = Path(segments_path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        config = tomllib.load(f)
    result = {}
    for seg_name, seg_cfg in config.get("segments", {}).items():
        result[seg_name] = {
            "min_risk": seg_cfg.get("min_risk"),
            "max_risk": seg_cfg.get("max_risk"),
            "optimum_risk": seg_cfg.get("optimum_risk"),
        }
    return result


def create_empty_state() -> dbc.Alert:
    """Create an empty state message when no scenarios are available."""
    return dbc.Alert(
        [
            html.H4("No Scenarios Available", className="alert-heading"),
            html.P("No efficient frontiers found in the output directory. Please run the pipeline first:"),
            html.Code("uv run python run_batch.py", className="d-block p-2 bg-light mt-2"),
            html.Hr(),
            html.P(
                "Once the pipeline has run, this app will detect the generated efficient frontiers "
                "and allow you to configure and run the global allocation interactively.",
                className="mb-0",
            ),
        ],
        color="info",
        className="mt-3",
    )


# --- Layout ---

app.layout = dbc.Container(
    [
        html.H1("Interactive Global Allocator", className="my-4 text-center"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Global Configuration"),
                                dbc.CardBody(
                                    [
                                        dbc.Label("Global Risk Target (%)", html_for="risk-target-input"),
                                        dbc.Input(
                                            id="risk-target-input", type="number", value=1.0, step=0.1, className="mb-3"
                                        ),
                                        dbc.Label("Optimization Method", html_for="method-input"),
                                        dbc.RadioItems(
                                            id="method-input",
                                            options=[
                                                {"label": "Exact (MILP)", "value": "exact"},
                                                {"label": "Greedy", "value": "greedy"},
                                            ],
                                            value="exact",
                                            inline=True,
                                            className="mb-3",
                                        ),
                                        dbc.Label("Scenario", html_for="scenario-dropdown"),
                                        dcc.Dropdown(
                                            id="scenario-dropdown",
                                            options=[],
                                            value=None,
                                            clearable=False,
                                            className="mb-3",
                                        ),
                                        dbc.Label(
                                            "Global Production Floor (optional)",
                                            html_for="global-prod-floor-input",
                                        ),
                                        dbc.Input(
                                            id="global-prod-floor-input",
                                            type="number",
                                            value=None,
                                            placeholder="No floor",
                                            className="mb-3",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Segment Constraints"),
                                dbc.CardBody(
                                    [
                                        html.P(
                                            "Edit per-segment constraints below. "
                                            "Values auto-populate from segments.toml when a scenario is selected.",
                                            className="text-muted small mb-2",
                                        ),
                                        dash_table.DataTable(
                                            id="constraints-table",
                                            columns=[
                                                {"name": "Segment", "id": "segment", "editable": False},
                                                {"name": "Min Risk", "id": "min_risk", "type": "numeric"},
                                                {"name": "Max Risk", "id": "max_risk", "type": "numeric"},
                                                {"name": "Min Production", "id": "min_production", "type": "numeric"},
                                                {"name": "Locked", "id": "locked", "presentation": "dropdown"},
                                            ],
                                            data=[],
                                            editable=True,
                                            row_deletable=False,
                                            style_table={"overflowX": "auto"},
                                            style_cell={"textAlign": "left", "padding": "4px 8px", "fontSize": "13px"},
                                            style_header={"fontWeight": "bold"},
                                            dropdown={
                                                "locked": {
                                                    "options": [
                                                        {"label": "No", "value": "No"},
                                                        {"label": "Yes", "value": "Yes"},
                                                    ]
                                                }
                                            },
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3",
                        ),
                        dbc.Button(
                            "Run Optimization",
                            id="run-button",
                            color="primary",
                            n_clicks=0,
                            className="w-100 mb-3",
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dcc.Loading(
                            id="loading-output",
                            type="default",
                            children=[
                                dbc.Card(
                                    [
                                        dbc.CardHeader("Optimization Results"),
                                        dbc.CardBody(
                                            id="results-output",
                                            children=[
                                                html.Div(
                                                    "Please configure and run the optimization.",
                                                    className="text-muted",
                                                )
                                            ],
                                        ),
                                    ]
                                )
                            ],
                        )
                    ],
                    md=8,
                ),
            ]
        ),
        dcc.Store(id="results-csv-store"),
        dcc.Download(id="download-csv"),
    ],
    fluid=True,
)

# --- Callbacks ---


@app.callback(
    Output("scenario-dropdown", "options"),
    Output("scenario-dropdown", "value"),
    Output("results-output", "children", allow_duplicate=True),
    Input("risk-target-input", "value"),  # Dummy input to trigger on load
    prevent_initial_call="initial_duplicate",
)
def update_scenario_dropdown(_):
    scenarios = get_available_scenarios(Path("output"))
    options = [{"label": s.capitalize(), "value": s} for s in scenarios]
    value = options[0]["value"] if options else None

    if not options:
        logger.warning("No scenarios found in output/")
        return options, value, create_empty_state()

    logger.info(f"Found {len(scenarios)} scenarios: {scenarios}")
    return options, value, html.Div("Please configure and run the optimization.", className="text-muted")


@app.callback(
    Output("constraints-table", "data"),
    Input("scenario-dropdown", "value"),
    prevent_initial_call=True,
)
def populate_constraints_table(scenario):
    """Auto-populate constraint table when scenario changes."""
    if not scenario:
        return []

    output_base = Path("output")
    segments = get_segments_for_scenario(output_base, scenario)
    if not segments:
        return []

    # Load defaults from segments.toml
    seg_config = load_segments_config()

    rows = []
    for seg in segments:
        cfg = seg_config.get(seg, {})
        rows.append(
            {
                "segment": seg,
                "min_risk": cfg.get("min_risk"),
                "max_risk": cfg.get("max_risk"),
                "min_production": None,
                "locked": "No",
            }
        )

    logger.info(f"Populated constraints table for {len(rows)} segments (scenario={scenario})")
    return rows


@app.callback(
    Output("results-output", "children"),
    Output("results-csv-store", "data"),
    Input("run-button", "n_clicks"),
    State("risk-target-input", "value"),
    State("method-input", "value"),
    State("scenario-dropdown", "value"),
    State("constraints-table", "data"),
    State("global-prod-floor-input", "value"),
    prevent_initial_call=True,
)
def run_optimization(n_clicks, target, method, scenario, table_data, global_prod_floor):
    if n_clicks == 0:
        return dash.no_update, dash.no_update

    if target is None:
        return dbc.Alert("Global Risk Target must be set.", color="danger"), dash.no_update

    if scenario is None:
        return dbc.Alert("No scenario selected. Run the pipeline first.", color="danger"), dash.no_update

    allocator = GlobalAllocator()
    output_base = Path("output")
    segments_found = []

    if not output_base.exists():
        return (
            dbc.Alert("Output directory 'output/' does not exist. Run run_batch.py first.", color="danger"),
            dash.no_update,
        )

    for segment_dir in output_base.iterdir():
        if segment_dir.is_dir() and not segment_dir.name.startswith(("_", ".")):
            frontier_path = segment_dir / "data" / f"efficient_frontier_{scenario}.csv"
            if frontier_path.exists():
                try:
                    df = pd.read_csv(frontier_path)
                    allocator.load_frontier(segment_dir.name, df)
                    segments_found.append(segment_dir.name)
                    logger.info(f"Loaded frontier for {segment_dir.name}")
                except Exception as e:
                    logger.error(f"Failed to load {frontier_path}: {e}")
                    return dbc.Alert(f"Failed to load {frontier_path}: {e}", color="danger"), dash.no_update

    if not segments_found:
        logger.warning(f"No efficient frontiers found for scenario '{scenario}'")
        return (
            dbc.Alert(f"No efficient frontiers found for scenario '{scenario}' in output/*/", color="danger"),
            dash.no_update,
        )

    # Build SegmentConstraints from table data
    constraints = {}
    if table_data:
        for row in table_data:
            seg = row.get("segment")
            if not seg:
                continue
            sc = SegmentConstraints(
                min_risk=row.get("min_risk") if row.get("min_risk") not in (None, "") else None,
                max_risk=row.get("max_risk") if row.get("max_risk") not in (None, "") else None,
                min_production=row.get("min_production") if row.get("min_production") not in (None, "") else None,
            )
            # Handle locking: if locked, use the current allocation's sol_fac (set after first run)
            # For now, locked=Yes without a prior result means no lock
            if row.get("locked") == "Yes" and row.get("locked_sol_fac") is not None:
                sc.locked_sol_fac = int(row["locked_sol_fac"])
            # Only add constraint if it has any non-None value
            if (
                sc.min_risk is not None
                or sc.max_risk is not None
                or sc.min_production is not None
                or sc.locked_sol_fac is not None
            ):
                constraints[seg] = sc

        if constraints:
            logger.info(f"Segment constraints loaded for: {list(constraints.keys())}")

    global_production_floor = global_prod_floor if global_prod_floor not in (None, "") else None

    logger.info(f"Running {method} optimization for target {target}% across {len(segments_found)} segments")

    try:
        result = allocator.optimize(
            target,
            constraints=constraints if constraints else None,
            global_production_floor=global_production_floor,
            method=method,
        )
        result_df = result.to_full_dataframe()
        logger.info(f"Optimization successful for {len(segments_found)} segments")

        # --- Binding constraints badges ---
        binding_badges = []
        for bc in result.binding_constraints:
            binding_badges.append(dbc.Badge(bc.replace("_", " ").title(), color="warning", className="me-1"))

        # --- Create Visualizations ---

        # Bar chart for allocated risk
        fig_risk = px.bar(
            result_df,
            x="segment",
            y="b2_ever_h6",
            title="Allocated Risk by Segment",
            labels={"b2_ever_h6": "Allocated Risk (%)"},
            color_discrete_sequence=[COLOR_RISK],
        )
        apply_plotly_style(fig_risk)

        # Pie chart for production
        fig_prod = px.pie(
            result_df,
            values="oa_amt_h0",
            names="segment",
            title="Production by Segment",
            labels={"oa_amt_h0": "Production"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        apply_plotly_style(fig_prod)

        # --- Format Table ---
        table_data_display = result_df.to_dict("records")
        table_columns = [{"name": col, "id": col} for col in result_df.columns]

        for item in table_data_display:
            if "b2_ever_h6" in item and pd.notna(item["b2_ever_h6"]) and np.isfinite(item["b2_ever_h6"]):
                item["b2_ever_h6"] = f"{item['b2_ever_h6']:.2%}"
            if "oa_amt_h0" in item and pd.notna(item["oa_amt_h0"]) and np.isfinite(item["oa_amt_h0"]):
                item["oa_amt_h0"] = f"\u20ac{item['oa_amt_h0']:,.0f}"

        table = dash_table.DataTable(
            columns=table_columns,
            data=table_data_display,
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left"},
        )

        # --- Summary Stats ---
        total_prod = result.global_production
        overall_risk = result.global_risk

        summary_cards = dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(
                                        f"\u20ac{total_prod:,.0f}"
                                        if pd.notna(total_prod) and np.isfinite(total_prod)
                                        else "N/A",
                                        className="card-title",
                                    ),
                                    html.P("Total Production", className="card-text"),
                                ]
                            )
                        ],
                        color="success",
                        inverse=True,
                    ),
                    md=6,
                ),
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardBody(
                                [
                                    html.H4(
                                        f"{overall_risk:.2%}"
                                        if pd.notna(overall_risk) and np.isfinite(overall_risk)
                                        else "N/A",
                                        className="card-title",
                                    ),
                                    html.P("Overall Risk", className="card-text"),
                                ]
                            )
                        ],
                        color="danger",
                        inverse=True,
                    ),
                    md=6,
                ),
            ]
        )

        # --- Store CSV data for deferred download ---
        csv_string = result_df.to_csv(index=False)

        # --- Layout Results ---
        results_children = [
            dbc.Alert(f"Optimization successful for {len(segments_found)} segments.", color="success"),
        ]
        if binding_badges:
            results_children.append(
                html.Div(
                    [html.Strong("Binding constraints: ")] + binding_badges,
                    className="mb-3",
                )
            )
        results_children.extend(
            [
                summary_cards,
                html.Hr(),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_risk), md=6),
                        dbc.Col(dcc.Graph(figure=fig_prod), md=6),
                    ],
                    className="mt-4",
                ),
                html.Hr(),
                html.H4("Detailed Results", className="mt-4"),
                dbc.Button("Download Results", id="download-button", color="secondary", className="mb-3"),
                table,
            ]
        )

        results_layout = html.Div(results_children)

        return results_layout, csv_string

    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return dbc.Alert(f"An unexpected error occurred during optimization: {e}", color="danger"), dash.no_update


@app.callback(
    Output("download-csv", "data"),
    Input("download-button", "n_clicks"),
    State("results-csv-store", "data"),
    prevent_initial_call=True,
)
def download_results(n_clicks, csv_string):
    if not csv_string:
        return dash.no_update
    return dict(content=csv_string, filename="allocation_results.csv", type="text/csv")


if __name__ == "__main__":
    args = parse_args()
    logger.info("Starting Interactive Global Allocator...")
    # debug=True enables Werkzeug's interactive console (RCE vector if network-accessible).
    # Refuse the flag unless DASHBOARD_DEBUG_ALLOWED=1 is explicitly set in the environment.
    debug_requested = bool(args.debug)
    debug_allowed = os.environ.get("DASHBOARD_DEBUG_ALLOWED") == "1"
    if debug_requested and not debug_allowed:
        logger.warning(
            "--debug requested but DASHBOARD_DEBUG_ALLOWED=1 is not set; "
            "refusing to enable Werkzeug debug console (RCE risk)."
        )
    effective_debug = debug_requested and debug_allowed
    app.run(debug=effective_debug, port=args.port)
