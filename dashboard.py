import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import glob
import os
import re

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Style for iframes
IFRAME_STYLE = {
    "width": "100%",
    "height": "800px",
    "border": "none",
    "margin-bottom": "20px"
}

def get_scenarios():
    """Parse available scenarios from data directory

    Scenarios are named: pessimistic, base, optimistic
    - pessimistic: optimum - step (more conservative)
    - base: optimum risk threshold
    - optimistic: optimum + step (more aggressive)
    """
    files = glob.glob("data/risk_production_summary_table_*.csv")
    scenarios = []

    # Define scenario order for sorting
    scenario_order = {'pessimistic': 0, 'base': 1, 'optimistic': 2}

    for f in files:
        filename = os.path.basename(f)

        # Skip MR files for this discovery (they are secondary)
        if "_mr" in filename:
            continue

        # Match new named format: risk_production_summary_table_pessimistic.csv
        match = re.search(r"risk_production_summary_table_(pessimistic|base|optimistic).csv", filename)
        if match:
            scenarios.append(match.group(1))
            continue

        # Match legacy numeric format: risk_production_summary_table_1.0.csv
        match = re.search(r"risk_production_summary_table_(\d+\.\d+).csv", filename)
        if match:
            scenarios.append(match.group(1))
            continue

    # Also check for base file without suffix
    if os.path.exists("data/risk_production_summary_table.csv") and 'base' not in scenarios:
        scenarios.append("base")

    # Sort by scenario order (pessimistic, base, optimistic) then alphabetically for others
    return sorted(scenarios, key=lambda x: (scenario_order.get(x, 99), x))

def load_table(csv_path):
    """Load CSV and return a dash compatible table"""
    if not os.path.exists(csv_path):
        return html.Div(f"Data file not found: {csv_path}", className="text-danger")
    
    df = pd.read_csv(csv_path)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True, size="sm")

import plotly.express as px

# ... (Imports) ...

def get_comparison_data():
    """Aggregate metrics across all scenarios

    Scenarios: pessimistic, base, optimistic
    """
    scenarios = get_scenarios()
    data = []

    # Map scenario names to relative risk order for display
    scenario_order = {'pessimistic': 0, 'base': 1, 'optimistic': 2}

    for s in scenarios:
        suffix = f"_{s}"
        main_path = f"data/risk_production_summary_table{suffix}.csv"
        mr_path = f"data/risk_production_summary_table_mr{suffix}.csv"
        opt_path = f"data/optimal_solution{suffix}.csv"

        # Fallback to unsuffixed files for base scenario
        if s == "base":
            if not os.path.exists(main_path):
                main_path = "data/risk_production_summary_table.csv"
            if not os.path.exists(mr_path):
                mr_path = "data/risk_production_summary_table_mr.csv"
            if not os.path.exists(opt_path):
                opt_path = "data/optimal_solution.csv"

        row = {'Scenario': s.capitalize()}
        try:
            # Main Metrics
            if os.path.exists(main_path):
                df_main = pd.read_csv(main_path)
                opt_row = df_main[df_main['Metric'] == 'Optimum selected']
                if not opt_row.empty:
                    row['Main_Risk'] = opt_row['Risk (%)'].values[0]
                    row['Main_Prod_Pct'] = opt_row['Production (%)'].values[0]

            # MR Metrics
            if os.path.exists(mr_path):
                df_mr = pd.read_csv(mr_path)
                opt_row_mr = df_mr[df_mr['Metric'] == 'Optimum selected']
                if not opt_row_mr.empty:
                    row['MR_Risk'] = opt_row_mr['Risk (%)'].values[0]
                    row['MR_Prod_Pct'] = opt_row_mr['Production (%)'].values[0]

            # Scenario order for comparison charts
            row['Scenario_Order'] = scenario_order.get(s, 99)

            data.append(row)
        except Exception as e:
            print(f"Error processing {s}: {e}")

    return pd.DataFrame(data)

# Layout
app.layout = dbc.Container([
    html.H1("Risk Scoring Optimization Dashboard", className="my-4 text-center"),
    
    dbc.Row([
        dbc.Col([
            html.Label("Select Optimum Risk Scenario (for detailed tabs):", className="fw-bold"),
            dcc.Dropdown(
                id="scenario-selector",
                options=[{'label': s, 'value': s} for s in get_scenarios()],
                value=get_scenarios()[0] if get_scenarios() else None,
                clearable=False
            )
        ], width=4, className="mb-4")
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Scenario Comparison", tab_id="tab-comp"),
        dbc.Tab(label="Main Period", tab_id="tab-main"),
        dbc.Tab(label="MR Period (Recent)", tab_id="tab-mr"),
    ], id="tabs", active_tab="tab-comp", className="mb-3"),
    
    html.Div(id="tab-content")
    
], fluid=True)

@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab"),
     Input("scenario-selector", "value")]
)
def render_content(active_tab, scenario):
    if active_tab == "tab-comp":
        df_comp = get_comparison_data()

        if df_comp.empty:
            return html.Div("No comparison data available.")

        # Sort by scenario order (pessimistic -> base -> optimistic)
        df_comp = df_comp.sort_values("Scenario_Order")

        # Figures
        fig_risk_prod = px.scatter(
            df_comp, x="Main_Risk", y="Main_Prod_Pct", text="Scenario",
            title="Main Period: Risk vs Production Trade-off",
            labels={"Main_Risk": "Risk (%)", "Main_Prod_Pct": "Production (%)"},
            template="plotly_white", size_max=15
        )
        fig_risk_prod.update_traces(textposition='top center', marker=dict(size=12))

        fig_sensitivity = px.bar(
            df_comp, x="Scenario", y=["Main_Risk", "MR_Risk"],
            title="Risk by Scenario (Main & MR Periods)",
            barmode="group",
            labels={"value": "Risk (%)", "variable": "Period"},
            template="plotly_white"
        )
        
        return html.Div([
            html.H4("Scenario Comparison", className="my-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_risk_prod), width=6),
                dbc.Col(dcc.Graph(figure=fig_sensitivity), width=6)
            ]),
            html.Hr(),
            html.H5("Detailed Metrics"),
            dbc.Table.from_dataframe(df_comp, striped=True, bordered=True, hover=True, size="sm")
        ])
    
    if not scenario:
        return html.Div("No scenarios found.")

    # Determine suffixes and paths
    # Base scenario has both suffixed (_base) and unsuffixed versions
    suffix = f"_{scenario}"

    # Main Period Paths (check for suffixed version first, then unsuffixed for base)
    main_viz = f"images/risk_production_visualizer{suffix}.html"
    main_csv = f"data/risk_production_summary_table{suffix}.csv"

    # Fallback to unsuffixed files for base scenario
    if scenario == "base":
        if not os.path.exists(main_csv):
            main_csv = "data/risk_production_summary_table.csv"
        if not os.path.exists(main_viz):
            main_viz = "images/risk_production_visualizer.html"

    # MR Period Paths
    mr_viz = f"images/b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html"
    mr_csv = f"data/risk_production_summary_table_mr{suffix}.csv"

    # Fallback for MR as well
    if scenario == "base":
        if not os.path.exists(mr_csv):
            mr_csv = "data/risk_production_summary_table_mr.csv"
        if not os.path.exists(mr_viz):
            mr_viz = "images/b2_ever_h6_vs_octroi_and_risk_score_mr.html"

    # Format scenario name for display
    scenario_display = scenario.capitalize()

    if active_tab == "tab-main":
        return html.Div([
            html.H4(f"Main Period Summary ({scenario_display})", className="my-3"),
            load_table(main_csv),
            html.Hr(),
            html.H4("Risk Production Visualizer"),
            html.Iframe(src=main_viz, style=IFRAME_STYLE)
        ])
        
    elif active_tab == "tab-mr":
        return html.Div([
            html.H4(f"MR Period Summary ({scenario_display})", className="my-3"),
            load_table(mr_csv),
            html.Hr(),
            html.H4("MR 3D Surface Plot"),
            html.Iframe(src=mr_viz, style=IFRAME_STYLE)
        ])
    
    return html.Div("Select a tab")

if __name__ == "__main__":
    app.run(debug=True)
