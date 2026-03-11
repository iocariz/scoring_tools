"""
Gradio Dashboard for Risk Scoring Optimization results.

Functional prototype covering the same tabs as dashboard.py but with ~10x less code
thanks to Gradio's declarative API.

Usage:
    uv run python gradio_dashboard.py
    uv run python gradio_dashboard.py --output output --segment seg1
    uv run python gradio_dashboard.py --port 7860
"""

import argparse
import json
import re
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ---------------------------------------------------------------------------
# Global state (set at startup from CLI)
# ---------------------------------------------------------------------------
OUTPUT_BASE: Path = Path(".")
CURRENT_SEGMENT: str | None = None

SCENARIO_ORDER: dict[str, int] = {"pessimistic": 0, "base": 1, "optimistic": 2}

# ---------------------------------------------------------------------------
# Path helpers (mirrors dashboard.py patterns)
# ---------------------------------------------------------------------------


def get_data_dir(segment: str | None = None) -> Path:
    if segment:
        return OUTPUT_BASE / segment / "data"
    if (OUTPUT_BASE / "data").exists():
        return OUTPUT_BASE / "data"
    return Path("data")


def get_images_dir(segment: str | None = None) -> Path:
    if segment:
        return OUTPUT_BASE / segment / "images"
    if (OUTPUT_BASE / "images").exists():
        return OUTPUT_BASE / "images"
    return Path("images")


def get_available_segments() -> list[str]:
    if not OUTPUT_BASE.exists():
        return []
    return sorted(
        d.name
        for d in OUTPUT_BASE.iterdir()
        if d.is_dir() and not d.name.startswith(("_", ".")) and (d / "data").exists()
    )


def _resolve_supersegment(segment: str) -> str | None:
    for toml_path in [Path("segments.toml"), OUTPUT_BASE / "segments.toml"]:
        if toml_path.exists():
            try:
                import tomllib

                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                for name, cfg in data.get("segments", {}).items():
                    if name == segment:
                        return cfg.get("supersegment")
            except Exception:
                pass
    return None


def get_models_dir(segment: str | None = None) -> Path:
    if segment:
        direct = OUTPUT_BASE / segment / "models"
        if direct.exists() and any(direct.iterdir()):
            return direct
        superseg = _resolve_supersegment(segment)
        if superseg:
            models_dir = OUTPUT_BASE / f"_supersegment_{superseg}" / "models"
            if models_dir.exists():
                return models_dir
        return direct
    if (OUTPUT_BASE / "models").exists():
        return OUTPUT_BASE / "models"
    return Path("models")


def get_scenarios(segment: str | None = None) -> list[str]:
    data_dir = get_data_dir(segment)
    if not data_dir.exists():
        return []
    scenarios: list[str] = []
    for f in data_dir.glob("risk_production_summary_table_*.csv"):
        if "_mr" in f.name:
            continue
        m = re.search(r"risk_production_summary_table_(pessimistic|base|optimistic)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))
            continue
        m = re.search(r"risk_production_summary_table_(\d+\.\d+)\.csv", f.name)
        if m:
            scenarios.append(m.group(1))
    if (data_dir / "risk_production_summary_table.csv").exists() and "base" not in scenarios:
        scenarios.append("base")
    return sorted(scenarios, key=lambda x: (SCENARIO_ORDER.get(x, 99), x))


def get_scenario_paths(scenario: str, segment: str | None = None) -> dict:
    data_dir = get_data_dir(segment)
    suffix = f"_{scenario}"
    paths = {
        "main_csv": data_dir / f"risk_production_summary_table{suffix}.csv",
        "mr_csv": data_dir / f"risk_production_summary_table_mr{suffix}.csv",
        "cutoffs": data_dir / f"optimal_solution{suffix}.csv",
        "summary_data": data_dir / f"data_summary_desagregado{suffix}.csv",
    }
    if scenario == "base":
        fallbacks = {
            "main_csv": data_dir / "risk_production_summary_table.csv",
            "mr_csv": data_dir / "risk_production_summary_table_mr.csv",
            "cutoffs": data_dir / "optimal_solution.csv",
            "summary_data": data_dir / "data_summary_desagregado.csv",
        }
        for key in paths:
            if not paths[key].exists() and fallbacks[key].exists():
                paths[key] = fallbacks[key]
    return paths


# ---------------------------------------------------------------------------
# KPI card helper (styled HTML via gr.Markdown)
# ---------------------------------------------------------------------------


def kpi_card(title: str, value: str, delta: str = "", border_color: str = COLOR_PRIMARY) -> str:
    delta_color = COLOR_NEUTRAL
    if delta.startswith("+"):
        delta_color = COLOR_GOOD
    elif delta.startswith("-") or delta.startswith("\u2212"):
        delta_color = COLOR_BAD
    delta_html = f'<span style="font-size:0.75rem;color:{delta_color}">{delta}</span>' if delta else ""
    return (
        f'<div style="border-left:4px solid {border_color};border-radius:4px;'
        f'padding:12px 16px;background:white;box-shadow:0 1px 3px rgba(0,0,0,.08);min-width:160px">'
        f'<div style="text-transform:uppercase;font-size:0.7rem;font-weight:600;color:{COLOR_NEUTRAL}">{title}</div>'
        f'<div style="font-size:1.4rem;font-weight:700;color:{COLOR_PRIMARY}">{value}</div>'
        f"{delta_html}</div>"
    )


def kpi_row(cards: list[str]) -> str:
    cells = "".join(f'<td style="padding:0 8px 0 0;vertical-align:top">{c}</td>' for c in cards)
    return f"<table><tr>{cells}</tr></table>"


# ---------------------------------------------------------------------------
# Tab 1 — Scenario Comparison
# ---------------------------------------------------------------------------


def load_comparison_data(segment: str | None) -> pd.DataFrame:
    scenarios = get_scenarios(segment)
    rows: list[dict] = []
    for sc in scenarios:
        paths = get_scenario_paths(sc, segment)
        row: dict = {"Scenario": sc.capitalize(), "Scenario_Order": SCENARIO_ORDER.get(sc, 99)}
        try:
            if paths["main_csv"].exists():
                df = pd.read_csv(paths["main_csv"])
                opt = df[df["Metric"] == "Optimum selected"]
                if not opt.empty:
                    row["Main_Risk"] = opt["Risk (%)"].values[0] if not pd.isna(opt["Risk (%)"].values[0]) else None
                    row["Main_Prod_Pct"] = (
                        opt["Production (%)"].values[0] if not pd.isna(opt["Production (%)"].values[0]) else None
                    )
            if paths["mr_csv"].exists():
                df_mr = pd.read_csv(paths["mr_csv"])
                opt_mr = df_mr[df_mr["Metric"] == "Optimum selected"]
                if not opt_mr.empty:
                    row["MR_Risk"] = opt_mr["Risk (%)"].values[0] if not pd.isna(opt_mr["Risk (%)"].values[0]) else None
                    row["MR_Prod_Pct"] = (
                        opt_mr["Production (%)"].values[0] if not pd.isna(opt_mr["Production (%)"].values[0]) else None
                    )
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error processing scenario {sc}: {e}")
    return pd.DataFrame(rows)


def build_comparison_charts(segment: str | None) -> tuple:
    """Return (scatter_fig, risk_bar_fig, prod_bar_fig, kpi_html)."""
    df = load_comparison_data(segment)
    if df.empty:
        empty = go.Figure().update_layout(title="No data available")
        return empty, empty, empty, "No optimization results found. Run the pipeline first."

    df = df.sort_values("Scenario_Order")
    has_mr_risk = "MR_Risk" in df.columns and df["MR_Risk"].notna().any()
    has_mr_prod = "MR_Prod_Pct" in df.columns and df["MR_Prod_Pct"].notna().any()

    # Risk vs Production scatter
    fig_scatter = px.scatter(df, x="Main_Risk", y="Main_Prod_Pct", text="Scenario", size_max=15)
    fig_scatter.update_traces(textposition="top center", marker=dict(size=14, color=COLOR_PRIMARY))
    apply_plotly_style(fig_scatter, title="Risk vs Production Trade-off", height=400)
    fig_scatter.update_xaxes(title="Risk (%)")
    fig_scatter.update_yaxes(title="Production (%)")

    # Risk bars
    fig_risk = go.Figure()
    fig_risk.add_trace(go.Bar(name="Main", x=df["Scenario"], y=df["Main_Risk"], marker_color=COLOR_PRIMARY))
    if has_mr_risk:
        fig_risk.add_trace(go.Bar(name="MR", x=df["Scenario"], y=df["MR_Risk"], marker_color=COLOR_RISK))
        for _, r in df.iterrows():
            delta = r.get("Main_Risk", 0) - r.get("MR_Risk", 0) if pd.notna(r.get("MR_Risk")) else None
            if delta is not None:
                color = COLOR_GOOD if delta <= 0 else COLOR_BAD
                fig_risk.add_annotation(
                    x=r["Scenario"],
                    y=max(r["Main_Risk"], r["MR_Risk"]),
                    text=f"\u0394 {delta:+.2f}pp",
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color=color, weight="bold"),
                )
    fig_risk.update_layout(barmode="group")
    apply_plotly_style(fig_risk, title="Risk by Scenario", height=400)

    # Production bars
    fig_prod = go.Figure()
    fig_prod.add_trace(go.Bar(name="Main", x=df["Scenario"], y=df["Main_Prod_Pct"], marker_color=COLOR_PRODUCTION))
    if has_mr_prod:
        fig_prod.add_trace(go.Bar(name="MR", x=df["Scenario"], y=df["MR_Prod_Pct"], marker_color=COLOR_PRIMARY))
        for _, r in df.iterrows():
            delta = r.get("Main_Prod_Pct", 0) - r.get("MR_Prod_Pct", 0) if pd.notna(r.get("MR_Prod_Pct")) else None
            if delta is not None:
                color = COLOR_GOOD if delta >= 0 else COLOR_BAD
                fig_prod.add_annotation(
                    x=r["Scenario"],
                    y=max(r["Main_Prod_Pct"], r["MR_Prod_Pct"]),
                    text=f"\u0394 {delta:+.1%}",
                    showarrow=False,
                    yshift=15,
                    font=dict(size=10, color=color, weight="bold"),
                )
    fig_prod.update_layout(barmode="group")
    apply_plotly_style(fig_prod, title="Production by Scenario", height=400)

    # KPI summary
    cards = []
    for _, r in df.iterrows():
        risk_val = f"{r['Main_Risk']:.2f}%" if pd.notna(r.get("Main_Risk")) else "-"
        prod_val = f"{r['Main_Prod_Pct']:.1%}" if pd.notna(r.get("Main_Prod_Pct")) else "-"
        cards.append(kpi_card(f"{r['Scenario']} Risk", risk_val, border_color=COLOR_RISK))
        cards.append(kpi_card(f"{r['Scenario']} Prod", prod_val, border_color=COLOR_PRODUCTION))
    kpi_html = kpi_row(cards)

    return fig_scatter, fig_risk, fig_prod, kpi_html


# ---------------------------------------------------------------------------
# Tabs 2-3 — Main / MR Period
# ---------------------------------------------------------------------------


def load_period_table(segment: str | None, scenario: str, period: str = "main") -> tuple:
    """Return (DataFrame for display, HTML viz path or None)."""
    paths = get_scenario_paths(scenario, segment)
    csv_key = "main_csv" if period == "main" else "mr_csv"
    csv_path = paths[csv_key]

    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    # Check for interactive viz HTML
    images_dir = get_images_dir(segment)
    suffix = f"_{scenario}"
    if period == "main":
        viz_name = f"risk_production_visualizer{suffix}.html"
        fallback = "risk_production_visualizer.html"
    else:
        viz_name = f"b2_ever_h6_vs_octroi_and_risk_score_mr{suffix}.html"
        fallback = "b2_ever_h6_vs_octroi_and_risk_score_mr.html"

    viz_path = images_dir / viz_name
    if not viz_path.exists() and scenario == "base":
        viz_path = images_dir / fallback

    viz_html = None
    if viz_path.exists():
        viz_html = viz_path.read_text()

    return df, viz_html


def build_period_tab(segment: str | None, scenario: str, period: str) -> tuple:
    """Return (table_df, viz_html, kpi_md)."""
    df, viz_html = load_period_table(segment, scenario, period)
    if df.empty:
        return pd.DataFrame({"Info": ["No data available"]}), "<p>No visualization found.</p>", "No data."

    # Build KPI cards from the summary table
    cards = []
    opt = df[df["Metric"] == "Optimum selected"]
    actual = df[df["Metric"] == "Actual"]
    if not opt.empty:
        risk = opt["Risk (%)"].values[0]
        prod_pct = opt["Production (%)"].values[0]
        prod_eur = opt["Production (EUR)"].values[0] if "Production (EUR)" in opt.columns else None
        cards.append(kpi_card("Optimum Risk", f"{risk:.2f}%", border_color=COLOR_RISK))
        cards.append(kpi_card("Optimum Prod", f"{prod_pct:.1%}", border_color=COLOR_PRODUCTION))
        if prod_eur is not None and not pd.isna(prod_eur):
            cards.append(kpi_card("Prod (EUR)", f"{prod_eur:,.0f}", border_color=COLOR_PRODUCTION))
    if not actual.empty:
        act_risk = actual["Risk (%)"].values[0]
        cards.append(kpi_card("Actual Risk", f"{act_risk:.2f}%", border_color=COLOR_BAD))

    kpi_md = kpi_row(cards) if cards else ""
    viz = viz_html if viz_html else "<p>Visualization not available.</p>"
    return df, viz, kpi_md


# ---------------------------------------------------------------------------
# Tab 4 — Cutoff Explorer
# ---------------------------------------------------------------------------


def load_cutoff_data(scenario: str, segment: str | None):
    """Return (summary_data, pareto_solutions, variables, inv_vars, multiplier, optimal_mask)."""
    from src.config import PreprocessingSettings

    inv_vars: list[str] = []
    config_variables: list[str] = []
    multiplier = 7.0
    try:
        candidates = []
        if segment:
            candidates.append(OUTPUT_BASE / segment / "config_segment.toml")
        candidates.append(Path("config.toml"))
        for cp in candidates:
            if cp.exists():
                settings = PreprocessingSettings.from_toml(str(cp))
                multiplier = settings.multiplier
                inv_vars = [v for v, d in settings.directions.items() if d == -1]
                config_variables = list(settings.variables)
                break
    except Exception as e:
        logger.warning(f"Could not load config: {e}")

    paths = get_scenario_paths(scenario, segment)
    data_dir = get_data_dir(segment)

    summary_data = None
    variables: list[str] = []
    if paths["summary_data"].exists():
        summary_data = pd.read_csv(paths["summary_data"])
        if config_variables:
            variables = [v for v in config_variables if v in summary_data.columns]
        if not variables:
            for col in summary_data.columns:
                if "clus" in col.lower():
                    variables.append(col)

    pareto_solutions = None
    pareto_path = data_dir / "pareto_optimal_solutions.csv"
    if pareto_path.exists():
        pareto_solutions = pd.read_csv(pareto_path)

    optimal_mask = None
    if paths["cutoffs"].exists():
        opt_sol = pd.read_csv(paths["cutoffs"])
        if len(variables) != 2 and not opt_sol.empty and "acceptance_mask" in opt_sol.columns:
            from src.optimization_utils import decode_mask

            optimal_mask = decode_mask(str(opt_sol["acceptance_mask"].iloc[0])).tolist()

    return summary_data, pareto_solutions, variables, inv_vars, multiplier, optimal_mask


def build_cutoff_heatmap(summary_data, variables, mask_or_cuts, inv_vars, multiplier, is_nd):
    """Build heatmap figure and KPI markdown from cutoff state."""
    if summary_data is None or len(variables) < 1:
        return go.Figure(), "No cutoff data."

    is_1d = len(variables) == 1

    if is_nd or is_1d:
        metrics = _calculate_metrics_from_mask(summary_data, mask_or_cuts, variables, multiplier)
    else:
        metrics = _calculate_metrics_from_custom_cuts(
            summary_data, mask_or_cuts, variables[0], variables[1], inv_vars, multiplier
        )

    # 1D: bar chart with accept/reject coloring
    if is_1d:
        from src.optimization_utils import CellGrid

        var0 = variables[0]
        grid = CellGrid.from_summary(summary_data, variables)
        mask_arr = np.asarray(mask_or_cuts, dtype=int)
        bin_values = list(grid.values_per_var[var0])
        bar_prod = [float(grid.cell_data.iloc[grid.cell_index[(bv,)]]["oa_amt_h0"]) for bv in bin_values]
        bar_colors = ["#2ECC71" if mask_arr[grid.cell_index[(bv,)]] == 1 else "#E74C3C" for bv in bin_values]
        fig = go.Figure(go.Bar(x=[str(v) for v in bin_values], y=bar_prod, marker_color=bar_colors))
        apply_plotly_style(fig, title=f"Acceptance Strip: {var0}", height=400)
        fig.update_xaxes(title=var0)
        fig.update_yaxes(title="Production (\u20ac)")

        cards = [
            kpi_card("Optimum Risk", f"{metrics['opt_risk']:.2f}%", border_color=COLOR_RISK),
            kpi_card("Optimum Prod", f"{metrics['opt_prod_pct']:.1%}", border_color=COLOR_PRODUCTION),
            kpi_card("Swap-in Prod", f"{metrics['swap_in_prod']:,.0f}", border_color=COLOR_GOOD),
            kpi_card("Swap-out Prod", f"{metrics['swap_out_prod']:,.0f}", border_color=COLOR_BAD),
        ]
        return fig, kpi_row(cards)

    # Build heatmap (2D projection on first two variables)
    var0, var1 = variables[0], variables[1]
    vals0 = sorted(summary_data[var0].unique())
    vals1 = sorted(summary_data[var1].unique())

    heatmap_data = np.zeros((len(vals1), len(vals0)))
    for i, v1 in enumerate(vals1):
        for j, v0 in enumerate(vals0):
            cell = summary_data[(summary_data[var0] == v0) & (summary_data[var1] == v1)]
            if not cell.empty:
                den = cell["todu_amt_pile_h6"].values[0] if "todu_amt_pile_h6" in cell.columns else 0
                num = cell["todu_30ever_h6"].values[0] if "todu_30ever_h6" in cell.columns else 0
                heatmap_data[i, j] = (num / den * multiplier) if den > 0 else 0

    fig = go.Figure(
        go.Heatmap(
            z=heatmap_data,
            x=[str(v) for v in vals0],
            y=[str(v) for v in vals1],
            colorscale="RdYlGn_r",
            colorbar=dict(title="Risk (%)"),
        )
    )

    # Add accept/reject overlay
    if not is_nd and isinstance(mask_or_cuts, dict):
        for _j, v0 in enumerate(vals0):
            cutoff = mask_or_cuts.get(float(v0), mask_or_cuts.get(v0))
            if cutoff is not None:
                for _i, v1 in enumerate(vals1):
                    accepted = v1 >= cutoff if var1 in inv_vars else v1 <= cutoff
                    marker = "\u2713" if accepted else "\u2717"
                    color = "green" if accepted else "red"
                    fig.add_annotation(
                        x=str(v0), y=str(v1), text=marker, showarrow=False, font=dict(size=14, color=color)
                    )

    apply_plotly_style(fig, title=f"Risk Heatmap: {var0} vs {var1}", height=450)
    fig.update_xaxes(title=var0)
    fig.update_yaxes(title=var1)

    # KPI summary
    cards = [
        kpi_card("Optimum Risk", f"{metrics['opt_risk']:.2f}%", border_color=COLOR_RISK),
        kpi_card("Optimum Prod", f"{metrics['opt_prod_pct']:.1%}", border_color=COLOR_PRODUCTION),
        kpi_card("Swap-in Prod", f"{metrics['swap_in_prod']:,.0f}", border_color=COLOR_GOOD),
        kpi_card("Swap-out Prod", f"{metrics['swap_out_prod']:,.0f}", border_color=COLOR_BAD),
    ]
    return fig, kpi_row(cards)


def _calculate_metrics_from_custom_cuts(data, cut_map, var0_col, var1_col, inv_vars, multiplier):
    df = data.copy()
    cut_map_norm = {float(k): v for k, v in cut_map.items()}
    df["cut_limit"] = df[var0_col].astype(float).map(cut_map_norm)
    if var1_col in inv_vars:
        df["passes_cut"] = df[var1_col] >= df["cut_limit"]
    else:
        df["passes_cut"] = df[var1_col] <= df["cut_limit"]

    def calc(subset, suffix):
        prod = subset[f"oa_amt_h0{suffix}"].sum() if f"oa_amt_h0{suffix}" in subset.columns else 0
        rn = subset[f"todu_30ever_h6{suffix}"].sum() if f"todu_30ever_h6{suffix}" in subset.columns else 0
        rd = subset[f"todu_amt_pile_h6{suffix}"].sum() if f"todu_amt_pile_h6{suffix}" in subset.columns else 0
        return prod, (rn / rd * multiplier) if rd > 0 else 0, rn, rd

    ap, ar, arn, ard = calc(df, "_boo")
    sip, sir, sirn, sird = calc(df[df["passes_cut"]], "_rep")
    sop, sor, sorn, sord = calc(df[~df["passes_cut"]], "_boo")
    opt_rn, opt_rd = (arn - sorn) + sirn, (ard - sord) + sird
    opt_prod = (ap - sop) + sip
    return {
        "actual_prod": ap,
        "actual_risk": ar,
        "swap_in_prod": sip,
        "swap_in_risk": sir,
        "swap_out_prod": sop,
        "swap_out_risk": sor,
        "opt_prod": opt_prod,
        "opt_risk": (opt_rn / opt_rd * multiplier) if opt_rd > 0 else 0,
        "opt_prod_pct": opt_prod / ap if ap > 0 else 0,
    }


def _calculate_metrics_from_mask(data, mask, variables, multiplier):
    from src.optimization_utils import CellGrid, evaluate_solution

    mask_arr = np.asarray(mask, dtype=int)
    grid = CellGrid.from_summary(data, variables)
    kpis = evaluate_solution(mask_arr, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier)
    actual_prod = kpis.get("oa_amt_h0_boo", 0) + kpis.get("oa_amt_h0_cut", 0)
    opt_prod = kpis.get("oa_amt_h0", 0)
    return {
        "actual_prod": actual_prod,
        "actual_risk": kpis.get("b2_ever_h6_boo", 0),
        "swap_in_prod": kpis.get("oa_amt_h0_rep", 0),
        "swap_in_risk": kpis.get("b2_ever_h6_rep", 0),
        "swap_out_prod": kpis.get("oa_amt_h0_cut", 0),
        "swap_out_risk": kpis.get("b2_ever_h6_cut", 0),
        "opt_prod": opt_prod,
        "opt_risk": kpis.get("b2_ever_h6", 0),
        "opt_prod_pct": opt_prod / actual_prod if actual_prod > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Tab 5 — Audit Trail
# ---------------------------------------------------------------------------


def build_audit_tab(segment: str | None, scenario: str, period: str = "main") -> tuple:
    """Return (donut_fig, detail_df, kpi_html)."""
    data_dir = get_data_dir(segment)
    suffix = f"_{scenario}_mr" if period == "mr" else f"_{scenario}"
    audit_path = data_dir / f"audit{suffix}.csv"
    if not audit_path.exists():
        empty = go.Figure().update_layout(title="Audit data not found")
        return empty, pd.DataFrame(), "No audit data."

    df = pd.read_csv(audit_path)
    if "classification" not in df.columns:
        empty = go.Figure().update_layout(title="Missing classification column")
        return empty, df, "Invalid audit format."

    # Summary by classification
    summary = (
        df.groupby("classification")
        .agg(
            count=("classification", "size"),
            total_oa_amt=("oa_amt", "sum"),
        )
        .reset_index()
    )

    color_map = {
        "keep": COLOR_PRIMARY,
        "swap_in": COLOR_GOOD,
        "swap_out": COLOR_BAD,
        "rejected": COLOR_NEUTRAL,
        "rejected_other": "#7F8C8D",
    }
    colors = [color_map.get(c, COLOR_NEUTRAL) for c in summary["classification"]]

    fig = go.Figure(
        go.Pie(
            labels=summary["classification"],
            values=summary["count"],
            hole=0.5,
            marker=dict(colors=colors),
            textinfo="label+percent",
        )
    )
    apply_plotly_style(fig, title="Classification Breakdown", height=400)

    # KPI cards
    cards = []
    for _, row in summary.iterrows():
        cls = row["classification"]
        bc = color_map.get(cls, COLOR_NEUTRAL)
        cards.append(kpi_card(cls.replace("_", " ").title(), f"{row['count']:,}", border_color=bc))
    kpi_html = kpi_row(cards)

    # Detail columns
    display_cols = [
        c
        for c in ["authorization_id", "status_name", "classification", "oa_amt", "oa_amt_adjusted", "cut_limit"]
        if c in df.columns
    ]
    return fig, df[display_cols].head(500), kpi_html


# ---------------------------------------------------------------------------
# Tab 6 — Model Details
# ---------------------------------------------------------------------------


def get_latest_model(segment: str | None = None) -> dict | None:
    models_dir = get_models_dir(segment)
    if not models_dir.exists():
        return None
    for md in sorted(models_dir.glob("model_*"), reverse=True):
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


def parse_coefficients(model_dir: str) -> list[dict]:
    summary_path = Path(model_dir) / "model_summary.txt"
    if not summary_path.exists():
        return []
    coefficients = []
    in_section = False
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if "MODEL COEFFICIENTS" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("---") or not line:
                continue
            if line.isupper() and ":" not in line:
                break
            if ":" in line:
                parts = line.rsplit(":", 1)
                if len(parts) == 2:
                    try:
                        coefficients.append({"feature": parts[0].strip(), "coefficient": float(parts[1].strip())})
                    except ValueError:
                        continue
    return coefficients


def build_model_tab(segment: str | None) -> tuple:
    """Return (info_md, perf_md, coef_fig)."""
    metadata = get_latest_model(segment)
    if metadata is None:
        empty = go.Figure().update_layout(title="No model found")
        return "No trained model found. Run the pipeline first.", "", empty

    model_dir = metadata.get("_model_dir", "")

    # Info card markdown
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
            "Zero Prop.",
            f"{metadata.get('zero_proportion', 0):.1%}" if metadata.get("zero_proportion") is not None else "N/A",
        ),
        ("Weighted", str(metadata.get("weighted_regression", "N/A"))),
    ]
    info_md = " | ".join(f"**{k}:** {v}" for k, v in info_items)

    # Performance metrics
    perf_metrics = [
        ("CV Mean RMSE", metadata.get("cv_mean_rmse")),
        ("Train R\u00b2", metadata.get("train_r2", metadata.get("full_r2"))),
        ("LOO-CV R\u00b2", metadata.get("loo_r2")),
        ("Step 1 CV RMSE", metadata.get("step1_cv_rmse")),
        ("Step 2 CV RMSE", metadata.get("step2_cv_rmse")),
    ]
    perf_lines = [f"- **{k}:** {v:.4f}" for k, v in perf_metrics if v is not None]
    perf_md = "\n".join(perf_lines) if perf_lines else "No performance metrics available."

    # Coefficients chart
    coefficients = parse_coefficients(model_dir)
    coef_display = [c for c in coefficients if c["feature"].lower() != "intercept"]
    if coef_display:
        cdf = pd.DataFrame(coef_display).sort_values("coefficient")
        colors = [COLOR_PRODUCTION if v >= 0 else COLOR_RISK for v in cdf["coefficient"]]
        fig = go.Figure(go.Bar(x=cdf["coefficient"], y=cdf["feature"], orientation="h", marker_color=colors))
        apply_plotly_style(fig, title="Feature Coefficients", height=max(250, len(coef_display) * 40))
        fig.update_layout(margin=dict(l=200))
    else:
        fig = go.Figure().update_layout(title="No coefficients available")

    return info_md, perf_md, fig


# ---------------------------------------------------------------------------
# Tab 7 — Allocations (batch mode)
# ---------------------------------------------------------------------------


def build_allocations_tab(scenario: str) -> tuple:
    """Return (treemap_fig, bar_fig)."""
    report_file = OUTPUT_BASE / "consolidated_risk_production.csv"
    if not report_file.exists():
        empty = go.Figure().update_layout(title="Run run_batch.py first")
        return empty, empty

    df = pd.read_csv(report_file)
    df_filtered = df[(df["scenario"] == scenario) & (df["period"] == "main")].copy()
    if df_filtered.empty:
        empty = go.Figure().update_layout(title=f"No data for scenario '{scenario}'")
        return empty, empty

    df_segs = df_filtered[df_filtered["n_segments"] == 1].copy()
    df_segs["supersegment"] = df_segs["group"].apply(lambda x: x.split("/")[0] if "/" in x else x)
    df_segs["segment"] = df_segs["group"].apply(lambda x: x.split("/")[-1])

    if df_segs.empty:
        empty = go.Figure().update_layout(title="No segment-level data")
        return empty, empty

    tree_path = ["segment"] if (df_segs["supersegment"] == df_segs["segment"]).all() else ["supersegment", "segment"]
    fig_tree = px.treemap(
        df_segs,
        path=tree_path,
        values="optimum_production",
        color="optimum_production",
        color_continuous_scale="Viridis",
        title=f"Optimum Production Allocation ({scenario.capitalize()})",
    )
    apply_plotly_style(fig_tree, height=500)

    df_sorted = df_segs.sort_values("actual_production", ascending=False)
    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(name="Actual", x=df_sorted["segment"], y=df_sorted["actual_production"], marker_color=COLOR_PRIMARY)
    )
    fig_bar.add_trace(
        go.Bar(name="Optimum", x=df_sorted["segment"], y=df_sorted["optimum_production"], marker_color=COLOR_PRODUCTION)
    )
    fig_bar.update_layout(barmode="group")
    apply_plotly_style(fig_bar, title="Production Comparison by Segment", height=400)

    return fig_tree, fig_bar


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------


def create_app() -> gr.Blocks:
    segments = get_available_segments()
    has_segments = len(segments) > 0
    initial_segment = CURRENT_SEGMENT or (segments[0] if segments else None)
    initial_scenarios = get_scenarios(initial_segment)
    initial_scenario = initial_scenarios[0] if initial_scenarios else "base"
    has_consolidated = (OUTPUT_BASE / "consolidated_risk_production.csv").exists()

    with gr.Blocks(
        title="Risk Scoring Dashboard (Gradio)",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=".gradio-container {max-width: 1400px !important}",
    ) as app:
        gr.Markdown(
            "# Risk Scoring Optimization Dashboard\n*Explore optimization results across scenarios and time periods*"
        )

        # --- Selectors ---
        with gr.Row():
            segment_dd = gr.Dropdown(
                choices=segments if has_segments else ["(none)"],
                value=initial_segment or "(none)",
                label="Segment",
                interactive=has_segments,
            )
            scenario_dd = gr.Dropdown(
                choices=initial_scenarios or ["base"],
                value=initial_scenario,
                label="Scenario",
            )

        # Update scenarios when segment changes
        def on_segment_change(seg):
            seg = seg if seg != "(none)" else None
            scenarios = get_scenarios(seg)
            default = scenarios[0] if scenarios else "base"
            return gr.update(choices=scenarios or ["base"], value=default)

        segment_dd.change(on_segment_change, inputs=[segment_dd], outputs=[scenario_dd])

        # --- Tabs ---
        with gr.Tabs():
            # ---- Tab 1: Scenario Comparison ----
            with gr.Tab("Scenario Comparison"):
                comp_kpi = gr.HTML()
                with gr.Row():
                    comp_scatter = gr.Plot(label="Risk vs Production")
                    comp_risk = gr.Plot(label="Risk by Scenario")
                with gr.Row():
                    comp_prod = gr.Plot(label="Production by Scenario")

                def refresh_comparison(seg, _scenario):
                    seg = seg if seg != "(none)" else None
                    scatter, risk, prod, kpi = build_comparison_charts(seg)
                    return kpi, scatter, risk, prod

                # Refresh on segment or scenario change
                for trigger in [segment_dd, scenario_dd]:
                    trigger.change(
                        refresh_comparison,
                        inputs=[segment_dd, scenario_dd],
                        outputs=[comp_kpi, comp_scatter, comp_risk, comp_prod],
                    )
                # Initial load
                app.load(
                    refresh_comparison,
                    inputs=[segment_dd, scenario_dd],
                    outputs=[comp_kpi, comp_scatter, comp_risk, comp_prod],
                )

            # ---- Tab 2: Main Period ----
            with gr.Tab("Main Period"):
                main_kpi = gr.HTML()
                main_table = gr.DataFrame(label="Summary Table")
                main_viz = gr.HTML(label="Risk-Production Visualization")

                def refresh_main(seg, sc):
                    seg = seg if seg != "(none)" else None
                    df, viz, kpi = build_period_tab(seg, sc, "main")
                    return kpi, df, viz or ""

                for trigger in [segment_dd, scenario_dd]:
                    trigger.change(
                        refresh_main, inputs=[segment_dd, scenario_dd], outputs=[main_kpi, main_table, main_viz]
                    )
                app.load(refresh_main, inputs=[segment_dd, scenario_dd], outputs=[main_kpi, main_table, main_viz])

            # ---- Tab 3: MR Period ----
            with gr.Tab("MR Period"):
                mr_kpi = gr.HTML()
                mr_table = gr.DataFrame(label="MR Summary Table")
                mr_viz = gr.HTML(label="MR Risk-Production Visualization")

                def refresh_mr(seg, sc):
                    seg = seg if seg != "(none)" else None
                    df, viz, kpi = build_period_tab(seg, sc, "mr")
                    return kpi, df, viz or ""

                for trigger in [segment_dd, scenario_dd]:
                    trigger.change(refresh_mr, inputs=[segment_dd, scenario_dd], outputs=[mr_kpi, mr_table, mr_viz])
                app.load(refresh_mr, inputs=[segment_dd, scenario_dd], outputs=[mr_kpi, mr_table, mr_viz])

            # ---- Tab 4: Cutoff Explorer ----
            with gr.Tab("Cutoff Explorer"):
                cutoff_kpi = gr.HTML()
                cutoff_plot = gr.Plot(label="Risk Heatmap")
                cutoff_info = gr.Markdown("Select a segment and scenario to explore cutoffs.")

                # State for cutoff data
                cutoff_state = gr.State(value=None)

                def init_cutoff(seg, sc):
                    seg = seg if seg != "(none)" else None
                    try:
                        summary_data, pareto, variables, inv_vars, multiplier, optimal_mask = load_cutoff_data(sc, seg)
                    except Exception as e:
                        empty = go.Figure().update_layout(title=f"Error: {e}")
                        return "", empty, f"Error loading cutoff data: {e}", None

                    if summary_data is None or len(variables) < 1:
                        empty = go.Figure().update_layout(title="No cutoff data")
                        return "", empty, "Summary data not found. Run the pipeline first.", None

                    is_1d = len(variables) == 1
                    is_nd = len(variables) > 2

                    # Determine initial cuts/mask
                    if is_1d and optimal_mask is not None:
                        fig, kpi = build_cutoff_heatmap(
                            summary_data, variables, optimal_mask, inv_vars, multiplier, False
                        )
                        state = {
                            "summary_data_json": summary_data.to_json(),
                            "variables": variables,
                            "inv_vars": inv_vars,
                            "multiplier": multiplier,
                            "is_1d": True,
                            "is_nd": False,
                            "mask": optimal_mask,
                        }
                        n_acc = sum(optimal_mask)
                        info = (
                            f"**1D mode** ({variables[0]}): {n_acc}/{len(optimal_mask)} bins accepted."
                        )
                        return kpi, fig, info, state

                    elif is_nd and optimal_mask is not None:
                        fig, kpi = build_cutoff_heatmap(
                            summary_data, variables, optimal_mask, inv_vars, multiplier, True
                        )
                        state = {
                            "summary_data_json": summary_data.to_json(),
                            "variables": variables,
                            "inv_vars": inv_vars,
                            "multiplier": multiplier,
                            "is_nd": True,
                            "mask": optimal_mask,
                        }
                        info = f"**N-D mode** ({len(variables)} variables): {', '.join(variables)}. Showing optimal mask ({sum(optimal_mask)}/{len(optimal_mask)} cells accepted)."
                    else:
                        # 2-var: build initial cut_map from Pareto or uniform max
                        var0, var1 = variables[0], variables[1]
                        vals0 = sorted(summary_data[var0].unique())
                        vals1 = sorted(summary_data[var1].unique())
                        cut_map = {float(v): float(max(vals1)) for v in vals0}

                        # Try to extract optimal cuts from pareto
                        if pareto is not None and not pareto.empty:
                            bin_cols = [
                                c
                                for c in pareto.columns
                                if c not in ["b2_ever_h6", "oa_amt_h0", "sol_fac", "acceptance_mask"]
                                and c not in variables
                            ]
                            float_cols = []
                            for c in bin_cols:
                                try:
                                    float(c)
                                    float_cols.append(c)
                                except ValueError:
                                    continue
                            if float_cols:
                                # Use middle Pareto solution
                                mid_idx = len(pareto) // 2
                                for c in float_cols:
                                    cut_map[float(c)] = float(pareto.iloc[mid_idx][c])

                        fig, kpi = build_cutoff_heatmap(summary_data, variables, cut_map, inv_vars, multiplier, False)
                        state = {
                            "summary_data_json": summary_data.to_json(),
                            "variables": variables,
                            "inv_vars": inv_vars,
                            "multiplier": multiplier,
                            "is_nd": False,
                            "cut_map": cut_map,
                        }
                        info = f"**2D mode**: {var0} vs {var1}. Adjust cutoffs using the Risk Slider below."

                    return kpi, fig, info, state

                # Risk slider for exploring Pareto frontier
                risk_slider = gr.Slider(minimum=0, maximum=5, step=0.05, value=1.0, label="Target Risk (%)")

                def on_risk_change(risk_val, state_data, seg, sc):
                    if state_data is None:
                        return "", go.Figure()
                    seg = seg if seg != "(none)" else None
                    summary_data = pd.read_json(state_data["summary_data_json"])
                    variables = state_data["variables"]
                    inv_vars = state_data["inv_vars"]
                    multiplier = state_data["multiplier"]
                    is_nd = state_data["is_nd"]

                    if is_nd:
                        # Re-solve with MILP at new risk target
                        try:
                            from src.optimization_utils import CellGrid, milp_solve_cutoffs

                            grid = CellGrid.from_summary(summary_data, variables)
                            mask = milp_solve_cutoffs(grid, risk_val, inv_vars, multiplier)
                            if mask is not None:
                                fig, kpi = build_cutoff_heatmap(
                                    summary_data, variables, mask.tolist(), inv_vars, multiplier, True
                                )
                                return kpi, fig
                        except Exception as e:
                            logger.warning(f"MILP solve failed: {e}")
                        return state_data.get("kpi", ""), go.Figure().update_layout(title="Solve failed")
                    else:
                        # Find closest Pareto solution
                        data_dir = get_data_dir(seg)
                        pareto_path = data_dir / "pareto_optimal_solutions.csv"
                        if pareto_path.exists():
                            pareto = pd.read_csv(pareto_path)
                            if not pareto.empty and "b2_ever_h6" in pareto.columns:
                                idx = (pareto["b2_ever_h6"] - risk_val).abs().idxmin()
                                row = pareto.loc[idx]
                                bin_cols = [
                                    c
                                    for c in pareto.columns
                                    if c not in ["b2_ever_h6", "oa_amt_h0", "sol_fac", "acceptance_mask"]
                                ]
                                float_cols = []
                                for c in bin_cols:
                                    try:
                                        float(c)
                                        float_cols.append(c)
                                    except ValueError:
                                        continue
                                if float_cols:
                                    cut_map = {float(c): float(row[c]) for c in float_cols}
                                    fig, kpi = build_cutoff_heatmap(
                                        summary_data, variables, cut_map, inv_vars, multiplier, False
                                    )
                                    return kpi, fig
                        return "", go.Figure().update_layout(title="No Pareto data")

                risk_slider.release(
                    on_risk_change,
                    inputs=[risk_slider, cutoff_state, segment_dd, scenario_dd],
                    outputs=[cutoff_kpi, cutoff_plot],
                )

                for trigger in [segment_dd, scenario_dd]:
                    trigger.change(
                        init_cutoff,
                        inputs=[segment_dd, scenario_dd],
                        outputs=[cutoff_kpi, cutoff_plot, cutoff_info, cutoff_state],
                    )
                app.load(
                    init_cutoff,
                    inputs=[segment_dd, scenario_dd],
                    outputs=[cutoff_kpi, cutoff_plot, cutoff_info, cutoff_state],
                )

            # ---- Tab 5: Audit Trail ----
            with gr.Tab("Audit Trail"):
                audit_period_dd = gr.Dropdown(choices=["main", "mr"], value="main", label="Period")
                audit_kpi = gr.HTML()
                with gr.Row():
                    audit_donut = gr.Plot(label="Classification Breakdown")
                audit_table = gr.DataFrame(label="Audit Details")

                def refresh_audit(seg, sc, period):
                    seg = seg if seg != "(none)" else None
                    fig, df, kpi = build_audit_tab(seg, sc, period)
                    return kpi, fig, df

                for trigger in [segment_dd, scenario_dd, audit_period_dd]:
                    trigger.change(
                        refresh_audit,
                        inputs=[segment_dd, scenario_dd, audit_period_dd],
                        outputs=[audit_kpi, audit_donut, audit_table],
                    )
                app.load(
                    refresh_audit,
                    inputs=[segment_dd, scenario_dd, audit_period_dd],
                    outputs=[audit_kpi, audit_donut, audit_table],
                )

            # ---- Tab 6: Model Details ----
            with gr.Tab("Model Details"):
                model_info = gr.Markdown()
                model_perf = gr.Markdown()
                model_coef = gr.Plot(label="Feature Coefficients")

                def refresh_model(seg, _sc):
                    seg = seg if seg != "(none)" else None
                    return build_model_tab(seg)

                segment_dd.change(
                    refresh_model, inputs=[segment_dd, scenario_dd], outputs=[model_info, model_perf, model_coef]
                )
                app.load(refresh_model, inputs=[segment_dd, scenario_dd], outputs=[model_info, model_perf, model_coef])

            # ---- Tab 7: Allocations (batch only) ----
            if has_consolidated:
                with gr.Tab("Segment Allocations"):
                    alloc_tree = gr.Plot(label="Production Allocation Treemap")
                    alloc_bar = gr.Plot(label="Production Comparison")

                    def refresh_alloc(_seg, sc):
                        return build_allocations_tab(sc)

                    scenario_dd.change(refresh_alloc, inputs=[segment_dd, scenario_dd], outputs=[alloc_tree, alloc_bar])
                    app.load(refresh_alloc, inputs=[segment_dd, scenario_dd], outputs=[alloc_tree, alloc_bar])

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Risk Scoring Dashboard (Gradio)")
    parser.add_argument("--output", "-o", default="output", help="Base output directory")
    parser.add_argument("--segment", "-s", default=None, help="Initial segment to display")
    parser.add_argument("--port", "-p", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create a public link")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_BASE = Path(args.output)

    # Auto-detect single-run mode
    if not OUTPUT_BASE.exists() and (Path("data").exists() or Path("images").exists()):
        OUTPUT_BASE = Path(".")

    CURRENT_SEGMENT = args.segment

    app = create_app()
    app.launch(server_port=args.port, share=args.share)
