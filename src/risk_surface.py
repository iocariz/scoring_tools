"""3D risk surfaces of ``b2_ever_h6`` over the two score bins, split by audit category.

For each grid cell the pipeline carries a **booked** component (observed) and a
**score-rejected / repesca** component (inferred). The accept/reject decision (the cutoff
mask) splits each into two audit categories:

| Population | Cell accepted | Cell rejected |
|:-----------|:--------------|:--------------|
| Booked     | **keep**      | **swap_out**  |
| Repesca    | **swap_in**   | **rejected**  |

This module classifies the per-cell summary into those four categories (each carrying the
risk of its own population), and renders a 3D figure: x/y = the two score bins,
z = ``b2_ever_h6`` (%), one coloured surface per category. When a third grid variable
(e.g. ``income_bin``) is present, the figure is faceted side-by-side, one 3D scene per value.

Pure / IO-light: ``plot_risk_surface.py`` handles discovery and file IO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import calculate_b2_ever_h6

# Audit categories and their fixed colours (legend order = drawing order).
CATEGORY_ORDER = ["keep", "swap_in", "swap_out", "rejected"]
CATEGORY_COLORS = {
    "keep": "#2ECC71",  # green — booked, accepted
    "swap_in": "#3498DB",  # blue — repesca, accepted
    "swap_out": "#F39C12",  # amber — booked, rejected
    "rejected": "#E74C3C",  # red — repesca, rejected
}
CATEGORY_LABELS = {
    "keep": "Keep (booked · accepted)",
    "swap_in": "Swap-in (repesca · accepted)",
    "swap_out": "Swap-out (booked · rejected)",
    "rejected": "Rejected (repesca · rejected)",
}

Cell = tuple[float, ...]


def classify_cells(
    summary: pd.DataFrame,
    accepted_set: set[Cell],
    variables: list[str],
) -> pd.DataFrame:
    """Explode the per-cell summary into a long booked+repesca table tagged by audit category.

    Returns one row per (cell, population) with columns ``variables + [category, t30, pile,
    exposure]``. ``category`` is keep/swap_out for the booked component and swap_in/rejected
    for the repesca component, per the cell's membership in ``accepted_set``.
    """
    df = summary.copy()
    coords = list(zip(*[df[v].astype("float64") for v in variables], strict=True))
    accepted = np.array([c in accepted_set for c in coords])

    def _component(suffix: str, accepted_cat: str, rejected_cat: str) -> pd.DataFrame:
        part = {v: df[v].to_numpy() for v in variables}
        part["category"] = np.where(accepted, accepted_cat, rejected_cat)
        part["t30"] = df.get(f"todu_30ever_h6_{suffix}", 0.0)
        part["pile"] = df.get(f"todu_amt_pile_h6_{suffix}", 0.0)
        part["exposure"] = df.get(f"acct_booked_h0_{suffix}", 0.0)
        return pd.DataFrame(part)

    booked = _component("boo", "keep", "swap_out")
    repesca = _component("rep", "swap_in", "rejected")
    return pd.concat([booked, repesca], ignore_index=True)


def aggregate_classified(
    long_df: pd.DataFrame,
    variables: list[str],
    multiplier: float = 7.0,
) -> pd.DataFrame:
    """Aggregate the long table to one risk value per (cell, category).

    Sums the exposure/numerator across rows (so several segments of a supersegment combine),
    drops cells with non-positive exposure, and computes ``b2_ever_h6`` (%, clipped at 0).
    """
    g = (
        long_df.groupby([*variables, "category"], as_index=False)
        .agg(t30=("t30", "sum"), pile=("pile", "sum"), exposure=("exposure", "sum"))
        .loc[lambda d: d["pile"] > 0]
        .reset_index(drop=True)
    )
    g["b2_ever_h6"] = calculate_b2_ever_h6(
        g["t30"].to_numpy(dtype="float64"),
        g["pile"].to_numpy(dtype="float64"),
        multiplier=multiplier,
        as_percentage=True,
    )
    return g


def build_risk_surface_figure(
    classified: pd.DataFrame,
    score_vars: list[str],
    facet_var: str | None,
    title: str,
) -> go.Figure:
    """Render the 4-category 3D risk surfaces, faceted side-by-side by ``facet_var`` if given.

    ``score_vars`` = the two score-bin columns (x, y); z = ``b2_ever_h6`` (%). One surface per
    audit category; a category surface only "lights up" the cells it owns (NaN elsewhere).
    """
    x_var, y_var = score_vars
    all_x = sorted(classified[x_var].unique())
    all_y = sorted(classified[y_var].unique())
    zmax = float(classified["b2_ever_h6"].max()) if not classified.empty else 1.0

    facet_vals: list = sorted(classified[facet_var].unique()) if facet_var else [None]
    n = len(facet_vals)
    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "surface"} for _ in facet_vals]],
        subplot_titles=[f"{facet_var} = {v:g}" for v in facet_vals] if facet_var else None,
        horizontal_spacing=0.01,
    )

    for col, fv in enumerate(facet_vals, start=1):
        sub = classified if fv is None else classified[classified[facet_var] == fv]
        for cat in CATEGORY_ORDER:
            cat_df = sub[sub["category"] == cat]
            grid = (
                cat_df.pivot_table(index=y_var, columns=x_var, values="b2_ever_h6", aggfunc="first").reindex(
                    index=all_y, columns=all_x
                )
                if not cat_df.empty
                else None
            )
            if grid is None or not np.isfinite(grid.to_numpy(dtype="float64")).any():
                continue
            color = CATEGORY_COLORS[cat]
            fig.add_trace(
                go.Surface(
                    x=all_x,
                    y=all_y,
                    z=grid.to_numpy(dtype="float64"),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.85,
                    name=CATEGORY_LABELS[cat],
                    legendgroup=cat,
                    showlegend=(col == 1),
                    connectgaps=False,
                    hovertemplate=(
                        f"{CATEGORY_LABELS[cat]}<br>{x_var}=%{{x}}<br>{y_var}=%{{y}}"
                        "<br>b2_ever_h6=%{z:.2f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

    scene = dict(
        xaxis=dict(title=x_var),
        yaxis=dict(title=y_var),
        zaxis=dict(title="b2_ever_h6 (%)", range=[0, zmax * 1.05]),
        aspectratio=dict(x=1, y=1, z=0.8),
    )
    layout_scenes = {("scene" if i == 1 else f"scene{i}"): scene for i in range(1, n + 1)}
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#2C3E50")),
        font=dict(family="Arial, sans-serif", size=12, color="#2C3E50"),
        width=max(700, 560 * n),
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=10, r=10, t=70, b=10),
        **layout_scenes,
    )
    return fig
