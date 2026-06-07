"""3D risk surfaces of ``b2_ever_h6`` over the two score bins, split by audit category.

For each grid cell the pipeline carries a **booked** component (observed) and a
**score-rejected / repesca** component (inferred). The accept/reject decision (the cutoff
mask) splits each into two audit categories:

| Population | Cell accepted | Cell rejected |
|:-----------|:--------------|:--------------|
| Booked     | **keep**      | **swap_out**  |
| Repesca    | **swap_in**   | **rejected**  |

This module classifies the per-cell summary into those four categories (each carrying the
risk of its own population), and renders a non-overlapping **grid** of 3D panels:
x/y = the two score bins, z = ``b2_ever_h6`` (%). Rows = audit category; columns = the third
grid variable (e.g. ``income_bin``) side-by-side, so booked (keep/swap_out) and repesca
(swap_in/rejected) risk are read in separate panels instead of stacked surfaces.

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
    """Render the audit-category risk surfaces as a non-overlapping grid of 3D panels.

    Layout: **rows = audit category**, **columns = ``facet_var``** (e.g. income bin) side-by-side.
    Each panel holds exactly one category's surface, so nothing overlaps — booked (keep/swap_out)
    and repesca (swap_in/rejected) risk are read in separate panels rather than stacked. x/y = the
    two score bins, z = ``b2_ever_h6`` (%), with a shared z-range across panels for comparability.
    """
    x_var, y_var = score_vars
    all_x = sorted(classified[x_var].unique())
    all_y = sorted(classified[y_var].unique())
    zmax = float(classified["b2_ever_h6"].max()) if not classified.empty else 1.0
    present_cats = [c for c in CATEGORY_ORDER if (classified["category"] == c).any()] or [CATEGORY_ORDER[0]]

    facet_vals: list = sorted(classified[facet_var].unique()) if facet_var else [None]
    nrows, ncols = len(present_cats), len(facet_vals)
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "surface"} for _ in range(ncols)] for _ in range(nrows)],
        column_titles=[f"{facet_var} = {v:g}" for v in facet_vals] if facet_var else None,
        row_titles=[CATEGORY_LABELS[c] for c in present_cats],
        horizontal_spacing=0.03,
        vertical_spacing=0.04,
    )

    for row, cat in enumerate(present_cats, start=1):
        color = CATEGORY_COLORS[cat]
        for col, fv in enumerate(facet_vals, start=1):
            sub = classified if fv is None else classified[classified[facet_var] == fv]
            cat_df = sub[sub["category"] == cat]
            if cat_df.empty:
                continue
            grid = cat_df.pivot_table(index=y_var, columns=x_var, values="b2_ever_h6", aggfunc="first").reindex(
                index=all_y, columns=all_x
            )
            if not np.isfinite(grid.to_numpy(dtype="float64")).any():
                continue
            fig.add_trace(
                go.Surface(
                    x=all_x,
                    y=all_y,
                    z=grid.to_numpy(dtype="float64"),
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=0.95,
                    name=CATEGORY_LABELS[cat],
                    showlegend=False,
                    connectgaps=False,
                    hovertemplate=(
                        f"{CATEGORY_LABELS[cat]}<br>{x_var}=%{{x}}<br>{y_var}=%{{y}}"
                        "<br>b2_ever_h6=%{z:.2f}%<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

    scene = dict(
        xaxis=dict(title=x_var),
        yaxis=dict(title=y_var),
        zaxis=dict(title="b2 (%)", range=[0, zmax * 1.05]),
        aspectratio=dict(x=1, y=1, z=0.7),
    )
    n_scenes = nrows * ncols
    layout_scenes = {("scene" if i == 1 else f"scene{i}"): scene for i in range(1, n_scenes + 1)}
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color="#2C3E50")),
        font=dict(family="Arial, sans-serif", size=11, color="#2C3E50"),
        width=max(720, 520 * ncols),
        height=max(420, 430 * nrows),
        margin=dict(l=10, r=70, t=80, b=10),
        **layout_scenes,
    )
    return fig
