"""One continuous 3D risk surface of ``b2_ever_h6`` over the two score bins, coloured by category.

Each grid cell maps to exactly one **audit category** from (status before × decision now):

| Was booked before? | Accepted now | Rejected now |
|:-------------------|:-------------|:-------------|
| **Yes** (had booked exposure) | **keep**      | **swap_out** |
| **No** (repesca-only / rejected before) | **swap_in** | **rejected** |

"Booked before" = the cell had booked exposure (it was being booked under the current policy);
"accepted now" = the cell is in the proposed cutoff mask. Because both are cell-level facts, a
single segment partitions cleanly — **one category per cell, no overlap**. (A supersegment can
have members disagree on a cell; it is resolved to one category per cell by exposure-weighted
dominance, so the surface stays single-valued.)

The figure is **one continuous surface** per facet — z = ``b2_ever_h6`` (%), coloured discretely
by category — faceted side-by-side, one per third grid variable (e.g. ``income_bin``).

Pure / IO-light: ``plot_risk_surface.py`` handles discovery and file IO.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import calculate_b2_ever_h6

# Audit categories: code (colour-band order), colour, legend label.
CATEGORY_ORDER = ["keep", "swap_in", "swap_out", "rejected"]
CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORY_ORDER)}
CATEGORY_COLORS = {
    "keep": "#2ECC71",  # green — booked before, accepted now
    "swap_in": "#3498DB",  # blue — rejected before, accepted now
    "swap_out": "#F39C12",  # amber — booked before, rejected now
    "rejected": "#E74C3C",  # red — rejected before and now
}
CATEGORY_LABELS = {
    "keep": "Keep (booked → accept)",
    "swap_in": "Swap-in (rejected → accept)",
    "swap_out": "Swap-out (booked → reject)",
    "rejected": "Rejected (rejected → reject)",
}

Cell = tuple[float, ...]


def _col(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[name].to_numpy(dtype="float64") if name in df.columns else np.zeros(len(df))


def classify_cells(summary: pd.DataFrame, accepted_set: set[Cell], variables: list[str]) -> pd.DataFrame:
    """Per-cell components for one segment: combined risk numerator/denominator + before/now weights.

    Returns one row per cell with ``variables + [t30, pile, boo_mass, acc_wt, rej_wt]``:
    ``t30``/``pile`` are the combined (booked+repesca) risk numerator/denominator, ``boo_mass`` is
    booked exposure (→ "booked before"), and ``acc_wt``/``rej_wt`` carry the cell's exposure on the
    accept / reject side of the proposed cutoff (so members combine by exposure for supersegments).
    """
    df = summary.copy()
    coords = list(zip(*[df[v].astype("float64") for v in variables], strict=True))
    new_accept = np.array([c in accepted_set for c in coords])
    pile = _col(df, "todu_amt_pile_h6")
    weight = np.abs(pile)
    out = {v: df[v].to_numpy() for v in variables}
    out["t30"] = _col(df, "todu_30ever_h6")
    out["pile"] = pile
    out["boo_mass"] = _col(df, "todu_amt_pile_h6_boo")
    out["acc_wt"] = np.where(new_accept, weight, 0.0)
    out["rej_wt"] = np.where(new_accept, 0.0, weight)
    return pd.DataFrame(out)


def aggregate_to_cells(long_df: pd.DataFrame, variables: list[str], multiplier: float = 7.0) -> pd.DataFrame:
    """Collapse to **one row per cell**: combined ``b2_ever_h6`` (%) + a single ``category``.

    Sums components across rows (so a supersegment's members combine), drops empty cells, then
    derives the category from (booked-before × accepted-now). "Accepted now" is exposure-weighted
    across members (``acc_wt ≥ rej_wt``) — for a single segment that is exactly the cell's own
    accept decision, so the result is one unambiguous category per cell.
    """
    g = (
        long_df.groupby([*variables], as_index=False)
        .agg(
            t30=("t30", "sum"),
            pile=("pile", "sum"),
            boo_mass=("boo_mass", "sum"),
            acc_wt=("acc_wt", "sum"),
            rej_wt=("rej_wt", "sum"),
        )
        .loc[lambda d: d["pile"] != 0]
        .reset_index(drop=True)
    )
    g["b2_ever_h6"] = calculate_b2_ever_h6(
        g["t30"].to_numpy(dtype="float64"),
        g["pile"].to_numpy(dtype="float64"),
        multiplier=multiplier,
        as_percentage=True,
    )
    old_accept = g["boo_mass"].to_numpy() > 0
    new_accept = g["acc_wt"].to_numpy() >= g["rej_wt"].to_numpy()
    g["category"] = np.select(
        [old_accept & new_accept, old_accept & ~new_accept, ~old_accept & new_accept],
        ["keep", "swap_out", "swap_in"],
        default="rejected",
    )
    return g[[*variables, "b2_ever_h6", "category"]]


def _discrete_colorscale() -> list:
    """A 4-band step colorscale (code 0..3 → keep/swap_in/swap_out/rejected) for cmin=-0.5,cmax=3.5."""
    scale = []
    for i, cat in enumerate(CATEGORY_ORDER):
        color = CATEGORY_COLORS[cat]
        scale += [[i / 4, color], [(i + 1) / 4, color]]
    return scale


def build_risk_surface_figure(
    cells: pd.DataFrame,
    score_vars: list[str],
    facet_var: str | None,
    title: str,
) -> go.Figure:
    """One continuous ``b2_ever_h6`` surface over the two score bins, coloured by audit category.

    x/y = the two score bins, z = ``b2_ever_h6`` (%); each cell is coloured by its single category
    via a discrete colour bar. Faceted side-by-side (one 3D scene per ``facet_var`` value), with a
    shared z-range for comparability.
    """
    x_var, y_var = score_vars
    all_x = sorted(cells[x_var].unique())
    all_y = sorted(cells[y_var].unique())
    zmax = float(cells["b2_ever_h6"].max()) if not cells.empty else 1.0
    facet_vals: list = sorted(cells[facet_var].unique()) if facet_var else [None]
    n = len(facet_vals)
    colorscale = _discrete_colorscale()

    fig = make_subplots(
        rows=1,
        cols=n,
        specs=[[{"type": "surface"} for _ in facet_vals]],
        subplot_titles=[f"{facet_var} = {v:g}" for v in facet_vals] if facet_var else None,
        horizontal_spacing=0.02,
    )

    def _grid(sub: pd.DataFrame, col: str, fill=np.nan):
        return (
            sub.pivot_table(index=y_var, columns=x_var, values=col, aggfunc="first")
            .reindex(index=all_y, columns=all_x)
            .to_numpy()
        )

    for col, fv in enumerate(facet_vals, start=1):
        sub = cells if fv is None else cells[cells[facet_var] == fv]
        z = _grid(sub, "b2_ever_h6").astype("float64")
        code = _grid(sub.assign(_code=sub["category"].map(CATEGORY_CODE)), "_code").astype("float64")
        label = _grid(sub, "category")
        fig.add_trace(
            go.Surface(
                x=all_x,
                y=all_y,
                z=z,
                surfacecolor=code,
                cmin=-0.5,
                cmax=3.5,
                colorscale=colorscale,
                connectgaps=False,
                customdata=label,
                showscale=(col == 1),
                colorbar=dict(
                    tickvals=[0, 1, 2, 3],
                    ticktext=[CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
                    len=0.55,
                    y=0.5,
                    x=1.0,
                )
                if col == 1
                else None,
                hovertemplate=(
                    f"{x_var}=%{{x}}<br>{y_var}=%{{y}}<br>b2_ever_h6=%{{z:.2f}}%<br>%{{customdata}}<extra></extra>"
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
        width=max(760, 600 * n),
        height=720,
        margin=dict(l=10, r=10, t=70, b=10),
        **layout_scenes,
    )
    return fig
