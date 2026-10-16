"""Charts for the Apple deck — extending the Equifax score from e-commerce to stores.

Reads the outputs of ``run_apple_study.py`` and returns matplotlib figures ready for PNG.

Design decisions (dataviz skill), in case someone reviews them:

* **One axis per panel, never a dual axis.** Risk and production/acceptance are
  magnitudes of different scale, so they go in stacked panels sharing the x axis
  instead of one chart with two scales — the alignment between two scales is
  arbitrary and invents correlations that are not in the data.
* **Validated palette**, not picked by eye: the six checks of the skill were ported
  (OKLCH lightness band, chroma floor, Machado-2009 CVD separation, normal-vision
  floor, WCAG contrast). The project palette's grey (``#BDC3C7``) **fails** as a data
  mark on white — L=0.814 out of band and 1.78:1 contrast — so the "today" series
  uses ``#7F8C8D`` (L=0.630, 3.48:1, CVD dE vs the blue 11.7). The palette's red and
  green pass the CVD gate but are never used together: risk and production never
  share a panel.
* **Selective direct labels**, never a number on every point; a legend only with two
  or more series; thin recessive grid; text wears text ink, never the series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette (validated; see docstring) ------------------------------------- #
INK = "#2C3E50"  # primary text
MUTED = "#5D6D7E"  # secondary text (5.31:1 on white)
GRID = "#E5E7EB"  # grid, one step above the surface
CURRENT = "#7F8C8D"  # "today" / current series
PROPOSED = "#3498DB"  # "proposed" series / accent
RISK = "#E74C3C"  # risk magnitude
SURFACE = "#FFFFFF"

BAR_MAX_PX = 24

# Scenario labels as the deck renders them (the study writes Spanish labels; the deck
# translates them once, on load, and every chart matches on these).
TODAY_MEASURED = "Today (measured)"
TODAY_SAME_BASE = "Today, at current store demand"
CURRENT_GRID = "Online grid in both channels"
SINGLE_CUTOFF = "single cutoff"
SHIFTED_GRID = "shifted grid"

# The two demand-based rates, chained by take-up: booking rate = system approval × take-up.
# Business calls the second one "financing rate"; the pipeline already uses "tasa de
# financiación" for booked / approved (the take-up), so the deck says "booking rate".
SYSTEM_APPROVAL = "System approval"
BOOKING_RATE = "Booking rate"


def num(value: float, decimals: int = 2, suffix: str = "") -> str:
    """English number format: decimal point, thousands comma. The deck is in English."""
    return f"{value:,.{decimals}f}{suffix}"


plt.rcParams.update(
    {
        # Arial to match the slide typography; DejaVu is matplotlib's bundled fallback so
        # the deck regenerates in Docker/CI without Arial.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "axes.grid": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def save_chart(fig: plt.Figure, path: str | Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def _style(ax, *, ylabel: str = "", xlabel: str = "", hgrid: bool = True) -> None:
    """Recessive axes: no frame, thin horizontal grid drawn below the data."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=9)
    if hgrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
        ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=6)


def _bar_width(n_slots: int, fig_width_in: float, dpi: int = 100) -> float:
    """Bar width in data units, capped at 24 px so the band breathes."""
    px_per_slot = fig_width_in * dpi / max(n_slots, 1)
    return min(0.82, BAR_MAX_PX / px_per_slot)


# --------------------------------------------------------------------------- #
# 1. The risk curve: the engine of the whole study
# --------------------------------------------------------------------------- #
def chart_risk_curve(curve: pd.DataFrame, edges: list[float]) -> plt.Figure:
    """Realized risk by EFX bin and the production behind each estimate.

    Two panels sharing the x axis instead of a dual axis: the top one answers "how
    much does the score discriminate", the bottom one "how much can I trust each
    point". The bins with residual production are shaded in both panels — that is
    where the estimate is weakest and exactly what the loosest scenarios open.
    """
    df = curve.dropna(subset=["b2_pct"]).sort_values("bin")
    bins = df["bin"].to_numpy()
    thin = df[df["booked_eur"] < 0.005 * df["booked_eur"].sum()]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.5, 5.6), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.1}
    )

    if not thin.empty:
        lo, hi = thin["bin"].min() - 0.5, thin["bin"].max() + 0.5
        for ax in (ax1, ax2):
            ax.axvspan(lo, hi, color=GRID, alpha=0.55, zorder=0)

    ax1.plot(bins, df["b2_pct"], color=RISK, linewidth=2, solid_capstyle="round", zorder=3)
    ax1.scatter(bins, df["b2_pct"], s=34, color=RISK, edgecolor=SURFACE, linewidth=2, zorder=4)
    _style(ax1, ylabel="Realized b2 risk (%)")
    ax1.set_ylim(0, df["b2_pct"].max() * 1.18)
    # labels where the cutoff is decided, not on every point
    for b in (bins[0], 8.0, 12.0, 16.0, bins[-1]):
        row = df[df["bin"] == b]
        if row.empty:
            continue
        ax1.annotate(
            num(float(row["b2_pct"].iloc[0]), 1, "%"),
            (b, float(row["b2_pct"].iloc[0])),
            textcoords="offset points",
            xytext=(0, 11),
            ha="center",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    if not thin.empty:
        ax1.annotate(
            f"bins {int(thin['bin'].min())}-{int(thin['bin'].max())}\nresidual production:\nunreliable risk",
            (thin["bin"].max() + 0.25, df["b2_pct"].max() * 0.60),
            ha="left",
            fontsize=8.5,
            color=MUTED,
        )
    ax1.set_title(
        "The EFX score ranks risk end to end in e-commerce",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )

    ax2.bar(bins, df["booked_eur"] / 1e6, 0.6, color=CURRENT, zorder=3)
    _style(ax2, ylabel="Production (€M)", xlabel="EFX bin  ·  score threshold")
    ax2.set_xticks(bins[::2])
    labels = []
    for b in bins[::2]:
        edge = edges[int(b) - 1]
        labels.append(f"{int(b)}\n{'≤' + num(edges[1], 0) if not np.isfinite(edge) else '>' + num(edge, 0)}")
    ax2.set_xticklabels(labels, fontsize=8.5)
    return fig


# --------------------------------------------------------------------------- #
# 2. Point 1 — Apple Online, measured
# --------------------------------------------------------------------------- #
def chart_channel_actual(actuals: pd.DataFrame, title: str) -> plt.Figure:
    """Realized risk by group, with the acceptance rate as a direct label (one magnitude per axis).

    Works for both channels: their current situation is measured, not imputed.
    """
    df = actuals[actuals["segmento"] != "TOTAL"].dropna(subset=["riesgo_real_pct"]).copy()
    df = df.sort_values("riesgo_real_pct")
    total = actuals[actuals["segmento"] == "TOTAL"].iloc[0]
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.barh(y, df["riesgo_real_pct"], height=0.42, color=RISK, zorder=3)
    xmax = df["riesgo_real_pct"].max() * 1.42
    for i, row in enumerate(df.itertuples()):
        # both values outside the bar: a short bar cannot hold text without clipping it
        ax.annotate(
            f"  {num(row.riesgo_real_pct, 2, '%')}",
            (row.riesgo_real_pct, i),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
        ax.annotate(f"booking {num(row.ta_pct, 1, '%')}", (xmax, i), ha="right", va="center", fontsize=9.5, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(df["segmento"], fontsize=10.5, color=INK)
    _style(ax, xlabel="Realized b2 risk (%)", hgrid=False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, xmax)
    ax.axvline(total["riesgo_real_pct"], color=INK, linewidth=1.4, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        f"channel average {num(total['riesgo_real_pct'], 2, '%')}",
        (total["riesgo_real_pct"], len(df) - 0.5),
        xytext=(7, 4),
        textcoords="offset points",
        fontsize=9,
        color=MUTED,
        annotation_clip=False,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=18)
    return fig


# --------------------------------------------------------------------------- #
# 3. Point 2 — stores: today vs the Online grid
# --------------------------------------------------------------------------- #
def chart_stores_swap(grid: pd.DataFrame, compact: bool = False) -> plt.Figure:
    """Two panels, one magnitude each: acceptance goes up and risk goes down at once.

    Colour marks **identity** — which policy — not state: both "today" bars are grey
    and both "EFX grid" bars are blue. Painting the new risk bar red would read as
    "bad" exactly when the message is that risk falls.

    ``compact`` returns a half-slide version: smaller canvas with the same type sizes,
    so the text stays legible when projected. Scaling the large version down would
    halve the fonts.
    """
    total = grid[grid["segmento"] == "TOTAL"].iloc[0]
    ta0, ta1 = total["ta_actual_pct"], total["ta_efectiva_pct"]
    r0, r1 = total["riesgo_imputado_actual_pct"], total["riesgo_imputado_pct"]
    # Verdicts are computed: a fixed text would lie as soon as the data changed sign.
    panels = [
        (f"{BOOKING_RATE} (% of € demand)", ta0, ta1, "higher booking rate" if ta1 >= ta0 else "lower booking rate"),
        ("Imputed risk (%)", r0, r1, "less risk" if r1 <= r0 else "more risk"),
    ]
    figsize = (6.8, 3.2) if compact else (10.5, 4.4)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, (title, before, after, verdict) in zip(axes, panels, strict=True):
        ax.bar([0, 1], [before, after], 0.34, color=[CURRENT, PROPOSED], zorder=3)
        for x, v in ((0, before), (1, after)):
            ax.annotate(
                num(v, 2, "%"),
                (x, v),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=12 if compact else 13,
                fontweight="bold",
                color=INK,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(
            ["Internal score\n(today)", "E-commerce\nEFX grid"], fontsize=9 if compact else 9.5, color=INK
        )
        _style(ax, ylabel=title)
        ax.set_ylim(0, max(before, after) * 1.32)
        ax.annotate(
            f"{'+' if after >= before else ''}{num(after - before, 2)} pp  ·  {verdict}",
            (0.5, max(before, after) * 1.20),
            ha="center",
            fontsize=9.5 if compact else 10.5,
            fontweight="bold",
            color=MUTED,
        )
    if not compact:
        fig.suptitle(
            "In stores, switching to the EFX score raises the booking rate and lowers risk at once",
            fontsize=13,
            fontweight="bold",
            x=0.085,
            ha="left",
            y=1.02,
            color=INK,
        )
    fig.tight_layout()
    return fig


def chart_stores_by_segment(grid: pd.DataFrame) -> plt.Figure:
    """Breakdown by segment: where the stores improvement comes from.

    Dumbbell, not grouped bars: what should be read is each segment's **change**, and
    a pair of joined dots shows it directly — the connector's length is the magnitude
    and its direction the sign. Two panels, one magnitude each, sharing the segment axis.
    """
    df = grid[(grid["segmento"] != "TOTAL") & grid["riesgo_imputado_pct"].notna()].copy()
    df = df.sort_values("demanda_eur")
    y = np.arange(len(df))
    panels = [
        (f"{BOOKING_RATE} (% of € demand)", "ta_actual_pct", "ta_efectiva_pct"),
        ("Imputed risk (%)", "riesgo_imputado_actual_pct", "riesgo_imputado_pct"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (title, col_before, col_after) in zip(axes, panels, strict=True):
        before, after = df[col_before].to_numpy(), df[col_after].to_numpy()
        ax.hlines(y, before, after, color=GRID, linewidth=3.5, zorder=2)
        ax.scatter(before, y, s=95, color=CURRENT, edgecolor=SURFACE, linewidth=2, zorder=3, label="Today")
        ax.scatter(after, y, s=95, color=PROPOSED, edgecolor=SURFACE, linewidth=2, zorder=4, label="EFX grid")
        span = max(after.max(), before.max())
        for i, value in enumerate(after):
            # only the arrival value: the starting one is given by the grey dot and the legend
            ax.annotate(
                num(value, 1, "%"),
                (value, i),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
                fontweight="bold",
                color=INK,
            )
        _style(ax, xlabel=title, hgrid=False)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xlim(0, span * 1.18)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["segmento"], fontsize=10.5, color=INK)
    axes[0].set_ylim(-0.6, len(df) - 0.4)
    legend = axes[0].legend(loc="lower right", frameon=False, fontsize=10, handletextpad=0.4)
    for text in legend.get_texts():
        text.set_color(MUTED)
    # The headline is computed: a group can lose a few tenths of acceptance (A-C does)
    # and then "every group gains" would be false. "Where it weighs most" is the group
    # with the most demand.
    gain = int((df["ta_efectiva_pct"] > df["ta_actual_pct"]).sum())
    biggest = df.iloc[-1]  # df is sorted by demand, ascending
    head = (
        "Every group's booking rate rises"
        if gain == len(df)
        else f"{gain} of {len(df)} groups raise their booking rate"
    )
    tail = (
        "risk falls where it weighs most"
        if biggest["riesgo_imputado_pct"] < biggest["riesgo_imputado_actual_pct"]
        else "risk rises where it weighs most"
    )
    fig.suptitle(
        f"{head}; {tail}",
        fontsize=13,
        fontweight="bold",
        x=0.075,
        ha="left",
        y=1.0,
        color=INK,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 4. Point 3 — the risk / acceptance frontier
# --------------------------------------------------------------------------- #
def chart_scenario_frontier(ladder: pd.DataFrame, picks: pd.DataFrame) -> plt.Figure:
    """Risk-acceptance frontier, with the current policy marked.

    The acceptance on the axis is the **effective** one (production over demand),
    the same base as point 1. The other rate the study handles — demand above the
    cutoff — is much higher and not comparable with "today"; it goes in the table,
    not here.

    The current-grid point is what turns the curve into a decision: without it one
    cannot see that all four targets are tightenings.
    """
    df = ladder.sort_values("riesgo_total_pct")
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(
        df["riesgo_total_pct"], df["ta_efectiva_pct"], color=PROPOSED, linewidth=2, solid_capstyle="round", zorder=3
    )
    ax.scatter(
        df["riesgo_total_pct"], df["ta_efectiva_pct"], s=24, color=PROPOSED, edgecolor=SURFACE, linewidth=1.6, zorder=4
    )

    marks = picks[picks["escenario"].str.contains(SINGLE_CUTOFF, na=False)].dropna(subset=["riesgo_total_pct"])
    ax.scatter(
        marks["riesgo_total_pct"],
        marks["ta_efectiva_pct"],
        s=170,
        facecolor=SURFACE,
        edgecolor=INK,
        linewidth=2.2,
        zorder=5,
    )
    for row in marks.itertuples():
        ax.annotate(
            f"target {row.escenario.split(' ·')[0]}",
            (row.riesgo_total_pct, row.ta_efectiva_pct),
            textcoords="offset points",
            xytext=(-14, 13),
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color=INK,
        )
    measured = picks[picks["escenario"] == TODAY_MEASURED]
    if not measured.empty:
        m = measured.iloc[0]
        ax.scatter(
            [m["riesgo_total_pct"]],
            [m["ta_efectiva_pct"]],
            s=200,
            color=CURRENT,
            edgecolor=SURFACE,
            linewidth=2.4,
            zorder=6,
        )
        ax.annotate(
            f"today, measured: {num(m['ta_efectiva_pct'], 1, '%')} booking rate",
            (m["riesgo_total_pct"], m["ta_efectiva_pct"]),
            textcoords="offset points",
            xytext=(14, -16),
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            color=MUTED,
        )
    base = picks[picks["escenario"].str.startswith(CURRENT_GRID)]
    if not base.empty:
        b = base.iloc[0]
        ax.scatter(
            [b["riesgo_total_pct"]],
            [b["ta_efectiva_pct"]],
            s=200,
            color=RISK,
            edgecolor=SURFACE,
            linewidth=2.4,
            zorder=6,
        )
        ax.annotate(
            f"Online grid in both channels: {num(b['ta_efectiva_pct'], 1, '%')} booking rate",
            (b["riesgo_total_pct"], b["ta_efectiva_pct"]),
            textcoords="offset points",
            xytext=(16, -5),
            ha="left",
            fontsize=11,
            fontweight="bold",
            color=RISK,
        )
    _style(ax, ylabel=f"{BOOKING_RATE} (production / demand, %)", xlabel="Global Apple risk (%)")
    # The axis is cropped to the decision zone: loosening beyond the current policy is
    # not on the table, and with the whole tail the four targets bunch up.
    if not base.empty:
        ax.set_xlim(0, float(base.iloc[0]["riesgo_total_pct"]) * 1.45)
        visible = df[df["riesgo_total_pct"] <= float(base.iloc[0]["riesgo_total_pct"]) * 1.45]
        ax.set_ylim(0, max(visible["ta_efectiva_pct"].max(), 1) * 1.3)
    else:
        ax.set_ylim(0, max(df["ta_efectiva_pct"].max(), 1) * 1.25)
    # The title is a reading of the data, so the data decides it: it only holds if the
    # current grid already sits at or under the highest target requested.
    targets = picks["escenario"].str.extract(r"^(\d+(?:\.\d+)?)%")[0].dropna().astype(float)
    top = float(targets.max()) if not targets.empty else float("nan")
    base_risk = float(base.iloc[0]["riesgo_total_pct"]) if not base.empty else float("nan")
    already_there = np.isfinite(top) and np.isfinite(base_risk) and base_risk <= top + 0.05
    ax.set_title(
        f"The {top:g}% target is already met by the current grid"
        if already_there
        else "Each risk target has its price in booking rate",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    return fig


# --------------------------------------------------------------------------- #
# 5. Seasonality — why a month is not a month
# --------------------------------------------------------------------------- #
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def chart_seasonality(index: pd.DataFrame) -> plt.Figure:
    """The two seasonal profiles, which do not look alike.

    Two series on the same axis — it is the same magnitude, a month-of-year index — so
    here they do share a chart: what must be read is precisely the divergence. Legend
    present because there are two series, and a direct label only in September, where
    they part ways.
    """
    df = index.sort_values("mes")
    months = df["mes"].to_numpy()
    has_stores = "indice_tienda" in df.columns
    fig, ax = plt.subplots(figsize=(11, 4.6))
    series = [("Online", df["indice"].to_numpy(), CURRENT)]
    if has_stores:
        series.append(("Stores", df["indice_tienda"].to_numpy(), PROPOSED))
    for label, values, color in series:
        ax.plot(months, values, color=color, linewidth=2, solid_capstyle="round", zorder=3, label=label)
        ax.scatter(months, values, s=46, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.axhline(1.0, color=INK, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate(
        "average month = 1.00",
        (5.0, 1.0),
        xytext=(0, 9),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        color=MUTED,
    )
    for _label, values, _color in series:
        ax.annotate(
            num(values[8], 2),
            (9, values[8]),
            xytext=(10, -4),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color=INK,
        )
    top = max(v.max() for _, v, _ in series)
    ax.set_xticks(months)
    ax.set_xticklabels([_MONTHS[m - 1] for m in months], fontsize=9.5)
    _style(ax, ylabel="Seasonal demand index")
    ax.set_ylim(0, top * 1.22)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10.5, handletextpad=0.5)
    for text in legend.get_texts():
        text.set_color(MUTED)
    if has_stores:
        ax.annotate(
            "September: the iPhone launch is an Online phenomenon;\nin stores it is one of the weakest months",
            (1.5, top * 0.10),
            fontsize=9.5,
            color=MUTED,
        )
    ax.set_title("The two channels do not share a seasonality", fontsize=13, fontweight="bold", loc="left", pad=14)
    return fig
