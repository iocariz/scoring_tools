"""Management-grade summary charts for the results presentation (matplotlib + the risk surfaces).

Builds static PNG-ready figures from the consolidated results
(``consolidated_risk_production.csv``) and the out-of-time backtest, plus 2D / 3D views of the
audit-category risk surfaces (reusing :mod:`src.risk_surface`). All chart functions return a
matplotlib ``Figure``; :func:`save_chart` writes a PNG. The 3D surface PNG uses Plotly + kaleido
and degrades gracefully (returns ``None``) when kaleido is unavailable, so the deck still builds.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, deterministic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from src import styles
from src.risk_surface import (
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    CATEGORY_ORDER,
)

_FONT = "DejaVu Sans"
plt.rcParams.update({"font.family": _FONT, "axes.titlecolor": styles.COLOR_PRIMARY, "axes.edgecolor": "#D1D5DB"})


def _seg_label(group: str) -> str:
    """'segment_no_premium_cd' -> 'no_premium_cd'; 'TOTAL' stays."""
    return group[len("segment_") :] if group.startswith("segment_") else group


def _segments_frame(consolidated: pd.DataFrame, scenario: str = "base", period: str = "main") -> pd.DataFrame:
    """Per-segment base/main rows (TOTAL excluded), sorted by optimum production desc."""
    df = consolidated[(consolidated["scenario"] == scenario) & (consolidated["period"] == period)].copy()
    df = df[df["group"] != "TOTAL"]
    df["segment"] = df["group"].map(_seg_label)
    return df.sort_values("optimum_production", ascending=False).reset_index(drop=True)


def _total_row(consolidated: pd.DataFrame, scenario: str = "base", period: str = "main") -> pd.Series | None:
    m = (consolidated["group"] == "TOTAL") & (consolidated["scenario"] == scenario) & (consolidated["period"] == period)
    return consolidated[m].iloc[0] if m.any() else None


def save_chart(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Summary charts
# --------------------------------------------------------------------------- #


def chart_production_by_segment(consolidated: pd.DataFrame) -> plt.Figure:
    """Grouped bars: current (actual) vs proposed (optimum) booked production per segment (€M)."""
    df = _segments_frame(consolidated)
    seg = df["segment"].to_numpy()
    actual = df["actual_production"].to_numpy() / 1e6
    optimum = df["optimum_production"].to_numpy() / 1e6
    x = np.arange(len(seg))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, actual, w, label="Current", color=styles.COLOR_NEUTRAL)
    ax.bar(x + w / 2, optimum, w, label="Proposed", color=styles.COLOR_PRODUCTION)
    ax.set_ylabel("Booked production (€M)")
    ax.set_title("Production by segment — current vs proposed cutoffs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seg, rotation=30, ha="right", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#EEF0F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_risk_by_segment(consolidated: pd.DataFrame) -> plt.Figure:
    """Grouped bars: current vs proposed annualised risk (b2_ever_h6 %) per segment."""
    df = _segments_frame(consolidated)
    seg = df["segment"].to_numpy()
    actual = df["actual_risk_pct"].to_numpy()
    optimum = df["optimum_risk_pct"].to_numpy()
    x = np.arange(len(seg))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, actual, w, label="Current risk", color=styles.COLOR_NEUTRAL)
    ax.bar(x + w / 2, optimum, w, label="Proposed risk", color=styles.COLOR_RISK)
    ax.set_ylabel("b2_ever_h6 (%)")
    ax.set_title("Risk by segment — current vs proposed cutoffs", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seg, rotation=30, ha="right", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#EEF0F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_swap_waterfall(consolidated: pd.DataFrame) -> plt.Figure:
    """Waterfall on the TOTAL row: current production → +swap-in − swap-out → proposed."""
    tr = _total_row(consolidated)
    if tr is None:
        df = _segments_frame(consolidated)
        actual, swin, swout, opt = (
            df["actual_production"].sum(),
            df["swap_in_production"].sum(),
            df["swap_out_production"].sum(),
            df["optimum_production"].sum(),
        )
    else:
        actual, swin, swout, opt = (
            tr["actual_production"],
            tr["swap_in_production"],
            tr["swap_out_production"],
            tr["optimum_production"],
        )
    labels = ["Current", "+ Swap-in", "− Swap-out", "Proposed"]
    vals = np.array([actual, swin, -swout, opt]) / 1e6
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cum = actual / 1e6
    # bar 0: current (absolute), bars 1-2: floating deltas, bar 3: proposed (absolute)
    ax.bar(0, vals[0], color=styles.COLOR_NEUTRAL)
    base1 = cum
    ax.bar(1, vals[1], bottom=base1, color=styles.COLOR_PRODUCTION)
    cum += vals[1]
    ax.bar(2, vals[2], bottom=cum + (0 if vals[2] >= 0 else 0), color=styles.COLOR_RISK)
    cum += vals[2]
    ax.bar(3, vals[3], color=styles.COLOR_ACCENT)
    for i, v in enumerate([vals[0], vals[1], vals[2], vals[3]]):
        ax.annotate(
            f"{v:+,.1f}" if i in (1, 2) else f"{v:,.1f}",
            (i, max(actual / 1e6, opt / 1e6) * 1.02),
            ha="center",
            fontsize=9,
            color=styles.COLOR_PRIMARY,
        )
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Booked production (€M)")
    ax.set_title("Portfolio production bridge: swap-in vs swap-out", fontweight="bold")
    ax.grid(axis="y", color="#EEF0F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_acceptance_by_segment(consolidated: pd.DataFrame) -> plt.Figure:
    """Grouped bars: current vs proposed acceptance rate (100 − rejection %) per segment."""
    df = _segments_frame(consolidated)
    seg = df["segment"].to_numpy()
    actual = 100 - df["actual_rejection_rate_pct"].to_numpy()
    optimum = 100 - df["optimum_rejection_rate_pct"].to_numpy()
    x = np.arange(len(seg))
    w = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, actual, w, label="Current", color=styles.COLOR_NEUTRAL)
    ax.bar(x + w / 2, optimum, w, label="Proposed", color=styles.COLOR_ACCENT)
    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("Acceptance rate by segment — current vs proposed", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seg, rotation=30, ha="right", fontsize=9)
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#EEF0F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def chart_backtest_validation(backtest: pd.DataFrame) -> plt.Figure:
    """Per backtested segment: predicted vs in-sample vs OOT realised risk with CIs + drift flag."""
    df = backtest.copy()

    def _ci(s):
        try:
            lo, hi = s if isinstance(s, (list, tuple)) else __import__("ast").literal_eval(str(s))
            return float(lo), float(hi)
        except Exception:
            return (np.nan, np.nan)

    seg = df["segment"].to_numpy()
    x = np.arange(len(seg))
    w = 0.25
    pred = df.get("predicted_risk_pct", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    ins = df.get("in_sample_realized_risk_pct", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    oot = df.get("oot_realized_risk_pct", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    oot_ci = [_ci(v) for v in df.get("oot_risk_ci", [None] * len(df))]
    oot_err = np.array([[o - lo, hi - o] for o, (lo, hi) in zip(oot, oot_ci)]).T
    oot_err = np.where(np.isfinite(oot_err), oot_err, 0.0)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.bar(x - w, pred, w, label="Predicted", color=styles.COLOR_SECONDARY)
    ax.bar(x, ins, w, label="In-sample realised", color=styles.COLOR_ACCENT)
    ax.bar(
        x + w,
        oot,
        w,
        label="Out-of-time realised",
        color=styles.COLOR_PRODUCTION,
        yerr=oot_err,
        capsize=4,
        error_kw=dict(ecolor=styles.COLOR_PRIMARY, lw=1),
    )
    for i, f in enumerate(df.get("drift_flag", [""] * len(df))):
        ax.annotate(
            str(f),
            (x[i], max(np.nanmax([pred, ins, oot]) * 1.05, 0.1)),
            ha="center",
            fontsize=8,
            color=styles.COLOR_PRIMARY,
        )
    ax.set_ylabel("b2_ever_h6 (%)")
    ax.set_title("Out-of-time validation — predicted vs realised risk (with CIs)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(seg, rotation=20, ha="right", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#EEF0F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Risk surfaces — 2D heatmap (matplotlib) + 3D snapshot (plotly/kaleido)
# --------------------------------------------------------------------------- #


def chart_category_heatmap(cells: pd.DataFrame, score_vars: list[str], facet_var: str | None, title: str) -> plt.Figure:
    """2D score-grid heatmap coloured by audit category, faceted per ``facet_var``; risk annotated."""
    x_var, y_var = score_vars
    all_x = sorted(cells[x_var].unique())
    all_y = sorted(cells[y_var].unique())
    facet_vals = sorted(cells[facet_var].unique()) if facet_var else [None]
    cmap = matplotlib.colors.ListedColormap([CATEGORY_COLORS[c] for c in CATEGORY_ORDER])
    code = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    fig, axes = plt.subplots(1, len(facet_vals), figsize=(5.4 * len(facet_vals), 5.0), squeeze=False)
    for ax, fv in zip(axes[0], facet_vals):
        sub = cells if fv is None else cells[cells[facet_var] == fv]
        grid = np.full((len(all_y), len(all_x)), np.nan)
        risk = np.full((len(all_y), len(all_x)), np.nan)
        for _, r in sub.iterrows():
            gi, gj = all_y.index(r[y_var]), all_x.index(r[x_var])
            grid[gi, gj] = code[r["category"]]
            risk[gi, gj] = r["b2_ever_h6"]
        ax.imshow(grid, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
        for gi in range(len(all_y)):
            for gj in range(len(all_x)):
                if np.isfinite(risk[gi, gj]):
                    ax.text(gj, gi, f"{risk[gi, gj]:.0f}", ha="center", va="center", fontsize=6.5, color="white")
        ax.set_xticks(range(len(all_x)))
        ax.set_xticklabels([f"{v:g}" for v in all_x], fontsize=8)
        ax.set_yticks(range(len(all_y)))
        ax.set_yticklabels([f"{v:g}" for v in all_y], fontsize=8)
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.set_title(f"{facet_var} = {fv:g}" if facet_var else "", fontsize=10)
    handles = [matplotlib.patches.Patch(color=CATEGORY_COLORS[c], label=CATEGORY_LABELS[c]) for c in CATEGORY_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, fontweight="bold", color=styles.COLOR_PRIMARY)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def export_surface_png(fig, path: str | Path, scale: int = 2) -> Path | None:
    """Write a Plotly figure to PNG via kaleido; return None (with a warning) if unavailable."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(path), scale=scale)
        return Path(path)
    except Exception as e:  # kaleido missing / export error — degrade to 2D-only
        logger.warning(f"3D surface PNG export skipped ({type(e).__name__}: {e}); install kaleido for 3D snapshots.")
        return None
