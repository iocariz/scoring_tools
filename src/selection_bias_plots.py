"""
Plotting functions for selection bias analysis.

All functions accept matplotlib Axes or create their own figures, and follow
the project styling conventions in ``src.styles``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.styles import (
    COLOR_ACCENT,
    COLOR_BAD,
    COLOR_GOOD,
    COLOR_PRIMARY,
    COLOR_PRODUCTION,
    COLOR_RISK,
    COLOR_SECONDARY,
    apply_matplotlib_style,
)

SCORE_COLORS = {
    "Score RF": COLOR_ACCENT,
    "Risk Score RF": COLOR_RISK,
    "Combined": COLOR_PRODUCTION,
}


# ---------------------------------------------------------------------------
# 1. Distribution truncation — KDE overlay
# ---------------------------------------------------------------------------
def plot_distribution_truncation(
    df_demand: pd.DataFrame,
    score_col: str,
    score_name: str,
    segment_name: str,
    *,
    score_negate: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Overlaid KDE of demand vs booked vs rejected score distributions."""
    sign = -1 if score_negate else 1

    demand = sign * df_demand[score_col].dropna()
    booked_mask = df_demand["status_name"] == "booked"
    booked = sign * df_demand.loc[booked_mask, score_col].dropna()
    rejected = sign * df_demand.loc[~booked_mask, score_col].dropna()

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if len(demand) > 1:
        demand.plot.kde(ax=ax, label=f"Demand (n={len(demand):,})", color=COLOR_PRIMARY, lw=2)
    if len(booked) > 1:
        booked.plot.kde(ax=ax, label=f"Booked (n={len(booked):,})", color=COLOR_GOOD, lw=2)
    if len(rejected) > 1:
        rejected.plot.kde(ax=ax, label=f"Rejected (n={len(rejected):,})", color=COLOR_BAD, lw=2, linestyle="--")

    sd_demand = float(demand.std()) if len(demand) > 1 else 0
    sd_booked = float(booked.std()) if len(booked) > 1 else 0
    u = sd_booked / sd_demand if sd_demand > 0 else 1.0

    ax.set_title(f"{segment_name} — {score_name}\nu = SD_booked/SD_demand = {u:.3f}", fontsize=13)
    ax.set_xlabel(score_name)
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    return ax


def plot_all_truncation_panels(
    df_demand: pd.DataFrame,
    score_columns: dict[str, dict],
    segment_name: str,
    output_dir: Path,
) -> Path:
    """One figure with one row per score showing demand/booked/rejected KDEs."""
    apply_matplotlib_style()
    n_scores = len(score_columns)
    fig, axes = plt.subplots(n_scores, 1, figsize=(12, 5 * n_scores), squeeze=False)

    for i, (score_name, info) in enumerate(score_columns.items()):
        plot_distribution_truncation(
            df_demand,
            info["column"],
            score_name,
            segment_name,
            score_negate=info.get("negate", False),
            ax=axes[i, 0],
        )

    fig.suptitle(
        f"Score Distribution Truncation: {segment_name}",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = output_dir / f"truncation_{segment_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Truncation plot saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 2. Thorndike correction — bar chart
# ---------------------------------------------------------------------------
def plot_thorndike_correction(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Grouped bar chart: observed vs corrected Gini per segment/score."""
    if df.empty:
        return None

    apply_matplotlib_style()

    df = df.copy()
    df["label"] = df["segment"] + " / " + df["score"]

    labels = df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(labels)), 6))
    bars_obs = ax.bar(x - width / 2, df["observed_gini"], width, label="Observed Gini", color=COLOR_SECONDARY)
    bars_corr = ax.bar(
        x + width / 2, df["corrected_gini"], width, label="Corrected Gini (Thorndike)", color=COLOR_ACCENT
    )

    for bar, val in zip(bars_obs, df["observed_gini"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val, att in zip(bars_corr, df["corrected_gini"], df["attenuation_pct"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}\n({att:.0f}% att.)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Gini")
    ax.set_title("Observed vs Thorndike-Corrected Gini", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0, color="black", lw=0.5)

    fig.tight_layout()
    fig_path = output_dir / "thorndike_correction.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Thorndike correction plot saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 3. Simulated range restriction — sensitivity curves
# ---------------------------------------------------------------------------
def plot_simulated_restriction(df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Line chart: Gini vs % of booked population retained, per segment."""
    if df.empty:
        return None

    apply_matplotlib_style()

    groups = df.groupby("segment")
    n_groups = len(groups)
    ncols = min(n_groups, 3)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, (seg_name, gdf) in enumerate(groups):
        ax = axes[idx // ncols, idx % ncols]
        for score_name in gdf["score"].unique():
            sdf = gdf[gdf["score"] == score_name].sort_values("pct_retained", ascending=False)
            color = SCORE_COLORS.get(score_name, COLOR_SECONDARY)
            ax.plot(
                sdf["pct_retained"] * 100,
                sdf["gini"],
                marker="o",
                markersize=5,
                lw=2,
                label=score_name,
                color=color,
            )

        ax.set_title(seg_name, fontsize=13)
        ax.set_xlabel("% of Booked Population Retained")
        ax.set_ylabel("Gini")
        ax.legend(fontsize=9)
        ax.grid(True, linestyle="--", lw=0.5)
        ax.invert_xaxis()

    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        "Simulated Range Restriction: Gini Sensitivity",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = output_dir / "simulated_restriction.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Simulated restriction plot saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 4. Temporal acceptance rate vs Gini — scatter with regression
# ---------------------------------------------------------------------------
def plot_temporal_acceptance_vs_gini(
    rolling_gini: pd.DataFrame,
    rolling_acceptance: pd.DataFrame,
    output_dir: Path,
) -> Path | None:
    """Within-segment scatter of rolling acceptance rate vs rolling Gini."""
    if rolling_gini.empty or rolling_acceptance.empty:
        return None

    apply_matplotlib_style()

    # Merge on (segment, period, month)
    merged = rolling_gini.merge(
        rolling_acceptance[["segment", "period", "month", "acceptance_rate"]],
        on=["segment", "period", "month"],
        how="inner",
    )

    if merged.empty:
        return None

    groups = merged.groupby(["segment", "score"])
    n_groups = len(groups)
    if n_groups == 0:
        return None

    ncols = min(n_groups, 3)
    nrows = (n_groups + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), squeeze=False)

    for idx, ((seg, sc), gdf) in enumerate(groups):
        ax = axes[idx // ncols, idx % ncols]
        x = gdf["acceptance_rate"].values
        y = gdf["gini"].values
        color = SCORE_COLORS.get(sc, COLOR_ACCENT)

        ax.scatter(x, y, color=color, s=40, zorder=3)

        if len(x) >= 3:
            slope, intercept, r_val, p_val, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_line, slope * x_line + intercept, color=color, lw=1.5, linestyle="--")
            ax.text(
                0.05,
                0.95,
                f"r={r_val:.3f}  p={p_val:.3f}",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_title(f"{seg} — {sc}", fontsize=12)
        ax.set_xlabel("Acceptance Rate (rolling 3m)")
        ax.set_ylabel("Gini (rolling 3m)")
        ax.grid(True, linestyle="--", lw=0.5)

    for idx in range(n_groups, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(
        "Temporal: Rolling Acceptance Rate vs Rolling Gini",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.02,
    )
    fig.tight_layout()
    fig_path = output_dir / "temporal_acceptance_vs_gini.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Temporal acceptance vs Gini plot saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 5. Rejection profile — stacked bar by decile
# ---------------------------------------------------------------------------
def plot_rejection_profile(
    profiles: dict[str, pd.DataFrame],
    segment_name: str,
    output_dir: Path,
) -> Path | None:
    """Stacked bar chart of booked vs rejected by score decile."""
    non_empty = {k: v for k, v in profiles.items() if not v.empty}
    if not non_empty:
        return None

    apply_matplotlib_style()

    n_scores = len(non_empty)
    fig, axes = plt.subplots(1, n_scores, figsize=(7 * n_scores, 5), squeeze=False)

    for i, (score_name, prof) in enumerate(non_empty.items()):
        ax = axes[0, i]
        x = prof["decile"].values
        ax.bar(x, prof["n_booked"], label="Booked", color=COLOR_GOOD)
        ax.bar(x, prof["n_rejected"], bottom=prof["n_booked"], label="Rejected", color=COLOR_BAD)

        ax2 = ax.twinx()
        ax2.plot(
            x, prof["rejection_rate"] * 100, color=COLOR_PRIMARY, marker="D", markersize=5, lw=2, label="Rejection %"
        )
        ax2.set_ylabel("Rejection Rate (%)", color=COLOR_PRIMARY)
        ax2.set_ylim(0, 105)

        ax.set_title(f"{score_name}", fontsize=13)
        ax.set_xlabel("Score Decile (1 = safest)")
        ax.set_ylabel("Count")
        ax.legend(loc="upper left", fontsize=9)
        ax2.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Rejection Profile by Score Decile: {segment_name}",
        fontsize=16,
        fontweight="bold",
        color=COLOR_PRIMARY,
        y=1.04,
    )
    fig.tight_layout()
    fig_path = output_dir / f"rejection_profile_{segment_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Rejection profile plot saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# 6. Cross-segment acceptance vs Gini — scatter
# ---------------------------------------------------------------------------
def plot_cross_segment_scatter(
    segment_data: list[dict],
    regression: dict,
    output_dir: Path,
) -> Path | None:
    """Scatter of acceptance rate vs Gini across segments."""
    if not segment_data or regression.get("slope") is None:
        return None

    apply_matplotlib_style()

    df = pd.DataFrame(segment_data)
    fig, ax = plt.subplots(figsize=(10, 6))

    for score_name in df["score"].unique():
        sdf = df[df["score"] == score_name]
        color = SCORE_COLORS.get(score_name, COLOR_ACCENT)
        ax.scatter(sdf["acceptance_rate"], sdf["gini"], color=color, s=60, label=score_name, zorder=3)

        # Annotate with segment name
        for _, row in sdf.iterrows():
            ax.annotate(
                row["segment"],
                (row["acceptance_rate"], row["gini"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                color=COLOR_SECONDARY,
            )

    # Regression line (pooled)
    x = df["acceptance_rate"].values
    if regression["slope"] is not None:
        x_line = np.linspace(x.min(), x.max(), 50)
        y_line = regression["slope"] * x_line + regression["intercept"]
        ax.plot(x_line, y_line, color=COLOR_PRIMARY, lw=1.5, linestyle="--", label="OLS fit")
        ax.text(
            0.05,
            0.05,
            f"r={regression['r_value']:.3f}  p={regression['p_value']:.3f}  n={regression['n_segments']}",
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_xlabel("Acceptance Rate")
    ax.set_ylabel("Observed Gini")
    ax.set_title("Cross-Segment: Acceptance Rate vs Gini", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", lw=0.5)

    fig.tight_layout()
    fig_path = output_dir / "cross_segment_acceptance_vs_gini.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Cross-segment scatter saved to {fig_path}")
    return fig_path
