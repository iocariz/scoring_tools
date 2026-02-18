"""
Full score monitoring report: discriminance, lift, precision-recall,
score distributions, and pairwise model comparison (DeLong).

Evaluates score_rf, risk_score_rf, and their logistic-regression combination
against the early_bad target, per segment, per supersegment, and for both
Main and MR periods.

Usage:
    uv run python run_score_metrics.py                             # All segments
    uv run python run_score_metrics.py --segments no_premium_ab    # Specific
    uv run python run_score_metrics.py --output output             # Custom dir
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

from run_batch import (
    load_and_standardize_data,
    load_base_config,
    load_segments_config,
    load_supersegments_config,
)
from src.metrics import (
    calculate_lift_table,
    compute_precision_recall,
    compute_score_discriminance,
    delong_test,
)
from src.plots import plot_precision_recall_curve, plot_score_distribution
from src.styles import (
    COLOR_ACCENT,
    COLOR_PRIMARY,
    COLOR_PRODUCTION,
    COLOR_RISK,
    COLOR_SECONDARY,
    apply_matplotlib_style,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_COLUMN = "early_bad"

SCORE_COLUMNS: dict[str, dict] = {
    "Score RF": {"column": "score_rf", "negate": True},
    "Risk Score RF": {"column": "risk_score_rf", "negate": True},
}

COMBINED_COLUMNS: dict[str, list] = {
    "Combined": ["score_rf", "risk_score_rf"],
}

SCORE_COLORS = {
    "Score RF": COLOR_ACCENT,
    "Risk Score RF": COLOR_RISK,
    "Combined": COLOR_PRODUCTION,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _filter_base_population(df: pd.DataFrame, segment_filter: str) -> pd.DataFrame:
    """Apply standard exclusion filters and segment match (same as preprocess_data)."""
    filtered = df.query("fuera_norma == 'n' & fraud_flag == 'n' & nature_holder != 'legal'").copy()
    if "|" in segment_filter:
        mask = filtered["segment_cut_off"].astype(str).str.match(segment_filter, na=False)
        return filtered[mask]
    return filtered[filtered["segment_cut_off"] == segment_filter]


def _filter_booked_period(df: pd.DataFrame, date_ini: str, date_fin: str) -> pd.DataFrame:
    """Filter to booked status within a date range."""
    booked = df[df["status_name"] == "booked"].copy()
    if date_ini and date_fin:
        booked["mis_date"] = pd.to_datetime(booked["mis_date"])
        start = pd.to_datetime(date_ini)
        end = pd.to_datetime(date_fin)
        booked = booked[(booked["mis_date"] >= start) & (booked["mis_date"] <= end)]
    return booked


def _validate_target(df: pd.DataFrame, label: str) -> bool:
    """Check that early_bad exists and has both classes."""
    if TARGET_COLUMN not in df.columns:
        warnings.warn(f"[{label}] Column '{TARGET_COLUMN}' not found — skipping.", stacklevel=2)
        return False
    nunique = df[TARGET_COLUMN].dropna().nunique()
    if nunique < 2:
        warnings.warn(f"[{label}] '{TARGET_COLUMN}' has {nunique} class(es) — skipping.", stacklevel=2)
        return False
    return True


def _prepare_scores(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Build negated score arrays from the dataframe, matching SCORE_COLUMNS config."""
    scores = {}
    for name, info in SCORE_COLUMNS.items():
        sign = -1 if info["negate"] else 1
        scores[name] = sign * df[info["column"]].values
    return scores


def _compute_for_period(
    df_base: pd.DataFrame,
    date_ini: str | None,
    date_fin: str | None,
    period: str,
    level: str,
    name: str,
    supersegment: str,
) -> dict | None:
    """
    Compute all monitoring metrics for a single period.

    Returns a dict with: discriminance_df, lift_tables, pr_data,
    delong_results, scores_dict, y_true, and metadata tags.
    """
    if not date_ini or not date_fin:
        return None

    df_booked = _filter_booked_period(df_base, date_ini, date_fin)
    label = f"{level}/{name}/{period}"

    if df_booked.empty:
        logger.warning(f"[{label}] No booked records in date range — skipping.")
        return None

    if not _validate_target(df_booked, label):
        return None

    required_cols = [TARGET_COLUMN] + [s["column"] for s in SCORE_COLUMNS.values()]
    df_booked = df_booked.dropna(subset=required_cols)

    if df_booked.empty or df_booked[TARGET_COLUMN].nunique() < 2:
        logger.warning(f"[{label}] Not enough data after dropping NaNs — skipping.")
        return None

    y_true = df_booked[TARGET_COLUMN].values
    scores_dict = _prepare_scores(df_booked)

    # 1. Discriminance (AUROC, Gini, KS)
    metrics_df = compute_score_discriminance(
        df_booked,
        target_column=TARGET_COLUMN,
        score_columns=SCORE_COLUMNS,
        combined_columns=COMBINED_COLUMNS,
    )
    metrics_df["level"] = level
    metrics_df["name"] = name
    metrics_df["supersegment"] = supersegment
    metrics_df["period"] = period

    # 2. Lift tables per score
    lift_tables = {}
    for score_name, score_arr in scores_dict.items():
        lift_tables[score_name] = calculate_lift_table(y_true, score_arr)

    # 3. Precision-Recall per score
    pr_data = {}
    for score_name, score_arr in scores_dict.items():
        precision, recall, _, ap = compute_precision_recall(y_true, score_arr)
        pr_data[score_name] = {"precision": precision, "recall": recall, "ap": ap}

    # 4. DeLong pairwise tests
    delong_results = []
    score_names = list(scores_dict.keys())
    for i in range(len(score_names)):
        for j in range(i + 1, len(score_names)):
            n1, n2 = score_names[i], score_names[j]
            result = delong_test(y_true, scores_dict[n1], scores_dict[n2])
            result["model_1"] = n1
            result["model_2"] = n2
            result["level"] = level
            result["name"] = name
            result["period"] = period
            delong_results.append(result)

    return {
        "discriminance_df": metrics_df,
        "lift_tables": lift_tables,
        "pr_data": pr_data,
        "delong_results": delong_results,
        "scores_dict": scores_dict,
        "y_true": y_true,
        "label": label,
        "level": level,
        "name": name,
        "period": period,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
METRIC_NAMES = ["auroc", "gini", "ks"]
METRIC_LABELS = {"auroc": "AUROC", "gini": "Gini", "ks": "KS"}


def plot_score_discriminance(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Create a grouped-bar chart of AUROC / Gini / KS per segment and period.

    One subplot per metric.  Within each subplot the x-axis shows
    ``name (period)`` and bars are grouped by score.

    Args:
        df: The results DataFrame produced by ``generate_score_discriminance_report``.
        output_dir: Directory where the PNG is saved.

    Returns:
        Path to the saved figure.
    """
    apply_matplotlib_style()

    df = df.copy()
    df["x_label"] = df.apply(
        lambda r: (
            f"[SS] {r['name']} ({r['period']})" if r["level"] == "supersegment" else f"{r['name']} ({r['period']})"
        ),
        axis=1,
    )

    label_order = (
        df.sort_values(["level", "name", "period"], ascending=[False, True, True])["x_label"].drop_duplicates().tolist()
    )
    scores = df["score"].unique().tolist()

    n_labels = len(label_order)
    n_scores = len(scores)
    bar_width = 0.8 / n_scores
    x = np.arange(n_labels)

    fig, axes = plt.subplots(1, 3, figsize=(6 + 2.5 * n_labels, 6), sharey=False)

    for ax, metric in zip(axes, METRIC_NAMES):
        for i, score in enumerate(scores):
            subset = df[df["score"] == score]
            vals = []
            for lbl in label_order:
                row = subset[subset["x_label"] == lbl]
                vals.append(row[metric].values[0] if len(row) else 0)

            color = SCORE_COLORS.get(score, "#95A5A6")
            bars = ax.bar(
                x + i * bar_width, vals, bar_width, label=score, color=color, edgecolor="white", linewidth=0.5
            )

            for bar, v in zip(bars, vals):
                if v != 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color=COLOR_PRIMARY,
                    )

        ax.set_title(METRIC_LABELS[metric])
        ax.set_xticks(x + bar_width * (n_scores - 1) / 2)
        ax.set_xticklabels(label_order, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.axhline(0.5, color=COLOR_SECONDARY, linestyle="--", linewidth=0.8, alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=n_scores, frameon=True, fontsize=10, bbox_to_anchor=(0.5, 1.02)
    )

    fig.suptitle("Score Discriminance Metrics", fontsize=18, fontweight="bold", color=COLOR_PRIMARY, y=1.06)
    fig.tight_layout()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "score_discriminance.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score discriminance plot saved to {fig_path}")
    return fig_path


def plot_monitoring_dashboard(period_result: dict, output_dir: Path) -> Path:
    """
    Generate a 2x2 monitoring dashboard for a single segment/period:
      [0,0] Precision-Recall curves
      [0,1] Score distribution (Score RF)
      [1,0] Score distribution (Risk Score RF)
      [1,1] Cumulative Gains chart

    Returns path to the saved figure.
    """
    apply_matplotlib_style()

    label = period_result["label"]
    y_true = period_result["y_true"]
    scores_dict = period_result["scores_dict"]
    lift_tables = period_result["lift_tables"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # [0,0] Precision-Recall
    plot_precision_recall_curve(axes[0, 0], y_true, scores_dict)

    # [0,1] Score distribution — first score
    score_names = list(scores_dict.keys())
    if len(score_names) >= 1:
        plot_score_distribution(axes[0, 1], y_true, scores_dict[score_names[0]], name=score_names[0])

    # [1,0] Score distribution — second score
    if len(score_names) >= 2:
        plot_score_distribution(axes[1, 0], y_true, scores_dict[score_names[1]], name=score_names[1])
    else:
        axes[1, 0].set_visible(False)

    # [1,1] Cumulative Gains chart
    ax_lift = axes[1, 1]
    for score_name, lift_df in lift_tables.items():
        color = SCORE_COLORS.get(score_name, "#95A5A6")
        ax_lift.plot(
            lift_df["cumulative_pct_population"],
            lift_df["cumulative_pct_bads"],
            marker="o",
            markersize=5,
            lw=2.5,
            label=score_name,
            color=color,
        )
    ax_lift.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax_lift.set_xlabel("Cumulative % Population", fontsize=14)
    ax_lift.set_ylabel("Cumulative % Bads Captured", fontsize=14)
    ax_lift.set_title("Cumulative Gains Chart", fontsize=16)
    ax_lift.legend(fontsize=11)
    ax_lift.set_xlim(0, 1.02)
    ax_lift.set_ylim(0, 1.05)
    ax_lift.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_lift.spines["top"].set_visible(False)
    ax_lift.spines["right"].set_visible(False)

    fig.suptitle(f"Score Monitoring: {label}", fontsize=18, fontweight="bold", color=COLOR_PRIMARY, y=1.02)
    fig.tight_layout()

    safe_label = label.replace("/", "_")
    fig_path = output_dir / f"monitoring_{safe_label}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Monitoring dashboard saved to {fig_path}")
    return fig_path


# ---------------------------------------------------------------------------
# Main report function (also used by run_batch.py)
# ---------------------------------------------------------------------------
def generate_score_discriminance_report(
    preloaded_data: pd.DataFrame,
    segments: dict,
    supersegments: dict,
    base_config: dict,
    output_path: str,
) -> pd.DataFrame:
    """
    Compute full score monitoring report for all segments and supersegments.

    Outputs:
    - score_discriminance.csv: AUROC / Gini / KS per segment, period, score.
    - score_discriminance.png: Grouped bar chart of the above.
    - lift_tables/{label}.csv: Decile lift table per segment/period/score.
    - delong_comparisons.csv: Pairwise DeLong AUC comparison results.
    - monitoring_{label}.png: 2x2 dashboard per segment/period.
    """
    date_ini_main = base_config.get("date_ini_book_obs")
    date_fin_main = base_config.get("date_fin_book_obs")
    date_ini_mr = base_config.get("date_ini_book_obs_mr")
    date_fin_mr = base_config.get("date_fin_book_obs_mr")

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    lift_dir = out_dir / "lift_tables"
    lift_dir.mkdir(parents=True, exist_ok=True)

    all_discriminance: list[pd.DataFrame] = []
    all_delong: list[dict] = []
    all_period_results: list[dict] = []

    segment_base_cache: dict[str, pd.DataFrame] = {}

    # --- Per-segment metrics ---
    for seg_name, seg_config in segments.items():
        segment_filter = seg_config.get("segment_filter")
        if not segment_filter:
            logger.warning(f"Segment '{seg_name}' has no segment_filter — skipping.")
            continue

        ss_name = seg_config.get("supersegment", "")

        logger.info(f"Processing segment: {seg_name}")
        df_base = _filter_base_population(preloaded_data, segment_filter)
        segment_base_cache[seg_name] = df_base

        if df_base.empty:
            logger.warning(f"Segment '{seg_name}': no data after base filter — skipping.")
            continue

        for period, d_ini, d_fin in [
            ("main", date_ini_main, date_fin_main),
            ("mr", date_ini_mr, date_fin_mr),
        ]:
            result = _compute_for_period(df_base, d_ini, d_fin, period, "segment", seg_name, ss_name)
            if result is None:
                continue

            all_discriminance.append(result["discriminance_df"])
            all_delong.extend(result["delong_results"])
            all_period_results.append(result)

            # Save lift tables
            for score_name, lift_df in result["lift_tables"].items():
                safe_name = f"{seg_name}_{period}_{score_name.replace(' ', '_')}"
                lift_df.to_csv(lift_dir / f"{safe_name}.csv", index=False)

    # --- Per-supersegment metrics ---
    for ss_name, _ss_config in supersegments.items():
        member_segments = [sn for sn, sc in segments.items() if sc.get("supersegment") == ss_name]

        if not member_segments:
            logger.warning(f"Supersegment '{ss_name}': no member segments found — skipping.")
            continue

        frames = [segment_base_cache[sn] for sn in member_segments if sn in segment_base_cache]
        if not frames:
            continue
        df_combined = pd.concat(frames, ignore_index=True)

        logger.info(f"Processing supersegment: {ss_name} ({len(frames)} segments, {len(df_combined):,} rows)")

        for period, d_ini, d_fin in [
            ("main", date_ini_main, date_fin_main),
            ("mr", date_ini_mr, date_fin_mr),
        ]:
            result = _compute_for_period(df_combined, d_ini, d_fin, period, "supersegment", ss_name, "")
            if result is None:
                continue

            all_discriminance.append(result["discriminance_df"])
            all_delong.extend(result["delong_results"])
            all_period_results.append(result)

            for score_name, lift_df in result["lift_tables"].items():
                safe_name = f"SS_{ss_name}_{period}_{score_name.replace(' ', '_')}"
                lift_df.to_csv(lift_dir / f"{safe_name}.csv", index=False)

    # --- Assemble & save ---
    if not all_discriminance:
        logger.warning("No metrics computed — output files will not be created.")
        return pd.DataFrame()

    # Discriminance CSV
    final_df = pd.concat(all_discriminance, ignore_index=True)
    col_order = [
        "level",
        "name",
        "supersegment",
        "period",
        "score",
        "auroc",
        "gini",
        "ks",
        "n_records",
        "n_bads",
        "bad_rate",
    ]
    final_df = final_df[[c for c in col_order if c in final_df.columns]]
    final_df.to_csv(out_dir / "score_discriminance.csv", index=False)
    logger.info(f"Score discriminance report saved to {out_dir / 'score_discriminance.csv'}")

    # Discriminance plot
    plot_score_discriminance(final_df, out_dir)

    # DeLong comparisons CSV
    if all_delong:
        delong_df = pd.DataFrame(all_delong)
        delong_col_order = [
            "level",
            "name",
            "period",
            "model_1",
            "model_2",
            "auc1",
            "auc2",
            "auc_diff",
            "se_diff",
            "z_statistic",
            "p_value",
        ]
        delong_df = delong_df[[c for c in delong_col_order if c in delong_df.columns]]
        delong_df = delong_df.round({"auc1": 4, "auc2": 4, "auc_diff": 4, "se_diff": 4, "z_statistic": 3, "p_value": 4})
        delong_df.to_csv(out_dir / "delong_comparisons.csv", index=False)
        logger.info(f"DeLong comparisons saved to {out_dir / 'delong_comparisons.csv'}")

    # Per-period monitoring dashboards
    for result in all_period_results:
        plot_monitoring_dashboard(result, out_dir)

    return final_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Full score monitoring: discriminance, lift, PR, distributions, DeLong.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--segments", "-s", nargs="+", help="Specific segments to evaluate (default: all)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--config", "-c", default="config.toml", help="Base config file (default: config.toml)")
    parser.add_argument(
        "--segments-config", default="segments.toml", help="Segments config file (default: segments.toml)"
    )

    args = parser.parse_args()

    # Load configs
    try:
        base_config = load_base_config(args.config)
        all_segments = load_segments_config(args.segments_config)
        all_supersegments = load_supersegments_config(args.segments_config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        return 1

    # Filter segments if requested
    if args.segments:
        segments = {n: c for n, c in all_segments.items() if n in args.segments}
        unknown = set(args.segments) - set(all_segments.keys())
        if unknown:
            print(f"Warning: Unknown segments will be skipped: {unknown}")
        if not segments:
            print("Error: No valid segments to process")
            return 1
    else:
        segments = all_segments

    # Load data
    data_path = base_config.get("data_path", "data/demanda_direct_out.sas7bdat")
    preloaded_data = load_and_standardize_data(data_path)
    if preloaded_data is None:
        print("Error: Could not load data.")
        return 1

    # Compute
    print(f"\nComputing full score monitoring for {len(segments)} segment(s)...")
    final_df = generate_score_discriminance_report(
        preloaded_data=preloaded_data,
        segments=segments,
        supersegments=all_supersegments,
        base_config=base_config,
        output_path=args.output,
    )

    if final_df.empty:
        print("\nNo metrics were computed.")
        return 1

    # Print summary
    print(f"\n{'=' * 90}")
    print("SCORE MONITORING SUMMARY")
    print(f"{'=' * 90}")

    print("\n--- Discriminance Metrics ---")
    print(final_df.to_string(index=False))

    # Print DeLong results
    delong_path = Path(args.output) / "delong_comparisons.csv"
    if delong_path.exists():
        delong_df = pd.read_csv(delong_path)
        print("\n--- DeLong AUC Comparisons ---")
        print(delong_df.to_string(index=False))

        sig = delong_df[delong_df["p_value"] < 0.05]
        if not sig.empty:
            print(f"\n  * {len(sig)} significant AUC difference(s) detected (p < 0.05)")
        else:
            print("\n  * No significant AUC differences detected")

    print(f"\nOutputs saved to {args.output}/:")
    print("  - score_discriminance.csv    (AUROC, Gini, KS)")
    print("  - score_discriminance.png    (bar chart)")
    print("  - delong_comparisons.csv     (pairwise AUC tests)")
    print("  - lift_tables/*.csv          (decile lift tables)")
    print("  - monitoring_*.png           (per-segment dashboards)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
