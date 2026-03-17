"""
Selection bias analysis: demonstrate that low observed Gini is caused by
range restriction from the acceptance policy.

Analyses run at the **reporting supersegment** and **total** level to reduce
noise from small individual segments.

Produces:
  - truncation_report.csv          Distribution truncation statistics
  - thorndike_correction.csv       Observed vs corrected Gini
  - simulated_restriction.csv      Gini sensitivity to progressive truncation
  - rolling_acceptance.csv         Rolling 3-month acceptance rates
  - temporal_gini_vs_acceptance.csv Merged rolling Gini + acceptance
  - rejection_profile_*.csv        Rejection counts by score decile
  - cross_group_scatter.csv        Acceptance rate vs Gini per group
  - *.png                          Visualizations for each analysis

Usage:
    uv run python run_selection_bias_analysis.py
    uv run python run_selection_bias_analysis.py --output output/selection_bias
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from run_batch import (
    load_and_standardize_data,
    load_base_config,
    load_segments_config,
)
from run_score_metrics import (
    SCORE_COLUMNS,
    TARGET_COLUMN,
    _compute_rolling_gini,
    _filter_base_population,
    _filter_booked_period,
    _load_reporting_supersegments,
)
from src.metrics import compute_metrics
from src.selection_bias import (
    compute_acceptance_gini_correlation,
    compute_selection_bias_report,
)
from src.selection_bias_plots import (
    plot_all_truncation_panels,
    plot_cross_segment_scatter,
    plot_rejection_profile,
    plot_simulated_restriction,
    plot_temporal_acceptance_vs_gini,
    plot_thorndike_correction,
)
from src.utils import resolve_reporting_supersegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _filter_demand_period(df_base: pd.DataFrame, date_ini: str, date_fin: str) -> pd.DataFrame:
    """Filter base population to a date range (all statuses, not just booked)."""
    df = df_base.copy()
    df["mis_date"] = pd.to_datetime(df["mis_date"])
    start = pd.to_datetime(date_ini)
    end = pd.to_datetime(date_fin)
    return df[(df["mis_date"] >= start) & (df["mis_date"] <= end)]


def _run_analyses_for_group(
    group_name: str,
    df_base: pd.DataFrame,
    date_ini: str | None,
    date_fin: str | None,
    period: str,
    out_dir: Path,
    profile_dir: Path,
    *,
    all_truncation: list,
    all_thorndike: list,
    all_simulated: list,
    all_rolling_acceptance: list,
    all_rolling_gini: list,
    cross_group_data: list,
):
    """Run all selection bias analyses for one group (supersegment or total) and period."""
    if not date_ini or not date_fin:
        return

    df_demand_period = _filter_demand_period(df_base, date_ini, date_fin)
    if df_demand_period.empty:
        return

    df_booked = _filter_booked_period(df_base, date_ini, date_fin)
    if df_booked.empty or TARGET_COLUMN not in df_booked.columns:
        return
    if df_booked[TARGET_COLUMN].dropna().nunique() < 2:
        return

    label = f"{group_name}_{period}"
    logger.info(f"  {group_name} / {period}: {len(df_demand_period):,} demand, {len(df_booked):,} booked")

    # --- Run all analyses ---
    report = compute_selection_bias_report(
        df_demand=df_demand_period,
        df_booked=df_booked,
        score_columns=SCORE_COLUMNS,
        target_col=TARGET_COLUMN,
        segment_name=group_name,
        period=period,
        date_ini=date_ini,
        date_fin=date_fin,
    )

    if not report["truncation"].empty:
        all_truncation.append(report["truncation"])

    if not report["thorndike"].empty:
        all_thorndike.append(report["thorndike"])

    if not report["simulated_restriction"].empty:
        all_simulated.append(report["simulated_restriction"])

    if not report["rolling_acceptance"].empty:
        all_rolling_acceptance.append(report["rolling_acceptance"])

    # Rolling Gini
    rg = _compute_rolling_gini(df_booked, "group", group_name, period)
    if not rg.empty:
        rg["segment"] = group_name
        all_rolling_gini.append(rg)

    # Rejection profiles
    for score_name, profile_df in report["rejection_profiles"].items():
        if not profile_df.empty:
            safe = f"{label}_{score_name.replace(' ', '_')}"
            profile_df.to_csv(profile_dir / f"{safe}.csv", index=False)

    # Plots
    plot_all_truncation_panels(df_demand_period, SCORE_COLUMNS, label, out_dir)
    plot_rejection_profile(report["rejection_profiles"], label, out_dir)

    # Cross-group scatter data
    required = [TARGET_COLUMN] + [s["column"] for s in SCORE_COLUMNS.values()]
    df_clean = df_booked.dropna(subset=required)
    if not df_clean.empty and df_clean[TARGET_COLUMN].nunique() >= 2:
        n_demand = len(df_demand_period)
        n_booked = int((df_demand_period["status_name"] == "booked").sum())
        acc_rate = n_booked / n_demand if n_demand > 0 else 0

        y_true = df_clean[TARGET_COLUMN].values
        for score_name, info in SCORE_COLUMNS.items():
            sign = -1 if info.get("negate") else 1
            scores = sign * df_clean[info["column"]].values
            try:
                gini, _, _, _ = compute_metrics(y_true, scores)
            except ValueError:
                continue
            cross_group_data.append(
                {
                    "segment": group_name,
                    "period": period,
                    "score": score_name,
                    "acceptance_rate": round(acc_rate, 4),
                    "gini": round(gini, 4),
                }
            )


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def generate_selection_bias_report(
    preloaded_data: pd.DataFrame,
    segments: dict,
    supersegments: dict,
    base_config: dict,
    output_path: str,
) -> dict[str, pd.DataFrame]:
    """Run selection bias analyses at the reporting-supersegment and total level."""
    date_ini_main = base_config.get("date_ini_book_obs")
    date_fin_main = base_config.get("date_fin_book_obs")
    date_ini_mr = base_config.get("date_ini_book_obs_mr")
    date_fin_mr = base_config.get("date_fin_book_obs_mr")

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = out_dir / "rejection_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    # Collectors
    all_truncation: list[pd.DataFrame] = []
    all_thorndike: list[pd.DataFrame] = []
    all_simulated: list[pd.DataFrame] = []
    all_rolling_acceptance: list[pd.DataFrame] = []
    all_rolling_gini: list[pd.DataFrame] = []
    cross_group_data: list[dict] = []

    # --- Build per-segment base DataFrames (demand = booked + rejected) ---
    segment_base_cache: dict[str, pd.DataFrame] = {}
    for seg_name, seg_config in segments.items():
        segment_filter = seg_config.get("segment_filter")
        if not segment_filter:
            continue
        df_base = _filter_base_population(preloaded_data, segment_filter)
        if not df_base.empty:
            segment_base_cache[seg_name] = df_base

    # --- Reporting supersegments ---
    for ss_name, ss_config in supersegments.items():
        # Find member segments: match on segment_filter in the supersegment's list
        ss_filters = set(ss_config.get("segment_filters", []))
        member_names = [
            sn
            for sn, sc in segments.items()
            if sc.get("segment_filter") in ss_filters or resolve_reporting_supersegment(sc) == ss_name
        ]

        frames = [segment_base_cache[sn] for sn in member_names if sn in segment_base_cache]
        if not frames:
            logger.warning(f"Supersegment '{ss_name}': no data — skipping.")
            continue

        df_combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Processing supersegment: {ss_name} ({len(frames)} segments, {len(df_combined):,} rows)")

        common_kwargs = {
            "out_dir": out_dir,
            "profile_dir": profile_dir,
            "all_truncation": all_truncation,
            "all_thorndike": all_thorndike,
            "all_simulated": all_simulated,
            "all_rolling_acceptance": all_rolling_acceptance,
            "all_rolling_gini": all_rolling_gini,
            "cross_group_data": cross_group_data,
        }

        for period, d_ini, d_fin in [
            ("main", date_ini_main, date_fin_main),
            ("mr", date_ini_mr, date_fin_mr),
        ]:
            _run_analyses_for_group(ss_name, df_combined, d_ini, d_fin, period, **common_kwargs)

    # --- Total (all segments) ---
    if segment_base_cache:
        df_total = pd.concat(segment_base_cache.values(), ignore_index=True)
        logger.info(f"Processing total: {len(segment_base_cache)} segments, {len(df_total):,} rows")

        common_kwargs = {
            "out_dir": out_dir,
            "profile_dir": profile_dir,
            "all_truncation": all_truncation,
            "all_thorndike": all_thorndike,
            "all_simulated": all_simulated,
            "all_rolling_acceptance": all_rolling_acceptance,
            "all_rolling_gini": all_rolling_gini,
            "cross_group_data": cross_group_data,
        }

        for period, d_ini, d_fin in [
            ("main", date_ini_main, date_fin_main),
            ("mr", date_ini_mr, date_fin_mr),
        ]:
            _run_analyses_for_group("total", df_total, d_ini, d_fin, period, **common_kwargs)

    # --- Save CSVs & plots ---
    outputs: dict[str, pd.DataFrame] = {}

    if all_truncation:
        trunc_df = pd.concat(all_truncation, ignore_index=True)
        trunc_df.to_csv(out_dir / "truncation_report.csv", index=False)
        outputs["truncation"] = trunc_df
        logger.info(f"Truncation report saved ({len(trunc_df)} rows)")

    if all_thorndike:
        thor_df = pd.concat(all_thorndike, ignore_index=True)
        thor_df.to_csv(out_dir / "thorndike_correction.csv", index=False)
        outputs["thorndike"] = thor_df
        logger.info(f"Thorndike correction saved ({len(thor_df)} rows)")
        plot_thorndike_correction(thor_df, out_dir)

    if all_simulated:
        sim_df = pd.concat(all_simulated, ignore_index=True)
        sim_df.to_csv(out_dir / "simulated_restriction.csv", index=False)
        outputs["simulated_restriction"] = sim_df
        logger.info(f"Simulated restriction saved ({len(sim_df)} rows)")
        plot_simulated_restriction(sim_df, out_dir)

    if all_rolling_acceptance:
        racc_df = pd.concat(all_rolling_acceptance, ignore_index=True)
        racc_df.to_csv(out_dir / "rolling_acceptance.csv", index=False)
        outputs["rolling_acceptance"] = racc_df

    # Temporal Gini vs Acceptance
    if all_rolling_gini and all_rolling_acceptance:
        rgini_df = pd.concat(all_rolling_gini, ignore_index=True)
        racc_df = outputs.get("rolling_acceptance", pd.DataFrame())
        if not racc_df.empty:
            merged = rgini_df.merge(
                racc_df[["segment", "period", "month", "acceptance_rate"]],
                on=["segment", "period", "month"],
                how="inner",
            )
            if not merged.empty:
                merged.to_csv(out_dir / "temporal_gini_vs_acceptance.csv", index=False)
                outputs["temporal"] = merged
                logger.info(f"Temporal Gini vs acceptance saved ({len(merged)} rows)")
            plot_temporal_acceptance_vs_gini(rgini_df, racc_df, out_dir)

    # Cross-group scatter
    if cross_group_data:
        cs_df = pd.DataFrame(cross_group_data)
        cs_df.to_csv(out_dir / "cross_group_scatter.csv", index=False)
        outputs["cross_group"] = cs_df

        for score_name in cs_df["score"].unique():
            sdf = cs_df[cs_df["score"] == score_name].to_dict("records")
            regression = compute_acceptance_gini_correlation(sdf)
            logger.info(
                f"Cross-group regression ({score_name}): r={regression.get('r_value')}, p={regression.get('p_value')}"
            )

        regression = compute_acceptance_gini_correlation(cross_group_data)
        plot_cross_segment_scatter(cross_group_data, regression, out_dir)

    return outputs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Selection bias analysis: prove low Gini is due to range restriction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--output", "-o", default="output/selection_bias", help="Output directory")
    parser.add_argument("--config", "-c", default="config.toml", help="Base config file")
    parser.add_argument("--segments-config", default="segments.toml", help="Segments config file")

    args = parser.parse_args()

    try:
        base_config = load_base_config(args.config)
        all_segments = load_segments_config(args.segments_config)
        all_supersegments = _load_reporting_supersegments(args.segments_config)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        return 1

    data_path = base_config.get("data_path", "data/demanda_direct_out.sas7bdat")
    preloaded_data = load_and_standardize_data(data_path)
    if preloaded_data is None:
        print("Error: Could not load data.")
        return 1

    n_groups = len(all_supersegments) + 1  # supersegments + total
    print(
        f"\nRunning selection bias analysis for {n_groups} group(s) "
        f"({len(all_supersegments)} supersegment(s) + total)..."
    )

    outputs = generate_selection_bias_report(
        preloaded_data=preloaded_data,
        segments=all_segments,
        supersegments=all_supersegments,
        base_config=base_config,
        output_path=args.output,
    )

    if not outputs:
        print("\nNo analyses were computed.")
        return 1

    # --- Print summary ---
    print(f"\n{'=' * 90}")
    print("SELECTION BIAS ANALYSIS SUMMARY")
    print(f"{'=' * 90}")

    if "truncation" in outputs:
        print("\n--- Distribution Truncation (u = SD_booked / SD_demand) ---")
        trunc = outputs["truncation"][
            ["segment", "period", "score", "acceptance_rate", "u_ratio", "sd_demand", "sd_booked"]
        ]
        print(trunc.to_string(index=False))

    if "thorndike" in outputs:
        print("\n--- Thorndike Case 2 Correction ---")
        thor = outputs["thorndike"][
            ["segment", "period", "score", "observed_gini", "u_ratio", "corrected_gini", "attenuation_pct"]
        ]
        print(thor.to_string(index=False))

    if "simulated_restriction" in outputs:
        sim = outputs["simulated_restriction"]
        endpoints = sim[sim["pct_retained"].isin([1.0, 0.5])]
        if not endpoints.empty:
            print("\n--- Simulated Restriction (Gini at 100% vs 50% retained) ---")
            pivot = endpoints.pivot_table(index=["segment", "score"], columns="pct_retained", values="gini")
            if 1.0 in pivot.columns and 0.5 in pivot.columns:
                pivot["gini_drop_pct"] = ((pivot[1.0] - pivot[0.5]) / pivot[1.0] * 100).round(1)
            print(pivot.to_string())

    if "cross_group" in outputs:
        print("\n--- Cross-Group: Acceptance Rate vs Gini ---")
        cs = outputs["cross_group"]
        for score_name in cs["score"].unique():
            sdf = cs[cs["score"] == score_name]
            regression = compute_acceptance_gini_correlation(sdf.to_dict("records"))
            r = regression.get("r_value", "N/A")
            p = regression.get("p_value", "N/A")
            print(f"  {score_name}: r={r}, p={p} (n={len(sdf)} groups)")

    print(f"\nOutputs saved to {args.output}/:")
    print("  - truncation_report.csv              (distribution statistics)")
    print("  - thorndike_correction.csv            (observed vs corrected Gini)")
    print("  - simulated_restriction.csv           (progressive truncation)")
    print("  - rolling_acceptance.csv              (3-month rolling acceptance)")
    print("  - temporal_gini_vs_acceptance.csv      (merged rolling Gini + acceptance)")
    print("  - cross_group_scatter.csv             (acceptance vs Gini per group)")
    print("  - rejection_profiles/*.csv            (booked/rejected by decile)")
    print("  - truncation_*.png                    (KDE overlays)")
    print("  - thorndike_correction.png            (bar chart)")
    print("  - simulated_restriction.png           (sensitivity curves)")
    print("  - temporal_acceptance_vs_gini.png     (within-group scatter)")
    print("  - rejection_profile_*.png             (stacked bars)")
    print("  - cross_segment_acceptance_vs_gini.png (scatter)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
