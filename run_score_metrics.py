"""
Score discriminance metrics (AUROC, KS, Gini) for booked population.

Evaluates the discriminatory power of score_rf, risk_score_rf, and their
logistic-regression combination against the early_bad target, per segment,
per supersegment, and for both Main and MR periods.

Usage:
    uv run python run_score_metrics.py                             # All segments
    uv run python run_score_metrics.py --segments no_premium_ab    # Specific
    uv run python run_score_metrics.py --output output             # Custom dir
"""

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd
from loguru import logger

from run_batch import (
    load_and_standardize_data,
    load_base_config,
    load_segments_config,
    load_supersegments_config,
)
from src.metrics import compute_score_discriminance

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


def _compute_for_period(
    df_base: pd.DataFrame,
    date_ini: str | None,
    date_fin: str | None,
    period: str,
    level: str,
    name: str,
    supersegment: str,
) -> pd.DataFrame | None:
    """Compute discriminance metrics for a single period, returning tagged rows."""
    if not date_ini or not date_fin:
        return None

    df_booked = _filter_booked_period(df_base, date_ini, date_fin)
    label = f"{level}/{name}/{period}"

    if df_booked.empty:
        logger.warning(f"[{label}] No booked records in date range — skipping.")
        return None

    if not _validate_target(df_booked, label):
        return None

    # Drop rows where scores or target are NaN
    required_cols = [TARGET_COLUMN] + [s["column"] for s in SCORE_COLUMNS.values()]
    df_booked = df_booked.dropna(subset=required_cols)

    if df_booked.empty or df_booked[TARGET_COLUMN].nunique() < 2:
        logger.warning(f"[{label}] Not enough data after dropping NaNs — skipping.")
        return None

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
    return metrics_df


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
    Compute score discriminance metrics for all segments and supersegments.

    Args:
        preloaded_data: Standardised raw DataFrame.
        segments: Segment configs from segments.toml.
        supersegments: Supersegment configs from segments.toml.
        base_config: Base config from config.toml (preprocessing section).
        output_path: Directory to write score_discriminance.csv.

    Returns:
        Concatenated results DataFrame.
    """
    date_ini_main = base_config.get("date_ini_book_obs")
    date_fin_main = base_config.get("date_fin_book_obs")
    date_ini_mr = base_config.get("date_ini_book_obs_mr")
    date_fin_mr = base_config.get("date_fin_book_obs_mr")

    all_results: list[pd.DataFrame] = []
    # Cache of base-filtered data per segment for supersegment aggregation
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
            if result is not None:
                all_results.append(result)

    # --- Per-supersegment metrics ---
    for ss_name, _ss_config in supersegments.items():
        # Find segments belonging to this supersegment
        member_segments = [
            sn for sn, sc in segments.items() if sc.get("supersegment") == ss_name
        ]

        if not member_segments:
            logger.warning(f"Supersegment '{ss_name}': no member segments found — skipping.")
            continue

        # Combine cached base populations
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
            if result is not None:
                all_results.append(result)

    # --- Assemble & save ---
    if not all_results:
        logger.warning("No metrics computed — output CSV will not be created.")
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)
    # Reorder columns
    col_order = ["level", "name", "supersegment", "period", "score", "auroc", "gini", "ks", "n_records", "n_bads", "bad_rate"]
    final_df = final_df[[c for c in col_order if c in final_df.columns]]

    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "score_discriminance.csv"
    final_df.to_csv(csv_path, index=False)
    logger.info(f"Score discriminance report saved to {csv_path}")

    return final_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute score discriminance metrics (AUROC, KS, Gini) on booked population.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--segments", "-s", nargs="+", help="Specific segments to evaluate (default: all)")
    parser.add_argument("--output", "-o", default="output", help="Output directory (default: output)")
    parser.add_argument("--config", "-c", default="config.toml", help="Base config file (default: config.toml)")
    parser.add_argument("--segments-config", default="segments.toml", help="Segments config file (default: segments.toml)")

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
    print(f"\nComputing score discriminance for {len(segments)} segment(s)...")
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

    # Print summary table
    print(f"\n{'=' * 90}")
    print("SCORE DISCRIMINANCE SUMMARY")
    print(f"{'=' * 90}")
    print(final_df.to_string(index=False))
    print(f"\nResults saved to: {args.output}/score_discriminance.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())
