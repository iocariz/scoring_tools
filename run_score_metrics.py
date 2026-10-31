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
from pathlib import Path

import pandas as pd

from src.data.data_manager import load_configured_data
from src.pipeline.batch_config import load_base_config, load_segments_config
from src.report.score_metrics import _load_reporting_supersegments, generate_score_discriminance_report


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
        all_supersegments = _load_reporting_supersegments(args.segments_config)
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
    preloaded_data = load_configured_data(base_config)
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
    print("  - score_discriminance.csv    (AUROC, Gini, KS + period dates)")
    print("  - score_discriminance.png    (bar chart)")
    print("  - rolling_gini.csv           (3-month rolling Gini)")
    print("  - rolling_gini.png           (rolling Gini line chart)")
    print("  - delong_comparisons.csv     (pairwise AUC tests)")
    print("  - lift_tables/*.csv          (decile lift tables)")
    print("  - monitoring_*.png           (per-segment dashboards)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
