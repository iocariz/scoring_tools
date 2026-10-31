import argparse

from loguru import logger

from src.pipeline.segment_runner import run_pipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Credit Risk Scoring and Portfolio Optimization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  uv run python main.py

  # Run with custom config file
  uv run python main.py --config path/to/config.toml

  # Skip data quality checks (faster, not recommended for production)
  uv run python main.py --skip-dq-checks

  # Training only mode (skip optimization and scenario analysis)
  uv run python main.py --training-only

  # Use pre-trained model for optimization
  uv run python main.py --model-path models/model_20240101_120000

Output files:
  data/optimal_solution_*.csv          - Optimal cutoff solutions per scenario
  data/risk_production_summary_*.csv   - Risk/production metrics
  data/cutoff_summary_by_segment.csv   - Cutoff points summary (long format)
  data/cutoff_summary_wide.csv         - Cutoff points summary (wide format)
  images/risk_production_*.html        - Interactive visualizations
  data/monthly_metrics_*.csv           - Monthly aggregated metrics
  data/trend_anomalies_*.csv           - Detected trend anomalies
  images/metric_trends_*.html          - Monthly metric trend charts
        """,
    )

    parser.add_argument(
        "--config", "-c", type=str, default="config.toml", help="Path to configuration TOML file (default: config.toml)"
    )

    parser.add_argument(
        "--model-path",
        "-m",
        type=str,
        default=None,
        help="Path to pre-trained model directory. Skips training and uses existing model.",
    )

    parser.add_argument(
        "--training-only",
        "-t",
        action="store_true",
        help="Run only data preprocessing and model training (skips optimization). Useful for supersegment training.",
    )

    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: show current portfolio as-is (no cutoff optimization, no swap-in/swap-out).",
    )

    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Run only the base scenario (skip pessimistic and optimistic).",
    )

    parser.add_argument(
        "--skip-dq-checks",
        action="store_true",
        help="Skip data quality checks (use with caution).",
    )

    parser.add_argument(
        "--allow-dq-warnings",
        action="store_true",
        help=(
            "Analyst escape hatch: proceed past non-critical DQ warnings (coverage gaps, outliers, "
            "small segments) instead of halting. DQ is fail-closed by default; FAILED-severity checks "
            "(negative counts/amounts, unparseable dates) still halt — use --skip-dq-checks to skip DQ entirely."
        ),
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to write all log output to a file (in addition to console).",
    )

    parser.add_argument(
        "--resimulate",
        type=float,
        nargs="+",
        default=None,
        metavar="RISK",
        help=(
            "Resimulation mode: skip data loading/preprocessing/training/optimization, "
            "load artifacts from a previous run, and re-run scenario analysis with the "
            "specified optimum_risk value(s). "
            "1 value → base only; 3 values → pessimistic/base/optimistic; "
            "N values → target_X.XX. Example: --resimulate 0.8 0.96 1.2"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    import sys

    args = parse_args()

    sink_id = None
    if args.log_file:
        sink_id = logger.add(args.log_file, rotation="50 MB", level="DEBUG")

    try:
        result = run_pipeline(
            config_path=args.config,
            model_path=args.model_path,
            training_only=args.training_only,
            baseline_mode=args.baseline,
            base_scenario_only=args.base_only,
            skip_dq_checks=args.skip_dq_checks,
            allow_dq_warnings=args.allow_dq_warnings,
            resimulate_risk=args.resimulate,
        )
    finally:
        if sink_id is not None:
            logger.remove(sink_id)

    # #61: exit non-zero on pipeline failure so CI / scripts can detect it.
    # main() returns the False sentinel on PipelineExecutionError; success
    # returns a tuple (full/training) or None (resimulation).
    if result is False:
        sys.exit(1)
