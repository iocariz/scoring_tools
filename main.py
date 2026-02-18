import argparse
import time

import pandas as pd
from loguru import logger
from pydantic import ValidationError

from src.config import OutputPaths
from src.data_manager import DataValidationError, load_and_prepare_data
from src.pipeline.config_loader import load_and_validate_config
from src.pipeline.inference import run_inference_phase
from src.pipeline.optimization import (
    _build_scenario_list,
    _compute_mr_annual_coef,
    _save_cutoff_summaries,
    run_optimization_phase,
    run_scenario_analysis,
)
from src.pipeline.preprocessing import run_preprocessing_phase


class PipelineExecutionError(RuntimeError):
    """Base exception for pipeline execution failures."""


class ConfigLoadError(PipelineExecutionError):
    """Raised when config load/validation fails."""


class DataLoadError(PipelineExecutionError):
    """Raised when data load/validation fails."""


class InferencePhaseError(PipelineExecutionError):
    """Raised when inference phase fails."""


def main(
    config_path: str = "config.toml",
    model_path: str | None = None,
    training_only: bool = False,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame | None = None,
    output: OutputPaths | None = None,
):
    """
    Load and preprocess SAS data using configuration.

    Args:
        config_path: Path to the configuration TOML file (default: config.toml)
        model_path: Optional path to a pre-trained model directory.
        training_only: If True, only runs data preprocessing and model training.
        skip_dq_checks: If True, skip data quality checks.
        preloaded_data: Optional pre-loaded and standardized DataFrame.

    Returns:
        Tuple of processed DataFrames, or None if processing fails
    """
    if output is None:
        output = OutputPaths()
    output.ensure_dirs()

    t0_total = time.perf_counter()

    try:
        # Step 1: Load and validate configuration
        try:
            settings, date_ini, date_fin, annual_coef = load_and_validate_config(config_path)
        except ValidationError as e:
            raise ConfigLoadError(f"Configuration validation failed for '{config_path}'") from e
        except Exception as e:
            raise ConfigLoadError(f"Error loading configuration from '{config_path}'") from e

        segment = settings.segment_filter

        # Step 2: Load and prepare data
        try:
            data = load_and_prepare_data(settings, preloaded_data)
        except (DataValidationError, FileNotFoundError) as e:
            raise DataLoadError(f"[{segment}] Data error") from e
        except Exception as e:
            raise DataLoadError(f"[{segment}] Unexpected data loading error") from e

        # Step 3: Preprocessing (DQ checks, binning, stress factor, transformation rate)
        result = run_preprocessing_phase(data, settings, skip_dq_checks, output=output)
        if result is None:
            return None
        data_clean, data_booked, data_demand, stress_factor, tasa_fin = result

        # Step 4: Risk inference (model training or loading)
        try:
            risk_inference, reg_todu_amt_pile = run_inference_phase(data_clean, settings, model_path, output=output)
        except Exception as e:
            raise InferencePhaseError(f"[{segment}] Inference phase failed") from e

        # Early return for training_only mode (supersegment training)
        if training_only:
            elapsed = time.perf_counter() - t0_total
            logger.info(
                f"[{segment}] Training only complete | "
                f"model={risk_inference.get('model_path', 'models/')} | {elapsed:.1f}s total"
            )
            return data_clean, data_booked, data_demand, risk_inference, reg_todu_amt_pile

        # Step 5: Optimization (feasible solutions, KPIs, Pareto front)
        (data_summary_desagregado, data_summary, data_summary_sample_no_opt, values_var0, values_var1) = (
            run_optimization_phase(
                data_booked,
                data_demand,
                risk_inference,
                reg_todu_amt_pile,
                stress_factor,
                tasa_fin,
                settings,
                annual_coef,
                output=output,
            )
        )

        # Step 6: Scenario analysis loop
        use_fixed_cutoffs = settings.fixed_cutoffs is not None and len(settings.fixed_cutoffs) > 0
        scenarios = _build_scenario_list(settings, use_fixed_cutoffs)
        annual_coef_mr = _compute_mr_annual_coef(settings)

        cutoff_summaries = []
        for scenario_risk, scenario_name in scenarios:
            summary = run_scenario_analysis(
                scenario_risk,
                scenario_name,
                data_summary=data_summary,
                data_summary_desagregado=data_summary_desagregado,
                data_summary_sample_no_opt=data_summary_sample_no_opt,
                data_clean=data_clean,
                data_booked=data_booked,
                settings=settings,
                risk_inference=risk_inference,
                reg_todu_amt_pile=reg_todu_amt_pile,
                stress_factor=stress_factor,
                tasa_fin=tasa_fin,
                annual_coef_mr=annual_coef_mr,
                values_var0=values_var0,
                values_var1=values_var1,
                output=output,
            )
            cutoff_summaries.append(summary)

        _save_cutoff_summaries(cutoff_summaries, settings, output=output)

        # Step 7: Temporal trend analysis (non-blocking)
        try:
            from src.trends import compute_monthly_metrics, detect_trend_changes, plot_metric_trends

            monthly = compute_monthly_metrics(data_clean, date_column="mis_date", segment_filter=segment)
            if not monthly.empty:
                # Save monthly metrics
                monthly.to_csv(output.monthly_metrics_csv(segment))

                # Plot key metrics
                plot_metric_trends(
                    monthly,
                    ["approval_rate", "total_records", "mean_production"],
                    output_path=output.metric_trends_html(segment),
                )

                # Detect anomalies in approval rate
                anomalies = detect_trend_changes(monthly, "approval_rate", window=3)
                anomaly_months = anomalies[anomalies["is_anomaly"]]
                if not anomaly_months.empty:
                    anomalies.to_csv(output.trend_anomalies_csv(segment))
                    logger.warning(f"[{segment}] Trend anomalies detected in {len(anomaly_months)} month(s)")
                else:
                    logger.info(f"[{segment}] No trend anomalies detected")
            else:
                logger.info(f"[{segment}] Insufficient data for trend analysis")
        except Exception as e:
            logger.warning(f"[{segment}] Trend analysis failed (non-blocking): {e}")

        elapsed_total = time.perf_counter() - t0_total
        logger.info(f"[{segment}] Pipeline complete | {len(scenarios)} scenarios | {elapsed_total:.1f}s total")

        return data_clean, data_booked, data_demand, data_summary_desagregado, data_summary
    except PipelineExecutionError as e:
        logger.exception(str(e))
        return None


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
        "--skip-dq-checks",
        action="store_true",
        help="Skip data quality checks (use with caution).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        config_path=args.config,
        model_path=args.model_path,
        training_only=args.training_only,
        skip_dq_checks=args.skip_dq_checks,
    )
