import time

import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.data_quality import run_data_quality_checks
from src.plots import calculate_and_plot_transformation_rate, plot_risk_vs_production
from src.preprocess_improved import complete_preprocessing_pipeline
from src.utils import calculate_stress_factor


def run_preprocessing_phase(
    data: pd.DataFrame,
    settings: PreprocessingSettings,
    skip_dq_checks: bool,
    output: OutputPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float] | None:
    """Run data quality checks, preprocessing pipeline, and compute derived metrics.

    Args:
        data: Input DataFrame
        settings: Configuration settings object
        skip_dq_checks: If True, skip data quality checks
        output: Output paths configuration. Defaults to current directory.

    Returns:
        Tuple of (data_clean, data_booked, data_demand, stress_factor, tasa_fin)
        Returns None if data quality checks fail.
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter

    # Data Quality Checks
    if skip_dq_checks:
        logger.warning(f"[{segment}] Skipping data quality checks (--skip-dq-checks flag set)")
    else:
        dq_report = run_data_quality_checks(data, settings, verbose=True)

        if not dq_report.is_valid:
            logger.error(f"[{segment}] Data quality validation failed. Use --skip-dq-checks to bypass.")
            return None

        if dq_report.warnings:
            logger.warning(f"[{segment}] {len(dq_report.warnings)} data quality warnings. Proceeding with caution.")

    # Preprocessing
    data_clean, data_booked, data_demand = complete_preprocessing_pipeline(data, settings)

    # Risk vs production plot
    fig = plot_risk_vs_production(data_clean, settings.indicators, settings.cz_config, data_booked)
    fig.write_html(output.risk_vs_production_html)
    logger.debug(f"[{segment}] Risk vs production plot saved to {output.risk_vs_production_html}")

    # Stress factor & transformation rate
    stress_factor = calculate_stress_factor(data_booked)
    result = calculate_and_plot_transformation_rate(
        data_clean, date_col="mis_date", amount_col="oa_amt", n_months=settings.n_months
    )
    result["figure"].write_html(output.transformation_rate_html)
    tasa_fin = result["overall_rate"]

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Preprocessing done | "
        f"clean={len(data_clean):,} booked={len(data_booked):,} demand={len(data_demand):,} | "
        f"stress={stress_factor:.4f} tasa_fin={tasa_fin:.2%} | {elapsed:.1f}s"
    )

    return data_clean, data_booked, data_demand, stress_factor, tasa_fin
