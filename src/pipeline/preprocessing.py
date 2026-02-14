import time

import pandas as pd
from loguru import logger

from src.config import PreprocessingSettings
from src.data_quality import run_data_quality_checks
from src.plots import calculate_and_plot_transformation_rate, plot_risk_vs_production
from src.preprocess_improved import PreprocessingConfig, complete_preprocessing_pipeline
from src.utils import calculate_stress_factor


def convert_bins(bins: list[float]) -> list[float]:
    """
    Convert bin values.
    Note: Pydantic handles infinite conversions but we keep this for compatibility
    if config dictionary is accessed directly or for display.
    """
    if not bins:
        return bins
    # For now, pydantic model should return floats, including inf.
    # The original implementation replaced inf strings with np.inf constants.
    # Since pydantic validation allows floats, we might not strictly need this if the model is robust.
    # We will keep it but as a pass-through if they are already floats.
    return bins


def run_preprocessing_phase(
    data: pd.DataFrame,
    settings: PreprocessingSettings,
    skip_dq_checks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float] | None:
    """Run data quality checks, preprocessing pipeline, and compute derived metrics.

    Args:
        data: Input DataFrame
        settings: Configuration settings object
        skip_dq_checks: If True, skip data quality checks

    Returns:
        Tuple of (data_clean, data_booked, data_demand, stress_factor, tasa_fin)
        Returns None if data quality checks fail.
    """
    t0 = time.perf_counter()
    segment = settings.segment_filter

    # Convert settings to dict for compatibility with existing functions that expect a dict
    # This is a temporary bridge until all downstream functions are updated to use the Settings object
    config_dict = settings.model_dump()

    # Data Quality Checks
    if skip_dq_checks:
        logger.warning(f"[{segment}] Skipping data quality checks (--skip-dq-checks flag set)")
    else:
        dq_report = run_data_quality_checks(data, config_dict, verbose=True)

        if not dq_report.is_valid:
            logger.error(f"[{segment}] Data quality validation failed. Use --skip-dq-checks to bypass.")
            return None

        if dq_report.warnings:
            logger.warning(f"[{segment}] {len(dq_report.warnings)} data quality warnings. Proceeding with caution.")

    # Preprocessing
    # Pydantic model already ensures bins are lists of floats
    octroi_bins = settings.octroi_bins
    efx_bins = settings.efx_bins

    config = PreprocessingConfig(
        keep_vars=settings.keep_vars,
        indicators=settings.indicators,
        segment_filter=settings.segment_filter,
        octroi_bins=octroi_bins,
        efx_bins=efx_bins,
        date_ini_book_obs=settings.date_ini_book_obs,
        date_fin_book_obs=settings.date_fin_book_obs,
        score_measures=settings.score_measures,
        log_level=settings.log_level,
    )

    data_clean, data_booked, data_demand = complete_preprocessing_pipeline(data, config)

    # Risk vs production plot
    fig = plot_risk_vs_production(data_clean, settings.indicators, settings.cz_config, data_booked)
    fig.write_html("images/risk_vs_production.html")
    logger.debug(f"[{segment}] Risk vs production plot saved to images/risk_vs_production.html")

    # Stress factor & transformation rate
    stress_factor = calculate_stress_factor(data_booked)
    result = calculate_and_plot_transformation_rate(
        data_clean, date_col="mis_date", amount_col="oa_amt", n_months=settings.n_months
    )
    result["figure"].write_html("images/transformation_rate.html")
    tasa_fin = result["overall_rate"]

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Preprocessing done | "
        f"clean={len(data_clean):,} booked={len(data_booked):,} demand={len(data_demand):,} | "
        f"stress={stress_factor:.4f} tasa_fin={tasa_fin:.2%} | {elapsed:.1f}s"
    )

    return data_clean, data_booked, data_demand, stress_factor, tasa_fin
