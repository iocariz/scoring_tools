import time
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.data_quality import run_data_quality_checks
from src.plots import (
    calculate_and_plot_transformation_rate,
    calculate_per_bin_transformation_rate,
    plot_bin_threshold_diagnostic,
    plot_risk_vs_production,
)
from src.preprocess_improved import complete_preprocessing_pipeline, filter_by_date
from src.utils import calculate_per_bin_stress_factors, calculate_stress_factor


@dataclass
class PreprocessingResult:
    """Result of the preprocessing phase.

    Encapsulates all outputs so that adding new fields does not break
    call sites that unpack by position.
    """

    data_clean: pd.DataFrame
    data_booked: pd.DataFrame
    data_demand: pd.DataFrame
    stress_factor: float
    tasa_fin: float
    per_bin_stress: pd.DataFrame | None = None
    per_bin_tasa_fin: pd.DataFrame | None = None


def run_preprocessing_phase(
    data: pd.DataFrame,
    settings: PreprocessingSettings,
    skip_dq_checks: bool,
    output: OutputPaths | None = None,
) -> PreprocessingResult | None:
    """Run data quality checks, preprocessing pipeline, and compute derived metrics.

    Args:
        data: Input DataFrame
        settings: Configuration settings object
        skip_dq_checks: If True, skip data quality checks
        output: Output paths configuration. Defaults to current directory.

    Returns:
        PreprocessingResult or None if data quality checks fail.
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
    fig = plot_risk_vs_production(
        data_clean, settings.indicators, settings.cz_config, data_booked, multiplier=settings.multiplier
    )
    fig.write_html(output.risk_vs_production_html)
    logger.debug(f"[{segment}] Risk vs production plot saved to {output.risk_vs_production_html}")

    # Bin threshold diagnostic charts (one per binning variable)
    for _var_name, bc in settings.bins.items():
        if bc.output_col in data_clean.columns:
            try:
                plot_bin_threshold_diagnostic(
                    data_clean,
                    bin_col=bc.output_col,
                    bin_edges=bc.bin_edges or None,
                    source_col=bc.source_col,
                    output_path=output.bin_diagnostic_html(bc.output_col),
                )
            except Exception as e:
                logger.warning(f"[{segment}] Bin diagnostic for '{bc.output_col}' failed (non-blocking): {e}")

    # -------------------------------------------------------------------------
    # Stress factor
    # -------------------------------------------------------------------------
    stress_mode = getattr(settings, "stress_mode", "global")
    per_bin_stress: pd.DataFrame | None = None

    if stress_mode == "disabled":
        stress_factor = 1.0
        logger.info(f"[{segment}] stress_mode=disabled — stress factor set to 1.0")
    elif stress_mode == "per_bin":
        stress_factor = 1.0  # scalar is neutral; per-bin DF carries the adjustment
        per_bin_stress = calculate_per_bin_stress_factors(data_booked, settings.variables)
        logger.info(
            f"[{segment}] stress_mode=per_bin — "
            f"{len(per_bin_stress)} bins, mean={per_bin_stress['stress_factor'].mean():.3f}"
        )
    else:
        # "global" (default / backward-compatible)
        stress_factor = calculate_stress_factor(data_booked)
        if settings.reject_inference_method == "parceling":
            logger.info(
                f"[{segment}] stress_mode=global with parceling active — "
                f"consider stress_mode='disabled' to avoid double-counting selection bias"
            )

    # -------------------------------------------------------------------------
    # Transformation rate (tasa_fin)
    # -------------------------------------------------------------------------
    # Avoid look-ahead / leakage: transformation rate rolling window must be computed
    # within the configured observation period, not relative to max(df) across the full dataset.
    data_for_tasa_fin = data_clean
    if settings.date_ini_book_obs and settings.date_fin_book_obs:
        data_for_tasa_fin = filter_by_date(
            data_clean, "mis_date", settings.date_ini_book_obs, settings.date_fin_book_obs
        )

    result = calculate_and_plot_transformation_rate(
        data_for_tasa_fin, date_col="mis_date", amount_col="oa_amt", n_months=settings.n_months
    )
    result["figure"].write_html(output.transformation_rate_html)
    tasa_fin = result["overall_rate"]
    if tasa_fin <= 0:
        logger.warning(f"[{segment}] tasa_fin={tasa_fin:.4f} is non-positive; defaulting to 1.0")
        tasa_fin = 1.0
    elif tasa_fin > 5.0:
        logger.warning(f"[{segment}] tasa_fin={tasa_fin:.4f} is unusually high (>5.0) — verify transformation rate")

    per_bin_tasa_fin: pd.DataFrame | None = None
    if getattr(settings, "per_bin_tasa_fin", False):
        per_bin_tasa_fin = calculate_per_bin_transformation_rate(
            data_for_tasa_fin,
            settings.variables,
            date_col="mis_date",
            amount_col="oa_amt",
            n_months=settings.n_months,
            global_tasa_fin=tasa_fin,
        )
        logger.info(
            f"[{segment}] per_bin_tasa_fin enabled — "
            f"{len(per_bin_tasa_fin)} bins, mean={per_bin_tasa_fin['tasa_fin'].mean():.3f}"
        )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Preprocessing done | "
        f"clean={len(data_clean):,} booked={len(data_booked):,} demand={len(data_demand):,} | "
        f"stress={stress_factor:.4f} tasa_fin={tasa_fin:.2%} | {elapsed:.1f}s"
    )

    return PreprocessingResult(
        data_clean=data_clean,
        data_booked=data_booked,
        data_demand=data_demand,
        stress_factor=stress_factor,
        tasa_fin=tasa_fin,
        per_bin_stress=per_bin_stress,
        per_bin_tasa_fin=per_bin_tasa_fin,
    )
