import argparse
import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger
from pydantic import ValidationError

from src.audit import save_audit_tables
from src.config import PreprocessingSettings
from src.data_quality import run_data_quality_checks
from src.data_manager import load_and_prepare_data, DataValidationError
from src.inference_optimized import (
    inference_pipeline,
    run_optimization_pipeline,
    todu_average_inference,
)
from src.mr_pipeline import process_mr_period
from src.persistence import load_model_for_prediction
from src.plots import RiskProductionVisualizer, plot_risk_vs_production
from src.preprocess_improved import PreprocessingConfig, complete_preprocessing_pipeline, filter_by_date
from src.utils import (
    calculate_and_plot_transformation_rate,
    calculate_annual_coef,
    calculate_b2_ever_h6,
    calculate_stress_factor,
    consolidate_cutoff_summaries,
    create_fixed_cutoff_solution,
    format_cutoff_summary_table,
    generate_cutoff_summary,
    get_fact_sol,
    get_optimal_solutions,
    kpi_of_fact_sol,
)


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


def load_and_validate_config(config_path: str) -> tuple[PreprocessingSettings, pd.Timestamp, pd.Timestamp, float]:
    """Load TOML config, validate it using Pydantic, parse dates, and compute annual coefficient.

    Args:
        config_path: Path to the configuration TOML file

    Returns:
        Tuple of (settings, date_ini, date_fin, annual_coef)

    Raises:
        ValidationError: If configuration validation fails
    """
    settings = PreprocessingSettings.from_toml(config_path)
    logger.debug(f"Configuration loaded from {config_path}")

    # Dates are already validated by Pydantic model
    date_ini = settings.get_date("date_ini_book_obs")
    date_fin = settings.get_date("date_fin_book_obs")
    
    annual_coef = calculate_annual_coef(date_ini, date_fin)

    segment = settings.segment_filter
    logger.info(
        f"Config validated | segment={segment} | "
        f"period={date_ini.date()} to {date_fin.date()} | annual_coef={annual_coef:.2f}"
    )

    return settings, date_ini, date_fin, annual_coef


def run_preprocessing_phase(
    data: pd.DataFrame, settings: PreprocessingSettings, skip_dq_checks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, float]:
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


def run_inference_phase(
    data_clean: pd.DataFrame, settings: PreprocessingSettings, model_path: str = None,
) -> tuple[dict, Any]:
    """Run risk inference: either load a pre-trained model or train a new one.

    Args:
        data_clean: Cleaned DataFrame from preprocessing
        settings: Configuration settings object
        model_path: Optional path to a pre-trained model directory

    Returns:
        Tuple of (risk_inference, reg_todu_amt_pile)

    Raises:
        Exception: If model loading or training fails
    """
    t0 = time.perf_counter()
    segment = settings.segment_filter

    if model_path:
        # Load pre-trained model from supersegment
        model, metadata, features = load_model_for_prediction(model_path)
        # Support both old (test_r2) and new (cv_mean_r2) metric formats
        if "cv_mean_r2" in metadata:
            r2_display = f"{metadata['cv_mean_r2']:.4f} +/- {metadata.get('cv_std_r2', 0.0):.4f}"
        else:
            r2_display = f"{metadata.get('test_r2', 0.0):.4f}"
        risk_inference = {
            "best_model_info": {
                "model": model,
                "name": metadata.get("model_type", "Unknown"),
                "cv_mean_r2": metadata.get("cv_mean_r2", metadata.get("test_r2", 0.0)),
                "cv_std_r2": metadata.get("cv_std_r2", 0.0),
            },
            "features": features,
            "model_path": model_path,
        }

        # Load todu model from the models directory (sibling to model subdirectory)
        todu_model_path = Path(model_path).parent / "todu_model.joblib"
        if not todu_model_path.exists():
            # Also check parent's parent (models/ directory)
            todu_model_path = Path(model_path).parent.parent / "todu_model.joblib"
        if todu_model_path.exists():
            reg_todu_amt_pile = joblib.load(todu_model_path)
            logger.debug(f"[{segment}] Loaded todu model from {todu_model_path}")
        else:
            # Fallback: train todu model on current segment data
            logger.warning(f"[{segment}] Todu model not found at {todu_model_path}, training on current data")
            _, reg_todu_amt_pile, _ = todu_average_inference(
                data=data_clean,
                variables=settings.variables,
                indicators=settings.indicators,
                feature_col="oa_amt",
                target_col="todu_amt_pile_h6",
                z_threshold=settings.z_threshold,
                plot_output_path="models/todu_avg_inference.html",
                model_output_path=None,  # Don't save, it's a fallback
            )

        elapsed = time.perf_counter() - t0
        model_name = risk_inference["best_model_info"]["name"]
        logger.info(f"[{segment}] Model loaded | {model_name} | R2={r2_display} | from {model_path} | {elapsed:.1f}s")
    else:
        # Train new model with feature selection
        risk_inference = inference_pipeline(
            data=data_clean,
            bins=(settings.octroi_bins, settings.efx_bins),
            variables=settings.variables,
            indicators=settings.indicators,
            target_var="b2_ever_h6",
            multiplier=settings.multiplier,
            test_size=0.4,
            include_hurdle=True,
            save_model=True,
            model_base_path="models",
            create_visualizations=True,
        )

        # Todu Average Inference
        _, reg_todu_amt_pile, _ = todu_average_inference(
            data=data_clean,
            variables=settings.variables,
            indicators=settings.indicators,
            feature_col="oa_amt",
            target_col="todu_amt_pile_h6",
            z_threshold=settings.z_threshold,
            plot_output_path="models/todu_avg_inference.html",
            model_output_path="models/todu_model.joblib",
        )

        elapsed = time.perf_counter() - t0
        info = risk_inference["best_model_info"]
        cv_r2 = info.get("cv_mean_r2", 0)
        cv_std = info.get("cv_std_r2", 0)
        logger.info(
            f"[{segment}] Inference done | {info['name']} ({info.get('model_type', 'N/A')}) | "
            f"features={info.get('feature_set', 'N/A')} | CV R2={cv_r2:.4f} +/- {cv_std:.4f} | {elapsed:.1f}s"
        )

    return risk_inference, reg_todu_amt_pile


def run_optimization_phase(
    data_booked: pd.DataFrame,
    data_demand: pd.DataFrame,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    settings: PreprocessingSettings,
    annual_coef: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list, list]:
    """Run the optimization pipeline: generate summary, find optimal cutoffs.

    Args:
        data_booked: Booked applications DataFrame
        data_demand: Demand applications DataFrame
        risk_inference: Risk inference results dictionary
        reg_todu_amt_pile: Trained todu regression model
        stress_factor: Calculated stress factor
        tasa_fin: Financing/transformation rate
        settings: Configuration settings object
        annual_coef: Annual coefficient for the observation period

    Returns:
        Tuple of (data_summary_desagregado, data_summary, data_summary_sample_no_opt,
                  values_var0, values_var1)
    """
    t0 = time.perf_counter()
    segment = settings.segment_filter
    config_dict = settings.model_dump()

    data_summary_desagregado = run_optimization_pipeline(
        data_booked=data_booked,
        data_demand=data_demand,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stressor=stress_factor,
        tasa_fin=tasa_fin,
        config_data=config_dict,
        annual_coef=annual_coef,
    )

    # Cutoff optimization
    values_var0 = sorted(data_summary_desagregado[settings.variables[0]].unique())
    values_var1 = sorted(data_summary_desagregado[settings.variables[1]].unique())

    # Check for fixed cutoffs (skip optimization if provided)
    fixed_cutoffs = settings.fixed_cutoffs
    use_fixed_cutoffs = fixed_cutoffs is not None and len(fixed_cutoffs) > 0

    if use_fixed_cutoffs:
        logger.info(f"[{segment}] Using fixed cutoffs (skipping optimization)")
        logger.debug(f"[{segment}] Fixed cutoffs: {fixed_cutoffs}")

        # Get validation settings from config
        strict_validation = fixed_cutoffs.get("strict_validation", False)
        inv_var1 = settings.inv_var1

        # Create single solution from fixed cutoffs with enhanced validation
        df_v = create_fixed_cutoff_solution(
            fixed_cutoffs=fixed_cutoffs,
            variables=settings.variables,
            values_var0=values_var0,
            values_var1=values_var1,
            strict_validation=strict_validation,
            inv_var1=inv_var1,
        )

        # Calculate KPIs for the fixed cutoff solution
        data_summary = kpi_of_fact_sol(
            df_v=df_v,
            values_var0=values_var0,
            data_sumary_desagregado=data_summary_desagregado,
            variables=settings.variables,
            indicadores=settings.indicators,
            chunk_size=100000,
        )

        # Merge df_v (with bin columns) into data_summary (with KPIs)
        # This is done automatically by get_optimal_solutions in the optimization path,
        # but must be done manually for fixed cutoffs since we skip optimization
        data_summary = data_summary.merge(df_v, on="sol_fac", how="left")

        # Log acceptance rate preview for fixed cutoffs
        if len(data_summary) > 0:
            row = data_summary.iloc[0]
            production = row.get("oa_amt_h0", 0)
            total_demand = data_summary_desagregado["oa_amt_h0"].sum() if "oa_amt_h0" in data_summary_desagregado.columns else 0
            acceptance_rate = (production / total_demand * 100) if total_demand > 0 else 0
            risk_str = ""
            if "todu_30ever_h6" in row and "todu_amt_pile_h6" in row:
                multiplier = settings.multiplier
                risk = calculate_b2_ever_h6(row["todu_30ever_h6"], row["todu_amt_pile_h6"], multiplier=multiplier, as_percentage=True)
                risk_str = f" | risk={risk:.4f}%"
            logger.info(
                f"[{segment}] Fixed cutoff preview | "
                f"production={production:,.0f} | demand={total_demand:,.0f} | "
                f"acceptance={acceptance_rate:.2f}%{risk_str}"
            )

        # For fixed cutoffs, there's only one solution (no sampling needed)
        data_summary_sample_no_opt = data_summary.copy()

        # Save the fixed cutoff solution as the only Pareto solution
        data_summary.to_csv("data/pareto_optimal_solutions.csv", index=False)
        logger.debug(f"[{segment}] Fixed cutoff solution saved to data/pareto_optimal_solutions.csv")

    else:
        # Get feasible solutions
        df_v = get_fact_sol(values_var0=values_var0, values_var1=values_var1, chunk_size=10000)

        # Calculate KPIs for feasible solutions
        data_summary = kpi_of_fact_sol(
            df_v=df_v,
            values_var0=values_var0,
            data_sumary_desagregado=data_summary_desagregado,
            variables=settings.variables,
            indicadores=settings.indicators,
            chunk_size=100000,
        )

        # Sample non-optimal solutions for visualization
        data_summary_sample_no_opt = data_summary.sample(min(10000, len(data_summary)))

        # Find optimal solutions
        data_summary = get_optimal_solutions(
            df_v=df_v,
            data_sumary=data_summary,
            chunk_size=100000,
        )

        # Save all Pareto-optimal solutions (for Cutoff Explorer risk slider)
        data_summary.to_csv("data/pareto_optimal_solutions.csv", index=False)
        logger.debug(f"[{segment}] Pareto solutions saved to data/pareto_optimal_solutions.csv")

    multiplier = settings.multiplier
    data_summary_desagregado["b2_ever_h6"] = calculate_b2_ever_h6(
        data_summary_desagregado["todu_30ever_h6"],
        data_summary_desagregado["todu_amt_pile_h6"],
        multiplier=multiplier,
        as_percentage=True,
    )
    data_summary_desagregado["text"] = data_summary_desagregado.apply(
        lambda x: str("{:,.2f}M".format(x["oa_amt_h0"] / 1000000)) + " " + str("{:.2%}".format(x["b2_ever_h6"] / 100)),
        axis=1,
    )

    elapsed = time.perf_counter() - t0
    mode = "fixed_cutoffs" if use_fixed_cutoffs else "pareto"
    logger.info(
        f"[{segment}] Optimization done | mode={mode} | "
        f"{len(data_summary)} solutions | {len(values_var0)}x{len(values_var1)} grid | {elapsed:.1f}s"
    )

    return data_summary_desagregado, data_summary, data_summary_sample_no_opt, values_var0, values_var1


def run_scenario_analysis(
    scenario_risk: float,
    scenario_name: str,
    *,
    data_summary: pd.DataFrame,
    data_summary_desagregado: pd.DataFrame,
    data_summary_sample_no_opt: pd.DataFrame,
    data_clean: pd.DataFrame,
    data_booked: pd.DataFrame,
    settings: PreprocessingSettings,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    annual_coef_mr: float,
    values_var0: list,
    values_var1: list,
) -> pd.DataFrame:
    """Run scenario analysis for a single risk threshold: visualization, MR processing, audit.

    Args:
        scenario_risk: Risk threshold for this scenario
        scenario_name: Name of the scenario (e.g., "base", "pessimistic", "optimistic")
        data_summary: Pareto-optimal solutions DataFrame
        data_summary_desagregado: Disaggregated summary DataFrame
        data_summary_sample_no_opt: Sample of non-optimal solutions for visualization
        data_clean: Cleaned data DataFrame
        data_booked: Booked applications DataFrame
        settings: Configuration settings object
        risk_inference: Risk inference results dictionary
        reg_todu_amt_pile: Trained todu regression model
        stress_factor: Calculated stress factor
        tasa_fin: Financing/transformation rate
        annual_coef_mr: Annual coefficient for the MR period
        values_var0: Sorted unique values for variable 0
        values_var1: Sorted unique values for variable 1

    Returns:
        Cutoff summary DataFrame for this scenario
    """
    t0 = time.perf_counter()
    segment = settings.segment_filter
    current_risk = float(round(scenario_risk, 1))
    config_dict = settings.model_dump()

    visualizer = RiskProductionVisualizer(
        data_summary=data_summary,
        data_summary_disaggregated=data_summary_desagregado,
        data_summary_sample_no_opt=data_summary_sample_no_opt,
        variables=settings.variables,
        values_var0=values_var0,
        values_var1=values_var1,
        optimum_risk=current_risk,
        tasa_fin=tasa_fin,
    )

    suffix = f"_{scenario_name}"

    # Save outputs
    visualizer.save_html(f"images/risk_production_visualizer{suffix}.html")
    summary_table = visualizer.get_summary_table()
    summary_table.to_csv(f"data/risk_production_summary_table{suffix}.csv", index=False)
    data_summary_desagregado.to_csv(f"data/data_summary_desagregado{suffix}.csv", index=False)
    opt_sol = visualizer.get_selected_solution()
    opt_sol.to_csv(f"data/optimal_solution{suffix}.csv", index=False)
    # Export full efficient frontier for global optimization
    data_summary.to_csv(f"data/efficient_frontier{suffix}.csv", index=False)
    logger.debug(f"[{segment}] Scenario {scenario_name} outputs saved with suffix '{suffix}'")

    # Extract risk and production values from summary table for cutoff summary
    optimum_row = summary_table[summary_table["Metric"] == "Optimum selected"]
    risk_pct = optimum_row["Risk (%)"].values[0] if not optimum_row.empty else None
    production = optimum_row["Production (€)"].values[0] if not optimum_row.empty else None

    # Generate cutoff summary for this scenario
    cutoff_summary = generate_cutoff_summary(
        optimal_solution_df=opt_sol,
        variables=settings.variables,
        segment_name=segment,
        scenario_name=scenario_name,
        risk_value=risk_pct,
        production_value=production,
    )

    if scenario_name == "base":
        # Also save as default filenames for backward compatibility
        visualizer.save_html("images/risk_production_visualizer.html")
        summary_table.to_csv("data/risk_production_summary_table.csv", index=False)
        opt_sol.to_csv("data/optimal_solution.csv", index=False)
        data_summary_desagregado.to_csv("data/data_summary_desagregado.csv", index=False)
        logger.debug(f"[{segment}] Base scenario outputs also saved to default filenames")

        # Base Scenario MR Processing (Default filenames)
        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            config_data=config_dict,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stress_factor=stress_factor,
            tasa_fin=tasa_fin,
            annual_coef=annual_coef_mr,
            optimal_solution_df=opt_sol,
            file_suffix="",
        )

    # Scenario MR Processing
    process_mr_period(
        data_clean=data_clean,
        data_booked=data_booked,
        config_data=config_dict,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stress_factor=stress_factor,
        tasa_fin=tasa_fin,
        annual_coef=annual_coef_mr,
        optimal_solution_df=opt_sol,
        file_suffix=suffix,
    )

    # Generate audit tables for this scenario
    data_main_period = filter_by_date(
        data_clean,
        "mis_date",
        settings.date_ini_book_obs,
        settings.date_fin_book_obs,
    )
    data_mr_period = filter_by_date(
        data_clean,
        "mis_date",
        settings.date_ini_book_obs_mr,
        settings.date_fin_book_obs_mr,
    )

    inv_var1 = settings.inv_var1

    # Calculate n_months for each period (for annualization)
    date_ini_main = settings.get_date("date_ini_book_obs")
    date_fin_main = settings.get_date("date_fin_book_obs")
    n_months_main = (
        (date_fin_main.year - date_ini_main.year) * 12
        + (date_fin_main.month - date_ini_main.month)
        + 1
    )

    date_ini_mr = settings.get_date("date_ini_book_obs_mr")
    date_fin_mr = settings.get_date("date_fin_book_obs_mr")
    n_months_mr = (
        (date_fin_mr.year - date_ini_mr.year) * 12
        + (date_fin_mr.month - date_ini_mr.month)
        + 1
    )

    save_audit_tables(
        data_main=data_main_period,
        data_mr=data_mr_period,
        optimal_solution_df=opt_sol,
        variables=settings.variables,
        scenario_name=scenario_name,
        output_dir="data",
        inv_var1=inv_var1,
        financing_rate=tasa_fin,
        n_months_main=n_months_main,
        n_months_mr=n_months_mr,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Scenario {scenario_name} done | risk={current_risk} | "
        f"main={n_months_main}mo MR={n_months_mr}mo | {elapsed:.1f}s"
    )

    return cutoff_summary


def _build_scenario_list(settings: PreprocessingSettings, use_fixed_cutoffs: bool) -> list[tuple[float, str]]:
    """Build the list of (risk_threshold, name) scenarios to run."""
    base_optimum_risk = settings.optimum_risk
    scenario_step = settings.scenario_step
    segment = settings.segment_filter

    if use_fixed_cutoffs:
        fixed_cutoffs = settings.fixed_cutoffs or {}
        run_all_scenarios = fixed_cutoffs.get("run_all_scenarios", False)
        if run_all_scenarios:
            scenarios = [
                (base_optimum_risk - scenario_step, "pessimistic"),
                (base_optimum_risk, "base"),
                (base_optimum_risk + scenario_step, "optimistic"),
            ]
            logger.debug(f"[{segment}] Fixed cutoffs: running all scenarios")
        else:
            scenarios = [(base_optimum_risk, "base")]
            logger.debug(f"[{segment}] Fixed cutoffs: running base scenario only")
    else:
        scenarios = [
            (base_optimum_risk - scenario_step, "pessimistic"),
            (base_optimum_risk, "base"),
            (base_optimum_risk + scenario_step, "optimistic"),
        ]

    return scenarios


def _compute_mr_annual_coef(settings: PreprocessingSettings) -> float:
    """Compute the annual coefficient for the MR period."""
    date_ini_mr = settings.get_date("date_ini_book_obs_mr")
    date_fin_mr = settings.get_date("date_fin_book_obs_mr")
    annual_coef_mr = calculate_annual_coef(date_ini_book_obs=date_ini_mr, date_fin_book_obs=date_fin_mr)
    logger.debug(f"MR annual_coef={annual_coef_mr:.2f} ({date_ini_mr.date()} to {date_fin_mr.date()})")
    return annual_coef_mr


def _save_cutoff_summaries(cutoff_summaries: list[pd.DataFrame], settings: PreprocessingSettings) -> None:
    """Consolidate and save cutoff summaries across scenarios."""
    segment = settings.segment_filter

    consolidated_cutoffs = consolidate_cutoff_summaries(
        summaries=cutoff_summaries, output_path="data/cutoff_summary_by_segment.csv"
    )

    if not consolidated_cutoffs.empty:
        wide_cutoffs = format_cutoff_summary_table(
            cutoff_summary=consolidated_cutoffs,
            variables=settings.variables,
        )
        wide_cutoffs.to_csv("data/cutoff_summary_wide.csv", index=False)
        logger.debug(f"[{segment}] Cutoff summaries saved to data/cutoff_summary_*.csv")

        logger.info(f"[{segment}] Cutoff summary:\n{wide_cutoffs.to_string()}")


def main(
    config_path: str = "config.toml",
    model_path: str = None,
    training_only: bool = False,
    skip_dq_checks: bool = False,
    preloaded_data: pd.DataFrame = None,
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
    t0_total = time.perf_counter()

    # Step 1: Load and validate configuration
    try:
        settings, date_ini, date_fin, annual_coef = load_and_validate_config(config_path)
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return None

    segment = settings.segment_filter

    # Step 2: Load and prepare data
    try:
        data = load_and_prepare_data(settings, preloaded_data)
    except (DataValidationError, FileNotFoundError) as e:
        logger.error(f"[{segment}] Data error: {e}")
        return None
    except Exception as e:
        logger.error(f"[{segment}] Error loading data: {e}")
        return None

    # Step 3: Preprocessing (DQ checks, binning, stress factor, transformation rate)
    result = run_preprocessing_phase(data, settings, skip_dq_checks)
    if result is None:
        return None
    data_clean, data_booked, data_demand, stress_factor, tasa_fin = result

    # Step 4: Risk inference (model training or loading)
    try:
        risk_inference, reg_todu_amt_pile = run_inference_phase(data_clean, settings, model_path)
    except Exception as e:
        logger.error(f"[{segment}] Inference error: {e}")
        # import traceback; logger.error(traceback.format_exc()) # Debugging aid
        return None

    # Early return for training_only mode (supersegment training)
    if training_only:
        elapsed = time.perf_counter() - t0_total
        logger.info(
            f"[{segment}] Training only complete | "
            f"model={risk_inference.get('model_path', 'models/')} | {elapsed:.1f}s total"
        )
        return data_clean, data_booked, data_demand, risk_inference, reg_todu_amt_pile

    # Step 5: Optimization (feasible solutions, KPIs, Pareto front)
    (data_summary_desagregado, data_summary, data_summary_sample_no_opt,
     values_var0, values_var1) = run_optimization_phase(
        data_booked, data_demand, risk_inference, reg_todu_amt_pile,
        stress_factor, tasa_fin, settings, annual_coef,
    )

    # Step 6: Scenario analysis loop
    use_fixed_cutoffs = settings.fixed_cutoffs is not None and len(settings.fixed_cutoffs) > 0
    scenarios = _build_scenario_list(settings, use_fixed_cutoffs)
    annual_coef_mr = _compute_mr_annual_coef(settings)

    cutoff_summaries = []
    for scenario_risk, scenario_name in scenarios:
        summary = run_scenario_analysis(
            scenario_risk, scenario_name,
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
        )
        cutoff_summaries.append(summary)

    _save_cutoff_summaries(cutoff_summaries, settings)

    elapsed_total = time.perf_counter() - t0_total
    logger.info(f"[{segment}] Pipeline complete | {len(scenarios)} scenarios | {elapsed_total:.1f}s total")

    return data_clean, data_booked, data_demand, data_summary_desagregado, data_summary


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
        help="Only run preprocessing and model training, skip optimization.",
    )

    parser.add_argument(
        "--skip-dq-checks", action="store_true", help="Skip data quality checks (not recommended for production)."
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
