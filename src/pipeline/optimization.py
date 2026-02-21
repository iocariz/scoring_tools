import time
from typing import Any

import pandas as pd
from loguru import logger

from src.audit import save_audit_tables
from src.config import OutputPaths, PreprocessingSettings
from src.inference_optimized import run_optimization_pipeline
from src.mr_pipeline import process_mr_period
from src.optimization_utils import (
    add_bin_columns,
    create_fixed_cutoff_solution,
    get_fact_sol,
    get_optimal_solutions,
    kpi_of_fact_sol,
    trace_pareto_frontier,
)
from src.plots import RiskProductionVisualizer
from src.preprocess_improved import filter_by_date
from src.utils import (
    calculate_annual_coef,
    calculate_b2_ever_h6,
    calculate_bootstrap_intervals,
    consolidate_cutoff_summaries,
    format_cutoff_summary_table,
    generate_cutoff_summary,
)


def run_optimization_phase(
    data_booked: pd.DataFrame,
    data_demand: pd.DataFrame,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    settings: PreprocessingSettings,
    annual_coef: float,
    output: OutputPaths | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list]]:
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
        output: Output paths configuration. Defaults to current directory.

    Returns:
        Tuple of (data_summary_desagregado, data_summary, data_summary_sample_no_opt,
                  values_per_var)
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter

    data_summary_desagregado = run_optimization_pipeline(
        data_booked=data_booked,
        data_demand=data_demand,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stressor=stress_factor,
        tasa_fin=tasa_fin,
        indicators=settings.indicators,
        variables=settings.variables,
        annual_coef=annual_coef,
        b2_output_path=output.b2_visualization_html,
        reject_inference_method=settings.reject_inference_method,
        reject_uplift_factor=settings.reject_uplift_factor,
        reject_max_risk_multiplier=settings.reject_max_risk_multiplier,
    )

    # Build values_per_var dict for all variables
    values_per_var = {var: sorted(data_summary_desagregado[var].unique()) for var in settings.variables}

    # Shorthand for 2-var backward compat
    values_var0 = values_per_var[settings.variables[0]]
    values_var1 = values_per_var[settings.variables[1]] if len(settings.variables) > 1 else []

    # Check for fixed cutoffs (skip optimization if provided)
    fixed_cutoffs = settings.fixed_cutoffs
    use_fixed_cutoffs = fixed_cutoffs is not None and len(fixed_cutoffs) > 0

    if use_fixed_cutoffs:
        logger.info(f"[{segment}] Using fixed cutoffs (skipping optimization)")
        logger.debug(f"[{segment}] Fixed cutoffs: {fixed_cutoffs}")

        # Get validation settings from config
        strict_validation = fixed_cutoffs.get("strict_validation", False)
        inv_var1 = settings.variables[1] in settings.inv_vars

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
        data_summary = data_summary.merge(df_v, on="sol_fac", how="left")

        # Log acceptance rate preview for fixed cutoffs
        if len(data_summary) > 0:
            row = data_summary.iloc[0]
            production = row.get("oa_amt_h0", 0)
            total_demand = (
                data_summary_desagregado["oa_amt_h0"].sum() if "oa_amt_h0" in data_summary_desagregado.columns else 0
            )
            acceptance_rate = (production / total_demand * 100) if total_demand > 0 else 0
            risk_str = ""
            if "todu_30ever_h6" in row and "todu_amt_pile_h6" in row:
                multiplier = settings.multiplier
                risk = calculate_b2_ever_h6(
                    row["todu_30ever_h6"], row["todu_amt_pile_h6"], multiplier=multiplier, as_percentage=True
                )
                risk_str = f" | risk={risk:.4f}%"
            logger.info(
                f"[{segment}] Fixed cutoff preview | "
                f"production={production:,.0f} | demand={total_demand:,.0f} | "
                f"acceptance={acceptance_rate:.2f}%{risk_str}"
            )

        # For fixed cutoffs, there's only one solution (no sampling needed)
        data_summary_sample_no_opt = data_summary.copy()

        # Save the fixed cutoff solution as the only Pareto solution
        data_summary.to_csv(output.pareto_solutions_csv, index=False)
        logger.debug(f"[{segment}] Fixed cutoff solution saved to {output.pareto_solutions_csv}")

    else:
        # MILP-based Pareto frontier optimization
        pareto_df, grid, masks = trace_pareto_frontier(
            data_summary_desagregado=data_summary_desagregado,
            variables=settings.variables,
            inv_vars=settings.inv_vars,
            multiplier=settings.multiplier,
            indicators=settings.indicators,
        )

        if pareto_df.empty:
            # Fallback to legacy enumeration if MILP produces no solutions
            logger.warning(f"[{segment}] MILP produced no solutions, falling back to legacy enumeration")
            df_v = get_fact_sol(values_var0=values_var0, values_var1=values_var1, chunk_size=10000)
            data_summary = kpi_of_fact_sol(
                df_v=df_v,
                values_var0=values_var0,
                data_sumary_desagregado=data_summary_desagregado,
                variables=settings.variables,
                indicadores=settings.indicators,
                chunk_size=100000,
            )
            data_summary_sample_no_opt = data_summary.sample(min(10000, len(data_summary)))
            data_summary = get_optimal_solutions(df_v=df_v, data_sumary=data_summary, chunk_size=100000)
        else:
            # Add bin columns for 2-var backward compat (cutoff extraction, viz, bootstrap)
            data_summary = add_bin_columns(pareto_df, masks, grid, settings.inv_vars)

            # MILP doesn't enumerate all solutions, so non-optimal sample is empty
            data_summary_sample_no_opt = pd.DataFrame(columns=["oa_amt_h0", "b2_ever_h6"])

        # Save all Pareto-optimal solutions
        data_summary.to_csv(output.pareto_solutions_csv, index=False)
        logger.debug(f"[{segment}] Pareto solutions saved to {output.pareto_solutions_csv}")

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
    mode = "fixed_cutoffs" if use_fixed_cutoffs else "milp_pareto"
    b2_min = data_summary["b2_ever_h6"].min() if not data_summary.empty else 0
    b2_max = data_summary["b2_ever_h6"].max() if not data_summary.empty else 0
    grid_desc = "x".join(str(len(values_per_var[v])) for v in settings.variables)
    logger.info(
        f"[{segment}] Optimization done | mode={mode} | "
        f"{len(data_summary)} solutions | {grid_desc} grid | "
        f"b2 range: [{b2_min:.2f}%, {b2_max:.2f}%] | optimum_risk={settings.optimum_risk:.1f}% | {elapsed:.1f}s"
    )

    return data_summary_desagregado, data_summary, data_summary_sample_no_opt, values_per_var


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
    values_per_var: dict[str, list],
    output: OutputPaths | None = None,
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
        values_per_var: Dict mapping variable names to sorted unique bin values

    Returns:
        Cutoff summary DataFrame for this scenario
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter
    current_risk = float(round(scenario_risk, 1))

    # Extract values_var0 for 2-var backward compat (bootstrap, cutoff extraction)
    values_var0 = values_per_var[settings.variables[0]]

    visualizer = RiskProductionVisualizer(
        data_summary=data_summary,
        data_summary_disaggregated=data_summary_desagregado,
        data_summary_sample_no_opt=data_summary_sample_no_opt,
        variables=settings.variables,
        optimum_risk=current_risk,
        tasa_fin=tasa_fin,
        values_per_var=values_per_var,
        directions=settings.directions,
    )

    suffix = f"_{scenario_name}"

    # Extract optimal solution for CI calculation
    opt_sol = visualizer.get_selected_solution()
    selected_b2 = opt_sol.iloc[0].get("b2_ever_h6", float("nan"))
    selected_prod = opt_sol.iloc[0].get("oa_amt_h0", float("nan"))
    logger.info(
        f"[{segment}] Scenario {scenario_name} | risk_threshold={current_risk:.1f}% | "
        f"selected b2={selected_b2:.2f}% | production={selected_prod:,.0f}"
    )

    # Create mapping of cuts from optimal solution
    cut_map = {}
    row = opt_sol.iloc[0]
    for bin_val in values_var0:
        if bin_val in row:
            cut_map[float(bin_val)] = float(row[bin_val])
        elif str(bin_val) in row:
            cut_map[float(bin_val)] = float(row[str(bin_val)])
        elif str(float(bin_val)) in row:
            cut_map[float(bin_val)] = float(row[str(float(bin_val))])

    inv_var1 = settings.variables[1] in settings.inv_vars
    # Pass model CV SE so bootstrap CI accounts for model prediction uncertainty
    model_cv_se = risk_inference.get("cv_std_r2") if risk_inference else None
    ci_data = calculate_bootstrap_intervals(
        data_booked=data_booked,
        cut_map=cut_map,
        variables=settings.variables,
        multiplier=settings.multiplier,
        n_bootstraps=1000,
        inv_var1=inv_var1,
        model_cv_se_risk=model_cv_se,
    )
    logger.info(f"[{segment}] Scenario {scenario_name} CI: {ci_data}")

    summary_table = visualizer.get_summary_table()

    # Add CI columns to summary table (only for Optimum selected row)
    summary_table["production_ci_lower"] = 0.0
    summary_table["production_ci_upper"] = 0.0
    summary_table["risk_ci_lower"] = 0.0
    summary_table["risk_ci_upper"] = 0.0

    if ci_data:
        mask_opt = summary_table["Metric"] == "Optimum selected"
        if mask_opt.any():
            summary_table.loc[mask_opt, "production_ci_lower"] = ci_data.get("production_ci_lower", 0.0)
            summary_table.loc[mask_opt, "production_ci_upper"] = ci_data.get("production_ci_upper", 0.0)
            # CI risk values from bootstrap are raw ratio (multiplier * num / den).
            # Convert to percentage to match the Risk (%) column.
            summary_table.loc[mask_opt, "risk_ci_lower"] = ci_data.get("risk_ci_lower", 0.0) * 100
            summary_table.loc[mask_opt, "risk_ci_upper"] = ci_data.get("risk_ci_upper", 0.0) * 100

    # Save outputs
    visualizer.save_html(output.risk_production_visualizer_html(suffix))
    summary_table.to_csv(output.risk_production_summary_csv(suffix), index=False)
    data_summary_desagregado.to_csv(output.data_summary_desagregado_csv(suffix), index=False)
    opt_sol.to_csv(output.optimal_solution_csv(suffix), index=False)
    # Export full efficient frontier for global optimization
    data_summary.to_csv(output.efficient_frontier_csv(suffix), index=False)
    logger.debug(f"[{segment}] Scenario {scenario_name} outputs saved with suffix '{suffix}'")

    # Extract risk and production values from summary table for cutoff summary
    optimum_row = summary_table[summary_table["Metric"] == "Optimum selected"]
    risk_pct = optimum_row["Risk (%)"].values[0] if not optimum_row.empty else None
    production = optimum_row["Production (€)"].values[0] if not optimum_row.empty else None

    # Calculate confidence intervals (Already done above)
    # cut_map logic removed as it's duped

    # Generate cutoff summary for this scenario
    cutoff_summary = generate_cutoff_summary(
        optimal_solution_df=opt_sol,
        variables=settings.variables,
        segment_name=segment,
        scenario_name=scenario_name,
        risk_value=risk_pct,
        production_value=production,
        ci_data=ci_data,
    )

    if scenario_name == "base":
        # Also save as default filenames for backward compatibility
        visualizer.save_html(output.risk_production_visualizer_html())
        summary_table.to_csv(output.risk_production_summary_csv(), index=False)
        opt_sol.to_csv(output.optimal_solution_csv(), index=False)
        data_summary_desagregado.to_csv(output.data_summary_desagregado_csv(), index=False)
        logger.debug(f"[{segment}] Base scenario outputs also saved to default filenames")

        # Base Scenario MR Processing (Default filenames)
        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stress_factor=stress_factor,
            tasa_fin=tasa_fin,
            annual_coef=annual_coef_mr,
            optimal_solution_df=opt_sol,
            file_suffix="",
            output=output,
        )

    # Scenario MR Processing
    process_mr_period(
        data_clean=data_clean,
        data_booked=data_booked,
        settings=settings,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stress_factor=stress_factor,
        tasa_fin=tasa_fin,
        annual_coef=annual_coef_mr,
        optimal_solution_df=opt_sol,
        file_suffix=suffix,
        output=output,
    )

    # Generate audit tables for this scenario
    data_main_period = filter_by_date(
        data_clean,
        "mis_date",
        settings.date_ini_book_obs,
        settings.date_fin_book_obs,
    )
    if settings.date_ini_book_obs_mr is not None and settings.date_fin_book_obs_mr is not None:
        data_mr_period = filter_by_date(
            data_clean,
            "mis_date",
            settings.date_ini_book_obs_mr,
            settings.date_fin_book_obs_mr,
        )
    else:
        data_mr_period = pd.DataFrame(columns=data_clean.columns)

    inv_var1 = settings.variables[1] in settings.inv_vars

    # Calculate n_months for each period (for annualization)
    date_ini_main = settings.get_date("date_ini_book_obs")
    date_fin_main = settings.get_date("date_fin_book_obs")
    n_months_main = (date_fin_main.year - date_ini_main.year) * 12 + (date_fin_main.month - date_ini_main.month) + 1

    if settings.date_ini_book_obs_mr is not None and settings.date_fin_book_obs_mr is not None:
        date_ini_mr = settings.get_date("date_ini_book_obs_mr")
        date_fin_mr = settings.get_date("date_fin_book_obs_mr")
        n_months_mr = (date_fin_mr.year - date_ini_mr.year) * 12 + (date_fin_mr.month - date_ini_mr.month) + 1
    else:
        n_months_mr = None

    try:
        save_audit_tables(
            data_main=data_main_period,
            data_mr=data_mr_period,
            optimal_solution_df=opt_sol,
            variables=settings.variables,
            scenario_name=scenario_name,
            output_dir=str(output.data_dir),
            inv_var1=inv_var1,
            financing_rate=tasa_fin,
            n_months_main=n_months_main,
            n_months_mr=n_months_mr,
        )
    except Exception as e:
        logger.error(f"[{segment}] Audit table generation failed for {scenario_name} (non-blocking): {e}")

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Scenario {scenario_name} done | risk={current_risk} | "
        f"main={n_months_main}mo MR={n_months_mr}mo | {elapsed:.1f}s"
    )

    return cutoff_summary


def _build_scenario_list(settings: PreprocessingSettings, use_fixed_cutoffs: bool) -> list[tuple[float, str]]:
    """Build the list of (risk_threshold, name) scenarios to run."""
    base_optimum_risk = settings.optimum_risk
    scenario_step = settings.risk_step
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
    """Compute the annual coefficient for the MR period.

    Returns 1.0 if MR dates are not configured.
    """
    if settings.date_ini_book_obs_mr is None or settings.date_fin_book_obs_mr is None:
        logger.warning("MR dates not configured — using annual_coef_mr=1.0")
        return 1.0
    date_ini_mr = settings.get_date("date_ini_book_obs_mr")
    date_fin_mr = settings.get_date("date_fin_book_obs_mr")
    annual_coef_mr = calculate_annual_coef(date_ini_book_obs=date_ini_mr, date_fin_book_obs=date_fin_mr)
    logger.debug(f"MR annual_coef={annual_coef_mr:.2f} ({date_ini_mr.date()} to {date_fin_mr.date()})")
    return annual_coef_mr


def run_sensitivity_phase(
    data_summary_desagregado: pd.DataFrame,
    data_summary: pd.DataFrame,
    settings: PreprocessingSettings,
    output: OutputPaths | None = None,
) -> None:
    """Run sensitivity analysis on the base scenario (non-blocking).

    Gated behind ``settings.run_sensitivity``. Saves results to CSV.

    Args:
        data_summary_desagregado: Aggregated summary data.
        data_summary: Pareto-optimal solutions DataFrame.
        settings: Configuration settings.
        output: Output paths configuration.
    """
    if not settings.run_sensitivity:
        return

    if output is None:
        output = OutputPaths()

    segment = settings.segment_filter
    logger.info(f"[{segment}] Running sensitivity analysis...")

    try:
        from src.optimization_utils import CellGrid, milp_solve_cutoffs
        from src.sensitivity import compute_cell_marginal_impact, run_sensitivity_analysis, sensitivity_cell_detail

        grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)

        # Get baseline mask from the base scenario's optimal solution
        baseline_mask = milp_solve_cutoffs(grid, settings.optimum_risk, settings.inv_vars, settings.multiplier)
        if baseline_mask is None:
            logger.warning(f"[{segment}] Sensitivity: baseline solve infeasible, skipping")
            return

        # Run sensitivity analysis
        sens_df = run_sensitivity_analysis(
            data_summary_desagregado,
            settings.variables,
            settings.inv_vars,
            settings.multiplier,
            settings.indicators,
            baseline_mask,
            settings.optimum_risk,
            perturbation_levels=settings.sensitivity_levels,
        )
        sens_path = output.sensitivity_analysis_csv("_base")
        sens_df.to_csv(sens_path, index=False)
        logger.info(f"[{segment}] Sensitivity analysis saved to {sens_path}")

        # Cell-level sensitivity detail
        cell_detail = sensitivity_cell_detail(
            data_summary_desagregado,
            settings.variables,
            settings.inv_vars,
            settings.multiplier,
            settings.indicators,
            baseline_mask,
            settings.optimum_risk,
            perturbation_levels=settings.sensitivity_levels,
        )
        cell_detail_path = output.sensitivity_analysis_csv("_cell_detail")
        cell_detail.to_csv(cell_detail_path, index=False)

        # Marginal impact
        marginal_df = compute_cell_marginal_impact(grid, baseline_mask, settings.indicators, settings.multiplier)
        marginal_path = output.cell_marginal_impact_csv("_base")
        marginal_df.to_csv(marginal_path, index=False)
        logger.info(f"[{segment}] Marginal impact saved to {marginal_path}")

    except Exception as e:
        logger.error(f"[{segment}] Sensitivity analysis failed (non-blocking): {e}")


def _save_cutoff_summaries(
    cutoff_summaries: list[pd.DataFrame],
    settings: PreprocessingSettings,
    output: OutputPaths | None = None,
) -> None:
    """Consolidate and save cutoff summaries across scenarios."""
    if output is None:
        output = OutputPaths()

    segment = settings.segment_filter

    consolidated_cutoffs = consolidate_cutoff_summaries(
        summaries=cutoff_summaries, output_path=output.cutoff_summary_by_segment_csv
    )

    if not consolidated_cutoffs.empty:
        wide_cutoffs = format_cutoff_summary_table(
            cutoff_summary=consolidated_cutoffs,
            variables=settings.variables,
        )
        wide_cutoffs.to_csv(output.cutoff_summary_wide_csv, index=False)
        logger.debug(f"[{segment}] Cutoff summaries saved to {output.cutoff_summary_by_segment_csv}")

        logger.info(f"[{segment}] Cutoff summary:\n{wide_cutoffs.to_string()}")
