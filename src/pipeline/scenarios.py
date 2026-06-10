"""Scenario-analysis phase: iterates pessimistic/base/optimistic risk targets over the pre-computed Pareto frontier, producing per-scenario RP summary tables, bootstrap CIs, MR validation, audits, and cutoff summaries.

Extracted from ``src/pipeline/optimization.py`` in R2b-iv (todo #63). Exposes :func:`run_scenario_analysis`, :func:`build_scenario_list`, :func:`compute_mr_annual_coef`, :func:`save_cutoff_summaries`.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.audit import (
    generate_audit_table,
    reconcile_risk_production_summary_with_audit,
    save_audit_tables,
    validate_audit_against_summary,
)
from src.config import OutputPaths, PreprocessingSettings
from src.constants import Columns, RejectReason, StatusName
from src.mr_pipeline import process_mr_period
from src.optimization_utils import (
    CellGrid,
    CutoffSpec,
)
from src.plots import RiskProductionVisualizer
from src.preprocess_improved import filter_by_date
from src.utils import (
    calculate_annual_coef,
    calculate_bootstrap_intervals,
    consolidate_cutoff_summaries,
    format_cutoff_summary_table,
    generate_cutoff_summary,
)


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
    grid: CellGrid | None = None,
    pareto_masks: list | None = None,
    output: OutputPaths | None = None,
    total_demand: float = 0.0,
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
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
        grid: CellGrid from MILP optimization (None for legacy/fixed-cutoff paths)
        pareto_masks: List of binary masks from Pareto frontier (None for legacy)

    Returns:
        Cutoff summary DataFrame for this scenario
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter
    current_risk = float(round(scenario_risk, 1))

    visualizer = RiskProductionVisualizer(
        data_summary=data_summary,
        data_summary_disaggregated=data_summary_desagregado,
        data_summary_sample_no_opt=data_summary_sample_no_opt,
        variables=settings.variables,
        optimum_risk=current_risk,
        tasa_fin=tasa_fin,
        values_per_var=values_per_var,
        directions=settings.directions,
        pareto_masks=pareto_masks,
        grid=grid,
        multiplier=settings.multiplier,
        total_demand=total_demand,
    )

    suffix = f"_{scenario_name}"

    # Extract optimal solution for CI calculation
    opt_sol = visualizer.get_selected_solution()
    if opt_sol.empty:
        logger.warning(
            f"[{segment}] Scenario {scenario_name} | risk_threshold={current_risk:.1f}% | "
            "no feasible solution found on Pareto frontier"
        )
        return pd.DataFrame()
    selected_b2 = opt_sol.iloc[0].get("b2_ever_h6", float("nan"))
    selected_prod = opt_sol.iloc[0].get("oa_amt_h0", float("nan"))
    logger.info(
        f"[{segment}] Scenario {scenario_name} | risk_threshold={current_risk:.1f}% | "
        f"selected b2={selected_b2:.2f}% | production={selected_prod:,.0f}"
    )

    inv_var1 = settings.variables[1] in settings.inv_vars if len(settings.variables) > 1 else False
    is_nd = len(settings.variables) != 2

    # Calculate main annual coef for production scaling
    date_ini_main = settings.get_date("date_ini_book_obs")
    date_fin_main = settings.get_date("date_fin_book_obs")
    n_months_main_calc = (
        (date_fin_main.year - date_ini_main.year) * 12 + (date_fin_main.month - date_ini_main.month) + 1
    )
    annual_coef_main = 12 / n_months_main_calc if n_months_main_calc > 0 else 1.0

    # Resolve the selected mask from the Pareto frontier (for N-d classify_by_mask usage)
    selected_mask = None
    if grid is not None and pareto_masks:
        # Match the selected solution's sol_fac back to its mask index
        selected_sol_fac = int(opt_sol.iloc[0].get("sol_fac", 0))
        if 0 <= selected_sol_fac < len(pareto_masks):
            selected_mask = pareto_masks[selected_sol_fac]

    # Build unified CutoffSpec (mask for N-d, cut_map for 2-var).
    spec: CutoffSpec | None = None
    if is_nd and selected_mask is not None and grid is not None:
        spec = CutoffSpec.from_mask(selected_mask, grid)
    elif not is_nd:
        spec = CutoffSpec.from_optimal_solution(
            opt_sol,
            settings.variables,
            data_summary=data_summary_desagregado,
            inv_vars=settings.inv_vars,
        )
    cut_map: dict[float, float] = (spec.as_2d_cut_map() or {}) if spec is not None else {}

    # Calculate repesca_production from data_summary_desagregado
    repesca_production = 0.0
    if "oa_amt_h0_rep" in data_summary_desagregado.columns and spec is not None:
        passes = spec.classify(data_summary_desagregado)
        repesca_production = data_summary_desagregado.loc[passes, "oa_amt_h0_rep"].sum()

    # Score-rejected loans of the main period: lets the bootstrap resample the
    # reject-inferred component so the risk CI matches the blended headline
    # basis instead of bracketing booked-only realized risk (audit #28).
    rejected_loans_main = pd.DataFrame()
    if "oa_amt_h0_rep" in data_summary_desagregado.columns:
        rejected_loans_main = filter_by_date(
            data_clean,
            "mis_date",
            settings.date_ini_book_obs,
            settings.date_fin_book_obs,
        )
        rejected_loans_main = rejected_loans_main[
            (rejected_loans_main[Columns.STATUS_NAME] == StatusName.REJECTED.value)
            & (rejected_loans_main[Columns.REJECT_REASON] == RejectReason.SCORE.value)
        ]

    ci_data = calculate_bootstrap_intervals(
        data_booked=data_booked,
        cut_map=cut_map,
        variables=settings.variables,
        multiplier=settings.multiplier,
        n_bootstraps=settings.n_bootstraps,
        inv_var1=inv_var1,
        annual_coef=annual_coef_main,
        repesca_production=repesca_production,
        mask=selected_mask,
        grid=grid,
        rejected_loans=rejected_loans_main,
        rep_cells=data_summary_desagregado,
    )
    logger.debug(f"[{segment}] Scenario {scenario_name} CI: {ci_data}")

    summary_table = visualizer.get_summary_table()

    # Loan-level audit totals are canonical for production (€): reconcile summary before save
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

    n_months_main = n_months_main_calc
    if settings.date_ini_book_obs_mr is not None and settings.date_fin_book_obs_mr is not None:
        date_ini_mr = settings.get_date("date_ini_book_obs_mr")
        date_fin_mr = settings.get_date("date_fin_book_obs_mr")
        n_months_mr = (date_fin_mr.year - date_ini_mr.year) * 12 + (date_fin_mr.month - date_ini_mr.month) + 1
    else:
        n_months_mr = None

    audit_main = generate_audit_table(
        data=data_main_period,
        optimal_solution_df=opt_sol,
        variables=settings.variables,
        financing_rate=tasa_fin,
        inv_var1=inv_var1,
        n_months=n_months_main,
        mask=selected_mask,
        grid=grid,
    )
    if len(data_mr_period) > 0:
        audit_mr = generate_audit_table(
            data=data_mr_period,
            optimal_solution_df=opt_sol,
            variables=settings.variables,
            financing_rate=tasa_fin,
            inv_var1=inv_var1,
            n_months=n_months_mr,
            mask=selected_mask,
            grid=grid,
        )
    else:
        audit_mr = pd.DataFrame()

    # In baseline mode the accept-all mask makes every rejected applicant look like
    # swap-in, producing misleading audit classifications.  Skip reconciliation so
    # the summary table keeps the correct Optimum = Actual / zero-swap values.
    if not settings.baseline_mode:
        summary_table = reconcile_risk_production_summary_with_audit(summary_table, audit_main)
        validate_audit_against_summary(audit_main, summary_table)

    # Add CI columns to summary table (only for Optimum selected row; others stay NaN).
    # risk_ci_* is on the blended booked+RI basis (matching the headline) when the
    # repesca context was available; risk_ci_basis records which basis was used and
    # risk_booked_ci_* always carries the booked-only realized CI (audit #28).
    summary_table["production_ci_lower"] = np.nan
    summary_table["production_ci_upper"] = np.nan
    summary_table["risk_ci_lower"] = np.nan
    summary_table["risk_ci_upper"] = np.nan
    summary_table["risk_ci_basis"] = None
    summary_table["risk_booked_ci_lower"] = np.nan
    summary_table["risk_booked_ci_upper"] = np.nan

    if ci_data:
        mask_opt = summary_table["Metric"] == "Optimum selected"
        if mask_opt.any():
            summary_table.loc[mask_opt, "production_ci_lower"] = ci_data.get("production_ci_lower", 0.0)
            summary_table.loc[mask_opt, "production_ci_upper"] = ci_data.get("production_ci_upper", 0.0)
            summary_table.loc[mask_opt, "risk_ci_lower"] = ci_data.get("risk_ci_lower", 0.0)
            summary_table.loc[mask_opt, "risk_ci_upper"] = ci_data.get("risk_ci_upper", 0.0)
            summary_table.loc[mask_opt, "risk_ci_basis"] = ci_data.get("risk_ci_basis", "booked_realized")
            summary_table.loc[mask_opt, "risk_booked_ci_lower"] = ci_data.get("risk_booked_ci_lower", np.nan)
            summary_table.loc[mask_opt, "risk_booked_ci_upper"] = ci_data.get("risk_booked_ci_upper", np.nan)

    # Add swap-in risk adjustment diagnostics to summary table (Swap-in row only)
    for diag_col, diag_label in [
        ("ri_multiplier_rep", "ri_multiplier"),
        ("stress_factor_rep", "stress_factor"),
    ]:
        summary_table[f"{diag_label}_min"] = None
        summary_table[f"{diag_label}_avg"] = None
        summary_table[f"{diag_label}_max"] = None
        if diag_col in data_summary_desagregado.columns:
            vals = data_summary_desagregado[diag_col].dropna()
            if not vals.empty:
                mask_swapin = summary_table["Metric"] == "Swap-in"
                if mask_swapin.any():
                    summary_table.loc[mask_swapin, f"{diag_label}_min"] = vals.min()
                    summary_table.loc[mask_swapin, f"{diag_label}_avg"] = vals.mean()
                    summary_table.loc[mask_swapin, f"{diag_label}_max"] = vals.max()

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
        mask=selected_mask if is_nd else None,
        grid=grid if is_nd else None,
    )

    # Generate acceptance grid visualization for N>2
    if is_nd and selected_mask is not None and grid is not None:
        try:
            from src.plots import plot_acceptance_grid_nd

            plot_acceptance_grid_nd(
                mask=selected_mask,
                grid=grid,
                output_path=output.acceptance_grid_html(suffix),
                multiplier=settings.multiplier,
                inv_vars=settings.inv_vars,
            )
        except Exception as e:
            logger.warning(f"[{segment}] Acceptance grid plot failed (non-blocking): {e}")

    # Save accepted cell coordinates for sequential cutoff ordering
    if scenario_name == "base" and grid is not None and selected_mask is not None:
        accepted_coords = []
        for coord, idx in grid.cell_index.items():
            if selected_mask[idx] == 1:
                accepted_coords.append({var: float(val) for var, val in zip(settings.variables, coord)})
        if accepted_coords:
            pd.DataFrame(accepted_coords).to_csv(output.accepted_cells_csv(suffix), index=False)
            logger.debug(f"[{segment}] Saved {len(accepted_coords)} accepted cells for cutoff ordering")

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
            mask=selected_mask,
            grid=grid,
            per_bin_stress=per_bin_stress,
            per_bin_tasa_fin=per_bin_tasa_fin,
            audit_mr_df=audit_mr if len(audit_mr) else None,
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
        mask=selected_mask,
        grid=grid,
        per_bin_stress=per_bin_stress,
        per_bin_tasa_fin=per_bin_tasa_fin,
        audit_mr_df=audit_mr if len(audit_mr) else None,
    )

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
            mask=selected_mask,
            grid=grid,
            audit_main=audit_main,
            audit_mr=audit_mr if len(audit_mr) else None,
        )
    except Exception as e:
        logger.error(f"[{segment}] Audit table generation failed for {scenario_name} (non-blocking): {e}")

    elapsed = time.perf_counter() - t0
    logger.info(
        f"[{segment}] Scenario {scenario_name} done | risk={current_risk} | "
        f"main={n_months_main}mo MR={n_months_mr}mo | {elapsed:.1f}s"
    )

    return cutoff_summary


def build_scenario_list(settings: PreprocessingSettings, use_fixed_cutoffs: bool) -> list[tuple[float, str]]:
    """Build the list of (risk_threshold, name) scenarios to run."""
    base_optimum_risk = settings.optimum_risk
    scenario_step = settings.risk_step
    segment = settings.segment_filter

    if settings.baseline_mode or settings.base_scenario_only:
        scenarios = [(base_optimum_risk, "base")]
        reason = "baseline mode" if settings.baseline_mode else "base_scenario_only"
        logger.debug(f"[{segment}] {reason}: running base scenario only")
        return scenarios

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


def compute_mr_annual_coef(settings: PreprocessingSettings) -> float:
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


def save_cutoff_summaries(
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
        if len(settings.variables) != 2:
            # N>2: cell-level summary, skip pivot (no cutoff_value column)
            consolidated_cutoffs.to_csv(output.cutoff_summary_wide_csv, index=False)
            logger.debug(f"[{segment}] Cell-level cutoff summaries saved to {output.cutoff_summary_wide_csv}")
            accepted_count = consolidated_cutoffs["accepted"].sum() if "accepted" in consolidated_cutoffs.columns else 0
            total_cells = len(consolidated_cutoffs)
            logger.debug(f"[{segment}] Cell-level cutoff summary: {int(accepted_count)}/{total_cells} cells accepted")
        else:
            wide_cutoffs = format_cutoff_summary_table(
                cutoff_summary=consolidated_cutoffs,
                variables=settings.variables,
            )
            wide_cutoffs.to_csv(output.cutoff_summary_wide_csv, index=False)
            logger.debug(f"[{segment}] Cutoff summaries saved to {output.cutoff_summary_by_segment_csv}")
            logger.debug(f"[{segment}] Cutoff summary:\n{wide_cutoffs.to_string()}")
