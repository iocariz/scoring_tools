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
from src.inference_optimized import run_optimization_pipeline
from src.mr_pipeline import process_mr_period
from src.optimization_utils import (
    CellGrid,
    add_bin_columns,
    classify_by_mask,
    create_fixed_cutoff_mask,
    create_fixed_cutoff_solution,
    evaluate_solution,
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
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
    floor_cells_path: str | None = None,
    floor_cells_mode: str = "floor",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, list], CellGrid | None, list]:
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
        per_bin_stress: Optional per-bin stress factors DataFrame.
        per_bin_tasa_fin: Optional per-bin transformation rate DataFrame.
        floor_cells_path: Optional path to CSV of accepted cell coordinates
            from a previous segment. Used for sequential cutoff ordering.
        floor_cells_mode: How to interpret the CSV cells:
            ``"floor"`` (bottom-up) — cells listed must be accepted.
            ``"ceiling"`` (top-down) — only cells listed may be accepted;
            all others are forced to rejected.

    Returns:
        Tuple of (data_summary_desagregado, data_summary, data_summary_sample_no_opt,
                  values_per_var, grid, pareto_masks)
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
        reject_parceling_method=settings.reject_parceling_method,
        reject_bayesian_smoothing=settings.reject_bayesian_smoothing,
        reject_bayesian_prior_strength=settings.reject_bayesian_prior_strength,
        reject_enforce_monotonicity=settings.reject_enforce_monotonicity,
        reject_include_all_rejections=settings.reject_include_all_rejections,
        reject_acceptance_recent_months=settings.reject_acceptance_recent_months,
        reject_acceptance_decay_half_life_months=settings.reject_acceptance_decay_half_life_months,
        reject_acceptance_date_col=settings.reject_acceptance_date_col,
        reject_apply_h3_multiplier=settings.reject_apply_h3_multiplier,
        multiplier=settings.multiplier,
        inv_vars=settings.inv_vars,
        per_bin_stress=per_bin_stress,
        per_bin_tasa_fin=per_bin_tasa_fin,
    )

    # Build values_per_var dict for all variables
    values_per_var = {var: sorted(data_summary_desagregado[var].unique()) for var in settings.variables}

    # Shorthand for 2-var backward compat
    values_var0 = values_per_var[settings.variables[0]]
    values_var1 = values_per_var[settings.variables[1]] if len(settings.variables) > 1 else []

    # Check for fixed cutoffs (skip optimization if provided)
    fixed_cutoffs = settings.fixed_cutoffs
    use_fixed_cutoffs = fixed_cutoffs is not None and len(fixed_cutoffs) > 0

    grid = None
    pareto_masks: list = []

    # Check for baseline mode (show current portfolio as-is, no optimization)
    if settings.baseline_mode:
        logger.info(f"[{segment}] Baseline mode: showing current portfolio (no optimization)")
        grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)
        accept_all_mask = np.ones(len(grid.cell_data), dtype=int)

        # Build single solution with booked-only metrics (zero repesca/cut)
        baseline_kpis: dict[str, float] = {"sol_fac": 0}
        cell = grid.cell_data
        for ind in settings.indicators:
            boo_col = f"{ind}_boo"
            rep_col = f"{ind}_rep"
            if boo_col in cell.columns:
                baseline_kpis[boo_col] = float(cell[boo_col].sum())
                baseline_kpis[ind] = float(cell[boo_col].sum())
            if rep_col in cell.columns:
                baseline_kpis[rep_col] = 0.0
            baseline_kpis[f"{ind}_cut"] = 0.0

        # Compute derived b2 metrics for each suffix group.
        # For _rep/_cut suffixes both numerator and denominator are 0 by design,
        # producing NaN from calculate_b2_ever_h6; replace with 0.0 for display.
        for suffix in ["", "_boo", "_rep", "_cut"]:
            t30_key = f"todu_30ever_h6{suffix}"
            tamt_key = f"todu_amt_pile_h6{suffix}"
            if t30_key in baseline_kpis and tamt_key in baseline_kpis:
                raw = float(
                    calculate_b2_ever_h6(
                        baseline_kpis[t30_key], baseline_kpis[tamt_key],
                        multiplier=settings.multiplier, as_percentage=True,
                    )
                )
                baseline_kpis[f"b2_ever_h6{suffix}"] = 0.0 if np.isnan(raw) else raw
            t30_h3 = f"todu_30ever_h3{suffix}"
            tamt_h3 = f"todu_amt_pile_h3{suffix}"
            if t30_h3 in baseline_kpis and tamt_h3 in baseline_kpis:
                raw_h3 = float(
                    calculate_b2_ever_h6(
                        baseline_kpis[t30_h3], baseline_kpis[tamt_h3],
                        multiplier=settings.multiplier_h3, as_percentage=True,
                    )
                )
                baseline_kpis[f"b2_ever_h3{suffix}"] = 0.0 if np.isnan(raw_h3) else raw_h3

        pareto_masks = [accept_all_mask]
        data_summary = pd.DataFrame([baseline_kpis])
        data_summary = add_bin_columns(data_summary, pareto_masks, grid, settings.inv_vars)
        data_summary_sample_no_opt = data_summary.copy()
        data_summary.to_csv(output.pareto_solutions_csv, index=False)

        # Save accepted cells for sequential cutoff ordering (downstream segments may depend on this)
        accepted_coords = [
            {var: float(val) for var, val in zip(settings.variables, coord)}
            for coord, idx in grid.cell_index.items()
            if accept_all_mask[idx] == 1
        ]
        if accepted_coords:
            pd.DataFrame(accepted_coords).to_csv(output.accepted_cells_csv("_base"), index=False)
            logger.debug(f"[{segment}] Baseline: saved {len(accepted_coords)} accepted cells")

    elif use_fixed_cutoffs:
        logger.info(f"[{segment}] Using fixed cutoffs (skipping optimization)")
        logger.debug(f"[{segment}] Fixed cutoffs: {fixed_cutoffs}")

        # Get validation settings from config
        strict_validation = fixed_cutoffs.get("strict_validation", False)

        if len(settings.variables) != 2:
            # N!=2 fixed cutoffs → mask-based path
            grid, fixed_mask = create_fixed_cutoff_mask(
                fixed_cutoffs=fixed_cutoffs,
                variables=settings.variables,
                data_summary_desagregado=data_summary_desagregado,
                inv_vars=settings.inv_vars,
                strict_validation=strict_validation,
            )
            kpis = evaluate_solution(
                fixed_mask, grid, settings.indicators, settings.multiplier, multiplier_h3=settings.multiplier_h3
            )
            data_summary = pd.DataFrame([{**kpis, "sol_fac": 0}])
            pareto_masks = [fixed_mask]
            data_summary = add_bin_columns(data_summary, pareto_masks, grid, settings.inv_vars)
            data_summary_sample_no_opt = data_summary.copy()
        else:
            # 2-var fixed cutoffs → legacy path
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
                multiplier=settings.multiplier,
                multiplier_h3=settings.multiplier_h3,
            )

            # Merge df_v (with bin columns) into data_summary (with KPIs)
            data_summary = data_summary.merge(df_v, on="sol_fac", how="left")

            # For fixed cutoffs, there's only one solution (no sampling needed)
            data_summary_sample_no_opt = data_summary.copy()

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

        # Save the fixed cutoff solution as the only Pareto solution
        data_summary.to_csv(output.pareto_solutions_csv, index=False)
        logger.debug(f"[{segment}] Fixed cutoff solution saved to {output.pareto_solutions_csv}")

    else:
        # Load constraint from previous segment (sequential cutoff ordering)
        floor_fixed_cells = None
        if floor_cells_path:
            floor_grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)
            floor_df = pd.read_csv(floor_cells_path)
            # Drop rows with NaN coordinates (would never match any grid cell)
            n_raw = len(floor_df)
            floor_df = floor_df.dropna(subset=settings.variables)
            n_nan_rows = n_raw - len(floor_df)
            if n_nan_rows > 0:
                logger.warning(
                    f"[{segment}] Floor cells CSV: dropped {n_nan_rows} row(s) with NaN coordinates"
                )
            # Build a float-normalized lookup to handle int/float type mismatches
            normalized_index = {
                tuple(float(v) for v in coord): idx
                for coord, idx in floor_grid.cell_index.items()
            }
            n_floor_rows = len(floor_df)

            if floor_cells_mode == "ceiling":
                # Top-down: cells in CSV are the ALLOWED set; everything else is rejected
                allowed_indices = set()
                for _, row in floor_df.iterrows():
                    coord = tuple(float(row[var]) for var in settings.variables)
                    if coord in normalized_index:
                        allowed_indices.add(normalized_index[coord])
                floor_fixed_cells = {
                    idx: 0 for idx in range(len(floor_grid.cell_data))
                    if idx not in allowed_indices
                }
                logger.info(
                    f"[{segment}] Ceiling constraint (top-down): {len(allowed_indices)} cells allowed, "
                    f"{len(floor_fixed_cells)} cells forced rejected (from {floor_cells_path})"
                )
            else:
                # Bottom-up: cells in CSV must be accepted (floor)
                floor_fixed_cells = {}
                for _, row in floor_df.iterrows():
                    coord = tuple(float(row[var]) for var in settings.variables)
                    if coord in normalized_index:
                        floor_fixed_cells[normalized_index[coord]] = 1
                n_matched = len(floor_fixed_cells)
                if n_matched < n_floor_rows:
                    logger.warning(
                        f"[{segment}] Floor constraint: {n_matched}/{n_floor_rows} cells matched "
                        f"(unmatched cells may be absent from this segment's grid)"
                    )
                logger.info(
                    f"[{segment}] Floor constraint (bottom-up): {n_matched} cells must be accepted "
                    f"(from {floor_cells_path})"
                )

        # Per-segment minimum accepted-bin thresholds:
        # force reject any cell with variable value below configured threshold.
        if settings.min_accepted_bin_by_variable:
            minbin_grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)
            var_pos = {v: i for i, v in enumerate(settings.variables)}
            income_pos = var_pos.get("income_bin")
            minbin_reject_cells: dict[int, int] = {}

            # Normalize conditional thresholds by variable, keyed by income_bin value.
            conditional_thresholds: dict[str, dict[float, float]] = {}
            for var, threshold_cfg in settings.min_accepted_bin_by_variable.items():
                if isinstance(threshold_cfg, dict):
                    if income_pos is None:
                        raise ValueError(
                            f"[{segment}] Conditional min_accepted_bin_by_variable for '{var}' requires "
                            "'income_bin' in settings.variables."
                        )
                    conditional_thresholds[var] = {float(k): float(v) for k, v in threshold_cfg.items()}

            for coord, idx in minbin_grid.cell_index.items():
                coord_f = tuple(float(v) for v in coord)
                income_val = coord_f[income_pos] if income_pos is not None else None
                reject = False
                for var, threshold_cfg in settings.min_accepted_bin_by_variable.items():
                    if isinstance(threshold_cfg, dict):
                        resolved = conditional_thresholds[var].get(float(income_val)) if income_val is not None else None
                        if resolved is None:
                            continue
                        threshold = resolved
                    else:
                        threshold = float(threshold_cfg)
                    if coord_f[var_pos[var]] < threshold:
                        reject = True
                        break
                if reject:
                    minbin_reject_cells[idx] = 0

            if floor_fixed_cells is None:
                floor_fixed_cells = {}
            conflicts = [idx for idx, val in minbin_reject_cells.items() if floor_fixed_cells.get(idx) == 1 and val == 0]
            if conflicts:
                raise ValueError(
                    f"[{segment}] min_accepted_bin_by_variable conflicts with must-accept floor constraints "
                    f"for {len(conflicts)} cell(s). Adjust segment settings."
                )
            floor_fixed_cells.update(minbin_reject_cells)
            logger.info(
                f"[{segment}] Min-bin constraint: {len(minbin_reject_cells)} cells forced rejected via "
                f"min_accepted_bin_by_variable={settings.min_accepted_bin_by_variable}"
            )

        # MILP-based Pareto frontier optimization
        pareto_df, grid, pareto_masks = trace_pareto_frontier(
            data_summary_desagregado=data_summary_desagregado,
            variables=settings.variables,
            inv_vars=settings.inv_vars,
            multiplier=settings.multiplier,
            indicators=settings.indicators,
            n_points=settings.pareto_n_points,
            max_swapin_production_pct=settings.max_swapin_production_pct,
            max_swapin_risk=settings.max_swapin_risk,
            multiplier_h3=settings.multiplier_h3,
            milp_time_limit=settings.milp_time_limit,
            monotonicity_relaxation_enabled=settings.monotonicity_relaxation_enabled,
            monotonicity_uncertainty_min_exposure=settings.monotonicity_uncertainty_min_exposure,
            monotonicity_uncertainty_z_threshold=settings.monotonicity_uncertainty_z_threshold,
            fixed_cells=floor_fixed_cells,
        )

        if pareto_df.empty:
            # Fallback depending on number of variables
            if len(settings.variables) != 2:
                # N!=2: try GA fallback instead of legacy enumeration
                logger.warning(f"[{segment}] MILP produced no solutions for N!=2, trying GA fallback")
                from src.optimization_utils import _ga_pareto_fallback

                pareto_df, grid, pareto_masks = _ga_pareto_fallback(
                    grid,
                    settings.inv_vars,
                    settings.multiplier,
                    settings.indicators,
                    settings.pareto_n_points,
                    monotonicity_relaxation_enabled=settings.monotonicity_relaxation_enabled,
                    monotonicity_uncertainty_min_exposure=settings.monotonicity_uncertainty_min_exposure,
                    monotonicity_uncertainty_z_threshold=settings.monotonicity_uncertainty_z_threshold,
                )
                if pareto_df.empty:
                    raise RuntimeError(
                        f"[{segment}] Both MILP and GA produced no solutions for N>2 "
                        f"({len(settings.variables)} variables)."
                    )
                data_summary = add_bin_columns(pareto_df, pareto_masks, grid, settings.inv_vars)
                data_summary_sample_no_opt = pd.DataFrame(columns=["oa_amt_h0", "b2_ever_h6"])
            else:
                logger.warning(f"[{segment}] MILP produced no solutions, falling back to legacy enumeration")
                try:
                    df_v = get_fact_sol(values_var0=values_var0, values_var1=values_var1, chunk_size=10000)
                    data_summary = kpi_of_fact_sol(
                        df_v=df_v,
                        values_var0=values_var0,
                        data_sumary_desagregado=data_summary_desagregado,
                        variables=settings.variables,
                        indicadores=settings.indicators,
                        chunk_size=100000,
                        multiplier=settings.multiplier,
                        multiplier_h3=settings.multiplier_h3,
                    )
                    data_summary_sample_no_opt = data_summary.sample(min(10000, len(data_summary)))
                    data_summary = get_optimal_solutions(df_v=df_v, data_sumary=data_summary, chunk_size=100000)
                    grid = None
                    pareto_masks = []
                except RuntimeError as e:
                    # Legacy enumeration can blow up in memory when the grid is large.
                    # If it fails, fall back to GA-based search (if pymoo is available)
                    # rather than crashing.
                    logger.warning(
                        f"[{segment}] Legacy enumeration failed ({e}). Trying GA fallback instead."
                    )
                    from src.optimization_utils import _ga_pareto_fallback

                    pareto_df, grid, pareto_masks = _ga_pareto_fallback(
                        grid,
                        settings.inv_vars,
                        settings.multiplier,
                        settings.indicators,
                        settings.pareto_n_points,
                        monotonicity_relaxation_enabled=settings.monotonicity_relaxation_enabled,
                        monotonicity_uncertainty_min_exposure=settings.monotonicity_uncertainty_min_exposure,
                        monotonicity_uncertainty_z_threshold=settings.monotonicity_uncertainty_z_threshold,
                    )
                    if pareto_df.empty:
                        raise RuntimeError(
                            f"[{segment}] Legacy enumeration failed and GA fallback produced no solutions."
                        ) from e
                    data_summary = add_bin_columns(pareto_df, pareto_masks, grid, settings.inv_vars)
                    data_summary_sample_no_opt = pd.DataFrame(columns=["oa_amt_h0", "b2_ever_h6"])
        else:
            # Add bin columns for 2-var backward compat (cutoff extraction, viz, bootstrap)
            data_summary = add_bin_columns(pareto_df, pareto_masks, grid, settings.inv_vars)

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
    # Compute complementary H3 risk metric when columns are available
    if "todu_30ever_h3" in data_summary_desagregado.columns:
        data_summary_desagregado["b2_ever_h3"] = calculate_b2_ever_h6(
            data_summary_desagregado["todu_30ever_h3"],
            data_summary_desagregado["todu_amt_pile_h3"],
            multiplier=settings.multiplier_h3,
            as_percentage=True,
        )
    data_summary_desagregado["text"] = data_summary_desagregado.apply(
        lambda x: str("{:,.2f}M".format(x["oa_amt_h0"] / 1000000)) + " " + str("{:.2%}".format(x["b2_ever_h6"] / 100)),
        axis=1,
    )

    elapsed = time.perf_counter() - t0
    mode = "baseline" if settings.baseline_mode else ("fixed_cutoffs" if use_fixed_cutoffs else "milp_pareto")
    b2_min = data_summary["b2_ever_h6"].min() if not data_summary.empty else 0
    b2_max = data_summary["b2_ever_h6"].max() if not data_summary.empty else 0
    grid_desc = "x".join(str(len(values_per_var[v])) for v in settings.variables)
    logger.info(
        f"[{segment}] Optimization done | mode={mode} | "
        f"{len(data_summary)} solutions | {grid_desc} grid | "
        f"b2 range: [{b2_min:.2f}%, {b2_max:.2f}%] | optimum_risk={settings.optimum_risk:.1f}% | {elapsed:.1f}s"
    )

    return data_summary_desagregado, data_summary, data_summary_sample_no_opt, values_per_var, grid, pareto_masks


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
        pareto_masks=pareto_masks,
        grid=grid,
        multiplier=settings.multiplier,
        total_demand=total_demand,
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

    # Build cut_map for 2-var path (also used even in N-d as fallback for cut_map consumers)
    cut_map: dict[float, float] = {}
    if not is_nd:
        row = opt_sol.iloc[0]
        for bin_val in values_var0:
            if bin_val in row:
                cut_map[float(bin_val)] = float(row[bin_val])
            elif str(bin_val) in row:
                cut_map[float(bin_val)] = float(row[str(bin_val)])
            elif str(float(bin_val)) in row:
                cut_map[float(bin_val)] = float(row[str(float(bin_val))])

    # Calculate repesca_production from data_summary_desagregado
    repesca_production = 0.0
    if "oa_amt_h0_rep" in data_summary_desagregado.columns:
        if is_nd and selected_mask is not None and grid is not None:
            passes = classify_by_mask(data_summary_desagregado, selected_mask, grid)
            repesca_production = data_summary_desagregado.loc[passes, "oa_amt_h0_rep"].sum()
        elif not is_nd and len(settings.variables) > 1:
            var0 = settings.variables[0]
            var1 = settings.variables[1]
            fallback = float("inf") if inv_var1 else float("-inf")
            full_cut_series = data_summary_desagregado[var0].map(cut_map).fillna(fallback)
            if inv_var1:
                passes_2d = data_summary_desagregado[var1] >= full_cut_series
            else:
                passes_2d = data_summary_desagregado[var1] <= full_cut_series
            repesca_production = data_summary_desagregado.loc[passes_2d, "oa_amt_h0_rep"].sum()

    # Pass model CV SE so bootstrap CI accounts for model prediction uncertainty
    model_cv_se = None
    if risk_inference:
        best_info = risk_inference.get("best_model_info", {})
        model_cv_se = best_info.get("cv_std_rmse") or best_info.get("cv_std_r2")
    ci_data = calculate_bootstrap_intervals(
        data_booked=data_booked,
        cut_map=cut_map,
        variables=settings.variables,
        multiplier=settings.multiplier,
        n_bootstraps=settings.n_bootstraps,
        inv_var1=inv_var1,
        model_cv_se_risk=model_cv_se,
        annual_coef=annual_coef_main,
        repesca_production=repesca_production,
        mask=selected_mask,
        grid=grid,
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

    summary_table = reconcile_risk_production_summary_with_audit(summary_table, audit_main)
    validate_audit_against_summary(audit_main, summary_table)

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
            summary_table.loc[mask_opt, "risk_ci_lower"] = ci_data.get("risk_ci_lower", 0.0)
            summary_table.loc[mask_opt, "risk_ci_upper"] = ci_data.get("risk_ci_upper", 0.0)

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


def _build_scenario_list(settings: PreprocessingSettings, use_fixed_cutoffs: bool) -> list[tuple[float, str]]:
    """Build the list of (risk_threshold, name) scenarios to run."""
    base_optimum_risk = settings.optimum_risk
    scenario_step = settings.risk_step
    segment = settings.segment_filter

    if settings.baseline_mode:
        scenarios = [(base_optimum_risk, "base")]
        logger.debug(f"[{segment}] Baseline mode: running base scenario only")
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
        baseline_mask = milp_solve_cutoffs(
            grid,
            settings.optimum_risk,
            settings.inv_vars,
            settings.multiplier,
            max_swapin_production_pct=settings.max_swapin_production_pct,
            max_swapin_risk=settings.max_swapin_risk,
            time_limit=settings.milp_time_limit,
            monotonicity_relaxation_enabled=settings.monotonicity_relaxation_enabled,
            monotonicity_uncertainty_min_exposure=settings.monotonicity_uncertainty_min_exposure,
            monotonicity_uncertainty_z_threshold=settings.monotonicity_uncertainty_z_threshold,
        )
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
            max_swapin_production_pct=settings.max_swapin_production_pct,
            max_swapin_risk=settings.max_swapin_risk,
            milp_time_limit=settings.milp_time_limit,
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
            max_swapin_production_pct=settings.max_swapin_production_pct,
            max_swapin_risk=settings.max_swapin_risk,
            milp_time_limit=settings.milp_time_limit,
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


def run_ri_optimizer_phase(
    data_booked: pd.DataFrame,
    data_demand: pd.DataFrame,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stress_factor: float,
    tasa_fin: float,
    settings: PreprocessingSettings,
    annual_coef: float,
    output: OutputPaths | None = None,
    data_booked_mr: pd.DataFrame | None = None,
    data_demand_mr: pd.DataFrame | None = None,
    annual_coef_mr: float | None = None,
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
) -> dict | None:
    """Run reject inference parameter optimization (non-blocking).

    Gated behind ``settings.run_ri_optimizer``. Sweeps over
    (reject_uplift_factor, reject_max_risk_multiplier) grid to find the pair
    that maximizes production at the configured risk target.

    Parameters
    ----------
    data_booked_mr:
        Optional MR-period booked data for out-of-time validation.
    data_demand_mr:
        Optional MR-period demand data for out-of-time validation.
    annual_coef_mr:
        Optional MR-period annual coefficient.

    Returns:
        Best parameter dict, or None if disabled/failed.
    """
    if not settings.run_ri_optimizer:
        return None

    if settings.reject_inference_method == "none":
        logger.warning("RI optimizer skipped: reject_inference_method is 'none'")
        return None

    if output is None:
        output = OutputPaths()

    segment = settings.segment_filter
    logger.info(f"[{segment}] Running reject inference parameter optimization...")

    try:
        from src.inference_optimized import compute_pre_reject_inference_data
        from src.reject_inference import compute_acceptance_rates
        from src.reject_inference_optimizer import (
            OptimizerInputs,
            run_reject_inference_optimization,
            run_reject_inference_optimization_optuna,
            validate_ri_with_mr,
        )

        # Step 1: Temporal train/validation split of main-period data
        # Both splits have fully mature H6 outcomes (unlike MR period).
        split_ratio = settings.ri_validation_split
        train_booked, train_demand = data_booked, data_demand
        val_booked, val_demand = None, None

        if split_ratio < 1.0 and "mis_date" in data_booked.columns and "mis_date" in data_demand.columns:
            months_booked = sorted(data_booked["mis_date"].dt.to_period("M").unique())
            n_train = max(1, int(len(months_booked) * split_ratio))
            train_months = set(months_booked[:n_train])
            val_months = set(months_booked[n_train:])

            if len(val_months) >= 1:
                train_mask_b = data_booked["mis_date"].dt.to_period("M").isin(train_months)
                train_booked = data_booked[train_mask_b].copy()
                val_booked = data_booked[~train_mask_b].copy()

                train_mask_d = data_demand["mis_date"].dt.to_period("M").isin(train_months)
                train_demand = data_demand[train_mask_d].copy()
                val_demand = data_demand[~train_mask_d].copy()

                train_range = f"{min(train_months)}–{max(train_months)}"
                val_range = f"{min(val_months)}–{max(val_months)}"
                logger.info(
                    f"[{segment}] RI temporal split: train={len(train_months)} months ({train_range}, "
                    f"{len(train_booked):,} booked), val={len(val_months)} months ({val_range}, "
                    f"{len(val_booked):,} booked)"
                )
            else:
                logger.warning(
                    f"[{segment}] RI temporal split: only {len(months_booked)} months, "
                    f"cannot split at {split_ratio:.0%}. Using all data for training."
                )
        elif split_ratio < 1.0:
            logger.warning(f"[{segment}] RI temporal split: mis_date not available. Using all data for training.")

        # Step 2: Compute invariant pre-reject-inference data (on training split)
        booked_summary, repesca_pre_ri = compute_pre_reject_inference_data(
            data_booked=train_booked,
            data_demand=train_demand,
            risk_inference=risk_inference,
            reg_todu_amt_pile=reg_todu_amt_pile,
            stressor=stress_factor,
            indicators=settings.indicators,
            variables=settings.variables,
            annual_coef=annual_coef,
            multiplier=settings.multiplier,
            per_bin_stress=per_bin_stress,
        )

        # Step 3: Compute acceptance rates (on training split)
        # Propagate Bayesian smoothing settings so optimizer uses the same rates as main pipeline
        acceptance_rates = compute_acceptance_rates(
            train_demand, settings.variables,
            bayesian_smoothing=settings.reject_bayesian_smoothing,
            bayesian_prior_strength=settings.reject_bayesian_prior_strength,
            include_all_rejections=settings.reject_include_all_rejections,
            recent_months=settings.reject_acceptance_recent_months,
            decay_half_life_months=settings.reject_acceptance_decay_half_life_months,
            date_col=settings.reject_acceptance_date_col,
        )

        # Step 4: Build optimizer inputs
        optimizer_inputs = OptimizerInputs(
            booked_summary=booked_summary,
            repesca_pre_ri=repesca_pre_ri,
            acceptance_rates=acceptance_rates,
            tasa_fin=tasa_fin,
            variables=settings.variables,
            indicators=settings.indicators,
            inv_vars=settings.inv_vars,
            multiplier=settings.multiplier,
            parceling_method=settings.reject_parceling_method,
            calibration_gamma=settings.ri_calibration_gamma,
            per_bin_tasa_fin=per_bin_tasa_fin,
            enforce_monotonicity=settings.reject_enforce_monotonicity,
            apply_h3_multiplier=settings.reject_apply_h3_multiplier,
        )

        # Step 5: Run optimization (grid or Optuna)
        if settings.ri_optimizer_method == "optuna":
            results_df, best_params = run_reject_inference_optimization_optuna(
                optimizer_inputs,
                risk_target=settings.optimum_risk,
                uplift_range=tuple(settings.ri_uplift_range),
                max_mult_range=tuple(settings.ri_max_mult_range),
                n_trials=settings.ri_optuna_n_trials,
            )
        else:
            results_df, best_params = run_reject_inference_optimization(
                optimizer_inputs,
                risk_target=settings.optimum_risk,
                uplift_range=tuple(settings.ri_uplift_range),
                uplift_steps=settings.ri_uplift_steps,
                max_mult_range=tuple(settings.ri_max_mult_range),
                max_mult_steps=settings.ri_max_mult_steps,
            )

        # Step 6: Out-of-time validation on held-out main-period months
        if best_params and val_booked is not None and val_demand is not None:
            try:
                val_booked_summary, val_repesca_pre_ri = compute_pre_reject_inference_data(
                    data_booked=val_booked,
                    data_demand=val_demand,
                    risk_inference=risk_inference,
                    reg_todu_amt_pile=reg_todu_amt_pile,
                    stressor=stress_factor,
                    indicators=settings.indicators,
                    variables=settings.variables,
                    annual_coef=annual_coef,
                    multiplier=settings.multiplier,
                    per_bin_stress=per_bin_stress,
                )
                val_acceptance_rates = compute_acceptance_rates(
                    val_demand, settings.variables,
                    bayesian_smoothing=settings.reject_bayesian_smoothing,
                    bayesian_prior_strength=settings.reject_bayesian_prior_strength,
                    include_all_rejections=settings.reject_include_all_rejections,
                    recent_months=settings.reject_acceptance_recent_months,
                    decay_half_life_months=settings.reject_acceptance_decay_half_life_months,
                    date_col=settings.reject_acceptance_date_col,
                )
                val_optimizer_inputs = OptimizerInputs(
                    booked_summary=val_booked_summary,
                    repesca_pre_ri=val_repesca_pre_ri,
                    acceptance_rates=val_acceptance_rates,
                    tasa_fin=tasa_fin,
                    variables=settings.variables,
                    indicators=settings.indicators,
                    inv_vars=settings.inv_vars,
                    multiplier=settings.multiplier,
                    parceling_method=settings.reject_parceling_method,
                    calibration_gamma=settings.ri_calibration_gamma,
                    per_bin_tasa_fin=per_bin_tasa_fin,
                    enforce_monotonicity=settings.reject_enforce_monotonicity,
                    apply_h3_multiplier=settings.reject_apply_h3_multiplier,
                )

                validation = validate_ri_with_mr(
                    optimizer_inputs, val_optimizer_inputs, best_params, settings.optimum_risk
                )
                best_params.update({f"val_{k}": v for k, v in validation.items()})
                logger.info(
                    f"[{segment}] RI temporal validation: "
                    f"train_error={validation['main_calibration_error']:.6f}, "
                    f"val_error={validation['mr_calibration_error']:.6f}, "
                    f"degradation={validation['degradation_ratio']:.2f}x"
                )

                # Append validation results to results CSV
                for key, val in validation.items():
                    col = f"val_{key}"
                    results_df[col] = None
                    results_df.loc[results_df["is_best"], col] = val

            except Exception as e:
                logger.warning(f"[{segment}] RI temporal validation failed (non-blocking): {e}")

        # Step 6: Save results
        csv_path = output.ri_optimizer_csv()
        results_df.to_csv(csv_path, index=False)
        logger.info(f"[{segment}] RI optimizer results saved to {csv_path}")

        if best_params:
            logger.info(
                f"[{segment}] RI optimizer best params: "
                f"uplift={best_params['uplift_factor']:.2f}, "
                f"max_mult={best_params['max_risk_multiplier']:.2f}"
            )

        return best_params

    except Exception as e:
        logger.error(f"[{segment}] RI optimizer failed (non-blocking): {e}")
        return None


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
