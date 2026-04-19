"""Cutoff-optimization phase: builds the Pareto frontier / fixed-cutoff solutions for a single segment.

Extracted from ``src/pipeline/optimization.py`` in R2b-iv (todo #63). Exposes :class:`OptimizationResult` and :func:`run_optimization_phase`.
"""

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.inference_optimized import run_optimization_pipeline
from src.optimization_utils import (
    CellGrid,
    add_bin_columns,
    create_fixed_cutoff_mask,
    create_fixed_cutoff_solution,
    evaluate_solution,
    get_fact_sol,
    get_optimal_solutions,
    kpi_of_fact_sol,
    trace_pareto_frontier,
)
from src.utils import (
    calculate_b2_ever_h6,
)


@dataclass
class OptimizationResult:
    """Result of the optimization phase (todo #63).

    Replaces the previous 7-tuple return from ``run_optimization_phase``.
    Attribute access matches the ``PreprocessingResult`` pattern used by
    :mod:`src.pipeline.preprocessing`. Supports positional unpacking via
    ``__iter__`` so legacy ``a, b, c, ... = run_optimization_phase(...)``
    call sites keep working during the migration window.
    """

    data_summary_desagregado: pd.DataFrame
    data_summary: pd.DataFrame
    data_summary_sample_no_opt: pd.DataFrame
    values_per_var: dict[str, list]
    grid: CellGrid | None
    pareto_masks: list = field(default_factory=list)
    floor_fixed_cells: dict[int, int] | None = None

    def __iter__(self):
        """Positional unpacking: preserves legacy 7-tuple call sites."""
        yield self.data_summary_desagregado
        yield self.data_summary
        yield self.data_summary_sample_no_opt
        yield self.values_per_var
        yield self.grid
        yield self.pareto_masks
        yield self.floor_fixed_cells


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
) -> OptimizationResult:
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
        :class:`OptimizationResult` with data_summary_desagregado,
        data_summary, data_summary_sample_no_opt, values_per_var,
        grid, pareto_masks, and floor_fixed_cells.
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter
    floor_fixed_cells: dict[int, int] | None = None

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

        # Zero out repesca columns in disaggregated data so downstream consumers
        # (visualizer, summary table, audit) see Optimum = Actual with zero swap-in.
        for col in data_summary_desagregado.columns:
            if col.endswith("_rep"):
                data_summary_desagregado[col] = 0
        # Recompute base totals (base = _boo + _rep, now _rep is 0 → base = _boo)
        for ind in settings.indicators:
            boo_col = f"{ind}_boo"
            if boo_col in data_summary_desagregado.columns:
                data_summary_desagregado[ind] = data_summary_desagregado[boo_col]

        grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)
        accept_all_mask = np.ones(len(grid.cell_data), dtype=int)

        # Use evaluate_solution for KPIs — same code path as all other modes
        baseline_kpis = evaluate_solution(
            accept_all_mask,
            grid,
            settings.indicators,
            settings.multiplier,
            multiplier_h3=settings.multiplier_h3,
        )
        baseline_kpis["sol_fac"] = 0

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
                logger.warning(f"[{segment}] Floor cells CSV: dropped {n_nan_rows} row(s) with NaN coordinates")
            # Build a float-normalized lookup to handle int/float type mismatches
            normalized_index = {tuple(float(v) for v in coord): idx for coord, idx in floor_grid.cell_index.items()}
            n_floor_rows = len(floor_df)

            if floor_cells_mode == "ceiling":
                # Top-down: cells in CSV are the ALLOWED set; everything else is rejected
                allowed_indices = set()
                for _, row in floor_df.iterrows():
                    coord = tuple(float(row[var]) for var in settings.variables)
                    if coord in normalized_index:
                        allowed_indices.add(normalized_index[coord])
                floor_fixed_cells = {idx: 0 for idx in range(len(floor_grid.cell_data)) if idx not in allowed_indices}
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
                        resolved = (
                            conditional_thresholds[var].get(float(income_val)) if income_val is not None else None
                        )
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
            conflicts = [
                idx for idx, val in minbin_reject_cells.items() if floor_fixed_cells.get(idx) == 1 and val == 0
            ]
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
                    logger.warning(f"[{segment}] Legacy enumeration failed ({e}). Trying GA fallback instead.")
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

    return OptimizationResult(
        data_summary_desagregado=data_summary_desagregado,
        data_summary=data_summary,
        data_summary_sample_no_opt=data_summary_sample_no_opt,
        values_per_var=values_per_var,
        grid=grid,
        pareto_masks=pareto_masks,
        floor_fixed_cells=floor_fixed_cells,
    )
