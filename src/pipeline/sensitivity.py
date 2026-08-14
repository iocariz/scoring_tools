"""Sensitivity analysis phase: perturbs per-bin risk and re-runs the MILP to measure cutoff stability.

Extracted from ``src/pipeline/optimization.py`` in R2b-iv (todo #63). Exposes :func:`run_sensitivity_phase`.
"""

import os

import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings


def run_sensitivity_phase(
    data_summary_desagregado: pd.DataFrame,
    data_summary: pd.DataFrame,
    settings: PreprocessingSettings,
    output: OutputPaths | None = None,
    fixed_cells: dict[int, int] | None = None,
) -> None:
    """Run sensitivity analysis on the base scenario (non-blocking).

    Gated behind ``settings.run_sensitivity``. Saves results to CSV.

    Args:
        data_summary_desagregado: Aggregated summary data.
        data_summary: Pareto-optimal solutions DataFrame.
        settings: Configuration settings.
        output: Output paths configuration.
        fixed_cells: Floor/ceiling constraints from sequential cutoff ordering.
    """
    if not settings.run_sensitivity:
        return

    if output is None:
        output = OutputPaths()

    segment = settings.segment_filter
    logger.info(f"[{segment}] Running sensitivity analysis...")

    try:
        from src.optimization_utils import CellGrid, decode_mask, milp_solve_cutoffs
        from src.sensitivity import compute_cell_marginal_impact, run_sensitivity_analysis, sensitivity_cell_detail

        grid = CellGrid.from_summary(data_summary_desagregado, settings.variables)

        # #38/#55: use the base scenario's ACTUALLY-SELECTED mask as the baseline
        # (persisted acceptance_mask in optimal_solution_base.csv), not a fresh
        # MILP re-solve at the raw, unrounded optimum_risk. Re-solving diverged
        # from the shipped policy whenever selection_risk_basis="ci_upper" chose a
        # different frontier point, or for fixed_cutoffs segments whose policy
        # never came from the MILP at all. Fall back to a re-solve only if the
        # frozen mask is unavailable.
        baseline_mask = None
        opt_path = output.optimal_solution_csv("_base")
        if os.path.exists(opt_path):
            try:
                opt_sol = pd.read_csv(opt_path)
                cell = opt_sol.iloc[0].get("acceptance_mask") if not opt_sol.empty else None
                if pd.notna(cell):
                    decoded = decode_mask(str(cell))
                    if len(decoded) == len(grid.cell_data):
                        baseline_mask = decoded
                    else:
                        logger.warning(
                            f"[{segment}] Sensitivity: frozen mask length {len(decoded)} != grid "
                            f"{len(grid.cell_data)}; re-solving the baseline."
                        )
            except (pd.errors.ParserError, OSError, ValueError) as e:
                logger.warning(f"[{segment}] Sensitivity: could not read frozen base mask ({e}); re-solving.")

        if baseline_mask is None:
            baseline_mask = milp_solve_cutoffs(
                grid,
                settings.optimum_risk,
                settings.inv_vars,
                settings.multiplier,
                fixed_cells=fixed_cells,
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
            fixed_cells=fixed_cells,
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
            fixed_cells=fixed_cells,
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
