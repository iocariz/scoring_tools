"""
Automated reject inference parameter optimizer.

Grid search over (reject_uplift_factor, reject_max_risk_multiplier) to find
the parameter pair that minimizes calibration error against the 1/acceptance_rate
selection model.

The calibration target for each cell is ``booked_risk / acceptance_rate``, based
on the standard selection-bias model: if a bin accepts the least-risky fraction
*a* of applicants, the true population risk is approximately ``observed / a``.
The optimizer selects parameters whose blended (booked + RI-corrected repesca)
risk estimates best match these targets (exposure-weighted mean squared relative
error).  Ties within 5% of the minimum error are broken by maximizing production.

Steps before reject inference (aggregate booked, aggregate repesca, apply risk
model) are invariant across parameter combinations and are computed once via
``compute_pre_reject_inference_data``.  The inner loop only re-runs:
apply_parceling_adjustment -> apply tasa_fin -> merge -> CellGrid -> MILP solve -> evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import Suffixes
from src.optimization_utils import CellGrid, evaluate_solution, milp_solve_cutoffs
from src.reject_inference import apply_parceling_adjustment
from src.utils import calculate_b2_ever_h6


@dataclass
class OptimizerInputs:
    """Pre-computed invariant data for the optimizer."""

    booked_summary: pd.DataFrame  # aggregated with _boo suffix
    repesca_pre_ri: pd.DataFrame  # after risk model, before reject inference
    acceptance_rates: pd.DataFrame  # per-bin acceptance rates
    tasa_fin: float
    variables: list[str]
    indicators: list[str]
    inv_vars: list[str] = field(default_factory=list)
    multiplier: float = 7.0
    parceling_method: str = "linear"


def _compute_calibration_error(
    merged: pd.DataFrame,
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    multiplier: float,
) -> float:
    """Compute exposure-weighted mean squared relative calibration error.

    For each cell the selection-bias model predicts:
        target_risk = booked_risk / clip(acceptance_rate, 0.05)

    The predicted (blended) risk after RI is:
        predicted_risk = multiplier * todu_30ever_h6 / todu_amt_pile_h6

    Returns the exposure-weighted mean of (relative_error ** 2).
    """
    df = merged.merge(acceptance_rates[variables + ["acceptance_rate"]], on=variables, how="left")
    acc = df["acceptance_rate"].clip(lower=0.05)

    denom_boo = df["todu_amt_pile_h6_boo"].replace(0, np.nan)
    booked_risk = multiplier * df["todu_30ever_h6_boo"] / denom_boo
    target_risk = booked_risk / acc

    denom_all = df["todu_amt_pile_h6"].replace(0, np.nan)
    predicted_risk = multiplier * df["todu_30ever_h6"] / denom_all

    relative_error = (predicted_risk - target_risk) / target_risk

    # Drop cells where target or predicted risk is undefined
    mask = relative_error.notna() & target_risk.notna() & (target_risk > 0)
    if not mask.any():
        return np.inf

    weights = df.loc[mask, "todu_amt_pile_h6"]
    total_weight = weights.sum()
    if total_weight == 0:
        return np.inf

    return float((weights * relative_error[mask] ** 2).sum() / total_weight)


def evaluate_ri_params(
    inputs: OptimizerInputs,
    uplift_factor: float,
    max_risk_multiplier: float,
    risk_target: float,
) -> dict:
    """Evaluate a single (uplift_factor, max_risk_multiplier) combination.

    Steps: apply_parceling_adjustment -> drop aux cols -> apply tasa_fin ->
    rename _rep -> merge with booked -> compute base indicators -> CellGrid ->
    milp_solve_cutoffs -> evaluate_solution.

    Returns:
        Dict with keys: uplift_factor, max_risk_multiplier, oa_amt_h0,
        b2_ever_h6, feasible, n_cells_accepted, calibration_error.
    """
    # Apply parceling with the candidate parameters
    repesca = apply_parceling_adjustment(
        repesca_summary=inputs.repesca_pre_ri.copy(),
        acceptance_rates=inputs.acceptance_rates,
        variables=inputs.variables,
        reject_uplift_factor=uplift_factor,
        max_risk_multiplier=max_risk_multiplier,
        method=inputs.parceling_method if hasattr(inputs, "parceling_method") else "linear",
    )

    # Drop auxiliary columns
    repesca = repesca.drop(columns=["acceptance_rate", "reject_risk_multiplier"], errors="ignore")

    # Apply financing rate
    repesca[inputs.indicators] *= inputs.tasa_fin

    # Rename with _rep suffix
    repesca = repesca.rename(columns={i: i + "_rep" for i in inputs.indicators})

    # Merge with booked summary
    merged = inputs.booked_summary.merge(repesca, on=inputs.variables, how="outer").fillna(0)

    # Compute base indicators (sum of _boo + _rep)
    for ind in inputs.indicators:
        merged[ind] = merged[ind + Suffixes.BOOKED] + merged[ind + "_rep"]

    # Calibration error (parameter-intrinsic, computed before MILP)
    calibration_error = _compute_calibration_error(merged, inputs.acceptance_rates, inputs.variables, inputs.multiplier)

    # Build grid and solve MILP
    grid = CellGrid.from_summary(merged, inputs.variables)
    mask = milp_solve_cutoffs(grid, risk_target, inputs.inv_vars, inputs.multiplier)

    result = {
        "uplift_factor": uplift_factor,
        "max_risk_multiplier": max_risk_multiplier,
    }

    if mask is None or mask.sum() == 0:
        result.update(
            {
                "oa_amt_h0": 0.0,
                "b2_ever_h6": 0.0,
                "feasible": False,
                "n_cells_accepted": 0,
                "calibration_error": calibration_error,
            }
        )
        return result

    kpis = evaluate_solution(mask, grid, inputs.indicators, inputs.multiplier)
    b2 = float(
        calculate_b2_ever_h6(
            kpis.get("todu_30ever_h6", 0.0),
            kpis.get("todu_amt_pile_h6", 0.0),
            multiplier=inputs.multiplier,
            as_percentage=True,
        )
    )

    result.update(
        {
            "oa_amt_h0": kpis.get("oa_amt_h0", 0.0),
            "b2_ever_h6": b2,
            "feasible": True,
            "n_cells_accepted": int(mask.sum()),
            "calibration_error": calibration_error,
        }
    )
    return result


def run_reject_inference_optimization(
    inputs: OptimizerInputs,
    risk_target: float,
    *,
    uplift_range: tuple[float, float] = (0.0, 5.0),
    uplift_steps: int = 11,
    max_mult_range: tuple[float, float] = (1.0, 5.0),
    max_mult_steps: int = 9,
) -> tuple[pd.DataFrame, dict]:
    """Grid search over (uplift_factor, max_risk_multiplier).

    Returns:
        Tuple of (results_df with all combos + KPIs, best_params dict).
        Best = min calibration_error among feasible solutions; ties within
        5% of the minimum error are broken by max oa_amt_h0.
        Empty dict if nothing feasible.
    """
    uplift_values = np.linspace(uplift_range[0], uplift_range[1], uplift_steps)
    max_mult_values = np.linspace(max_mult_range[0], max_mult_range[1], max_mult_steps)

    total = len(uplift_values) * len(max_mult_values)
    logger.info(
        f"RI optimizer: {total} combinations "
        f"(uplift={uplift_range[0]:.1f}-{uplift_range[1]:.1f} x{uplift_steps}, "
        f"max_mult={max_mult_range[0]:.1f}-{max_mult_range[1]:.1f} x{max_mult_steps})"
    )

    results = []
    for uplift in uplift_values:
        for max_mult in max_mult_values:
            row = evaluate_ri_params(inputs, float(uplift), float(max_mult), risk_target)
            results.append(row)

    results_df = pd.DataFrame(results)

    # Find best: min calibration_error among feasible; tie-break by max production
    feasible = results_df[results_df["feasible"]]
    if feasible.empty:
        logger.warning("RI optimizer: no feasible solution found")
        results_df["is_best"] = False
        return results_df, {}

    min_error = feasible["calibration_error"].min()
    tolerance = min_error * 1.05  # 5% tolerance band
    near_best = feasible[feasible["calibration_error"] <= tolerance]
    best_idx = near_best["oa_amt_h0"].idxmax()

    results_df["is_best"] = False
    results_df.loc[best_idx, "is_best"] = True

    best_row = results_df.loc[best_idx]
    best_params = {
        "uplift_factor": float(best_row["uplift_factor"]),
        "max_risk_multiplier": float(best_row["max_risk_multiplier"]),
        "oa_amt_h0": float(best_row["oa_amt_h0"]),
        "b2_ever_h6": float(best_row["b2_ever_h6"]),
        "calibration_error": float(best_row["calibration_error"]),
    }

    logger.info(
        f"RI optimizer best: uplift={best_params['uplift_factor']:.2f}, "
        f"max_mult={best_params['max_risk_multiplier']:.2f} | "
        f"calibration_error={best_params['calibration_error']:.6f} | "
        f"production={best_params['oa_amt_h0']:,.0f} | "
        f"risk={best_params['b2_ever_h6']:.4f}%"
    )

    return results_df, best_params
