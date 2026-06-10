"""
Automated reject inference parameter optimizer.

Grid search over (reject_uplift_factor, reject_max_risk_multiplier) to find
the parameter pair that minimizes calibration error against the 1/acceptance_rate
selection model.

The calibration target for each cell is ``booked_risk / acceptance_rate^gamma``,
based on the standard selection-bias model: if a bin accepts the least-risky
fraction *a* of applicants, the true population risk is approximately
``observed / a^gamma``.  With ``gamma=1.0`` (default) this recovers the classic
model; lower gamma produces less aggressive targets.

The optimizer selects parameters whose blended (booked + RI-corrected repesca)
risk estimates best match these targets (exposure-weighted mean squared relative
error). The calibration blend is STRESS-FREE and NOT financing-rate weighted
(audit #29) so it shares the raw-demand basis of the target; the production
blend that feeds the MILP/KPIs keeps stress + tasa_fin. Ties within 5% of the
minimum error are broken CONSERVATIVELY (lowest production at the risk target).

Honesty note: the target is itself a modeling assumption (the 1/a^gamma
selection model) — this procedure calibrates one assumption against another,
never against realized rejected outcomes (none exist). It disciplines the
uplift choice; it does not validate it. The M4 out-of-time backtest remains
the only realized-outcome check of the resulting cutoffs.

Steps before reject inference (aggregate booked, aggregate repesca, apply risk
model) are invariant across parameter combinations and are computed once via
``compute_pre_reject_inference_data``.  The inner loop only re-runs:
apply_parceling_adjustment -> apply tasa_fin -> merge -> CellGrid -> MILP solve -> evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import DEFAULT_RANDOM_STATE, Suffixes
from src.optimization_utils import CellGrid, evaluate_solution, milp_solve_cutoffs
from src.reject_inference import _shrink_acceptance_rate, apply_parceling_adjustment
from src.utils import calculate_b2_ever_h6

ParcelingMethod = Literal["linear", "power", "sigmoid"]


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
    parceling_method: ParcelingMethod = "linear"
    calibration_gamma: float = 1.0
    per_bin_tasa_fin: pd.DataFrame | None = None
    enforce_monotonicity: bool = False
    apply_h3_multiplier: bool = False
    no_demand_anchor_percentile: float = 0.10
    confidence_scale: float = 10.0
    #: STRESS-FREE repesca frame for the calibration objective (audit #29).
    #: The 1/a^gamma target is built from raw booked outcomes, so candidates
    #: must be scored on the same basis — otherwise the chosen uplift silently
    #: cancels the stress factor. ``None`` falls back to ``repesca_pre_ri``
    #: (correct when stress is disabled; the financing-rate exclusion below
    #: still applies).
    repesca_pre_ri_calibration: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        """Runtime validation of Literal-typed fields (todo #52).

        Python does not enforce Literal at construction time; callers that
        bypass Pydantic (tests, ad-hoc scripts) can still pass arbitrary
        strings. Fail loudly here rather than silently mis-dispatching in
        apply_parceling_adjustment.
        """
        allowed = get_args(ParcelingMethod)
        if self.parceling_method not in allowed:
            raise ValueError(f"parceling_method must be one of {allowed}, got {self.parceling_method!r}")


def _compute_calibration_error(
    merged: pd.DataFrame,
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    multiplier: float,
    calibration_gamma: float = 1.0,
    *,
    no_demand_anchor_percentile: float = 0.10,
    confidence_scale: float = 10.0,
) -> float:
    """Compute exposure-weighted mean squared relative calibration error.

    For each cell the selection-bias model predicts:
        target_risk = booked_risk / clip(acceptance_rate, 0.05) ^ gamma

    The predicted (blended) risk after RI is:
        predicted_risk = multiplier * todu_30ever_h6 / todu_amt_pile_h6

    The acceptance_rate is score-only (08-other rejections are excluded), so
    the target population matches the blended (booked + score-rejected) basis.

    Returns the exposure-weighted mean of (relative_error ** 2).
    """
    rate_col = (
        "smoothed_acceptance_rate" if "smoothed_acceptance_rate" in acceptance_rates.columns else "acceptance_rate"
    )
    df = merged
    # Keep the calibration objective aligned with runtime RI adjustment (apply_parceling_adjustment):
    # no/low-demand bins are shrunk toward the conservative anchor via the SAME shared helper, so the
    # optimizer scores candidates against the same rate definition the runtime applies (not the old,
    # anti-conservative median fallback).
    rate_eff, _anchor, _n_no_demand = _shrink_acceptance_rate(
        df,
        acceptance_rates,
        variables,
        rate_col,
        anchor_percentile=no_demand_anchor_percentile,
        confidence_scale=confidence_scale,
    )
    acc = rate_eff.clip(lower=0.05)

    # Note: multiplier cancels in the relative error below; it is kept explicit
    # so both quantities read as actual annualized risks and stay correct if the
    # predicted/booked scaling factors ever diverge.
    denom_boo = df["todu_amt_pile_h6_boo"].replace(0, np.nan)
    booked_risk = multiplier * df["todu_30ever_h6_boo"] / denom_boo
    target_risk = booked_risk / (acc**calibration_gamma)

    denom_all = df["todu_amt_pile_h6"].replace(0, np.nan)
    predicted_risk = multiplier * df["todu_30ever_h6"] / denom_all

    relative_error = (predicted_risk - target_risk) / target_risk

    # Drop cells where target or predicted risk is undefined
    mask = relative_error.notna() & target_risk.notna() & (target_risk > 0)
    if not mask.any():
        logger.debug("RI calibration: no valid cells with target_risk > 0; returning inf")
        return np.inf

    weights = df.loc[mask, "todu_amt_pile_h6"]
    total_weight = weights.sum()
    if total_weight == 0:
        logger.debug("RI calibration: all valid cells have zero exposure weight; returning inf")
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
    _AUX_COLS = [
        "acceptance_rate",
        "smoothed_acceptance_rate",
        "reject_risk_multiplier",
        "ri_effective_acceptance_rate",
        "ri_confidence",
        "ri_bin_count",
    ]

    def _parcel(repesca_frame: pd.DataFrame) -> pd.DataFrame:
        out = apply_parceling_adjustment(
            repesca_summary=repesca_frame.copy(),
            acceptance_rates=inputs.acceptance_rates,
            variables=inputs.variables,
            reject_uplift_factor=uplift_factor,
            max_risk_multiplier=max_risk_multiplier,
            method=inputs.parceling_method,
            enforce_monotonicity=inputs.enforce_monotonicity,
            inv_vars=inputs.inv_vars,
            apply_h3_multiplier=inputs.apply_h3_multiplier,
            no_demand_anchor_percentile=inputs.no_demand_anchor_percentile,
            confidence_scale=inputs.confidence_scale,
            quiet=True,
        )
        return out.drop(columns=_AUX_COLS, errors="ignore")

    def _blend(repesca_frame: pd.DataFrame) -> pd.DataFrame:
        rep = repesca_frame.rename(columns={i: i + "_rep" for i in inputs.indicators})
        # Merge with booked summary (preserve integer dtypes on merge-key columns
        # that can be upcast to float by NaN introduction during outer merge).
        var_dtypes = {v: inputs.booked_summary[v].dtype for v in inputs.variables}
        out = inputs.booked_summary.merge(rep, on=inputs.variables, how="outer").fillna(0)
        for v, dt in var_dtypes.items():
            if out[v].dtype != dt:
                out[v] = out[v].astype(dt)
        # Compute base indicators (sum of _boo + _rep)
        for ind in inputs.indicators:
            out[ind] = out[ind + Suffixes.BOOKED] + out[ind + "_rep"]
        return out

    # --- Production blend: stressed + financing-rate scaled (feeds MILP/KPIs) ---
    repesca = _parcel(inputs.repesca_pre_ri)
    if inputs.per_bin_tasa_fin is not None and not inputs.per_bin_tasa_fin.empty:
        repesca = repesca.merge(inputs.per_bin_tasa_fin, on=inputs.variables, how="left")
        repesca["tasa_fin"] = repesca["tasa_fin"].fillna(inputs.tasa_fin)
        for ind in inputs.indicators:
            repesca[ind] *= repesca["tasa_fin"]
        repesca = repesca.drop(columns=["tasa_fin"])
    else:
        repesca[inputs.indicators] *= inputs.tasa_fin
    merged = _blend(repesca)

    # --- Calibration blend (audit #29): the 1/a^gamma target describes the raw
    # demand population (booked + score-rejected, one applicant = one unit), so
    # candidates are scored on a blend that is STRESS-FREE (via
    # repesca_pre_ri_calibration) and NOT financing-rate weighted. The old code
    # calibrated the stressed, tasa_fin-scaled production blend against the raw
    # target — the chosen uplift cancelled the stress factor and varied with the
    # financing rate, neither of which has anything to do with selection bias. ---
    rep_cal_src = (
        inputs.repesca_pre_ri_calibration if inputs.repesca_pre_ri_calibration is not None else inputs.repesca_pre_ri
    )
    merged_cal = _blend(_parcel(rep_cal_src))

    # Calibration error (parameter-intrinsic, computed before MILP)
    calibration_gamma = inputs.calibration_gamma if hasattr(inputs, "calibration_gamma") else 1.0
    calibration_error = _compute_calibration_error(
        merged_cal,
        inputs.acceptance_rates,
        inputs.variables,
        inputs.multiplier,
        calibration_gamma,
        no_demand_anchor_percentile=inputs.no_demand_anchor_percentile,
        confidence_scale=inputs.confidence_scale,
    )

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


def _select_best(results_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Select the best parameter combination from optimizer results.

    Best = min calibration_error among feasible; ties within the 5% tolerance
    band are broken CONSERVATIVELY (audit #29): the parameters being chosen
    are *assumptions* about unobserved rejected-population risk, so within a
    statistically indistinguishable band we take the candidate with the LOWEST
    production at the risk target — i.e. the most prudent assumption — instead
    of the old max-production rule, which by construction picked the loosest
    uplift the noise band allowed. Deterministic secondary keys: higher uplift,
    lower cap.

    Returns:
        ``(results_df, best_params)`` — *results_df* gains an ``is_best`` column;
        *best_params* is a dict with the winning parameters, or empty dict if
        no feasible solution exists or all calibration errors are non-finite.
    """
    feasible = results_df[results_df["feasible"]]
    if feasible.empty:
        logger.warning("RI optimizer: no feasible solution found")
        results_df["is_best"] = False
        return results_df, {}

    feasible = feasible[np.isfinite(feasible["calibration_error"])]
    if feasible.empty:
        logger.warning("RI optimizer: all feasible solutions have non-finite calibration_error")
        results_df["is_best"] = False
        return results_df, {}

    min_error = feasible["calibration_error"].min()
    tolerance = min_error * 1.05  # 5% tolerance band
    near_best = feasible[feasible["calibration_error"] <= tolerance]
    near_best = near_best.sort_values(
        ["oa_amt_h0", "uplift_factor", "max_risk_multiplier"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    best_idx = near_best.index[0]
    if len(near_best) > 1:
        prod_spread = float(near_best["oa_amt_h0"].max() - near_best["oa_amt_h0"].min())
        logger.info(
            f"RI optimizer tie-break: {len(near_best)} candidates within the 5% calibration band "
            f"(production spread €{prod_spread:,.0f}); choosing the most conservative "
            f"(the old rule took the band's production maximum)."
        )

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
    return _select_best(results_df)


def run_reject_inference_optimization_optuna(
    inputs: OptimizerInputs,
    risk_target: float,
    *,
    uplift_range: tuple[float, float] = (0.0, 5.0),
    max_mult_range: tuple[float, float] = (1.0, 5.0),
    n_trials: int = 100,
) -> tuple[pd.DataFrame, dict]:
    """Optuna TPE-based optimization over (uplift_factor, max_risk_multiplier).

    Uses Tree-structured Parzen Estimator (TPE) sampler for efficient search.
    Same best-selection logic as grid search (min calibration error, 5% tolerance,
    conservative tie-break — see _select_best).

    Returns:
        Tuple of (results_df with all trials, best_params dict).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    results: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        uplift = trial.suggest_float("uplift_factor", uplift_range[0], uplift_range[1])
        max_mult = trial.suggest_float("max_risk_multiplier", max_mult_range[0], max_mult_range[1])

        row = evaluate_ri_params(inputs, uplift, max_mult, risk_target)
        results.append(row)

        if not row["feasible"]:
            return float("inf")
        return row["calibration_error"]

    sampler = optuna.samplers.TPESampler(seed=DEFAULT_RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"RI Optuna optimizer: {n_trials} trials completed")

    results_df = pd.DataFrame(results)
    return _select_best(results_df)


def validate_ri_with_mr(
    inputs: OptimizerInputs,
    mr_inputs: OptimizerInputs,
    best_params: dict,
    risk_target: float,
) -> dict:
    """Validate best RI params on MR-period data (out-of-time validation).

    Applies the best RI parameters found on the main period to MR-period data
    and computes calibration error. Returns comparison metrics.

    Parameters
    ----------
    inputs:
        Main-period optimizer inputs (for computing main calibration error).
    mr_inputs:
        MR-period optimizer inputs.
    best_params:
        Best parameter dict from main-period optimization.
    risk_target:
        Target risk threshold for MILP.

    Returns
    -------
    Dict with keys: main_calibration_error, mr_calibration_error, degradation_ratio.
    """
    uplift = best_params["uplift_factor"]
    max_mult = best_params["max_risk_multiplier"]

    main_result = evaluate_ri_params(inputs, uplift, max_mult, risk_target)
    mr_result = evaluate_ri_params(mr_inputs, uplift, max_mult, risk_target)

    main_error = main_result["calibration_error"]
    mr_error = mr_result["calibration_error"]

    degradation = mr_error / main_error if main_error > 0 and np.isfinite(main_error) else float("nan")

    logger.info(
        f"RI MR validation: main_error={main_error:.6f}, mr_error={mr_error:.6f}, degradation={degradation:.2f}x"
    )

    return {
        "main_calibration_error": main_error,
        "mr_calibration_error": mr_error,
        "degradation_ratio": degradation,
        "mr_feasible": mr_result.get("feasible", False),
        "mr_oa_amt_h0": mr_result.get("oa_amt_h0", 0.0),
        "mr_b2_ever_h6": mr_result.get("b2_ever_h6", 0.0),
    }
