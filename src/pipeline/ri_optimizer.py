"""Reject-inference optimizer phase: grid / Optuna search over ``reject_uplift_factor`` and ``reject_max_risk_multiplier`` to minimize calibration error against observed swap-in risk.

Extracted from ``src/pipeline/optimization.py`` in R2b-iv (todo #63). Exposes :func:`run_ri_optimizer_phase`.
"""

from typing import Any

import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings


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
            train_demand,
            settings.variables,
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
            no_demand_anchor_percentile=settings.reject_no_demand_anchor_percentile,
            confidence_scale=settings.reject_confidence_scale,
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
                    val_demand,
                    settings.variables,
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
                    no_demand_anchor_percentile=settings.reject_no_demand_anchor_percentile,
                    confidence_scale=settings.reject_confidence_scale,
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
