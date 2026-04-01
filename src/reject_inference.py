"""
Reject inference adjustments for score-rejected (repesca) applications.

When the risk model is trained only on booked applications, predictions for
score-rejected bins suffer from selection bias.  This module provides per-bin
acceptance-rate corrections that uplift the predicted risk for rejected
populations.

Supported methods:
- ``"none"``      – no adjustment (default, preserves current behavior)
- ``"parceling"`` – per-bin risk multiplier based on acceptance rate
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger

from src.constants import RejectReason, StatusName


def compute_acceptance_rates(
    data_demand: pd.DataFrame,
    variables: list[str],
    *,
    bayesian_smoothing: bool = False,
    bayesian_prior_strength: float = 10.0,
    include_all_rejections: bool = False,
    recent_months: int | None = None,
    decay_half_life_months: float | None = None,
    date_col: str = "mis_date",
) -> pd.DataFrame:
    """Compute per-bin acceptance rates from the demand population.

    For each unique combination of *variables* bins, the acceptance rate is
    defined as::

        acceptance_rate = n_booked / (n_booked + n_rejected)

    By default, only ``09-score`` rejections are counted (``08-other`` are
    excluded because they are not candidates for cutoff changes).  When
    *include_all_rejections* is True, **all** rejections are included in the
    denominator, producing lower (more conservative) acceptance rates and
    higher reject-inference multipliers.

    Parameters
    ----------
    data_demand:
        Full demand DataFrame containing both booked and rejected records.
        Must have columns ``status_name``, ``reject_reason``, and the columns
        listed in *variables*.
    variables:
        The two binning variable names (e.g. ``["sc_octroi_new_clus", "new_efx_clus"]``).
    bayesian_smoothing:
        If True, apply Beta-Binomial posterior smoothing to acceptance rates.
    bayesian_prior_strength:
        Strength of the Beta prior (higher = more shrinkage toward global rate).
    include_all_rejections:
        If True, include all rejections (not just score rejections) in the
        denominator.  Use when non-score rejections are a material share of
        demand (>5-10%).
    recent_months:
        If set (and *decay_half_life_months* is None), restrict the demand
        population to the most recent N months before computing acceptance
        rates.  Uses a hard date filter — counts remain integer.
    decay_half_life_months:
        If set, apply exponential time-decay weights to demand records.
        More recent records contribute more to bin-level acceptance rates.
        Takes precedence over *recent_months* when both are provided.
        Counts become float "effective counts".
    date_col:
        Name of the date column used for temporal filtering/weighting.

    Returns
    -------
    DataFrame with columns ``[*variables, "n_booked", "n_score_rejected", "acceptance_rate"]``.
    When *include_all_rejections* is True, ``"n_score_rejected"`` contains the
    count of **all** rejections.
    When *bayesian_smoothing* is True, also includes ``"smoothed_acceptance_rate"``.
    """
    if "status_name" not in data_demand.columns:
        raise ValueError("data_demand must contain 'status_name' column for reject inference.")

    # Time-awareness: either filter to a recent window or apply exponential
    # time-decay weights (based on `date_col`).
    # - recent_months (and no decay) uses a hard filter (counts-based).
    # - decay_half_life_months applies exponential weights (float "effective counts").
    if recent_months is not None or decay_half_life_months is not None:
        if date_col not in data_demand.columns:
            logger.warning(
                f"Reject inference acceptance-rate time-awareness requested but '{date_col}' column is missing; "
                "falling back to unweighted acceptance rates."
            )
            recent_months = None
            decay_half_life_months = None
        else:
            # Normalize dates for robust comparisons; drop rows with unparseable dates in weighting path.
            data_demand = data_demand.copy()
            data_demand[date_col] = pd.to_datetime(data_demand[date_col], errors="coerce")
            n_invalid_dates = int(data_demand[date_col].isna().sum())
            if n_invalid_dates > 0 and decay_half_life_months is not None:
                share = 100.0 * n_invalid_dates / max(len(data_demand), 1)
                logger.warning(
                    f"Reject inference decay mode: {n_invalid_dates:,} row(s) ({share:.2f}%) have invalid '{date_col}' "
                    "and will receive zero weight."
                )
            max_date = data_demand[date_col].max()
            if pd.isna(max_date):
                logger.warning(
                    f"Reject inference acceptance-rate time-awareness could not parse '{date_col}'; "
                    "falling back to unweighted acceptance rates."
                )
                recent_months = None
                decay_half_life_months = None
            else:
                # Hard recent window (counts remain integer-like)
                if recent_months is not None and decay_half_life_months is None:
                    cutoff = max_date - pd.DateOffset(months=int(recent_months))
                    data_demand = data_demand.loc[data_demand[date_col] >= cutoff]

    booked = data_demand[data_demand["status_name"] == StatusName.BOOKED.value]

    if include_all_rejections:
        rejected = data_demand[data_demand["status_name"] == StatusName.REJECTED.value]
        logger.debug("Acceptance rates use ALL rejections in denominator (include_all_rejections=True)")
    else:
        rejected = data_demand[
            (data_demand["status_name"] == StatusName.REJECTED.value)
            & (data_demand["reject_reason"] == RejectReason.SCORE.value)
        ]

    if decay_half_life_months is not None:
        # Exponential decay weights anchored at the maximum date in the (possibly filtered) dataset.
        # age_months ~ days/30.437. Effective counts are floats.
        data_tmp = data_demand[[*variables, "status_name", date_col]].copy()
        max_date = pd.to_datetime(data_tmp[date_col], errors="coerce").max()
        if pd.isna(max_date):
            # Should already be handled above, but keep it safe.
            logger.warning(
                f"Reject inference decay weights could not compute max date from '{date_col}'; "
                "falling back to unweighted acceptance rates."
            )
            decay_half_life_months = None
        else:
            age_days = (max_date - pd.to_datetime(data_tmp[date_col], errors="coerce")).dt.total_seconds() / (3600 * 24)
            age_months = age_days / 30.437
            weights = np.power(2.0, -age_months.to_numpy(dtype=float) / float(decay_half_life_months))
            data_tmp["__w"] = weights

            booked_w = data_demand.loc[data_demand["status_name"] == StatusName.BOOKED.value, variables].copy()
            booked_w["__w"] = data_tmp.loc[data_demand["status_name"] == StatusName.BOOKED.value, "__w"].to_numpy()
            n_booked = booked_w.groupby(variables)["__w"].sum().reset_index(name="n_booked")
            n_booked_raw = booked_w.groupby(variables).size().reset_index(name="n_booked_raw")

            if include_all_rejections:
                rejected_w = data_demand.loc[data_demand["status_name"] == StatusName.REJECTED.value, variables].copy()
                rejected_w["__w"] = data_tmp.loc[data_demand["status_name"] == StatusName.REJECTED.value, "__w"].to_numpy()
                n_rejected = rejected_w.groupby(variables)["__w"].sum().reset_index(name="n_score_rejected")
                n_rejected_raw = rejected_w.groupby(variables).size().reset_index(name="n_score_rejected_raw")
            else:
                rej_mask = (data_demand["status_name"] == StatusName.REJECTED.value) & (
                    data_demand["reject_reason"] == RejectReason.SCORE.value
                )
                rejected_w = data_demand.loc[rej_mask, variables].copy()
                rejected_w["__w"] = data_tmp.loc[rej_mask, "__w"].to_numpy()
                n_rejected = rejected_w.groupby(variables)["__w"].sum().reset_index(name="n_score_rejected")
                n_rejected_raw = rejected_w.groupby(variables).size().reset_index(name="n_score_rejected_raw")
    else:
        # Counts-based acceptance rates (original behavior)
        n_booked = booked.groupby(variables).size().reset_index(name="n_booked")
        n_rejected = rejected.groupby(variables).size().reset_index(name="n_score_rejected")
        n_booked_raw = n_booked.rename(columns={"n_booked": "n_booked_raw"})
        n_rejected_raw = n_rejected.rename(columns={"n_score_rejected": "n_score_rejected_raw"})

    # Save original dtypes of merge-key columns before outer merge (NaN introduction can
    # upcast int → float; we restore after fillna to keep downstream joins stable).
    var_dtypes = {v: n_booked[v].dtype for v in variables}
    rates = n_booked.merge(n_rejected, on=variables, how="outer").fillna(0)
    rates = rates.merge(n_booked_raw, on=variables, how="left").merge(n_rejected_raw, on=variables, how="left").fillna(0)
    for v, dt in var_dtypes.items():
        if rates[v].dtype != dt:
            rates[v] = rates[v].astype(dt)
    if decay_half_life_months is None:
        # Keep types stable for non-decay mode (tests rely on exact ints).
        rates["n_booked"] = rates["n_booked"].astype(int)
        rates["n_score_rejected"] = rates["n_score_rejected"].astype(int)
    rates["n_booked_raw"] = rates["n_booked_raw"].astype(int)
    rates["n_score_rejected_raw"] = rates["n_score_rejected_raw"].astype(int)

    total = rates["n_booked"] + rates["n_score_rejected"]
    rates["acceptance_rate"] = (rates["n_booked"] / total).where(total > 0, 0.0)

    # Bayesian smoothing: Beta-Binomial posterior
    if bayesian_smoothing:
        global_rate = rates["n_booked"].sum() / max(total.sum(), 1)
        # Empirical-Bayes adjustment:
        # The configured `bayesian_prior_strength` is treated as a baseline
        # shrinkage (equivalent sample size). We estimate an additional
        # empirical shrinkage level from cross-bin variability, and blend the
        # two to reduce staleness across time/segments.
        #
        # This is intentionally conservative and bounded to keep downstream
        # behavior stable.
        effective_prior_strength = float(bayesian_prior_strength)
        p = float(global_rate)
        if 0.0 < p < 1.0:
            bins_for_emp = rates.loc[total > 0, "acceptance_rate"]
            n_i = total.loc[total > 0].astype(float).to_numpy()
            if len(bins_for_emp) >= 2 and np.all(n_i > 0):
                sample_var = float(bins_for_emp.var(ddof=1))
                # Average within-bin (binomial) variance of the acceptance-rate estimator
                # Var(p_hat_i | p) ~ p(1-p)/n_i.
                within_var_mean = float((p * (1 - p) / n_i).mean())
                between_var = sample_var - within_var_mean
                if np.isfinite(between_var) and between_var <= 0:
                    logger.warning(
                        f"Bayesian smoothing: between-bin variance is non-positive "
                        f"({between_var:.6f}), data may not support random-effects model. "
                        f"Using configured prior_strength={bayesian_prior_strength:.1f}."
                    )
                if np.isfinite(between_var) and between_var > 0:
                    empirical_strength = (p * (1 - p) / between_var) - 1.0
                    if np.isfinite(empirical_strength) and empirical_strength > 0:
                        # Conservative blend keeps configured strength relevant while adapting.
                        effective_prior_strength = 0.5 * float(bayesian_prior_strength) + 0.5 * float(
                            empirical_strength
                        )

        effective_prior_strength = float(np.clip(effective_prior_strength, 0.5, 1000.0))
        alpha = max(effective_prior_strength * global_rate, 0.5)
        beta = max(effective_prior_strength * (1 - global_rate), 0.5)
        rates["smoothed_acceptance_rate"] = (rates["n_booked"] + alpha) / (total + alpha + beta)
        logger.debug(
            f"Bayesian smoothing applied | prior_strength(cfg)={bayesian_prior_strength:.3f} | "
            f"prior_strength(eff)={effective_prior_strength:.3f} | global_rate={global_rate:.3f} | "
            f"alpha={alpha:.2f}, beta={beta:.2f}"
        )

    # Warn about non-score rejections that are excluded from acceptance rate.
    # The denominator is n_booked + n_score_rejected (not all rejected).
    # Non-score rejections (manual review, fraud, missing docs) are excluded
    # because they are not candidates for cutoff changes.  However, excluding
    # them inflates acceptance rates when they are a material share of demand.
    n_other_rejected = len(
        data_demand[
            (data_demand["status_name"] == StatusName.REJECTED.value)
            & (data_demand["reject_reason"] != RejectReason.SCORE.value)
        ]
    )
    n_total_rejected = len(data_demand[data_demand["status_name"] == StatusName.REJECTED.value])
    n_total_demand = len(data_demand)
    if n_other_rejected > 0:
        other_pct = n_other_rejected / n_total_demand * 100
        other_of_rejected_pct = n_other_rejected / max(n_total_rejected, 1) * 100
        log_fn = logger.warning if other_pct > 5.0 else logger.info
        log_fn(
            f"Acceptance rates exclude {n_other_rejected:,} non-score rejections "
            f"({other_pct:.1f}% of demand, {other_of_rejected_pct:.1f}% of all rejections). "
            f"Bins with many non-score rejections may have overstated acceptance rates."
        )
        if other_pct > 10.0:
            logger.warning(
                "Non-score rejections exceed 10% of demand — acceptance rate bias may be material. "
                "Consider including all rejections in the denominator or adjusting reject_uplift_factor."
            )

    logger.debug(
        f"Acceptance rates computed for {len(rates)} bins | "
        f"mean={rates['acceptance_rate'].mean():.3f} | "
        f"min={rates['acceptance_rate'].min():.3f} | "
        f"max={rates['acceptance_rate'].max():.3f}"
    )

    return rates


def compute_ri_confidence(
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    *,
    scale: float = 50.0,
) -> pd.DataFrame:
    """Compute per-bin confidence scores for reject inference adjustments.

    Confidence is based on the total number of observations (booked + score-rejected)
    in each bin::

        confidence = 1 - exp(-n_total / scale)

    Parameters
    ----------
    acceptance_rates:
        Output of :func:`compute_acceptance_rates`.
    variables:
        Binning variable names.
    scale:
        Controls the rate of confidence growth. Higher = more observations needed
        for high confidence.

    Returns
    -------
    DataFrame with columns ``[*variables, "ri_confidence", "ri_bin_count"]``.
    Also includes:
    - ``ri_bin_count_effective``: float effective count (time-decay aware)
    - ``ri_bin_count_raw``: integer raw count
    """
    n_total_effective = (acceptance_rates["n_booked"] + acceptance_rates["n_score_rejected"]).astype(float)
    if {"n_booked_raw", "n_score_rejected_raw"}.issubset(acceptance_rates.columns):
        n_total_raw = (acceptance_rates["n_booked_raw"] + acceptance_rates["n_score_rejected_raw"]).astype(int)
    else:
        n_total_raw = np.round(n_total_effective).astype(int)

    confidence = 1.0 - np.exp(-n_total_effective / scale)

    return pd.DataFrame(
        {
            **{v: acceptance_rates[v] for v in variables},
            "ri_confidence": confidence,
            # Backward-compatible primary count column remains integer raw counts.
            "ri_bin_count": n_total_raw,
            # Additional explicit columns to avoid ambiguity in decay mode.
            "ri_bin_count_effective": n_total_effective,
            "ri_bin_count_raw": n_total_raw,
        }
    )


def _enforce_multiplier_monotonicity(
    result: pd.DataFrame,
    variables: list[str],
    inv_vars: list[str] | None = None,
    *,
    max_iterations: int = 10,
    tol: float = 1e-6,
    quiet: bool = False,
) -> pd.DataFrame:
    """Enforce monotonicity on reject_risk_multiplier per variable axis.

    Uses alternating isotonic regression passes until convergence to ensure
    that multipliers are monotonic along each variable axis (marginal
    monotonicity).  Iterating avoids the single-pass problem where the last
    variable's projection overwrites earlier ones.

    Parameters
    ----------
    result:
        DataFrame with ``reject_risk_multiplier`` and *variables* columns.
    variables:
        Binning variable names.
    inv_vars:
        Variables whose higher bin values indicate *lower* risk. Isotonic
        regression uses ``increasing=False`` for these.
    max_iterations:
        Maximum number of full passes over all variables.
    tol:
        Convergence tolerance on max absolute change between iterations.

    Returns
    -------
    DataFrame with monotonicity-adjusted ``reject_risk_multiplier``.
    """
    from sklearn.isotonic import IsotonicRegression

    if result.empty or "reject_risk_multiplier" not in result.columns:
        return result

    inv_set = set(inv_vars) if inv_vars else set()

    for iteration in range(max_iterations):
        prev_values = result["reject_risk_multiplier"].values.copy()

        for var in variables:
            increasing = var not in inv_set
            other_vars = [v for v in variables if v != var]

            if not other_vars:
                # Single variable: apply isotonic on the whole column
                sorted_df = result.sort_values(var)
                if len(sorted_df) < 2:
                    continue
                iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                iso_values = iso.fit_transform(
                    sorted_df[var].values.astype(float), sorted_df["reject_risk_multiplier"].values
                )
                result.loc[sorted_df.index, "reject_risk_multiplier"] = iso_values
            else:
                # Per-slice isotonic: for each unique combo of other vars,
                # apply isotonic along the target variable axis.
                # This preserves cross-dimensional variation.
                for _, slice_idx in result.groupby(other_vars, observed=True).groups.items():
                    slice_df = result.loc[slice_idx].sort_values(var)
                    if len(slice_df) < 2:
                        continue
                    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
                    iso_values = iso.fit_transform(
                        slice_df[var].values.astype(float), slice_df["reject_risk_multiplier"].values
                    )
                    result.loc[slice_df.index, "reject_risk_multiplier"] = iso_values

        max_change = np.abs(result["reject_risk_multiplier"].values - prev_values).max()
        if max_change < tol:
            logger.debug(f"Isotonic monotonicity converged after {iteration + 1} iteration(s) (max_change={max_change:.2e})")
            break
    else:
        logger.debug(f"Isotonic monotonicity did not converge after {max_iterations} iterations (max_change={max_change:.2e})")

    # Post-hoc: verify and fix partial-order violations across all dimensions.
    # A cell a dominates b if a[v] >= b[v] for all v (respecting direction).
    # For dominated pairs: multiplier[riskier] >= multiplier[safer].
    if len(variables) >= 2 and len(result) > 1:
        n_violations = _fix_partial_order_violations(result, variables, inv_set)
        if n_violations > 0:
            log_fn = logger.debug if quiet else logger.warning
            log_fn(
                f"Isotonic post-hoc: fixed {n_violations} partial-order violation(s) "
                f"remaining after alternating projections."
            )

    return result


def _fix_partial_order_violations(
    result: pd.DataFrame,
    variables: list[str],
    inv_set: set[str],
) -> int:
    """Check and fix partial-order violations by averaging violating pairs.

    Returns the number of violations fixed.
    """
    vals = result["reject_risk_multiplier"].values.copy()
    coords = result[variables].values  # (n_cells, n_vars)
    n = len(result)

    # Build direction signs: +1 if higher bin = riskier, -1 if inverted
    signs = np.array([(-1 if v in inv_set else 1) for v in variables])

    # Orient coordinates so that higher = riskier in all dimensions
    oriented = coords * signs

    violations_fixed = 0
    # Iterate until no violations remain (typically 1-2 passes)
    for _ in range(5):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j (i is riskier or equal in all dims)
                diff = oriented[i] - oriented[j]
                if np.all(diff >= 0) and np.any(diff > 0):
                    # i is strictly riskier → multiplier[i] should >= multiplier[j]
                    if vals[i] < vals[j]:
                        avg = (vals[i] + vals[j]) / 2
                        vals[i] = avg
                        vals[j] = avg
                        violations_fixed += 1
                        changed = True
                elif np.all(diff <= 0) and np.any(diff < 0):
                    # j is strictly riskier → multiplier[j] should >= multiplier[i]
                    if vals[j] < vals[i]:
                        avg = (vals[i] + vals[j]) / 2
                        vals[j] = avg
                        vals[i] = avg
                        violations_fixed += 1
                        changed = True
        if not changed:
            break

    if violations_fixed > 0:
        result["reject_risk_multiplier"] = vals
    return violations_fixed


def apply_parceling_adjustment(
    repesca_summary: pd.DataFrame,
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    *,
    reject_uplift_factor: float = 1.5,
    max_risk_multiplier: float = 3.0,
    method: Literal["linear", "power", "sigmoid"] = "linear",
    enforce_monotonicity: bool = False,
    inv_vars: list[str] | None = None,
    apply_h3_multiplier: bool = False,
    quiet: bool = False,
) -> pd.DataFrame:
    """Apply per-bin risk uplift to repesca summary based on acceptance rates.

    Three methods are available:

    - ``"linear"`` (default): ``multiplier = 1 + factor * (1 - acceptance_rate)``
    - ``"power"``: ``multiplier = (1 / acceptance_rate) ^ factor``.  This is
      grounded in the assumption that rejected applicants are drawn from the
      riskier tail, so risk scales as a power of the inverse acceptance rate.
      It produces a non-linear curve that grows faster at low acceptance rates.
    - ``"sigmoid"``: ``multiplier = 1 + factor / (1 + exp(steepness * (rate - midpoint)))``
      Produces a smooth S-curve: gentle at extremes, steep transition around 50%
      acceptance.

    Only ``todu_30ever_h6`` is adjusted (revenue columns are left unchanged
    because ``oa_amt`` is observable for rejected records).

    Parameters
    ----------
    repesca_summary:
        Aggregated repesca DataFrame with at least columns
        ``[*variables, "todu_30ever_h6"]``.
    acceptance_rates:
        Output of :func:`compute_acceptance_rates`.
    variables:
        Binning variable names.
    reject_uplift_factor:
        Scaling coefficient.  For ``"linear"``: additive slope on reject ratio.
        For ``"power"``: exponent on inverse acceptance rate.
        For ``"sigmoid"``: max uplift magnitude.
    max_risk_multiplier:
        Upper cap for the per-bin multiplier.
    method:
        ``"linear"``, ``"power"``, or ``"sigmoid"`` (see above).
    enforce_monotonicity:
        If True, apply isotonic regression to ensure multipliers are
        non-decreasing along each variable axis.

    Returns
    -------
    Copy of *repesca_summary* with ``todu_30ever_h6`` adjusted in place.
    Auxiliary columns ``acceptance_rate`` and ``reject_risk_multiplier`` are
    included for diagnostics but should be dropped before downstream merges.
    """
    # Use smoothed rates if available (from Bayesian smoothing)
    rate_col = (
        "smoothed_acceptance_rate" if "smoothed_acceptance_rate" in acceptance_rates.columns else "acceptance_rate"
    )
    merge_cols = variables + ["acceptance_rate"]
    if "smoothed_acceptance_rate" in acceptance_rates.columns:
        merge_cols = merge_cols + ["smoothed_acceptance_rate"]

    result = repesca_summary.merge(
        acceptance_rates[merge_cols],
        on=variables,
        how="left",
    )

    # Bins missing from acceptance_rates (no demand data): use median observed rate
    # as a conservative default (1.0 would mean "all accepted" = no adjustment)
    median_rate = acceptance_rates[rate_col].median()
    fallback_rate = median_rate if pd.notna(median_rate) and median_rate > 0 else 0.5
    n_missing = result[rate_col].isna().sum() if rate_col in result.columns else result["acceptance_rate"].isna().sum()
    if n_missing > 0:
        logger.warning(
            f"Parceling: {n_missing} repesca bin(s) have no demand data; "
            f"filling acceptance_rate with median={fallback_rate:.3f}"
        )
    result["acceptance_rate"] = result["acceptance_rate"].fillna(fallback_rate)
    if rate_col == "smoothed_acceptance_rate":
        result["smoothed_acceptance_rate"] = result["smoothed_acceptance_rate"].fillna(fallback_rate)

    effective_rate = result[rate_col] if rate_col in result.columns else result["acceptance_rate"]

    if method == "power":
        # Power-law: multiplier = (1 / acceptance_rate) ^ factor
        # Clamp acceptance_rate away from 0 to avoid infinity
        safe_rate = effective_rate.clip(lower=0.01)
        raw_multiplier = (1.0 / safe_rate) ** reject_uplift_factor
    elif method == "sigmoid":
        # Sigmoid: multiplier = 1 + max_uplift / (1 + exp(steepness * (rate - midpoint)))
        steepness = 10.0
        midpoint = 0.5
        raw_multiplier = 1.0 + reject_uplift_factor / (1.0 + np.exp(steepness * (effective_rate - midpoint)))
    else:
        # Linear: multiplier = 1 + factor * reject_ratio
        reject_ratio = 1.0 - effective_rate
        raw_multiplier = 1.0 + reject_uplift_factor * reject_ratio

    result["reject_risk_multiplier"] = raw_multiplier.clip(lower=1.0, upper=max_risk_multiplier)

    # Log bins with extreme acceptance rates
    if not quiet:
        n_zero = int((effective_rate <= 0.01).sum())
        n_full = int((effective_rate >= 0.99).sum())
        if n_zero > 0 or n_full > 0:
            logger.debug(
                f"Acceptance rate extremes: {n_zero} bin(s) at ≤1% (all rejected), "
                f"{n_full} bin(s) at ≥99% (all accepted)"
            )

    # Enforce monotonicity if requested
    if enforce_monotonicity:
        result = _enforce_multiplier_monotonicity(result, variables, inv_vars=inv_vars, quiet=quiet)
        # Re-clip after isotonic adjustment
        result["reject_risk_multiplier"] = result["reject_risk_multiplier"].clip(lower=1.0, upper=max_risk_multiplier)

    # Warn about bins with extreme adjustments or very few observations
    extreme_bins = (result["reject_risk_multiplier"] >= max_risk_multiplier * 0.9).sum()
    if extreme_bins > 0:
        logger.debug(
            f"Reject inference ({method}): {extreme_bins}/{len(result)} bins have multipliers "
            f"near or at the cap ({max_risk_multiplier:.1f}x). Consider reviewing reject_uplift_factor."
        )

    result["todu_30ever_h6"] = result["todu_30ever_h6"] * result["reject_risk_multiplier"]

    # Optionally apply the same uplift to H3 risk numerator.
    # When True: preserves the observed H6/H3 ratio for downstream H3→H6
    # extrapolation when that ratio is stable.
    # When False (default): keeps H3 original so extrapolation uses unscaled H3
    # — useful when the H6/H3 ratio differs between booked and rejected
    # populations (e.g., cohort maturity or selection effects).
    if "todu_30ever_h3" in result.columns:
        if apply_h3_multiplier:
            result["todu_30ever_h3"] = result["todu_30ever_h3"] * result["reject_risk_multiplier"]
        else:
            logger.debug(
                "Reject inference: H3 multiplier NOT applied (apply_h3_multiplier=False). "
                "H3 risk numerator preserved for unbiased H3→H6 extrapolation."
            )

    adjusted_bins = (result["reject_risk_multiplier"] > 1.0).sum()
    if adjusted_bins > 0:
        logger.debug(
            f"Reject inference (parceling): adjusted {adjusted_bins}/{len(result)} bins | "
            f"avg multiplier={result['reject_risk_multiplier'].mean():.3f} | "
            f"max multiplier={result['reject_risk_multiplier'].max():.3f}"
        )

    return result


def apply_reject_inference(
    repesca_summary: pd.DataFrame,
    data_demand: pd.DataFrame,
    variables: list[str],
    method: Literal["none", "parceling"] = "none",
    *,
    reject_uplift_factor: float = 1.5,
    max_risk_multiplier: float = 3.0,
    parceling_method: Literal["linear", "power", "sigmoid"] = "linear",
    bayesian_smoothing: bool = False,
    bayesian_prior_strength: float = 10.0,
    enforce_monotonicity: bool = False,
    inv_vars: list[str] | None = None,
    include_all_rejections: bool = False,
    apply_h3_multiplier: bool = False,
    acceptance_recent_months: int | None = None,
    acceptance_decay_half_life_months: float | None = None,
    acceptance_date_col: str = "mis_date",
) -> pd.DataFrame:
    """Dispatcher: apply reject-inference adjustment to repesca risk predictions.

    Parameters
    ----------
    repesca_summary:
        Aggregated repesca DataFrame (output of ``calculate_risk_values``).
    data_demand:
        Full demand population (booked + rejected).
    variables:
        Binning variable names.
    method:
        ``"none"`` returns *repesca_summary* unchanged.
        ``"parceling"`` applies per-bin acceptance-rate correction.
    reject_uplift_factor:
        Passed to :func:`apply_parceling_adjustment`.
    max_risk_multiplier:
        Passed to :func:`apply_parceling_adjustment`.
    parceling_method:
        ``"linear"``, ``"power"``, or ``"sigmoid"``, passed to :func:`apply_parceling_adjustment`.
    bayesian_smoothing:
        If True, apply Beta-Binomial posterior smoothing to acceptance rates.
    bayesian_prior_strength:
        Strength of the Beta prior for Bayesian smoothing.
    enforce_monotonicity:
        If True, enforce monotonicity on multipliers via isotonic regression.
    inv_vars:
        Variables whose higher bin values indicate lower risk. Used to set
        isotonic regression direction when *enforce_monotonicity* is True.
    include_all_rejections:
        If True, include all rejections in acceptance rate denominator.
    apply_h3_multiplier:
        If True, apply the reject risk multiplier to the H3 numerator as well
        as H6, preserving the observed H6/H3 ratio when that is stable.  Default
        False: H3 numerator is left unchanged so H3→H6 extrapolation uses
        unscaled H3 (see ``reject_apply_h3_multiplier`` in config).

    Returns
    -------
    Adjusted (or unchanged) repesca summary DataFrame.

    Raises
    ------
    ValueError
        If *method* is not a recognized value.
    """
    if method == "none":
        return repesca_summary

    if method == "parceling":
        acceptance_rates = compute_acceptance_rates(
            data_demand,
            variables,
            bayesian_smoothing=bayesian_smoothing,
            bayesian_prior_strength=bayesian_prior_strength,
            include_all_rejections=include_all_rejections,
            recent_months=acceptance_recent_months,
            decay_half_life_months=acceptance_decay_half_life_months,
            date_col=acceptance_date_col,
        )

        result = apply_parceling_adjustment(
            repesca_summary,
            acceptance_rates,
            variables,
            reject_uplift_factor=reject_uplift_factor,
            max_risk_multiplier=max_risk_multiplier,
            method=parceling_method,
            enforce_monotonicity=enforce_monotonicity,
            inv_vars=inv_vars,
            apply_h3_multiplier=apply_h3_multiplier,
        )

        # Merge per-bin confidence scores
        confidence = compute_ri_confidence(acceptance_rates, variables)
        result = result.merge(confidence, on=variables, how="left")
        result["ri_confidence"] = result["ri_confidence"].fillna(0.0)
        result["ri_bin_count"] = result["ri_bin_count"].fillna(0).astype(int)
        result["ri_bin_count_raw"] = result["ri_bin_count_raw"].fillna(0).astype(int)
        result["ri_bin_count_effective"] = result["ri_bin_count_effective"].fillna(0.0).astype(float)

        return result

    raise ValueError(f"Unknown reject inference method: {method!r}. Supported methods: 'none', 'parceling'.")
