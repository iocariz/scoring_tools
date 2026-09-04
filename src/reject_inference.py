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

#: Lower clamp on the effective acceptance rate, shared by the runtime parceling
#: uplift AND the RI-optimizer's calibration target (#58). Both must clamp at the
#: SAME floor: the optimizer scores candidate uplifts against ``booked_risk /
#: acc**gamma`` and the runtime applies ``(1/acc)**factor`` — if the optimizer
#: floored at 0.05 while the runtime floored at 0.01, bins with effective
#: acceptance in (0.01, 0.05) — the deepest-reject bins RI exists for — were
#: calibrated against a rate up to 5x the one actually applied. 0.01 matches the
#: anchor's own floor (``max(..., 0.01)`` below), so the effective rate never
#: sits below it by construction except for well-observed sub-1%-acceptance bins.
MIN_EFFECTIVE_ACCEPTANCE_RATE = 0.01

_INCLUDE_ALL_REJECTIONS_DEPRECATION_WARNED = False


def _warn_include_all_rejections_deprecated() -> None:
    """Warn once that include_all_rejections is deprecated and ignored."""
    global _INCLUDE_ALL_REJECTIONS_DEPRECATION_WARNED
    if not _INCLUDE_ALL_REJECTIONS_DEPRECATION_WARNED:
        logger.warning(
            "reject_include_all_rejections / include_all_rejections is deprecated and ignored: "
            "reject inference always uses score-only acceptance rates because the swap-in "
            "(repesca) population consists solely of score-rejected accounts. Including "
            "08-other rejections in the denominator biased inferred risk upward."
        )
        _INCLUDE_ALL_REJECTIONS_DEPRECATION_WARNED = True


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

    Only ``09-score`` rejections are counted.  ``08-other`` rejections are not
    candidates for cutoff changes — the swap-in (repesca) population consists
    solely of score-rejected accounts — so they must not enter the
    acceptance-rate denominator (doing so biases the inferred risk upward).

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
        Without time-decay the prior strength is auto-tuned via an empirical-Bayes
        random-effects estimator over the integer bin counts. Under time-decay
        (*decay_half_life_months* set) the posterior is computed on the Kish effective
        sample size ``n_eff = (Σw)² / Σw²`` using the configured *bayesian_prior_strength*
        directly — the empirical-Bayes auto-tuning is skipped, because the moment
        estimator is only valid on integer Binomial counts, not on float effective counts.
    bayesian_prior_strength:
        Strength of the Beta prior (higher = more shrinkage toward global rate). Used as a
        baseline blended with the empirical estimate on the count path, and used directly
        (no blend) on the decay path.
    include_all_rejections:
        Deprecated and ignored.  Reject inference always uses score-only
        acceptance rates because the swap-in (repesca) population consists
        solely of score-rejected accounts.  Passing ``True`` logs a one-time
        deprecation warning and otherwise has no effect.
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
    ``"n_score_rejected"`` always counts ``09-score`` rejections only.
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

    if include_all_rejections:
        _warn_include_all_rejections_deprecated()

    booked = data_demand[data_demand["status_name"] == StatusName.BOOKED.value]
    # Score-only: 08-other rejections are not swap-in candidates and must not
    # enter the acceptance-rate denominator (see include_all_rejections deprecation).
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
            # The counts-based `else` branch below is NOT reached (the outer `if decay is
            # not None` was True when evaluated), so compute the unweighted aggregates here —
            # otherwise `n_booked`/`n_rejected` stay undefined and the merge below NameErrors.
            n_booked = booked.groupby(variables).size().reset_index(name="n_booked")
            n_rejected = rejected.groupby(variables).size().reset_index(name="n_score_rejected")
            n_booked_raw = n_booked.rename(columns={"n_booked": "n_booked_raw"})
            n_rejected_raw = n_rejected.rename(columns={"n_score_rejected": "n_score_rejected_raw"})
        else:
            age_days = (max_date - pd.to_datetime(data_tmp[date_col], errors="coerce")).dt.total_seconds() / (3600 * 24)
            age_months = age_days / 30.437
            weights = np.power(2.0, -age_months.to_numpy(dtype=float) / float(decay_half_life_months))
            data_tmp["__w"] = weights

            booked_w = data_demand.loc[data_demand["status_name"] == StatusName.BOOKED.value, variables].copy()
            booked_w["__w"] = data_tmp.loc[data_demand["status_name"] == StatusName.BOOKED.value, "__w"].to_numpy()
            booked_w["__w2"] = booked_w["__w"] ** 2
            n_booked = booked_w.groupby(variables)["__w"].sum().reset_index(name="n_booked")
            n_booked_raw = booked_w.groupby(variables).size().reset_index(name="n_booked_raw")
            # Σw² per bin → needed for the Kish effective sample size used by Bayesian smoothing.
            sumw2_booked = booked_w.groupby(variables)["__w2"].sum().reset_index(name="__sumw2_booked")

            rej_mask = (data_demand["status_name"] == StatusName.REJECTED.value) & (
                data_demand["reject_reason"] == RejectReason.SCORE.value
            )
            rejected_w = data_demand.loc[rej_mask, variables].copy()
            rejected_w["__w"] = data_tmp.loc[rej_mask, "__w"].to_numpy()
            rejected_w["__w2"] = rejected_w["__w"] ** 2
            n_rejected = rejected_w.groupby(variables)["__w"].sum().reset_index(name="n_score_rejected")
            n_rejected_raw = rejected_w.groupby(variables).size().reset_index(name="n_score_rejected_raw")
            sumw2_rej = rejected_w.groupby(variables)["__w2"].sum().reset_index(name="__sumw2_rej")
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
    rates = (
        rates.merge(n_booked_raw, on=variables, how="left").merge(n_rejected_raw, on=variables, how="left").fillna(0)
    )
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

    # Effective sample size for the per-bin acceptance proportion. Under time-decay the
    # per-bin "total" is Σw (float effective counts), which understates the information
    # content of a weighted proportion. The correct value is the Kish effective sample
    # size n_eff = (Σw)² / Σw², bounded by Σw ≤ n_eff ≤ n_raw. Used only by Bayesian
    # smoothing below; kept as a local Series (the public schema is unchanged).
    if decay_half_life_months is not None:
        rates = rates.merge(sumw2_booked, on=variables, how="left").merge(sumw2_rej, on=variables, how="left").fillna(0)
        # Derive Σw and Σw² from the merged frame so n_eff aligns with `rates` by
        # construction — independent of merge row order (don't reuse the pre-merge `total`).
        sumw = rates["n_booked"] + rates["n_score_rejected"]
        sumw2_total = rates["__sumw2_booked"] + rates["__sumw2_rej"]
        n_eff = ((sumw**2) / sumw2_total).where(sumw2_total > 0, 0.0)
        rates = rates.drop(columns=["__sumw2_booked", "__sumw2_rej"])
    else:
        n_eff = total

    # Bayesian smoothing: Beta-Binomial posterior.
    #
    # The empirical-Bayes prior auto-tuning below estimates the shrinkage strength
    # from cross-bin variability using a method-of-moments random-effects model. That
    # estimator assumes integer Binomial counts: it is only statistically valid on the
    # unweighted count path. Under time-decay the per-bin evidence is Σw (float
    # effective counts), so:
    #   - we compute the posterior on the Kish effective sample size n_eff (above),
    #     not on Σw, to avoid systematically over-shrinking toward the global rate; and
    #   - we use the analyst-configured `bayesian_prior_strength` directly rather than
    #     auto-tuning a prior from a handful of weighted bins (where the moment
    #     estimator is unsound).
    # This deliberate asymmetry keeps the count path bit-identical while making the
    # decay path defensible under independent model validation.
    if bayesian_smoothing:
        # Zero-guarded denominator. The old max(total.sum(), 1) clamped the total to >= 1, which
        # is fine for integer counts but WRONG under time-decay where `total` is Σw (float
        # effective weight): when Σw < 1 (heavily-decayed / sparse bins) the clamp understated
        # global_rate. Divide by the true Σw; fall back to 0 only when there is no weight at all.
        _denom = float(total.sum())
        global_rate = float(rates["n_booked"].sum() / _denom) if _denom > 0 else 0.0
        if decay_half_life_months is None:
            # Empirical-Bayes adjustment (integer-count path only):
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
        else:
            # Decay path: no auto-tune — use the configured prior strength directly.
            effective_prior_strength = float(bayesian_prior_strength)

        effective_prior_strength = float(np.clip(effective_prior_strength, 0.5, 1000.0))
        # Unbiased Beta prior: mean alpha/(alpha+beta) == global_rate by construction, so
        # smoothing shrinks toward the true global acceptance rate. The old per-component
        # max(., 0.5) floors were applied ASYMMETRICALLY: when global_rate was near 0 (or 1)
        # they raised the smaller component to 0.5, pulling the prior mean toward 0.5 — e.g. a
        # deep-reject global_rate=0.016 became a prior mean of 0.025 (+55%), inflating inferred
        # acceptance in exactly the riskiest bins (anti-conservative). Keep the configured
        # strength (already clipped); clip only global_rate away from exact 0/1 to avoid a
        # degenerate zero component.
        gr = float(np.clip(global_rate, 1e-6, 1.0 - 1e-6))
        alpha = effective_prior_strength * gr
        beta = effective_prior_strength * (1.0 - gr)
        if decay_half_life_months is None:
            # Literal count-based posterior (kept bit-identical for regression tests).
            rates["smoothed_acceptance_rate"] = (rates["n_booked"] + alpha) / (total + alpha + beta)
        else:
            # Weighted posterior on the Kish effective sample size.
            rates["smoothed_acceptance_rate"] = (rates["acceptance_rate"] * n_eff + alpha) / (n_eff + alpha + beta)
        logger.debug(
            f"Bayesian smoothing applied | decay={'on' if decay_half_life_months is not None else 'off'} | "
            f"auto_tune={'off' if decay_half_life_months is not None else 'on'} | "
            f"prior_strength(cfg)={bayesian_prior_strength:.3f} | "
            f"prior_strength(eff)={effective_prior_strength:.3f} | global_rate={global_rate:.3f} | "
            f"alpha={alpha:.2f}, beta={beta:.2f}"
        )

    # Non-score (08-other) rejections are excluded from the denominator by design
    # (they are not swap-in candidates).  Surface their share so sparse or skewed
    # bins can be assessed — but stabilize via Bayesian smoothing / min-obs, not by
    # polluting the rate with rejections that can never be swapped in.
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
            f"({other_pct:.1f}% of demand, {other_of_rejected_pct:.1f}% of all rejections) "
            f"by design (not swap-in candidates)."
        )
        if other_pct > 10.0:
            logger.warning(
                "Non-score rejections exceed 10% of demand. If bin-level acceptance rates "
                "look noisy, rely on Bayesian smoothing / min-obs rather than including "
                "non-score rejections."
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


def _shrink_acceptance_rate(
    target: pd.DataFrame,
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    rate_col: str,
    *,
    anchor_percentile: float = 0.10,
    confidence_scale: float = 10.0,
) -> tuple[pd.Series, float, int]:
    """Confidence-weighted shrinkage of per-bin acceptance rates toward a conservative anchor.

    Cells with little or no demand carry the highest selection-bias uncertainty, so their
    acceptance rate is shrunk toward a conservative *low* anchor (low acceptance ⇒ high reject
    multiplier) rather than trusted at face value or filled with the (anti-conservative) median::

        rate_eff = conf · bin_rate + (1 − conf) · anchor
        conf     = 1 − exp(−(n_booked + n_score_rejected) / confidence_scale)
        anchor   = max(low-percentile of observed rates, 0.01)   # fixed 0.05 if none observed

    No-demand cells (absent from *acceptance_rates*) get ``conf = 0`` ⇒ ``rate_eff = anchor``.
    Well-observed cells get ``conf ≈ 1`` ⇒ ``rate_eff ≈ bin_rate`` (near-unchanged). This is a
    deliberate risk-pessimism prior — the anchor is a *low* percentile, not the global mean.

    Shared by :func:`apply_parceling_adjustment` (runtime) and the RI optimizer's calibration
    objective so both score sparse cells on the same rate definition.

    Parameters
    ----------
    target:
        Frame whose rows (keyed by *variables*) need an effective acceptance rate.
    acceptance_rates:
        Output of :func:`compute_acceptance_rates`; supplies the observed ``rate_col`` and the
        ``n_booked`` / ``n_score_rejected`` counts used for confidence and the anchor.
    rate_col:
        ``"smoothed_acceptance_rate"`` when Bayesian smoothing is on, else ``"acceptance_rate"``.
    anchor_percentile:
        Percentile of observed rates used as the conservative anchor (lower = more conservative).
    confidence_scale:
        Count scale in ``conf = 1 − exp(−n / scale)``; smaller = trust counts faster, so only
        genuinely sparse cells shrink.

    Returns
    -------
    ``(rate_eff, anchor, n_no_demand)`` — ``rate_eff`` is a Series aligned to *target*'s index,
    ``anchor`` the conservative anchor used, ``n_no_demand`` the count of zero-demand target cells.
    """
    if len(acceptance_rates) and {"n_booked", "n_score_rejected"}.issubset(acceptance_rates.columns):
        counts = (acceptance_rates["n_booked"] + acceptance_rates["n_score_rejected"]).to_numpy(dtype=float)
    else:
        # No counts available (e.g. hand-built rates): treat every observed bin as fully confident.
        counts = np.full(len(acceptance_rates), np.inf)

    # Conservative anchor: a low percentile of rates on cells with real demand. Robust to a single
    # all-rejected bin (unlike min); adapts across segments (unlike a magic constant).
    observed = acceptance_rates[rate_col].to_numpy(dtype=float)[counts > 0] if len(acceptance_rates) else np.array([])
    observed = observed[np.isfinite(observed)]
    anchor = (
        max(float(np.quantile(observed, anchor_percentile)), MIN_EFFECTIVE_ACCEPTANCE_RATE) if observed.size else 0.05
    )

    conf_tbl = acceptance_rates[variables].copy()
    conf_tbl["__n"] = counts
    conf_tbl["__rate"] = acceptance_rates[rate_col].to_numpy(dtype=float)
    merged = target[variables].merge(conf_tbl, on=variables, how="left")

    n = merged["__n"].fillna(0.0).to_numpy(dtype=float)
    bin_rate = merged["__rate"].fillna(anchor).to_numpy(dtype=float)
    # conf = 1 - exp(-n/scale); n=inf (no counts) ⇒ conf=1 (use bin_rate); n=0 ⇒ conf=0 (use anchor).
    conf = 1.0 - np.exp(-n / float(confidence_scale))
    rate_eff = conf * bin_rate + (1.0 - conf) * anchor
    n_no_demand = int((n <= 0).sum())
    return pd.Series(rate_eff, index=target.index), anchor, n_no_demand


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
            logger.debug(
                f"Isotonic monotonicity converged after {iteration + 1} iteration(s) (max_change={max_change:.2e})"
            )
            break
    else:
        logger.debug(
            f"Isotonic monotonicity did not converge after {max_iterations} iterations (max_change={max_change:.2e})"
        )

    # Post-hoc: project onto monotonicity over the full cell poset (block-pooling PAVA, audit #17).
    # A cell a dominates b if a[v] >= b[v] for all v (respecting direction) ⇒ multiplier[a] >= multiplier[b].
    # Guarantees zero residual violations after the alternating axis-wise warm-start.
    if len(variables) >= 2 and len(result) > 1:
        n_merges = _fix_partial_order_violations(result, variables, inv_set)
        if n_merges > 0:
            log_fn = logger.debug if quiet else logger.warning
            log_fn(
                f"Isotonic post-hoc: pooled {n_merges} block(s) to clear partial-order violations "
                f"remaining after the axis-wise warm-start."
            )

    return result


def _fix_partial_order_violations(
    result: pd.DataFrame,
    variables: list[str],
    inv_set: set[str],
    *,
    tol: float = 1e-9,
) -> int:
    """Project the multiplier surface onto monotonicity over the cell poset (audit #17).

    A generalized PAVA: whenever a dominating pair violates the order (cell ``i`` is riskier than
    ``j`` in every dimension but ``mult[i] < mult[j]``), the two cells' **blocks** are pooled to the
    unweighted mean of their members via union-find, and the scan repeats until no violation remains.
    Pooling whole blocks (level sets) rather than averaging isolated pairs is what makes this a valid
    isotonic projection — and because blocks only ever merge (≤ n−1 merges) it is guaranteed to
    terminate with **zero** residual partial-order violations (the old pairwise-average + ``range(5)``
    loop could re-break pairs and exit non-converged).

    Returns the number of block merges performed.
    """
    n = len(result)
    if n < 2:
        return 0

    vals = result["reject_risk_multiplier"].to_numpy(dtype=float)
    # Orient coordinates so that higher = riskier in all dimensions.
    signs = np.array([(-1 if v in inv_set else 1) for v in variables])
    oriented = result[variables].to_numpy() * signs

    # Strictly-dominating ordered pairs: i dominates j ⇒ constraint mult[i] >= mult[j].
    dom_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = oriented[i] - oriented[j]
            if np.all(diff >= 0) and np.any(diff > 0):
                dom_pairs.append((i, j))
    if not dom_pairs:
        return 0

    # Union-find over cells; a block's value is the unweighted mean of its members' multipliers.
    parent = list(range(n))
    block_sum = vals.copy()
    block_size = np.ones(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    while True:
        changed = False
        for i, j in dom_pairs:
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            if block_sum[ri] / block_size[ri] < block_sum[rj] / block_size[rj] - tol:
                # i dominates j but block(i) is less risky than block(j): pool the two blocks.
                parent[rj] = ri
                block_sum[ri] += block_sum[rj]
                block_size[ri] += block_size[rj]
                merges += 1
                changed = True
        if not changed:
            break

    if merges:
        result["reject_risk_multiplier"] = np.array([block_sum[find(k)] / block_size[find(k)] for k in range(n)])
    return merges


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
    no_demand_anchor_percentile: float = 0.10,
    confidence_scale: float = 10.0,
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
    no_demand_anchor_percentile:
        Percentile of observed acceptance rates used as the conservative anchor that
        no/low-demand bins are shrunk toward (lower ⇒ more conservative). See
        :func:`_shrink_acceptance_rate`.
    confidence_scale:
        Count scale in ``conf = 1 − exp(−n / scale)``; smaller ⇒ confidence saturates
        faster, so only genuinely sparse bins shrink toward the anchor.

    Notes
    -----
    Before applying the chosen method, each bin's acceptance rate is replaced by a
    confidence-weighted blend toward a conservative low anchor (see
    :func:`_shrink_acceptance_rate`): no-demand bins (the highest selection-bias
    uncertainty) collapse to the anchor → high uplift, while well-observed bins are
    left essentially unchanged. When Bayesian smoothing is on this composes with the
    smoothed rate (smoothing pulls toward the global rate, this toward the low anchor).

    Returns
    -------
    Copy of *repesca_summary* with ``todu_30ever_h6`` adjusted in place.
    Auxiliary columns ``acceptance_rate``, ``ri_effective_acceptance_rate`` (the shrunk
    rate actually used) and ``reject_risk_multiplier`` are included for diagnostics but
    should be dropped before downstream merges.
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

    # No-demand / low-demand bins carry the highest selection-bias uncertainty. Instead of the
    # (anti-conservative) median observed rate, shrink every bin's rate toward a conservative LOW
    # anchor by confidence: no-demand → anchor (high uplift), well-observed → ~unchanged. This also
    # de-trusts sparse-but-nonzero bins (audit #37/#81). See _shrink_acceptance_rate.
    effective_rate, anchor, n_no_demand = _shrink_acceptance_rate(
        result,
        acceptance_rates,
        variables,
        rate_col,
        anchor_percentile=no_demand_anchor_percentile,
        confidence_scale=confidence_scale,
    )
    if n_no_demand > 0 and not quiet:
        logger.warning(
            f"Parceling: {n_no_demand} repesca bin(s) have no demand data; "
            f"shrinking toward conservative anchor={anchor:.3f} (confidence-weighted)."
        )
    # Persisted diagnostics: keep the observed acceptance_rate (anchor-filled so no NaN) for the
    # extremes log below; expose the shrunk rate actually used as ri_effective_acceptance_rate.
    result["acceptance_rate"] = result["acceptance_rate"].fillna(anchor)
    if rate_col == "smoothed_acceptance_rate" and "smoothed_acceptance_rate" in result.columns:
        result["smoothed_acceptance_rate"] = result["smoothed_acceptance_rate"].fillna(anchor)
    result["ri_effective_acceptance_rate"] = effective_rate.to_numpy()

    if method == "power":
        # Power-law: multiplier = (1 / acceptance_rate) ^ factor
        # Clamp acceptance_rate away from 0 to avoid infinity (#58: the SAME
        # floor the optimizer's calibration target uses).
        safe_rate = effective_rate.clip(lower=MIN_EFFECTIVE_ACCEPTANCE_RATE)
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
        n_zero = int((effective_rate <= MIN_EFFECTIVE_ACCEPTANCE_RATE).sum())
        n_full = int((effective_rate >= 0.99).sum())
        if n_zero > 0 or n_full > 0:
            logger.debug(
                f"Acceptance rate extremes: {n_zero} bin(s) at ≤1% (all rejected), "
                f"{n_full} bin(s) at ≥99% (all accepted)"
            )

    # Enforce monotonicity if requested. Clip FIRST, then enforce (audit #17): a uniform clip is
    # order-preserving, and pooled means of in-range values stay in range, so the monotone projection
    # is the final word and the output is provably both monotone and within [1, max_risk_multiplier].
    if enforce_monotonicity:
        result["reject_risk_multiplier"] = result["reject_risk_multiplier"].clip(lower=1.0, upper=max_risk_multiplier)
        result = _enforce_multiplier_monotonicity(result, variables, inv_vars=inv_vars, quiet=quiet)

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
    no_demand_anchor_percentile: float = 0.10,
    confidence_scale: float = 10.0,
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
    no_demand_anchor_percentile:
        Percentile of observed acceptance rates used as the conservative anchor for
        no/low-demand bins (lower = more conservative).  See :func:`_shrink_acceptance_rate`.
    confidence_scale:
        Count scale controlling how fast confidence saturates; smaller means only genuinely
        sparse bins shrink toward the anchor.  Also used for the ``ri_confidence`` diagnostic.

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
            no_demand_anchor_percentile=no_demand_anchor_percentile,
            confidence_scale=confidence_scale,
        )

        # Merge per-bin confidence scores (same scale as the shrinkage above, for consistency)
        confidence = compute_ri_confidence(acceptance_rates, variables, scale=confidence_scale)
        result = result.merge(confidence, on=variables, how="left")
        result["ri_confidence"] = result["ri_confidence"].fillna(0.0)
        result["ri_bin_count"] = result["ri_bin_count"].fillna(0).astype(int)
        result["ri_bin_count_raw"] = result["ri_bin_count_raw"].fillna(0).astype(int)
        result["ri_bin_count_effective"] = result["ri_bin_count_effective"].fillna(0.0).astype(float)

        return result

    raise ValueError(f"Unknown reject inference method: {method!r}. Supported methods: 'none', 'parceling'.")
