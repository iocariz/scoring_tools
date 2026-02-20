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

import pandas as pd
from loguru import logger

from src.constants import RejectReason, StatusName


def compute_acceptance_rates(
    data_demand: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """Compute per-bin acceptance rates from the demand population.

    For each unique combination of *variables* bins, the acceptance rate is
    defined as::

        acceptance_rate = n_booked / (n_booked + n_score_rejected)

    Only ``09-score`` rejections are counted (``08-other`` are excluded because
    they are not candidates for cutoff changes).

    Parameters
    ----------
    data_demand:
        Full demand DataFrame containing both booked and rejected records.
        Must have columns ``status_name``, ``reject_reason``, and the columns
        listed in *variables*.
    variables:
        The two binning variable names (e.g. ``["sc_octroi_new_clus", "new_efx_clus"]``).

    Returns
    -------
    DataFrame with columns ``[*variables, "n_booked", "n_score_rejected", "acceptance_rate"]``.
    """
    booked = data_demand[data_demand["status_name"] == StatusName.BOOKED.value]
    score_rejected = data_demand[
        (data_demand["status_name"] == StatusName.REJECTED.value)
        & (data_demand["reject_reason"] == RejectReason.SCORE.value)
    ]

    n_booked = booked.groupby(variables).size().reset_index(name="n_booked")
    n_rejected = score_rejected.groupby(variables).size().reset_index(name="n_score_rejected")

    rates = n_booked.merge(n_rejected, on=variables, how="outer").fillna(0)
    rates["n_booked"] = rates["n_booked"].astype(int)
    rates["n_score_rejected"] = rates["n_score_rejected"].astype(int)

    total = rates["n_booked"] + rates["n_score_rejected"]
    rates["acceptance_rate"] = (rates["n_booked"] / total).where(total > 0, 0.0)

    # Warn about non-score rejections that are excluded from acceptance rate
    n_other_rejected = len(
        data_demand[
            (data_demand["status_name"] == StatusName.REJECTED.value)
            & (data_demand["reject_reason"] != RejectReason.SCORE.value)
        ]
    )
    n_total_demand = len(data_demand)
    if n_other_rejected > 0:
        other_pct = n_other_rejected / n_total_demand * 100
        logger.info(
            f"Acceptance rates exclude {n_other_rejected:,} non-score rejections "
            f"({other_pct:.1f}% of demand). Bins with many non-score rejections "
            f"may have overstated acceptance rates."
        )

    logger.debug(
        f"Acceptance rates computed for {len(rates)} bins | "
        f"mean={rates['acceptance_rate'].mean():.3f} | "
        f"min={rates['acceptance_rate'].min():.3f} | "
        f"max={rates['acceptance_rate'].max():.3f}"
    )

    return rates


def apply_parceling_adjustment(
    repesca_summary: pd.DataFrame,
    acceptance_rates: pd.DataFrame,
    variables: list[str],
    *,
    reject_uplift_factor: float = 1.5,
    max_risk_multiplier: float = 3.0,
    method: Literal["linear", "power"] = "linear",
) -> pd.DataFrame:
    """Apply per-bin risk uplift to repesca summary based on acceptance rates.

    Two methods are available:

    - ``"linear"`` (default): ``multiplier = 1 + factor * (1 - acceptance_rate)``
    - ``"power"``: ``multiplier = (1 / acceptance_rate) ^ factor``.  This is
      grounded in the assumption that rejected applicants are drawn from the
      riskier tail, so risk scales as a power of the inverse acceptance rate.
      It produces a non-linear curve that grows faster at low acceptance rates.

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
    max_risk_multiplier:
        Upper cap for the per-bin multiplier.
    method:
        ``"linear"`` or ``"power"`` (see above).

    Returns
    -------
    Copy of *repesca_summary* with ``todu_30ever_h6`` adjusted in place.
    Auxiliary columns ``acceptance_rate`` and ``reject_risk_multiplier`` are
    included for diagnostics but should be dropped before downstream merges.
    """
    result = repesca_summary.merge(
        acceptance_rates[variables + ["acceptance_rate"]],
        on=variables,
        how="left",
    )

    # Bins missing from acceptance_rates (no demand data) get no adjustment
    result["acceptance_rate"] = result["acceptance_rate"].fillna(1.0)

    if method == "power":
        # Power-law: multiplier = (1 / acceptance_rate) ^ factor
        # Clamp acceptance_rate away from 0 to avoid infinity
        safe_rate = result["acceptance_rate"].clip(lower=0.01)
        raw_multiplier = (1.0 / safe_rate) ** reject_uplift_factor
    else:
        # Linear: multiplier = 1 + factor * reject_ratio
        reject_ratio = 1.0 - result["acceptance_rate"]
        raw_multiplier = 1.0 + reject_uplift_factor * reject_ratio

    result["reject_risk_multiplier"] = raw_multiplier.clip(lower=1.0, upper=max_risk_multiplier)

    # Warn about bins with extreme adjustments or very few observations
    extreme_bins = (result["reject_risk_multiplier"] >= max_risk_multiplier * 0.9).sum()
    if extreme_bins > 0:
        logger.warning(
            f"Reject inference ({method}): {extreme_bins}/{len(result)} bins have multipliers "
            f"near or at the cap ({max_risk_multiplier:.1f}x). Consider reviewing reject_uplift_factor."
        )

    result["todu_30ever_h6"] = result["todu_30ever_h6"] * result["reject_risk_multiplier"]

    adjusted_bins = (result["reject_risk_multiplier"] > 1.0).sum()
    if adjusted_bins > 0:
        logger.info(
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
        acceptance_rates = compute_acceptance_rates(data_demand, variables)
        return apply_parceling_adjustment(
            repesca_summary,
            acceptance_rates,
            variables,
            reject_uplift_factor=reject_uplift_factor,
            max_risk_multiplier=max_risk_multiplier,
        )

    raise ValueError(f"Unknown reject inference method: {method!r}. Supported methods: 'none', 'parceling'.")
