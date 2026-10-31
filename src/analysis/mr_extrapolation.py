"""H3 -> H6 risk extrapolation (curve fitting + application), used by the MR pipeline."""

import numpy as np
import pandas as pd
from loguru import logger


def extrapolate_h3_to_h6(
    b2_h3: pd.Series | np.ndarray | float,
    h6_h3_ratio: pd.Series | np.ndarray | float,
    method: str = "linear",
    curvature: float = 1.0,
    b2_h3_main: pd.Series | np.ndarray | float | None = None,
) -> pd.Series | np.ndarray | float:
    """Extrapolate H6 risk from observed H3 risk using a configurable curve.

    Args:
        b2_h3: Observed H3 risk metric values (MR period).
        h6_h3_ratio: Main-period H6/H3 ratio used as base scaling factor.
        method: Extrapolation curve — ``"linear"``, ``"power"``, or ``"logistic"``.
        curvature: Tuning parameter. For **power**: exponent alpha (1.0 = linear).
            For **logistic**: steepness *k*. Ignored for linear.
        b2_h3_main: Main-period H3 risk per bin (required for ``"power"``).
            Used as the reference point so that the power law is applied to
            the deviation ``b2_h3 / b2_h3_main`` rather than to the ratio.

    Returns:
        Extrapolated H6 risk values, same type/shape as *b2_h3*.
    """
    if method == "linear":
        return b2_h3 * h6_h3_ratio
    elif method == "power":
        # Power law: b2_h6 = exp(c) * b2_h3^alpha  fitted via log-log regression.
        # Using h6_h3_ratio = b2_h6_main / b2_h3_main and the fitted alpha:
        #   b2_h6_mr = b2_h3_mr * ratio * (b2_h3_mr / b2_h3_main)^(alpha - 1)
        # This matches the fitted model and degrades to linear when alpha=1.
        if b2_h3_main is not None:
            main_arr = np.asarray(b2_h3_main, dtype=float)
            safe_main = np.where(main_arr > 0, main_arr, np.nan)
            deviation = np.asarray(b2_h3, dtype=float) / safe_main
            # Clip deviation to [0.01, 100] to prevent extreme extrapolation:
            # 0.01 lower bound avoids near-zero base causing huge inverse power,
            # 100 upper bound caps at 100× the main-period risk level.
            power_result = b2_h3 * h6_h3_ratio * np.power(np.clip(deviation, 0.01, 100.0), curvature - 1.0)
            # Fall back to linear for elements where b2_h3_main is NaN/0 (e.g. MR-only bins)
            linear_result = b2_h3 * h6_h3_ratio
            return np.where(np.isfinite(power_result), power_result, linear_result)
        # Fallback when b2_h3_main not available: apply curvature to ratio (legacy).
        # This path can diverge from the fitted log-log model used when b2_h3_main
        # is available — results should be treated as approximate.
        logger.warning(
            "H3→H6 power extrapolation: b2_h3_main not available, using legacy "
            "ratio^curvature fallback (may diverge from fitted log-log model)."
        )
        return b2_h3 * np.power(h6_h3_ratio, curvature)
    elif method == "logistic":
        # Smoothly caps the scaling for extreme ratios while approaching
        # linear for moderate ratios (ratio ≈ 1).
        # Uses tanh to compress deviations: cap at 1 + 2/curvature.
        # Slope at ratio=1 equals 1 (matches linear for small deviations).
        scale = 1 + 2 * np.tanh(curvature * (h6_h3_ratio - 1) / 2) / curvature
        return b2_h3 * scale
    else:
        raise ValueError(f"Unknown extrapolation method: {method!r}. Use 'linear', 'power', or 'logistic'.")


def fit_h3_extrapolation_curve(
    b2_h3: np.ndarray,
    b2_h6: np.ndarray,
    weights: np.ndarray | None = None,
    min_bins: int = 4,
) -> tuple[str, float, dict]:
    """Fit the H3→H6 extrapolation curvature from observed main-period data.

    Performs weighted log-log regression ``log(b2_h6) = c + alpha * log(b2_h3)``
    to determine whether the relationship is convex (alpha > 1), linear (~1),
    or concave (alpha < 1).

    ECOLOGICAL-TRANSFER CAVEAT: this alpha is a **cross-bin / cross-sectional** slope (variation is
    across score bins in the mature main period). It is then applied to project a single bin's OWN
    observed H3 forward to H6 over time (within-bin, temporal) — assuming the cross-sectional
    curvature equals the temporal maturation curvature within a bin. That need not hold. When the
    main period has ≥2 sub-periods, use :func:`fit_h3_curvature_within_bin_temporal` to estimate the
    within-bin temporal alpha and compare: a large gap means this cross-bin alpha is a poor proxy for
    maturation and the extrapolated MR risk carries model risk beyond the reported ``se``.

    Args:
        b2_h3: Per-bin H3 risk values (main period).
        b2_h6: Per-bin H6 risk values (main period).
        weights: Optional observation counts per bin for weighted regression.
        min_bins: Minimum valid bins required for fitting.

    Returns:
        ``(method, curvature, diagnostics)`` where *method* is ``"linear"`` or
        ``"power"``, *curvature* is the fitted alpha (clipped to [0.3, 3.0]),
        and *diagnostics* is a dict with ``alpha``, ``se``, ``r_squared``,
        ``n_bins``, and ``weighting_scheme`` (``"inverse_variance_proxy"`` when
        weights are provided, ``"uniform"`` otherwise).
    """
    valid = (np.asarray(b2_h3) > 0) & (np.asarray(b2_h6) > 0)
    n_valid = int(valid.sum())

    fallback_diag = {"alpha": 1.0, "se": float("nan"), "r_squared": float("nan"), "n_bins": n_valid}
    if n_valid < min_bins:
        return ("linear", 1.0, {**fallback_diag, "note": "insufficient bins"})

    log_h3 = np.log(np.asarray(b2_h3)[valid])
    log_h6 = np.log(np.asarray(b2_h6)[valid])
    w_raw = np.asarray(weights)[valid].astype(float) if weights is not None else None

    # Variance-aware weighting (heteroscedasticity proxy):
    # For log-risk fits, lower-risk / lower-exposure bins are noisier.
    # Build an inverse-variance style weight using exposure (weights) and
    # risk levels from both H3 and H6:
    #   var_proxy ~ 1/(n*b2_h6) + 1/(n*b2_h3)
    # then polyfit weight uses inverse std-dev:
    #   w_fit ~ 1/sqrt(var_proxy)
    #
    # If no weights are provided, keep unweighted fit for backward compatibility.
    if w_raw is not None:
        b3 = np.asarray(b2_h3)[valid].astype(float)
        b6 = np.asarray(b2_h6)[valid].astype(float)
        eps = 1e-12
        n_eff = np.clip(w_raw, eps, None)
        b3 = np.clip(b3, eps, None)
        b6 = np.clip(b6, eps, None)
        var_proxy = 1.0 / (n_eff * b6) + 1.0 / (n_eff * b3)
        var_proxy = np.clip(
            var_proxy, eps, np.nanpercentile(var_proxy, 99.5) if len(var_proxy) > 5 else np.max(var_proxy)
        )
        w_fit = 1.0 / np.sqrt(var_proxy)
        eff_w = w_fit**2
        weighting_scheme = "inverse_variance_proxy"
    else:
        w_fit = None
        eff_w = np.ones_like(log_h3)
        weighting_scheme = "uniform"
    x_mean = np.average(log_h3, weights=eff_w)
    ss_x = float(np.sum(eff_w * (log_h3 - x_mean) ** 2))

    if not np.isfinite(ss_x) or ss_x <= 1e-10:
        return ("linear", 1.0, {**fallback_diag, "note": "ill_conditioned_design"})

    coeffs = np.polyfit(log_h3, log_h6, 1, w=w_fit)
    alpha = float(coeffs[0])
    intercept = float(coeffs[1])

    predicted = alpha * log_h3 + intercept
    residuals = log_h6 - predicted
    ss_res = float(np.sum(eff_w * residuals**2))
    ss_tot = float(np.sum(eff_w * (log_h6 - np.average(log_h6, weights=eff_w)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mse = ss_res / max(n_valid - 2, 1)
    se = float(np.sqrt(mse / ss_x)) if ss_x > 0 else float("inf")

    diagnostics = {
        "alpha": alpha,
        "se": se,
        "r_squared": r_squared,
        "n_bins": n_valid,
        "weighting_scheme": weighting_scheme,
    }

    # Decision: if 95% CI includes 1.0, use linear
    if abs(alpha - 1.0) < 2.0 * se:
        return ("linear", 1.0, diagnostics)

    curvature = float(np.clip(alpha, 0.3, 3.0))
    return ("power", curvature, diagnostics)


def fit_h3_curvature_within_bin_temporal(
    bin_ids: np.ndarray,
    b2_h3: np.ndarray,
    b2_h6: np.ndarray,
    weights: np.ndarray | None = None,
    min_bins: int = 2,
    min_pairs: int = 4,
) -> tuple[float, dict]:
    """Estimate the H3→H6 curvature from **within-bin temporal** variation (ecological-transfer check).

    :func:`fit_h3_extrapolation_curve` fits a **cross-bin** (cross-sectional) slope: it regresses
    ``log(b2_h6)`` on ``log(b2_h3)`` across the score bins in the mature main period. That slope is
    then applied to project a single bin's OWN observed H3 forward to H6 over time (within-bin,
    temporal). That is an **ecological transfer** — it assumes the cross-sectional H6~H3 curvature
    equals the temporal H3→H6 maturation curvature *within* a bin, which need not hold (the cross
    section mixes bins of different populations; maturation is a different mechanism).

    This function is the direct validation when the main period has ≥2 sub-periods. Given
    ``(bin, period)`` cells — each carrying that bin's H3 and H6 rate in one period — it removes each
    bin's own (weighted) mean in log space (a bin fixed effect) and fits the pooled slope on the
    residual variation, i.e. how a bin's H6 moves with its OWN H3 across time. Comparing the returned
    ``alpha_within`` to the cross-bin alpha validates (or refutes) the ecological transfer: close ⇒
    the transfer holds on this data; far apart ⇒ the cross-bin curvature is a poor proxy for
    maturation and the extrapolated MR risk carries model risk beyond the reported SE.

    Args:
        bin_ids: Bin identifier per cell (rows with the same id are the same score bin in different
            periods). Any hashable/array dtype; only equality is used.
        b2_h3: H3 risk per cell (same order as ``bin_ids``).
        b2_h6: H6 risk per cell.
        weights: Optional exposure/count per cell (e.g. booked accounts); used both to weight each
            bin's mean and the pooled slope. Uniform if omitted.
        min_bins: Minimum number of bins that appear in ≥2 periods (else return NaN).
        min_pairs: Minimum usable cells across those multi-period bins (else return NaN).

    Returns:
        ``(alpha_within, diagnostics)``. ``alpha_within`` is ``nan`` when there is insufficient
        within-bin temporal variation. ``diagnostics`` carries ``alpha``, ``se``, ``n_bins``
        (multi-period bins used), ``n_pairs`` (cells used), and a ``note`` when degenerate.
    """
    bin_ids = np.asarray(bin_ids)
    b2_h3 = np.asarray(b2_h3, dtype=float)
    b2_h6 = np.asarray(b2_h6, dtype=float)
    w = np.asarray(weights, dtype=float) if weights is not None else np.ones_like(b2_h3)

    valid = np.isfinite(b2_h3) & np.isfinite(b2_h6) & (b2_h3 > 0) & (b2_h6 > 0) & np.isfinite(w) & (w > 0)
    nan_diag = {"alpha": float("nan"), "se": float("nan"), "n_bins": 0, "n_pairs": 0}
    if valid.sum() < min_pairs:
        return (float("nan"), {**nan_diag, "note": "insufficient valid cells"})

    bids = bin_ids[valid]
    log_h3 = np.log(b2_h3[valid])
    log_h6 = np.log(b2_h6[valid])
    wv = w[valid]

    # Keep only bins that appear in ≥2 periods (≥2 cells) — those carry temporal variation.
    rx, ry, rw = [], [], []
    n_bins_used = 0
    for b in np.unique(bids):
        m = bids == b
        if m.sum() < 2:
            continue
        wb = wv[m]
        wsum = wb.sum()
        if wsum <= 0:
            continue
        # De-mean in log space with the bin's own weighted mean (remove the bin fixed effect),
        # leaving only within-bin temporal variation.
        mx = np.average(log_h3[m], weights=wb)
        my = np.average(log_h6[m], weights=wb)
        rx.append(log_h3[m] - mx)
        ry.append(log_h6[m] - my)
        rw.append(wb)
        n_bins_used += 1

    if n_bins_used < min_bins:
        return (float("nan"), {**nan_diag, "note": "too few multi-period bins", "n_bins": n_bins_used})

    rx = np.concatenate(rx)
    ry = np.concatenate(ry)
    rw = np.concatenate(rw)
    n_pairs = int(rx.size)

    ss_x = float(np.sum(rw * rx**2))
    if not np.isfinite(ss_x) or ss_x <= 1e-10:
        return (
            float("nan"),
            {**nan_diag, "note": "no within-bin log-H3 spread", "n_bins": n_bins_used, "n_pairs": n_pairs},
        )

    # Slope through the origin (data are within-bin de-meaned ⇒ intercept is 0 by construction).
    alpha = float(np.sum(rw * rx * ry) / ss_x)
    residuals = ry - alpha * rx
    # dof: n_pairs observations − n_bins_used means already spent − 1 slope.
    dof = max(n_pairs - n_bins_used - 1, 1)
    mse = float(np.sum(rw * residuals**2) / dof)
    se = float(np.sqrt(mse / ss_x)) if ss_x > 0 else float("inf")

    return (alpha, {"alpha": alpha, "se": se, "n_bins": n_bins_used, "n_pairs": n_pairs})
