"""Stress factors (global and per-bin) applied to the booked risk during preprocessing."""

import pandas as pd
from loguru import logger

from src.core.constants import DEFAULT_RISK_MULTIPLIER

# A stress factor multiplies the risk predicted for rejected applicants by how much worse the
# worst-score booked tail is than the booked average. It makes risk worse or leaves it alone:
# a tail that shows LESS risk than the average is sampling noise on a handful of loans, not
# evidence that rejected applicants are safer (audit 2026-11-27 A20).
NEUTRAL_STRESS_FACTOR = 1.0


def calculate_stress_factor(
    df: pd.DataFrame,
    status_col: str = "status_name",
    score_col: str = "risk_score_rf",
    num_col: str = "todu_30ever_h6",
    den_col: str = "todu_amt_pile_h6",
    frac: float = 0.05,
    target_status: str = "booked",
    bad_rate: float = 0.05,
    higher_is_worse: bool = False,
) -> float:
    # Filter for target status
    df_target = df[df[status_col] == target_status].copy()

    if df_target.empty:
        logger.warning(f"No records found with {status_col} = {target_status}; returning neutral stress factor 1.0")
        return 1.0

    # Calculate overall bad rate
    total_num = df_target[num_col].sum()
    total_den = df_target[den_col].sum()

    overall_bad_rate = (total_num / total_den * DEFAULT_RISK_MULTIPLIER) if total_den > 0 else bad_rate

    # Select worst population based on score cutoff
    if higher_is_worse:
        cutoff_score = df_target[score_col].quantile(1.0 - frac)
        df_worst = df_target[df_target[score_col] >= cutoff_score]
    else:
        cutoff_score = df_target[score_col].quantile(frac)
        df_worst = df_target[df_target[score_col] <= cutoff_score]

    logger.debug(f"Score cutoff (frac={frac}): {cutoff_score}")
    logger.debug(
        f"Selected {len(df_worst)}/{len(df_target)} records ({len(df_worst) / len(df_target):.2%}) as worst population"
    )

    # Calculate bad rate for worst fraction
    worst_num = df_worst[num_col].sum()
    worst_den = df_worst[den_col].sum()

    if worst_den <= 0:
        # #54: no exposure in the worst-score slice — the stress factor is
        # UNDEFINED, not zero. Returning 0 would multiply predicted risk to zero
        # downstream (maximally anti-conservative). Fall back to neutral 1.0.
        logger.warning("Worst-slice denominator is 0; returning neutral stress factor 1.0")
        return NEUTRAL_STRESS_FACTOR
    if worst_num <= 0:
        # Exposure but no defaults in the worst-score slice: the tail rate is not measured as
        # zero risk, it is unmeasured. A factor of 0 here multiplied swap-in risk to nothing.
        logger.warning("Worst-slice has no defaults; returning neutral stress factor 1.0")
        return NEUTRAL_STRESS_FACTOR
    worst_bad_rate = worst_num / worst_den * DEFAULT_RISK_MULTIPLIER

    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        logger.warning("Overall bad rate is 0; returning neutral stress factor 1.0")
        stress_factor = NEUTRAL_STRESS_FACTOR

    if stress_factor < NEUTRAL_STRESS_FACTOR:
        logger.warning(
            f"Worst-slice risk is below the overall risk (ratio {stress_factor:.3f}); a stress factor never "
            "reduces risk — returning neutral 1.0."
        )
        return NEUTRAL_STRESS_FACTOR
    return float(stress_factor)


# Defaults of the per-bin shrinkage (config: stress_per_bin_prior_strength / stress_per_bin_max_factor).
DEFAULT_PER_BIN_PRIOR_STRENGTH = 50.0
DEFAULT_PER_BIN_MAX_FACTOR = 5.0


def calculate_per_bin_stress_factors(
    df: pd.DataFrame,
    variables: list[str],
    status_col: str = "status_name",
    score_col: str = "risk_score_rf",
    num_col: str = "todu_30ever_h6",
    den_col: str = "todu_amt_pile_h6",
    frac: float = 0.05,
    target_status: str = "booked",
    higher_is_worse: bool = False,
    min_obs_per_bin: int = 20,
    prior_strength: float = DEFAULT_PER_BIN_PRIOR_STRENGTH,
    max_factor: float = DEFAULT_PER_BIN_MAX_FACTOR,
) -> pd.DataFrame:
    """Compute per-bin stress factors, shrunk toward the global factor.

    A bin's OWN ratio is the risk rate of its worst ``frac`` fraction (by score) over the bin's
    overall rate. That slice is a handful of loans, so the ratio is mostly noise: most bins have
    no default in it (ratio 0) and one default in a one-loan slice gives ~30x (audit 2026-11-27
    A32). The factor is therefore a weighted mean of the bin's ratio and the global factor::

        factor = w * min(own_ratio, max_factor) + (1 - w) * global,   w = n / (n + prior_strength)

    with ``n`` the number of loans in the bin's worst slice — the same shape as the acceptance-rate
    smoothing in reject inference. ``max_factor`` caps the noisy component only (the global factor
    passes through), and no factor goes below neutral (1.0). ``prior_strength = 0`` uses the
    bin's own ratio.

    Bins with no usable ratio — fewer than ``min_obs_per_bin`` records, no exposure in the bin or
    in its worst slice, or no default in the whole bin — take the global factor.

    Returns
    -------
    DataFrame with ``[*variables, "stress_factor"]``.
    """
    df_target = df[df[status_col] == target_status].copy()
    if df_target.empty:
        logger.warning("No target records for per-bin stress; returning empty DataFrame")
        return pd.DataFrame(columns=variables + ["stress_factor"])

    global_stress = calculate_stress_factor(
        df,
        status_col=status_col,
        score_col=score_col,
        num_col=num_col,
        den_col=den_col,
        frac=frac,
        target_status=target_status,
        higher_is_worse=higher_is_worse,
    )

    rows: list[dict] = []
    weights: list[float] = []
    for bin_vals, group in df_target.groupby(variables, observed=True):
        if not isinstance(bin_vals, tuple):
            bin_vals = (bin_vals,)
        row = dict(zip(variables, bin_vals))
        row["stress_factor"] = global_stress
        rows.append(row)

        total_den = group[den_col].sum()
        if len(group) < min_obs_per_bin or total_den <= 0:
            continue
        overall_rate = DEFAULT_RISK_MULTIPLIER * group[num_col].sum() / total_den
        if overall_rate <= 0:
            continue  # no default anywhere in the bin: a tail ratio does not exist

        if higher_is_worse:
            worst = group[group[score_col] >= group[score_col].quantile(1.0 - frac)]
        else:
            worst = group[group[score_col] <= group[score_col].quantile(frac)]
        worst_den = worst[den_col].sum()
        if worst_den <= 0:
            continue  # #54: an undefined worst-slice rate, not a zero one

        # A slice with exposure and no defaults has own ratio 0 here. Unlike the unshrunk
        # version (where 0 zeroed the bin's swap-in risk, A20), that is information: it pulls
        # the bin a little below the global factor, by the weight its few loans deserve.
        own_ratio = min(DEFAULT_RISK_MULTIPLIER * worst[num_col].sum() / worst_den / overall_rate, max_factor)
        n_tail = len(worst)
        weight = n_tail / (n_tail + prior_strength) if (n_tail + prior_strength) > 0 else 0.0
        row["stress_factor"] = max(weight * own_ratio + (1.0 - weight) * global_stress, NEUTRAL_STRESS_FACTOR)
        weights.append(weight)

    result = pd.DataFrame(rows)
    mean_weight = sum(weights) / len(weights) if weights else 0.0
    logger.info(
        f"Per-bin stress factors: {len(result)} bins, {len(weights)} with their own ratio "
        f"(mean weight {mean_weight:.2f} vs the global factor {global_stress:.3f}; prior strength "
        f"{prior_strength:g}, own-ratio cap {max_factor:g}) | mean={result['stress_factor'].mean():.3f} | "
        f"min={result['stress_factor'].min():.3f} | max={result['stress_factor'].max():.3f}"
    )
    return result
