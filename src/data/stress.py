"""Stress factors (global and per-bin) applied to the booked risk during preprocessing."""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from src.core.constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RISK_MULTIPLIER
from src.core.utils import (
    MAX_PARALLEL_JOBS,
    calculate_annual_coef,
    calculate_b2_ever_h6,
    calculate_todu_30ever_from_b2,
    mask_incomplete_outcome_pairs,
)


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
        return 1.0
    worst_bad_rate = worst_num / worst_den * DEFAULT_RISK_MULTIPLIER

    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        logger.warning("Overall bad rate is 0; returning neutral stress factor 1.0")
        stress_factor = 1.0

    return float(stress_factor)


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
) -> pd.DataFrame:
    """Compute per-bin stress factors.

    For each bin combination, the stress factor is the ratio of the risk rate
    in the worst ``frac`` fraction (by score) to the overall risk rate within
    that bin.  Bins with fewer than ``min_obs_per_bin`` records use the global
    stress factor as a fallback.

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
    for bin_vals, group in df_target.groupby(variables, observed=True):
        if not isinstance(bin_vals, tuple):
            bin_vals = (bin_vals,)
        row = dict(zip(variables, bin_vals))

        if len(group) < min_obs_per_bin:
            row["stress_factor"] = global_stress
            rows.append(row)
            continue

        total_den = group[den_col].sum()
        if total_den == 0:
            row["stress_factor"] = global_stress
            rows.append(row)
            continue

        overall_rate = DEFAULT_RISK_MULTIPLIER * group[num_col].sum() / total_den

        if higher_is_worse:
            cutoff = group[score_col].quantile(1.0 - frac)
            worst = group[group[score_col] >= cutoff]
        else:
            cutoff = group[score_col].quantile(frac)
            worst = group[group[score_col] <= cutoff]

        worst_den = worst[den_col].sum()
        if worst_den <= 0:
            # #54: undefined worst-slice rate → neutral global fallback, not a
            # 0 stress factor that would zero out this bin's predicted risk.
            row["stress_factor"] = global_stress
            rows.append(row)
            continue
        worst_rate = DEFAULT_RISK_MULTIPLIER * worst[num_col].sum() / worst_den

        row["stress_factor"] = worst_rate / overall_rate if overall_rate > 0 else global_stress
        rows.append(row)

    result = pd.DataFrame(rows)
    logger.debug(
        f"Per-bin stress factors: {len(result)} bins | "
        f"mean={result['stress_factor'].mean():.3f} | "
        f"min={result['stress_factor'].min():.3f} | "
        f"max={result['stress_factor'].max():.3f}"
    )
    return result
