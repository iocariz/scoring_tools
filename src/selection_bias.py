"""
Selection bias analysis for credit risk score discriminance.

When the acceptance policy uses the same scores being evaluated, the booked
(accepted) population has a truncated score distribution.  This range
restriction mechanically depresses observed Gini/AUROC/KS below the scores'
true discriminative power.

This module provides statistical tools to quantify and demonstrate the effect:

1. **Distribution truncation** — variance ratio between demand and booked.
2. **Thorndike Case 2 correction** — theoretical Gini recovery.
3. **Simulated range restriction** — progressive truncation sensitivity.
4. **Temporal acceptance-rate vs Gini** — natural experiment.
5. **Rejection profile by decile** — where rejected mass falls.
6. **Cross-segment acceptance vs Gini** — supplementary scatter.
"""

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from src.constants import RejectReason, StatusName
from src.metrics import compute_metrics
from src.stability import calculate_psi


# ---------------------------------------------------------------------------
# Analysis 1: Distribution truncation
# ---------------------------------------------------------------------------
def compute_distribution_truncation(
    df_demand: pd.DataFrame,
    score_col: str,
    *,
    score_negate: bool = False,
) -> dict:
    """Compute statistics comparing demand vs booked score distributions.

    Parameters
    ----------
    df_demand:
        Full demand population (booked + rejected), already filtered to
        the segment and base quality criteria.
    score_col:
        Raw score column name (e.g. ``"score_rf"``).
    score_negate:
        If True, negate the score before analysis (higher = riskier).

    Returns
    -------
    Dictionary with keys:
        n_demand, n_booked, n_rejected, acceptance_rate,
        mean_demand, mean_booked, mean_rejected,
        sd_demand, sd_booked, sd_rejected,
        u_ratio (SD_booked / SD_demand),
        iqr_demand, iqr_booked,
        ks_demand_vs_booked (2-sample KS statistic + p-value).
    """
    sign = -1 if score_negate else 1

    demand = (sign * df_demand[score_col].dropna()).values
    booked_mask = df_demand["status_name"] == StatusName.BOOKED.value
    booked = (sign * df_demand.loc[booked_mask, score_col].dropna()).values
    rejected = (sign * df_demand.loc[~booked_mask, score_col].dropna()).values

    n_demand = len(demand)
    n_booked = len(booked)
    n_rejected = len(rejected)

    sd_demand = float(np.std(demand, ddof=1)) if n_demand > 1 else 0.0
    sd_booked = float(np.std(booked, ddof=1)) if n_booked > 1 else 0.0
    sd_rejected = float(np.std(rejected, ddof=1)) if n_rejected > 1 else 0.0

    u_ratio = sd_booked / sd_demand if sd_demand > 0 else 1.0

    ks_stat, ks_pval = stats.ks_2samp(demand, booked) if n_booked > 0 and n_demand > 0 else (0.0, 1.0)

    def _iqr(arr):
        if len(arr) < 4:
            return 0.0
        return float(np.percentile(arr, 75) - np.percentile(arr, 25))

    return {
        "n_demand": n_demand,
        "n_booked": n_booked,
        "n_rejected": n_rejected,
        "acceptance_rate": round(n_booked / n_demand, 4) if n_demand > 0 else 0.0,
        "mean_demand": round(float(np.mean(demand)), 4) if n_demand > 0 else 0.0,
        "mean_booked": round(float(np.mean(booked)), 4) if n_booked > 0 else 0.0,
        "mean_rejected": round(float(np.mean(rejected)), 4) if n_rejected > 0 else 0.0,
        "sd_demand": round(sd_demand, 4),
        "sd_booked": round(sd_booked, 4),
        "sd_rejected": round(sd_rejected, 4),
        "u_ratio": round(u_ratio, 4),
        "iqr_demand": round(_iqr(demand), 4),
        "iqr_booked": round(_iqr(booked), 4),
        "ks_demand_vs_booked": round(ks_stat, 4),
        "ks_pvalue": round(ks_pval, 6),
    }


# ---------------------------------------------------------------------------
# Analysis 2: Thorndike Case 2 correction
# ---------------------------------------------------------------------------
def _auc_to_r(auc: float) -> float:
    """Convert AUC to point-biserial correlation via the normal-ogive model.

    d' = sqrt(2) * Phi^{-1}(AUC), then r = d' / sqrt(1 + d'^2).
    """
    auc = np.clip(auc, 0.501, 0.999)
    d_prime = np.sqrt(2) * stats.norm.ppf(auc)
    return d_prime / np.sqrt(1 + d_prime**2)


def _r_to_auc(r: float) -> float:
    """Convert point-biserial correlation back to AUC."""
    r = np.clip(r, 0.001, 0.999)
    d_prime = r / np.sqrt(1 - r**2)
    return float(stats.norm.cdf(d_prime / np.sqrt(2)))


def thorndike_case2_correction(
    observed_gini: float,
    u_ratio: float,
) -> dict:
    """Estimate unrestricted Gini using Thorndike Case 2 formula.

    Case 2 assumes *direct* selection on the predictor itself:

        r_corrected = r_observed / (u * sqrt(1 + r_observed^2 * (1/u^2 - 1)))

    where u = SD_restricted / SD_unrestricted.

    Parameters
    ----------
    observed_gini:
        Gini measured on the selected (booked) population.
    u_ratio:
        SD_booked / SD_demand.

    Returns
    -------
    Dictionary with corrected_gini, attenuation_pct, and intermediates.
    """
    if u_ratio >= 1.0 or u_ratio <= 0.0:
        return {
            "observed_gini": round(observed_gini, 4),
            "u_ratio": round(u_ratio, 4),
            "corrected_gini": round(observed_gini, 4),
            "attenuation_pct": 0.0,
            "r_observed": 0.0,
            "r_corrected": 0.0,
        }

    observed_auc = (observed_gini + 1) / 2
    r_obs = _auc_to_r(observed_auc)

    # Thorndike Case 2
    denom = u_ratio * np.sqrt(1 + r_obs**2 * (1 / u_ratio**2 - 1))
    r_corr = r_obs / denom if denom > 0 else r_obs
    r_corr = np.clip(r_corr, 0.0, 0.999)

    corrected_auc = _r_to_auc(r_corr)
    corrected_gini = 2 * corrected_auc - 1

    # A Case-2 range-restriction correction (u<1) can only raise the estimate; the 0.999 r-clip
    # above can otherwise invert that and yield corrected < observed -> a nonsensical negative
    # attenuation. Clamp so corrected is never below observed (attenuation_pct is then >= 0).
    corrected_gini = max(corrected_gini, observed_gini)

    attenuation = (1 - observed_gini / corrected_gini) * 100 if corrected_gini > 0 else 0.0

    return {
        "observed_gini": round(observed_gini, 4),
        "u_ratio": round(u_ratio, 4),
        "corrected_gini": round(corrected_gini, 4),
        "attenuation_pct": round(attenuation, 1),
        "r_observed": round(r_obs, 4),
        "r_corrected": round(r_corr, 4),
    }


# ---------------------------------------------------------------------------
# Analysis 3: Simulated range restriction
# ---------------------------------------------------------------------------
def simulate_range_restriction(
    df_booked: pd.DataFrame,
    score_columns: dict[str, dict],
    target_col: str,
    pct_steps: list[float] | None = None,
) -> pd.DataFrame:
    """Progressively truncate the booked population and measure Gini decay.

    At each step, the riskiest tail (per the score's risk orientation, i.e.
    after applying ``negate``) is removed, simulating a tighter acceptance
    cutoff: the retained population is the safest ``pct`` of the book.

    Parameters
    ----------
    df_booked:
        Booked population with observed outcomes.
    score_columns:
        Score config dict (e.g. ``{"Score RF": {"column": "score_rf", "negate": True}}``).
    target_col:
        Binary target column name.
    pct_steps:
        Fractions of the booked population to retain (descending).
        Default: [1.0, 0.95, 0.90, ..., 0.50].

    Returns
    -------
    DataFrame with columns: [pct_retained, score, gini, n_records, n_bads].
    """
    if pct_steps is None:
        pct_steps = [round(1.0 - i * 0.05, 2) for i in range(11)]  # 1.0 → 0.50

    required = [target_col] + [s["column"] for s in score_columns.values()]
    df = df_booked.dropna(subset=required).copy()

    if df.empty or df[target_col].nunique() < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for score_name, info in score_columns.items():
        col = info["column"]
        sign = -1 if info.get("negate") else 1
        scores_raw = df[col].values

        # sign*score is higher = riskier (module convention), so an ASCENDING
        # sort puts the safest first; keeping [:n_keep] removes the riskiest
        # tail — a tighter cutoff. (Audit #23: the old descending sort kept the
        # riskiest pct and removed the safest — the inverted restriction.)
        order = np.argsort(sign * scores_raw)
        sorted_target = df[target_col].values[order]
        sorted_scores = (sign * scores_raw)[order]

        n_total = len(sorted_target)
        for pct in pct_steps:
            n_keep = max(int(n_total * pct), 10)
            y_sub = sorted_target[:n_keep]
            s_sub = sorted_scores[:n_keep]

            if len(np.unique(y_sub)) < 2:
                continue

            try:
                gini, _, _, _ = compute_metrics(y_sub, s_sub)
            except ValueError:
                continue

            rows.append(
                {
                    "pct_retained": pct,
                    "score": score_name,
                    "gini": round(gini, 4),
                    "n_records": n_keep,
                    "n_bads": int(y_sub.sum()),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 4: Rolling temporal acceptance rate vs Gini
# ---------------------------------------------------------------------------
def compute_rolling_acceptance_rate(
    df_demand: pd.DataFrame,
    date_ini: str,
    date_fin: str,
    window_months: int = 3,
) -> pd.DataFrame:
    """Compute acceptance rate in a rolling window of months.

    Mirrors the rolling Gini windows in ``run_score_metrics.py`` so the
    two can be joined for a natural-experiment scatter.

    Parameters
    ----------
    df_demand:
        Full demand (booked + rejected) for a single segment, already
        base-filtered.
    date_ini, date_fin:
        Period date bounds.
    window_months:
        Width of the rolling window (default 3, matching rolling Gini).

    Returns
    -------
    DataFrame with columns: [month, n_demand, n_booked, acceptance_rate].
    """
    if not date_ini or not date_fin:
        return pd.DataFrame()

    df = df_demand.copy()
    df["mis_date"] = pd.to_datetime(df["mis_date"])
    start = pd.to_datetime(date_ini)
    end = pd.to_datetime(date_fin)
    df = df[(df["mis_date"] >= start) & (df["mis_date"] <= end)]

    if df.empty:
        return pd.DataFrame()

    df["year_month"] = df["mis_date"].dt.to_period("M")
    months_sorted = sorted(df["year_month"].unique())

    if len(months_sorted) < window_months:
        return pd.DataFrame()

    rows: list[dict] = []
    for i in range(len(months_sorted) - window_months + 1):
        window = months_sorted[i : i + window_months]
        window_end = window[-1]
        df_win = df[df["year_month"].isin(window)]

        n_demand = len(df_win)
        n_booked = int((df_win["status_name"] == StatusName.BOOKED.value).sum())

        rows.append(
            {
                "month": str(window_end),
                "n_demand": n_demand,
                "n_booked": n_booked,
                "acceptance_rate": round(n_booked / n_demand, 4) if n_demand > 0 else 0.0,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 5: Rejection profile by score decile
# ---------------------------------------------------------------------------
def compute_rejection_profile(
    df_demand: pd.DataFrame,
    score_col: str,
    *,
    score_negate: bool = False,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute booked vs rejected counts by score decile on full demand.

    Parameters
    ----------
    df_demand:
        Full demand population for a segment.
    score_col:
        Raw score column.
    score_negate:
        Negate so that higher = riskier.
    n_bins:
        Number of quantile bins (default 10 = deciles).

    Returns
    -------
    DataFrame with columns:
        [decile, score_min, score_max, n_booked, n_rejected, n_total,
         rejection_rate, pct_of_total_rejections].
    """
    df = df_demand.dropna(subset=[score_col]).copy()
    if df.empty:
        return pd.DataFrame()

    sign = -1 if score_negate else 1
    df["_score"] = sign * df[score_col]

    # Create decile bins from full demand
    try:
        df["_decile"] = pd.qcut(df["_score"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        logger.warning(f"Could not create {n_bins} quantile bins for {score_col}")
        return pd.DataFrame()

    agg = (
        df.groupby("_decile")
        .agg(
            score_min=("_score", "min"),
            score_max=("_score", "max"),
            n_booked=("status_name", lambda s: (s == StatusName.BOOKED.value).sum()),
            n_total=("status_name", "count"),
        )
        .reset_index()
        .rename(columns={"_decile": "decile"})
    )

    agg["n_rejected"] = agg["n_total"] - agg["n_booked"]
    agg["rejection_rate"] = (agg["n_rejected"] / agg["n_total"]).round(4)

    total_rejected = agg["n_rejected"].sum()
    agg["pct_of_total_rejections"] = (agg["n_rejected"] / total_rejected).round(4) if total_rejected > 0 else 0.0

    return agg[
        [
            "decile",
            "score_min",
            "score_max",
            "n_booked",
            "n_rejected",
            "n_total",
            "rejection_rate",
            "pct_of_total_rejections",
        ]
    ]


# ---------------------------------------------------------------------------
# Analysis 6: Cross-segment acceptance rate vs Gini
# ---------------------------------------------------------------------------
def compute_acceptance_gini_correlation(
    segment_data: list[dict],
) -> dict:
    """Regress observed Gini on acceptance rate across segments.

    Parameters
    ----------
    segment_data:
        List of dicts with keys ``segment``, ``score``, ``acceptance_rate``, ``gini``.

    Returns
    -------
    Dictionary with: slope, intercept, r_value, p_value, n_segments.
    Returns None if fewer than 3 data points.
    """
    if len(segment_data) < 3:
        return {"slope": None, "intercept": None, "r_value": None, "p_value": None, "n_segments": len(segment_data)}

    df = pd.DataFrame(segment_data)
    x = df["acceptance_rate"].values
    y = df["gini"].values

    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)

    return {
        "slope": round(slope, 4),
        "intercept": round(intercept, 4),
        "r_value": round(r_value, 4),
        "p_value": round(p_value, 4),
        "n_segments": len(segment_data),
    }


# ---------------------------------------------------------------------------
# Analysis 7: Booked-vs-Rejected discriminance (no target needed)
# ---------------------------------------------------------------------------
def compute_booked_vs_rejected_discriminance(
    df_demand: pd.DataFrame,
    score_columns: dict[str, dict],
) -> pd.DataFrame:
    """Compute Gini/AUROC/KS of each score for separating booked from rejected.

    Uses ``status_name`` as a binary target (booked=1, rejected=0) on the
    full demand population.  This does NOT require ``early_bad``.

    A high Gini here means the score is good at predicting who gets accepted.
    If a score has high booked-vs-rejected Gini but low early_bad Gini, that
    is the classic selection-bias signature.
    """
    df = df_demand.dropna(subset=[s["column"] for s in score_columns.values()]).copy()
    if df.empty:
        return pd.DataFrame()

    y_binary = (df["status_name"] == StatusName.BOOKED.value).astype(int).values
    if len(np.unique(y_binary)) < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for score_name, info in score_columns.items():
        # Use raw score (no negation): higher score → more likely booked
        scores = df[info["column"]].values
        try:
            gini, auroc, ks, _ = compute_metrics(y_binary, scores)
        except ValueError:
            continue
        rows.append(
            {
                "score": score_name,
                "target": "booked_vs_rejected",
                "auroc": round(auroc, 4),
                "gini": round(gini, 4),
                "ks": round(ks, 4),
                "n_booked": int(y_binary.sum()),
                "n_rejected": int(len(y_binary) - y_binary.sum()),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 8: Score-rejection-only truncation
# ---------------------------------------------------------------------------
def compute_score_rejection_only_truncation(
    df_demand: pd.DataFrame,
    score_columns: dict[str, dict],
) -> pd.DataFrame:
    """Recompute truncation stats using only score-based rejections.

    Filters demand to ``booked + (rejected AND reject_reason == "09-score")``,
    excluding non-score rejections ("08-other") that are orthogonal to the score.
    This gives a cleaner measure of score-based selection intensity.
    """
    is_booked = df_demand["status_name"] == StatusName.BOOKED.value
    is_score_rejected = (df_demand["status_name"] == StatusName.REJECTED.value) & (
        df_demand["reject_reason"] == RejectReason.SCORE.value
    )
    df_filtered = df_demand[is_booked | is_score_rejected].copy()

    if df_filtered.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for score_name, info in score_columns.items():
        trunc = compute_distribution_truncation(df_filtered, info["column"], score_negate=info.get("negate", False))
        trunc["score"] = score_name
        trunc["filter"] = "score_rejection_only"
        rows.append(trunc)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 9: Bad rate gradient by score decile (within booked)
# ---------------------------------------------------------------------------
def compute_bad_rate_by_decile(
    df_booked: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    score_negate: bool = False,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute bad rate (early_bad) by score decile within booked population.

    A flat curve means the score has no rank-ordering power.
    A steep gradient means the score discriminates even within the selected range.
    """
    df = df_booked.dropna(subset=[score_col, target_col]).copy()
    if df.empty:
        return pd.DataFrame()

    sign = -1 if score_negate else 1
    df["_score"] = sign * df[score_col]

    try:
        df["_decile"] = pd.qcut(df["_score"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        return pd.DataFrame()

    agg = (
        df.groupby("_decile")
        .agg(
            score_min=("_score", "min"),
            score_max=("_score", "max"),
            n_records=(target_col, "count"),
            n_bads=(target_col, "sum"),
        )
        .reset_index()
        .rename(columns={"_decile": "decile"})
    )

    agg["n_bads"] = agg["n_bads"].astype(int)
    agg["bad_rate"] = (agg["n_bads"] / agg["n_records"]).round(4)

    return agg


# ---------------------------------------------------------------------------
# Analysis 10: Score-to-score correlation
# ---------------------------------------------------------------------------
def compute_score_correlations(
    df_demand: pd.DataFrame,
    score_columns: dict[str, dict],
) -> pd.DataFrame:
    """Compute pairwise Pearson and Spearman correlations between scores.

    Computed on the full demand population (booked + rejected).
    High correlation means selection on one score automatically truncates the other.
    """
    cols = {name: info["column"] for name, info in score_columns.items()}
    df = df_demand[list(cols.values())].dropna()
    if df.empty:
        return pd.DataFrame()

    names = list(cols.keys())
    raw_cols = list(cols.values())
    rows: list[dict] = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x = df[raw_cols[i]].values
            y = df[raw_cols[j]].values
            pearson_r, pearson_p = stats.pearsonr(x, y)
            spearman_r, spearman_p = stats.spearmanr(x, y)
            rows.append(
                {
                    "score_1": names[i],
                    "score_2": names[j],
                    "pearson_r": round(pearson_r, 4),
                    "pearson_p": round(pearson_p, 6),
                    "spearman_r": round(spearman_r, 4),
                    "spearman_p": round(spearman_p, 6),
                    "n_records": len(x),
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Analysis 11: Reject inference Gini (bin-level imputation)
# ---------------------------------------------------------------------------
def compute_ri_gini(
    df_demand: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    score_negate: bool = False,
    n_bins: int = 10,
    uplift_factor: float = 1.5,
) -> dict:
    """Estimate full-population Gini by imputing outcomes for rejected applicants.

    For each score decile, the rejected bad rate is imputed as::

        imputed_bad_rate = min(booked_bad_rate * uplift_factor, 1.0)

    Binary outcomes are then sampled for rejected records using this rate,
    and Gini is computed on the combined (booked-observed + rejected-imputed)
    population.  Repeated 50 times to produce a mean and CI.

    This metric is **mechanically inflated** and must NOT be read as an unbiased discrimination
    estimate: the imputed bad rate is a monotone function of the same score the Gini then ranks by,
    so the score trivially "predicts" the imputed-rejected portion by construction. It is an upper
    bound *under that assumption* only (the returned dict carries a ``note`` saying so). For a
    principled unrestricted estimate use the Thorndike-corrected Gini instead.
    """
    rng = np.random.RandomState(42)

    df = df_demand.dropna(subset=[score_col]).copy()
    if df.empty:
        return {}

    sign = -1 if score_negate else 1
    df["_score"] = sign * df[score_col]

    try:
        df["_decile"] = pd.qcut(df["_score"], q=n_bins, labels=False, duplicates="drop") + 1
    except ValueError:
        return {}

    is_booked = df["status_name"] == StatusName.BOOKED.value
    df_booked = df[is_booked].dropna(subset=[target_col])
    df_rejected = df[~is_booked]

    if df_booked.empty or df_booked[target_col].nunique() < 2 or df_rejected.empty:
        return {}

    # Compute per-decile bad rate from booked
    bin_rates = df_booked.groupby("_decile")[target_col].mean().to_dict()

    # Impute and compute Gini multiple times
    gini_samples = []
    for _ in range(50):
        imputed_bad = np.zeros(len(df_rejected))
        for dec, rate in bin_rates.items():
            mask = df_rejected["_decile"].values == dec
            imputed_rate = min(rate * uplift_factor, 1.0)
            imputed_bad[mask] = rng.binomial(1, imputed_rate, mask.sum())

        y_full = np.concatenate([df_booked[target_col].values, imputed_bad])
        s_full = np.concatenate([df_booked["_score"].values, df_rejected["_score"].values])

        if len(np.unique(y_full)) < 2:
            continue
        try:
            gini, _, _, _ = compute_metrics(y_full, s_full)
            gini_samples.append(gini)
        except ValueError:
            continue

    if not gini_samples:
        return {}

    return {
        "ri_gini_mean": round(float(np.mean(gini_samples)), 4),
        "ri_gini_ci_lo": round(float(np.percentile(gini_samples, 2.5)), 4),
        "ri_gini_ci_hi": round(float(np.percentile(gini_samples, 97.5)), 4),
        "n_booked": len(df_booked),
        "n_rejected": len(df_rejected),
        "uplift_factor": uplift_factor,
        "note": (
            "circular: rejected bad outcomes are imputed as a monotone function of the same score "
            "the Gini ranks by — an upper bound under that assumption, NOT an unbiased discrimination "
            "estimate"
        ),
    }


# ---------------------------------------------------------------------------
# Analysis 12: Score PSI over monthly vintages (within booked)
# ---------------------------------------------------------------------------
def compute_monthly_psi(
    df_booked: pd.DataFrame,
    score_col: str,
    *,
    score_negate: bool = False,
    reference_months: int = 3,
) -> pd.DataFrame:
    """Compute PSI of a score's distribution across monthly vintages.

    The first *reference_months* months form the baseline distribution.
    PSI is then computed for each subsequent month against that baseline.

    High PSI indicates the score distribution is shifting (model drift),
    which is a separate explanation from selection bias for changing Gini.
    """
    df = df_booked.dropna(subset=[score_col]).copy()
    if df.empty:
        return pd.DataFrame()

    sign = -1 if score_negate else 1
    df["_score"] = sign * df[score_col]
    df["mis_date"] = pd.to_datetime(df["mis_date"])
    df["year_month"] = df["mis_date"].dt.to_period("M")

    months_sorted = sorted(df["year_month"].unique())
    if len(months_sorted) <= reference_months:
        return pd.DataFrame()

    # Baseline: first N months
    baseline_months = months_sorted[:reference_months]
    baseline_scores = df[df["year_month"].isin(baseline_months)]["_score"]

    rows: list[dict] = []
    for month in months_sorted[reference_months:]:
        comparison_scores = df[df["year_month"] == month]["_score"]
        if comparison_scores.empty:
            continue
        psi_val, _ = calculate_psi(baseline_scores, comparison_scores, bins=10)
        rows.append(
            {
                "month": str(month),
                "psi": round(psi_val, 4),
                "n_baseline": len(baseline_scores),
                "n_comparison": len(comparison_scores),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Master function: full selection bias report for one segment/period
# ---------------------------------------------------------------------------
def compute_selection_bias_report(
    df_demand: pd.DataFrame,
    df_booked: pd.DataFrame,
    score_columns: dict[str, dict],
    target_col: str,
    segment_name: str,
    period: str,
    date_ini: str | None = None,
    date_fin: str | None = None,
) -> dict:
    """Run all selection bias analyses for a single segment and period.

    Returns a dictionary with DataFrames/dicts for each analysis, keyed by
    analysis name.
    """
    results: dict = {
        "segment": segment_name,
        "period": period,
        "date_ini": date_ini,
        "date_fin": date_fin,
    }

    # 1. Distribution truncation per score
    truncation_rows = []
    for score_name, info in score_columns.items():
        trunc = compute_distribution_truncation(df_demand, info["column"], score_negate=info.get("negate", False))
        trunc["score"] = score_name
        trunc["segment"] = segment_name
        trunc["period"] = period
        truncation_rows.append(trunc)
    results["truncation"] = pd.DataFrame(truncation_rows)

    # 2. Thorndike correction (needs observed Gini from booked)
    correction_rows = []
    required_cols = [target_col] + [s["column"] for s in score_columns.values()]
    df_clean = df_booked.dropna(subset=required_cols)

    if not df_clean.empty and df_clean[target_col].nunique() >= 2:
        y_true = df_clean[target_col].values
        for score_name, info in score_columns.items():
            sign = -1 if info.get("negate") else 1
            scores = sign * df_clean[info["column"]].values
            try:
                gini, _, _, _ = compute_metrics(y_true, scores)
            except ValueError:
                continue

            u_ratio = truncation_rows[[r for r, row in enumerate(truncation_rows) if row["score"] == score_name][0]][
                "u_ratio"
            ]
            correction = thorndike_case2_correction(gini, u_ratio)
            correction["score"] = score_name
            correction["segment"] = segment_name
            correction["period"] = period
            correction_rows.append(correction)

    results["thorndike"] = pd.DataFrame(correction_rows)

    # 3. Simulated range restriction
    results["simulated_restriction"] = simulate_range_restriction(df_booked, score_columns, target_col)
    if not results["simulated_restriction"].empty:
        results["simulated_restriction"]["segment"] = segment_name
        results["simulated_restriction"]["period"] = period

    # 4. Rolling acceptance rate (for temporal analysis)
    results["rolling_acceptance"] = compute_rolling_acceptance_rate(df_demand, date_ini, date_fin)
    if not results["rolling_acceptance"].empty:
        results["rolling_acceptance"]["segment"] = segment_name
        results["rolling_acceptance"]["period"] = period

    # 5. Rejection profile per score
    rejection_profiles = {}
    for score_name, info in score_columns.items():
        profile = compute_rejection_profile(df_demand, info["column"], score_negate=info.get("negate", False))
        if not profile.empty:
            profile["score"] = score_name
            profile["segment"] = segment_name
            profile["period"] = period
        rejection_profiles[score_name] = profile
    results["rejection_profiles"] = rejection_profiles

    # 7. Booked-vs-Rejected discriminance
    bvr = compute_booked_vs_rejected_discriminance(df_demand, score_columns)
    if not bvr.empty:
        bvr["segment"] = segment_name
        bvr["period"] = period
    results["booked_vs_rejected"] = bvr

    # 8. Score-rejection-only truncation
    sro = compute_score_rejection_only_truncation(df_demand, score_columns)
    if not sro.empty:
        sro["segment"] = segment_name
        sro["period"] = period
    results["score_rejection_truncation"] = sro

    # 9. Bad rate gradient by score decile
    bad_rate_profiles = {}
    for score_name, info in score_columns.items():
        br = compute_bad_rate_by_decile(df_booked, info["column"], target_col, score_negate=info.get("negate", False))
        if not br.empty:
            br["score"] = score_name
            br["segment"] = segment_name
            br["period"] = period
        bad_rate_profiles[score_name] = br
    results["bad_rate_profiles"] = bad_rate_profiles

    # 10. Score-to-score correlation
    corr = compute_score_correlations(df_demand, score_columns)
    if not corr.empty:
        corr["segment"] = segment_name
        corr["period"] = period
    results["score_correlations"] = corr

    # 11. Reject inference Gini
    ri_rows = []
    for score_name, info in score_columns.items():
        ri = compute_ri_gini(df_demand, info["column"], target_col, score_negate=info.get("negate", False))
        if ri:
            ri["score"] = score_name
            ri["segment"] = segment_name
            ri["period"] = period
            ri_rows.append(ri)
    results["ri_gini"] = pd.DataFrame(ri_rows)

    # 12. Monthly PSI (score stability)
    psi_profiles = {}
    for score_name, info in score_columns.items():
        psi_df = compute_monthly_psi(df_booked, info["column"], score_negate=info.get("negate", False))
        if not psi_df.empty:
            psi_df["score"] = score_name
            psi_df["segment"] = segment_name
            psi_df["period"] = period
        psi_profiles[score_name] = psi_df
    results["monthly_psi"] = psi_profiles

    return results
