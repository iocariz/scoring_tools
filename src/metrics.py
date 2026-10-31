"""
Statistical metrics and model evaluation functions for credit risk scoring.

This module provides functions for evaluating credit risk model performance:
- Classification metrics (Gini, AUC, KS statistic)
- Precision-Recall metrics
- Bootstrap confidence intervals
- Rejection threshold analysis
- Lift and cumulative gains analysis
- DeLong test for AUC comparison
- Comprehensive model summary reports

Key functions:
- compute_metrics: Calculate Gini, AUC, KS, and CAP curve
- compute_precision_recall: Calculate precision, recall, and average precision
- bootstrap_confidence_interval: Compute CI for Gini and KS via bootstrap
- calculate_lift_table: Decile-level lift and cumulative gains analysis
- model_summary: Generate comprehensive model performance report
- delong_test: Statistical test for comparing two AUC values
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RANDOM_STATE


def combined_score_cv(X: pd.DataFrame, y, n_splits: int = 5, random_state: int = DEFAULT_RANDOM_STATE) -> np.ndarray:
    """OUT-OF-FOLD logistic-regression combined score (audit #35).

    Unlike the raw scores it is compared against, the combined score involves
    FITTING — evaluating it in-sample is not like-for-like. Each row's combined
    score here comes from a model that never saw that row (StratifiedKFold;
    the scaler is fit on the train fold only). Rows in folds that cannot be
    fit (single-class train fold) come back NaN — callers mask them.
    """
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y)
    out = np.full(len(Xv), np.nan)
    n_splits = min(n_splits, max(2, int(min(np.sum(yv == 1), np.sum(yv == 0)))))
    if n_splits < 2:
        return out
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in skf.split(Xv, yv):
        if len(np.unique(yv[tr])) < 2:
            continue
        scaler = StandardScaler().fit(Xv[tr])
        model = LogisticRegression().fit(scaler.transform(Xv[tr]), yv[tr])
        out[te] = model.decision_function(scaler.transform(Xv[te]))
    return out


def _warn_if_many_bootstrap_skips(n_kept: int, n_iterations: int, label: str) -> None:
    """Surface how many bootstrap resamples were single-class and dropped.

    A CI computed on far fewer than ``n_iterations`` resamples (tiny / heavily
    imbalanced population) is unreliable — silently returning it reads as a solid
    interval. Warn when >20% were skipped; info otherwise.
    """
    n_skipped = n_iterations - n_kept
    if n_skipped <= 0:
        return
    msg = (
        f"Bootstrap CI ({label}): {n_skipped}/{n_iterations} resamples were single-class and "
        f"dropped; the interval rests on {n_kept} resample(s)."
    )
    (logger.warning if n_kept < 0.8 * n_iterations else logger.info)(msg)


def _bootstrap_ci_combined(
    X: pd.DataFrame,
    y,
    n_iterations: int = DEFAULT_N_BOOTSTRAPS,
    alpha: float = 0.05,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Bootstrap CI for the combined score, REFITTING per resample (audit #35).

    The plain bootstrap held the fitted coefficients fixed and resampled scored
    rows, ignoring coefficient-estimation uncertainty — too narrow for a fitted
    score. Each replicate refits scaler+logistic on the resampled rows and
    evaluates Gini/KS on the OUT-OF-BAG rows (those not drawn), so the
    replicate metric is itself out-of-sample.

    Basis caveat: this CI is deliberately CONSERVATIVE and is not a bracket around
    the ``combined_score_cv`` point estimate. Each replicate trains on the in-bag
    sample (~63% unique rows) and scores on OOB (~37%), whereas the point estimate
    is K-fold OOF (trains on ~(k-1)/k of the data, scores every row once). The
    smaller training set + noisier evaluation bias the replicate metrics DOWN, so
    the interval typically sits below the point estimate. Read it as a
    lower-leaning uncertainty band, not a symmetric CI.
    """
    Xv = np.asarray(X, dtype=float)
    yv = np.asarray(y)
    rng = np.random.RandomState(random_state)
    n = len(yv)
    ginis, kss = [], []
    for _ in range(n_iterations):
        idx = rng.choice(n, n, replace=True)
        oob = np.setdiff1d(np.arange(n), idx, assume_unique=False)
        if len(np.unique(yv[idx])) < 2 or len(oob) == 0 or len(np.unique(yv[oob])) < 2:
            continue
        scaler = StandardScaler().fit(Xv[idx])
        model = LogisticRegression().fit(scaler.transform(Xv[idx]), yv[idx])
        scores_oob = model.decision_function(scaler.transform(Xv[oob]))
        gini, _, ks, _ = compute_metrics(yv[oob], scores_oob)
        ginis.append(gini)
        kss.append(ks)
    if not ginis:
        return (np.nan, np.nan), (np.nan, np.nan)
    _warn_if_many_bootstrap_skips(len(ginis), n_iterations, "combined-score OOB")
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return (
        (float(np.percentile(ginis, lo)), float(np.percentile(ginis, hi))),
        (float(np.percentile(kss, lo)), float(np.percentile(kss, hi))),
    )


def ks_statistic(y_true, y_scores):
    """Calculate KS statistic."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return max(tpr - fpr)


def compute_metrics(y_true, scores):
    """Compute GINI, AUC, and KS, and CAP Curve"""
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    gini = 2 * roc_auc - 1
    ks = float(max(tpr - fpr))

    # CAP Curve Calculation
    y_true_array = np.array(y_true)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_true_values = y_true_array[sorted_indices]
    cumulative_true_positive = np.cumsum(sorted_true_values)

    return gini, roc_auc, ks, cumulative_true_positive


def bootstrap_confidence_interval(
    y_true, y_scores, n_iterations=DEFAULT_N_BOOTSTRAPS, alpha=0.05, random_state=DEFAULT_RANDOM_STATE
):
    """Compute bootstrap confidence interval for Gini and KS."""
    if random_state is None:
        random_state = DEFAULT_RANDOM_STATE
    rng = np.random.RandomState(random_state)

    gini_scores = []
    ks_scores = []

    for _ in range(n_iterations):
        sample_indices = rng.choice(len(y_true), len(y_true), replace=True)

        sampled_y_true = y_true.iloc[sample_indices].values
        sampled_y_scores = y_scores.iloc[sample_indices].values

        # Skip degenerate resamples that contain only one class
        if len(np.unique(sampled_y_true)) < 2:
            continue

        # Compute GINI
        fpr, tpr, _ = roc_curve(sampled_y_true, sampled_y_scores)
        auc_score = auc(fpr, tpr)
        gini_scores.append(2 * auc_score - 1)

        # Compute KS
        ks_scores.append(ks_statistic(sampled_y_true, sampled_y_scores))

    if not gini_scores:
        # Every resample was single-class (tiny / imbalanced population): the CI is
        # UNDEFINED. Return NaN, not (0.0, 0.0) — the latter reads downstream as a
        # confident zero-width CI (Gini is precisely 0), which is not what happened.
        nan_ci = (float("nan"), float("nan"))
        return nan_ci, nan_ci

    _warn_if_many_bootstrap_skips(len(gini_scores), n_iterations, "Gini/KS")
    gini_ci = (np.percentile(gini_scores, 100 * alpha / 2.0), np.percentile(gini_scores, 100 * (1 - alpha / 2.0)))
    ks_ci = (np.percentile(ks_scores, 100 * alpha / 2.0), np.percentile(ks_scores, 100 * (1 - alpha / 2.0)))

    return gini_ci, ks_ci


def calculate_rejection_thresholds(y_true: np.ndarray, scores: np.ndarray, thresholds: list | None = None) -> dict:
    """
    Calculate the percentage of bad accounts captured at various rejection thresholds.

    For each threshold percentage, calculates what proportion of total bad accounts
    would be captured if rejecting that percentage of applications (sorted by score).

    Args:
        y_true: Binary array of true outcomes (1=bad, 0=good).
        scores: Model scores (higher = higher risk).
        thresholds: List of rejection percentages to evaluate.

    Returns:
        Dictionary mapping threshold strings (e.g., "5%") to capture percentages.
        Example: {"5%": 25.3, "10%": 42.1} means rejecting top 5% captures 25.3% of bads.
    """
    if thresholds is None:
        thresholds = [5, 10, 15, 20]
    y_true_array = np.array(y_true)
    total_bad = np.sum(y_true_array)

    if total_bad == 0:
        return {f"{t}%": 0.0 for t in thresholds}

    sorted_indices = np.argsort(scores)[::-1]
    sorted_true_values = y_true_array[sorted_indices]

    results = {}
    for threshold in thresholds:
        cutoff_index = int(len(scores) * (threshold / 100))
        bad_accounts = np.sum(sorted_true_values[:cutoff_index])
        results[f"{threshold}%"] = (bad_accounts / total_bad) * 100

    return results


def _nan_psi_frame() -> pd.DataFrame:
    """A single-row PSI result whose PSI is NaN (undefined) — never 0 (false "stable")."""
    return pd.DataFrame(
        {
            "Bucket": [1],
            "Breakpoint Value": [np.inf],
            "Expected Count": [0],
            "Actual Count": [0],
            "Expected Percent": [0.0],
            "Actual Percent": [0.0],
            "PSI": [float("nan")],
        }
    )


def model_summary(
    df: pd.DataFrame,
    target_column: str,
    score_columns: dict,
    combined_columns: dict | None = None,
    plot: bool = True,
    n_iterations: int = DEFAULT_N_BOOTSTRAPS,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Generate comprehensive model performance summary with metrics and visualizations.

    Evaluates multiple scoring models on the same dataset, computing Gini, AUC, KS,
    confidence intervals, and rejection threshold analysis. Optionally creates
    combined scores using logistic regression.

    Args:
        df: DataFrame containing target and score columns.
        target_column: Name of the binary target column (1=bad, 0=good).
        score_columns: Dictionary of score configurations. Format:
            {"Model Name": {"column": "score_col", "negate": False}}
            Set negate=True if lower scores indicate higher risk.
        combined_columns: Optional dictionary for creating combined scores via
            logistic regression. Format: {"Combined Name": ["col1", "col2"]}
        plot: Whether to display visualization dashboard.
        n_iterations: Number of bootstrap iterations for confidence intervals.
        alpha: Significance level for confidence intervals (default 0.05 = 95% CI).

    Returns:
        DataFrame with columns: Model, Gini Score, AUC, KS Value, Gini CI, KS CI,
        and rejection thresholds at 5%, 10%, 15%, 20%.

    Raises:
        ValueError: If specified columns not found in DataFrame.
    """
    # Local imports to avoid circular dependencies
    from .plots import plot_gini_confidence_intervals, visualize_metrics

    y_true = df[target_column]

    # Check if required columns exist in dataframe
    for info in score_columns.values():
        if info["column"] not in df.columns:
            raise ValueError(f"Column '{info['column']}' not found in dataframe.")

    # Prepare Data
    scores_dict = {
        name: (-1 if info["negate"] else 1) * df[info["column"]].values for name, info in score_columns.items()
    }

    combined_names = set()
    if combined_columns:
        for name, columns in combined_columns.items():
            # OUT-OF-FOLD combined score (audit #35): the in-sample fit inflated
            # the combined model's metrics vs the unfitted raw scores.
            scores_dict[name] = combined_score_cv(df[list(columns)], y_true)
            combined_names.add(name)

    # Compute Metrics and Rejection Thresholds
    metrics_data = {
        "Model": [],
        "Gini Score": [],
        "AUC": [],
        "KS Value": [],
        "Gini CI": [],
        "KS CI": [],
        "5% Rejection": [],
        "10% Rejection": [],
        "15% Rejection": [],
        "20% Rejection": [],
    }

    for name, scores in scores_dict.items():
        scores = np.asarray(scores, dtype=float)
        valid = ~np.isnan(scores)
        y_eval = y_true[valid] if not valid.all() else y_true
        s_eval = scores[valid] if not valid.all() else scores
        gini, roc_auc, ks, _ = compute_metrics(y_eval, s_eval)
        if name in combined_names:
            # Refit per resample (audit #35): the plain bootstrap held the fitted
            # coefficients fixed — too narrow for a fitted score.
            gini_ci, ks_ci = _bootstrap_ci_combined(df[list(combined_columns[name])], y_true, n_iterations, alpha)
        else:
            gini_ci, ks_ci = bootstrap_confidence_interval(
                pd.Series(np.asarray(y_eval)), pd.Series(s_eval), n_iterations, alpha
            )
        rejection_thresholds = calculate_rejection_thresholds(y_eval, s_eval)

        metrics_data["Model"].append(name)
        metrics_data["Gini Score"].append(gini)
        metrics_data["AUC"].append(roc_auc)
        metrics_data["KS Value"].append(ks)
        metrics_data["Gini CI"].append(gini_ci)
        metrics_data["KS CI"].append(ks_ci)
        metrics_data["5% Rejection"].append(rejection_thresholds["5%"])
        metrics_data["10% Rejection"].append(rejection_thresholds["10%"])
        metrics_data["15% Rejection"].append(rejection_thresholds["15%"])
        metrics_data["20% Rejection"].append(rejection_thresholds["20%"])

    # Format and Return
    summary_df = pd.DataFrame(metrics_data)
    summary_df["Gini CI"] = summary_df["Gini CI"].apply(lambda x: (round(x[0], 4), round(x[1], 4)))
    summary_df["KS CI"] = summary_df["KS CI"].apply(lambda x: (round(x[0], 4), round(x[1], 4)))

    # Visualization
    if plot:
        fig, axes = plt.subplots(1, 5, figsize=(25, 7))  # Adjust subplot dimensions as needed
        plt.subplots_adjust(wspace=0.4)

        visualize_metrics(y_true, scores_dict, ax=axes[:4])

        plot_gini_confidence_intervals(axes[4], summary_df)

        fig.suptitle("Model Performance Evaluation", fontsize=22, y=1.05)
        plt.tight_layout()
        plt.show()

    return summary_df


def compute_score_discriminance(
    df: pd.DataFrame,
    target_column: str,
    score_columns: dict[str, dict],
    combined_columns: dict[str, list] | None = None,
) -> pd.DataFrame:
    """
    Compute AUROC, Gini, and KS for each score on a given population.

    This is a lightweight alternative to ``model_summary`` — no plots, no
    bootstrap CIs, no rejection-threshold analysis.

    Args:
        df: DataFrame containing the target and score columns.
        target_column: Binary target column (1 = bad, 0 = good).
        score_columns: Score configurations, e.g.
            ``{"Score RF": {"column": "score_rf", "negate": True}}``.
        combined_columns: Optional logistic-regression combinations, e.g.
            ``{"Combined": ["score_rf", "risk_score_rf"]}``.

    Returns:
        DataFrame with columns:
        ``[score, auroc, gini, ks, n_records, n_bads, bad_rate]``.
    """
    y_true = df[target_column]

    scores_dict: dict[str, np.ndarray] = {}
    for name, info in score_columns.items():
        scores_dict[name] = (-1 if info.get("negate") else 1) * df[info["column"]].values

    if combined_columns:
        for name, columns in combined_columns.items():
            # OUT-OF-FOLD combined score (audit #35) — like-for-like with the raw scores.
            scores_dict[name] = combined_score_cv(df[list(columns)], y_true)

    rows = []
    for name, scores in scores_dict.items():
        scores = np.asarray(scores, dtype=float)
        valid = ~np.isnan(scores)
        # Count on the population the metric was ACTUALLY computed on. A combined (OOF) score
        # is NaN for rows in unfittable folds; labelling its gini with the full-population
        # n_records/n_bads overstates the sample the discriminance rests on.
        y_valid = y_true[valid] if not valid.all() else y_true
        s_valid = scores[valid] if not valid.all() else scores
        gini, roc_auc, ks, _ = compute_metrics(y_valid, s_valid)
        n_records = int(len(y_valid))
        n_bads = int(np.asarray(y_valid).sum())
        bad_rate = n_bads / n_records if n_records > 0 else 0.0
        rows.append(
            {
                "score": name,
                "auroc": round(roc_auc, 4),
                "gini": round(gini, 4),
                "ks": round(ks, 4),
                "n_records": n_records,
                "n_bads": n_bads,
                "bad_rate": round(bad_rate, 4),
            }
        )

    return pd.DataFrame(rows)


def compute_precision_recall(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute precision-recall curve and average precision.

    Useful for evaluating models on imbalanced datasets (low default rates)
    where ROC curves can be misleadingly optimistic.

    Args:
        y_true: Binary array of true outcomes (1=bad, 0=good).
        y_scores: Model scores (higher = higher risk of being class 1).

    Returns:
        Tuple of (precision, recall, thresholds, average_precision).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    return precision, recall, thresholds, ap


def calculate_lift_table(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Generate a decile-level lift and cumulative gains table.

    Sorts the population by score (descending risk), divides into quantile bins,
    and computes per-bin and cumulative statistics.

    Args:
        y_true: Binary array of true outcomes (1=bad, 0=good).
        y_scores: Model scores (higher = higher risk).
        n_bins: Number of quantile bins (default 10 for deciles).

    Returns:
        DataFrame with columns:
        - bin: Bin number (1 = highest risk).
        - n_records: Number of records in the bin.
        - n_bads: Number of bads (target=1) in the bin.
        - bad_rate: Bad rate within the bin.
        - pct_population: Percentage of total population in the bin.
        - pct_bads: Percentage of total bads captured by the bin.
        - cumulative_pct_bads: Cumulative percentage of bads captured.
        - cumulative_pct_population: Cumulative percentage of population.
        - lift: Ratio of bin bad rate to overall bad rate.
        - cumulative_lift: Cumulative lift up to and including this bin.
    """
    y_true_arr = np.asarray(y_true)
    y_scores_arr = np.asarray(y_scores)

    total_records = len(y_true_arr)
    total_bads = y_true_arr.sum()
    overall_bad_rate = total_bads / total_records if total_records > 0 else 0.0

    # Sort by descending score
    sorted_indices = np.argsort(y_scores_arr)[::-1]
    sorted_true = y_true_arr[sorted_indices]

    # Create bins (1 = highest risk)
    bin_edges = np.linspace(0, total_records, n_bins + 1, dtype=int)

    rows = []
    cumulative_bads = 0
    cumulative_records = 0

    for i in range(n_bins):
        start, end = bin_edges[i], bin_edges[i + 1]
        bin_true = sorted_true[start:end]
        n_records = len(bin_true)
        n_bads = int(bin_true.sum())
        bad_rate = n_bads / n_records if n_records > 0 else 0.0

        cumulative_bads += n_bads
        cumulative_records += n_records

        pct_population = n_records / total_records
        pct_bads = n_bads / total_bads if total_bads > 0 else 0.0
        cumulative_pct_bads = cumulative_bads / total_bads if total_bads > 0 else 0.0
        cumulative_pct_population = cumulative_records / total_records

        lift = bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0
        cumulative_bad_rate = cumulative_bads / cumulative_records if cumulative_records > 0 else 0.0
        cumulative_lift = cumulative_bad_rate / overall_bad_rate if overall_bad_rate > 0 else 0.0

        rows.append(
            {
                "bin": i + 1,
                "n_records": n_records,
                "n_bads": n_bads,
                "bad_rate": round(bad_rate, 4),
                "pct_population": round(pct_population, 4),
                "pct_bads": round(pct_bads, 4),
                "cumulative_pct_bads": round(cumulative_pct_bads, 4),
                "cumulative_pct_population": round(cumulative_pct_population, 4),
                "lift": round(lift, 4),
                "cumulative_lift": round(cumulative_lift, 4),
            }
        )

    return pd.DataFrame(rows)


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for the DeLong test.

    Uses scipy.stats.rankdata (C-level implementation) for performance
    instead of a Python while-loop over sorted indices.
    """
    return rankdata(x, method="average")


def _fast_delong(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Core DeLong computation for two paired AUC estimates.

    Implementation based on:
    Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm for Comparing
    the Areas Under Correlated Receiver Operating Characteristic Curves"

    Returns:
        Tuple of (auc1, auc2, covariance_matrix).
    """
    positive_mask = y_true == 1
    negative_mask = y_true == 0
    m = positive_mask.sum()  # number of positives
    n = negative_mask.sum()  # number of negatives

    if m < 1 or n < 1:
        return np.array([np.nan, np.nan]), np.full((2, 2), np.nan)

    aucs = []
    structural_components = []

    for scores in [scores1, scores2]:
        # Compute the structural components (placement values)
        # Following Sun & Xu (2014) exactly:
        #   V10[i] = (rank_combined[i] - rank_among_positives[i]) / n
        #   V01[j] = (rank_combined[m+j] - rank_among_negatives[j]) / m
        ordered = np.concatenate([scores[positive_mask], scores[negative_mask]])

        midranks = _compute_midrank(ordered)

        positive_ranks = midranks[:m]
        auc_val = (positive_ranks.sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc_val)

        # Structural components for variance estimation
        positive_ranks_within = _compute_midrank(ordered[:m])
        v_positive = (positive_ranks - positive_ranks_within) / n

        negative_ranks_combined = midranks[m:]
        negative_ranks_within = _compute_midrank(ordered[m:])
        v_negative = (negative_ranks_combined - negative_ranks_within) / m

        structural_components.append((v_positive, v_negative))

    # Compute 2x2 covariance matrix (ddof=1 for unbiased variance per DeLong et al. 1988)
    cov = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            s10 = np.cov(structural_components[i][0], structural_components[j][0], ddof=1)[0, 1] if m > 1 else 0.0
            s01 = np.cov(structural_components[i][1], structural_components[j][1], ddof=1)[0, 1] if n > 1 else 0.0
            cov[i, j] = s10 / m + s01 / n

    return np.array(aucs), cov


def delong_test(
    y_true: np.ndarray,
    scores1: np.ndarray,
    scores2: np.ndarray,
) -> dict[str, float]:
    """
    DeLong test for comparing two correlated AUC values.

    Tests the null hypothesis that two models have equal AUC on the same dataset.

    Based on:
    DeLong et al. (1988) "Comparing the Areas under Two or More Correlated
    Receiver Operating Characteristic Curves: A Nonparametric Approach"

    Args:
        y_true: Binary array of true outcomes (1=bad, 0=good).
        scores1: Predicted scores from model 1.
        scores2: Predicted scores from model 2.

    Returns:
        Dictionary with keys:
        - auc1: AUC for model 1.
        - auc2: AUC for model 2.
        - z_statistic: Z-score of the difference.
        - p_value: Two-sided p-value.
        - auc_diff: AUC1 - AUC2.
        - se_diff: Standard error of the difference.

    Example:
        >>> result = delong_test(y_true, model_a_scores, model_b_scores)
        >>> if result["p_value"] < 0.05:
        ...     print("Models have significantly different AUCs")
    """
    y_true_arr = np.asarray(y_true)
    scores1_arr = np.asarray(scores1, dtype=float)
    scores2_arr = np.asarray(scores2, dtype=float)

    aucs, cov = _fast_delong(y_true_arr, scores1_arr, scores2_arr)

    # Single-class subset: AUC is undefined, return non-significant result.
    if np.any(np.isnan(aucs)):
        return {
            "auc1": np.nan,
            "auc2": np.nan,
            "z_statistic": 0.0,
            "p_value": 1.0,
            "auc_diff": np.nan,
            "se_diff": np.nan,
        }

    # Variance of the difference: Var(AUC1 - AUC2) = Var(AUC1) + Var(AUC2) - 2*Cov(AUC1, AUC2)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se_diff = np.sqrt(max(var_diff, 0))

    if se_diff == 0:
        z = 0.0
    else:
        z = (aucs[0] - aucs[1]) / se_diff

    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "auc1": aucs[0],
        "auc2": aucs[1],
        "z_statistic": z,
        "p_value": p_value,
        "auc_diff": aucs[0] - aucs[1],
        "se_diff": se_diff,
    }
