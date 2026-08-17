"""
Statistical metrics and model evaluation functions for credit risk scoring.

This module provides functions for evaluating credit risk model performance:
- Classification metrics (Gini, AUC, KS statistic)
- Precision-Recall metrics
- Population Stability Index (PSI) calculation
- Bootstrap confidence intervals
- Rejection threshold analysis
- Lift and cumulative gains analysis
- Information Value (IV) calculation
- DeLong test for AUC comparison
- Comprehensive model summary reports

Key functions:
- compute_metrics: Calculate Gini, AUC, KS, and CAP curve
- compute_precision_recall: Calculate precision, recall, and average precision
- bootstrap_confidence_interval: Compute CI for Gini and KS via bootstrap
- calculate_psi_by_period: Population Stability Index with visualization (date-based)
- calculate_lift_table: Decile-level lift and cumulative gains analysis
- model_summary: Generate comprehensive model performance report
- calc_iv: Calculate Information Value for feature selection
- delong_test: Statistical test for comparing two AUC values
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.constants import DEFAULT_N_BOOTSTRAPS, DEFAULT_RANDOM_STATE, PSI_EPSILON


def train_logistic_regression(X, y):
    """Train logistic regression on standardized data.

    NOTE (audit #35): scoring this model on its own training rows inflates the
    combined score's Gini/AUC/KS relative to the raw (unfitted) scores it is
    compared against. Use :func:`combined_score_cv` for evaluation.
    """
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    log_reg = LogisticRegression().fit(X_standardized, y)
    return log_reg, X_standardized


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


def calculate_psi_by_period(
    data: pd.DataFrame,
    date_column: str,
    score_column: str,
    start_date_ref: pd.Timestamp,
    end_date_ref: pd.Timestamp,
    start_date_act: pd.Timestamp,
    end_date_act: pd.Timestamp,
    buckets: int | None = None,
    breakpoints: list | None = None,
    show_plots: bool = False,
) -> pd.DataFrame:
    """
    Calculate Population Stability Index (PSI) between reference and actual periods.

    PSI measures how much a score distribution has shifted between two time periods.
    Values are interpreted as:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, action required

    Zero handling (audit #11): epsilon protects only the log term; the difference term keeps the true
    proportions. This variant is symmetric in appearing/disappearing bins and equals the textbook
    epsilon-in-both-terms PSI to O(epsilon), so the thresholds above remain valid.

    Args:
        data: DataFrame containing scores and dates.
        date_column: Name of the date column.
        score_column: Name of the score column to analyze.
        start_date_ref: Start date for reference (expected) period.
        end_date_ref: End date for reference period.
        start_date_act: Start date for actual period.
        end_date_act: End date for actual period.
        buckets: Number of equal-frequency buckets (mutually exclusive with breakpoints).
        breakpoints: Custom breakpoints for buckets (mutually exclusive with buckets).
        show_plots: Whether to display KDE and CDF comparison plots.

    Returns:
        DataFrame with PSI calculation by bucket, including breakpoints, counts, and PSI values.

    Raises:
        ValueError: If dates are not pd.Timestamp or neither buckets nor breakpoints provided.
    """
    # Ensure valid inputs
    if not all(isinstance(i, pd.Timestamp) for i in [start_date_ref, end_date_ref, start_date_act, end_date_act]):
        raise ValueError("Date inputs should be of type pd.Timestamp")
    if buckets is None and breakpoints is None:
        raise ValueError("Either buckets or breakpoints should be provided")

    # Filter data for expected and actual
    expected = data.loc[(data[date_column] >= start_date_ref) & (data[date_column] <= end_date_ref)]
    actual = data.loc[(data[date_column] >= start_date_act) & (data[date_column] <= end_date_act)]

    # Create buckets
    if breakpoints is None:
        breakpoints = np.percentile(expected[score_column], np.linspace(0, 100, buckets + 1))
    # Drop duplicate edges (audit #12): on a low-cardinality score np.percentile yields repeated
    # breakpoints, which would make np.histogram raise "bins must increase monotonically".
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        # Near-constant reference: PSI undefined. Return a NaN result (not a crash, not a 0 that
        # would read as "stable").
        logger.warning(f"PSI by period: fewer than 2 distinct score breakpoints for '{score_column}'. Returning NaN.")
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
    # Extend edges to (-inf, +inf) so out-of-range actual values are captured
    # (mirrors stability.calculate_psi behaviour).
    breakpoints = list(breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    breakpoints = np.array(breakpoints)
    buckets = len(breakpoints) - 1

    # Calculate bucket counts for expected and actual data
    initial_counts = np.histogram(expected[score_column], breakpoints)[0]
    new_counts = np.histogram(actual[score_column], breakpoints)[0]

    # Generate dataframe
    df = pd.DataFrame(
        {
            "Bucket": np.arange(1, buckets + 1),
            "Breakpoint Value": breakpoints[1:],
            "Expected Count": initial_counts,
            "Actual Count": new_counts,
        }
    )

    df["Expected Percent"] = df["Expected Count"] / len(expected)
    df["Actual Percent"] = df["Actual Count"] / len(actual)

    # Apply epsilon only in the log term to avoid log(0) without distorting
    # the actual distributions (re-normalizing would change divergence scale).
    epsilon = PSI_EPSILON
    expected_safe = df["Expected Percent"].where(df["Expected Percent"] > 0, epsilon)
    actual_safe = df["Actual Percent"].where(df["Actual Percent"] > 0, epsilon)

    # Compute PSI (use original percentages for the difference, epsilon only in log)
    df["PSI"] = (df["Actual Percent"] - df["Expected Percent"]) * np.log(actual_safe / expected_safe)
    total_psi = df["PSI"].sum()

    # Plotting
    if show_plots:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))

        # Set the global title for the entire figure
        fig.suptitle(
            f"Population Stability Report\nExpected: {start_date_ref.strftime('%Y-%m-%d')} to {end_date_ref.strftime('%Y-%m-%d')}\nActual: {start_date_act.strftime('%Y-%m-%d')} to {end_date_act.strftime('%Y-%m-%d')}",
            fontsize=16,
            fontweight="bold",
            y=1.08,
        )

        # KDE Plot
        sns.set_style("whitegrid")
        sns.kdeplot(data=actual, x=score_column, fill=True, label="Actual", ax=axes[0], legend=True)
        sns.kdeplot(data=expected, x=score_column, fill=True, label="Expected", ax=axes[0], legend=True)
        axes[0].set_title("Kernel Density Plot of Scores")
        axes[0].tick_params(labelsize=12)
        axes[0].legend()
        sns.despine(left=True, ax=axes[0])

        # CDF plot
        sorted_actual_scores = np.sort(actual[score_column])
        sorted_expected_scores = np.sort(expected[score_column])
        p_actual = np.arange(1, len(sorted_actual_scores) + 1) / len(sorted_actual_scores)
        p_expected = np.arange(1, len(sorted_expected_scores) + 1) / len(sorted_expected_scores)
        axes[1].plot(sorted_actual_scores, p_actual, marker=".", linestyle="none", label="Actual")
        axes[1].plot(sorted_expected_scores, p_expected, marker=".", linestyle="none", label="Expected")
        axes[1].set_title(f"CDF Plot of Scores (PSI = {total_psi:.5f})")
        axes[1].set_xlabel(score_column)
        axes[1].set_ylabel("CDF")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

    return df


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
    n_records = len(y_true)
    n_bads = int(y_true.sum())
    bad_rate = n_bads / n_records if n_records > 0 else 0.0

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
        gini, roc_auc, ks, _ = compute_metrics(
            y_true[valid] if not valid.all() else y_true, scores[valid] if not valid.all() else scores
        )
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


def calc_iv(df: pd.DataFrame, var: str, target: str) -> float:
    """
    Calculate Information Value (IV) for a categorical variable.

    IV measures the predictive power of a variable for binary classification.
    Interpretation:
    - IV < 0.02: Not useful for prediction
    - 0.02 <= IV < 0.1: Weak predictive power
    - 0.1 <= IV < 0.3: Medium predictive power
    - IV >= 0.3: Strong predictive power

    Args:
        df: DataFrame containing the variable and target.
        var: Name of the categorical variable to evaluate.
        target: Name of the binary target column (1=bad, 0=good).

    Returns:
        Information Value as a float.
    """
    df_tmp = df.groupby(var).agg({target: ["sum", "count"]})
    df_tmp.columns = ["sum", "count"]
    df_tmp = df_tmp.reset_index()

    total_bad = df_tmp["sum"].sum()
    total_good = df_tmp["count"].sum() - total_bad

    # Guard against degenerate cases (all-bad or all-good)
    if total_bad == 0 or total_good == 0:
        return 0.0

    # Laplace smoothing: add 0.5 to bad and good counts per bin to stabilize
    # WOE for zero-event bins, instead of epsilon substitution which produces
    # extreme WOE values that inflate IV (#13).
    n_bins = len(df_tmp)
    bad_smooth = df_tmp["sum"] + 0.5
    good_smooth = (df_tmp["count"] - df_tmp["sum"]) + 0.5
    total_bad_smooth = total_bad + 0.5 * n_bins
    total_good_smooth = total_good + 0.5 * n_bins

    df_tmp["perc_bad"] = bad_smooth / total_bad_smooth
    df_tmp["perc_good"] = good_smooth / total_good_smooth
    df_tmp["woe"] = np.log(df_tmp["perc_good"] / df_tmp["perc_bad"])
    df_tmp["iv"] = (df_tmp["perc_good"] - df_tmp["perc_bad"]) * df_tmp["woe"]
    iv = df_tmp["iv"].sum()

    return iv


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
