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
from scipy import stats
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from src.constants import DEFAULT_RANDOM_STATE


def train_logistic_regression(X, y):
    """Train logistic regression on standardized data."""
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    log_reg = LogisticRegression().fit(X_standardized, y)
    return log_reg, X_standardized


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


def bootstrap_confidence_interval(y_true, y_scores, n_iterations=100, alpha=0.05, random_state=42):
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

        # Compute GINI
        fpr, tpr, _ = roc_curve(sampled_y_true, sampled_y_scores)
        auc_score = auc(fpr, tpr)
        gini_scores.append(2 * auc_score - 1)

        # Compute KS
        ks_scores.append(ks_statistic(sampled_y_true, sampled_y_scores))

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
    buckets: int = None,
    breakpoints: list = None,
    show_plots: bool = True,
) -> pd.DataFrame:
    """
    Calculate Population Stability Index (PSI) between reference and actual periods.

    PSI measures how much a score distribution has shifted between two time periods.
    Values are interpreted as:
    - PSI < 0.1: No significant change
    - 0.1 <= PSI < 0.25: Moderate change, investigate
    - PSI >= 0.25: Significant change, action required

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

    # Handle edge cases for computing PSI
    epsilon = 0.001  # small value to prevent log(0)
    df["Expected Percent"] = df["Expected Percent"].replace(0, epsilon)
    df["Actual Percent"] = df["Actual Percent"].replace(0, epsilon)

    # Compute PSI
    df["PSI"] = (df["Actual Percent"] - df["Expected Percent"]) * np.log(df["Actual Percent"] / df["Expected Percent"])
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
    combined_columns: dict = None,
    plot: bool = True,
    n_iterations: int = 100,
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

    if combined_columns:
        for name, columns in combined_columns.items():
            log_reg, X_standardized = train_logistic_regression(df[list(columns)], y_true)
            scores_dict[name] = (log_reg.coef_[0] * X_standardized).sum(axis=1)

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
        gini, roc_auc, ks, _ = compute_metrics(y_true, scores)
        gini_ci, ks_ci = bootstrap_confidence_interval(y_true, pd.Series(scores), n_iterations, alpha)
        rejection_thresholds = calculate_rejection_thresholds(y_true, scores)

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
            log_reg, X_std = train_logistic_regression(df[list(columns)], y_true)
            scores_dict[name] = (log_reg.coef_[0] * X_std).sum(axis=1)

    rows = []
    for name, scores in scores_dict.items():
        gini, roc_auc, ks, _ = compute_metrics(y_true, scores)
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
    df_tmp["perc_bad"] = df_tmp["sum"] / df_tmp["sum"].sum()
    df_tmp["perc_good"] = (df_tmp["count"] - df_tmp["sum"]) / (df_tmp["count"].sum() - df_tmp["sum"].sum())
    df_tmp["woe"] = np.log((df_tmp["perc_good"] + 0.00001) / (df_tmp["perc_bad"] + 0.00001))
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

    aucs = []
    structural_components = []

    for scores in [scores1, scores2]:
        # Compute the structural components (placement values)
        # For positives: fraction of negatives with lower score
        # For negatives: fraction of positives with lower score
        ordered = np.concatenate([scores[positive_mask], scores[negative_mask]])

        midranks = _compute_midrank(ordered)

        positive_ranks = midranks[:m]
        auc_val = (positive_ranks.sum() - m * (m + 1) / 2) / (m * n)
        aucs.append(auc_val)

        # Structural components for variance estimation
        v_positive = (positive_ranks - np.arange(1, m + 1)) / n
        v_negative = 1.0 - (_compute_midrank(-ordered)[m:] - np.arange(1, n + 1)) / m

        structural_components.append((v_positive, v_negative))

    # Compute 2x2 covariance matrix
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
