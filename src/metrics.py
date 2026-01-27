"""
Statistical metrics and model evaluation functions for credit risk scoring.

This module provides functions for evaluating credit risk model performance:
- Classification metrics (Gini, AUC, KS statistic)
- Population Stability Index (PSI) calculation
- Bootstrap confidence intervals
- Rejection threshold analysis
- Information Value (IV) calculation
- Comprehensive model summary reports

Key functions:
- compute_metrics: Calculate Gini, AUC, KS, and CAP curve
- bootstrap_confidence_interval: Compute CI for Gini and KS via bootstrap
- calculate_psi: Population Stability Index with visualization
- model_summary: Generate comprehensive model performance report
- calc_iv: Calculate Information Value for feature selection
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
    ks = ks_statistic(y_true, scores)

    # CAP Curve Calculation
    y_true_array = np.array(y_true)
    sorted_indices = np.argsort(scores)[::-1]
    sorted_true_values = y_true_array[sorted_indices]
    cumulative_true_positive = np.cumsum(sorted_true_values)

    return gini, roc_auc, ks, cumulative_true_positive

def bootstrap_confidence_interval(y_true, y_scores, n_iterations=100, alpha=0.05):
    """Compute bootstrap confidence interval for Gini and KS."""
    
    gini_scores = []
    ks_scores = []
    
    for _ in range(n_iterations):
        sample_indices = np.random.choice(len(y_true), len(y_true), replace=True)
        
        sampled_y_true = y_true.iloc[sample_indices].values
        sampled_y_scores = y_scores.iloc[sample_indices].values
        
        # Compute GINI
        fpr, tpr, _ = roc_curve(sampled_y_true, sampled_y_scores)
        auc_score = auc(fpr, tpr)
        gini_scores.append(2 * auc_score - 1)
        
        # Compute KS
        ks_scores.append(ks_statistic(sampled_y_true, sampled_y_scores))
    
    gini_ci = (np.percentile(gini_scores, 100 * alpha / 2.), np.percentile(gini_scores, 100 * (1 - alpha / 2.)))
    ks_ci = (np.percentile(ks_scores, 100 * alpha / 2.), np.percentile(ks_scores, 100 * (1 - alpha / 2.)))
    
    return gini_ci, ks_ci

def calculate_rejection_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: list = [5, 10, 15, 20]
) -> dict:
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

def calculate_psi(
    data: pd.DataFrame,
    date_column: str,
    score_column: str,
    start_date_ref: pd.Timestamp,
    end_date_ref: pd.Timestamp,
    start_date_act: pd.Timestamp,
    end_date_act: pd.Timestamp,
    buckets: int = None,
    breakpoints: list = None,
    show_plots: bool = True
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
    if not all([isinstance(i, pd.Timestamp) for i in [start_date_ref, end_date_ref, start_date_act, end_date_act]]):
        raise ValueError("Date inputs should be of type pd.Timestamp")
    if buckets is None and breakpoints is None:
        raise ValueError("Either buckets or breakpoints should be provided")

    # Filter data for expected and actual
    expected = data.loc[(data[date_column] >= start_date_ref) & (data[date_column] <= end_date_ref)]
    actual = data.loc[(data[date_column] >= start_date_act) & (data[date_column] <= end_date_act)]

    # Create buckets
    if breakpoints is None:
        breakpoints = np.percentile(expected[score_column], np.linspace(0, 100, buckets+1))
    buckets = len(breakpoints) - 1

    # Calculate bucket counts for expected and actual data
    initial_counts = np.histogram(expected[score_column], breakpoints)[0]
    new_counts = np.histogram(actual[score_column], breakpoints)[0]

    # Generate dataframe
    df = pd.DataFrame({
        'Bucket': np.arange(1, buckets+1),
        'Breakpoint Value': breakpoints[1:],
        'Expected Count': initial_counts,
        'Actual Count': new_counts
    })

    df['Expected Percent'] = df['Expected Count'] / len(expected)
    df['Actual Percent'] = df['Actual Count'] / len(actual)

    # Handle edge cases for computing PSI
    epsilon = 0.001  # small value to prevent log(0)
    df['Expected Percent'] = df['Expected Percent'].replace(0, epsilon)
    df['Actual Percent'] = df['Actual Percent'].replace(0, epsilon)

    # Compute PSI
    df['PSI'] = (df['Actual Percent'] - df['Expected Percent']) * np.log(df['Actual Percent'] / df['Expected Percent'])
    total_psi = df['PSI'].sum()

    # Plotting
    if show_plots:
        fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
        
        # Set the global title for the entire figure
        fig.suptitle(f"Population Stability Report\nExpected: {start_date_ref.strftime('%Y-%m-%d')} to {end_date_ref.strftime('%Y-%m-%d')}\nActual: {start_date_act.strftime('%Y-%m-%d')} to {end_date_act.strftime('%Y-%m-%d')}", fontsize=16, fontweight='bold', y=1.08)  

        # KDE Plot
        sns.set_style("whitegrid")
        sns.kdeplot(data=actual, x=score_column, fill=True, label='Actual', ax=axes[0], legend=True)
        sns.kdeplot(data=expected , x=score_column, fill=True, label='Expected', ax=axes[0], legend=True)
        axes[0].set_title("Kernel Density Plot of Scores")
        axes[0].tick_params(labelsize=12)
        axes[0].legend()
        sns.despine(left=True, ax=axes[0])

        # CDF plot
        sorted_actual_scores = np.sort(actual[score_column])
        sorted_expected_scores = np.sort(expected[score_column])
        p_actual = np.arange(1, len(sorted_actual_scores)+1) / len(sorted_actual_scores)
        p_expected = np.arange(1, len(sorted_expected_scores)+1) / len(sorted_expected_scores)
        axes[1].plot(sorted_actual_scores, p_actual, marker='.', linestyle='none', label='Actual')
        axes[1].plot(sorted_expected_scores, p_expected, marker='.', linestyle='none', label='Expected')
        axes[1].set_title(f'CDF Plot of Scores (PSI = {total_psi:.5f})')
        axes[1].set_xlabel(score_column)
        axes[1].set_ylabel('CDF')
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
    alpha: float = 0.05
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
    from .plots import visualize_metrics, plot_gini_confidence_intervals
    
    y_true = df[target_column]

    # Check if required columns exist in dataframe
    for info in score_columns.values():
        if info['column'] not in df.columns:
            raise ValueError(f"Column '{info['column']}' not found in dataframe.")

    # Prepare Data
    scores_dict = {name: (-1 if info['negate'] else 1) * df[info['column']].values for name, info in score_columns.items()}

    if combined_columns:
        for name, columns in combined_columns.items():
            log_reg, X_standardized = train_logistic_regression(df[list(columns)], y_true)
            scores_dict[name] = (log_reg.coef_[0] * X_standardized).sum(axis=1)

    # Compute Metrics and Rejection Thresholds
    metrics_data = {'Model': [], 'Gini Score': [], 'AUC': [], 'KS Value': [], 'Gini CI': [], 'KS CI': [], '5% Rejection': [], '10% Rejection': [], '15% Rejection': [], '20% Rejection': []}

    for name, scores in scores_dict.items():
        gini, roc_auc, ks, _ = compute_metrics(y_true, scores)
        gini_ci, ks_ci = bootstrap_confidence_interval(y_true, pd.Series(scores), n_iterations, alpha)
        rejection_thresholds = calculate_rejection_thresholds(y_true, scores)

        metrics_data['Model'].append(name)
        metrics_data['Gini Score'].append(gini)
        metrics_data['AUC'].append(roc_auc)
        metrics_data['KS Value'].append(ks)
        metrics_data['Gini CI'].append(gini_ci)
        metrics_data['KS CI'].append(ks_ci)
        metrics_data['5% Rejection'].append(rejection_thresholds['5%'])
        metrics_data['10% Rejection'].append(rejection_thresholds['10%'])
        metrics_data['15% Rejection'].append(rejection_thresholds['15%'])
        metrics_data['20% Rejection'].append(rejection_thresholds['20%'])

    # Format and Return
    summary_df = pd.DataFrame(metrics_data)
    summary_df['Gini CI'] = summary_df['Gini CI'].apply(lambda x: (round(x[0], 4), round(x[1], 4)))
    summary_df['KS CI'] = summary_df['KS CI'].apply(lambda x: (round(x[0], 4), round(x[1], 4)))

    # Visualization
    if plot:
        fig, axes = plt.subplots(1,5, figsize=(25,7))  # Adjust subplot dimensions as needed
        plt.subplots_adjust(wspace=0.4)

        visualize_metrics(y_true, scores_dict, ax=axes[:4])

        plot_gini_confidence_intervals(axes[4], summary_df)

        fig.suptitle('Model Performance Evaluation', fontsize=22, y=1.05)
        plt.tight_layout()
        plt.show()

    return summary_df

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
    df_tmp = df.groupby(var).agg({target: ['sum', 'count']})
    df_tmp.columns = ['sum', 'count']
    df_tmp = df_tmp.reset_index()
    df_tmp['perc_bad'] = df_tmp['sum'] / df_tmp['sum'].sum()
    df_tmp['perc_good'] = (df_tmp['count'] - df_tmp['sum']) / (df_tmp['count'].sum() - df_tmp['sum'].sum())
    df_tmp['woe'] = np.log((df_tmp['perc_good'] + 0.00001) / (df_tmp['perc_bad'] + 0.00001))
    df_tmp['iv'] = (df_tmp['perc_good'] - df_tmp['perc_bad']) * df_tmp['woe']
    iv = df_tmp['iv'].sum()
    
    return iv
