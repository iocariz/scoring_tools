"""
Utility functions for credit risk scoring and portfolio optimization.

This module provides core utility functions used throughout the scoring tools:
- Risk metric calculations (b2_ever_h6)
- Data optimization and memory management

"""

import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from .constants import DEFAULT_RISK_MULTIPLIER

# Cap parallel workers to avoid OOM on many-core servers.
# Override with SCORING_TOOLS_MAX_JOBS environment variable.
MAX_PARALLEL_JOBS = int(os.environ.get("SCORING_TOOLS_MAX_JOBS", min(4, os.cpu_count() or 4)))


def calculate_b2_ever_h6(
    numerator: pd.Series | np.ndarray | float,
    denominator: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    as_percentage: bool = False,
    decimals: int = 2,
) -> pd.Series | np.ndarray | float:
    """
    Calculate the b2_ever_h6 risk metric.

    Formula: multiplier * numerator / denominator

    Args:
        numerator: todu_30ever_h6 values
        denominator: todu_amt_pile_h6 values
        multiplier: Risk multiplier (default: 7)
        as_percentage: If True, multiply result by 100
        decimals: Number of decimal places to round to

    Returns:
        Calculated b2_ever_h6 values, with division-by-zero handled as NaN.
        Callers that need 0 instead of NaN should apply np.nan_to_num() at the
        display/output boundary.
    """
    # Handle division by zero
    if isinstance(denominator, (pd.Series, np.ndarray)):
        safe_denominator = np.where(denominator == 0, np.nan, denominator)
    else:
        safe_denominator = np.nan if denominator == 0 else denominator

    result = multiplier * numerator / safe_denominator

    if as_percentage:
        result = result * 100

    return np.round(result, decimals)


def calculate_todu_30ever_from_b2(
    b2_ever_h6: pd.Series | np.ndarray | float,
    todu_amt_pile_h6: pd.Series | np.ndarray | float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
) -> pd.Series | np.ndarray | float:
    """
    Calculate todu_30ever_h6 from b2_ever_h6 and todu_amt_pile_h6.

    This is the inverse of calculate_b2_ever_h6:
        todu_30ever_h6 = b2_ever_h6 * todu_amt_pile_h6 / multiplier

    Args:
        b2_ever_h6: Risk metric values
        todu_amt_pile_h6: Exposure values
        multiplier: Risk multiplier (default: 7)

    Returns:
        Calculated todu_30ever_h6 values
    """
    return b2_ever_h6 * todu_amt_pile_h6 / multiplier


def get_data_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Display DataFrame information and return a DataFrame with variable details.
    """
    # Display DataFrame information
    logger.info(f"Number of rows/records: {df.shape[0]}")
    logger.info(f"Number of columns/variables: {df.shape[1]}")
    logger.info("-" * 50)

    # Create a DataFrame with variable information
    variables_df = pd.DataFrame(
        {
            "Variable": df.columns,
            "Number of unique values": df.nunique(),
            "Variable Type": df.dtypes,
            "Number of missing values": df.isnull().sum(),
            "Percentage missing values": df.isnull().mean() * 100,
        }
    )

    # Sort variables by percentage of missing values
    variables_df = variables_df.sort_values(by="Percentage missing values", ascending=False)

    # Return the DataFrame with variable information
    return variables_df


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by choosing appropriate integer dtypes.

    Works on a copy to avoid mutating the input. Float64 columns are preserved
    to avoid precision loss on financial data.
    """
    df = df.copy()
    for col in df.columns:
        # Convert integer columns
        if df[col].dtype == "int64":
            if df[col].min() >= 0:
                if df[col].max() <= 255:
                    df[col] = df[col].astype("uint8")
                elif df[col].max() <= 65535:
                    df[col] = df[col].astype("uint16")
                else:
                    df[col] = df[col].astype("uint32")
            else:
                if df[col].min() >= -128 and df[col].max() <= 127:
                    df[col] = df[col].astype("int8")
                elif df[col].min() >= -32768 and df[col].max() <= 32767:
                    df[col] = df[col].astype("int16")
                else:
                    df[col] = df[col].astype("int32")

        # Float64 columns are intentionally preserved to avoid precision loss

    return df


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
        logger.warning(f"No records found with {status_col} = {target_status}")
        return 0.0

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

    worst_bad_rate = (worst_num / worst_den * DEFAULT_RISK_MULTIPLIER) if worst_den > 0 else 0.0

    # Calculate stress factor
    if overall_bad_rate > 0:
        stress_factor = worst_bad_rate / overall_bad_rate
    else:
        stress_factor = 0.0

    return float(stress_factor)


def calculate_annual_coef(date_ini_book_obs: pd.Timestamp, date_fin_book_obs: pd.Timestamp) -> float:
    """
    Calculate annual coefficient based on the time range.
    """
    n_month = (
        (date_fin_book_obs.year - date_ini_book_obs.year) * 12 + (date_fin_book_obs.month - date_ini_book_obs.month) + 1
    )
    annual_coef = 12 / n_month
    return annual_coef


def _bootstrap_worker(
    df: pd.DataFrame,
    cut_map: dict[float, float],
    variables: list[str],
    multiplier: float,
    random_state: int | None = None,
    inv_var1: bool = False,
) -> tuple[float, float]:
    """Worker function for bootstrap resampling."""
    # Resample with replacement
    sample = df.sample(frac=1.0, replace=True, random_state=random_state)

    # Apply cuts
    var0 = variables[0]
    var1 = variables[1]

    # Map cuts to each row based on var0 bin
    # For missing bins, default to strict rejection:
    #   non-inverted (var1 <= cutoff): fillna(-inf) → always rejects
    #   inverted (var1 >= cutoff): fillna(+inf) → always rejects
    fallback = np.inf if inv_var1 else -np.inf
    full_cut_series = sample[var0].map(cut_map).fillna(fallback)

    # Filter passed — inverted variables use >= (higher bin = safer)
    if inv_var1:
        passes = sample[var1] >= full_cut_series
    else:
        passes = sample[var1] <= full_cut_series

    # Calculate metrics on passed (Production) and Risk (B2)
    # Production: sum of filtered oa_amt (assuming oa_amt is the production column,
    # usually it's oa_amt_h0 for optimization but we need to check what column to use.
    # In optimization pipeline, we optimize 'oa_amt_h0'.
    # But usually we want the Total Production of the SELECTED portfolio.

    # Actually, Risk B2 is calculated using todu_30ever_h6 and todu_amt_pile_h6
    # for the selected portfolio.

    passed_df = sample[passes]

    # Use oa_amt_h0 to match the optimization pipeline metric
    prod_col = "oa_amt_h0" if "oa_amt_h0" in passed_df.columns else "oa_amt"
    production = passed_df[prod_col].sum() if not passed_df.empty else 0.0

    risk_num = passed_df["todu_30ever_h6"].sum() if not passed_df.empty else 0.0
    risk_den = passed_df["todu_amt_pile_h6"].sum() if not passed_df.empty else 0.0

    risk = calculate_b2_ever_h6(risk_num, risk_den, multiplier=multiplier, as_percentage=False, decimals=6)

    return production, float(risk)


def calculate_bootstrap_intervals(
    data_booked: pd.DataFrame,
    cut_map: dict[float, float],
    variables: list[str],
    multiplier: float,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | None = 42,
    inv_var1: bool = False,
    model_cv_se_risk: float | None = None,
) -> dict[str, float]:
    """
    Calculate confidence intervals for Risk and Production using bootstrap resampling.

    Args:
        data_booked: DataFrame of booked accounts (patient-level data)
        cut_map: Dictionary mapping var0 bin values to var1 cutoff thresholds
        variables: List of [var0, var1] names
        multiplier: Risk multiplier
        n_bootstraps: Number of bootstrap iterations
        confidence_level: Confidence level (e.g., 0.95)
        random_state: Seed for reproducibility (default: 42)
        inv_var1: If True, use >= comparison for var1 (higher bin = safer)
        model_cv_se_risk: Optional standard error of the risk model's CV predictions.
            When provided, the risk CI is widened to account for model prediction
            uncertainty (which the bootstrap alone does not capture, since it only
            resamples booked records and ignores model inference error on the
            rejected/swap-in population).  The total SE is computed as
            ``sqrt(bootstrap_se² + model_cv_se²)``.

    Returns:
        Dictionary with lower/upper bounds for production and risk
    """
    logger.info(f"Calculating {confidence_level:.0%} CI with {n_bootstraps} bootstraps...")

    # Generate per-iteration seeds for reproducibility
    if random_state is not None:
        rng = np.random.RandomState(random_state)
        seeds = rng.randint(0, 2**31, size=n_bootstraps)
    else:
        seeds = [None] * n_bootstraps

    # Parallel execution (capped to avoid OOM)
    results = Parallel(n_jobs=MAX_PARALLEL_JOBS)(
        delayed(_bootstrap_worker)(
            data_booked,
            cut_map,
            variables,
            multiplier,
            random_state=int(seed) if seed is not None else None,
            inv_var1=inv_var1,
        )
        for seed in seeds
    )

    # Unzip results
    productions, risks = zip(*results)

    # Calculate percentiles
    alpha = (1 - confidence_level) / 2
    lower_p = alpha * 100
    upper_p = (1 - alpha) * 100

    prod_lower = np.percentile(productions, lower_p)
    prod_upper = np.percentile(productions, upper_p)
    risk_lower = np.percentile(risks, lower_p)
    risk_upper = np.percentile(risks, upper_p)

    # Inflate risk CI to account for model prediction uncertainty
    if model_cv_se_risk is not None and model_cv_se_risk > 0:
        from scipy.stats import norm

        z = norm.ppf(1 - alpha)
        risk_mean = float(np.mean(risks))
        bootstrap_se = float(np.std(risks, ddof=1))
        total_se = np.sqrt(bootstrap_se**2 + model_cv_se_risk**2)
        risk_lower = risk_mean - z * total_se
        risk_upper = risk_mean + z * total_se
        logger.info(
            f"  Risk CI inflated for model uncertainty: bootstrap_se={bootstrap_se:.6f}, "
            f"model_se={model_cv_se_risk:.6f}, total_se={total_se:.6f}"
        )

    return {
        "production_ci_lower": prod_lower,
        "production_ci_upper": prod_upper,
        "risk_ci_lower": risk_lower,
        "risk_ci_upper": risk_upper,
    }


def generate_cutoff_summary(
    optimal_solution_df: pd.DataFrame,
    variables: list[str],
    segment_name: str,
    scenario_name: str = "base",
    risk_value: float | None = None,
    production_value: float | None = None,
    ci_data: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Generate a readable summary of cutoff points by segment.

    This function transforms the optimal solution into a human-readable format
    showing the cutoff threshold for each bin of the first variable.

    Args:
        optimal_solution_df: DataFrame containing the optimal solution with bin columns
        variables: List of two variable names [var0, var1] used in the optimization
        segment_name: Name of the segment being analyzed
        scenario_name: Name of the scenario (e.g., 'pessimistic', 'base', 'optimistic')
        risk_value: Optional risk (b2_ever_h6) value for this solution
        production_value: Optional production (oa_amt_h0) value for this solution

    Returns:
        DataFrame with columns:
        - segment: Segment name
        - scenario: Scenario name
        - var0_bin: Bin number of the first variable
        - var0_name: Name of the first variable
        - cutoff_value: Maximum value of var1 to accept for this bin
        - var1_name: Name of the second variable
        - risk_pct: Risk percentage (if provided)
        - production: Production amount (if provided)

    Example output:
        segment     scenario  var0_bin  var0_name          cutoff_value  var1_name
        loan_known  base      1         sc_octroi_new_clus  2            new_efx_clus
        loan_known  base      2         sc_octroi_new_clus  4            new_efx_clus
        ...
    """
    if optimal_solution_df is None or optimal_solution_df.empty:
        logger.warning("No optimal solution provided for cutoff summary")
        return pd.DataFrame()

    var0_name = variables[0] if len(variables) > 0 else "var0"
    var1_name = variables[1] if len(variables) > 1 else "var1"

    # Get the first (selected) solution row
    opt_row = optimal_solution_df.iloc[0]

    # Find bin columns (numeric column names representing bins)
    bin_columns = []
    for col in optimal_solution_df.columns:
        try:
            bin_val = int(col) if isinstance(col, str) else col
            if isinstance(bin_val, (int, float)) and not pd.isna(bin_val):
                bin_columns.append((bin_val, col))
        except (ValueError, TypeError):
            continue

    # Sort bins numerically
    bin_columns.sort(key=lambda x: x[0])

    if not bin_columns:
        logger.warning("No bin columns found in optimal solution")
        return pd.DataFrame()

    # Build summary rows
    summary_rows = []
    for bin_val, col_name in bin_columns:
        cutoff = opt_row[col_name]

        if pd.notna(cutoff) and np.isfinite(cutoff):
            safe_cutoff = int(cutoff)
        elif pd.notna(cutoff):
            safe_cutoff = float(cutoff)  # Keep as inf or -inf
        else:
            safe_cutoff = None

        row_data = {
            "segment": segment_name,
            "scenario": scenario_name,
            f"{var0_name}_bin": int(bin_val),
            "var0_name": var0_name,
            "cutoff_value": safe_cutoff,
            "var1_name": var1_name,
            "risk_pct": risk_value,
            "production": production_value,
        }

        if ci_data:
            row_data["production_ci_lower"] = ci_data.get("production_ci_lower")
            row_data["production_ci_upper"] = ci_data.get("production_ci_upper")
            # Risk CI is raw, convert to % if needed
            row_data["risk_ci_lower"] = ci_data.get("risk_ci_lower", 0) * 100
            row_data["risk_ci_upper"] = ci_data.get("risk_ci_upper", 0) * 100

        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)

    logger.info(f"Generated cutoff summary for segment '{segment_name}', scenario '{scenario_name}'")
    logger.info(
        f"  Bins: {len(bin_columns)}, Cutoff range: "
        f"[{summary_df['cutoff_value'].min()}, {summary_df['cutoff_value'].max()}]"
    )

    return summary_df


def format_cutoff_summary_table(
    cutoff_summary: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    Format cutoff summary into a wide pivot table for easier reading.

    Args:
        cutoff_summary: DataFrame from generate_cutoff_summary
        variables: List of variable names [var0, var1]

    Returns:
        Pivoted DataFrame with segments/scenarios as rows and bins as columns
    """
    if cutoff_summary.empty:
        return pd.DataFrame()

    var0_name = variables[0] if len(variables) > 0 else "var0"
    bin_col = f"{var0_name}_bin"

    # Pivot to wide format
    pivot_df = cutoff_summary.pivot_table(
        index=["segment", "scenario", "risk_pct", "production"], columns=bin_col, values="cutoff_value", aggfunc="first"
    ).reset_index()

    # Rename bin columns to be more readable
    pivot_df.columns = [f"bin_{col}" if isinstance(col, (int, float)) else col for col in pivot_df.columns]

    return pivot_df


def consolidate_cutoff_summaries(
    summaries: list[pd.DataFrame],
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Consolidate multiple cutoff summaries into a single DataFrame.

    Args:
        summaries: List of DataFrames from generate_cutoff_summary
        output_path: Optional path to save the consolidated summary as CSV

    Returns:
        Consolidated DataFrame with all summaries
    """
    if not summaries:
        logger.warning("No summaries provided for consolidation")
        return pd.DataFrame()

    # Filter out empty DataFrames
    valid_summaries = [s for s in summaries if not s.empty]

    if not valid_summaries:
        logger.warning("All provided summaries are empty")
        return pd.DataFrame()

    consolidated = pd.concat(valid_summaries, ignore_index=True)

    if output_path:
        consolidated.to_csv(output_path, index=False)
        logger.info(f"Consolidated cutoff summary saved to {output_path}")

    return consolidated
