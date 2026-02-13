"""
Machine learning models and data transformation functions for credit risk scoring.

This module provides model training, data transformation, and risk calculation functions:
- Decision tree-based optimal binning
- Variable transformations for risk modeling
- Risk value calculations (B2, RV, TODU)
- Financing rate calculations

Key functions:
- optimal_splits_using_tree: Find optimal variable splits using decision trees
- transform_variables: Apply polynomial transformations for regression
- calculate_risk_values: Calculate inferred risk values using trained models
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn import tree as sktree
from sklearn.tree import DecisionTreeClassifier

from .constants import Columns, StatusName


def extract_splits_from_tree(tree, feature_names):
    """
    Extract split thresholds from decision tree.
    """
    tree_ = tree.tree_
    feature = tree_.feature
    threshold = tree_.threshold

    splits = []
    for i, f in enumerate(feature):
        if f != sktree._tree.TREE_UNDEFINED:
            splits.append(threshold[i])

    return sorted(splits)


def optimal_splits_using_tree(data_frame, numeric_variable, binary_outcome, num_groups):
    """
    Bin the numeric_variable using decision trees and assign rows to groups.
    """
    X = data_frame[[numeric_variable]]
    y = data_frame[binary_outcome]

    tree = DecisionTreeClassifier(max_leaf_nodes=num_groups, min_samples_leaf=500)
    tree.fit(X, y)

    splits = extract_splits_from_tree(tree, [numeric_variable])

    data_frame["group"] = pd.cut(
        data_frame[numeric_variable], bins=[-float("inf")] + splits + [float("inf")], labels=range(1, len(splits) + 2)
    )

    return data_frame


def calculate_financing_rates(data, date_ini_demand, lm=6):
    """
    Calculates financing rates based on given data and displays the results.

    Args:
        data (pd.DataFrame): Input data with columns:
            - 'se_decision_id'
            - 'status_name'
            - 'oa_amt'
            - 'mis_date'
        date_ini_demand (datetime): Start date for analysis.
        lm (int, optional): Number of last months to consider (default=6).

    Returns:
        float: Selected (mean) financing rate.
    """

    # Filter data once for efficiency
    relevant_data = data[data[Columns.SE_DECISION_ID] == "ok"]

    # Use loc for filtering for better readability
    booked_data = relevant_data.loc[relevant_data[Columns.STATUS_NAME] == StatusName.BOOKED.value]

    # Group and calculate rates (no need to reset_index)
    financing_rate_by_month = (
        booked_data.groupby(Columns.MIS_DATE)[Columns.OA_AMT].sum()
        / relevant_data.groupby(Columns.MIS_DATE)[Columns.OA_AMT].sum()
    )

    # Calculate recent financing rates
    recent_data = relevant_data.loc[relevant_data[Columns.MIS_DATE] >= date_ini_demand]
    recent_booked_data = recent_data.loc[recent_data[Columns.STATUS_NAME] == StatusName.BOOKED.value]
    financing_rate_UM = recent_booked_data[Columns.OA_AMT].sum() / recent_data[Columns.OA_AMT].sum()

    # Plotting (use more descriptive labels and better formatting)
    plt.figure(figsize=(10, 6))  # Adjust figure size
    financing_rate_by_month.plot(title=f"Financing Rate Over Time (since {date_ini_demand.strftime('%Y-%m')})")
    plt.ylabel("Financing Rate (%)")
    plt.xlabel("Month")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.show()

    # Calculate and display results with consistent rounding
    mean_financing_rate = np.round(financing_rate_by_month.tail(lm).mean() * 100, 1)
    logger.info(f"Financing Rate (Mean of last {lm} months): {mean_financing_rate:.1f}%")
    logger.info(f"Financing Rate (Weighted for last {lm} months): {np.round(financing_rate_UM * 100, 1):.1f}%")

    # Indicate selected rate more clearly
    logger.info("--------------------------------")
    logger.info(f"Selected Financing Rate: {mean_financing_rate:.1f}%")

    return mean_financing_rate / 100


def transform_variables(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Apply polynomial and interaction transformations to create regression features.

    Creates the following features from two input variables (var0, var1):
    - Interaction: var0 * var1
    - Squares: var0^2, var1^2
    - Cubes: var0^3, var1^3
    - Complements: 9-var0, 9-var1
    - Complement interactions: (9-var0) x var1, etc.

    Args:
        df: Input DataFrame containing the base variables.
        variables: List of two variable names [var0, var1] to transform.

    Returns:
        DataFrame with original columns plus all transformed features.

    Note:
        The complement (9 - var) is used because risk scores typically range 0-9,
        where 9 represents the best credit quality.
    """
    # Destructure the variables for clarity
    var0, var1 = variables

    # Compute transformations directly without repeated assignments
    df[f"{var0}_{var1}"] = df[var0] * df[var1]
    df[f"{var0}^2"] = df[var0] ** 2
    df[f"{var1}^2"] = df[var1] ** 2
    df[f"{var0}^3"] = df[var0] ** 3
    df[f"{var1}^3"] = df[var1] ** 3

    # Computing transformations related to (9 - var0)
    df[f"9-{var0}"] = 9 - df[var0]
    df[f"9-{var1}"] = 9 - df[var1]
    df[f"(9-{var0}) x {var1}"] = df[f"9-{var0}"] * df[var1]
    df[f"(9-{var0})^2 x {var1}"] = df[f"9-{var0}"] ** 2 * df[var1]
    df[f"(9-{var0}) x {var1}^2"] = df[f"9-{var0}"] * df[var1] ** 2
    df[f"(9-{var0})^2"] = df[f"9-{var0}"] ** 2
    df[f"(9-{var1})^2"] = df[f"9-{var1}"] ** 2
    df[f"(9-{var0})^3"] = df[f"9-{var0}"] ** 3

    return df


def preprocess_data(
    data: pd.DataFrame,
    octroi_limits: list[float],
    efx_limits: list[float],
    variables: list[str],
    indicadores: list[str],
    var_target: str,
    multiplier: float,
) -> pd.DataFrame:
    """
    Preprocess data by creating a grid and aggregating booked records.

    Creates a grid of all possible combinations of binned variables,
    applies transformations, merges with aggregated booked data, and
    calculates the target risk metric.

    Args:
        data: Input DataFrame with raw records.
        octroi_limits: Bin boundaries for the first variable (octroi).
        efx_limits: Bin boundaries for the second variable (efx).
        variables: List of two variable names for grid dimensions.
        indicadores: List of indicator columns to aggregate (sum).
        var_target: Name of the target variable to calculate.
        multiplier: Multiplier for target calculation (typically 7 for b2_ever_h6).

    Returns:
        DataFrame with grid, transformations, aggregated indicators, and target variable.
    """
    # Creating a MultiIndex for efficient grid generation
    index = pd.MultiIndex.from_product([range(len(octroi_limits) - 1), range(len(efx_limits) - 1)], names=variables)
    data_train = pd.DataFrame(index=index).reset_index()

    data_train = transform_variables(data_train, variables)

    # Merging with improved filtering
    booked_data = data.loc[data[Columns.STATUS_NAME] == StatusName.BOOKED.value, variables + indicadores]
    data_train = data_train.merge(booked_data.groupby(variables).sum().reset_index(), on=variables, how="left")

    # Calculating the target variable more directly
    data_train[var_target] = np.round(multiplier * data_train["todu_30ever_h6"] / data_train["todu_amt_pile_h6"], 2)

    return data_train


def calculate_B2(
    df: pd.DataFrame, model_risk, variables: list[str], stressor: float, var_reg: list[str]
) -> pd.DataFrame:
    """
    Apply the B2 risk model to calculate b2_ever_h6 predictions.

    Transforms variables and applies the trained risk model with a stress
    factor to predict the B2 risk metric. Results are clipped to non-negative.

    Args:
        df: Input DataFrame with base variables.
        model_risk: Trained sklearn model with predict() method.
        variables: List of two variable names for transformation.
        stressor: Stress multiplier to apply to predictions.
        var_reg: List of regression feature names for model input.

    Returns:
        DataFrame with transformed variables and 'b2_ever_h6' predictions.
    """
    data_out = transform_variables(df.copy(), variables)
    data_out["b2_ever_h6"] = np.clip(stressor * model_risk.predict(data_out[var_reg]), a_min=0, a_max=None)
    return data_out


def calculate_RV(df: pd.DataFrame, model_rv) -> pd.DataFrame:
    """
    Apply the RV (cumulative revenue) model to predict todu_amt_pile_h6.

    Uses the trained revenue model to predict cumulative TODU amount
    based on the original amount (oa_amt).

    Args:
        df: Input DataFrame containing 'oa_amt' column.
        model_rv: Trained sklearn model with predict() method.

    Returns:
        DataFrame with 'todu_amt_pile_h6' predictions added.
    """
    df["todu_amt_pile_h6"] = model_rv.predict(df[["oa_amt"]])
    return df


def calculate_risk_values(
    df: pd.DataFrame, model_risk, model_rv, variables: list[str], stressor: float, var_reg: list[str]
) -> pd.DataFrame:
    """
    Calculate all inferred risk values using trained models.

    Orchestrates the full risk calculation pipeline:
    1. Predicts cumulative revenue (todu_amt_pile_h6) using RV model
    2. Predicts B2 risk metric (b2_ever_h6) using risk model
    3. Calculates TODU 30-day ever metric from B2 and revenue

    Args:
        df: Input DataFrame with 'oa_amt' and base variables.
        model_risk: Trained risk model for B2 predictions.
        model_rv: Trained revenue model for TODU amount predictions.
        variables: List of two variable names for transformation.
        stressor: Stress multiplier to apply to risk predictions.
        var_reg: List of regression feature names for risk model.

    Returns:
        DataFrame with all risk metrics: 'todu_amt_pile_h6', 'b2_ever_h6',
        and 'todu_30ever_h6'.
    """
    df = calculate_RV(df, model_rv)
    df = calculate_B2(df, model_risk, variables, stressor, var_reg)
    df["todu_30ever_h6"] = df["b2_ever_h6"] * df["todu_amt_pile_h6"] / 7
    return df
