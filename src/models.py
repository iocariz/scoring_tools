import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree as sktree
import matplotlib.pyplot as plt
from typing import List
from loguru import logger

def train_logistic_regression(X, y):
    """Train logistic regression on standardized data."""
    scaler = StandardScaler()
    X_standarized = scaler.fit_transform(X)
    log_reg = LogisticRegression().fit(X_standarized, y)
    return log_reg, X_standarized

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
    
    data_frame['group'] = pd.cut(data_frame[numeric_variable],
                                 bins=[-float("inf")] + splits + [float("inf")],
                                 labels=range(1, len(splits) + 2))
    
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
    relevant_data = data[data['se_decision_id'] == 'ok']

    # Use loc for filtering for better readability
    booked_data = relevant_data.loc[relevant_data['status_name'] == 'booked']

    # Group and calculate rates (no need to reset_index)
    financing_rate_by_month = (booked_data.groupby('mis_date')['oa_amt'].sum() /
                               relevant_data.groupby('mis_date')['oa_amt'].sum())

    # Calculate recent financing rates
    recent_data = relevant_data.loc[relevant_data['mis_date'] >= date_ini_demand]
    recent_booked_data = recent_data.loc[recent_data['status_name'] == 'booked']
    financing_rate_UM = recent_booked_data['oa_amt'].sum() / recent_data['oa_amt'].sum()

    # Plotting (use more descriptive labels and better formatting)
    plt.figure(figsize=(10, 6))  # Adjust figure size
    financing_rate_by_month.plot(title=f"Financing Rate Over Time (since {date_ini_demand.strftime('%Y-%m')})")
    plt.ylabel('Financing Rate (%)')
    plt.xlabel('Month')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Calculate and display results with consistent rounding
    mean_financing_rate = np.round(financing_rate_by_month.tail(lm).mean() * 100, 1)
    logger.info(f"Financing Rate (Mean of last {lm} months): {mean_financing_rate:.1f}%")
    logger.info(f"Financing Rate (Weighted for last {lm} months): {np.round(financing_rate_UM * 100, 1):.1f}%")

    # Indicate selected rate more clearly
    logger.info("--------------------------------")
    logger.info(f"Selected Financing Rate: {mean_financing_rate:.1f}%")

    return mean_financing_rate/100

def transform_variables(df, variables):
    # Destructure the variables for clarity
    var0, var1 = variables

    # Compute transformations directly without repeated assignments
    df[f"{var0}_{var1}"] = df[var0] * df[var1]
    df[f"{var0}^2"] = df[var0]**2
    df[f"{var1}^2"] = df[var1]**2
    df[f"{var0}^3"] = df[var0]**3
    df[f"{var1}^3"] = df[var1]**3

    # Computing transformations related to (9 - var0)
    df[f"9-{var0}"] = 9 - df[var0]
    df[f"9-{var1}"] = 9 - df[var1]
    df[f"(9-{var0}) x {var1}"] = df[f"9-{var0}"] * df[var1]
    df[f"(9-{var0})^2 x {var1}"] = df[f"9-{var0}"]**2 * df[var1]
    df[f"(9-{var0}) x {var1}^2"] = df[f"9-{var0}"] * df[var1]**2
    df[f"(9-{var0})^2"] = df[f"9-{var0}"]**2
    df[f"(9-{var1})^2"] = df[f"9-{var1}"]**2
    df[f"(9-{var0})^3"] = df[f"9-{var0}"]**3
    
    return df

def preprocess_data(data, octroi_limits, efx_limits, variables, indicadores, var_target, multiplier):
    # Creating a MultiIndex for efficient grid generation
    index = pd.MultiIndex.from_product([range(len(octroi_limits) - 1),
                                       range(len(efx_limits) - 1)],
                                      names=variables)
    data_train = pd.DataFrame(index=index).reset_index()

    data_train = transform_variables(data_train, variables)

    # Merging with improved filtering
    booked_data = data.loc[data['status_name'] == 'booked', variables + indicadores]
    data_train = data_train.merge(
        booked_data.groupby(variables).sum().reset_index(),
        on=variables,
        how='left'
    )

    # Calculating the target variable more directly
    data_train[var_target] = np.round(
        multiplier * data_train['todu_30ever_h6'] / data_train['todu_amt_pile_h6'], 2)

    return data_train

# funcion para aplicar el modelo de B2
def calculate_B2(df, model_risk, variables, stressor, var_reg):
    data_out = transform_variables(df.copy(), variables)
    data_out['b2_ever_h6'] = np.clip(stressor*model_risk.predict(data_out[var_reg]), a_min=0, a_max=None)
    return(data_out)

# funcion para aplicar el modelo de RV acumulado
def calculate_RV(df, model_rv):
    df['todu_amt_pile_h6'] = model_rv.predict(df[['oa_amt']])
    return(df)

# funcion para calcularlo svalores de riesgo inferidos
def calculate_risk_values(df, model_risk, model_rv, variables, stressor, var_reg):
    df = calculate_RV(df, model_rv)
    df = calculate_B2(df, model_risk, variables, stressor, var_reg)
    df['todu_30ever_h6'] = df['b2_ever_h6']*df['todu_amt_pile_h6']/7
    return(df)
