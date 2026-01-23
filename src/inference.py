from src.models import preprocess_data
# Standard lbrary imports
from IPython.display import clear_output, display, HTML
import pandas as pd
import numpy as np
from pathlib import Path
import random
from dateutil.relativedelta import relativedelta
from IPython.display import display
from pyarrow import feather

# Visualization libraries
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ipywidgets as widgets
from ipywidgets import interact
import anywidget

# Sklearn imports
from sklearn.metrics import roc_curve, auc, make_scorer, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, precision_recall_curve, r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression, ElasticNet
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn import set_config
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_selector, make_column_transformer, ColumnTransformer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy.stats import kurtosis, skew, zscore, randint, uniform
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.base import clone  # <-- add this import at the top of the module

# Miscelaneous
from tqdm import tqdm
from colorama import Fore, Back, Style
import warnings
import importlib
import logging
from typing import List, Dict, Tuple, Union, Optional
import time
import sys
import pickle
import joblib
import json
from datetime import datetime 

def calculate_target_metric(df, multiplier, numerator, denominator):
    """Calculate target metric with proper handling of edge cases."""
    result = multiplier * df[numerator] / df[denominator].replace(0, np.nan)
    return np.round(result, 2)

def split_and_prepare_data(data, bins, variables, indicators, target_var, multiplier, test_size=0.4):
    """
    Split data into train/test sets first, then preprocess each separately to avoid data leakage.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data frame
    bins : tuple
        Tuple of (octroi_bins, efx_bins) for variable binning
    variables : list
        List of variables to use in modeling
    indicators : list
        Indicator columns for calculations
    target_var : str
        Name of the target variable to compute
    multiplier : float
        Multiplier for the target calculation
    test_size : float, optional
        Proportion of data to use for testing (default=0.2)
        
    Returns:
    --------
    tuple
        (train_data, test_data, var_reg, var_reg_extended)
    """
    from sklearn.model_selection import train_test_split
    
    # First, filter to only booked data (if needed for your use case)
    booked_data = data[data['status_name'] == 'booked'].copy()
    
    # Split data at individual record level BEFORE any grouping
    train_raw, test_raw = train_test_split(
        booked_data, 
        test_size=test_size, 
        random_state=42
    )
    
    print(f"Training set: {len(train_raw)} records")
    print(f"Test set: {len(test_raw)} records")
    
    # Define regression variables dynamically
    var_reg = [
        f'{variables[1]}',
        f'9-{variables[0]}',
        f'(9-{variables[0]}) x {variables[1]}'
    ]
    
    # Additional potentially useful variables for testing
    var_reg_extended = [
        f'{variables[1]}^2', 
        f'(9-{variables[0]})^2',
        f'{variables[1]}', 
        f'9-{variables[0]}',
        f'(9-{variables[0]}) x {variables[1]}'
    ]
    
    # Process train and test data separately
    train_processed = process_dataset(
        train_raw, bins, variables, indicators, target_var, multiplier, var_reg
    )
    
    test_processed = process_dataset(
        test_raw, bins, variables, indicators, target_var, multiplier, var_reg
    )
    
    return train_processed, test_processed, var_reg, var_reg_extended

def process_dataset(data, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold=3):
    """
    Process a single dataset (either train or test) with proper outlier handling.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Dataset to process
    bins : tuple
        Tuple of (octroi_bins, efx_bins) for variable binning
    variables : list
        List of variables to use in modeling
    indicators : list
        Indicator columns for calculations
    target_var : str
        Name of the target variable to compute
    multiplier : float
        Multiplier for the target calculation
    var_reg : list
        Regression variables to use for grouping
    z_threshold : float, optional
        Z-score threshold for outlier detection (default=3)
        
    Returns:
    --------
    pd.DataFrame
        Processed data ready for modeling
    """
    from scipy.stats import zscore
    from sklearn.ensemble import IsolationForest
    
    octroi_bins, efx_bins = bins
    
    # Preprocess data with existing function
    processed_data = preprocess_data(
        data, octroi_bins, efx_bins, variables, indicators, target_var, multiplier
    )
    
    # Group by variables for aggregation (this mimics your original approach)
    groupby_vars = list(set(variables + var_reg))
    
    # Aggregate data - single operation
    processed_data = (processed_data
                     .groupby(groupby_vars)
                     .sum()
                     .reset_index()
                     .sort_values(by=variables)
    )
    
    # Calculate target variable
    processed_data[target_var] = calculate_target_metric(
        processed_data, multiplier, 'todu_30ever_h6', 'todu_amt_pile_h6'
    )
    
    # Clean data - chain operations for better readability
    processed_data = (processed_data
                     .dropna()
                     .loc[lambda df: np.abs(zscore(df[target_var])) < z_threshold]
    )
    
    # Clean data - chain operations for better readability
    #processed_data = (processed_data
    #                 .dropna()
    #                 .loc[lambda df: np.abs(zscore(df[target_var])) < z_threshold]
    #                 .loc[lambda df: df[target_var] > 0]
    #)
    
    # isolation_forest = IsolationForest(random_state=42, contamination='auto')
    # inlier_mask = isolation_forest.fit_predict(processed_data) == 1
    # processed_data = processed_data.loc[inlier_mask].copy()
    
    return processed_data

def evaluate_models_for_aggregated_data(X_train, X_test, y_train, y_test, var_reg):
    """
    Evaluate multiple regression models optimized for aggregated datasets.
    Uses Test R² as primary metric since aggregated data has limited samples.
    """
    # Define models to evaluate
    models = {
        'Linear Regression': LinearRegression(fit_intercept=False),
        'Ridge (α=0.3)': Ridge(alpha=0.3, fit_intercept=False),
        'Ridge (α=0.5)': Ridge(alpha=0.5, fit_intercept=False),
        'Ridge (α=0.8)': Ridge(alpha=0.8, fit_intercept=False),
        'Ridge (α=1.2)': Ridge(alpha=1.2, fit_intercept=False),
        'Lasso (α=0.01)': Lasso(alpha=0.01, fit_intercept=False),
        'Lasso (α=0.05)': Lasso(alpha=0.05, fit_intercept=False),
        'Lasso (α=0.1)': Lasso(alpha=0.1, fit_intercept=False),
        'ElasticNet (α=0.05, l1=0.5)': ElasticNet(alpha=0.05, l1_ratio=0.5, fit_intercept=False),
        'ElasticNet (α=0.1, l1=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False),
        #'Random Forest (estimators=100)': RandomForestRegressor(n_estimators=100, random_state=42),
        #'Random Forest (estimators=300)': RandomForestRegressor(n_estimators=300, random_state=42),
        #'Random Forest (estimators=500)': RandomForestRegressor(n_estimators=500, random_state=42),
        #'Gradient Boosting (estimators=100)': GradientBoostingRegressor(n_estimators=100, random_state=42),
        #'Gradient Boosting (estimators=300)': GradientBoostingRegressor(n_estimators=300, random_state=42),
        #'Gradient Boosting (estimators=500)': GradientBoostingRegressor(n_estimators=500, random_state=42)
    }
    
    results = []
    best_model = None
    best_test_score = -float('inf')
    
    print("\n" + "="*100)
    print("MODEL EVALUATION FOR AGGREGATED DATA")
    print("="*100)
    print(f"{'Model':<35} {'Train R²':<10} {'Test R²':<10} {'MSE':<12} {'MAE':<10} {'RMSE':<10}")
    print("-"*100)
    
    for name, model in models.items():
        # Fit model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mse)
        
        # Check for overfitting
        overfitting_warning = "⚠️" if (train_r2 - test_r2) > 0.15 else ""
        
        # Print results
        print(f"{name:<35} {train_r2:.4f}     {test_r2:.4f}     {mse:.4f}      {mae:.4f}     {rmse:.4f} {overfitting_warning}")
        
        # Store results
        results.append({
            'Model': name,
            'Train R²': train_r2,
            'Test R²': test_r2,
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'Overfit Gap': train_r2 - test_r2
        })
        
        # Update best model based on Test R²
        if test_r2 > best_test_score:
            best_test_score = test_r2
            best_model = model
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    # Print best model summary
    print("-"*100)
    best_result = results_df.iloc[0]
    print(f"\n✅ BEST MODEL: {best_result['Model']}")
    print(f"   Test R²: {best_result['Test R²']:.4f}")
    print(f"   Train R²: {best_result['Train R²']:.4f}")
    print(f"   RMSE: {best_result['RMSE']:.4f}")
    print(f"   Overfitting Gap: {best_result['Overfit Gap']:.4f}")
    
    # Print coefficients if available
    if hasattr(best_model, 'coef_'):
        print(f"\n   Coefficients:")
        for var, coef in zip(var_reg, best_model.coef_):
            print(f"     {var}: {coef:.4f}")
    
    return best_model, results_df


def optimize_feature_combinations(train_data, test_data, y_train, y_test, var_reg_extended, best_model_class):
    """
    Test different feature combinations using the best model type from Step 1.
    
    Parameters:
    -----------
    best_model_class : sklearn model instance
        The best performing model from Step 1 to use for testing feature sets
    """
    # Define comprehensive feature sets to test
    feature_sets = {
        'Basic (3 features)': [
            var_reg_extended[2],  # var1
            var_reg_extended[3],  # 9-var0
            var_reg_extended[4]   # interaction
        ],
        'Extended (5 features)': var_reg_extended,  # All features
        'Interaction Only': [var_reg_extended[4]],  # Just interaction
        'Main Effects': [
            var_reg_extended[2],  # var1
            var_reg_extended[3]   # 9-var0
        ],
        'With var1²': [
            var_reg_extended[0],  # var1^2
            var_reg_extended[2],  # var1
            var_reg_extended[3],  # 9-var0
            var_reg_extended[4]   # interaction
        ],
        'With (9-var0)²': [
            var_reg_extended[1],  # (9-var0)^2
            var_reg_extended[2],  # var1
            var_reg_extended[3],  # 9-var0
            var_reg_extended[4]   # interaction
        ],
        'Quadratic + Interaction': [
            var_reg_extended[0],  # var1^2
            var_reg_extended[1],  # (9-var0)^2
            var_reg_extended[4]   # interaction
        ]
    }
    
    print("\n" + "="*100)
    print("FEATURE SET OPTIMIZATION USING BEST MODEL FROM STEP 1")
    print("="*100)
    print(f"Using model: {type(best_model_class).__name__}")
    if hasattr(best_model_class, 'get_params'):
        params = best_model_class.get_params()
        relevant_params = {k: v for k, v in params.items() if k in ['alpha', 'l1_ratio', 'fit_intercept']}
        if relevant_params:
            print(f"Parameters: {relevant_params}")
    print("-"*100)
    print(f"{'Feature Set':<30} {'N Features':<12} {'Train R²':<10} {'Test R²':<10} {'RMSE':<10} {'Gap':<10}")
    print("-"*100)
    
    best_test_score = -float('inf')
    best_feature_set = None
    best_model = None
    best_set_name = None
    results = []
    
    for set_name, features in feature_sets.items():
        X_train = train_data[features]
        X_test = test_data[features]
        
        # Clone the best model from Step 1 with same parameters
        from sklearn.base import clone
        model = clone(best_model_class)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, model.predict(X_train))
        test_r2 = r2_score(y_test, model.predict(X_test))
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        gap = train_r2 - test_r2
        
        # Store results
        results.append({
            'Feature Set': set_name,
            'Num Features': len(features),
            'Train R²': train_r2,
            'Test R²': test_r2,
            'RMSE': rmse,
            'Overfit Gap': gap
        })
        
        # Print results
        overfitting_warning = "⚠️" if gap > 0.15 else ""
        print(f"{set_name:<30} {len(features):<12} {train_r2:.4f}     {test_r2:.4f}     {rmse:.4f}     {gap:.4f} {overfitting_warning}")
        
        # Update best model based on Test R²
        if test_r2 > best_test_score:
            best_test_score = test_r2
            best_feature_set = features
            best_model = model
            best_set_name = set_name
    
    results_df = pd.DataFrame(results)
    
    print("-"*100)
    print(f"\n✅ BEST FEATURE SET: {best_set_name}")
    print(f"   Features: {', '.join(best_feature_set)}")
    print(f"   Test R²: {best_test_score:.4f}")
    
    # Print coefficients for best model
    if hasattr(best_model, 'coef_'):
        print(f"\n   Coefficients:")
        for var, coef in zip(best_feature_set, best_model.coef_):
            print(f"     {var}: {coef:.4f}")
    
    return best_feature_set, best_model, results_df


def save_model_for_production(model, features, metadata, base_path='saved_models'):
    """
    Save trained model with comprehensive metadata for production use.
    Optimized for aggregated data models.
    """
    # Create directory structure
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create subdirectory for this model version
    version_path = Path(base_path) / f"model_{timestamp}"
    version_path.mkdir(exist_ok=True)
    
    # Save model
    model_path = version_path / "model.pkl"
    joblib.dump(model, model_path)
    
    # Enhance metadata
    metadata_enhanced = {
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        'features': features,
        'num_features': len(features),
        'aggregated_data': True,  # Flag indicating this is for aggregated data
        **metadata
    }
    
    # Save metadata
    metadata_path = version_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata_enhanced, f, indent=4, default=str)
    
    # Save feature list
    features_path = version_path / "features.txt"
    with open(features_path, 'w') as f:
        f.write("# Features used in model (order matters)\n")
        f.write(f"# Model: {type(model).__name__}\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Test R²: {metadata.get('test_r2', 'N/A'):.4f}\n\n")
        for i, feature in enumerate(features):
            f.write(f"{i+1}. {feature}\n")
    
    # Save model summary
    summary_path = version_path / "model_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Number of Features: {len(features)}\n")
        f.write(f"Aggregated Data: Yes\n\n")
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*30 + "\n")
        f.write(f"Training R²: {metadata.get('train_r2', 'N/A'):.4f}\n")
        f.write(f"Test R²: {metadata.get('test_r2', 'N/A'):.4f}\n")
        f.write(f"Test RMSE: {metadata.get('test_rmse', 'N/A'):.4f}\n")
        f.write(f"Test MAE: {metadata.get('test_mae', 'N/A'):.4f}\n")
        f.write(f"Training Samples: {metadata.get('train_samples', 'N/A')}\n")
        f.write(f"Test Samples: {metadata.get('test_samples', 'N/A')}\n")
        
        if hasattr(model, 'coef_'):
            f.write("\nMODEL COEFFICIENTS:\n")
            f.write("-"*30 + "\n")
            for feature, coef in zip(features, model.coef_):
                f.write(f"{feature}: {coef:.6f}\n")
    
    print("\n" + "="*60)
    print("MODEL SAVED SUCCESSFULLY")
    print("="*60)
    print(f"📁 Directory: {version_path}")
    print(f"   - Model: model.pkl")
    print(f"   - Metadata: metadata.json")
    print(f"   - Features: features.txt")
    print(f"   - Summary: model_summary.txt")
    
    return str(version_path)


def load_model_for_prediction(model_path):
    """
    Load a saved model for making predictions on new aggregated data.
    """
    model_dir = Path(model_path)
    
    # Load model
    model = joblib.load(model_dir / "model.pkl")
    
    # Load metadata
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    features = metadata['features']
    
    print(f"✅ Model loaded successfully")
    print(f"   Type: {metadata['model_type']}")
    print(f"   Features: {metadata['num_features']}")
    print(f"   Test R²: {metadata.get('test_r2', 'N/A'):.4f}")
    
    return model, metadata, features


def plot_3d_surface(model, train_data, test_data, variables, target_var, features):
    """
    Create a 3D plot showing model predictions as a surface with actual data points.
    """
    try:
        # Extract variables
        var0, var1 = variables
        print(f"Creating 3D plot for {var0}, {var1}, {target_var}")
       
        # Create mesh grid for predictions
        n_points = 20
        x_min, x_max = min(train_data[var0].min(), test_data[var0].min()), max(train_data[var0].max(), test_data[var0].max())
        y_min, y_max = min(train_data[var1].min(), test_data[var1].min()), max(train_data[var1].max(), test_data[var1].max())
       
        x_range = np.linspace(x_min, x_max, n_points)
        y_range = np.linspace(y_min, y_max, n_points)
       
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
       
        # Create inputs for model
        points = []
        for i in range(n_points):
            for j in range(n_points):
                points.append([x_range[j], y_range[i]])
       
        mesh_df = pd.DataFrame(points, columns=[var0, var1])
       
        # Create features for model
        feature_df = pd.DataFrame(index=mesh_df.index)
       
        # Transform each feature explicitly
        for feature in features:
            if feature == f'{var1}':
                feature_df[feature] = mesh_df[var1]
            elif feature == f'9-{var0}':
                feature_df[feature] = 9 - mesh_df[var0]
            elif feature == f'(9-{var0}) x {var1}':
                feature_df[feature] = (9 - mesh_df[var0]) * mesh_df[var1]
            elif feature == f'{var1}^2':
                feature_df[feature] = mesh_df[var1] ** 2
            elif feature == f'(9-{var0})^2':
                feature_df[feature] = (9 - mesh_df[var0]) ** 2
            else:
                print(f"Warning: No rule for feature {feature}")
       
        # Make predictions
        z_pred = model.predict(feature_df)
       
        # Reshape for the surface
        z_mesh = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(n_points):
                z_mesh[i, j] = z_pred[i * n_points + j]
       
        # Create the 3D plot
        fig = go.Figure(data=[
            # Training data
            go.Scatter3d(
                x=train_data[var0],
                y=train_data[var1],
                z=train_data[target_var],
                mode='markers',
                marker=dict(
                    size=5,
                    color='blue',
                    opacity=0.7
                ),
                name='Training Data'
            ),
            # Test data
            go.Scatter3d(
                x=test_data[var0],
                y=test_data[var1],
                z=test_data[target_var],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',
                    opacity=0.7
                ),
                name='Test Data'
            ),
            # Prediction surface
            go.Surface(
                z=z_mesh,
                x=x_mesh,
                y=y_mesh,
                colorscale='Viridis',
                opacity=0.8,
                name='Model Predictions'
            )
        ])
       
        # Update layout
        fig.update_layout(
            title=f"{target_var} vs {var0} and {var1} with Model Predictions",
            scene=dict(
                xaxis_title=var0,
                yaxis_title=var1,
                zaxis_title=target_var,
                aspectratio=dict(x=1, y=1, z=0.8)
            ),
            width=1000,
            height=800
        )
       
        return fig
       
    except Exception as e:
        print(f"Error creating 3D plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None