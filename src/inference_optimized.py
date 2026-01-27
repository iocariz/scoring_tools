"""
Optimized inference module for fraud detection system.
Key improvements:
- Consolidated imports
- Removed redundant operations
- Improved memory efficiency
- Better error handling
- More maintainable code structure
"""

# Standard library imports
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from scipy.stats import zscore

# Scikit-learn imports
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Project imports
from src.models import preprocess_data, transform_variables
from src.utils import calculate_b2_ever_h6
from src.constants import (
    Columns, StatusName, Suffixes,
    DEFAULT_TEST_SIZE, DEFAULT_Z_THRESHOLD, DEFAULT_RANDOM_STATE
)
from src import styles


# Module-specific constants
DEFAULT_N_POINTS_3D = 20


class HurdleRegressor(BaseEstimator, RegressorMixin):
    """
    Hurdle Regression for zero-inflated data.
    
    Two-stage model:
    1. Classification: Predict if value is zero or non-zero (logistic regression)
    2. Regression: Predict magnitude for non-zero values (linear regression)
    
    This is ideal when you have many zero values that create a flat regression surface.
    
    Parameters:
    -----------
    classifier : sklearn classifier, optional
        Model for predicting zero vs non-zero (default: LogisticRegression)
    regressor : sklearn regressor, optional
        Model for predicting non-zero values (default: Ridge)
    zero_threshold : float, optional
        Values below this are considered "zero" (default: 1e-10)
    
    Attributes:
    -----------
    classifier_ : fitted classifier
    regressor_ : fitted regressor
    
    Example:
    --------
    >>> from sklearn.linear_model import Ridge, LogisticRegression
    >>> hurdle = HurdleRegressor(
    ...     classifier=LogisticRegression(max_iter=1000),
    ...     regressor=Ridge(alpha=0.5, fit_intercept=False)
    ... )
    >>> hurdle.fit(X_train, y_train, sample_weight=weights_train)
    >>> predictions = hurdle.predict(X_test)
    """
    
    def __init__(self, classifier=None, regressor=None, zero_threshold=1e-10):
        self.classifier = classifier
        self.regressor = regressor
        self.zero_threshold = zero_threshold
    
    def fit(self, X, y, sample_weight=None):
        """
        Fit the hurdle model.
        
        Args:
            X: Features
            y: Target variable
            sample_weight: Optional sample weights
        
        Returns:
            self
        """
        # Initialize models if not provided
        if self.classifier is None:
            self.classifier_ = LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE)
        else:
            from sklearn.base import clone
            self.classifier_ = clone(self.classifier)
        
        if self.regressor is None:
            self.regressor_ = Ridge(alpha=0.5, fit_intercept=False)
        else:
            from sklearn.base import clone
            self.regressor_ = clone(self.regressor)
        
        # Create binary target: 0 if zero, 1 if non-zero
        y_binary = (np.abs(y) > self.zero_threshold).astype(int)
        
        # Stage 1: Fit classifier (zero vs non-zero)
        self.classifier_.fit(X, y_binary, sample_weight=sample_weight)
        
        # Stage 2: Fit regressor on non-zero values only
        non_zero_mask = y_binary == 1
        
        if non_zero_mask.sum() > 0:
            X_nonzero = X[non_zero_mask]
            y_nonzero = y[non_zero_mask]
            weights_nonzero = sample_weight[non_zero_mask] if sample_weight is not None else None
            
            self.regressor_.fit(X_nonzero, y_nonzero, sample_weight=weights_nonzero)
        else:
            # Fallback: if no non-zero values, just fit on all data
            self.regressor_.fit(X, y, sample_weight=sample_weight)
        
        return self
    
    def predict(self, X):
        """
        Predict using the hurdle model.
        
        Process:
        1. Predict probability of non-zero
        2. Predict magnitude if non-zero
        3. Combine: P(non-zero) * predicted_magnitude
        
        Args:
            X: Features
        
        Returns:
            predictions: Array of predictions
        """
        # Stage 1: Predict probability of being non-zero
        prob_nonzero = self.classifier_.predict_proba(X)[:, 1]
        
        # Stage 2: Predict magnitude for all observations
        magnitude = self.regressor_.predict(X)
        
        # Combine: Expected value = P(non-zero) * E(Y | Y > 0)
        predictions = prob_nonzero * magnitude
        
        return predictions
    
    def predict_binary(self, X):
        """Predict binary outcome (zero vs non-zero) only."""
        return self.classifier_.predict(X)
    
    def predict_magnitude(self, X):
        """Predict magnitude (without zero-inflation adjustment)."""
        return self.regressor_.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'zero_threshold': self.zero_threshold
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


def calculate_target_metric(
    df: pd.DataFrame, 
    multiplier: float, 
    numerator: str, 
    denominator: str
) -> np.ndarray:
    """
    Calculate target metric with proper handling of edge cases.
    
    Args:
        df: DataFrame containing the data
        multiplier: Multiplier for the target calculation
        numerator: Column name for numerator
        denominator: Column name for denominator
    
    Returns:
        Calculated metric rounded to 2 decimals
    """
    result = multiplier * df[numerator] / df[denominator].replace(0, np.nan)
    return np.round(result, 2)


def _generate_regression_variables(variables: List[str]) -> Tuple[List[str], List[str]]:
    """
    Generate regression variable names dynamically.
    
    Args:
        variables: List of base variable names
    
    Returns:
        Tuple of (var_reg, var_reg_extended)
    """
    var_reg = [
        f'{variables[1]}',
        f'9-{variables[0]}',
        f'(9-{variables[0]}) x {variables[1]}'
    ]
    
    var_reg_extended = [
        f'{variables[1]}^2', 
        f'(9-{variables[0]})^2',
        f'{variables[1]}', 
        f'9-{variables[0]}',
        f'(9-{variables[0]}) x {variables[1]}'
    ]
    
    return var_reg, var_reg_extended


def split_and_prepare_data(
    data: pd.DataFrame,
    bins: Tuple,
    variables: List[str],
    indicators: List[str],
    target_var: str,
    multiplier: float,
    test_size: float = DEFAULT_TEST_SIZE
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Split data into train/test sets and preprocess each separately to avoid data leakage.
    
    Args:
        data: Input DataFrame
        bins: Tuple of (octroi_bins, efx_bins) for variable binning
        variables: List of variables to use in modeling
        indicators: Indicator columns for calculations
        target_var: Name of the target variable to compute
        multiplier: Multiplier for the target calculation
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (train_data, test_data, var_reg, var_reg_extended)
    """
    # Filter to booked data only
    booked_data = data[data[Columns.STATUS_NAME] == StatusName.BOOKED.value].copy()
    
    # Split at record level before grouping
    train_raw, test_raw = train_test_split(
        booked_data, 
        test_size=test_size, 
        random_state=DEFAULT_RANDOM_STATE
    )
    
    logger.info(f"Training set: {len(train_raw):,} records")
    logger.info(f"Test set: {len(test_raw):,} records")
    
    # Generate regression variables
    var_reg, var_reg_extended = _generate_regression_variables(variables)
    
    # Process datasets separately
    train_processed = process_dataset(
        train_raw, bins, variables, indicators, target_var, multiplier, var_reg
    )
    
    test_processed = process_dataset(
        test_raw, bins, variables, indicators, target_var, multiplier, var_reg
    )
    
    return train_processed, test_processed, var_reg, var_reg_extended


def process_dataset(
    data: pd.DataFrame,
    bins: Tuple,
    variables: List[str],
    indicators: List[str],
    target_var: str,
    multiplier: float,
    var_reg: List[str],
    z_threshold: float = DEFAULT_Z_THRESHOLD
) -> pd.DataFrame:
    """
    Process a single dataset with proper outlier handling.
    
    Args:
        data: Dataset to process
        bins: Tuple of (octroi_bins, efx_bins) for variable binning
        variables: List of variables to use in modeling
        indicators: Indicator columns for calculations
        target_var: Name of the target variable to compute
        multiplier: Multiplier for the target calculation
        var_reg: Regression variables to use for grouping
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        Processed data ready for modeling with 'n_observations' column for weighting
    """
    octroi_bins, efx_bins = bins
    
    # Preprocess data - Use direct transformation instead of grid merge to preserve data
    # Filter to relevant columns + variables + indicators
    # Note: data is already filtered to 'booked' in split_and_prepare_data but we ensure variables exist
    processed_data = data.copy()
    
    # Transform variables to create regression features
    processed_data = transform_variables(processed_data, variables)
    
    # Aggregate data - combine variables for groupby to avoid duplication
    groupby_vars = list(set(variables + var_reg))
    
    # Track number of observations per group for weighted regression
    # Perform groupby
    grouped = processed_data.groupby(groupby_vars, observed=True)
    
    # Aggregate sums
    # Filter to numeric columns only for summation
    numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
    agg_dict = {col: 'sum' for col in numeric_cols if col not in groupby_vars}
    processed_data = grouped.agg(agg_dict)
    
    # Add n_observations (size of each group)
    processed_data['n_observations'] = grouped.size()
    
    # Reset index and sort
    processed_data = processed_data.reset_index().sort_values(by=variables)
    
    # Calculate target variable
    processed_data[target_var] = calculate_target_metric(
        processed_data, multiplier, 'todu_30ever_h6', 'todu_amt_pile_h6'
    )
    
    # Clean data with method chaining
    processed_data = (
        processed_data
        .dropna()
        .loc[lambda df: np.abs(zscore(df[target_var])) < z_threshold]
        .copy()  # Create a copy to avoid SettingWithCopyWarning
    )
    
    return processed_data


def evaluate_models_for_aggregated_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    var_reg: List[str],
    weights_train: Optional[pd.Series] = None,
    weights_test: Optional[pd.Series] = None,
    include_hurdle: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Evaluate multiple regression models optimized for aggregated datasets.
    Uses Test R² as primary metric with optional sample weighting.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        var_reg: Regression variables used
        weights_train: Optional weights for training samples (e.g., n_observations)
        weights_test: Optional weights for test samples (e.g., n_observations)
        include_hurdle: Whether to include Hurdle models for zero-inflated data
    
    Returns:
        Tuple of (results_df, best_model_dict)
    """
    # Calculate zero proportion for information
    zero_prop_train = (np.abs(y_train) < 1e-10).mean()
    zero_prop_test = (np.abs(y_test) < 1e-10).mean()
    
    logger.info(f"Data characteristics:")
    logger.info(f"  Zero proportion - Train: {zero_prop_train:.1%}, Test: {zero_prop_test:.1%}")
    
    if zero_prop_train > 0.1 and include_hurdle:
        logger.warning(f"High zero proportion detected - Hurdle models recommended!")
    
    # Define models with better organization
    alpha_values = {
        'Ridge': [0.3, 0.5, 0.8, 1.2],
        'Lasso': [0.01, 0.05, 0.1],
        'ElasticNet': [(0.05, 0.5), (0.1, 0.5)]
    }
    
    models = {'Linear Regression': LinearRegression(fit_intercept=False)}
    
    # Add Ridge models
    for alpha in alpha_values['Ridge']:
        models[f'Ridge (α={alpha})'] = Ridge(alpha=alpha, fit_intercept=False)
    
    # Add Lasso models
    for alpha in alpha_values['Lasso']:
        models[f'Lasso (α={alpha})'] = Lasso(alpha=alpha, fit_intercept=False)
    
    # Add ElasticNet models
    for alpha, l1_ratio in alpha_values['ElasticNet']:
        models[f'ElasticNet (α={alpha}, l1={l1_ratio})'] = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False
        )
    
    # Add Hurdle models if requested
    if include_hurdle:
        logger.info(f"Including Hurdle models for zero-inflated data")
        
        # Hurdle with different regression backbones
        for alpha in [0.3, 0.5, 0.8]:
            models[f'Hurdle-Ridge (α={alpha})'] = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
                regressor=Ridge(alpha=alpha, fit_intercept=False)
            )
        
        for alpha in [0.01, 0.05]:
            models[f'Hurdle-Lasso (α={alpha})'] = HurdleRegressor(
                classifier=LogisticRegression(max_iter=1000, random_state=DEFAULT_RANDOM_STATE),
                regressor=Lasso(alpha=alpha, fit_intercept=False)
            )
    
    # Evaluate models
    results = []

    logger.info("Evaluating models...")
    if weights_train is not None:
        logger.info(f"Using weighted regression with sample weights")
        logger.info(f"Weight range - Train: [{weights_train.min():.0f}, {weights_train.max():.0f}], "
              f"Test: [{weights_test.min():.0f}, {weights_test.max():.0f}]")
    logger.info("-" * 80)
    
    for name, model in models.items():
        try:
            # Train model with optional weights
            model.fit(X_train, y_train, sample_weight=weights_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate metrics (weighted if weights provided)
            train_r2 = r2_score(y_train, y_train_pred, sample_weight=weights_train)
            test_r2 = r2_score(y_test, y_test_pred, sample_weight=weights_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred, sample_weight=weights_test))
            test_mae = mean_absolute_error(y_test, y_test_pred, sample_weight=weights_test)
            
            # Additional metrics for hurdle models
            if isinstance(model, HurdleRegressor):
                # Classification accuracy on zero vs non-zero
                y_test_binary = (np.abs(y_test) > 1e-10).astype(int)
                binary_acc = (model.predict_binary(X_test) == y_test_binary).mean()
            else:
                binary_acc = None
            
            # Store results
            results.append({
                'Model': name,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Test RMSE': test_rmse,
                'Test MAE': test_mae,
                'Overfit': train_r2 - test_r2,
                'Zero/NonZero Acc': binary_acc,
                'model_object': model
            })
        except Exception as e:
            logger.error(f"Failed to train {name}:")
            logger.exception("Training failed")
            continue
    
    if not results:
        raise RuntimeError("All models failed to train. Check your data for NaNs, infinite values, or shape mismatches.")

    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
    
    # Display top models
    display_cols = ['Model', 'Train R²', 'Test R²', 'Test RMSE', 'Overfit']
    if include_hurdle:
        display_cols.append('Zero/NonZero Acc')

    logger.info("Top 10 Models by Test R²:" if len(results_df) > 10 else "All Models by Test R²:")
    logger.info(f"\n{results_df[display_cols].head(10).to_string()}")
    
    # Get best model
    best_idx = results_df['Test R²'].idxmax()
    best_model_info = {
        'model': results_df.loc[best_idx, 'model_object'],
        'name': results_df.loc[best_idx, 'Model'],
        'test_r2': results_df.loc[best_idx, 'Test R²'],
        'train_r2': results_df.loc[best_idx, 'Train R²'],
        'test_rmse': results_df.loc[best_idx, 'Test RMSE'],
        'test_mae': results_df.loc[best_idx, 'Test MAE'],
        'weighted': weights_train is not None,
        'is_hurdle': isinstance(results_df.loc[best_idx, 'model_object'], HurdleRegressor)
    }
    
    logger.info(f"Best Model: {best_model_info['name']}")
    logger.info(f"   Test R²: {best_model_info['test_r2']:.4f}")
    if weights_train is not None:
        logger.info(f"   (Weighted by sample size)")
    if best_model_info['is_hurdle']:
        logger.info(f"   (Hurdle model - handles zero-inflation)")
    
    return results_df.drop('model_object', axis=1), best_model_info


def save_model_with_metadata(
    model,
    features: List[str],
    metadata: Dict,
    base_path: str = 'models'
) -> str:
    """
    Save trained model with comprehensive metadata for production use.
    
    Args:
        model: Trained model object
        features: List of feature names
        metadata: Dictionary containing model metadata
        base_path: Base directory for saving models
    
    Returns:
        Path to saved model directory
    """
    # Create directory structure
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create version directory
    version_path = base_path / f"model_{timestamp}"
    version_path.mkdir(exist_ok=True)
    
    # Save model
    model_path = version_path / "model.pkl"
    joblib.dump(model, model_path, compress=3)  # Add compression
    
    # Enhance metadata
    metadata_enhanced = {
        'timestamp': timestamp,
        'model_type': type(model).__name__,
        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
        'features': features,
        'num_features': len(features),
        'aggregated_data': True,
        **metadata
    }
    
    # Save metadata
    with open(version_path / "metadata.json", 'w') as f:
        json.dump(metadata_enhanced, f, indent=2, default=str)
    
    # Save features
    with open(version_path / "features.txt", 'w') as f:
        f.write(f"# Features for {type(model).__name__}\n")
        f.write(f"# Timestamp: {timestamp}\n")
        f.write(f"# Test R²: {metadata.get('test_r2', 'N/A'):.4f}\n\n")
        for i, feature in enumerate(features, 1):
            f.write(f"{i}. {feature}\n")
    
    # Save model summary
    _save_model_summary(version_path, model, features, metadata, timestamp)
    
    logger.info("=" * 60)
    logger.info("MODEL SAVED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Directory: {version_path}")
    logger.info(f"   - Model: model.pkl")
    logger.info(f"   - Metadata: metadata.json")
    logger.info(f"   - Features: features.txt")
    logger.info(f"   - Summary: model_summary.txt")
    
    return str(version_path)


def _save_model_summary(
    version_path: Path,
    model,
    features: List[str],
    metadata: Dict,
    timestamp: str
) -> None:
    """Helper function to save model summary."""
    with open(version_path / "model_summary.txt", 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Number of Features: {len(features)}\n")
        f.write(f"Aggregated Data: Yes\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        for key in ['train_r2', 'test_r2', 'test_rmse', 'test_mae']:
            value = metadata.get(key, 'N/A')
            display_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
            f.write(f"{key.replace('_', ' ').title()}: {display_value}\n")
        
        f.write(f"Training Samples: {metadata.get('train_samples', 'N/A')}\n")
        f.write(f"Test Samples: {metadata.get('test_samples', 'N/A')}\n")
        
        if hasattr(model, 'coef_'):
            f.write("\nMODEL COEFFICIENTS:\n")
            f.write("-" * 30 + "\n")
            for feature, coef in zip(features, model.coef_):
                f.write(f"{feature}: {coef:.6f}\n")


def load_model_for_prediction(model_path: str) -> Tuple[object, Dict, List[str]]:
    """
    Load a saved model for making predictions.
    
    Args:
        model_path: Path to model directory
    
    Returns:
        Tuple of (model, metadata, features)
    """
    model_dir = Path(model_path)
    
    # Load model
    model = joblib.load(model_dir / "model.pkl")
    
    # Load metadata
    with open(model_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    features = metadata['features']
    
    logger.info("Model loaded successfully")
    logger.info(f"   Type: {metadata['model_type']}")
    logger.info(f"   Features: {metadata['num_features']}")
    test_r2 = metadata.get('test_r2', 'N/A')
    logger.info(f"   Test R²: {test_r2:.4f}" if isinstance(test_r2, (int, float)) else f"   Test R²: {test_r2}")
    
    return model, metadata, features


def _create_feature_dataframe(
    mesh_df: pd.DataFrame,
    features: List[str],
    var0: str,
    var1: str
) -> pd.DataFrame:
    """
    Create feature DataFrame for predictions.
    
    Args:
        mesh_df: DataFrame with mesh grid points
        features: List of feature names
        var0: First variable name
        var1: Second variable name
    
    Returns:
        DataFrame with computed features
    """
    feature_df = pd.DataFrame(index=mesh_df.index)
    
    # Feature computation mapping
    feature_rules = {
        f'{var1}': lambda df: df[var1],
        f'9-{var0}': lambda df: 9 - df[var0],
        f'(9-{var0}) x {var1}': lambda df: (9 - df[var0]) * df[var1],
        f'{var1}^2': lambda df: df[var1] ** 2,
        f'(9-{var0})^2': lambda df: (9 - df[var0]) ** 2
    }
    
    for feature in features:
        if feature in feature_rules:
            feature_df[feature] = feature_rules[feature](mesh_df)
        else:
            warnings.warn(f"No rule defined for feature: {feature}")
    
    return feature_df


def plot_3d_surface(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    variables: List[str],
    target_var: str,
    features: List[str],
    n_points: int = DEFAULT_N_POINTS_3D
) -> Optional[go.Figure]:
    """
    Create a 3D plot showing model predictions as a surface with actual data points.
    
    Args:
        model: Trained model
        train_data: Training dataset
        test_data: Test dataset
        variables: List of variable names
        target_var: Target variable name
        features: List of feature names
        n_points: Number of points for mesh grid
    
    Returns:
        Plotly Figure object or None if error occurs
    """
    try:
        var0, var1 = variables
        logger.info(f"Creating 3D plot for {var0}, {var1}, {target_var}")
        
        # Create mesh grid for predictions
        x_min = min(train_data[var0].min(), test_data[var0].min())
        x_max = max(train_data[var0].max(), test_data[var0].max())
        y_min = min(train_data[var1].min(), test_data[var1].min())
        y_max = max(train_data[var1].max(), test_data[var1].max())
        
        x_range = np.linspace(x_min, x_max, n_points)
        y_range = np.linspace(y_min, y_max, n_points)
        
        x_mesh, y_mesh = np.meshgrid(x_range, y_range)
        
        # Create DataFrame for mesh points
        mesh_df = pd.DataFrame({
            var0: x_mesh.ravel(),
            var1: y_mesh.ravel()
        })
        
        # Create features for model
        feature_df = _create_feature_dataframe(mesh_df, features, var0, var1)
        
        # Make predictions and reshape
        z_pred = model.predict(feature_df)
        z_mesh = z_pred.reshape(n_points, n_points)
        
        # Create the 3D plot
        fig = go.Figure(data=[
            # Training data
            go.Scatter3d(
                x=train_data[var0],
                y=train_data[var1],
                z=train_data[target_var],
                mode='markers',
                marker=dict(size=5, color=styles.COLOR_ACCENT, opacity=0.7),
                name='Training Data'
            ),
            # Test data
            go.Scatter3d(
                x=test_data[var0],
                y=test_data[var1],
                z=test_data[target_var],
                mode='markers',
                marker=dict(size=5, color=styles.COLOR_RISK, opacity=0.7),
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
        styles.apply_plotly_style(
            fig,
            title=f"{target_var} vs {var0} and {var1} with Model Predictions",
            width=1000,
            height=800
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=var0,
                yaxis_title=var1,
                zaxis_title=target_var,
                aspectratio=dict(x=1, y=1, z=0.8)
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating 3D plot: {str(e)}")
        logger.exception("3D Plot generation failed")
        return None

"""
Complete End-to-End Pipeline Example
=====================================

This script demonstrates the full workflow from raw data to a deployed model,
including:
- Data preparation with train/test split
- Feature engineering with weighted observations
- Model evaluation (standard + hurdle models)
- Model saving with metadata
- Model loading and prediction
- 3D visualization

Author: Your Name
Date: 2024
"""

def inference_pipeline(
    data: pd.DataFrame,
    bins: tuple,
    variables: list,
    indicators: list,
    target_var: str,
    multiplier: float,
    test_size: float = 0.4,
    include_hurdle: bool = True,
    save_model: bool = True,
    model_base_path: str = 'models',
    create_visualizations: bool = True
):
    """
    Execute complete modeling pipeline from data to deployed model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw input data containing all records
    bins : tuple
        Tuple of (octroi_bins, efx_bins) for variable binning
    variables : list
        List of variable names for modeling (e.g., ['octroi_binned', 'efx_binned'])
    indicators : list
        List of indicator column names for calculations
    target_var : str
        Name of the target variable to predict
    multiplier : float
        Multiplier for target variable calculation
    test_size : float, optional
        Proportion of data for testing (default: 0.4)
    include_hurdle : bool, optional
        Whether to include Hurdle models (default: True)
    save_model : bool, optional
        Whether to save the best model (default: True)
    model_base_path : str, optional
        Base path for saving models (default: 'models')
    create_visualizations : bool, optional
        Whether to create 3D plots (default: True)
    
    Returns:
    --------
    dict
        Dictionary containing all pipeline outputs:
        - train_data: Processed training data
        - test_data: Processed test data
        - features: List of feature names
        - results_df: DataFrame with all model results
        - best_model_info: Dictionary with best model details
        - model_path: Path to saved model (if save_model=True)
        - visualization: Plotly figure (if create_visualizations=True)
    
    Example:
    --------
    >>> # Load your data
    >>> data = pd.read_feather('your_data.feather')
    >>> 
    >>> # Define parameters
    >>> bins = (octroi_bins, efx_bins)
    >>> variables = ['octroi_binned', 'efx_binned']
    >>> indicators = ['indicator1', 'indicator2']
    >>> 
    >>> # Run full pipeline
    >>> results = run_full_pipeline(
    ...     data=data,
    ...     bins=bins,
    ...     variables=variables,
    ...     indicators=indicators,
    ...     target_var='tpr_30ever',
    ...     multiplier=100
    ... )
    >>> 
    >>> # Access results
    >>> best_model = results['best_model_info']['model']
    >>> model_path = results['model_path']
    """
    
    logger.info("=" * 80)
    logger.info("FULL MODELING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Target variable: {target_var}")
    logger.info(f"Test size: {test_size:.1%}")
    logger.info(f"Include Hurdle models: {include_hurdle}")
    logger.info("=" * 80)
    
    # ========================================================================
    # STEP 1: DATA PREPARATION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("=" * 80)
    
    train_data, test_data, var_reg, var_reg_extended = split_and_prepare_data(
        data=data,
        bins=bins,
        variables=variables,
        indicators=indicators,
        target_var=target_var,
        multiplier=multiplier,
        test_size=test_size
    )
    
    logger.info(f"Training data shape: {train_data.shape}")
    logger.info(f"Test data shape: {test_data.shape}")
    logger.info(f"Features (var_reg): {var_reg}")
    
    # Check for n_observations column
    if 'n_observations' in train_data.columns:
        logger.info(f"Weights available for weighted regression")
        logger.info(f"  Train weights: [{train_data['n_observations'].min():.0f}, "
              f"{train_data['n_observations'].max():.0f}]")
        logger.info(f"  Test weights: [{test_data['n_observations'].min():.0f}, "
              f"{test_data['n_observations'].max():.0f}]")
    
    # ========================================================================
    # STEP 2: FEATURE EXTRACTION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 2: FEATURE EXTRACTION")
    logger.info("=" * 80)
    
    # Extract features and target
    X_train = train_data[var_reg]
    X_test = test_data[var_reg]
    y_train = train_data[target_var]
    y_test = test_data[target_var]
    
    # Extract weights if available
    weights_train = train_data['n_observations'] if 'n_observations' in train_data.columns else None
    weights_test = test_data['n_observations'] if 'n_observations' in test_data.columns else None
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    
    # Display target statistics
    # Display target statistics
    logger.info(f"Target variable statistics:")
    logger.info(f"  Train - Mean: {y_train.mean():.4f}, Std: {y_train.std():.4f}")
    logger.info(f"  Train - Min: {y_train.min():.4f}, Max: {y_train.max():.4f}")
    logger.info(f"  Test  - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}")
    logger.info(f"  Test  - Min: {y_test.min():.4f}, Max: {y_test.max():.4f}")
    
    # Check zero proportion
    zero_prop_train = (np.abs(y_train) < 1e-10).mean()
    zero_prop_test = (np.abs(y_test) < 1e-10).mean()
    logger.info(f"Zero proportion:")
    logger.info(f"  Train: {zero_prop_train:.1%}")
    logger.info(f"  Test: {zero_prop_test:.1%}")
    
    if zero_prop_train > 0.1:
        logger.warning(f"High zero proportion - Hurdle models recommended!")
    
    # ========================================================================
    # STEP 3: MODEL EVALUATION
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 3: MODEL EVALUATION")
    logger.info("=" * 80)
    
    results_df, best_model_info = evaluate_models_for_aggregated_data(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        var_reg=var_reg,
        weights_train=weights_train,
        weights_test=weights_test,
        include_hurdle=include_hurdle
    )
    
    # ========================================================================
    # STEP 4: MODEL ANALYSIS
    # ========================================================================
    logger.info("=" * 80)
    logger.info("STEP 4: MODEL ANALYSIS")
    logger.info("=" * 80)
    
    # Display detailed results for best model
    # Display detailed results for best model
    logger.info(f"Best Model Details:")
    logger.info(f"  Name: {best_model_info['name']}")
    logger.info(f"  Test R²: {best_model_info['test_r2']:.4f}")
    logger.info(f"  Train R²: {best_model_info['train_r2']:.4f}")
    logger.info(f"  Test RMSE: {best_model_info['test_rmse']:.4f}")
    logger.info(f"  Test MAE: {best_model_info['test_mae']:.4f}")
    logger.info(f"  Overfit: {best_model_info['train_r2'] - best_model_info['test_r2']:.4f}")
    
    if best_model_info.get('weighted', False):
        logger.info(f"  Weighted: Yes")
    
    if best_model_info.get('is_hurdle', False):
        logger.info(f"  Type: Hurdle Model (handles zero-inflation)")
    
    # Display model coefficients if available
    best_model = best_model_info['model']
    if hasattr(best_model, 'coef_'):

        logger.info(f"Model Coefficients:")
        for feature, coef in zip(var_reg, best_model.coef_):
            logger.info(f"  {feature}: {coef:.6f}")
    elif hasattr(best_model, 'regressor_') and hasattr(best_model.regressor_, 'coef_'):
        # Hurdle model
        logger.info(f"Hurdle Model - Regression Stage Coefficients:")
        for feature, coef in zip(var_reg, best_model.regressor_.coef_):
            logger.info(f"  {feature}: {coef:.6f}")
    
    # Compare top models
    logger.info("Top 5 Models Comparison:")
    display_cols = ['Model', 'Test R²', 'Test RMSE', 'Overfit']
    if 'Zero/NonZero Acc' in results_df.columns:
        display_cols.append('Zero/NonZero Acc')
    logger.info(f"\n{results_df[display_cols].head().to_string()}")
    
    # ========================================================================
    # STEP 5: MODEL SAVING
    # ========================================================================
    model_path = None
    if save_model:
        logger.info("=" * 80)
        logger.info("STEP 5: MODEL SAVING")
        logger.info("=" * 80)
        
        # Prepare metadata
        model_metadata = {
            'test_r2': best_model_info['test_r2'],
            'train_r2': best_model_info['train_r2'],
            'test_rmse': best_model_info['test_rmse'],
            'test_mae': best_model_info['test_mae'],
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'target_variable': target_var,
            'multiplier': multiplier,
            'weighted_regression': best_model_info.get('weighted', False),
            'is_hurdle': best_model_info.get('is_hurdle', False),
            'zero_proportion_train': float(zero_prop_train),
            'zero_proportion_test': float(zero_prop_test)
        }
        
        # Add weight statistics if weighted
        if weights_train is not None:
            model_metadata['weight_stats'] = {
                'train_min': float(weights_train.min()),
                'train_max': float(weights_train.max()),
                'train_mean': float(weights_train.mean()),
                'train_median': float(weights_train.median()),
                'test_min': float(weights_test.min()),
                'test_max': float(weights_test.max()),
                'test_mean': float(weights_test.mean()),
                'test_median': float(weights_test.median())
            }
        
        # Add hurdle-specific metadata
        if best_model_info.get('is_hurdle', False):
            model_metadata['hurdle_classifier'] = type(best_model.classifier_).__name__
            model_metadata['hurdle_regressor'] = type(best_model.regressor_).__name__
            model_metadata['zero_threshold'] = best_model.zero_threshold
        
        # Save model
        model_path = save_model_with_metadata(
            model=best_model,
            features=var_reg,
            metadata=model_metadata,
            base_path=model_base_path
        )
        
        logger.info(f"Model saved successfully")
        logger.info(f"  Path: {model_path}")
    
    # ========================================================================
    # STEP 6: VISUALIZATION
    # ========================================================================
    fig = None
    if create_visualizations and len(variables) == 2:
        logger.info("=" * 80)
        logger.info("STEP 6: 3D VISUALIZATION")
        logger.info("=" * 80)
        
        try:
            fig = plot_3d_surface(
                model=best_model,
                train_data=train_data,
                test_data=test_data,
                variables=variables,
                target_var=target_var,
                features=var_reg,
                n_points=20
            )
            
            if fig is not None:
                logger.info(f"3D surface plot created")
                # Optionally save the plot
                if model_path:
                    plot_path = Path(model_path) / "prediction_surface.html"
                    fig.write_html(str(plot_path))
                    logger.info(f"  Plot saved to: {plot_path}")
            else:
                logger.warning(f"Could not create 3D plot")
        except Exception as e:
            logger.error(f"Visualization failed: {str(e)}")
    
    # ========================================================================
    # STEP 7: PIPELINE SUMMARY
    # ========================================================================
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    logger.info(f"Pipeline completed successfully!")
    logger.info(f"Key Results:")
    logger.info(f"  - Best Model: {best_model_info['name']}")
    logger.info(f"  - Test R²: {best_model_info['test_r2']:.4f}")
    logger.info(f"  - Test RMSE: {best_model_info['test_rmse']:.4f}")
    logger.info(f"  - Training samples: {len(X_train):,}")
    logger.info(f"  - Test samples: {len(X_test):,}")
    
    if model_path:
        logger.info(f"  - Model saved: {model_path}")

    logger.info("=" * 80)
    
    # ========================================================================
    # RETURN ALL RESULTS
    # ========================================================================
    return {
        'train_data': train_data,
        'test_data': test_data,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'weights_train': weights_train,
        'weights_test': weights_test,
        'features': var_reg,
        'features_extended': var_reg_extended,
        'results_df': results_df,
        'best_model_info': best_model_info,
        'model_path': model_path,
        'visualization': fig
    }


def predict_on_new_data(model_path: str, new_data: pd.DataFrame):
    """
    Load a saved model and make predictions on new data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model directory
    new_data : pd.DataFrame
        New data with the same features as training
    
    Returns:
    --------
    np.ndarray
        Predictions for the new data
    
    Example:
    --------
    >>> # Load model and predict
    >>> predictions = predict_on_new_data(
    ...     model_path='models/production/model_20240121_143022',
    ...     new_data=new_df
    ... )
    """
    logger.info("=" * 80)
    logger.info("PREDICTION ON NEW DATA")
    logger.info("=" * 80)
    
    # Load model
    model, metadata, features = load_model_for_prediction(model_path)
    
    # Verify features exist in new data
    missing_features = [f for f in features if f not in new_data.columns]
    if missing_features:
        raise ValueError(f"Missing features in new data: {missing_features}")
    
    # Make predictions
    predictions = model.predict(new_data[features])
    
    logger.info(f"Predictions generated")
    logger.info(f"  Number of predictions: {len(predictions):,}")
    logger.info(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    logger.info(f"  Mean prediction: {predictions.mean():.4f}")
    
    # Additional info for Hurdle models
    if metadata.get('is_hurdle', False):
        from src.inference_optimized import HurdleRegressor
        if isinstance(model, HurdleRegressor):
            binary_pred = model.predict_binary(new_data[features])
            prob_nonzero = model.classifier_.predict_proba(new_data[features])[:, 1]
            logger.info(f"Hurdle Model Details:")
            logger.info(f"  Predicted non-zero: {binary_pred.sum():,} ({binary_pred.mean():.1%})")
            logger.info(f"  Mean P(non-zero): {prob_nonzero.mean():.2%}")
    
    return predictions


def todu_average_inference(
    data: pd.DataFrame, 
    variables: List[str], 
    indicators: List[str],
    feature_col: str = 'oa_amt',        # Renamed from hardcoded var_reg_oa
    target_col: str = 'todu_amt_pile_h6', # Renamed from hardcoded var_target_oa
    z_threshold: float = 3.0, 
    plot_output_path: Optional[str] = "images/todu_avg_inference.html",
    model_output_path: Optional[str] = "models/todu_model.joblib"
) -> Tuple[go.Figure, Any, float]:
    """
        Calculate linear regression between a feature and a target (Todu), 
        visualize results, and optionally save the trained model.
        
        Args:
            data: Input DataFrame.
            variables: List of grouping variables (e.g., ['year', 'month']).
            indicators: List of indicator columns to sum during aggregation.
            feature_col: The independent variable (X axis), default 'oa_amt'.
            target_col: The dependent variable (Y axis), default 'todu_amt_pile_h6'.
            z_threshold: Z-score threshold for outlier removal.
            plot_output_path: Path to save the HTML plot. If None, does not save plot.
            model_output_path: Path to save the trained model (e.g. 'model.joblib'). 
                            If None, does not save model.
            
        Returns:
            Tuple containing: (Plotly Figure, Fitted Sklearn Model, R-squared score)
    """ 
    logger.info("Calculating todu average inference...")

    # 1. Validation
    # ---------------------------------------------------------
    required_cols = variables + indicators
    missing = [c for c in required_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")
        
    if feature_col not in indicators or target_col not in indicators:
         raise ValueError(f"Feature '{feature_col}' and Target '{target_col}' must be present in 'indicators' list.")

    # 2. Data Preparation
    # ---------------------------------------------------------
    # Filter for 'booked' status (case insensitive)
    mask_booked = data[Columns.STATUS_NAME].astype(str).str.lower() == StatusName.BOOKED.value
    
    # Aggregation
    df_grouped = (
        data[mask_booked]
        .groupby(variables, as_index=False) # as_index=False keeps variables as columns
        .agg({col: 'sum' for col in indicators})
        .dropna()
    )
    
    if df_grouped.empty:
        logger.warning("No data found after filtering for 'booked' status.")
        return go.Figure(), None, 0.0

    # 3. Outlier Removal
    # ---------------------------------------------------------
    # Calculate Z-scores on the target variable
    # We use a mask to avoid SettingWithCopy warnings and ensure alignment
    z_scores = np.abs(zscore(df_grouped[target_col]))
    df_train = df_grouped[z_scores < z_threshold].copy()
    
    n_dropped = len(df_grouped) - len(df_train)
    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped} outliers based on Z-threshold {z_threshold}")

    if df_train.empty:
        logger.warning("All data points were removed as outliers.")
        return go.Figure(), None, 0.0

    # 4. Modeling (Linear Regression)
    # ---------------------------------------------------------
    # Scikit-learn expects 2D array for features: [[x1], [x2], ...]
    X = df_train[[feature_col]] 
    y = df_train[target_col]

    model = LinearRegression(fit_intercept=False) # Forced through origin as per original code
    model.fit(X, y)   
    
    r_sq = model.score(X, y)
    coef = model.coef_[0]
    
    logger.info(f"Model Coefficient: {coef:.4f}")
    logger.info(f"Model R²: {r_sq:.4f}")

    # --- SAVE MODEL FUNCTIONALITY ---
    if model_output_path:
        try:
            joblib.dump(model, model_output_path)
            logger.info(f"Trained model saved successfully to: {model_output_path}")
        except Exception as e:
            logger.error(f"Failed to save model to {model_output_path}: {e}")
    # --------------------------------

    # Generate predictions
    df_train['y_pred'] = model.predict(X)

    # 5. Visualization
    # ---------------------------------------------------------
    fig = go.Figure()

    # Create hover text showing the grouping variables (e.g., Year, Region)
    hover_text = df_train[variables].apply(lambda row: '<br>'.join([f"{col}: {val}" for col, val in row.items()]), axis=1)

    # Actual Data Scatter
    fig.add_trace(go.Scatter(
        x=df_train[feature_col], 
        y=df_train[target_col], 
        mode='markers', 
        name='Actual Data',
        marker=dict(color='#1f77b4', opacity=0.6, size=8),
        text=hover_text,
        hovertemplate=(
            f"<b>Actual</b><br>" +
            f"{feature_col}: %{{x:,.0f}}<br>" +
            f"{target_col}: %{{y:,.0f}}<br>" +
            "%{text}<extra></extra>"
        )
    ))

    # Regression Line
    fig.add_trace(go.Scatter(
        x=df_train[feature_col], 
        y=df_train['y_pred'], 
        mode='lines', 
        name=f'Fit (Coef: {coef:.4f})',
        line=dict(color='#d62728', width=3), # Red line
        hovertemplate=f"Predicted: %{{y:,.0f}}<extra></extra>"
    ))

    # Layout Updates
    fig.update_layout(
        title=dict(
            text=f'<b>Inference Model: {target_col} vs {feature_col}</b><br><sup>R²: {r_sq:.3f} | Coefficient: {coef:.4f}</sup>',
            x=0.05
        ),
        xaxis=dict(title=feature_col, gridcolor='lightgrey', zeroline=False),
        yaxis=dict(title=target_col, gridcolor='lightgrey', zeroline=False),
        plot_bgcolor='white',
        width=1000,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # 6. Save Output
    # ---------------------------------------------------------
    if plot_output_path:
        try:
            fig.write_html(plot_output_path)
            logger.info(f"Todu average inference plot saved to {plot_output_path}")
        except Exception as e:
            logger.error(f"Failed to save Todu average inference plot: {e}")

    return fig, model, r_sq

def run_optimization_pipeline(data_booked, data_demand, risk_inference, reg_todu_amt_pile, stressor, tasa_fin, config_data, annual_coef):
    """
    Runs the optimization pipeline: aggregates data, applies risk models, and generates visualizations.
    """
    logger.info("Running optimization pipeline...")
    
    # Define variables needed for optimization
    final_model = risk_inference['best_model_info']['model']
    final_features = risk_inference['features']
    var_target = 'b2_ever_h6'

    # Define Constants
    INDICADORES = config_data.get('indicators')
    VARIABLES = config_data.get('variables')
    
    # Calculate aggregate data for booked and repesca cases
    def calculate_aggregate_data(data, status, reject_reason=None):
        filtered_data = data[data[Columns.STATUS_NAME] == status]
        if reject_reason:
            filtered_data = filtered_data[filtered_data[Columns.REJECT_REASON] == reject_reason]
        return (
            filtered_data.groupby(VARIABLES)
            .agg({col: "sum" for col in INDICADORES})
            .mul(annual_coef)  # Multiply by annual coefficient
            .reset_index()[VARIABLES + INDICADORES]
        )

    data_sumary_desagregado_booked = calculate_aggregate_data(data_booked, StatusName.BOOKED.value)
    data_sumary_desagregado_booked.rename(columns={i: i + Suffixes.BOOKED for i in INDICADORES}, inplace=True)

    data_sumary_desagregado_repesca = calculate_aggregate_data(
        data_demand, StatusName.REJECTED.value, "09-score"
    )
    
    # Apply Risk Model to Repesca
    # Note: calculate_risk_values is imported from src.models
    from src.models import calculate_risk_values
    data_sumary_desagregado_repesca = calculate_risk_values(
        data_sumary_desagregado_repesca, final_model, reg_todu_amt_pile, VARIABLES, stressor, final_features
    )[VARIABLES + INDICADORES]

    data_sumary_desagregado_repesca[INDICADORES] *= tasa_fin
    data_sumary_desagregado_repesca.rename(columns={i: i + '_rep' for i in INDICADORES}, inplace=True)

    # Merge and adjust indicators
    data_sumary_desagregado = data_sumary_desagregado_booked.merge(
        data_sumary_desagregado_repesca, on=VARIABLES, how='outer'
    ).fillna(0)

    for indicador in INDICADORES:
        data_sumary_desagregado[indicador] = (
            data_sumary_desagregado[indicador + '_boo'] +
            data_sumary_desagregado[indicador + '_rep']
        )

    # Visualization
    logger.info("Generating optimization visualization...")
    fig = go.Figure()
    data_surf = data_sumary_desagregado.copy()
    data_surf['b2_ever_h6'] = calculate_b2_ever_h6(
        data_surf['todu_30ever_h6'],
        data_surf['todu_amt_pile_h6']
    )
    data_surf_pivot = data_surf.pivot(index=VARIABLES[1], columns=VARIABLES[0], values='b2_ever_h6')
    
    fig = fig.add_trace(go.Surface(
        x=data_surf_pivot.columns, y=data_surf_pivot.index, z=data_surf_pivot.values, colorscale='turbo'
    ))
    
    styles.apply_plotly_style(
        fig,
        title="B2 Ever H6 vs. Octroi and Risk Score",
        width=1500,
        height=700
    )
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(title=VARIABLES[0]),
            yaxis=dict(title=VARIABLES[1]),
            zaxis=dict(title=var_target),
            aspectratio=dict(x=1, y=1, z=1)
        )
    )
    fig.write_html("images/b2_ever_h6_vs_octroi_and_risk_score.html")
    logger.info("Optimization visualization saved to images/b2_ever_h6_vs_octroi_and_risk_score.html")
    
    return data_sumary_desagregado