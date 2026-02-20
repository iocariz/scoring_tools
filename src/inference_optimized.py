"""
Optimized inference module for credit risk scoring system.
Key improvements:
- Consolidated imports
- Removed redundant operations
- Improved memory efficiency
- Better error handling
- More maintainable code structure
"""

# Standard library imports
from pathlib import Path
from typing import Any

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger
from scipy.stats import zscore

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from src import styles
from src.constants import (
    DEFAULT_N_POINTS_3D,
    DEFAULT_RANDOM_STATE,
    DEFAULT_Z_THRESHOLD,
    Columns,
    StatusName,
    Suffixes,
)
from src.estimators import HurdleRegressor

# Project imports
from src.models import transform_variables
from src.optuna_tuning import tune_linear_models, tune_tree_models
from src.persistence import (
    save_model_with_metadata,
)
from src.utils import calculate_b2_ever_h6


def _get_model_complexity(model_name: str) -> int:
    """Return a complexity rank for a model type (lower = simpler)."""
    model_name_lower = model_name.lower()
    complexity_map = [
        ("linear regression", 1),
        ("ridge", 2),
        ("lasso", 3),
        ("elasticnet", 4),
        ("elastic", 4),
        ("tweedie", 5),
        ("hurdle-ridge", 6),
        ("hurdle-lasso", 7),
        ("xgboost", 8),
        ("lightgbm", 9),
    ]
    for pattern, rank in complexity_map:
        if pattern in model_name_lower:
            return rank
    return 10  # unknown models get highest complexity


def _apply_one_se_rule(results_df: pd.DataFrame, complexity_col: str) -> int:
    """Apply the one-standard-error rule: among models within 1 SE of the best,
    pick the simplest.

    The 'CV Std R²' column is expected to contain the standard error of the
    mean (i.e., ``std(fold_scores, ddof=1) / sqrt(k)``), not the raw standard
    deviation of fold scores.

    Args:
        results_df: DataFrame with 'CV Mean R²', 'CV Std R²', and complexity_col.
        complexity_col: Column name with complexity metric (lower = simpler).

    Returns:
        Index of the selected row in results_df.
    """
    best_idx = results_df["CV Mean R²"].idxmax()
    best_mean = results_df.loc[best_idx, "CV Mean R²"]
    best_se = results_df.loc[best_idx, "CV Std R²"]
    threshold = best_mean - best_se

    # Models within 1 SE of best
    eligible = results_df[results_df["CV Mean R²"] >= threshold]

    # Pick simplest among eligible
    selected_idx = eligible[complexity_col].idxmin()

    selected_name = results_df.loc[selected_idx].get("Model", results_df.loc[selected_idx].get("Feature Set", ""))
    best_name = results_df.loc[best_idx].get("Model", results_df.loc[best_idx].get("Feature Set", ""))
    if selected_idx != best_idx:
        logger.info(
            f"1SE rule: selected '{selected_name}' (R²={results_df.loc[selected_idx, 'CV Mean R²']:.4f}) "
            f"over '{best_name}' (R²={best_mean:.4f}±{best_se:.4f}, threshold={threshold:.4f})"
        )
    else:
        logger.info(f"1SE rule: best model '{best_name}' is also the simplest eligible")

    return selected_idx


def calculate_target_metric(df: pd.DataFrame, multiplier: float, numerator: str, denominator: str) -> np.ndarray:
    """
    Calculate target metric with proper handling of edge cases.

    Delegates to calculate_b2_ever_h6 for consistent division-by-zero handling.

    Args:
        df: DataFrame containing the data
        multiplier: Multiplier for the target calculation
        numerator: Column name for numerator
        denominator: Column name for denominator

    Returns:
        Calculated metric rounded to 2 decimals as ndarray
    """
    return calculate_b2_ever_h6(df[numerator], df[denominator], multiplier=multiplier).values


def _generate_regression_variables(variables: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """
    Generate regression variable names dynamically.

    For 2 variables: uses the legacy hardcoded feature sets for backward compatibility.
    For N > 2 variables: builds feature sets at degree 1/2/3 from PolynomialFeatures
    on [variables + complements].

    Args:
        variables: List of base variable names (len >= 2)

    Returns:
        Tuple of (var_reg, feature_sets) where:
        - var_reg: Base feature set
        - feature_sets: Dictionary of named feature sets for comparison
    """
    if len(variables) == 2:
        return _generate_regression_variables_2d(variables)
    return _generate_regression_variables_nd(variables)


def _generate_regression_variables_2d(variables: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """Standard 2-variable feature set generation."""
    var0, var1 = variables

    var_reg = [
        f"{var1}",
        f"{var0}",
        f"{var0}_{var1}",
    ]

    squared_terms = [
        f"{var1}^2",
        f"{var0}^2",
    ]

    cubic_terms = [
        f"{var1}^3",
        f"{var0}^3",
    ]

    extra_interactions = [
        f"{var0}^2 x {var1}",
        f"{var0} x {var1}^2",
    ]

    var_simple = [
        f"{var1}",
        f"{var0}",
    ]

    feature_sets = {
        "original": list(variables),
        "simple": var_simple,
        "base": var_reg,
        "base + squared": var_reg + squared_terms,
        "base + cubic": var_reg + cubic_terms,
        "base + all_poly": var_reg + squared_terms + cubic_terms,
        "base + interactions": var_reg + extra_interactions,
        "full": var_reg + squared_terms + cubic_terms + extra_interactions,
    }

    return var_reg, feature_sets


def _generate_regression_variables_nd(variables: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """N-variable feature set generation using PolynomialFeatures naming."""
    from sklearn.preprocessing import PolynomialFeatures

    input_cols = list(variables)

    feature_sets: dict[str, list[str]] = {}
    var_reg: list[str] = []

    for deg, label in [(1, "degree_1"), (2, "degree_2"), (3, "degree_3")]:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        poly.fit_transform(np.zeros((1, len(input_cols))))
        names = list(poly.get_feature_names_out(input_cols))
        if deg == 1:
            # Degree-1 features are the base variables themselves
            feature_sets[label] = names
            var_reg = names
        else:
            # Higher degrees: remove raw variable names (already columns in df)
            feature_names = [n for n in names if n not in variables]
            feature_sets[label] = feature_names

    # Also add a combined set
    feature_sets["original"] = list(variables)
    feature_sets["full"] = feature_sets["degree_3"]

    return var_reg, feature_sets


def process_dataset(
    data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    var_reg: list[str],
    z_threshold: float = DEFAULT_Z_THRESHOLD,
) -> pd.DataFrame:
    """
    Aggregate raw records by bin groups and compute regression features.

    Aggregates indicator columns (sum) by the grouping variables, then
    computes polynomial/interaction features on the aggregated group values.
    Features are computed *after* aggregation so that each group's feature
    values reflect the group's bin coordinates, not a sum of per-record
    polynomial terms.

    Args:
        data: Dataset to process (caller must pre-filter to booked records).
        bins: Tuple of (octroi_bins, efx_bins) for variable binning.
        variables: List of two grouping variable names (bin columns).
        indicators: Indicator columns to aggregate (summed per group).
        target_var: Name of the target variable to compute from aggregated indicators.
        multiplier: Multiplier for the target calculation.
        var_reg: Regression variable names (used only for documentation; features
            are generated by transform_variables after aggregation).
        z_threshold: Z-score threshold for outlier removal.

    Returns:
        Processed DataFrame with one row per bin group, containing:
        - grouping variables
        - summed indicator columns
        - polynomial/interaction features (from transform_variables)
        - n_observations (group size for weighted regression)
        - target_var (computed risk metric)
    """
    # ---- 1. Aggregate indicator columns by bin groups ----
    # Only sum the indicator columns; polynomial features will be computed
    # on the group-level bin values afterwards.
    agg_dict = {col: "sum" for col in indicators if col in data.columns}
    grouped = data.groupby(variables, observed=True)

    processed_data = grouped.agg(agg_dict)
    processed_data["n_observations"] = grouped.size()
    processed_data = processed_data.reset_index().sort_values(by=variables)

    # ---- 2. Compute polynomial/interaction features on aggregated bin values ----
    # Each row now represents a unique bin combination; (9-var0), var1^2, etc.
    # are computed on the bin coordinate values, not summed per-record values.
    processed_data = transform_variables(processed_data, variables)

    # ---- 3. Calculate target variable from aggregated indicators ----
    processed_data[target_var] = calculate_target_metric(
        processed_data, multiplier, "todu_30ever_h6", "todu_amt_pile_h6"
    )

    # ---- 4. Filter missing targets ----
    processed_data = processed_data.dropna(subset=[target_var]).copy()

    # ---- 5. Remove outlier bins by z-score on target variable ----
    if z_threshold > 0 and len(processed_data) > 2:
        target_vals = processed_data[target_var]
        target_median = target_vals.median()
        target_mad = np.median(np.abs(target_vals - target_median))
        
        if target_mad == 0:
             target_mad = np.mean(np.abs(target_vals - target_median))

        if target_mad > 0:
            z_scores = 0.6745 * np.abs(target_vals - target_median) / target_mad
            outlier_mask = z_scores >= z_threshold
            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                logger.debug(f"process_dataset: removed {n_outliers} outlier bins (z >= {z_threshold})")
                processed_data = processed_data[~outlier_mask].copy()

    return processed_data


def _create_feature_dataframe(mesh_df: pd.DataFrame, features: list[str], var0: str, var1: str) -> pd.DataFrame:
    """
    Create feature DataFrame for model predictions on a mesh grid.

    Uses the same ``transform_variables`` function that generates training
    features, ensuring the prediction surface is consistent with the model's
    training data regardless of which feature set was selected.

    Args:
        mesh_df: DataFrame with mesh grid points (columns: var0, var1).
        features: List of feature column names the model expects.
        var0: First variable name.
        var1: Second variable name.

    Returns:
        DataFrame containing only the requested feature columns, in order.
    """
    transformed = transform_variables(mesh_df.copy(), [var0, var1])
    missing = [f for f in features if f not in transformed.columns]
    if missing:
        raise ValueError(
            f"transform_variables did not produce expected features: {missing}. "
            f"Available columns: {sorted(transformed.columns.tolist())}"
        )
    return transformed[features]


def plot_3d_surface(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    variables: list[str],
    target_var: str,
    features: list[str],
    n_points: int = DEFAULT_N_POINTS_3D,
) -> go.Figure | None:
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
        mesh_df = pd.DataFrame({var0: x_mesh.ravel(), var1: y_mesh.ravel()})

        # Create features for model
        feature_df = _create_feature_dataframe(mesh_df, features, var0, var1)

        # Make predictions and reshape
        z_pred = model.predict(feature_df)
        z_mesh = z_pred.reshape(n_points, n_points)

        # Create the 3D plot
        fig = go.Figure(
            data=[
                # Training data
                go.Scatter3d(
                    x=train_data[var0],
                    y=train_data[var1],
                    z=train_data[target_var],
                    mode="markers",
                    marker=dict(size=5, color=styles.COLOR_ACCENT, opacity=0.7),
                    name="Training Data",
                ),
                # Test data
                go.Scatter3d(
                    x=test_data[var0],
                    y=test_data[var1],
                    z=test_data[target_var],
                    mode="markers",
                    marker=dict(size=5, color=styles.COLOR_RISK, opacity=0.7),
                    name="Test Data",
                ),
                # Prediction surface
                go.Surface(z=z_mesh, x=x_mesh, y=y_mesh, colorscale="Viridis", opacity=0.8, name="Model Predictions"),
            ]
        )

        # Update layout
        styles.apply_plotly_style(
            fig, title=f"{target_var} vs {var0} and {var1} with Model Predictions", width=1000, height=800
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=var0,
                yaxis_title=var1,
                zaxis_title=target_var,
                xaxis=dict(range=[1, max(1, x_max)]),
                yaxis=dict(range=[1, max(1, y_max)]),
                aspectratio=dict(x=1, y=1, z=0.8),
            )
        )

        return fig

    except (ValueError, KeyError, ImportError) as e:
        logger.error(f"Error creating 3D plot: {str(e)}")
        logger.exception("3D Plot generation failed")
        return None


def _select_model_type_cv(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    z_threshold: float,
    var_reg: list[str],
    cv_folds: int,
    include_hurdle: bool,
) -> tuple[pd.DataFrame, dict]:
    """
    Select the best model type using k-fold cross-validation on base features.

    Evaluates candidate model types using automated Optuna tuning on var_reg features,
    and returns the best model type based on mean CV R².

    Args:
        raw_data: Un-aggregated dataset with features and target.
        bins: Bin definitions.
        variables: Demographic features to group by.
        indicators: Metrics required for aggregation.
        target_var: Target column name.
        multiplier: Scaler for target derivation.
        z_threshold: Outlier cutoff threshold.
        var_reg: Base feature column names.
        cv_folds: Number of cross-validation folds.
        include_hurdle: Whether to include Hurdle model variants.

    Returns:
        Tuple of (results_df, best_model_info) where best_model_info contains
        the unfitted model object, name, and CV scores.
    """
    # Run Optuna tuning for all linear and GLM regression models
    results_df, _ = tune_linear_models(
        raw_data=raw_data,
        bins=bins,
        variables=variables,
        indicators=indicators,
        target_var=target_var,
        multiplier=multiplier,
        z_threshold=z_threshold,
        var_reg=var_reg,
        cv_folds=cv_folds,
        n_trials=20,
        include_hurdle=include_hurdle,
        random_state=DEFAULT_RANDOM_STATE,
    )

    if results_df.empty:
        raise RuntimeError(
            "All linear/GLM models failed during cross-validation. Check your data for NaNs, infinites, or shape mismatches."
        )

    # Log top results
    display_cols = ["Model", "CV Mean R²", "CV Std R²"]
    logger.info("Model type CV results:")
    logger.info(f"\n{results_df[display_cols].head(10).to_string()}")

    # Apply 1SE rule: pick simplest model within 1 SE of the best
    results_df["_complexity"] = results_df["Model"].apply(_get_model_complexity)
    best_idx = _apply_one_se_rule(results_df, "_complexity")
    results_df = results_df.drop(columns=["_complexity"])
    best_row = results_df.loc[best_idx]

    best_model_info = {
        "model_template": best_row["model_template"],
        "name": best_row["Model"],
        "cv_mean_r2": best_row["CV Mean R²"],
        "cv_std_r2": best_row["CV Std R²"],
    }

    logger.info(f"Best model type: {best_model_info['name']}")
    logger.info(f"  CV R²: {best_model_info['cv_mean_r2']:.4f} ± {best_model_info['cv_std_r2']:.4f}")

    return results_df, best_model_info


def _select_feature_set_cv(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    z_threshold: float,
    feature_sets: dict[str, list[str]],
    model_template,
    cv_folds: int,
) -> tuple[pd.DataFrame, dict]:
    """
    Select the best feature set using k-fold cross-validation with a fixed model type.
    """
    from sklearn.base import clone

    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=DEFAULT_RANDOM_STATE)
    feature_results = []

    for feature_name, features in feature_sets.items():
        cv_scores = []
        for train_idx, val_idx in kfold.split(raw_data):
            raw_train, raw_val = raw_data.iloc[train_idx].copy(), raw_data.iloc[val_idx].copy()

            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, features, z_threshold
            )
            val_agg = process_dataset(
                raw_val, bins, variables, indicators, target_var, multiplier, features, z_threshold
            )

            missing_train = [f for f in features if f not in train_agg.columns]
            if missing_train:
                continue

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                logger.debug(f"Fold skipped for {feature_name}: only {len(val_agg)} validation bins")
                continue

            X_train, y_train = train_agg[features], train_agg[target_var]
            X_val, y_val = val_agg[features], val_agg[target_var]

            w_train = train_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in train_agg.columns else None
            w_val = val_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in val_agg.columns else None

            model_clone = clone(model_template)
            model_clone.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model_clone.predict(X_val)
            if len(y_val) >= 2:
                fold_r2 = r2_score(y_val, y_pred, sample_weight=w_val)
                cv_scores.append(fold_r2)

        if not cv_scores:
            logger.warning(f"Skipping {feature_name}: execution failed internally.")
            continue

        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores, ddof=1) / np.sqrt(len(cv_scores))

        feature_results.append(
            {
                "Feature Set": feature_name,
                "Num Features": len(features),
                "Features": features,
                "CV Mean R²": cv_mean,
                "CV Std R²": cv_std,
            }
        )

        logger.info(f"  {feature_name} ({len(features)} features): CV R² = {cv_mean:.4f} ± {cv_std:.4f}")

    if not feature_results:
        raise RuntimeError("All feature sets failed or were skipped. Check data columns.")

    results_df = pd.DataFrame(feature_results)

    # Apply 1SE rule: pick simplest feature set within 1 SE of the best
    best_idx = _apply_one_se_rule(results_df, "Num Features")
    best_row = results_df.loc[best_idx]

    best_feature_info = {
        "feature_set_name": best_row["Feature Set"],
        "features": best_row["Features"],
        "cv_mean_r2": best_row["CV Mean R²"],
        "cv_std_r2": best_row["CV Std R²"],
    }

    # Log results table
    display_cols = ["Feature Set", "Num Features", "CV Mean R²", "CV Std R²"]
    logger.info("Feature set CV results:")
    logger.info(f"\n{results_df[display_cols].to_string()}")
    logger.info(f"Best feature set: {best_feature_info['feature_set_name']}")
    logger.info(f"  CV R²: {best_feature_info['cv_mean_r2']:.4f} ± {best_feature_info['cv_std_r2']:.4f}")

    return results_df, best_feature_info


def _prepare_pipeline_data(
    data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
    """Filter booked records and determine base models."""
    var_reg, feature_sets = _generate_regression_variables(variables)

    booked_data = data[data[Columns.STATUS_NAME] == StatusName.BOOKED.value].copy()

    # We remove records where the demographic variables or target indicators are null
    req_cols = variables + indicators
    booked_data = booked_data.dropna(subset=req_cols).copy()

    logger.info(f"Booked valid records: {len(booked_data):,} of {len(data):,} total")
    logger.info(f"Base features (var_reg): {var_reg}")

    return booked_data, var_reg, feature_sets


def _select_best_model_and_features(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    multiplier: float,
    z_threshold: float,
    var_reg: list[str],
    feature_sets: dict[str, list[str]],
    target_var: str,
    cv_folds: int,
    include_hurdle: bool,
    directions: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    """Run model type CV + feature set CV, and Optuna tuning for tree models."""
    # Step 1: Tune Tree Models on Original Features
    logger.info("-" * 40)
    logger.info(f"STEP 2A: TUNE TREE MODELS ON ORIGINAL FEATURES ({cv_folds}-fold CV)")
    logger.info("-" * 40)
    tree_results_df, tree_models = tune_tree_models(
        raw_data=raw_data,
        bins=bins,
        variables=variables,
        indicators=indicators,
        target_var=target_var,
        multiplier=multiplier,
        z_threshold=z_threshold,
        cv_folds=cv_folds,
        n_trials=30,
        directions=directions,
    )

    # Step 2: Evaluate Linear/GLM models on var_reg
    logger.info("-" * 40)
    logger.info(f"STEP 2B: LINEAR/GLM MODEL TYPE SELECTION ({cv_folds}-fold CV)")
    logger.info("-" * 40)
    results_step1, best_model_type = _select_model_type_cv(
        raw_data=raw_data,
        bins=bins,
        variables=variables,
        indicators=indicators,
        target_var=target_var,
        multiplier=multiplier,
        z_threshold=z_threshold,
        var_reg=var_reg,
        cv_folds=cv_folds,
        include_hurdle=include_hurdle,
    )

    # Combine results to find the global winner
    combined_results = pd.concat([tree_results_df, results_step1], ignore_index=True)
    combined_results = combined_results.sort_values("CV Mean R²", ascending=False)

    logger.info("Combined Model CV results:")
    display_cols = ["Model", "CV Mean R²", "CV Std R²"]
    logger.info(f"\n{combined_results[display_cols].head(10).to_string()}")

    # Apply 1SE rule on combined tree+linear results
    combined_results["_complexity"] = combined_results["Model"].apply(_get_model_complexity)
    best_combined_idx = _apply_one_se_rule(combined_results, "_complexity")
    combined_results = combined_results.drop(columns=["_complexity"])
    best_global_name = combined_results.loc[best_combined_idx, "Model"]

    if "Optuna Tuned" in best_global_name:
        # A tree model won. We do not need step 3 for feature sets,
        # tree models inherently use "original" feature set.
        best_row = combined_results.loc[best_combined_idx]
        best_model_template = best_row["model_template"]
        logger.info(f"Tree model '{best_global_name}' won overall! Skipping Step 3.")

        # Override the outputs to fast-path using "original" features
        best_model_type = {
            "model_template": best_model_template,
            "name": best_global_name,
            "cv_mean_r2": best_row["CV Mean R²"],
            "cv_std_r2": best_row["CV Std R²"],
        }

        # Simulate results of step 2 for tree models
        results_step2 = pd.DataFrame(
            [
                {
                    "Feature Set": "original",
                    "Num Features": len(variables),
                    "Features": variables,
                    "CV Mean R²": best_row["CV Mean R²"],
                    "CV Std R²": best_row["CV Std R²"],
                }
            ]
        )

        best_feature_info = {
            "feature_set_name": "original",
            "features": variables,
            "cv_mean_r2": best_row["CV Mean R²"],
            "cv_std_r2": best_row["CV Std R²"],
        }
    else:
        # A linear/GLM model won. Proceed with Step 3.
        best_model_name = best_model_type["name"]
        best_model_template = best_model_type["model_template"]

        logger.info("-" * 40)
        logger.info(f"STEP 3: FEATURE SET SELECTION ({best_model_name}, {cv_folds}-fold CV)")
        logger.info("-" * 40)

        results_step2, best_feature_info = _select_feature_set_cv(
            raw_data=raw_data,
            bins=bins,
            variables=variables,
            indicators=indicators,
            target_var=target_var,
            multiplier=multiplier,
            z_threshold=z_threshold,
            feature_sets=feature_sets,
            model_template=best_model_template,
            cv_folds=cv_folds,
        )

    return (
        combined_results.drop(columns=["model_template"], errors="ignore"),
        best_model_type,
        results_step2,
        best_feature_info,
    )


def _train_and_evaluate_final_model(
    best_model_template,
    all_data: pd.DataFrame,
    final_features: list[str],
    target_var: str,
    weights: pd.Series | None,
) -> tuple[Any, float]:
    """Clone, fit, predict, compute R², return model + train_r2."""
    from sklearn.base import clone

    final_model = clone(best_model_template)
    y_all = all_data[target_var]
    final_model.fit(all_data[final_features], y_all, sample_weight=weights)

    y_all_pred = final_model.predict(all_data[final_features])
    
    if len(y_all) >= 2:
        train_r2 = r2_score(y_all, y_all_pred, sample_weight=weights)
    else:
        train_r2 = float('nan')

    return final_model, train_r2


def _compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict | None:
    """Compute SHAP values for the trained model. Returns dict or None on failure."""
    try:
        import shap

        if hasattr(model, "coef_"):
            # Linear models: exact and fast
            explainer = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        elif type(model).__name__ in ("XGBRegressor", "LGBMRegressor"):
            # Tree models: exact and fast
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            # Non-linear models (Hurdle, Tweedie): use sampling-based explainer
            # Use a small background sample for efficiency
            background = shap.sample(X, min(50, len(X)))

            # Wrap predict to prevent shap from trying to access/set sklearn-specific properties on the model
            def safe_predict(data):
                if isinstance(data, pd.DataFrame):
                    return model.predict(data)
                return model.predict(pd.DataFrame(data, columns=X.columns))

            explainer = shap.KernelExplainer(safe_predict, background)
            shap_values = explainer.shap_values(X, nsamples=100)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        logger.info(f"SHAP values computed: {shap_values.shape}")

        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }

    except (ImportError, ValueError, TypeError, AttributeError) as e:
        logger.warning(f"SHAP computation failed (non-blocking): {e}")
        return None


def _save_model_to_disk(
    final_model,
    final_features: list[str],
    best_model_info: dict,
    best_model_type: dict,
    best_feature_info: dict,
    all_data: pd.DataFrame,
    target_var: str,
    multiplier: float,
    cv_folds: int,
    zero_prop: float,
    weights: pd.Series | None,
    model_base_path: str,
) -> str:
    """Build metadata, save model with metadata, return path."""
    model_metadata = {
        "cv_mean_r2": best_model_info["cv_mean_r2"],
        "cv_std_r2": best_model_info["cv_std_r2"],
        "train_r2": best_model_info["train_r2"],
        "full_r2": best_model_info["train_r2"],  # Full and train R2 are identical because final model uses all data
        "cv_folds": cv_folds,
        "total_samples": len(all_data),
        "target_variable": target_var,
        "multiplier": multiplier,
        "model_type_selected": best_model_info["model_type"],
        "feature_set_selected": best_model_info["feature_set"],
        "weighted_regression": best_model_info["weighted"],
        "is_hurdle": best_model_info["is_hurdle"],
        "zero_proportion": float(zero_prop),
        "step1_cv_r2": best_model_type["cv_mean_r2"],
        "step2_cv_r2": best_feature_info["cv_mean_r2"],
    }

    if weights is not None:
        model_metadata["weight_stats"] = {
            "min": float(weights.min()),
            "max": float(weights.max()),
            "mean": float(weights.mean()),
        }

    model_path = save_model_with_metadata(
        model=final_model, features=final_features, metadata=model_metadata, base_path=model_base_path
    )
    logger.info(f"Model saved to: {model_path}")

    return model_path


def _create_pipeline_visualization(
    final_model,
    all_data: pd.DataFrame,
    variables: list[str],
    target_var: str,
    final_features: list[str],
    plot_output_path: str | None,
) -> go.Figure | None:
    """Create 3D surface plot and save HTML."""
    try:
        fig = plot_3d_surface(
            model=final_model,
            train_data=all_data,
            test_data=all_data,
            variables=variables,
            target_var=target_var,
            features=final_features,
            n_points=20,
        )

        if fig is not None:
            logger.info("3D surface plot created")
            if plot_output_path:
                fig.write_html(str(plot_output_path))
                logger.info(f"  Plot saved to: {plot_output_path}")

        return fig
    except (ValueError, KeyError, OSError) as e:
        logger.error(f"Visualization failed: {str(e)}")
        return None


def inference_pipeline(
    data: pd.DataFrame,
    bins: tuple,
    variables: list,
    indicators: list,
    target_var: str,
    multiplier: float,
    cv_folds: int = 4,
    include_hurdle: bool = True,
    save_model: bool = True,
    model_base_path: str = "models",
    create_visualizations: bool = True,
    directions: dict[str, int] | None = None,
):
    """
    Two-step inference pipeline using cross-validation throughout.

    Both model type selection and feature set selection are done via k-fold
    cross-validation on the full dataset, avoiding data leakage. The final
    model is retrained on all data with the best model type + feature set.

    Steps:
        1. Prepare data: filter booked records, aggregate by bin groups,
           compute target metric and regression features.
        2. Select model type: evaluate all candidate model types (Linear,
           Ridge, Lasso, ElasticNet, Hurdle variants) via k-fold CV on
           base features (var_reg). Pick the best by mean CV R².
        3. Select feature set: using the best model type, evaluate 7
           feature sets via k-fold CV. Pick the best by mean CV R².
        4. Train final model on all data with best type + features.
        5. Optionally save model and create 3D visualization.

    Args:
        data: Raw input data containing all records.
        bins: Tuple of (octroi_bins, efx_bins) for variable binning.
        variables: List of variable names for modeling.
        indicators: List of indicator column names for calculations.
        target_var: Name of the target variable to predict.
        multiplier: Multiplier for target variable calculation.
        cv_folds: Number of cross-validation folds (default: 5).
        include_hurdle: Whether to include Hurdle models (default: True).
        save_model: Whether to save the best model (default: True).
        model_base_path: Base path for saving models (default: 'models').
        create_visualizations: Whether to create 3D plots (default: True).
        directions: Dict mapping variable names to monotonic constraint
            directions (-1=decreasing, 1=increasing). Used for tree models.

    Returns:
        Dictionary containing all pipeline outputs:
        - all_data: Processed dataset used for training
        - features: Best feature set selected
        - var_reg: Base feature set
        - feature_sets: Dictionary of all available feature sets
        - step1_results: Model type CV comparison results
        - step2_results: Feature set CV comparison results
        - best_model_info: Final model details (model, name, cv_mean_r2,
          cv_std_r2, train_r2, model_type, feature_set, weighted, is_hurdle)
        - model_path: Path to saved model (if save_model=True)
        - visualization: Plotly figure (if create_visualizations=True)
    """
    logger.info("=" * 80)
    logger.info("INFERENCE PIPELINE (CV-based)")
    logger.info("=" * 80)
    logger.info(f"Target: {target_var} | CV folds: {cv_folds} | Hurdle: {include_hurdle}")

    # STEP 1: DATA PREPARATION
    logger.info("-" * 40)
    logger.info("STEP 1: DATA PREPARATION")
    logger.info("-" * 40)

    all_data, var_reg, feature_sets = _prepare_pipeline_data(data, bins, variables, indicators, target_var, multiplier)

    # Global aggregation happens purely at the final stage for training/saving
    final_agg = process_dataset(
        all_data, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold=DEFAULT_Z_THRESHOLD
    )
    weights_all = final_agg["todu_amt_pile_h6"] if "todu_amt_pile_h6" in final_agg.columns else None
    zero_prop = (np.abs(final_agg[target_var]) < 1e-10).mean()

    logger.info(f"Processed full data scope: {final_agg.shape[0]} groups")
    logger.info(f"Zero proportion: {zero_prop:.1%}")
    if weights_all is not None:
        logger.info(f"Weight range: [{weights_all.min():.0f}, {weights_all.max():.0f}]")

    # STEPS 2-3: MODEL TYPE + FEATURE SET SELECTION
    logger.info("-" * 40)
    logger.info("STEP 2: MODEL TYPE SELECTION (CV on var_reg)")
    logger.info("-" * 40)

    results_step1, best_model_type, results_step2, best_feature_info = _select_best_model_and_features(
        raw_data=all_data,
        bins=bins,
        variables=variables,
        indicators=indicators,
        multiplier=multiplier,
        z_threshold=DEFAULT_Z_THRESHOLD,
        var_reg=var_reg,
        feature_sets=feature_sets,
        target_var=target_var,
        cv_folds=cv_folds,
        include_hurdle=include_hurdle,
        directions=directions,
    )

    best_model_name = best_model_type["name"]
    final_features = best_feature_info["features"]

    # STEP 4: TRAIN FINAL MODEL
    logger.info("-" * 40)
    logger.info("STEP 4: FINAL MODEL TRAINING (all data)")
    logger.info("-" * 40)

    final_model, train_r2 = _train_and_evaluate_final_model(
        best_model_type["model_template"], final_agg, final_features, target_var, weights_all
    )

    logger.info(f"Final model: {best_model_name} + {best_feature_info['feature_set_name']}")
    logger.info(f"  CV R²: {best_feature_info['cv_mean_r2']:.4f} ± {best_feature_info['cv_std_r2']:.4f}")
    logger.info(f"  Train R² (in-sample): {train_r2:.4f}")

    if hasattr(final_model, "coef_"):
        logger.debug("Model Coefficients:")
        for feature, coef in zip(final_features, final_model.coef_):
            logger.debug(f"  {feature}: {coef:.6f}")
    elif hasattr(final_model, "regressor_") and hasattr(final_model.regressor_, "coef_"):
        logger.debug("Hurdle Model - Regression Stage Coefficients:")
        for feature, coef in zip(final_features, final_model.regressor_.coef_):
            logger.debug(f"  {feature}: {coef:.6f}")

    best_model_info = {
        "model": final_model,
        "name": f"{best_model_name} + {best_feature_info['feature_set_name']}",
        "cv_mean_r2": best_feature_info["cv_mean_r2"],
        "cv_std_r2": best_feature_info["cv_std_r2"],
        "train_r2": train_r2,
        "model_type": best_model_name,
        "feature_set": best_feature_info["feature_set_name"],
        "weighted": weights_all is not None,
        "is_hurdle": isinstance(final_model, HurdleRegressor),
    }

    # SHAP interpretability (non-blocking)
    shap_result = _compute_shap_values(final_model, final_agg[final_features], final_features)
    if shap_result is not None:
        best_model_info["shap_values"] = shap_result["shap_values"]
        best_model_info["shap_feature_names"] = shap_result["feature_names"]
        best_model_info["mean_abs_shap"] = shap_result["mean_abs_shap"]

        # Export SHAP summary plot (non-blocking)
        try:
            from src.plots import plot_shap_summary

            shap_plot_path = str(Path(model_base_path) / "shap_summary.html")
            plot_shap_summary(
                shap_result["shap_values"],
                shap_result["feature_names"],
                output_path=shap_plot_path,
            )
            logger.info(f"SHAP summary plot saved to {shap_plot_path}")
        except (ImportError, ValueError, OSError) as e:
            logger.warning(f"SHAP plot export failed (non-blocking): {e}")

    # STEP 5: MODEL SAVING
    model_path = None
    if save_model:
        logger.info("-" * 40)
        logger.info("STEP 5: MODEL SAVING")
        logger.info("-" * 40)

        model_path = _save_model_to_disk(
            final_model,
            final_features,
            best_model_info,
            best_model_type,
            best_feature_info,
            final_agg,
            target_var,
            multiplier,
            cv_folds,
            zero_prop,
            weights_all,
            model_base_path,
        )

    # STEP 6: VISUALIZATION
    fig = None
    if create_visualizations and len(variables) == 2:
        logger.info("-" * 40)
        logger.info("STEP 6: 3D VISUALIZATION")
        logger.info("-" * 40)

        # Save visualization alongside models (in images/ sibling directory)
        if model_path:
            images_dir = Path(model_path).parent.parent / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            viz_output_path = str(images_dir / "prediction_surface.html")
        else:
            viz_output_path = None
        fig = _create_pipeline_visualization(
            final_model, final_agg, variables, target_var, final_features, viz_output_path
        )

    # PIPELINE SUMMARY
    logger.info("=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info(f"  Model type (Step 1): {best_model_name} (CV R²: {best_model_type['cv_mean_r2']:.4f})")
    logger.info(
        f"  Feature set (Step 2): {best_feature_info['feature_set_name']} "
        f"(CV R²: {best_feature_info['cv_mean_r2']:.4f} ± {best_feature_info['cv_std_r2']:.4f})"
    )
    logger.info(f"  Train R² (in-sample): {train_r2:.4f}")
    if model_path:
        logger.info(f"  Model saved: {model_path}")
    logger.info("=" * 80)

    return {
        "all_data": final_agg,
        "features": final_features,
        "var_reg": var_reg,
        "feature_sets": feature_sets,
        "step1_results": results_step1,
        "step2_results": results_step2,
        "best_model_info": best_model_info,
        "model_path": model_path,
        "visualization": fig,
    }


def todu_average_inference(
    data: pd.DataFrame,
    variables: list[str],
    indicators: list[str],
    feature_col: str = "oa_amt",  # Renamed from hardcoded var_reg_oa
    target_col: str = "todu_amt_pile_h6",  # Renamed from hardcoded var_target_oa
    z_threshold: float = 3.0,
    plot_output_path: str | None = "images/todu_avg_inference.html",
    model_output_path: str | None = "models/todu_model.joblib",
) -> tuple[go.Figure, Any, float]:
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
        .groupby(variables, as_index=False)  # as_index=False keeps variables as columns
        .agg(dict.fromkeys(indicators, "sum"))
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

    model = LinearRegression(fit_intercept=True)  # Updated: Mathematical risk requires floating intercept bias
    model.fit(X, y)

    r_sq = model.score(X, y)
    coef = model.coef_[0]

    # LOO-CV R² for unbiased performance estimate
    if len(X) > 2:
        from sklearn.metrics import r2_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        loo_preds = cross_val_predict(LinearRegression(fit_intercept=True), X, y, cv=LeaveOneOut())
        loo_r2 = float(r2_score(y, loo_preds))
        logger.info(f"Model Coefficient: {coef:.4f}")
        logger.info(f"Model R² (in-sample): {r_sq:.4f} | LOO-CV R²: {loo_r2:.4f}")
    else:
        logger.info(f"Model Coefficient: {coef:.4f}")
        logger.info(f"Model R² (in-sample): {r_sq:.4f} (too few points for LOO-CV)")

    # --- SAVE MODEL FUNCTIONALITY ---
    if model_output_path:
        try:
            joblib.dump(model, model_output_path)
            logger.info(f"Trained model saved successfully to: {model_output_path}")
        except OSError as e:
            logger.error(f"Failed to save model to {model_output_path}: {e}")
    # --------------------------------

    # Generate predictions
    df_train["y_pred"] = model.predict(X)

    # 5. Visualization
    # ---------------------------------------------------------
    fig = go.Figure()

    # Create hover text showing the grouping variables (e.g., Year, Region)
    hover_text = df_train[variables].apply(
        lambda row: "<br>".join([f"{col}: {val}" for col, val in row.items()]), axis=1
    )

    # Actual Data Scatter
    fig.add_trace(
        go.Scatter(
            x=df_train[feature_col],
            y=df_train[target_col],
            mode="markers",
            name="Actual Data",
            marker=dict(color="#1f77b4", opacity=0.6, size=8),
            text=hover_text,
            hovertemplate=(
                "<b>Actual</b><br>"
                + f"{feature_col}: %{{x:,.0f}}<br>"
                + f"{target_col}: %{{y:,.0f}}<br>"
                + "%{text}<extra></extra>"
            ),
        )
    )

    # Regression Line
    fig.add_trace(
        go.Scatter(
            x=df_train[feature_col],
            y=df_train["y_pred"],
            mode="lines",
            name=f"Fit (Coef: {coef:.4f})",
            line=dict(color="#d62728", width=3),  # Red line
            hovertemplate="Predicted: %{y:,.0f}<extra></extra>",
        )
    )

    # Layout Updates
    fig.update_layout(
        title=dict(
            text=f"<b>Inference Model: {target_col} vs {feature_col}</b><br><sup>R²: {r_sq:.3f} | Coefficient: {coef:.4f}</sup>",
            x=0.05,
        ),
        xaxis=dict(title=feature_col, gridcolor="lightgrey", zeroline=False),
        yaxis=dict(title=target_col, gridcolor="lightgrey", zeroline=False),
        plot_bgcolor="white",
        width=1000,
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # 6. Save Output
    # ---------------------------------------------------------
    if plot_output_path:
        try:
            fig.write_html(plot_output_path)
            logger.info(f"Todu average inference plot saved to {plot_output_path}")
        except (ValueError, OSError) as e:
            logger.error(f"Failed to save Todu average inference plot: {e}")

    return fig, model, r_sq


def run_optimization_pipeline(
    data_booked,
    data_demand,
    risk_inference,
    reg_todu_amt_pile,
    stressor,
    tasa_fin,
    *,
    indicators: list[str],
    variables: list[str],
    annual_coef,
    b2_output_path: str = "images/b2_ever_h6_vs_octroi_and_risk_score.html",
    reject_inference_method: str = "none",
    reject_uplift_factor: float = 1.5,
    reject_max_risk_multiplier: float = 3.0,
):
    """
    Runs the optimization pipeline: aggregates data, applies risk models, and generates visualizations.
    """
    logger.info("Running optimization pipeline...")

    # Define variables needed for optimization
    final_model = risk_inference["best_model_info"]["model"]
    final_features = risk_inference["features"]
    var_target = "b2_ever_h6"

    # Define Constants
    INDICADORES = indicators
    VARIABLES = variables

    # Calculate aggregate data for booked and repesca cases
    def calculate_aggregate_data(data, status, reject_reason=None):
        filtered_data = data[data[Columns.STATUS_NAME] == status]
        if reject_reason:
            filtered_data = filtered_data[filtered_data[Columns.REJECT_REASON] == reject_reason]
        return (
            filtered_data.groupby(VARIABLES)
            .agg(dict.fromkeys(INDICADORES, "sum"))
            .mul(annual_coef)  # Multiply by annual coefficient
            .reset_index()[VARIABLES + INDICADORES]
        )

    data_sumary_desagregado_booked = calculate_aggregate_data(data_booked, StatusName.BOOKED.value)
    data_sumary_desagregado_booked = data_sumary_desagregado_booked.rename(
        columns={i: i + Suffixes.BOOKED for i in INDICADORES}
    )

    data_sumary_desagregado_repesca = calculate_aggregate_data(data_demand, StatusName.REJECTED.value, "09-score")

    # Apply Risk Model to Repesca
    # Note: calculate_risk_values is imported from src.models
    from src.models import calculate_risk_values

    data_sumary_desagregado_repesca = calculate_risk_values(
        data_sumary_desagregado_repesca, final_model, reg_todu_amt_pile, VARIABLES, stressor, final_features
    )[VARIABLES + INDICADORES]

    # Apply reject inference adjustment (after stressor, before tasa_fin)
    if reject_inference_method != "none":
        from src.reject_inference import apply_reject_inference

        data_sumary_desagregado_repesca = apply_reject_inference(
            repesca_summary=data_sumary_desagregado_repesca,
            data_demand=data_demand,
            variables=VARIABLES,
            method=reject_inference_method,
            reject_uplift_factor=reject_uplift_factor,
            max_risk_multiplier=reject_max_risk_multiplier,
        )
        # Drop auxiliary columns before downstream merge
        data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.drop(
            columns=["acceptance_rate", "reject_risk_multiplier"], errors="ignore"
        )

    data_sumary_desagregado_repesca[INDICADORES] *= tasa_fin
    data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.rename(
        columns={i: i + "_rep" for i in INDICADORES}
    )

    # Merge and adjust indicators
    data_sumary_desagregado = data_sumary_desagregado_booked.merge(
        data_sumary_desagregado_repesca, on=VARIABLES, how="outer"
    ).fillna(0)

    for indicador in INDICADORES:
        data_sumary_desagregado[indicador] = (
            data_sumary_desagregado[indicador + "_boo"] + data_sumary_desagregado[indicador + "_rep"]
        )

    # Visualization (only for 2-variable case)
    if len(VARIABLES) == 2:
        logger.info("Generating optimization visualization...")
        fig = go.Figure()
        data_surf = data_sumary_desagregado.copy()
        data_surf["b2_ever_h6"] = calculate_b2_ever_h6(
            data_surf["todu_30ever_h6"], data_surf["todu_amt_pile_h6"], as_percentage=True
        )
        data_surf_pivot = data_surf.pivot(index=VARIABLES[1], columns=VARIABLES[0], values="b2_ever_h6")

        fig = fig.add_trace(
            go.Surface(x=data_surf_pivot.columns, y=data_surf_pivot.index, z=data_surf_pivot.values, colorscale="turbo")
        )

        styles.apply_plotly_style(fig, title="B2 Ever H6 vs. Octroi and Risk Score", width=1500, height=700)

        fig.update_layout(
            scene=dict(
                xaxis=dict(title=VARIABLES[0]),
                yaxis=dict(title=VARIABLES[1]),
                zaxis=dict(title=var_target),
                aspectratio=dict(x=1, y=1, z=1),
            )
        )
        fig.write_html(b2_output_path)
        logger.info(f"Optimization visualization saved to {b2_output_path}")
    else:
        logger.info(f"Skipping 3D surface visualization for {len(VARIABLES)}-variable case")

    return data_sumary_desagregado
