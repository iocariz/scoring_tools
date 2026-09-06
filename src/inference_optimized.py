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
from typing import Any, Literal, get_args

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from loguru import logger

# Scikit-learn imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, StratifiedKFold

from src import styles
from src.constants import (
    DEFAULT_N_POINTS_3D,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RISK_MULTIPLIER,
    DEFAULT_Z_THRESHOLD,
    Columns,
    RejectReason,
    StatusName,
    Suffixes,
)
from src.estimators import HurdleRegressor, prepare_model_input

# Project imports
from src.models import transform_variables
from src.optuna_tuning import tune_linear_models, tune_tree_models
from src.persistence import (
    save_model_with_metadata,
)
from src.utils import calculate_b2_ever_h6

#: Loan-level default numerator — the binarized stratification target for CV.
_STRATIFY_COL = "todu_30ever_h6"

#: A final model whose predictions vary by less than this fraction of the target's
#: own spread across the grid is treated as constant (no discriminatory power).
_DEGENERATE_PRED_RANGE_FRAC = 1e-3


class DegenerateModelError(ValueError):
    """Raised when the selected risk model has no discriminatory power across the
    score grid (constant / near-constant predictions) — a flat risk surface that
    would make the downstream optimization meaningless."""


def _stratify_labels(raw_data: pd.DataFrame, indicators: list[str], min_per_class: int) -> pd.Series | None:
    """Binary stratification labels binarized on the DEFAULT indicator.

    Tries ``todu_30ever_h6 > 0`` (a loan defaulted) first, then the configured
    indicators. Returns ``None`` when no candidate column yields two classes
    with at least ``min_per_class`` members each.
    """
    candidates = [_STRATIFY_COL] + [c for c in (indicators or []) if c != _STRATIFY_COL]
    for strat_col in candidates:
        if strat_col not in raw_data.columns:
            continue
        y_strat = (raw_data[strat_col] > 0).astype(int)
        if (
            y_strat.nunique() >= 2
            and y_strat.sum() >= min_per_class
            and (len(y_strat) - y_strat.sum()) >= min_per_class
        ):
            return y_strat
    return None


def _stratified_cv_splitter(raw_data: pd.DataFrame, cv_folds: int, random_state: int, indicators: list[str]):
    """Create a StratifiedKFold splitter binarized on the DEFAULT indicator.

    Stratifies on ``todu_30ever_h6 > 0`` (a loan defaulted) so the bad rate is
    stable across folds. Audit #33: the old code stratified on ``indicators[0]``
    — ``acct_booked_h0``, ~all 1s on the booked training population — so the
    single-class check failed and it silently fell back to plain KFold on every
    real run. Falls back to KFold (loudly) when the default indicator is
    missing or single-class, trying the configured indicators next.
    """
    y_strat = _stratify_labels(raw_data, indicators, min_per_class=cv_folds)
    if y_strat is not None:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        return skf.split(raw_data, y_strat)
    logger.warning(
        f"Stratified CV unavailable (no usable two-class indicator among ['{_STRATIFY_COL}'] + {indicators}) "
        "— falling back to plain KFold."
    )
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return kfold.split(raw_data)


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

    The 'CV Std RMSE' column is expected to contain the Nadeau–Bengio
    corrected standard error of the CV mean (audit #32d — the naive
    ``std/sqrt(k)`` underestimates it ~2x because fold scores are positively
    correlated, halving the eligibility band), not the raw fold std.

    Args:
        results_df: DataFrame with 'CV Mean RMSE', 'CV Std RMSE', and complexity_col.
        complexity_col: Column name with complexity metric (lower = simpler).

    Returns:
        Index of the selected row in results_df.
    """
    best_idx = results_df["CV Mean RMSE"].idxmin()
    best_mean = results_df.loc[best_idx, "CV Mean RMSE"]
    best_se = results_df.loc[best_idx, "CV Std RMSE"]
    threshold = best_mean + best_se

    # Models within 1 SE of best
    eligible = results_df[results_df["CV Mean RMSE"] <= threshold]

    # Pick simplest among eligible
    selected_idx = eligible[complexity_col].idxmin()

    selected_name = results_df.loc[selected_idx].get("Model", results_df.loc[selected_idx].get("Feature Set", ""))
    best_name = results_df.loc[best_idx].get("Model", results_df.loc[best_idx].get("Feature Set", ""))
    if selected_idx != best_idx:
        logger.info(
            f"1SE rule: selected '{selected_name}' (RMSE={results_df.loc[selected_idx, 'CV Mean RMSE']:.4f}) "
            f"over '{best_name}' (RMSE={best_mean:.4f}±{best_se:.4f}, threshold={threshold:.4f})"
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
    # Always use ND path — transform_variables() uses PolynomialFeatures
    # naming (spaces, not underscores) so variable generation must match.
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
            # Include all terms (main effects + higher-order) to respect
            # the principle of marginality — a polynomial model without
            # lower-order terms is numerically unstable and uninterpretable.
            feature_names = [n for n in names if n not in variables]
            feature_sets[label] = var_reg + feature_names

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
    outlier_stats: tuple[float, float] | None = None,
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
        outlier_stats: Optional (median, MAD) tuple computed from training data.
            When provided, these statistics are used for outlier removal instead
            of computing them from this dataset. This prevents data leakage when
            processing validation folds in cross-validation.

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

    # ---- 5. Winsorize outlier bins by z-score on the target variable ----
    # #56: the target-based outlier step used to DROP bins whose realized risk
    # (the dependent variable) was extreme — overwhelmingly the high-risk tail on
    # right-skewed credit-risk data. Dropping them let the smooth fitted surface
    # extrapolate LOWER risk into exactly the cells where the accept/reject cutoff
    # is most sensitive (anti-conservative). Winsorize instead: clip an extreme
    # bin's target to the ±z_threshold boundary so EVERY observed bin keeps
    # influencing the fit, while a single corrupt bin is still bounded. Set
    # z_threshold == 0 to disable entirely.
    if z_threshold > 0 and len(processed_data) > 2:
        target_vals = processed_data[target_var]

        if outlier_stats is not None:
            # Use pre-computed stats from training data (prevents CV leakage)
            target_median, target_mad = outlier_stats
        else:
            # Compute stats from this dataset (used for non-CV paths)
            target_median = target_vals.median()
            target_mad = np.median(np.abs(target_vals - target_median))
            if target_mad == 0:
                target_mad = np.mean(np.abs(target_vals - target_median))

        if target_mad > 0:
            # |z| >= z_threshold  <=>  |target - median| >= z_threshold * MAD / 0.6745
            bound = z_threshold * target_mad / 0.6745
            lower, upper = target_median - bound, target_median + bound
            n_clipped = int(((target_vals < lower) | (target_vals > upper)).sum())
            if n_clipped > 0:
                logger.debug(
                    f"process_dataset: winsorized {n_clipped} outlier bin(s) to "
                    f"[{lower:.4f}, {upper:.4f}] (|z| >= {z_threshold}); all bins retained."
                )
                processed_data = processed_data.copy()
                processed_data[target_var] = target_vals.clip(lower, upper)
        else:
            logger.debug(
                f"process_dataset: target MAD=0 (all {len(target_vals)} bins have identical target); "
                f"skipping outlier winsorization"
            )

    return processed_data


def _check_model_discriminates(
    model, agg: pd.DataFrame, features: list, target_var: str, model_name: str, train_r2: float
) -> None:
    """Raise :class:`DegenerateModelError` if the model has no discriminatory power.

    A risk model whose predictions are (near-)constant across the score grid gives
    the MILP a FLAT risk surface — every cell has the same predicted risk, so
    acceptance is driven only by production/exposure and the "optimization" is
    meaningless. This used to ship silently (the pipeline merely logged
    ``Train R²: 0.0000``). The classic cause is a tree booster forced to a single
    leaf when a fixed ``min_child_weight`` / ``min_child_samples`` search range
    exceeds the small bin count (also fixed at the source via dynamic leaf bounds
    in ``optuna_tuning``). The check is model-agnostic — it catches zero-split
    boosters and any other constant model.
    """
    if len(agg) < 2:
        return
    preds = np.asarray(model.predict(prepare_model_input(agg, features, model)), dtype=float)
    pred_range = float(np.ptp(preds)) if preds.size else 0.0
    target_range = float(np.ptp(agg[target_var]))
    if pred_range < max(1e-9, _DEGENERATE_PRED_RANGE_FRAC * target_range):
        raise DegenerateModelError(
            f"Selected risk model '{model_name}' produces (near-)constant predictions across "
            f"{len(agg)} score bins (prediction range {pred_range:.3e} vs target range {target_range:.3e}, "
            f"train R²={train_r2:.2e}) — a flat risk surface with no discriminatory power, which makes the "
            "downstream optimization meaningless. Likely causes: too few bins for the tuned tree leaf bounds, "
            "or no bin-level signal in the data. Investigate the grid/data, or restrict the model family "
            "(e.g. linear-only)."
        )


def compute_outlier_stats(processed_data: pd.DataFrame, target_var: str) -> tuple[float, float]:
    """Compute outlier detection statistics (median, MAD) from a processed dataset.

    These stats should be computed on the training fold and passed to
    ``process_dataset(outlier_stats=...)`` for validation folds to prevent
    data leakage during cross-validation.
    """
    target_vals = processed_data[target_var].dropna()
    target_median = target_vals.median()
    target_mad = float(np.median(np.abs(target_vals - target_median)))
    if target_mad == 0:
        target_mad = float(np.mean(np.abs(target_vals - target_median)))
    return (float(target_median), target_mad)


def _get_regression_weights(processed_data: pd.DataFrame) -> pd.Series | None:
    for col in ("todu_amt_pile_h6", "oa_amt_h0", "n_observations"):
        if col in processed_data.columns:
            w = processed_data[col].copy()
            # Validate: non-negativity, clip extremes, normalize (#20)
            w = w.clip(lower=0)
            if w.sum() == 0:
                return None
            # Clip outliers at 99th percentile to prevent one bin from dominating
            cap = w.quantile(0.99)
            if cap > 0:
                w = w.clip(upper=cap)
            # Normalize so weights sum to N (sklearn expects relative scale)
            n = len(w)
            w = w * (n / w.sum())
            return w
    return None


def _weight_sizeref(*datasets: pd.DataFrame, max_marker_size: float = 22.0) -> float | None:
    """Plotly area-mode sizeref so the largest oa_amt_h0 maps to max_marker_size px.

    Returns None when any dataset lacks the weight column or all weights are
    non-positive, signalling callers to fall back to fixed-size markers.
    """
    weight_cols = [d[Columns.OA_AMT_H0].astype(float).clip(lower=0) for d in datasets if Columns.OA_AMT_H0 in d.columns]
    if len(weight_cols) != len(datasets):
        return None
    w_max = max(c.max() for c in weight_cols)
    if not np.isfinite(w_max) or w_max <= 0:
        return None
    return 2.0 * w_max / (max_marker_size**2)


def _weighted_marker(data: pd.DataFrame, color: str, sizeref: float | None) -> tuple[dict, list[str] | None]:
    """Marker dict with size proportional to oa_amt_h0, plus hover text with the value.

    Falls back to a fixed-size marker (no hover text) when sizeref is None.
    """
    if sizeref is None:
        return dict(size=5, color=color, opacity=0.7), None
    weights = data[Columns.OA_AMT_H0].astype(float).clip(lower=0)
    marker = dict(size=weights, sizemode="area", sizeref=sizeref, sizemin=3, color=color, opacity=0.7)
    hover = [f"{Columns.OA_AMT_H0}: {w:,.0f}" for w in weights]
    return marker, hover


def _create_feature_dataframe(mesh_df: pd.DataFrame, features: list[str], var0: str, var1: str | None) -> pd.DataFrame:
    """
    Create feature DataFrame for model predictions on a mesh grid.

    Uses the same ``transform_variables`` function that generates training
    features, ensuring the prediction surface is consistent with the model's
    training data regardless of which feature set was selected.

    Args:
        mesh_df: DataFrame with mesh grid points (columns: var0, and optionally var1).
        features: List of feature column names the model expects.
        var0: First variable name.
        var1: Second variable name (None for 1D case).

    Returns:
        DataFrame containing only the requested feature columns, in order.
    """
    variables = [var0] if var1 is None else [var0, var1]
    transformed = transform_variables(mesh_df.copy(), variables)
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
        var0 = variables[0]

        # 1D case: 2D scatter + line plot instead of 3D surface
        if len(variables) < 2:
            logger.info(f"Creating 2D plot for {var0}, {target_var}")
            x_min = min(train_data[var0].min(), test_data[var0].min())
            x_max = max(train_data[var0].max(), test_data[var0].max())
            x_range = np.linspace(x_min, x_max, n_points)
            mesh_df = pd.DataFrame({var0: x_range})
            feature_df = _create_feature_dataframe(mesh_df, features, var0, None)
            y_pred = model.predict(feature_df)

            sizeref = _weight_sizeref(train_data, test_data)
            train_marker, train_hover = _weighted_marker(train_data, styles.COLOR_ACCENT, sizeref)
            test_marker, test_hover = _weighted_marker(test_data, styles.COLOR_RISK, sizeref)
            name_suffix = f" (size ∝ {Columns.OA_AMT_H0})" if sizeref is not None else ""

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=train_data[var0],
                    y=train_data[target_var],
                    mode="markers",
                    marker=train_marker,
                    text=train_hover,
                    name=f"Training Data{name_suffix}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_data[var0],
                    y=test_data[target_var],
                    mode="markers",
                    marker=test_marker,
                    text=test_hover,
                    name=f"Test Data{name_suffix}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_pred,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0.8)", width=2),
                    name="Model Predictions",
                )
            )
            styles.apply_plotly_style(
                fig, title=f"{target_var} vs {var0} with Model Predictions", width=1000, height=600
            )
            fig.update_xaxes(title_text=var0)
            fig.update_yaxes(title_text=target_var)
            return fig

        var1 = variables[1]
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

        sizeref = _weight_sizeref(train_data, test_data, max_marker_size=18.0)
        train_marker, train_hover = _weighted_marker(train_data, styles.COLOR_ACCENT, sizeref)
        test_marker, test_hover = _weighted_marker(test_data, styles.COLOR_RISK, sizeref)
        name_suffix = f" (size ∝ {Columns.OA_AMT_H0})" if sizeref is not None else ""

        # Create the 3D plot
        fig = go.Figure(
            data=[
                # Training data
                go.Scatter3d(
                    x=train_data[var0],
                    y=train_data[var1],
                    z=train_data[target_var],
                    mode="markers",
                    marker=train_marker,
                    text=train_hover,
                    name=f"Training Data{name_suffix}",
                ),
                # Test data
                go.Scatter3d(
                    x=test_data[var0],
                    y=test_data[var1],
                    z=test_data[target_var],
                    mode="markers",
                    marker=test_marker,
                    text=test_hover,
                    name=f"Test Data{name_suffix}",
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
                # Start at the actual minimum bin (x_min/y_min), not a hard-coded 1 — the old
                # range clipped bin 0 (a real score band) out of the surface entirely.
                xaxis=dict(range=[x_min, max(x_min, x_max)]),
                yaxis=dict(range=[y_min, max(y_min, y_max)]),
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
    display_cols = ["Model", "CV Mean RMSE", "CV Std RMSE"]
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
        "cv_mean_rmse": best_row["CV Mean RMSE"],
        "cv_std_rmse": best_row["CV Std RMSE"],
    }

    logger.info(f"Best model type: {best_model_info['name']}")
    logger.info(f"  CV RMSE: {best_model_info['cv_mean_rmse']:.4f} ± {best_model_info['cv_std_rmse']:.4f}")

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

    feature_results = []

    for feature_name, features in feature_sets.items():
        cv_scores = []
        cv_r2_scores = []
        splits = _stratified_cv_splitter(raw_data, cv_folds, DEFAULT_RANDOM_STATE, indicators)
        for train_idx, val_idx in splits:
            raw_train, raw_val = raw_data.iloc[train_idx].copy(), raw_data.iloc[val_idx].copy()

            train_agg = process_dataset(
                raw_train, bins, variables, indicators, target_var, multiplier, features, z_threshold
            )
            # Compute outlier stats from training fold and apply to validation
            # to prevent data leakage in cross-validation.
            train_outlier_stats = compute_outlier_stats(train_agg, target_var) if len(train_agg) > 2 else None
            val_agg = process_dataset(
                raw_val,
                bins,
                variables,
                indicators,
                target_var,
                multiplier,
                features,
                z_threshold,
                outlier_stats=train_outlier_stats,
            )

            missing_train = [f for f in features if f not in train_agg.columns]
            if missing_train:
                continue

            # Skip fold if validation has too few bins for reliable R²
            if len(val_agg) < 3:
                logger.debug(f"Fold skipped for {feature_name}: only {len(val_agg)} validation bins")
                continue

            X_train = prepare_model_input(train_agg, features, model_template)
            X_val = prepare_model_input(val_agg, features, model_template)
            y_train, y_val = train_agg[target_var], val_agg[target_var]

            w_train = _get_regression_weights(train_agg)
            w_val = _get_regression_weights(val_agg)

            model_clone = clone(model_template)
            if isinstance(model_template, HurdleRegressor) and "_hurdle_r" in raw_train.columns:
                # Per-loan training (audit #6); score on aggregated val like the other candidates.
                raw_train_t = transform_variables(raw_train, variables)
                model_clone.fit(
                    prepare_model_input(raw_train_t, features, model_clone),
                    raw_train["_hurdle_r"].to_numpy(),
                    sample_weight=np.asarray(raw_train["_hurdle_w"], dtype=float),
                )
            else:
                model_clone.fit(X_train, y_train, sample_weight=w_train)
            y_pred = model_clone.predict(X_val)
            if len(y_val) >= 2:
                from sklearn.metrics import mean_squared_error, r2_score

                fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred, sample_weight=w_val))
                cv_scores.append(fold_rmse)
                fold_r2 = r2_score(y_val, y_pred, sample_weight=w_val)
                cv_r2_scores.append(fold_r2)

        if not cv_scores:
            logger.warning(f"Skipping {feature_name}: execution failed internally.")
            continue

        cv_mean = np.mean(cv_scores)
        # Nadeau–Bengio corrected SE (audit #32d): the naive std/sqrt(k) treats
        # correlated fold scores as independent and roughly halves the 1SE band.
        cv_std = _nadeau_bengio_cv_se(np.std(cv_scores, ddof=1), len(cv_scores)) if len(cv_scores) > 1 else 0.0
        # NaN (not 0.0) when R² is undefined: no scored folds, or a single fold (std undefined).
        # A stored 0.0 reads as "computed, no skill / perfectly stable" — a false signal.
        cv_r2_mean = np.mean(cv_r2_scores) if cv_r2_scores else float("nan")
        cv_r2_std = (
            _nadeau_bengio_cv_se(np.std(cv_r2_scores, ddof=1), len(cv_r2_scores))
            if len(cv_r2_scores) > 1
            else float("nan")
        )

        feature_results.append(
            {
                "Feature Set": feature_name,
                "Num Features": len(features),
                "Features": features,
                "CV Mean RMSE": cv_mean,
                "CV Std RMSE": cv_std,
                "CV Mean R2": cv_r2_mean,
                "CV Std R2": cv_r2_std,
            }
        )

        logger.info(f"  {feature_name} ({len(features)} features): CV RMSE = {cv_mean:.4f} ± {cv_std:.4f}")

    if not feature_results:
        raise RuntimeError("All feature sets failed or were skipped. Check data columns.")

    results_df = pd.DataFrame(feature_results)

    # Apply 1SE rule: pick simplest feature set within 1 SE of the best
    best_idx = _apply_one_se_rule(results_df, "Num Features")
    best_row = results_df.loc[best_idx]

    best_feature_info = {
        "feature_set_name": best_row["Feature Set"],
        "features": best_row["Features"],
        "cv_mean_rmse": best_row["CV Mean RMSE"],
        "cv_std_rmse": best_row["CV Std RMSE"],
        "cv_mean_r2": best_row.get("CV Mean R2", float("nan")),
        "cv_std_r2": best_row.get("CV Std R2", float("nan")),
    }

    # Log results table
    display_cols = ["Feature Set", "Num Features", "CV Mean RMSE", "CV Std RMSE", "CV Mean R2"]
    available_cols = [c for c in display_cols if c in results_df.columns]
    logger.info("Feature set CV results:")
    logger.info(f"\n{results_df[available_cols].to_string()}")
    logger.info(f"Best feature set: {best_feature_info['feature_set_name']}")
    logger.info(f"  CV RMSE: {best_feature_info['cv_mean_rmse']:.4f} ± {best_feature_info['cv_std_rmse']:.4f}")
    logger.info(f"  CV R²:   {best_feature_info['cv_mean_r2']:.4f} ± {best_feature_info['cv_std_r2']:.4f}")

    return results_df, best_feature_info


def _nadeau_bengio_cv_se(std: float, n_folds: int) -> float:
    """Nadeau–Bengio corrected standard error of a k-fold CV mean (audit #9).

    Cross-fold predictions are positively correlated (the folds share training data), so the naive
    ``std / sqrt(k)`` treats them as independent and underestimates the SE (typically by ~2x). The
    corrected resampled variance multiplies the per-fold sample variance by ``(1/k + n_test/n_train)``;
    for standard k-fold ``n_test/n_train = 1/(k-1)``, giving ``SE = std * sqrt(1/k + 1/(k-1))``. Falls
    back to the naive ``std / sqrt(k)`` when ``n_folds < 2`` (no correction term defined).
    """
    if n_folds < 2:
        return std / np.sqrt(max(n_folds, 1))
    return std * np.sqrt(1.0 / n_folds + 1.0 / (n_folds - 1))


def compute_cell_level_ci(
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    z_threshold: float,
    final_features: list[str],
    model_template,
    cv_folds: int = 5,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """K-fold CV per-cell prediction intervals (report-only).

    For each fold: train on train split, aggregate val split by bins,
    predict target_var per cell, record predictions.
    Then per cell: compute mean, std, and a CI from cross-fold predictions using the
    **Nadeau–Bengio corrected** SE (``std * sqrt(1/k + 1/(k-1))``, not the naive ``std/sqrt(k)``)
    to account for the positive correlation between folds (audit #9). These intervals are for
    human-facing reporting (dashboard / CSV); no cutoff or optimization logic consumes them.

    Args:
        raw_data: Raw (un-aggregated) booked data.
        bins: Bin definitions.
        variables: Grouping variable names.
        indicators: Indicator columns to aggregate.
        target_var: Target variable name.
        multiplier: Multiplier for target calculation.
        z_threshold: Outlier z-score threshold.
        final_features: Feature columns for the model.
        model_template: Unfitted sklearn estimator (will be cloned).
        cv_folds: Number of cross-validation folds.
        confidence_level: Confidence level for CI (default 0.95).

    Returns:
        DataFrame with columns: var0, var1, ..., pred_mean, pred_std,
        ci_lower, ci_upper, n_folds_observed.
        Cells with <2 fold observations get NaN CIs.
    """
    from collections import defaultdict

    from scipy import stats
    from sklearn.base import clone

    splits = _stratified_cv_splitter(raw_data, cv_folds, DEFAULT_RANDOM_STATE, indicators)
    cell_predictions: dict[tuple, list[float]] = defaultdict(list)

    for train_idx, val_idx in splits:
        raw_train = raw_data.iloc[train_idx].copy()
        raw_val = raw_data.iloc[val_idx].copy()

        train_agg = process_dataset(
            raw_train, bins, variables, indicators, target_var, multiplier, final_features, z_threshold
        )
        # Compute outlier stats from training fold and apply to validation
        # to prevent data leakage in cross-validation.
        train_outlier_stats = compute_outlier_stats(train_agg, target_var) if len(train_agg) > 2 else None
        val_agg = process_dataset(
            raw_val,
            bins,
            variables,
            indicators,
            target_var,
            multiplier,
            final_features,
            z_threshold,
            outlier_stats=train_outlier_stats,
        )

        missing = [f for f in final_features if f not in train_agg.columns]
        if missing or len(val_agg) < 1:
            continue

        X_train = prepare_model_input(train_agg, final_features, model_template)
        y_train = train_agg[target_var]
        w_train = _get_regression_weights(train_agg)

        model_clone = clone(model_template)
        model_clone.fit(X_train, y_train, sample_weight=w_train)

        X_val = prepare_model_input(val_agg, final_features, model_template)
        preds = model_clone.predict(X_val)

        # Record predictions per cell (keyed by variable tuple)
        for i, (_, row) in enumerate(val_agg.iterrows()):
            cell_key = tuple(row[v] for v in variables)
            cell_predictions[cell_key].append(float(preds[i]))

    # Build result DataFrame
    alpha = 1 - confidence_level
    z_val = stats.norm.ppf(1 - alpha / 2)

    rows = []
    for cell_key, preds_list in cell_predictions.items():
        row = {variables[d]: cell_key[d] for d in range(len(variables))}
        n_folds = len(preds_list)
        row["n_folds_observed"] = n_folds
        row["pred_mean"] = np.mean(preds_list)

        if n_folds >= 2:
            std = np.std(preds_list, ddof=1)
            se = _nadeau_bengio_cv_se(std, n_folds)  # corrected for inter-fold correlation (audit #9)
            crit_val = stats.t.ppf(1 - alpha / 2, df=n_folds - 1) if n_folds < 30 else z_val
            row["pred_std"] = std
            row["ci_lower"] = row["pred_mean"] - crit_val * se
            row["ci_upper"] = row["pred_mean"] + crit_val * se
        else:
            row["pred_std"] = np.nan
            row["ci_lower"] = np.nan
            row["ci_upper"] = np.nan

        rows.append(row)

    return (
        pd.DataFrame(rows)
        if rows
        else pd.DataFrame(columns=variables + ["pred_mean", "pred_std", "ci_lower", "ci_upper", "n_folds_observed"])
    )


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
    booked_data = booked_data.dropna(subset=req_cols)

    # Per-loan hurdle training targets (audit #6). `_hurdle_r` = per-loan b2 ratio (= 0 for
    # non-defaulting loans → real zero mass the hurdle's classifier needs); `_hurdle_w` = exposure
    # weight so the severity stage is exposure-weighted and the cell prediction reconciles to the
    # exposure-weighted aggregate b2 = mult·Σnum/Σden. These extra columns are invisible to
    # `process_dataset` (it only sums the named `indicators`), so the aggregated candidates are
    # unaffected; they are consumed only when a HurdleRegressor is trained per-loan.
    num_col, den_col = "todu_30ever_h6", "todu_amt_pile_h6"
    if num_col in booked_data.columns and den_col in booked_data.columns:
        booked_data["_hurdle_r"] = calculate_b2_ever_h6(
            booked_data[num_col], booked_data[den_col], multiplier=multiplier
        ).fillna(0.0)
        if booked_data[den_col].abs().sum() > 0:
            w = booked_data[den_col].astype(float)
        elif "oa_amt_h0" in booked_data.columns:
            w = booked_data["oa_amt_h0"].astype(float)
        else:
            w = pd.Series(1.0, index=booked_data.index)
        booked_data["_hurdle_w"] = w.clip(lower=0.0).fillna(0.0)

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
    combined_results = combined_results.sort_values("CV Mean RMSE", ascending=True)

    logger.info("Combined Model CV results:")
    display_cols = ["Model", "CV Mean RMSE", "CV Std RMSE"]
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
            "cv_mean_rmse": best_row["CV Mean RMSE"],
            "cv_std_rmse": best_row["CV Std RMSE"],
        }

        # Simulate results of step 2 for tree models
        results_step2 = pd.DataFrame(
            [
                {
                    "Feature Set": "original",
                    "Num Features": len(variables),
                    "Features": variables,
                    "CV Mean RMSE": best_row["CV Mean RMSE"],
                    "CV Std RMSE": best_row["CV Std RMSE"],
                }
            ]
        )

        best_feature_info = {
            "feature_set_name": "original",
            "features": variables,
            "cv_mean_rmse": best_row["CV Mean RMSE"],
            "cv_std_rmse": best_row["CV Std RMSE"],
            # Tree models are tuned/selected on RMSE only — CV R² is NOT computed for them, so the
            # column is absent here. Record NaN ("not computed"), never 0.0: a stored 0.0 is
            # indistinguishable from a genuinely flat model (R²≈0, the #65 degenerate case).
            "cv_mean_r2": best_row.get("CV Mean R2", float("nan")),
            "cv_std_r2": best_row.get("CV Std R2", float("nan")),
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


def evaluate_holdout_rmse(
    model_template,
    raw_data: pd.DataFrame,
    bins: tuple,
    variables: list[str],
    indicators: list[str],
    target_var: str,
    multiplier: float,
    features: list[str],
    z_threshold: float,
    random_state: int,
    test_size: float = 0.2,
) -> float | None:
    """POST-SELECTION held-out RMSE for the SELECTED model only (audit #7, honesty per #32b).

    A single stratified train/test split is carved here — but from the SAME data every selection CV
    fold ran on, so the held-out rows participated in choosing the model type/hyperparameters/feature
    set. Refitting on the 80% reduces — but does not remove — selection optimism; do NOT read this as
    an unbiased generalization estimate (the M4 out-of-time backtest is that). The model is refit on
    the train portion and scored (exposure-weighted RMSE) on the independently-aggregated test portion
    (train-derived outlier stats applied to test → no within-split leakage). Returns ``None`` when the
    data is too small. The split is independent of, and never reused by, the k-fold selection metric.
    """
    from sklearn.base import clone
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    if len(raw_data) < 200:
        return None
    # Same stratification basis as the selection CV splitter (audit #33): the
    # default indicator, not indicators[0] (~all 1s on booked → never stratified).
    stratify = _stratify_labels(raw_data, indicators, min_per_class=5)
    try:
        train_raw, test_raw = train_test_split(
            raw_data, test_size=test_size, random_state=random_state, stratify=stratify
        )
        train_agg = process_dataset(
            train_raw, bins, variables, indicators, target_var, multiplier, features, z_threshold
        )
        outlier_stats = compute_outlier_stats(train_agg, target_var) if len(train_agg) > 2 else None
        test_agg = process_dataset(
            test_raw,
            bins,
            variables,
            indicators,
            target_var,
            multiplier,
            features,
            z_threshold,
            outlier_stats=outlier_stats,
        )
        if len(train_agg) < 3 or len(test_agg) < 3:
            return None
        m = clone(model_template)
        if isinstance(m, HurdleRegressor) and "_hurdle_r" in train_raw.columns:
            # Mirror the per-loan hurdle fit (audit #6) so the report reflects the real model.
            train_t = transform_variables(train_raw, variables)
            m.fit(
                prepare_model_input(train_t, features, m),
                train_raw["_hurdle_r"].to_numpy(),
                sample_weight=np.asarray(train_raw["_hurdle_w"], dtype=float),
            )
        else:
            m.fit(
                prepare_model_input(train_agg, features, m),
                train_agg[target_var],
                sample_weight=_get_regression_weights(train_agg),
            )
        pred = m.predict(prepare_model_input(test_agg, features, m))
        return float(
            np.sqrt(mean_squared_error(test_agg[target_var], pred, sample_weight=_get_regression_weights(test_agg)))
        )
    except Exception:
        logger.debug("Winner held-out RMSE evaluation failed", exc_info=True)
        return None


def _train_and_evaluate_final_model(
    best_model_template,
    agg_data: pd.DataFrame,
    final_features: list[str],
    target_var: str,
    weights: pd.Series | None,
    per_loan_data: pd.DataFrame | None = None,
    variables: list[str] | None = None,
) -> tuple[Any, float]:
    """Clone, fit, predict, compute R², return model + train_r2.

    For a HurdleRegressor with per-loan data available (audit #6), fit on PER-LOAN rows (real zero
    mass); R² is still computed against the aggregated cells so it is comparable to other models.
    """
    from sklearn.base import clone

    final_model = clone(best_model_template)
    y_all = agg_data[target_var]
    X_all = prepare_model_input(agg_data, final_features, final_model)

    if (
        isinstance(final_model, HurdleRegressor)
        and per_loan_data is not None
        and variables is not None
        and "_hurdle_r" in per_loan_data.columns
    ):
        pl_t = transform_variables(per_loan_data, variables)
        final_model.fit(
            prepare_model_input(pl_t, final_features, final_model),
            per_loan_data["_hurdle_r"].to_numpy(),
            sample_weight=np.asarray(per_loan_data["_hurdle_w"], dtype=float),
        )
    else:
        final_model.fit(X_all, y_all, sample_weight=weights)

    y_all_pred = final_model.predict(X_all)

    if len(y_all) >= 2:
        train_r2 = r2_score(y_all, y_all_pred, sample_weight=weights)
    else:
        train_r2 = float("nan")

    return final_model, train_r2


def _compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: list[str],
) -> dict | None:
    """Compute SHAP values for the trained model. Returns dict or None on failure."""
    try:
        import shap

        # Bound SHAP workload for stability on large datasets.
        max_explain_rows = 500
        X_explain = X if len(X) <= max_explain_rows else shap.sample(X, max_explain_rows)

        if hasattr(model, "coef_"):
            # Linear models: exact and fast
            explainer = shap.LinearExplainer(model, X_explain)
            shap_values = explainer.shap_values(X_explain)
        elif type(model).__name__ in ("XGBRegressor", "LGBMRegressor"):
            # Tree models: exact and fast
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_explain)
            except ValueError as ve:
                if "could not convert string to float" in str(ve):
                    logger.warning(
                        "XGBoost/SHAP TreeExplainer incompatibility detected. Falling back to KernelExplainer."
                    )
                    background = shap.sample(X_explain, min(50, len(X_explain)))

                    def safe_predict_tree(data):
                        return model.predict(pd.DataFrame(data, columns=feature_names))

                    explainer = shap.KernelExplainer(safe_predict_tree, background)
                    shap_values = explainer.shap_values(X_explain, nsamples=50)
                else:
                    raise
        else:
            # Non-linear models (Hurdle, Tweedie): use sampling-based explainer
            # Use a small background sample for efficiency
            background = shap.sample(X_explain, min(50, len(X_explain)))

            # KernelExplainer wrapper to ensure input is a DataFrame with correct feature names
            def safe_predict(data):
                return model.predict(pd.DataFrame(data, columns=feature_names))

            explainer = shap.KernelExplainer(safe_predict, background)
            # Using a very small sample size for KernelExplainer to ensure it finishes quickly and doesn't crash on high dims
            shap_values = explainer.shap_values(X_explain, nsamples=50)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        logger.info(f"SHAP values computed: {shap_values.shape} (explained_rows={len(X_explain)}/{len(X)})")

        return {
            "shap_values": shap_values,
            "feature_names": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }

    except Exception:
        import traceback

        err_msg = traceback.format_exc()
        logger.error(f"SHAP computation failed (non-blocking):\n{err_msg}")
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
    model_variables: list[str] | None = None,
    bin_edges: tuple | None = None,
) -> str:
    """Build metadata, save model with metadata, return path."""
    model_metadata = {
        "cv_mean_rmse": best_model_info["cv_mean_rmse"],
        "cv_std_rmse": best_model_info["cv_std_rmse"],
        "cv_mean_r2": best_model_info.get("cv_mean_r2", 0.0),
        "cv_std_r2": best_model_info.get("cv_std_r2", 0.0),
        "train_r2": best_model_info["train_r2"],
        "full_r2": best_model_info["train_r2"],  # Full and train R2 are identical because final model uses all data
        "holdout_rmse": best_model_info.get("holdout_rmse"),  # POST-selection winner-only 20% report (#7/#32b)
        "cv_folds": cv_folds,
        "total_samples": len(all_data),
        "target_variable": target_var,
        "multiplier": multiplier,
        "model_type_selected": best_model_info["model_type"],
        "feature_set_selected": best_model_info["feature_set"],
        "weighted_regression": best_model_info["weighted"],
        "is_hurdle": best_model_info["is_hurdle"],
        "zero_proportion": float(zero_prop),
        "step1_cv_rmse": best_model_type["cv_mean_rmse"],
        "step2_cv_rmse": best_feature_info["cv_mean_rmse"],
    }

    if model_variables is not None:
        model_metadata["model_variables"] = model_variables

    # Persist the grid identity so a reused model can be validated against the current config (#40):
    # the model learned "bin index → risk" for THESE edges, so reusing it under different edges maps
    # the same cell indices onto different score regions. Stored per model_variable, aligned by order.
    if bin_edges is not None and model_variables is not None:
        model_metadata["bin_edges"] = {var: [float(e) for e in edges] for var, edges in zip(model_variables, bin_edges)}

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
    include_hurdle: bool = False,
    save_model: bool = True,
    model_base_path: str = "models",
    create_visualizations: bool = True,
    directions: dict[str, int] | None = None,
    z_threshold: float = DEFAULT_Z_THRESHOLD,
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
        include_hurdle: Whether to offer the (per-loan-trained) Hurdle candidate (default: False; audit #6).
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
        - best_model_info: Final model details (model, name, cv_mean_rmse,
          cv_std_rmse, train_r2, model_type, feature_set, weighted, is_hurdle)
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
        all_data, bins, variables, indicators, target_var, multiplier, var_reg, z_threshold=z_threshold
    )
    weights_all = _get_regression_weights(final_agg)
    # Zero mass for the hurdle (#6): measured PER-LOAN on the default-amount numerator, where real
    # zeros exist (non-defaulting loans). The bin-aggregated target ratio is ~never exactly 0, so
    # the old post-aggregation diagnostic was misleading. This per-loan value is what makes (or
    # doesn't make) a two-part hurdle meaningful.
    if "todu_30ever_h6" in all_data.columns and len(all_data):
        zero_prop = float((all_data["todu_30ever_h6"] <= 0).mean())
    else:
        zero_prop = float((np.abs(final_agg[target_var]) < 1e-10).mean())

    logger.info(f"Processed full data scope: {final_agg.shape[0]} groups")
    logger.info(f"Per-loan zero proportion (no-default loans): {zero_prop:.1%}")
    if weights_all is not None:
        logger.info(f"Weight range: [{weights_all.min():.0f}, {weights_all.max():.0f}]")

    # Degenerate-zero-mass gate (#6): a two-part hurdle is only meaningful when the per-loan default
    # indicator has real two-class signal. If essentially all loans default (or none do), the
    # classifier stage is degenerate — skip the hurdle candidate even if requested.
    if include_hurdle and not (0.02 <= zero_prop <= 0.999):
        logger.warning(
            f"Hurdle requested but per-loan zero mass ({zero_prop:.1%}) is degenerate "
            f"(outside [2%, 99.9%]); skipping the hurdle candidate."
        )
        include_hurdle = False

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
        z_threshold=z_threshold,
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
        best_model_type["model_template"],
        final_agg,
        final_features,
        target_var,
        weights_all,
        per_loan_data=all_data,
        variables=variables,
    )

    logger.info(f"Final model: {best_model_name} + {best_feature_info['feature_set_name']}")
    logger.info(f"  CV RMSE: {best_feature_info['cv_mean_rmse']:.4f} ± {best_feature_info['cv_std_rmse']:.4f}")
    logger.info(f"  Train R² (in-sample): {train_r2:.4f}")

    # Hard gate against a no-discriminatory-power risk model (constant-model bug).
    _check_model_discriminates(final_model, final_agg, final_features, target_var, best_model_name, train_r2)

    if hasattr(final_model, "coef_"):
        logger.debug("Model Coefficients:")
        for feature, coef in zip(final_features, final_model.coef_):
            logger.debug(f"  {feature}: {coef:.6f}")
    elif hasattr(final_model, "regressor_") and hasattr(final_model.regressor_, "coef_"):
        logger.debug("Hurdle Model - Regression Stage Coefficients:")
        for feature, coef in zip(final_features, final_model.regressor_.coef_):
            logger.debug(f"  {feature}: {coef:.6f}")

    # Winner-only held-out report (audit #7): a fresh 20% split scored on the SELECTED model only,
    # never used for selection — an unbiased generalization estimate for the model-risk record.
    holdout_rmse = evaluate_holdout_rmse(
        best_model_type["model_template"],
        all_data,
        bins,
        variables,
        indicators,
        target_var,
        multiplier,
        final_features,
        DEFAULT_Z_THRESHOLD,
        DEFAULT_RANDOM_STATE,
    )
    if holdout_rmse is not None:
        logger.info(
            f"  Winner held-out RMSE (20%, post-selection — selection saw these rows; "
            f"reduced, not zero, optimism): {holdout_rmse:.4f}"
        )

    best_model_info = {
        "model": final_model,
        "name": f"{best_model_name} + {best_feature_info['feature_set_name']}",
        "cv_mean_rmse": best_feature_info["cv_mean_rmse"],
        "cv_std_rmse": best_feature_info["cv_std_rmse"],
        "cv_mean_r2": best_feature_info.get("cv_mean_r2", 0.0),
        "cv_std_r2": best_feature_info.get("cv_std_r2", 0.0),
        "train_r2": train_r2,
        "holdout_rmse": holdout_rmse,
        "model_type": best_model_name,
        "feature_set": best_feature_info["feature_set_name"],
        "weighted": weights_all is not None,
        "is_hurdle": isinstance(final_model, HurdleRegressor),
    }

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
            model_variables=variables,
            bin_edges=bins,
        )

    # SHAP interpretability (non-blocking)
    if model_path:
        shap_result = _compute_shap_values(final_model, final_agg[final_features], final_features)
        if shap_result is not None:
            best_model_info["shap_values"] = shap_result["shap_values"]
            best_model_info["shap_feature_names"] = shap_result["feature_names"]
            best_model_info["mean_abs_shap"] = shap_result["mean_abs_shap"]

            # Export SHAP summary plot (non-blocking)
            try:
                from src.plots import plot_shap_dependence, plot_shap_summary

                images_dir = Path(model_path).parent.parent / "images"
                images_dir.mkdir(parents=True, exist_ok=True)

                shap_plot_path = str(images_dir / "shap_summary.html")
                plot_shap_summary(
                    shap_result["shap_values"],
                    shap_result["feature_names"],
                    output_path=shap_plot_path,
                )
                logger.info(f"SHAP summary plot saved to {shap_plot_path}")

                # Top features for dependence plots
                mean_abs_shap = shap_result["mean_abs_shap"]
                feature_names = shap_result["feature_names"]
                top_indices = np.argsort(mean_abs_shap)[::-1]
                n_top_features = min(3, len(feature_names))

                for i in range(n_top_features):
                    top_feature_idx = top_indices[i]
                    top_feature_name = feature_names[top_feature_idx]

                    # Sanitize feature name for filename
                    safe_feature_name = "".join(c for c in top_feature_name if c.isalnum() or c in "._-").rstrip()
                    dependence_path = str(images_dir / f"shap_dependence_{safe_feature_name}.html")

                    plot_shap_dependence(
                        shap_values=shap_result["shap_values"],
                        feature_names=feature_names,
                        feature_data=final_agg[final_features],
                        target_feature=top_feature_name,
                        output_path=dependence_path,
                    )
                    logger.debug(f"SHAP dependence plot for '{top_feature_name}' saved to {dependence_path}")

            except (ImportError, ValueError, OSError) as e:
                logger.warning(f"SHAP plot export failed (non-blocking): {e}")

    # CELL-LEVEL CI (non-blocking)
    try:
        cell_ci_df = compute_cell_level_ci(
            raw_data=all_data,
            bins=bins,
            variables=variables,
            indicators=indicators,
            target_var=target_var,
            multiplier=multiplier,
            z_threshold=z_threshold,
            final_features=final_features,
            model_template=best_model_type["model_template"],
            cv_folds=cv_folds,
        )
        if not cell_ci_df.empty and model_path:
            ci_path = Path(model_path).parent / "cell_level_ci.csv"
            cell_ci_df.to_csv(ci_path, index=False)
            logger.info(f"Cell-level CI saved to {ci_path} ({len(cell_ci_df)} cells)")
            best_model_info["cell_ci_path"] = str(ci_path)
    except Exception as e:
        logger.warning(f"Cell-level CI computation failed (non-blocking): {e}")

    # STEP 6: VISUALIZATION
    fig = None
    if create_visualizations and len(variables) >= 2:
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
    logger.info(f"  Model type (Step 1): {best_model_name} (CV RMSE: {best_model_type['cv_mean_rmse']:.4f})")
    logger.info(
        f"  Feature set (Step 2): {best_feature_info['feature_set_name']} "
        f"(CV RMSE: {best_feature_info['cv_mean_rmse']:.4f} ± {best_feature_info['cv_std_rmse']:.4f})"
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
        "model_variables": variables,
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
    )
    n_before_dropna = len(df_grouped)
    df_grouped = df_grouped.dropna()
    n_dropped = n_before_dropna - len(df_grouped)
    if n_dropped > 0:
        logger.warning(f"Dropped {n_dropped}/{n_before_dropna} bin(s) with NaN in aggregated indicators")

    if df_grouped.empty:
        logger.warning("No data found after filtering for 'booked' status.")
        return go.Figure(), None, 0.0

    # 3. Outlier Removal
    # ---------------------------------------------------------
    # Use robust MAD-based z-score (consistent with process_dataset)
    # Standard zscore is sensitive to the very outliers it's trying to detect
    median_val = df_grouped[target_col].median()
    mad = np.median(np.abs(df_grouped[target_col] - median_val))
    if mad > 0:
        z_scores = 0.6745 * np.abs(df_grouped[target_col] - median_val) / mad
    else:
        z_scores = np.abs(df_grouped[target_col] - median_val)
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

    model = LinearRegression(fit_intercept=False)  # Zero loan amount → zero exposure (prevents negative predictions)
    model.fit(X, y)

    r_sq = model.score(X, y)
    coef = model.coef_[0]

    # LOO-CV R² for unbiased performance estimate
    if len(X) > 2:
        from sklearn.metrics import r2_score
        from sklearn.model_selection import LeaveOneOut, cross_val_predict

        loo_preds = cross_val_predict(LinearRegression(fit_intercept=False), X, y, cv=LeaveOneOut())
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
            # SHA-256 integrity sidecar (todo #44) — consumed by safe_joblib_load
            from src.persistence import write_integrity_sidecar

            write_integrity_sidecar(model_output_path)
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


def compute_pre_reject_inference_data(
    data_booked: pd.DataFrame,
    data_demand: pd.DataFrame,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stressor: float,
    *,
    indicators: list[str],
    variables: list[str],
    annual_coef: float,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    model_variables: list[str] | None = None,
    per_bin_stress: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute booked summary (_boo suffix) and repesca summary (post-risk-model, pre-reject-inference).

    These results are invariant to reject inference parameters and can be
    computed once then reused across multiple parameter evaluations.

    Args:
        data_booked: Booked-only records for the period.
        data_demand: Full demand population (booked + rejected).
        risk_inference: Dict with ``best_model_info``, ``features``, etc.
        reg_todu_amt_pile: Fitted regression model for exposure prediction.
        stressor: Global stress multiplier (use 1.0 when per_bin_stress is set).
        indicators: Indicator columns to aggregate (e.g. todu_30ever_h6).
        variables: Binning variable names for groupby.
        annual_coef: Annualization coefficient.
        multiplier: Risk multiplier (default 7).
        model_variables: Variables the risk model was trained on. When fewer
            than ``variables``, the model predicts using this subset while
            aggregation still groups by all ``variables``.
        per_bin_stress: Optional per-bin stress factors DataFrame with
            ``[*variables, "stress_factor"]``.  When provided, the scalar
            *stressor* should be 1.0 and per-bin factors are applied after
            the model prediction step.

    Returns:
        Tuple of (booked_summary with _boo suffix, repesca_summary post-risk-model).
    """
    final_model = risk_inference["best_model_info"]["model"]
    final_features = risk_inference["features"]
    risk_vars = model_variables or variables

    def _ensure_unique_merge_keys(
        df: pd.DataFrame, key_cols: list[str], value_cols: list[str], *, name: str
    ) -> pd.DataFrame:
        if not df.duplicated(key_cols).any():
            return df
        dup_count = int(df.duplicated(key_cols).sum())
        logger.warning(
            f"{name}: detected non-unique merge keys (duplicates={dup_count}); collapsing by mean over value columns."
        )
        return df.groupby(key_cols, as_index=False)[value_cols].mean()

    def _aggregate(data, status, reject_reason=None):
        filtered_data = data[data[Columns.STATUS_NAME] == status]
        if reject_reason:
            filtered_data = filtered_data[filtered_data[Columns.REJECT_REASON] == reject_reason]
        return (
            filtered_data.groupby(variables)
            .agg(dict.fromkeys(indicators, "sum"))
            .mul(annual_coef)
            .reset_index()[variables + indicators]
        )

    booked_summary = _aggregate(data_booked, StatusName.BOOKED.value)
    booked_summary = booked_summary.rename(columns={i: i + Suffixes.BOOKED for i in indicators})

    repesca_summary = _aggregate(data_demand, StatusName.REJECTED.value, RejectReason.SCORE.value)

    from src.models import calculate_risk_values

    repesca_summary = calculate_risk_values(
        repesca_summary, final_model, reg_todu_amt_pile, risk_vars, stressor, final_features, multiplier=multiplier
    )[variables + indicators]

    # Apply per-bin stress factors when provided (scalar stressor is 1.0 in this case)
    if per_bin_stress is not None and not per_bin_stress.empty:
        per_bin_stress = _ensure_unique_merge_keys(
            per_bin_stress.copy(), variables, ["stress_factor"], name="per_bin_stress"
        )
        n_before = len(repesca_summary)
        repesca_summary = repesca_summary.merge(per_bin_stress, on=variables, how="left", validate="one_to_one")
        if len(repesca_summary) != n_before:
            raise RuntimeError(
                f"per_bin_stress merge expanded rows unexpectedly (before={n_before}, after={len(repesca_summary)})"
            )
        global_fallback = per_bin_stress["stress_factor"].median()
        repesca_summary["stress_factor"] = repesca_summary["stress_factor"].fillna(global_fallback)
        sf = repesca_summary["stress_factor"]
        logger.info(
            f"Swap-in per-bin stress: min={sf.min():.4f}, avg={sf.mean():.4f}, max={sf.max():.4f}, bins={len(sf)}"
        )
        repesca_summary["todu_30ever_h6"] *= sf
        if "todu_30ever_h3" in repesca_summary.columns:
            repesca_summary["todu_30ever_h3"] *= sf
        # Keep stress_factor_applied for downstream diagnostics
        repesca_summary = repesca_summary.rename(columns={"stress_factor": "stress_factor_applied"})
    else:
        # Global stressor — tag all bins with the scalar value
        repesca_summary["stress_factor_applied"] = stressor
        if abs(stressor - 1.0) > 1e-6:
            logger.info(f"Swap-in global stress: {stressor:.4f} (applied to all bins)")

    return booked_summary, repesca_summary


def run_optimization_pipeline(
    data_booked: pd.DataFrame,
    data_demand: pd.DataFrame,
    risk_inference: dict,
    reg_todu_amt_pile: Any,
    stressor: float,
    tasa_fin: float,
    *,
    indicators: list[str],
    variables: list[str],
    annual_coef: float,
    b2_output_path: str = "images/b2_ever_h6_vs_octroi_and_risk_score.html",
    reject_inference_method: Literal["none", "parceling"] = "none",
    reject_uplift_factor: float = 1.5,
    reject_max_risk_multiplier: float = 3.0,
    reject_parceling_method: Literal["linear", "power", "sigmoid"] = "linear",
    reject_bayesian_smoothing: bool = False,
    reject_bayesian_prior_strength: float = 10.0,
    reject_enforce_monotonicity: bool = False,
    reject_include_all_rejections: bool = False,
    reject_acceptance_recent_months: int | None = None,
    reject_acceptance_decay_half_life_months: float | None = None,
    reject_acceptance_date_col: str = "mis_date",
    reject_apply_h3_multiplier: bool = False,
    reject_no_demand_anchor_percentile: float = 0.10,
    reject_confidence_scale: float = 10.0,
    multiplier: float = DEFAULT_RISK_MULTIPLIER,
    inv_vars: list[str] | None = None,
    per_bin_stress: pd.DataFrame | None = None,
    per_bin_tasa_fin: pd.DataFrame | None = None,
):
    """Run the optimization pipeline: aggregate data, apply risk models, reject inference, and generate visualizations.

    Args:
        data_booked: Booked-only records for the observation period.
        data_demand: Full demand population (booked + rejected + canceled).
        risk_inference: Dict with ``best_model_info``, ``features``, ``model_variables``.
        reg_todu_amt_pile: Fitted regression model for exposure (todu_amt_pile) prediction.
        stressor: Global stress multiplier (1.0 when per_bin_stress is used).
        tasa_fin: Global financing rate (fraction of demand that converts).
        indicators: Indicator columns to aggregate (summed per bin).
        variables: Binning variable names for the optimization grid.
        annual_coef: Annualization coefficient for risk scaling.
        b2_output_path: Path for the risk heatmap HTML output.
        reject_inference_method: ``"none"`` or ``"parceling"``.
        reject_uplift_factor: Scaling coefficient for reject-inference multiplier.
        reject_max_risk_multiplier: Upper cap for per-bin reject multiplier.
        reject_parceling_method: ``"linear"``, ``"power"``, or ``"sigmoid"``.
        reject_bayesian_smoothing: Apply Beta-Binomial smoothing to acceptance rates.
        reject_bayesian_prior_strength: Strength of Bayesian prior for smoothing.
        reject_enforce_monotonicity: Enforce monotonic reject multipliers across bins.
        reject_include_all_rejections: Include non-score rejections in acceptance denominator.
        reject_acceptance_recent_months: If set, compute acceptance rates using only the last N months.
        reject_acceptance_decay_half_life_months: If set, apply exponential time-decay to acceptance rates.
        reject_acceptance_date_col: Date column for temporal weighting in acceptance rates.
        reject_apply_h3_multiplier: Apply reject multiplier to H3 numerator alongside H6.
        multiplier: Risk multiplier (default 7 for H6).
        inv_vars: Variables with inverted monotonicity (higher bin = lower risk).
        per_bin_stress: Per-bin stress factors DataFrame.
        per_bin_tasa_fin: Per-bin financing rate DataFrame.
    """
    logger.info("Running optimization pipeline...")

    var_target = "b2_ever_h6"

    INDICADORES = indicators
    VARIABLES = variables

    def _ensure_unique_merge_keys(
        df: pd.DataFrame, key_cols: list[str], value_cols: list[str], *, name: str
    ) -> pd.DataFrame:
        if not df.duplicated(key_cols).any():
            return df
        dup_count = int(df.duplicated(key_cols).sum())
        logger.warning(
            f"{name}: detected non-unique merge keys (duplicates={dup_count}); collapsing by mean over value columns."
        )
        return df.groupby(key_cols, as_index=False)[value_cols].mean()

    # Compute invariant pre-reject-inference data
    data_sumary_desagregado_booked, data_sumary_desagregado_repesca = compute_pre_reject_inference_data(
        data_booked=data_booked,
        data_demand=data_demand,
        risk_inference=risk_inference,
        reg_todu_amt_pile=reg_todu_amt_pile,
        stressor=stressor,
        indicators=INDICADORES,
        variables=VARIABLES,
        annual_coef=annual_coef,
        multiplier=multiplier,
        model_variables=risk_inference.get("model_variables"),
        per_bin_stress=per_bin_stress,
    )

    # Apply reject inference adjustment (after stressor, before tasa_fin)
    if reject_inference_method != "none":
        from src.reject_inference import apply_reject_inference

        # Runtime validation of Literal-typed params (todo #52). Pydantic
        # validates these when loaded from config.toml, but direct callers
        # (tests, ad-hoc scripts) can still pass arbitrary strings.
        _allowed_methods = get_args(Literal["none", "parceling"])
        _allowed_parceling = get_args(Literal["linear", "power", "sigmoid"])
        if reject_inference_method not in _allowed_methods:
            raise ValueError(
                f"reject_inference_method must be one of {_allowed_methods}, got {reject_inference_method!r}"
            )
        if reject_parceling_method not in _allowed_parceling:
            raise ValueError(
                f"reject_parceling_method must be one of {_allowed_parceling}, got {reject_parceling_method!r}"
            )

        data_sumary_desagregado_repesca = apply_reject_inference(
            repesca_summary=data_sumary_desagregado_repesca,
            data_demand=data_demand,
            variables=VARIABLES,
            method=reject_inference_method,
            reject_uplift_factor=reject_uplift_factor,
            max_risk_multiplier=reject_max_risk_multiplier,
            parceling_method=reject_parceling_method,
            bayesian_smoothing=reject_bayesian_smoothing,
            bayesian_prior_strength=reject_bayesian_prior_strength,
            enforce_monotonicity=reject_enforce_monotonicity,
            inv_vars=inv_vars,
            include_all_rejections=reject_include_all_rejections,
            acceptance_recent_months=reject_acceptance_recent_months,
            acceptance_decay_half_life_months=reject_acceptance_decay_half_life_months,
            acceptance_date_col=reject_acceptance_date_col,
            apply_h3_multiplier=reject_apply_h3_multiplier,
            no_demand_anchor_percentile=reject_no_demand_anchor_percentile,
            confidence_scale=reject_confidence_scale,
        )
        # Log RI multiplier stats at INFO level for visibility
        if "reject_risk_multiplier" in data_sumary_desagregado_repesca.columns:
            ri_mult = data_sumary_desagregado_repesca["reject_risk_multiplier"]
            logger.info(
                f"Swap-in RI multiplier: min={ri_mult.min():.3f}, avg={ri_mult.mean():.3f}, "
                f"max={ri_mult.max():.3f}, bins={len(ri_mult)}"
            )
        # Drop auxiliary columns before downstream merge, but keep reject_risk_multiplier
        # for downstream diagnostics (renamed to _rep suffix with other indicators)
        data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.drop(
            columns=[
                "acceptance_rate",
                "smoothed_acceptance_rate",
                "ri_effective_acceptance_rate",
                "ri_confidence",
                "ri_bin_count",
            ],
            errors="ignore",
        )

    if per_bin_tasa_fin is not None and not per_bin_tasa_fin.empty:
        per_bin_tasa_fin = _ensure_unique_merge_keys(
            per_bin_tasa_fin.copy(), VARIABLES, ["tasa_fin"], name="per_bin_tasa_fin"
        )
        n_before = len(data_sumary_desagregado_repesca)
        data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.merge(
            per_bin_tasa_fin, on=VARIABLES, how="left", validate="one_to_one"
        )
        if len(data_sumary_desagregado_repesca) != n_before:
            raise RuntimeError(
                "per_bin_tasa_fin merge expanded rows unexpectedly "
                f"(before={n_before}, after={len(data_sumary_desagregado_repesca)})"
            )
        data_sumary_desagregado_repesca["tasa_fin"] = data_sumary_desagregado_repesca["tasa_fin"].fillna(tasa_fin)
        for ind in INDICADORES:
            data_sumary_desagregado_repesca[ind] *= data_sumary_desagregado_repesca["tasa_fin"]
        data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.drop(columns=["tasa_fin"])
    else:
        data_sumary_desagregado_repesca[INDICADORES] *= tasa_fin
    rename_map = {i: i + "_rep" for i in INDICADORES}
    # Preserve diagnostic columns as _rep for downstream reporting
    if "stress_factor_applied" in data_sumary_desagregado_repesca.columns:
        rename_map["stress_factor_applied"] = "stress_factor_rep"
    if "reject_risk_multiplier" in data_sumary_desagregado_repesca.columns:
        rename_map["reject_risk_multiplier"] = "ri_multiplier_rep"
    data_sumary_desagregado_repesca = data_sumary_desagregado_repesca.rename(columns=rename_map)

    # Merge and adjust indicators (preserve merge-key dtypes that can be
    # upcast from int → float by NaN introduction during outer merge).
    var_dtypes = {v: data_sumary_desagregado_booked[v].dtype for v in VARIABLES}
    data_sumary_desagregado = data_sumary_desagregado_booked.merge(
        data_sumary_desagregado_repesca, on=VARIABLES, how="outer"
    )
    # Fill indicator NaNs with 0, but preserve diagnostic _rep columns (NaN for booked-only rows)
    diag_cols = {"stress_factor_rep", "ri_multiplier_rep"}
    fill_cols = [c for c in data_sumary_desagregado.columns if c not in diag_cols]
    data_sumary_desagregado[fill_cols] = data_sumary_desagregado[fill_cols].fillna(0)
    for v, dt in var_dtypes.items():
        if data_sumary_desagregado[v].dtype != dt:
            data_sumary_desagregado[v] = data_sumary_desagregado[v].astype(dt)

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
            data_surf["todu_30ever_h6"], data_surf["todu_amt_pile_h6"], multiplier=multiplier, as_percentage=True
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
