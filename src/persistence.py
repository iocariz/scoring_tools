"""
Model persistence utilities for saving and loading trained models.

This module provides functions for:
- Saving models with comprehensive metadata
- Loading models for prediction
- Making predictions on new data
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.estimators import HurdleRegressor


def save_model_with_metadata(model, features: list[str], metadata: dict, base_path: str = "models") -> str:
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create version directory
    version_path = base_path / f"model_{timestamp}"
    version_path.mkdir(exist_ok=True)

    # Save model
    model_path = version_path / "model.pkl"
    joblib.dump(model, model_path, compress=3)  # Add compression

    # Enhance metadata
    metadata_enhanced = {
        "timestamp": timestamp,
        "model_type": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
        "features": features,
        "num_features": len(features),
        "aggregated_data": True,
        **metadata,
    }

    # Save metadata
    with open(version_path / "metadata.json", "w") as f:
        json.dump(metadata_enhanced, f, indent=2, default=str)

    # Save features
    with open(version_path / "features.txt", "w") as f:
        f.write(f"# Features for {type(model).__name__}\n")
        f.write(f"# Timestamp: {timestamp}\n")
        # Support both old (test_r2) and new (cv_mean_r2) metric names
        if "cv_mean_r2" in metadata:
            cv_std = metadata.get("cv_std_r2", 0)
            f.write(f"# CV R²: {metadata['cv_mean_r2']:.4f} ± {cv_std:.4f}\n\n")
        elif "test_r2" in metadata:
            f.write(f"# Test R²: {metadata['test_r2']:.4f}\n\n")
        else:
            f.write("# R²: N/A\n\n")
        for i, feature in enumerate(features, 1):
            f.write(f"{i}. {feature}\n")

    # Save model summary
    _save_model_summary(version_path, model, features, metadata, timestamp)

    logger.info("=" * 60)
    logger.info("MODEL SAVED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Directory: {version_path}")
    logger.info("   - Model: model.pkl")
    logger.info("   - Metadata: metadata.json")
    logger.info("   - Features: features.txt")
    logger.info("   - Summary: model_summary.txt")

    return str(version_path)


def _save_model_summary(version_path: Path, model, features: list[str], metadata: dict, timestamp: str) -> None:
    """Helper function to save model summary."""
    with open(version_path / "model_summary.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model Type: {type(model).__name__}\n")
        f.write(f"Training Date: {timestamp}\n")
        f.write(f"Number of Features: {len(features)}\n")
        f.write("Aggregated Data: Yes\n\n")

        f.write("PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        # Support both old and new metric formats
        if "cv_mean_r2" in metadata:
            # New CV-based metrics
            for key in ["cv_mean_r2", "cv_std_r2", "full_r2"]:
                value = metadata.get(key, "N/A")
                display_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
                label = key.replace("_", " ").replace("cv ", "CV ").replace("r2", "R²").title()
                f.write(f"{label}: {display_value}\n")
            f.write(f"CV Folds: {metadata.get('cv_folds', 'N/A')}\n")
            f.write(f"Total Samples: {metadata.get('total_samples', 'N/A')}\n")
        else:
            # Old train/test metrics
            for key in ["train_r2", "test_r2", "test_rmse", "test_mae"]:
                value = metadata.get(key, "N/A")
                display_value = f"{value:.4f}" if isinstance(value, (int, float)) else value
                f.write(f"{key.replace('_', ' ').title()}: {display_value}\n")
            f.write(f"Training Samples: {metadata.get('train_samples', 'N/A')}\n")
            f.write(f"Test Samples: {metadata.get('test_samples', 'N/A')}\n")

        if hasattr(model, "coef_"):
            f.write("\nMODEL COEFFICIENTS:\n")
            f.write("-" * 30 + "\n")
            for feature, coef in zip(features, model.coef_):
                f.write(f"{feature}: {coef:.6f}\n")


def load_model_for_prediction(model_path: str) -> tuple[Any, dict, list[str]]:
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
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)

    features = metadata["features"]

    logger.info("Model loaded successfully")
    logger.info(f"   Type: {metadata['model_type']}")
    logger.info(f"   Features: {metadata['num_features']}")
    # Support both old and new metric formats
    if "cv_mean_r2" in metadata:
        cv_r2 = metadata.get("cv_mean_r2")
        cv_std = metadata.get("cv_std_r2", 0)
        logger.info(f"   CV R²: {cv_r2:.4f} ± {cv_std:.4f}")
    else:
        test_r2 = metadata.get("test_r2", "N/A")
        logger.info(f"   Test R²: {test_r2:.4f}" if isinstance(test_r2, (int, float)) else f"   Test R²: {test_r2}")

    return model, metadata, features


def predict_on_new_data(model_path: str, new_data: pd.DataFrame) -> np.ndarray:
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

    logger.info("Predictions generated")
    logger.info(f"  Number of predictions: {len(predictions):,}")
    logger.info(f"  Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    logger.info(f"  Mean prediction: {predictions.mean():.4f}")

    # Additional info for Hurdle models
    if metadata.get("is_hurdle", False):
        if isinstance(model, HurdleRegressor):
            binary_pred = model.predict_binary(new_data[features])
            prob_nonzero = model.classifier_.predict_proba(new_data[features])[:, 1]
            logger.info("Hurdle Model Details:")
            logger.info(f"  Predicted non-zero: {binary_pred.sum():,} ({binary_pred.mean():.1%})")
            logger.info(f"  Mean P(non-zero): {prob_nonzero.mean():.2%}")

    return predictions
