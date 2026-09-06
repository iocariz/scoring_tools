"""
Model persistence utilities for saving and loading trained models.

This module provides functions for:
- Saving models with comprehensive metadata (plus a SHA-256 integrity sidecar)
- Loading models for prediction, with mandatory integrity + trusted-path checks
- Making predictions on new data

Security model
--------------
`joblib.load` is equivalent to `pickle.load` — a crafted file executes
arbitrary Python on load (RCE).  To mitigate:

1. Every `.pkl`/`.joblib` file saved through this module gets a sibling
   ``<file>.sha256`` written atomically alongside it.  The sidecar is a
   single line: the hex SHA-256 digest of the pickle bytes.
2. Every load through this module refuses to proceed unless the sidecar
   exists, is well-formed, and its digest matches the file's actual
   digest on disk.
3. Every load additionally refuses paths that do not resolve under one
   of the trusted model roots (default: ``<cwd>/models``; override via
   the ``SCORING_TRUSTED_MODEL_ROOTS`` env var, colon-separated).

Existing pre-R0 models have no sidecar and will refuse to load.  Run
``scripts/sign_existing_models.py`` once to sign trusted artifacts.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger

from src.estimators import HurdleRegressor

if TYPE_CHECKING:
    from src.config import PreprocessingSettings

# ---------------------------------------------------------------------------
# Integrity helpers (todo #44 — pickle RCE mitigation)
# ---------------------------------------------------------------------------

SIDE_CAR_SUFFIX = ".sha256"
_TRUSTED_ROOTS_ENV = "SCORING_TRUSTED_MODEL_ROOTS"

# Capture the project root at module import time, BEFORE run_batch.py's
# per-segment os.chdir moves cwd around (todo #67).  This gives default
# trusted roots a stable anchor instead of "whatever directory happens to
# be current when the load fires".
_INITIAL_CWD = Path.cwd().resolve()


class ModelIntegrityError(RuntimeError):
    """Raised when a pickle file fails integrity or trusted-path checks."""


def _compute_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path*'s contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _trusted_roots() -> list[Path]:
    """Resolve the list of directories from which pickle loads are allowed.

    If ``SCORING_TRUSTED_MODEL_ROOTS`` is set (colon-separated paths), that
    is the authoritative list.  Otherwise default to the project's two
    pipeline-managed locations, anchored at the initial cwd (before any
    per-segment ``os.chdir``):

    - ``<project>/models``  — explicit save target from ``save_model_with_metadata``
    - ``<project>/output``  — per-segment and supersegment artifact tree
                              (``output/<segment>/models/...`` and
                              ``output/_supersegment_<ss>/models/...``)

    Both subtrees are entirely pipeline-generated, so widening the default
    to cover them does not loosen protection against user-supplied
    ``--model-path=/tmp/evil.pkl`` style attacks.

    All returned paths are fully resolved (symlinks followed).
    """
    env_val = os.environ.get(_TRUSTED_ROOTS_ENV)
    if env_val:
        return [Path(p).expanduser().resolve() for p in env_val.split(":") if p]
    return [
        (_INITIAL_CWD / "models").resolve(),
        (_INITIAL_CWD / "output").resolve(),
    ]


def write_integrity_sidecar(pkl_path: str | Path) -> Path:
    """Compute SHA-256 of *pkl_path* and write ``<pkl_path>.sha256`` beside it.

    Written atomically: digest goes to a ``.tmp`` file then ``os.replace``
    moves it into place.  Returns the sidecar path.  Raises if the pickle
    itself does not exist.
    """
    pkl = Path(pkl_path).resolve()
    if not pkl.is_file():
        raise ModelIntegrityError(f"cannot sign non-existent file: {pkl}")
    digest = _compute_sha256(pkl)
    sidecar = pkl.with_name(pkl.name + SIDE_CAR_SUFFIX)
    tmp = sidecar.with_name(sidecar.name + ".tmp")
    tmp.write_text(digest + "\n", encoding="ascii")
    os.replace(tmp, sidecar)
    return sidecar


def _verify_integrity_sidecar(pkl_path: Path) -> None:
    """Raise `ModelIntegrityError` if the sidecar is missing/malformed/mismatched."""
    sidecar = pkl_path.with_name(pkl_path.name + SIDE_CAR_SUFFIX)
    if not sidecar.is_file():
        raise ModelIntegrityError(
            f"refusing to load {pkl_path}: integrity sidecar {sidecar.name} is missing. "
            f"Run scripts/sign_existing_models.py on trusted artifacts."
        )
    expected = sidecar.read_text(encoding="ascii").strip().split()
    if not expected or len(expected[0]) != 64 or not all(c in "0123456789abcdef" for c in expected[0].lower()):
        raise ModelIntegrityError(
            f"refusing to load {pkl_path}: sidecar {sidecar.name} is malformed (expected a 64-char hex SHA-256 digest)."
        )
    actual = _compute_sha256(pkl_path)
    if actual.lower() != expected[0].lower():
        raise ModelIntegrityError(
            f"refusing to load {pkl_path}: SHA-256 mismatch "
            f"(sidecar={expected[0][:16]}..., actual={actual[:16]}...). "
            f"Pickle may have been tampered with."
        )


def _verify_trusted_path(pkl_path: Path) -> Path:
    """Resolve *pkl_path* and require it to live under a trusted root.

    Returns the resolved path on success.  Raises `ModelIntegrityError`
    otherwise.  Resolution follows symlinks, so a symlink escaping the
    trusted root is caught.
    """
    resolved = pkl_path.resolve()
    roots = _trusted_roots()
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    roots_display = ", ".join(str(r) for r in roots)
    raise ModelIntegrityError(
        f"refusing to load {pkl_path}: resolved path {resolved} is not under any "
        f"trusted root ({roots_display}). Set {_TRUSTED_ROOTS_ENV} to extend the "
        f"allowlist if this path is legitimate."
    )


def safe_joblib_load(pkl_path: str | Path) -> Any:
    """Load a pickle with trusted-path + SHA-256 integrity enforcement.

    This is the **only** sanctioned way to read pickle files produced by
    this project.  Direct `joblib.load` / `pickle.load` on user-influenced
    paths is forbidden because a crafted file executes arbitrary code.
    """
    pkl = Path(pkl_path)
    resolved = _verify_trusted_path(pkl)
    _verify_integrity_sidecar(resolved)
    return joblib.load(resolved)


def save_model_with_metadata(
    model,
    features: list[str],
    metadata: dict,
    base_path: str | Path = "models",
) -> str:
    """
    Save trained model with comprehensive metadata for production use.

    Args:
        model: Trained model object
        features: List of feature names
        metadata: Dictionary containing model metadata
        base_path: Base directory for saving models. Accepts ``str`` or
            ``pathlib.Path``; coerced internally.

    Returns:
        Path to saved model directory
    """
    # Create directory structure. Keep the parameter's type contract honest:
    # do not rebind `base_path` to a different type — coerce into a local.
    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create version directory. Two saves in the same second (e.g. a fast sequential
    # batch) would otherwise reuse the same model_<ts> dir and silently OVERWRITE the
    # first model's .pkl. Disambiguate with a counter suffix so no model is lost.
    version_path = base_dir / f"model_{timestamp}"
    if version_path.exists():
        n = 1
        while (base_dir / f"model_{timestamp}_{n}").exists():
            n += 1
        version_path = base_dir / f"model_{timestamp}_{n}"
        logger.warning(f"Model dir model_{timestamp} already exists; saving as {version_path.name} to avoid overwrite.")
    version_path.mkdir(parents=True, exist_ok=False)

    # Save model and write integrity sidecar
    model_path = version_path / "model.pkl"
    joblib.dump(model, model_path, compress=3)  # Add compression
    write_integrity_sidecar(model_path)

    # Collect environment info for reproducibility
    import importlib.metadata
    import sys

    package_versions = {}
    for pkg in ("pandas", "scikit-learn", "scipy", "numpy", "joblib"):
        try:
            package_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    # Enhance metadata
    metadata_enhanced = {
        "timestamp": timestamp,
        "model_type": type(model).__name__,
        "model_params": model.get_params() if hasattr(model, "get_params") else {},
        "features": features,
        "num_features": len(features),
        "aggregated_data": True,
        "python_version": sys.version,
        "package_versions": package_versions,
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

    # Save SHAP values if provided
    if "shap_values" in metadata and metadata["shap_values"] is not None:
        shap_path = version_path / "shap_values.npy"
        np.save(shap_path, metadata["shap_values"])
        logger.info(f"   - SHAP values: shap_values.npy ({metadata['shap_values'].shape})")

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
        if "cv_mean_rmse" in metadata or "cv_mean_r2" in metadata:
            # New CV-based metrics (RMSE or R2)
            metric_keys = ["cv_mean_rmse", "cv_std_rmse", "cv_mean_r2", "cv_std_r2", "train_r2", "full_r2", "loo_r2"]
            for key in metric_keys:
                if key in metadata:
                    value = metadata.get(key, "N/A")
                    display_value = f"{value:.4f}" if isinstance(value, (int, float)) and not np.isnan(value) else value
                    label = (
                        key.replace("_", " ").replace("cv ", "CV ").replace("r2", "R²").replace("rmse", "RMSE").title()
                    )
                    label = label.replace("Rmse", "RMSE").replace("Loo ", "LOO-CV ")
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

            # Print Intercept if it exists
            if hasattr(model, "intercept_"):
                # Handle array-like intercepts (e.g. from some scikit-learn setups)
                intercept_val = (
                    model.intercept_[0]
                    if isinstance(model.intercept_, (np.ndarray, list)) and len(model.intercept_) > 0
                    else model.intercept_
                )
                f.write(f"Intercept: {intercept_val:.6f}\n")

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

    # Load model (enforces SHA-256 sidecar + trusted-root allowlist; todo #44)
    model = safe_joblib_load(model_dir / "model.pkl")

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


def validate_reused_model_config(metadata: dict, settings: "PreprocessingSettings") -> None:
    """Validate a REUSED model's persisted config against the current settings (#40).

    A reused model (``--model-path`` / batch supersegment model sharing) learned "bin index → risk"
    for a SPECIFIC (multiplier, inference variables, per-variable bin edges). Applying it under a
    different grid maps the same cell indices onto different score regions → silently wrong
    predictions that no metric surfaces. This raises ``ValueError`` on a clear mismatch (multiplier,
    inference variables, or bin edges) and warns (does not fail) when the model predates metadata
    capture, so those fields simply cannot be validated.

    The metadata keys are populated by ``_save_model_to_disk`` (``multiplier``, ``model_variables``,
    ``bin_edges``); older models lack ``bin_edges`` (→ warn).
    """
    seg = getattr(settings, "segment_filter", "?")
    problems: list[str] = []

    saved_mult = metadata.get("multiplier")
    if saved_mult is not None and float(saved_mult) != float(settings.multiplier):
        problems.append(f"multiplier {saved_mult} != config {settings.multiplier}")

    inf_vars = list(settings.inference_variables)
    saved_vars = metadata.get("model_variables")
    if saved_vars is not None and list(saved_vars) != inf_vars:
        problems.append(f"inference variables {list(saved_vars)} != config {inf_vars}")

    saved_edges = metadata.get("bin_edges")
    if isinstance(saved_edges, dict) and getattr(settings, "bins", None):
        for var in inf_vars:
            if var not in saved_edges or var not in settings.bins:
                continue
            se = [float(e) for e in saved_edges[var]]
            ce = [float(e) for e in settings.bins[var].bin_edges]
            if se != ce:
                problems.append(f"bin_edges['{var}'] changed ({len(se)}→{len(ce)} edges / values differ)")

    if problems:
        raise ValueError(
            f"[{seg}] Reused model is INCOMPATIBLE with the current config (#40): "
            + "; ".join(problems)
            + ". The model's cell indices map to different score regions under this config — retrain "
            "for this config, or point --model-path at a model trained on it."
        )

    absent = [k for k in ("multiplier", "model_variables", "bin_edges") if metadata.get(k) is None]
    if absent:
        logger.warning(
            f"[{seg}] Reused model predates metadata capture for {absent}; those cannot be validated "
            "against the current config. Retrain to enable full model-reuse validation (#40)."
        )


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

    # Additional info for Hurdle models (diagnostic only — must never abort a valid
    # prediction run). A hurdle classifier trained on a single-class fold has one column
    # in predict_proba, so the old [:, 1] raised IndexError.
    if metadata.get("is_hurdle", False) and isinstance(model, HurdleRegressor):
        try:
            binary_pred = model.predict_binary(new_data[features])
            proba = model.classifier_.predict_proba(new_data[features])
            if proba.shape[1] > 1:
                prob_nonzero = proba[:, 1]
            else:
                # Single-class classifier: P(non-zero) is 1 iff its sole class is the positive one.
                prob_nonzero = np.full(len(proba), float(model.classifier_.classes_[0] == 1))
            logger.info("Hurdle Model Details:")
            logger.info(f"  Predicted non-zero: {binary_pred.sum():,} ({binary_pred.mean():.1%})")
            logger.info(f"  Mean P(non-zero): {prob_nonzero.mean():.2%}")
        except Exception as e:
            logger.warning(f"Could not compute hurdle diagnostics (non-fatal): {e}")

    return predictions
