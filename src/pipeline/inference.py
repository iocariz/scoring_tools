import time
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.inference_optimized import inference_pipeline, todu_average_inference
from src.persistence import load_model_for_prediction, safe_joblib_load, validate_reused_model_config


def run_inference_phase(
    data_clean: pd.DataFrame,
    settings: PreprocessingSettings,
    model_path: str = None,
    output: OutputPaths | None = None,
) -> tuple[dict, Any]:
    """Run risk inference: either load a pre-trained model or train a new one.

    Args:
        data_clean: Cleaned DataFrame from preprocessing
        settings: Configuration settings object
        model_path: Optional path to a pre-trained model directory
        output: Output paths configuration. Defaults to current directory.

    Returns:
        Tuple of (risk_inference, reg_todu_amt_pile)

    Raises:
        Exception: If model loading or training fails
    """
    if output is None:
        output = OutputPaths()

    t0 = time.perf_counter()
    segment = settings.segment_filter

    # Restrict MODEL TRAINING by booking date (audit #1). data_clean is the FULL cleaned
    # demand (all dates — the MR/holdout cohort is sliced from it downstream). The UPPER
    # bound (mis_date <= date_fin_book_obs) is ALWAYS applied: rows after it are the
    # immature MR/holdout cohort whose H6 is not yet realized, so training the risk AND
    # exposure models on them leaks future/immature outcomes into the fit. The LOWER bound
    # (drop applications booked before date_ini_book_obs) is opt-in via
    # settings.train_from_date_ini — OFF by default so all MATURE past applications are used
    # (more data; pre-window loans are older → their H6 is realized). The caller's data_clean
    # is untouched: optimization and the MR check still use the full period.
    train_data = data_clean
    if settings.date_fin_book_obs:
        if "mis_date" in data_clean.columns:
            mis = pd.to_datetime(data_clean["mis_date"])
            mask = mis <= pd.to_datetime(settings.date_fin_book_obs)
            lower = (
                settings.date_ini_book_obs if (settings.train_from_date_ini and settings.date_ini_book_obs) else None
            )
            if lower is not None:
                mask &= mis >= pd.to_datetime(lower)
            train_data = data_clean[mask]
            n_excluded = len(data_clean) - len(train_data)
            logger.info(
                f"[{segment}] Model training restricted to booking window "
                f"[{lower or '-inf'} .. {settings.date_fin_book_obs}]: "
                f"{len(train_data)}/{len(data_clean)} rows kept ({n_excluded} out-of-window rows excluded)."
            )
        else:
            logger.warning(
                f"[{segment}] mis_date column absent — cannot restrict training by booking date; "
                f"training on all supplied rows."
            )

    if model_path:
        # Load pre-trained model from supersegment
        model, metadata, features = load_model_for_prediction(model_path)
        # Fail loudly if the reused model was trained under an incompatible grid (#40): a different
        # multiplier / inference variables / bin edges means its cell indices map to different score
        # regions. Old models (no bin_edges) warn instead of failing.
        validate_reused_model_config(metadata, settings)
        # Support both old (test_r2) and new (cv_mean_r2) metric formats
        if "cv_mean_r2" in metadata:
            r2_display = f"{metadata['cv_mean_r2']:.4f} +/- {metadata.get('cv_std_r2', 0.0):.4f}"
        else:
            r2_display = f"{metadata.get('test_r2', 0.0):.4f}"
        model_variables = metadata.get("model_variables", settings.inference_variables)
        risk_inference = {
            "best_model_info": {
                "model": model,
                "name": metadata.get("model_type", "Unknown"),
                "cv_mean_r2": metadata.get("cv_mean_r2", metadata.get("test_r2", 0.0)),
                "cv_std_r2": metadata.get("cv_std_r2", 0.0),
                "cv_std_rmse": metadata.get("cv_std_rmse", 0.0),
            },
            "features": features,
            "model_path": model_path,
            "model_variables": model_variables,
        }

        # Load todu model from the models directory (sibling to model subdirectory)
        todu_model_path = Path(model_path).parent / "todu_model.joblib"
        if not todu_model_path.exists():
            # Also check parent's parent (models/ directory)
            todu_model_path = Path(model_path).parent.parent / "todu_model.joblib"
        if todu_model_path.exists():
            # safe_joblib_load enforces SHA-256 sidecar + trusted-root allowlist (todo #44)
            reg_todu_amt_pile = safe_joblib_load(todu_model_path)
            logger.debug(f"[{segment}] Loaded todu model from {todu_model_path}")
        else:
            # Fallback: train todu model on current segment data
            logger.warning(f"[{segment}] Todu model not found at {todu_model_path}, training on current data")
            _, reg_todu_amt_pile, _ = todu_average_inference(
                data=train_data,
                variables=settings.variables,
                indicators=settings.indicators,
                feature_col="oa_amt",
                target_col="todu_amt_pile_h6",
                z_threshold=settings.z_threshold,
                plot_output_path=output.todu_avg_inference_html,
                model_output_path=None,  # Don't save, it's a fallback
            )

        elapsed = time.perf_counter() - t0
        model_name = risk_inference["best_model_info"]["name"]
        logger.info(f"[{segment}] Model loaded | {model_name} | R2={r2_display} | from {model_path} | {elapsed:.1f}s")
    else:
        inference_vars = settings.inference_variables

        # Build bins tuple filtered to inference variables only
        if settings.bins:
            missing_inference_bins = [var for var in inference_vars if var not in settings.bins]
            if missing_inference_bins:
                raise ValueError(f"Missing bin configuration for inference variables: {missing_inference_bins}")
            bins_tuple = tuple(settings.bins[var].bin_edges for var in inference_vars)
        else:
            bins_tuple = (settings.octroi_bins, settings.efx_bins)

        if inference_vars != settings.variables:
            logger.info(
                f"[{segment}] Inference uses {len(inference_vars)} variables {inference_vars}, "
                f"optimization uses {len(settings.variables)} variables {settings.variables}"
            )

        # Train new model with feature selection
        risk_inference = inference_pipeline(
            data=train_data,
            bins=bins_tuple,
            variables=inference_vars,
            indicators=settings.indicators,
            target_var="b2_ever_h6",
            multiplier=settings.multiplier,
            cv_folds=settings.cv_folds,
            include_hurdle=settings.model_hurdle_per_loan,
            save_model=True,
            model_base_path=output.model_base_path,
            create_visualizations=True,
            directions=settings.directions or None,
            z_threshold=settings.z_threshold,
        )

        # Todu Average Inference
        _, reg_todu_amt_pile, _ = todu_average_inference(
            data=train_data,
            variables=settings.variables,
            indicators=settings.indicators,
            feature_col="oa_amt",
            target_col="todu_amt_pile_h6",
            z_threshold=settings.z_threshold,
            plot_output_path=output.todu_avg_inference_html,
            model_output_path=output.todu_model_joblib,
        )

        elapsed = time.perf_counter() - t0
        info = risk_inference["best_model_info"]
        cv_r2 = info.get("cv_mean_r2", 0)
        cv_std = info.get("cv_std_r2", 0)
        logger.info(
            f"[{segment}] Inference done | {info['name']} ({info.get('model_type', 'N/A')}) | "
            f"features={info.get('feature_set', 'N/A')} | CV R2={cv_r2:.4f} +/- {cv_std:.4f} | {elapsed:.1f}s"
        )

    return risk_inference, reg_todu_amt_pile
