import time
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from loguru import logger

from src.config import OutputPaths, PreprocessingSettings
from src.inference_optimized import inference_pipeline, todu_average_inference
from src.persistence import load_model_for_prediction


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

    if model_path:
        # Load pre-trained model from supersegment
        model, metadata, features = load_model_for_prediction(model_path)
        # Support both old (test_r2) and new (cv_mean_r2) metric formats
        if "cv_mean_r2" in metadata:
            r2_display = f"{metadata['cv_mean_r2']:.4f} +/- {metadata.get('cv_std_r2', 0.0):.4f}"
        else:
            r2_display = f"{metadata.get('test_r2', 0.0):.4f}"
        risk_inference = {
            "best_model_info": {
                "model": model,
                "name": metadata.get("model_type", "Unknown"),
                "cv_mean_r2": metadata.get("cv_mean_r2", metadata.get("test_r2", 0.0)),
                "cv_std_r2": metadata.get("cv_std_r2", 0.0),
            },
            "features": features,
            "model_path": model_path,
        }

        # Load todu model from the models directory (sibling to model subdirectory)
        todu_model_path = Path(model_path).parent / "todu_model.joblib"
        if not todu_model_path.exists():
            # Also check parent's parent (models/ directory)
            todu_model_path = Path(model_path).parent.parent / "todu_model.joblib"
        if todu_model_path.exists():
            reg_todu_amt_pile = joblib.load(todu_model_path)
            logger.debug(f"[{segment}] Loaded todu model from {todu_model_path}")
        else:
            # Fallback: train todu model on current segment data
            logger.warning(f"[{segment}] Todu model not found at {todu_model_path}, training on current data")
            _, reg_todu_amt_pile, _ = todu_average_inference(
                data=data_clean,
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
        # Train new model with feature selection
        risk_inference = inference_pipeline(
            data=data_clean,
            bins=(settings.octroi_bins, settings.efx_bins),
            variables=settings.variables,
            indicators=settings.indicators,
            target_var="b2_ever_h6",
            multiplier=settings.multiplier,
            test_size=0.4,
            include_hurdle=True,
            save_model=True,
            model_base_path=output.model_base_path,
            create_visualizations=True,
        )

        # Todu Average Inference
        _, reg_todu_amt_pile, _ = todu_average_inference(
            data=data_clean,
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
