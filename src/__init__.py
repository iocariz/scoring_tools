"""
Scoring tools package for credit risk analysis.

This package provides modules for:
- Data preprocessing and cleaning
- Risk inference and modeling
- Model persistence (save/load)
- Optimization pipeline
- Visualization and plotting
"""

from src.constants import (
    DEFAULT_N_POINTS_3D,
    DEFAULT_RANDOM_STATE,
    DEFAULT_RISK_MULTIPLIER,
    DEFAULT_TEST_SIZE,
    DEFAULT_Z_THRESHOLD,
    DEFAULT_ZERO_THRESHOLD,
    Columns,
    RejectReason,
    StatusName,
    Suffixes,
)
from src.estimators import HurdleRegressor
from src.persistence import (
    load_model_for_prediction,
    predict_on_new_data,
    save_model_with_metadata,
)
from src.preprocess_improved import (
    complete_preprocessing_pipeline,
)

__all__ = [
    # Constants and enums
    "StatusName",
    "RejectReason",
    "Columns",
    "Suffixes",
    "DEFAULT_RISK_MULTIPLIER",
    "DEFAULT_Z_THRESHOLD",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_N_POINTS_3D",
    "DEFAULT_ZERO_THRESHOLD",
    # Estimators
    "HurdleRegressor",
    # Persistence
    "save_model_with_metadata",
    "load_model_for_prediction",
    "predict_on_new_data",
    # Preprocessing
    "complete_preprocessing_pipeline",
]
