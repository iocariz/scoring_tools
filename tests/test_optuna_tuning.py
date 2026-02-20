import numpy as np
import pandas as pd

from src.optuna_tuning import tune_tree_models


def test_tune_tree_models_runs_without_crashing():
    """Test that the Optuna tree models tuning pipeline runs and evaluates successfully."""
    # Create simple dummy data
    np.random.seed(42)
    X = pd.DataFrame({
        'var0': np.random.rand(100),
        'var1': np.random.rand(100)
    })
    y = pd.Series(np.random.rand(100))
    weights = pd.Series(np.ones(100))

    # Run tuning with very few trials and folds for speed
    results_df, models = tune_tree_models(
        X=X,
        y=y,
        weights=weights,
        cv_folds=2,
        n_trials=2,
        random_state=42
    )

    assert len(results_df) == 2
    assert "XGBoost (Optuna Tuned)" in models
    assert "LightGBM (Optuna Tuned)" in models
    assert "CV Mean R²" in results_df.columns
