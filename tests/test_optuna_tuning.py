import numpy as np
import pandas as pd

from src.optuna_tuning import tune_tree_models


def test_tune_tree_models_runs_without_crashing():
    """Test that the Optuna tree models tuning pipeline runs and evaluates successfully."""
    # Create simple dummy data
    np.random.seed(42)
    # Create simple dummy data with mock columns
    X = pd.DataFrame(
        {
            "var_x": np.random.rand(100),
            "var_y": np.random.rand(100),
            "todu_30ever_h6": np.random.rand(100) * 100,
            "todu_amt_pile_h6": np.random.rand(100) * 1000,
            "status_name": ["Booked"] * 100,
        }
    )

    # Run tuning with very few trials and folds for speed
    results_df, models = tune_tree_models(
        raw_data=X,
        bins=None,
        variables=["var_x", "var_y"],
        indicators=["todu_30ever_h6", "todu_amt_pile_h6"],
        target_var="b2_ever_h6",
        multiplier=100.0,
        z_threshold=3.0,
        cv_folds=2,
        n_trials=2,
        random_state=42,
    )

    assert len(results_df) == 2
    assert "XGBoost (Optuna Tuned)" in models
    assert "LightGBM (Optuna Tuned)" in models
    assert "CV Mean R²" in results_df.columns
