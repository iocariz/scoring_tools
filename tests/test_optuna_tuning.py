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
    assert "CV Mean RMSE" in results_df.columns


def _per_loan_linear_data(seed=0):
    """Per-loan booked data with real zero mass + the _hurdle_r/_hurdle_w columns the per-loan
    hurdle adapter consumes (audit #6)."""
    rng = np.random.RandomState(seed)
    rows = []
    for v0 in range(1, 4):
        for v1 in range(1, 4):
            p = 0.05 + 0.10 * (v0 + v1)
            for _ in range(120):
                den = rng.uniform(800.0, 1200.0)
                defaulted = rng.random() < p
                num = den * rng.uniform(0.02, 0.06) if defaulted else 0.0
                rows.append(
                    {
                        "var0": v0,
                        "var1": v1,
                        "todu_30ever_h6": num,
                        "todu_amt_pile_h6": den,
                        "oa_amt_h0": den * 0.9,
                        "status_name": "Booked",
                    }
                )
    df = pd.DataFrame(rows)
    df["_hurdle_r"] = (7.0 * df["todu_30ever_h6"] / df["todu_amt_pile_h6"]).fillna(0.0)
    df["_hurdle_w"] = df["todu_amt_pile_h6"].astype(float)
    return df


def test_tune_linear_models_hurdle_per_loan_flag():
    """include_hurdle=True offers the per-loan hurdle (and it trains without crashing on per-loan
    rows); include_hurdle=False omits it entirely (audit #6)."""
    from src.inference_optimized import _generate_regression_variables
    from src.optuna_tuning import tune_linear_models

    df = _per_loan_linear_data()
    variables = ["var0", "var1"]
    indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    var_reg, _ = _generate_regression_variables(variables)

    res_on, _ = tune_linear_models(
        df,
        None,
        variables,
        indicators,
        "b2_ever_h6",
        7.0,
        3.0,
        var_reg,
        cv_folds=2,
        n_trials=2,
        include_hurdle=True,
        random_state=42,
    )
    assert any("Hurdle" in m for m in res_on["Model"])

    res_off, _ = tune_linear_models(
        df,
        None,
        variables,
        indicators,
        "b2_ever_h6",
        7.0,
        3.0,
        var_reg,
        cv_folds=2,
        n_trials=2,
        include_hurdle=False,
        random_state=42,
    )
    assert not any("Hurdle" in m for m in res_off["Model"])
