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


def test_selection_metric_reports_real_cv_se():
    """Audit #7: candidate rows are scored by fresh-seed k-fold CV, so CV Std RMSE is a real
    standard error (> 0), not the degenerate 0.0 of the old single shared holdout — this is what
    lets the downstream 1-SE rule form a proper band instead of collapsing to argmin."""
    from src.inference_optimized import _generate_regression_variables
    from src.optuna_tuning import tune_linear_models

    df = _per_loan_linear_data()
    variables = ["var0", "var1"]
    indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    var_reg, _ = _generate_regression_variables(variables)

    results_df, _ = tune_linear_models(
        df,
        None,
        variables,
        indicators,
        "b2_ever_h6",
        7.0,
        3.0,
        var_reg,
        cv_folds=3,
        n_trials=2,
        include_hurdle=False,
        random_state=42,
    )
    assert (results_df["CV Std RMSE"] >= 0).all()
    assert (results_df["CV Std RMSE"] > 0).any()  # real k-fold SE, not the all-zero holdout placeholder


def test_val_fold_outlier_stats_come_from_train_fold(monkeypatch):
    """Audit #32a: the validation fold must be outlier-filtered with TRAIN-fold
    stats — the old call sites omitted outlier_stats, so process_dataset
    computed median/MAD from the val fold's own target and silently dropped
    exactly the riskiest validation bins from scoring."""
    import src.inference_optimized as io_mod

    real_process_dataset = io_mod.process_dataset
    calls = []

    def recording_process_dataset(*args, **kwargs):
        calls.append(kwargs.get("outlier_stats"))
        return real_process_dataset(*args, **kwargs)

    monkeypatch.setattr(io_mod, "process_dataset", recording_process_dataset)

    np.random.seed(42)
    X = pd.DataFrame(
        {
            "var_x": np.random.rand(120),
            "var_y": np.random.rand(120),
            "todu_30ever_h6": np.random.rand(120) * 100,
            "todu_amt_pile_h6": np.random.rand(120) * 1000,
            "status_name": ["Booked"] * 120,
        }
    )
    tune_tree_models(
        raw_data=X,
        bins=None,
        variables=["var_x", "var_y"],
        indicators=["todu_30ever_h6", "todu_amt_pile_h6"],
        target_var="b2_ever_h6",
        multiplier=100.0,
        z_threshold=3.0,
        cv_folds=2,
        n_trials=1,
        random_state=42,
    )

    assert calls, "process_dataset was never invoked"
    passed_stats = [c for c in calls if c is not None]
    assert passed_stats, "no val-fold call received train-derived outlier_stats (audit #32a leak)"
    # calls alternate train (None) / val (train stats): half of them carry stats
    assert len(passed_stats) == len(calls) // 2
