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


def test_val_fold_scored_on_realized_target_not_winsorized(monkeypatch):
    """Audit #5 (endpoint of #32a): the validation fold is scored on the REALIZED
    target — process_dataset is called with z_threshold=0 for val (no winsorization),
    so the riskiest val bins keep their true value instead of being clipped, which
    otherwise biases model selection toward under-prediction. The TRAIN fold stays
    winsorized at the configured z_threshold (#56) to bound the fitted surface."""
    import src.inference_optimized as io_mod

    real_process_dataset = io_mod.process_dataset
    z_args = []

    def recording_process_dataset(*args, **kwargs):
        # z_threshold is the 8th positional arg (data, bins, variables, indicators,
        # target_var, multiplier, features, z_threshold); both sites pass it positionally.
        z = args[7] if len(args) > 7 else kwargs.get("z_threshold")
        z_args.append(z)
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

    assert z_args, "process_dataset was never invoked"
    # Each fold processes train (z=3.0, winsorized) then val (z=0.0, realized).
    train_calls = [z for z in z_args if z == 3.0]
    val_calls = [z for z in z_args if z == 0.0]
    assert val_calls, "val fold was not processed with z_threshold=0 (audit #5)"
    assert train_calls, "train fold was not winsorized at the configured z_threshold (#56)"
    assert len(train_calls) == len(val_calls)  # one train + one val per fold


def test_tree_leaf_bounds_scale_to_small_grid():
    """Constant-model bug: min_child_weight / min_child_samples must scale to the
    bin count. On a 16-bin grid the search ceiling is 16 // 3 = 5, so no booster is
    forced to a single leaf. The old fixed [10, 100] range required >= 10 bins per
    leaf, collapsing every tree to a constant on small (e.g. pooled/retail) grids."""
    rng = np.random.RandomState(0)
    rows = []
    for vx in range(1, 5):
        for vy in range(1, 5):
            for _ in range(25):  # 25 loans/bin over a 4x4 = 16-bin grid
                pile = 100.0 + rng.rand() * 10
                rows.append(
                    {
                        "var_x": vx,
                        "var_y": vy,
                        "todu_30ever_h6": (vx + vy) * 0.01 * pile + rng.rand(),  # real bin-level signal
                        "todu_amt_pile_h6": pile,
                        "status_name": "Booked",
                    }
                )
    X = pd.DataFrame(rows)

    _, models = tune_tree_models(
        raw_data=X,
        bins=None,
        variables=["var_x", "var_y"],
        indicators=["todu_30ever_h6", "todu_amt_pile_h6"],
        target_var="b2_ever_h6",
        multiplier=100.0,
        z_threshold=3.0,
        cv_folds=2,
        n_trials=3,
        random_state=0,
    )

    # leaf_hi = max(2, min(100, 16 // 3)) = 5 — both bounded well below the old 10.
    assert 1 <= models["XGBoost (Optuna Tuned)"].get_params()["min_child_weight"] <= 5
    assert 1 <= models["LightGBM (Optuna Tuned)"].get_params()["min_child_samples"] <= 5
