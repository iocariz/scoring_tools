import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from src.constants import Columns, RejectReason, StatusName
from src.inference_optimized import (
    compute_pre_reject_inference_data,
    run_optimization_pipeline,
    todu_average_inference,
)


def test_compute_pre_reject_inference_collapses_duplicate_per_bin_stress(monkeypatch):
    indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    variables = ["var0", "var1"]

    # Patch model-based risk calculation to a deterministic passthrough.
    monkeypatch.setattr(
        "src.models.calculate_risk_values",
        lambda df, *args, **kwargs: df.assign(todu_30ever_h6=df["todu_30ever_h6"]),
    )

    data_booked = pd.DataFrame(
        {
            "var0": [1],
            "var1": [1],
            "status_name": [StatusName.BOOKED.value],
            "reject_reason": [None],
            "todu_30ever_h6": [1.0],
            "todu_amt_pile_h6": [10.0],
            "oa_amt_h0": [100.0],
        }
    )
    data_demand = pd.DataFrame(
        {
            "var0": [1],
            "var1": [1],
            "status_name": [StatusName.REJECTED.value],
            "reject_reason": [RejectReason.SCORE.value],
            "todu_30ever_h6": [2.0],
            "todu_amt_pile_h6": [20.0],
            "oa_amt_h0": [200.0],
        }
    )
    # Duplicate key for (1,1): mean stress factor = 3.0
    per_bin_stress = pd.DataFrame({"var0": [1, 1], "var1": [1, 1], "stress_factor": [2.0, 4.0]})

    _, repesca = compute_pre_reject_inference_data(
        data_booked=data_booked,
        data_demand=data_demand,
        risk_inference={"best_model_info": {"model": object()}, "features": []},
        reg_todu_amt_pile=None,
        stressor=1.0,
        indicators=indicators,
        variables=variables,
        annual_coef=1.0,
        per_bin_stress=per_bin_stress,
    )

    assert len(repesca) == 1
    assert repesca.iloc[0]["todu_30ever_h6"] == 6.0  # 2.0 * mean(2,4)


def test_run_optimization_pipeline_collapses_duplicate_per_bin_tasa_fin(monkeypatch):
    indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
    variables = ["var0", "var1", "var2"]  # 3 vars avoids 2D plot side-effects

    booked_summary = pd.DataFrame(
        {
            "var0": [1],
            "var1": [1],
            "var2": [1],
            "todu_30ever_h6_boo": [1.0],
            "todu_amt_pile_h6_boo": [10.0],
            "oa_amt_h0_boo": [100.0],
        }
    )
    repesca_summary = pd.DataFrame(
        {
            "var0": [1],
            "var1": [1],
            "var2": [1],
            "todu_30ever_h6": [2.0],
            "todu_amt_pile_h6": [20.0],
            "oa_amt_h0": [200.0],
        }
    )

    monkeypatch.setattr(
        "src.inference_optimized.compute_pre_reject_inference_data",
        lambda **kwargs: (booked_summary.copy(), repesca_summary.copy()),
    )

    # Duplicate key for (1,1,1): mean tasa_fin = 3.0
    per_bin_tasa_fin = pd.DataFrame({"var0": [1, 1], "var1": [1, 1], "var2": [1, 1], "tasa_fin": [2.0, 4.0]})

    out = run_optimization_pipeline(
        data_booked=pd.DataFrame(),
        data_demand=pd.DataFrame(),
        risk_inference={"best_model_info": {"model": object()}, "features": []},
        reg_todu_amt_pile=None,
        stressor=1.0,
        tasa_fin=1.0,
        indicators=indicators,
        variables=variables,
        annual_coef=1.0,
        reject_inference_method="none",
        per_bin_tasa_fin=per_bin_tasa_fin,
    )

    assert len(out) == 1
    # repesca contribution should be scaled by mean tasa_fin=3.0: 2*3=6; total = 1+6
    assert out.iloc[0]["todu_30ever_h6"] == 7.0


def test_tree_winner_cv_r2_is_nan_not_zero(monkeypatch):
    """When a tree model wins, its CV R² was never computed (tree tuning is RMSE-only), so the
    metadata must record NaN ('not computed') — never 0.0, which is indistinguishable from a
    genuinely flat model (the #65 degenerate case)."""
    import numpy as np
    from sklearn.linear_model import LinearRegression

    import src.inference_optimized as io

    tree_df = pd.DataFrame(
        [
            {
                "Model": "XGBoost (Optuna Tuned)",
                "CV Mean RMSE": 0.10,  # clearly best
                "CV Std RMSE": 0.01,
                "model_template": LinearRegression(),
            }
        ]
    )
    lin_df = pd.DataFrame(
        [{"Model": "Ridge", "CV Mean RMSE": 0.50, "CV Std RMSE": 0.02, "model_template": LinearRegression()}]
    )
    monkeypatch.setattr(io, "tune_tree_models", lambda **kw: (tree_df, {}))
    monkeypatch.setattr(
        io,
        "_select_model_type_cv",
        lambda **kw: (
            lin_df,
            {"name": "Ridge", "model_template": LinearRegression(), "cv_mean_rmse": 0.5, "cv_std_rmse": 0.02},
        ),
    )

    _, _, _, best_feature_info = io._select_best_model_and_features(
        raw_data=pd.DataFrame(),
        bins=(),
        variables=["v0", "v1"],
        indicators=[],
        multiplier=7.0,
        z_threshold=3.0,
        var_reg=[],
        feature_sets={},
        target_var="t",
        cv_folds=3,
        include_hurdle=False,
    )
    assert np.isnan(best_feature_info["cv_mean_r2"]), "tree-winner CV R² must be NaN, not 0.0"
    assert np.isnan(best_feature_info["cv_std_r2"])


def _exposure_frame(n=30, seed=42):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "b1": rng.randint(0, 3, n),
            "b2": rng.randint(0, 3, n),
            "oa_amt": rng.uniform(1000, 5000, n),
            "todu_amt_pile_h6": rng.uniform(10, 200, n),
            Columns.STATUS_NAME: StatusName.BOOKED.value,
        }
    )


def test_todu_average_inference_z_threshold_zero_disables_outlier_removal():
    # z_threshold == 0 must DISABLE outlier removal (same contract as the risk fit
    # in process_dataset), NOT empty the training set. Before the guard, the filter
    # `z_scores < 0` dropped every bin -> None exposure model -> MR predict crash.
    df = _exposure_frame()
    variables = ["b1", "b2"]
    indicators = ["oa_amt", "todu_amt_pile_h6"]

    _, model_z0, r2_z0 = todu_average_inference(
        df, variables, indicators, z_threshold=0.0, plot_output_path=None, model_output_path=None
    )
    assert model_z0 is not None, "z_threshold=0 must still fit an exposure model"
    # the model is usable downstream (MR prediction path)
    assert model_z0.predict(pd.DataFrame({"oa_amt": [1000.0]})).shape == (1,)
    assert not np.isnan(r2_z0)


def test_todu_average_inference_z_threshold_positive_still_fits():
    # Behaviour for the live default (z_threshold=3.0) is unchanged: a model is fit.
    df = _exposure_frame()
    variables = ["b1", "b2"]
    indicators = ["oa_amt", "todu_amt_pile_h6"]

    _, model_z3, _ = todu_average_inference(
        df, variables, indicators, z_threshold=3.0, plot_output_path=None, model_output_path=None
    )
    assert model_z3 is not None
