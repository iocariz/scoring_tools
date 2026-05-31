"""Pure-function coverage tests for src.inference_optimized helpers.

Covers the small stateless helpers that don't require a full model training run:
complexity ranking, 1SE rule, CV splitter, feature-set generation, outlier stats,
and weight computation.
"""

import numpy as np
import pandas as pd
import pytest

from src import inference_optimized as io


class TestGetModelComplexity:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Linear Regression", 1),
            ("Ridge", 2),
            ("Lasso", 3),
            ("ElasticNet", 4),
            ("Tweedie", 5),
            # Note: substring match ordering — "ridge" matches before "hurdle-ridge"
            # in the current implementation, so "Hurdle-Ridge" → 2 (ridge rank).
            ("Hurdle-Ridge", 2),
            ("Hurdle-Lasso", 3),
            ("XGBoost", 8),
            ("LightGBM", 9),
            ("SomeUnknownModel", 10),
        ],
    )
    def test_complexity_ranks(self, name, expected):
        assert io._get_model_complexity(name) == expected


class TestApplyOneSERule:
    def test_prefers_simpler_model_within_one_se(self):
        df = pd.DataFrame(
            {
                "Model": ["Linear Regression", "XGBoost"],
                "CV Mean RMSE": [0.50, 0.48],  # XGBoost is nominally better
                "CV Std RMSE": [0.05, 0.05],  # within 1 SE
                "complexity": [1, 8],
            }
        )
        idx = io._apply_one_se_rule(df, "complexity")
        # 1SE rule should pick simpler (Linear Regression)
        assert df.loc[idx, "Model"] == "Linear Regression"

    def test_returns_best_when_simplest_already(self):
        df = pd.DataFrame(
            {
                "Model": ["Linear Regression", "XGBoost"],
                "CV Mean RMSE": [0.50, 0.51],
                "CV Std RMSE": [0.01, 0.01],
                "complexity": [1, 8],
            }
        )
        idx = io._apply_one_se_rule(df, "complexity")
        assert df.loc[idx, "Model"] == "Linear Regression"


class TestStratifiedCvSplitter:
    def test_uses_stratified_when_indicator_has_two_classes(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame({"ind1": rng.randint(0, 2, 100), "other": rng.rand(100)})
        splits = list(io._stratified_cv_splitter(df, cv_folds=5, random_state=0, indicators=["ind1"]))
        assert len(splits) == 5
        # Each split produces (train_idx, val_idx) arrays
        for train, val in splits:
            assert len(train) + len(val) == len(df)

    def test_falls_back_to_kfold_when_indicator_missing(self):
        df = pd.DataFrame({"other": np.arange(20)})
        splits = list(io._stratified_cv_splitter(df, cv_folds=4, random_state=0, indicators=["missing"]))
        assert len(splits) == 4

    def test_falls_back_to_kfold_when_only_one_class(self):
        df = pd.DataFrame({"ind1": [0] * 20, "other": np.arange(20)})
        splits = list(io._stratified_cv_splitter(df, cv_folds=4, random_state=0, indicators=["ind1"]))
        assert len(splits) == 4

    def test_no_indicators_uses_kfold(self):
        df = pd.DataFrame({"other": np.arange(20)})
        splits = list(io._stratified_cv_splitter(df, cv_folds=4, random_state=0, indicators=[]))
        assert len(splits) == 4


class TestGenerateRegressionVariables:
    def test_2var_generates_feature_sets(self):
        var_reg, feature_sets = io._generate_regression_variables(["v0", "v1"])
        assert "degree_1" in feature_sets
        assert "degree_2" in feature_sets
        assert "degree_3" in feature_sets
        assert "original" in feature_sets
        assert feature_sets["original"] == ["v0", "v1"]

    def test_nvar_generates_feature_sets(self):
        var_reg, feature_sets = io._generate_regression_variables_nd(["v0", "v1", "v2"])
        assert set(feature_sets.keys()) >= {"degree_1", "degree_2", "degree_3", "original", "full"}
        # degree_1 features should include all base variables
        for v in ["v0", "v1", "v2"]:
            assert v in feature_sets["degree_1"]
        # higher degrees should include more features
        assert len(feature_sets["degree_2"]) > len(feature_sets["degree_1"])
        assert len(feature_sets["degree_3"]) > len(feature_sets["degree_2"])


class TestCalculateTargetMetric:
    def test_basic_calculation(self):
        df = pd.DataFrame({"num": [1, 2, 3], "den": [10, 20, 30]})
        result = io.calculate_target_metric(df, multiplier=100.0, numerator="num", denominator="den")
        assert len(result) == 3
        # 1/10 * 100 = 10
        assert result[0] == pytest.approx(10.0, rel=1e-6)

    def test_zero_denominator_handled(self):
        df = pd.DataFrame({"num": [1, 2], "den": [10, 0]})
        result = io.calculate_target_metric(df, multiplier=100.0, numerator="num", denominator="den")
        assert len(result) == 2
        # Second row should be NaN or 0 depending on implementation
        assert np.isnan(result[1]) or result[1] == 0.0


class TestComputeOutlierStats:
    def test_basic_median_and_mad(self):
        df = pd.DataFrame({"target": [1.0, 2.0, 3.0, 4.0, 5.0]})
        median, mad = io.compute_outlier_stats(df, "target")
        assert median == pytest.approx(3.0)
        assert mad > 0

    def test_all_same_value_falls_back_to_mean_abs_deviation(self):
        df = pd.DataFrame({"target": [5.0, 5.0, 5.0]})
        median, mad = io.compute_outlier_stats(df, "target")
        assert median == 5.0
        # When MAD=0, falls back to mean(|x - median|) which is also 0
        assert mad == 0

    def test_skew_distribution(self):
        df = pd.DataFrame({"target": [1.0, 1.0, 1.0, 1.0, 100.0]})
        median, mad = io.compute_outlier_stats(df, "target")
        assert median == 1.0


class TestGetRegressionWeights:
    def test_prefers_todu_amt_pile_h6(self):
        df = pd.DataFrame(
            {
                "todu_amt_pile_h6": [100.0, 200.0, 300.0],
                "oa_amt_h0": [1.0, 2.0, 3.0],
                "n_observations": [10, 20, 30],
            }
        )
        w = io._get_regression_weights(df)
        assert w is not None
        # Normalized so weights sum to N
        assert w.sum() == pytest.approx(len(df))

    def test_falls_back_to_oa_amt_h0(self):
        df = pd.DataFrame({"oa_amt_h0": [100.0, 200.0], "n_observations": [10, 20]})
        w = io._get_regression_weights(df)
        assert w is not None
        assert w.sum() == pytest.approx(len(df))

    def test_falls_back_to_n_observations(self):
        df = pd.DataFrame({"n_observations": [10, 20, 30]})
        w = io._get_regression_weights(df)
        assert w is not None

    def test_returns_none_when_no_weight_columns(self):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        assert io._get_regression_weights(df) is None

    def test_returns_none_when_all_zeros(self):
        df = pd.DataFrame({"todu_amt_pile_h6": [0.0, 0.0]})
        assert io._get_regression_weights(df) is None

    def test_negative_values_clipped_to_zero(self):
        df = pd.DataFrame({"todu_amt_pile_h6": [-5.0, 100.0, 200.0]})
        w = io._get_regression_weights(df)
        assert w is not None
        # Negative entry should be clipped to 0
        assert (w >= 0).all()


class TestProcessDataset:
    def test_aggregates_and_computes_target(self):
        df = pd.DataFrame(
            {
                "var0": [1, 1, 2, 2],
                "var1": [1, 2, 1, 2],
                "todu_30ever_h6": [1.0, 2.0, 3.0, 4.0],
                "todu_amt_pile_h6": [10.0, 20.0, 30.0, 40.0],
                "oa_amt_h0": [100.0, 200.0, 300.0, 400.0],
            }
        )
        out = io.process_dataset(
            df,
            bins=(None, None),
            variables=["var0", "var1"],
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            target_var="target",
            multiplier=100.0,
            var_reg=[],
            z_threshold=0,
        )
        assert "target" in out.columns
        assert "n_observations" in out.columns
        assert len(out) == 4  # 4 unique bin combinations


class TestEvaluateHoldoutRmse:
    """Winner-only held-out RMSE report (audit #7)."""

    @staticmethod
    def _data(n=400, seed=0):
        rng = np.random.RandomState(seed)
        rows = []
        for _ in range(n):
            v0, v1 = int(rng.randint(1, 5)), int(rng.randint(1, 5))
            den = rng.uniform(800.0, 1200.0)
            num = den * (0.01 + 0.01 * (v0 + v1)) * rng.uniform(0.5, 1.5)
            rows.append(
                {"var0": v0, "var1": v1, "todu_30ever_h6": num, "todu_amt_pile_h6": den, "oa_amt_h0": den * 0.9}
            )
        return pd.DataFrame(rows)

    def test_returns_finite_rmse_on_adequate_data(self):
        from sklearn.linear_model import LinearRegression

        rmse = io.evaluate_holdout_rmse(
            LinearRegression(),
            self._data(),
            None,
            ["var0", "var1"],
            ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            "b2_ever_h6",
            7.0,
            ["var0", "var1"],
            3.0,
            42,
        )
        assert rmse is not None and np.isfinite(rmse) and rmse >= 0

    def test_returns_none_on_tiny_data(self):
        from sklearn.linear_model import LinearRegression

        rmse = io.evaluate_holdout_rmse(
            LinearRegression(),
            self._data(n=50),
            None,
            ["var0", "var1"],
            ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            "b2_ever_h6",
            7.0,
            ["var0", "var1"],
            3.0,
            42,
        )
        assert rmse is None  # < 200 rows ⇒ no held-out report


class TestNadeauBengioCvSe:
    """Audit #9: corrected SE of a k-fold CV mean (std·√(1/k + 1/(k-1))), wider than std/√k."""

    def test_k4_matches_formula(self):
        std = 0.02
        assert io._nadeau_bengio_cv_se(std, 4) == pytest.approx(std * np.sqrt(1 / 4 + 1 / 3))

    def test_k2_matches_formula(self):
        std = 0.02
        assert io._nadeau_bengio_cv_se(std, 2) == pytest.approx(std * np.sqrt(1.5))

    def test_wider_than_naive_for_all_k(self):
        std = 0.05
        for k in (2, 3, 4, 5, 10):
            assert io._nadeau_bengio_cv_se(std, k) > std / np.sqrt(k)

    def test_fallback_below_two_folds(self):
        # No correction term defined for k<2; falls back to the naive std/√n.
        assert io._nadeau_bengio_cv_se(0.02, 1) == pytest.approx(0.02)
        assert io._nadeau_bengio_cv_se(0.0, 4) == pytest.approx(0.0)
