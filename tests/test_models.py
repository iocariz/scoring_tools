import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from src.model.models import (
    calculate_B2,
    calculate_risk_values,
    calculate_RV,
    extract_splits_from_tree,
    optimal_splits_using_tree,
    preprocess_data,
    transform_variables,
)

# =============================================================================
# extract_splits_from_tree Tests
# =============================================================================


class TestExtractSplitsFromTree:
    def test_basic_splits(self):
        np.random.seed(42)
        X = pd.DataFrame({"x": np.random.randn(200)})
        y = (X["x"] > 0).astype(int)
        tree = DecisionTreeClassifier(max_leaf_nodes=3, min_samples_leaf=10)
        tree.fit(X, y)
        splits = extract_splits_from_tree(tree, ["x"])
        assert isinstance(splits, list)
        assert len(splits) > 0
        # Splits should be sorted
        assert splits == sorted(splits)

    def test_single_leaf(self):
        """A tree with only one leaf has no splits."""
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        y = [0, 0, 0, 0, 0]  # No variation
        tree = DecisionTreeClassifier(max_leaf_nodes=2, min_samples_leaf=10)
        tree.fit(X, y)
        splits = extract_splits_from_tree(tree, ["x"])
        # Tree may not split at all with constant target
        assert isinstance(splits, list)


# =============================================================================
# optimal_splits_using_tree Tests
# =============================================================================


class TestOptimalSplitsUsingTree:
    def test_creates_groups(self):
        np.random.seed(42)
        df = pd.DataFrame({"score": np.random.randn(1000), "target": np.random.randint(0, 2, 1000)})
        result = optimal_splits_using_tree(df, "score", "target", 3)
        assert "group" in result.columns
        assert result["group"].nunique() <= 3


# =============================================================================
# transform_variables Tests
# =============================================================================


class TestTransformVariables:
    def test_creates_expected_columns(self):
        """PolynomialFeatures(degree=3) produces polynomial + interaction terms."""
        df = pd.DataFrame({"var0": [1, 2, 3], "var1": [4, 5, 6]})
        result = transform_variables(df, ["var0", "var1"])

        # PolynomialFeatures names: "var0^2", "var0 var1", "var1^2",
        # "var0^3", "var0^2 var1", "var0 var1^2", "var1^3"
        assert "var0 var1" in result.columns
        assert "var0^2" in result.columns
        assert "var1^2" in result.columns
        assert "var0^3" in result.columns
        assert "var1^3" in result.columns
        assert "var0^2 var1" in result.columns
        assert "var0 var1^2" in result.columns

    def test_interaction_term(self):
        df = pd.DataFrame({"a": [2, 3], "b": [4, 5]})
        result = transform_variables(df, ["a", "b"])
        assert result["a b"].iloc[0] == 8
        assert result["a b"].iloc[1] == 15

    def test_square_term(self):
        df = pd.DataFrame({"a": [3, 4], "b": [1, 2]})
        result = transform_variables(df, ["a", "b"])
        assert result["a^2"].iloc[0] == 9
        assert result["a^2"].iloc[1] == 16

    def test_preserves_original_columns(self):
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = transform_variables(df, ["x", "y"])
        assert "x" in result.columns
        assert "y" in result.columns

    def test_does_not_mutate_input(self):
        """transform_variables should not modify the input DataFrame."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        original_cols = list(df.columns)
        _ = transform_variables(df, ["a", "b"])
        assert list(df.columns) == original_cols

    def test_consistent_2d_and_3d(self):
        """2-var and 3-var paths should both use PolynomialFeatures."""
        df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result2 = transform_variables(df2, ["a", "b"])
        # Both should produce PolynomialFeatures-style names
        assert "a b" in result2.columns  # not "a_b"
        assert "a^2 b" in result2.columns  # not "a^2 x b"

        df3 = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
        result3 = transform_variables(df3, ["a", "b", "c"])
        assert "a b" in result3.columns
        assert "a c" in result3.columns


# =============================================================================
# preprocess_data Tests
# =============================================================================


class TestPreprocessData:
    def test_basic_preprocessing(self):
        data = pd.DataFrame(
            {
                "sc_octroi_new_clus": [0, 0, 1, 1, 0, 1],
                "new_efx_clus": [0, 1, 0, 1, 0, 1],
                "status_name": ["booked"] * 6,
                "todu_30ever_h6": [10, 20, 30, 40, 15, 25],
                "todu_amt_pile_h6": [1000, 2000, 3000, 4000, 1500, 2500],
            }
        )
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6"]
        octroi_limits = [-float("inf"), 0.5, float("inf")]  # 2 bins
        efx_limits = [-float("inf"), 0.5, float("inf")]  # 2 bins

        result = preprocess_data(data, octroi_limits, efx_limits, variables, indicators, "b2_ever_h6", 7)
        assert isinstance(result, pd.DataFrame)
        assert "b2_ever_h6" in result.columns
        assert len(result) == 4  # 2x2 grid


# =============================================================================
# calculate_B2 Tests
# =============================================================================


class TestCalculateB2:
    def test_basic_b2(self):
        df = pd.DataFrame({"var0": [0, 1, 2], "var1": [0, 1, 2]})
        variables = ["var0", "var1"]

        # Create a simple model mock
        model = Ridge(fit_intercept=False)
        # Fit on transformed data so we know the features
        from src.model.models import transform_variables as tv

        train_df = tv(df.copy(), variables)
        var_reg = [c for c in train_df.columns if c not in ["var0", "var1"]]
        y = np.array([0.1, 0.2, 0.3])
        model.fit(train_df[var_reg], y)

        result = calculate_B2(df, model, variables, stressor=1.0, var_reg=var_reg)
        assert "b2_ever_h6" in result.columns
        assert (result["b2_ever_h6"] >= 0).all()

    def test_stressor_scaling(self):
        df = pd.DataFrame({"v0": [1, 2], "v1": [1, 2]})
        variables = ["v0", "v1"]

        model = Ridge(fit_intercept=False)
        from src.model.models import transform_variables as tv

        train_df = tv(df.copy(), variables)
        var_reg = [c for c in train_df.columns if c not in ["v0", "v1"]]
        y = np.array([0.5, 1.0])
        model.fit(train_df[var_reg], y)

        result_1x = calculate_B2(df.copy(), model, variables, stressor=1.0, var_reg=var_reg)
        result_2x = calculate_B2(df.copy(), model, variables, stressor=2.0, var_reg=var_reg)

        # With double stress, predictions should roughly double (clipped to >= 0)
        np.testing.assert_array_almost_equal(
            result_2x["b2_ever_h6"].values,
            np.clip(2.0 * result_1x["b2_ever_h6"].values, 0, None),
        )


# =============================================================================
# calculate_RV Tests
# =============================================================================


class TestCalculateRV:
    def test_basic_rv(self):
        df = pd.DataFrame({"oa_amt": [100, 200, 300]})
        model = LinearRegression()
        # Fit with DataFrame to preserve feature names, matching usage in calculate_RV
        model.fit(pd.DataFrame({"oa_amt": [100, 200, 300]}), np.array([110, 220, 330]))
        result = calculate_RV(df, model)
        assert "todu_amt_pile_h6" in result.columns
        assert len(result) == 3


# =============================================================================
# calculate_risk_values Tests
# =============================================================================


class TestCalculateRiskValues:
    def test_full_pipeline(self):
        df = pd.DataFrame({"oa_amt": [100, 200, 300], "v0": [0, 1, 2], "v1": [0, 1, 2]})
        variables = ["v0", "v1"]

        # Create RV model
        model_rv = LinearRegression()
        # Fit with DataFrame to preserve feature names
        model_rv.fit(pd.DataFrame({"oa_amt": [100, 200, 300]}), np.array([1000, 2000, 3000]))

        # Create risk model
        from src.model.models import transform_variables as tv

        train_df = tv(df[["v0", "v1"]].copy(), variables)
        var_reg = [c for c in train_df.columns if c not in ["v0", "v1"]]
        model_risk = Ridge(fit_intercept=False)
        model_risk.fit(train_df[var_reg], np.array([0.1, 0.2, 0.3]))

        result = calculate_risk_values(df, model_risk, model_rv, variables, stressor=1.0, var_reg=var_reg)

        assert "todu_amt_pile_h6" in result.columns
        assert "b2_ever_h6" in result.columns
        assert "todu_30ever_h6" in result.columns

    def test_empty_input_returns_empty_frame_not_error(self):
        """A 0-row frame is a legitimate input, not a failure.

        An MR cohort can contain no score-rejected demand at all (e.g. a segment that
        accepts every cell), which makes the repesca aggregate empty. sklearn refuses to
        predict on 0 samples, so this used to raise ValueError deep inside the MR run —
        swallowed by process_mr_period, leaving the segment silently without MR,
        stability or drift artifacts while still reporting success.
        """
        variables = ["v0", "v1"]
        model_rv = LinearRegression()
        model_rv.fit(pd.DataFrame({"oa_amt": [100.0, 200.0, 300.0]}), np.array([1000.0, 2000.0, 3000.0]))

        from src.model.models import transform_variables as tv

        train_df = tv(pd.DataFrame({"v0": [0, 1, 2], "v1": [0, 1, 2]}), variables)
        var_reg = [c for c in train_df.columns if c not in ["v0", "v1"]]
        model_risk = Ridge(fit_intercept=False)
        model_risk.fit(train_df[var_reg], np.array([0.1, 0.2, 0.3]))

        empty = pd.DataFrame({c: pd.Series(dtype=float) for c in ["oa_amt", "v0", "v1"]})
        result = calculate_risk_values(empty, model_risk, model_rv, variables, stressor=1.0, var_reg=var_reg)

        assert len(result) == 0
        for col in ("todu_amt_pile_h6", "b2_ever_h6", "todu_30ever_h6"):
            assert col in result.columns


# =============================================================================
# transform_variables Edge Case Tests
# =============================================================================


class TestTransformVariablesEdgeCases:
    def test_all_columns_created(self):
        """Verify all expected polynomial columns are created for 2 variables."""
        df = pd.DataFrame({"var0": [1, 2], "var1": [3, 4]})
        result = transform_variables(df, ["var0", "var1"])
        expected_new_columns = [
            "var0 var1",
            "var0^2",
            "var1^2",
            "var0^3",
            "var1^3",
            "var0^2 var1",
            "var0 var1^2",
        ]
        for col in expected_new_columns:
            assert col in result.columns, f"Missing expected column: {col}"


# =============================================================================
# Audit 2026-11-27 A21: HRI repesca rate — cap and stress
# =============================================================================


class _ConstantRateModel:
    """Predicts the same rate for every row; no feature names, so inputs pass through."""

    def __init__(self, rate: float):
        self.rate = rate

    def predict(self, X):
        return np.full(len(X), self.rate)


class TestHriRepescaRate:
    VARIABLES = ["var0", "var1"]

    def _score(self, predicted_rate: float, stressor: float, rate_cap: float | None) -> float:
        from src.model.models import calculate_hri_values
        from src.model.models import transform_variables as tv

        df = pd.DataFrame({"var0": [1.0], "var1": [1.0], "oa_amt": [1000.0]})
        var_reg = [c for c in tv(df.copy(), self.VARIABLES).columns if c not in df.columns]
        out = calculate_hri_values(
            df,
            _ConstantRateModel(predicted_rate),
            _ConstantRateModel(600.0),
            self.VARIABLES,
            stressor,
            var_reg,
            rate_cap=rate_cap,
        )
        return float(out["hri_h6"].iloc[0])

    def test_stress_applies_on_top_of_the_capped_rate(self):
        """The cap bounds the MODEL ("as bad as the worst observed bin"); stress then applies to
        it. Stressing first and clipping after wiped the stress out for exactly the capped cells."""
        assert self._score(0.50, stressor=2.0, rate_cap=0.20) == pytest.approx(0.40)

    def test_uncapped_rate_is_stressed_as_before(self):
        assert self._score(0.10, stressor=2.0, rate_cap=0.20) == pytest.approx(0.20)
        assert self._score(0.10, stressor=2.0, rate_cap=None) == pytest.approx(0.20)

    def test_cap_alone_without_stress(self):
        assert self._score(0.50, stressor=1.0, rate_cap=0.20) == pytest.approx(0.20)

    def test_observed_cap_ignores_winsorization(self):
        """The fit's target column is winsorized (#56); the cap must be the worst bin as OBSERVED,
        from the raw sums. Nine bins near 1% and one at 28.8% gave a cap of 1.24%."""
        from src.model.inference_optimized import _observed_hri_cap

        train = pd.DataFrame(
            {
                "h_num_h6": [1.0] * 9 + [28.8],
                "h_den_h6": [100.0] * 10,
                "hri_h6": [0.01] * 9 + [0.0124],  # clipped target the model was fitted on
            }
        )
        assert _observed_hri_cap(train) == pytest.approx(0.288)

    def test_observed_cap_skips_bins_without_exposure_and_missing_columns(self):
        from src.model.inference_optimized import _observed_hri_cap

        train = pd.DataFrame({"h_num_h6": [1.0, 5.0, np.nan], "h_den_h6": [100.0, 0.0, 100.0]})
        assert _observed_hri_cap(train) == pytest.approx(0.01)
        assert _observed_hri_cap(pd.DataFrame({"hri_h6": [0.3]})) is None
        assert _observed_hri_cap(None) is None

    def test_no_signal_guard_only_fires_on_a_measured_r2(self):
        """A tree winner records CV R² = NaN ("not computed"). `float(nan or 0.0) <= 0` is False,
        so it has taken the model branch since #175 — by accident. That is the right branch (a
        tree cannot extrapolate beyond its training targets); the rule is now explicit."""
        from src.model.inference_optimized import _hri_fit_has_no_signal

        assert _hri_fit_has_no_signal(0.0) is True
        assert _hri_fit_has_no_signal(-0.3) is True
        assert _hri_fit_has_no_signal(0.2) is False
        assert _hri_fit_has_no_signal(float("nan")) is False
        assert _hri_fit_has_no_signal(None) is True  # legacy metadata without the field
