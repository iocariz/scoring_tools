import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

from src.constants import SCORE_SCALE_MAX
from src.models import (
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
        df = pd.DataFrame({"var0": [1, 2, 3], "var1": [4, 5, 6]})
        result = transform_variables(df, ["var0", "var1"])

        assert "var0_var1" in result.columns
        assert "var0^2" in result.columns
        assert "var1^2" in result.columns
        assert "var0^3" in result.columns
        assert "var1^3" in result.columns
        assert "var0^2 x var1" in result.columns
        assert "var0 x var1^2" in result.columns

    def test_interaction_term(self):
        df = pd.DataFrame({"a": [2, 3], "b": [4, 5]})
        result = transform_variables(df, ["a", "b"])
        assert result["a_b"].iloc[0] == 8
        assert result["a_b"].iloc[1] == 15

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
        from src.models import transform_variables as tv

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
        from src.models import transform_variables as tv

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
        from src.models import transform_variables as tv

        train_df = tv(df[["v0", "v1"]].copy(), variables)
        var_reg = [c for c in train_df.columns if c not in ["v0", "v1"]]
        model_risk = Ridge(fit_intercept=False)
        model_risk.fit(train_df[var_reg], np.array([0.1, 0.2, 0.3]))

        result = calculate_risk_values(df, model_risk, model_rv, variables, stressor=1.0, var_reg=var_reg)

        assert "todu_amt_pile_h6" in result.columns
        assert "b2_ever_h6" in result.columns
        assert "todu_30ever_h6" in result.columns


# =============================================================================
# transform_variables Edge Case Tests
# =============================================================================


class TestTransformVariablesEdgeCases:
    def test_all_columns_created(self):
        """Verify all expected columns are created by transform_variables."""
        df = pd.DataFrame({"var0": [1, 2], "var1": [3, 4]})
        result = transform_variables(df, ["var0", "var1"])
        expected_new_columns = [
            "var0_var1",
            "var0^2",
            "var1^2",
            "var0^3",
            "var1^3",
            "var0^2 x var1",
            "var0 x var1^2",
        ]
        for col in expected_new_columns:
            assert col in result.columns, f"Missing expected column: {col}"
