import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.utils import (
    DEFAULT_RISK_MULTIPLIER,
    calculate_annual_coef,
    calculate_b2_ever_h6,
    calculate_stress_factor,
    consolidate_cutoff_summaries,
    format_cutoff_summary_table,
    generate_cutoff_summary,
    get_data_information,
    optimize_dtypes,
)

# =============================================================================
# calculate_b2_ever_h6 Tests
# =============================================================================


class TestCalculateB2EverH6:
    """Tests for the calculate_b2_ever_h6 utility function."""

    def test_basic_calculation(self):
        """Test basic risk calculation."""
        result = calculate_b2_ever_h6(100, 1000)
        expected = 7 * 100 / 1000  # 0.7
        assert np.isclose(result, expected)

    def test_with_custom_multiplier(self):
        """Test calculation with custom multiplier."""
        result = calculate_b2_ever_h6(100, 1000, multiplier=10)
        expected = 10 * 100 / 1000  # 1.0
        assert np.isclose(result, expected)

    def test_as_percentage(self):
        """Test calculation with percentage output."""
        result = calculate_b2_ever_h6(100, 1000, as_percentage=True)
        expected = 7 * 100 / 1000 * 100  # 70.0
        assert np.isclose(result, expected)

    def test_division_by_zero_scalar(self):
        """Test that division by zero returns NaN for scalar."""
        result = calculate_b2_ever_h6(100, 0)
        assert np.isnan(result)

    def test_with_pandas_series(self):
        """Test calculation with pandas Series."""
        numerator = pd.Series([100, 200, 300])
        denominator = pd.Series([1000, 1000, 1000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert isinstance(result, (pd.Series, np.ndarray))
        assert len(result) == 3
        assert np.isclose(result[0], 0.7)
        assert np.isclose(result[1], 1.4)
        assert np.isclose(result[2], 2.1)

    def test_with_numpy_array(self):
        """Test calculation with numpy arrays."""
        numerator = np.array([100, 200, 300])
        denominator = np.array([1000, 1000, 1000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_division_by_zero_in_series(self):
        """Test that division by zero in series returns NaN for that element."""
        numerator = pd.Series([100, 200, 300])
        denominator = pd.Series([1000, 0, 1000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert np.isclose(result[0], 0.7)
        assert np.isnan(result[1])
        assert np.isclose(result[2], 2.1)

    def test_rounding(self):
        """Test that result is rounded to specified decimals."""
        result = calculate_b2_ever_h6(100, 300, decimals=4)
        expected = round(7 * 100 / 300, 4)
        assert result == expected

    def test_custom_decimals(self):
        """Test custom decimal places."""
        result = calculate_b2_ever_h6(100, 300, decimals=6)
        expected = round(7 * 100 / 300, 6)
        assert result == expected

    def test_default_multiplier_constant(self):
        """Test that default multiplier constant is correct."""
        assert DEFAULT_RISK_MULTIPLIER == 7

    def test_zero_numerator(self):
        """Test that zero numerator returns zero."""
        result = calculate_b2_ever_h6(0, 1000)
        assert result == 0.0

    def test_negative_values(self):
        """Test calculation with negative values."""
        result = calculate_b2_ever_h6(-100, 1000)
        expected = 7 * -100 / 1000
        assert np.isclose(result, expected)

    def test_mixed_series_with_zeros(self):
        """Test series with multiple zeros in denominator."""
        numerator = pd.Series([100, 200, 300, 400])
        denominator = pd.Series([1000, 0, 0, 2000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert np.isclose(result[0], 0.7)
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isclose(result[3], 1.4)

    def test_empty_series(self):
        """Test with empty series."""
        numerator = pd.Series([], dtype=float)
        denominator = pd.Series([], dtype=float)

        result = calculate_b2_ever_h6(numerator, denominator)

        assert len(result) == 0

    def test_single_element_series(self):
        """Test with single element series."""
        numerator = pd.Series([100])
        denominator = pd.Series([1000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert len(result) == 1
        assert np.isclose(result[0], 0.7)

    def test_percentage_with_series(self):
        """Test percentage output with series."""
        numerator = pd.Series([100, 200])
        denominator = pd.Series([1000, 1000])

        result = calculate_b2_ever_h6(numerator, denominator, as_percentage=True)

        assert np.isclose(result[0], 70.0)
        assert np.isclose(result[1], 140.0)


# =============================================================================
# Integration Tests
# =============================================================================


class TestB2EverH6Integration:
    """Integration tests for b2_ever_h6 calculation in realistic scenarios."""

    def test_realistic_risk_data(self):
        """Test with realistic risk data."""
        # Simulate aggregated loan data
        data = pd.DataFrame(
            {
                "todu_30ever_h6": [10, 25, 50, 100],
                "todu_amt_pile_h6": [1000, 2000, 3000, 4000],
            }
        )

        result = calculate_b2_ever_h6(data["todu_30ever_h6"], data["todu_amt_pile_h6"])

        # Expected: 7 * num / den
        expected = 7 * data["todu_30ever_h6"] / data["todu_amt_pile_h6"]

        np.testing.assert_array_almost_equal(result, expected.round(2))

    def test_dataframe_column_assignment(self):
        """Test assigning result to DataFrame column."""
        df = pd.DataFrame(
            {
                "todu_30ever_h6": [10, 25, 50],
                "todu_amt_pile_h6": [1000, 2000, 3000],
            }
        )

        df["b2_ever_h6"] = calculate_b2_ever_h6(df["todu_30ever_h6"], df["todu_amt_pile_h6"], as_percentage=True)

        assert "b2_ever_h6" in df.columns
        assert not df["b2_ever_h6"].isna().any()





# =============================================================================
# optimize_dtypes Tests
# =============================================================================


class TestOptimizeDtypes:
    def test_int64_small_positive(self):
        df = pd.DataFrame({"a": pd.array([1, 2, 100], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.uint8

    def test_int64_medium_positive(self):
        df = pd.DataFrame({"a": pd.array([1, 2, 500], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.uint16

    def test_int64_large_positive(self):
        df = pd.DataFrame({"a": pd.array([1, 2, 100000], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.uint32

    def test_int64_negative_small(self):
        df = pd.DataFrame({"a": pd.array([-10, 0, 50], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.int8

    def test_int64_negative_medium(self):
        df = pd.DataFrame({"a": pd.array([-200, 0, 200], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.int16

    def test_int64_negative_large(self):
        df = pd.DataFrame({"a": pd.array([-50000, 0, 50000], dtype="int64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.int32

    def test_float64_to_float32(self):
        df = pd.DataFrame({"a": pd.array([1.1, 2.2, 3.3], dtype="float64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.float32

    def test_preserves_string_columns(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = optimize_dtypes(df)
        # String columns should not be converted to numeric
        assert not pd.api.types.is_numeric_dtype(result["a"])


# =============================================================================
# get_data_information Tests
# =============================================================================


class TestGetDataInformation:
    def test_returns_dataframe(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})
        result = get_data_information(df)
        assert isinstance(result, pd.DataFrame)
        assert "Variable" in result.columns
        assert "Number of missing values" in result.columns

    def test_missing_values_reported(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = get_data_information(df)
        a_row = result[result["Variable"] == "a"]
        assert a_row["Number of missing values"].values[0] == 1





# =============================================================================
# calculate_stress_factor Tests
# =============================================================================


class TestCalculateStressFactor:
    def test_basic_stress(self):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "status_name": ["booked"] * 100,
                "risk_score_rf": np.random.rand(100),
                "todu_30ever_h6": np.random.rand(100) * 10,
                "todu_amt_pile_h6": np.random.rand(100) * 1000 + 100,
            }
        )
        stress = calculate_stress_factor(df)
        assert isinstance(stress, float)
        assert stress >= 0

    def test_empty_target_status(self):
        df = pd.DataFrame(
            {
                "status_name": ["rejected"] * 10,
                "risk_score_rf": np.random.rand(10),
                "todu_30ever_h6": np.random.rand(10),
                "todu_amt_pile_h6": np.random.rand(10),
            }
        )
        stress = calculate_stress_factor(df)
        assert stress == 0.0

    def test_stress_factor_greater_than_1(self):
        """Worst fraction should typically have higher risk than overall."""
        np.random.seed(42)
        n = 1000
        scores = np.random.rand(n)
        # Lower scores = higher risk
        risk = np.where(scores < 0.1, 50, 5)
        df = pd.DataFrame(
            {
                "status_name": ["booked"] * n,
                "risk_score_rf": scores,
                "todu_30ever_h6": risk,
                "todu_amt_pile_h6": np.ones(n) * 100,
            }
        )
        stress = calculate_stress_factor(df, frac=0.10)
        assert stress > 1.0


# =============================================================================
# calculate_annual_coef Tests
# =============================================================================


class TestCalculateAnnualCoef:
    def test_12_months(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-12-01")
        coef = calculate_annual_coef(start, end)
        assert coef == pytest.approx(1.0)

    def test_6_months(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2023-06-01")
        coef = calculate_annual_coef(start, end)
        assert coef == pytest.approx(2.0)

    def test_24_months(self):
        start = pd.Timestamp("2023-01-01")
        end = pd.Timestamp("2024-12-01")
        coef = calculate_annual_coef(start, end)
        assert coef == pytest.approx(0.5)


# =============================================================================
# generate_cutoff_summary Tests
# =============================================================================


class TestGenerateCutoffSummary:
    def test_basic_summary(self):
        opt_df = pd.DataFrame({"sol_fac": [0], 1: [2], 2: [3], 3: [4]})
        result = generate_cutoff_summary(opt_df, ["var0", "var1"], "segment_a")
        assert len(result) == 3
        assert "segment" in result.columns
        assert "cutoff_value" in result.columns

    def test_empty_solution(self):
        result = generate_cutoff_summary(pd.DataFrame(), ["var0", "var1"], "seg")
        assert result.empty

    def test_none_solution(self):
        result = generate_cutoff_summary(None, ["var0", "var1"], "seg")
        assert result.empty

    def test_scenario_and_risk_values(self):
        opt_df = pd.DataFrame({"sol_fac": [0], 1: [5], 2: [6]})
        result = generate_cutoff_summary(
            opt_df, ["var0", "var1"], "seg", scenario_name="optimistic", risk_value=2.5, production_value=1000
        )
        assert result["scenario"].iloc[0] == "optimistic"
        assert result["risk_pct"].iloc[0] == 2.5
        assert result["production"].iloc[0] == 1000


# =============================================================================
# format_cutoff_summary_table Tests
# =============================================================================


class TestFormatCutoffSummaryTable:
    def test_basic_format(self):
        summary = pd.DataFrame(
            {
                "segment": ["seg_a", "seg_a"],
                "scenario": ["base", "base"],
                "var0_bin": [1, 2],
                "var0_name": ["var0", "var0"],
                "cutoff_value": [3, 5],
                "var1_name": ["var1", "var1"],
                "risk_pct": [2.0, 2.0],
                "production": [1000, 1000],
            }
        )
        result = format_cutoff_summary_table(summary, ["var0", "var1"])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1  # One row per segment+scenario

    def test_empty_input(self):
        result = format_cutoff_summary_table(pd.DataFrame(), ["var0", "var1"])
        assert result.empty


# =============================================================================
# consolidate_cutoff_summaries Tests
# =============================================================================


class TestConsolidateCutoffSummaries:
    def test_basic_consolidation(self):
        s1 = pd.DataFrame({"segment": ["a"], "cutoff": [1]})
        s2 = pd.DataFrame({"segment": ["b"], "cutoff": [2]})
        result = consolidate_cutoff_summaries([s1, s2])
        assert len(result) == 2

    def test_empty_list(self):
        result = consolidate_cutoff_summaries([])
        assert result.empty

    def test_all_empty_summaries(self):
        result = consolidate_cutoff_summaries([pd.DataFrame(), pd.DataFrame()])
        assert result.empty

    def test_mixed_empty_and_valid(self):
        valid = pd.DataFrame({"segment": ["a"], "cutoff": [1]})
        result = consolidate_cutoff_summaries([pd.DataFrame(), valid])
        assert len(result) == 1

    def test_save_to_csv(self, tmp_path):
        s1 = pd.DataFrame({"segment": ["a"], "cutoff": [1]})
        output_path = str(tmp_path / "consolidated.csv")
        result = consolidate_cutoff_summaries([s1], output_path=output_path)
        assert len(result) == 1
        # Verify file was created
        saved = pd.read_csv(output_path)
        assert len(saved) == 1
