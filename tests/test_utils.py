import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    DEFAULT_RISK_MULTIPLIER,
    calculate_annual_coef,
    calculate_b2_ever_h6,
    calculate_stress_factor,
    calculate_todu_30ever_from_b2,
    consolidate_cutoff_summaries,
    extrapolate_h3_to_h6,
    fit_h3_extrapolation_curve,
    format_cutoff_summary_table,
    generate_cutoff_summary,
    get_data_information,
    optimize_dtypes,
    resolve_modelling_supersegment,
    resolve_reporting_supersegment,
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
        """Test that negative risk is clipped to zero."""
        result = calculate_b2_ever_h6(-100, 1000)
        assert result == 0.0

    def test_mixed_series_with_zeros(self):
        """Test series with multiple zeros in denominator."""
        numerator = pd.Series([100, 200, 300, 400])
        denominator = pd.Series([1000, 0, 0, 2000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert np.isclose(result[0], 0.7)
        assert np.isnan(result[1])
        assert np.isnan(result[2])
        assert np.isclose(result[3], 1.4)

    def test_returns_series_when_any_input_is_series(self):
        """todo #55 contract: any Series input → Series output with `.isna()`."""
        num_series = pd.Series([100, 200, 0], index=["a", "b", "c"])
        denom_array = np.array([1000.0, 0.0, 500.0])
        result = calculate_b2_ever_h6(num_series, denom_array)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["a", "b", "c"]
        assert hasattr(result, "isna")
        assert int(result.isna().sum()) == 1  # denom=0 → NaN

        # Reverse: scalar numerator, Series denominator → Series output
        result2 = calculate_b2_ever_h6(100.0, pd.Series([1000.0, 500.0], index=["x", "y"]))
        assert isinstance(result2, pd.Series)
        assert list(result2.index) == ["x", "y"]

    def test_returns_scalar_when_both_inputs_scalar(self):
        """todo #55 contract: both scalar → float (no .isna)."""
        result = calculate_b2_ever_h6(100, 1000)
        # The function returns a numpy scalar; plain `isinstance(result, float)`
        # would miss np.float64. Check both the value type and the absence of
        # Series / ndarray attributes.
        assert not isinstance(result, (pd.Series, np.ndarray))
        assert isinstance(result, (float, np.floating))

    def test_returns_ndarray_when_both_inputs_ndarray(self):
        """todo #55 contract: ndarray+ndarray → ndarray."""
        result = calculate_b2_ever_h6(np.array([100.0, 200.0]), np.array([1000.0, 500.0]))
        assert isinstance(result, np.ndarray)

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

        result = calculate_b2_ever_h6(data["todu_30ever_h6"], data["todu_amt_pile_h6"], decimals=2)

        # Expected: 7 * num / den, rounded to 2 decimals (display precision)
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

    def test_float64_preserved(self):
        """Float64 is preserved to avoid precision loss on financial data."""
        df = pd.DataFrame({"a": pd.array([1.1, 2.2, 3.3], dtype="float64")})
        result = optimize_dtypes(df)
        assert result["a"].dtype == np.float64

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
        # Returns neutral 1.0 (not 0.0) to avoid zeroing out all risk predictions
        assert stress == 1.0

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

    def test_2var_with_mask_and_grid_produces_cell_level(self):
        # With mask/grid the summary must be cell-level for ANY N (the
        # consolidated acceptance grids need the `accepted` column) — the
        # classic per-bin cutoff table is only the mask-less fallback.
        from src.optimization_utils import CellGrid

        summary_df = pd.DataFrame(
            {
                "var0": [1.0, 1.0, 2.0, 2.0],
                "var1": [1.0, 2.0, 1.0, 2.0],
                "oa_amt_h0": [100.0, 100.0, 100.0, 100.0],
            }
        )
        grid = CellGrid.from_summary(summary_df, ["var0", "var1"])
        mask = np.array([1, 0, 1, 1])
        opt_df = pd.DataFrame({"sol_fac": [0], 1: [2], 2: [1]})

        result = generate_cutoff_summary(
            opt_df, ["var0", "var1"], "seg", risk_value=1.5, production_value=300, mask=mask, grid=grid
        )

        assert "accepted" in result.columns
        assert len(result) == 4
        assert int(result["accepted"].sum()) == 3
        assert set(result.columns) >= {"var0", "var1", "segment", "scenario", "risk_pct", "production"}


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


# =============================================================================
# calculate_b2_ever_h6 Edge Case Tests
# =============================================================================


class TestCalculateB2EverH6EdgeCases:
    """Additional edge case tests for calculate_b2_ever_h6."""

    def test_all_nan_numerator(self):
        """All-NaN numerator should produce all-NaN result."""
        numerator = pd.Series([np.nan, np.nan, np.nan])
        denominator = pd.Series([1000, 2000, 3000])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert len(result) == 3
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_all_zero_denominator_series(self):
        """All-zero denominator Series should produce all-NaN result."""
        numerator = pd.Series([100, 200, 300])
        denominator = pd.Series([0, 0, 0])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert len(result) == 3
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert np.isnan(result[2])

    def test_very_large_values_no_overflow(self):
        """Very large values (1e15) should not overflow."""
        numerator = pd.Series([1e15, 2e15])
        denominator = pd.Series([1e15, 1e15])

        result = calculate_b2_ever_h6(numerator, denominator)

        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isclose(result[0], 7.0)
        assert np.isclose(result[1], 14.0)

    def test_empty_series_inputs(self):
        """Empty Series inputs should return an empty result."""
        numerator = pd.Series([], dtype=float)
        denominator = pd.Series([], dtype=float)

        result = calculate_b2_ever_h6(numerator, denominator)

        assert len(result) == 0


# =============================================================================
# calculate_todu_30ever_from_b2 Tests
# =============================================================================


class TestCalculateTodu30everFromB2:
    """Tests for the calculate_todu_30ever_from_b2 inverse function."""

    def test_basic_calculation(self):
        """Test basic inverse calculation: 0.7 * 1000 / 7 = 100.0."""
        result = calculate_todu_30ever_from_b2(0.7, 1000)
        expected = 0.7 * 1000 / 7  # 100.0
        assert np.isclose(result, expected)

    def test_with_custom_multiplier(self):
        """Test calculation with a custom multiplier."""
        result = calculate_todu_30ever_from_b2(1.0, 1000, multiplier=10)
        expected = 1.0 * 1000 / 10  # 100.0
        assert np.isclose(result, expected)

    def test_with_pandas_series(self):
        """Test calculation with pandas Series inputs."""
        b2 = pd.Series([0.7, 1.4, 2.1])
        pile = pd.Series([1000, 1000, 1000])

        result = calculate_todu_30ever_from_b2(b2, pile)

        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert np.isclose(result[0], 100.0)
        assert np.isclose(result[1], 200.0)
        assert np.isclose(result[2], 300.0)

    def test_inverse_of_b2_ever_h6(self):
        """Verify round-trip: calculate_b2_ever_h6(calculate_todu_30ever_from_b2(b2, pile), pile) == b2."""
        b2_values = pd.Series([0.5, 1.0, 1.5, 2.0])
        pile_values = pd.Series([1000, 2000, 3000, 4000])

        todu = calculate_todu_30ever_from_b2(b2_values, pile_values)
        recovered_b2 = calculate_b2_ever_h6(todu, pile_values)

        np.testing.assert_array_almost_equal(recovered_b2, b2_values, decimal=2)


# =============================================================================
# extrapolate_h3_to_h6 Tests
# =============================================================================


class TestExtrapolateH3ToH6:
    """Tests for the extrapolate_h3_to_h6 function."""

    def test_linear_matches_direct_multiplication(self):
        """Linear method should give h3 * ratio."""
        h3 = pd.Series([0.5, 1.0, 2.0])
        ratio = pd.Series([2.0, 1.5, 3.0])
        result = extrapolate_h3_to_h6(h3, ratio, method="linear")
        expected = h3 * ratio
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_curvature_1_equals_linear(self):
        """Power with alpha=1.0 should match linear."""
        h3 = pd.Series([0.5, 1.0, 2.0])
        ratio = pd.Series([2.0, 1.5, 3.0])
        linear = extrapolate_h3_to_h6(h3, ratio, method="linear")
        power = extrapolate_h3_to_h6(h3, ratio, method="power", curvature=1.0)
        np.testing.assert_array_almost_equal(power, linear)

    def test_power_convex(self):
        """Alpha > 1 should produce higher values than linear for ratio > 1."""
        h3 = 1.0
        ratio = 2.0
        linear = extrapolate_h3_to_h6(h3, ratio, method="linear")
        power = extrapolate_h3_to_h6(h3, ratio, method="power", curvature=1.5)
        assert power > linear

    def test_power_concave(self):
        """Alpha < 1 should produce lower values than linear for ratio > 1."""
        h3 = 1.0
        ratio = 2.0
        linear = extrapolate_h3_to_h6(h3, ratio, method="linear")
        power = extrapolate_h3_to_h6(h3, ratio, method="power", curvature=0.5)
        assert power < linear

    def test_logistic_caps_extreme_ratios(self):
        """Logistic output should grow slower than linear for large ratios."""
        h3 = 1.0
        ratio = 5.0
        linear = extrapolate_h3_to_h6(h3, ratio, method="linear")
        logistic = extrapolate_h3_to_h6(h3, ratio, method="logistic", curvature=1.0)
        # Logistic scaling caps extreme ratios, so result < linear
        assert logistic < linear

    def test_logistic_near_linear_for_moderate_ratios(self):
        """Logistic should approximate linear when ratio is close to 1."""
        h3 = 1.0
        ratio = 1.05
        linear = extrapolate_h3_to_h6(h3, ratio, method="linear")
        logistic = extrapolate_h3_to_h6(h3, ratio, method="logistic", curvature=1.0)
        assert abs(logistic - linear) < 0.01

    def test_scalar_series_array_support(self):
        """All input types (scalar, Series, ndarray) should work."""
        # Scalar
        s = extrapolate_h3_to_h6(1.0, 2.0, method="power", curvature=1.5)
        assert isinstance(s, float)

        # Series
        ser = extrapolate_h3_to_h6(pd.Series([1.0, 2.0]), pd.Series([2.0, 1.5]), method="power", curvature=1.5)
        assert isinstance(ser, pd.Series)
        assert len(ser) == 2

        # ndarray
        arr = extrapolate_h3_to_h6(np.array([1.0, 2.0]), np.array([2.0, 1.5]), method="power", curvature=1.5)
        assert isinstance(arr, np.ndarray)
        assert len(arr) == 2

    def test_unknown_method_raises(self):
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown extrapolation method"):
            extrapolate_h3_to_h6(1.0, 2.0, method="cubic")


# =============================================================================
# fit_h3_extrapolation_curve Tests
# =============================================================================


class TestFitH3ExtrapolationCurve:
    """Tests for the fit_h3_extrapolation_curve function."""

    def test_linear_relationship_returns_linear(self):
        """Bins with constant H6/H3 ratio → alpha ≈ 1 → 'linear'."""
        rng = np.random.RandomState(42)
        b2_h3 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        b2_h6 = b2_h3 * 2.0 + rng.normal(0, 0.001, len(b2_h3))  # Nearly perfect linear scaling
        method, curvature, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert method == "linear"
        assert curvature == 1.0
        assert abs(diag["alpha"] - 1.0) < 0.1

    def test_convex_relationship_returns_power_gt1(self):
        """Bins where H6/H3 ratio increases with H3 → alpha > 1 → 'power'."""
        b2_h3 = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        # H6 = H3^1.5 → log-log slope = 1.5
        b2_h6 = b2_h3**1.5
        method, curvature, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert method == "power"
        assert curvature > 1.0
        assert np.isclose(diag["alpha"], 1.5, atol=0.01)

    def test_concave_relationship_returns_power_lt1(self):
        """Bins where ratio decreases with H3 → alpha < 1 → 'power'."""
        b2_h3 = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        # H6 = H3^0.5 → log-log slope = 0.5
        b2_h6 = b2_h3**0.5
        method, curvature, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert method == "power"
        assert curvature < 1.0
        assert np.isclose(diag["alpha"], 0.5, atol=0.01)

    def test_insufficient_bins_returns_linear(self):
        """Fewer than min_bins valid bins → fallback to linear."""
        b2_h3 = np.array([1.0, 2.0, 3.0])  # Only 3 < default min_bins=4
        b2_h6 = b2_h3**1.5
        method, curvature, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert method == "linear"
        assert curvature == 1.0
        assert "note" in diag

    def test_ill_conditioned_design_returns_linear(self):
        b2_h3 = np.array([1.0, 1.0 + 1e-12, 1.0 + 2e-12, 1.0 + 3e-12])
        b2_h6 = np.array([2.0, 2.0 + 1e-12, 2.0 + 2e-12, 2.0 + 3e-12])
        method, curvature, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert method == "linear"
        assert curvature == 1.0
        assert diag["note"] == "ill_conditioned_design"

    def test_weights_influence_fit(self):
        """Heavier weight on convex bins shifts alpha up."""
        b2_h3 = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        # Mix of linear and convex points
        b2_h6 = np.array([1.0, 2.0, 4.0, 12.0, 40.0])
        # Uniform weights
        _, _, diag_uniform = fit_h3_extrapolation_curve(b2_h3, b2_h6, weights=np.ones(5))
        # Heavy weight on the convex tail
        heavy_weights = np.array([1.0, 1.0, 1.0, 100.0, 100.0])
        _, _, diag_heavy = fit_h3_extrapolation_curve(b2_h3, b2_h6, weights=heavy_weights)
        assert diag_heavy["alpha"] > diag_uniform["alpha"]

    def test_diagnostics_dict_keys(self):
        """Returned diagnostics dict has alpha, se, r_squared, n_bins."""
        b2_h3 = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        b2_h6 = b2_h3 * 2.0
        _, _, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6)
        assert "alpha" in diag
        assert "se" in diag
        assert "r_squared" in diag
        assert "n_bins" in diag
        assert "weighting_scheme" in diag

    def test_variance_aware_weighting_scheme_when_weights_provided(self):
        """Providing bin exposures should activate inverse-variance proxy weighting."""
        b2_h3 = np.array([0.4, 0.8, 1.2, 1.6, 2.0, 2.4])
        b2_h6 = b2_h3 * 1.8
        weights = np.array([20, 50, 100, 200, 400, 800], dtype=float)
        _, _, diag = fit_h3_extrapolation_curve(b2_h3, b2_h6, weights=weights)
        assert diag["weighting_scheme"] == "inverse_variance_proxy"


# =============================================================================
# calculate_stress_factor Edge Case Tests
# =============================================================================


class TestCalculateStressFactorEdgeCases:
    """Additional edge case tests for calculate_stress_factor."""

    def test_all_zero_denominators(self):
        """All-zero denominators in both overall and worst populations."""
        df = pd.DataFrame(
            {
                "status_name": ["booked"] * 20,
                "risk_score_rf": np.linspace(0, 1, 20),
                "todu_30ever_h6": np.ones(20) * 10,
                "todu_amt_pile_h6": np.zeros(20),
            }
        )

        stress = calculate_stress_factor(df)

        # With zero denominators, the function should fall back gracefully
        assert isinstance(stress, float)
        assert np.isfinite(stress)

    def test_empty_dataframe_after_filtering(self):
        """DataFrame with no matching status should return neutral 1.0."""
        df = pd.DataFrame(
            {
                "status_name": ["rejected"] * 10 + ["cancelled"] * 5,
                "risk_score_rf": np.random.rand(15),
                "todu_30ever_h6": np.random.rand(15) * 10,
                "todu_amt_pile_h6": np.random.rand(15) * 1000 + 100,
            }
        )

        stress = calculate_stress_factor(df, target_status="booked")

        # Returns neutral 1.0 (not 0.0) to avoid zeroing out all risk predictions
        assert stress == 1.0


# =============================================================================
# Supersegment resolution helpers
# =============================================================================


class TestResolveModellingSuperSegment:
    def test_legacy_supersegment_field(self):
        assert resolve_modelling_supersegment({"supersegment": "total"}) == "total"

    def test_modelling_supersegment_overrides_legacy(self):
        cfg = {"supersegment": "total", "modelling_supersegment": "model_group"}
        assert resolve_modelling_supersegment(cfg) == "model_group"

    def test_no_field_returns_none(self):
        assert resolve_modelling_supersegment({}) is None

    def test_reporting_does_not_affect_modelling(self):
        cfg = {"reporting_supersegment": "report_group"}
        assert resolve_modelling_supersegment(cfg) is None


class TestResolveReportingSuperSegment:
    def test_legacy_supersegment_field(self):
        assert resolve_reporting_supersegment({"supersegment": "total"}) == "total"

    def test_reporting_supersegment_overrides_legacy(self):
        cfg = {"supersegment": "total", "reporting_supersegment": "report_group"}
        assert resolve_reporting_supersegment(cfg) == "report_group"

    def test_no_field_returns_none(self):
        assert resolve_reporting_supersegment({}) is None

    def test_modelling_does_not_affect_reporting(self):
        cfg = {"modelling_supersegment": "model_group"}
        assert resolve_reporting_supersegment(cfg) is None

    def test_both_set_independently(self):
        cfg = {"modelling_supersegment": "model_group", "reporting_supersegment": "report_group"}
        assert resolve_modelling_supersegment(cfg) == "model_group"
        assert resolve_reporting_supersegment(cfg) == "report_group"
