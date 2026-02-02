import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    DEFAULT_RISK_MULTIPLIER,
    calculate_b2_ever_h6,
    create_fixed_cutoff_solution,
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
# create_fixed_cutoff_solution Tests
# =============================================================================


class TestCreateFixedCutoffSolution:
    """Tests for the create_fixed_cutoff_solution utility function."""

    def test_basic_fixed_cutoff(self):
        """Test creating a fixed cutoff solution."""
        fixed_cutoffs = {
            "sc_octroi_new_clus": [1.0, 2.0, 3.0, 4.0],
            "new_efx_clus": [2, 2, 3, 4],
        }
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        values_var0 = [1.0, 2.0, 3.0, 4.0]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        # Check structure
        assert "sol_fac" in result.columns
        assert result["sol_fac"].iloc[0] == 0
        assert len(result) == 1

        # Check cutoff values
        assert result[1.0].iloc[0] == 2
        assert result[2.0].iloc[0] == 2
        assert result[3.0].iloc[0] == 3
        assert result[4.0].iloc[0] == 4

    def test_fixed_cutoff_with_integers(self):
        """Test with integer bin values."""
        fixed_cutoffs = {
            "var0": [1, 2, 3],
            "var1": [5, 6, 7],
        }
        variables = ["var0", "var1"]
        values_var0 = [1, 2, 3]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 5
        assert result[2.0].iloc[0] == 6
        assert result[3.0].iloc[0] == 7

    def test_missing_variable_raises_error(self):
        """Test that missing variable raises ValueError."""
        fixed_cutoffs = {
            "sc_octroi_new_clus": [1.0, 2.0, 3.0],
            # missing new_efx_clus
        }
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="must contain both variables"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [2, 3],  # Wrong length
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Length mismatch"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

    def test_single_bin(self):
        """Test with single bin."""
        fixed_cutoffs = {
            "var0": [1.0],
            "var1": [5],
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 5

    def test_monotonicity_warning(self):
        """Test that non-monotonic cutoffs trigger warning (captured via loguru sink)."""
        from io import StringIO

        from loguru import logger

        # Add a temporary sink to capture logs
        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0, 4.0],
                "var1": [5, 3, 4, 6],  # Non-monotonic: 5 -> 3 decreases
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0, 4.0]

            result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

            # Should succeed but log warning
            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic cutoffs detected" in log_output
        finally:
            logger.remove(handler_id)

    def test_monotonicity_strict_raises_error(self):
        """Test that non-monotonic cutoffs raise error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [5, 3, 4],  # Non-monotonic
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Non-monotonic cutoffs"):
            create_fixed_cutoff_solution(
                fixed_cutoffs, variables, values_var0, strict_validation=True
            )

    def test_monotonicity_inverted(self):
        """Test monotonicity with inv_var1=True (cutoffs should decrease)."""
        from io import StringIO

        from loguru import logger

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            # For inv_var1=True, cutoffs should be non-increasing
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [5, 6, 7],  # Increasing - wrong for inv_var1
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]

            result = create_fixed_cutoff_solution(
                fixed_cutoffs, variables, values_var0, inv_var1=True
            )

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic cutoffs detected" in log_output
            assert "non-increasing" in log_output
        finally:
            logger.remove(handler_id)

    def test_monotonicity_inverted_valid(self):
        """Test valid monotonicity with inv_var1=True."""
        from io import StringIO

        from loguru import logger

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            # For inv_var1=True, cutoffs should be non-increasing (valid case)
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [7, 5, 3],  # Decreasing - correct for inv_var1
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]

            result = create_fixed_cutoff_solution(
                fixed_cutoffs, variables, values_var0, inv_var1=True
            )

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic" not in log_output
        finally:
            logger.remove(handler_id)

    def test_data_bounds_warning(self):
        """Test warning when cutoffs are outside data range."""
        from io import StringIO

        from loguru import logger

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [5, 6, 15],  # 15 is outside range [1, 10]
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]
            values_var1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            result = create_fixed_cutoff_solution(
                fixed_cutoffs, variables, values_var0, values_var1=values_var1
            )

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "outside data range" in log_output
            assert "[15]" in log_output
        finally:
            logger.remove(handler_id)

    def test_data_bounds_strict_raises_error(self):
        """Test that out-of-bounds cutoffs raise error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [5, 20],  # 20 is outside range
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0]
        values_var1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        with pytest.raises(ValueError, match="outside data range"):
            create_fixed_cutoff_solution(
                fixed_cutoffs,
                variables,
                values_var0,
                values_var1=values_var1,
                strict_validation=True,
            )

    def test_bin_mismatch_strict_raises_error(self):
        """Test that bin mismatch raises error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [5, 6, 7],
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 4.0]  # 4.0 instead of 3.0

        with pytest.raises(ValueError, match="don't exactly match"):
            create_fixed_cutoff_solution(
                fixed_cutoffs, variables, values_var0, strict_validation=True
            )

    def test_valid_monotonic_cutoffs(self):
        """Test that valid monotonic cutoffs don't trigger warnings."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0, 4.0],
            "var1": [2, 3, 3, 5],  # Non-decreasing (equal is OK)
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0, 4.0]

        # Should not raise or warn
        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 2
        assert result[2.0].iloc[0] == 3
        assert result[3.0].iloc[0] == 3
        assert result[4.0].iloc[0] == 5
