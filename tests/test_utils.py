import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.utils import calculate_b2_ever_h6, DEFAULT_RISK_MULTIPLIER


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
        data = pd.DataFrame({
            'todu_30ever_h6': [10, 25, 50, 100],
            'todu_amt_pile_h6': [1000, 2000, 3000, 4000],
        })

        result = calculate_b2_ever_h6(
            data['todu_30ever_h6'],
            data['todu_amt_pile_h6']
        )

        # Expected: 7 * num / den
        expected = 7 * data['todu_30ever_h6'] / data['todu_amt_pile_h6']

        np.testing.assert_array_almost_equal(result, expected.round(2))

    def test_dataframe_column_assignment(self):
        """Test assigning result to DataFrame column."""
        df = pd.DataFrame({
            'todu_30ever_h6': [10, 25, 50],
            'todu_amt_pile_h6': [1000, 2000, 3000],
        })

        df['b2_ever_h6'] = calculate_b2_ever_h6(
            df['todu_30ever_h6'],
            df['todu_amt_pile_h6'],
            as_percentage=True
        )

        assert 'b2_ever_h6' in df.columns
        assert not df['b2_ever_h6'].isna().any()
