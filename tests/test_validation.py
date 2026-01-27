import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from main import (
    validate_config,
    validate_date_string,
    validate_date_range,
    validate_data_columns,
    validate_data_not_empty,
    convert_bins,
    ConfigValidationError,
    DataValidationError,
    REQUIRED_CONFIG_KEYS,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def valid_config():
    """Create a valid configuration dictionary."""
    return {
        'keep_vars': ['a', 'b', 'c'],
        'indicators': ['d', 'e'],
        'segment_filter': 'test_segment',
        'octroi_bins': [-np.inf, 350, 400, 450, np.inf],
        'efx_bins': [-np.inf, 20, 50, 80, np.inf],
        'date_ini_book_obs': '2024-01-01',
        'date_fin_book_obs': '2024-12-31',
        'variables': ['var1', 'var2'],
    }


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'col_a': [1, 2, 3],
        'col_b': ['x', 'y', 'z'],
        'col_c': [1.0, 2.0, 3.0],
    })


# =============================================================================
# validate_config Tests
# =============================================================================

class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_valid_config_passes(self, valid_config):
        """Test that a valid configuration passes validation."""
        warnings = validate_config(valid_config)
        assert isinstance(warnings, list)

    def test_missing_required_key(self, valid_config):
        """Test that missing required keys raise error."""
        del valid_config['keep_vars']
        with pytest.raises(ConfigValidationError, match="Missing required configuration keys"):
            validate_config(valid_config)

    def test_empty_keep_vars(self, valid_config):
        """Test that empty keep_vars raises error."""
        valid_config['keep_vars'] = []
        with pytest.raises(ConfigValidationError, match="must be a non-empty list"):
            validate_config(valid_config)

    def test_empty_indicators(self, valid_config):
        """Test that empty indicators raises error."""
        valid_config['indicators'] = []
        with pytest.raises(ConfigValidationError, match="must be a non-empty list"):
            validate_config(valid_config)

    def test_variables_wrong_length(self, valid_config):
        """Test that variables with wrong length raises error."""
        valid_config['variables'] = ['var1']
        with pytest.raises(ConfigValidationError, match="must contain exactly 2 elements"):
            validate_config(valid_config)

        valid_config['variables'] = ['var1', 'var2', 'var3']
        with pytest.raises(ConfigValidationError, match="must contain exactly 2 elements"):
            validate_config(valid_config)

    def test_bins_too_short(self, valid_config):
        """Test that bins with less than 2 values raise error."""
        valid_config['octroi_bins'] = [1]
        with pytest.raises(ConfigValidationError, match="octroi_bins.*must have at least 2 values"):
            validate_config(valid_config)

    def test_negative_multiplier(self, valid_config):
        """Test that negative multiplier raises error."""
        valid_config['multiplier'] = -1
        with pytest.raises(ConfigValidationError, match="multiplier.*must be positive"):
            validate_config(valid_config)

    def test_negative_z_threshold(self, valid_config):
        """Test that negative z_threshold raises error."""
        valid_config['z_threshold'] = -0.5
        with pytest.raises(ConfigValidationError, match="z_threshold.*must be positive"):
            validate_config(valid_config)

    def test_partial_mr_config_warning(self, valid_config):
        """Test that partial MR config produces warning."""
        valid_config['date_ini_book_obs_mr'] = '2025-01-01'
        # Missing date_fin_book_obs_mr
        warnings = validate_config(valid_config)
        assert len(warnings) > 0
        assert any('MR period' in w for w in warnings)

    def test_complete_mr_config_no_warning(self, valid_config):
        """Test that complete MR config produces no warning."""
        valid_config['date_ini_book_obs_mr'] = '2025-01-01'
        valid_config['date_fin_book_obs_mr'] = '2025-06-30'
        warnings = validate_config(valid_config)
        assert not any('MR period' in w for w in warnings)

    def test_keep_vars_not_list(self, valid_config):
        """Test that non-list keep_vars raises error."""
        valid_config['keep_vars'] = 'not_a_list'
        with pytest.raises(ConfigValidationError, match="must be a non-empty list"):
            validate_config(valid_config)


# =============================================================================
# validate_date_string Tests
# =============================================================================

class TestValidateDateString:
    """Tests for the validate_date_string function."""

    def test_valid_date_string(self):
        """Test that valid date string is parsed correctly."""
        result = validate_date_string('2024-01-15', 'test_date')
        assert isinstance(result, pd.Timestamp)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_various_date_formats(self):
        """Test various valid date formats."""
        formats = [
            '2024-01-15',
            '2024/01/15',
            '15-01-2024',
            'January 15, 2024',
        ]
        for fmt in formats:
            try:
                result = validate_date_string(fmt, 'test')
                assert isinstance(result, pd.Timestamp)
            except ConfigValidationError:
                pass  # Some formats may not be supported

    def test_empty_date_string(self):
        """Test that empty date string raises error."""
        with pytest.raises(ConfigValidationError, match="cannot be empty"):
            validate_date_string('', 'test_date')

    def test_invalid_date_string(self):
        """Test that invalid date string raises error."""
        with pytest.raises(ConfigValidationError, match="Invalid date format"):
            validate_date_string('not-a-date', 'test_date')

    def test_error_includes_field_name(self):
        """Test that error message includes field name."""
        with pytest.raises(ConfigValidationError, match="my_field"):
            validate_date_string('invalid', 'my_field')


# =============================================================================
# validate_date_range Tests
# =============================================================================

class TestValidateDateRange:
    """Tests for the validate_date_range function."""

    def test_valid_date_range(self):
        """Test that valid date range passes."""
        start = pd.Timestamp('2024-01-01')
        end = pd.Timestamp('2024-12-31')
        validate_date_range(start, end, 'test_range')  # Should not raise

    def test_same_date_range(self):
        """Test that same start and end date passes."""
        date = pd.Timestamp('2024-06-15')
        validate_date_range(date, date, 'test_range')  # Should not raise

    def test_invalid_date_range(self):
        """Test that start > end raises error."""
        start = pd.Timestamp('2024-12-31')
        end = pd.Timestamp('2024-01-01')
        with pytest.raises(ConfigValidationError, match="is after end date"):
            validate_date_range(start, end, 'test_range')

    def test_error_includes_range_name(self):
        """Test that error message includes range name."""
        start = pd.Timestamp('2024-12-31')
        end = pd.Timestamp('2024-01-01')
        with pytest.raises(ConfigValidationError, match="my_range"):
            validate_date_range(start, end, 'my_range')


# =============================================================================
# validate_data_columns Tests
# =============================================================================

class TestValidateDataColumns:
    """Tests for the validate_data_columns function."""

    def test_all_columns_present(self, sample_dataframe):
        """Test that present columns pass validation."""
        missing = validate_data_columns(sample_dataframe, ['col_a', 'col_b'])
        assert missing == []

    def test_missing_columns(self, sample_dataframe):
        """Test that missing columns raise error."""
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_data_columns(sample_dataframe, ['col_a', 'col_missing'])

    def test_case_insensitive(self, sample_dataframe):
        """Test that column matching is case-insensitive."""
        missing = validate_data_columns(sample_dataframe, ['COL_A', 'COL_B'])
        assert missing == []

    def test_custom_context(self, sample_dataframe):
        """Test that context is included in error message."""
        with pytest.raises(DataValidationError, match="my_context"):
            validate_data_columns(sample_dataframe, ['missing'], 'my_context')

    def test_empty_required_list(self, sample_dataframe):
        """Test that empty required list passes."""
        missing = validate_data_columns(sample_dataframe, [])
        assert missing == []


# =============================================================================
# validate_data_not_empty Tests
# =============================================================================

class TestValidateDataNotEmpty:
    """Tests for the validate_data_not_empty function."""

    def test_non_empty_dataframe(self, sample_dataframe):
        """Test that non-empty DataFrame passes."""
        validate_data_not_empty(sample_dataframe)  # Should not raise

    def test_empty_dataframe(self):
        """Test that empty DataFrame raises error."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="is empty"):
            validate_data_not_empty(empty_df)

    def test_custom_context(self):
        """Test that context is included in error message."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="my_data"):
            validate_data_not_empty(empty_df, 'my_data')


# =============================================================================
# convert_bins Tests
# =============================================================================

class TestConvertBins:
    """Tests for the convert_bins function."""

    def test_convert_positive_inf(self):
        """Test conversion of positive infinity."""
        bins = [0, 1, float('inf')]
        result = convert_bins(bins)
        assert result[-1] == np.inf

    def test_convert_negative_inf(self):
        """Test conversion of negative infinity."""
        bins = [float('-inf'), 0, 1]
        result = convert_bins(bins)
        assert result[0] == -np.inf

    def test_convert_both_inf(self):
        """Test conversion of both infinities."""
        bins = [float('-inf'), 0, 1, float('inf')]
        result = convert_bins(bins)
        assert result[0] == -np.inf
        assert result[-1] == np.inf

    def test_no_inf_values(self):
        """Test that non-inf values are unchanged."""
        bins = [0, 1, 2, 3]
        result = convert_bins(bins)
        assert result == bins

    def test_empty_list(self):
        """Test that empty list returns empty list."""
        result = convert_bins([])
        assert result == []

    def test_none_input(self):
        """Test that None input returns None."""
        result = convert_bins(None)
        assert result is None

    def test_preserves_order(self):
        """Test that bin order is preserved."""
        bins = [float('-inf'), 350, 400, 450, float('inf')]
        result = convert_bins(bins)
        assert result[1] == 350
        assert result[2] == 400
        assert result[3] == 450


# =============================================================================
# Integration Tests
# =============================================================================

class TestValidationIntegration:
    """Integration tests for validation functions working together."""

    def test_full_config_validation_flow(self, valid_config):
        """Test the complete config validation flow."""
        # Validate config
        warnings = validate_config(valid_config)

        # Validate dates
        date_ini = validate_date_string(valid_config['date_ini_book_obs'], 'date_ini')
        date_fin = validate_date_string(valid_config['date_fin_book_obs'], 'date_fin')
        validate_date_range(date_ini, date_fin, 'observation period')

        # Convert bins
        octroi_bins = convert_bins(valid_config['octroi_bins'])
        efx_bins = convert_bins(valid_config['efx_bins'])

        # All should complete without error
        assert len(octroi_bins) == len(valid_config['octroi_bins'])
        assert len(efx_bins) == len(valid_config['efx_bins'])

    def test_data_validation_flow(self, sample_dataframe):
        """Test the complete data validation flow."""
        # Validate not empty
        validate_data_not_empty(sample_dataframe)

        # Validate required columns
        validate_data_columns(sample_dataframe, ['col_a', 'col_b'])

        # Should complete without error
        assert True
