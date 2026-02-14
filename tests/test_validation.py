import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import PreprocessingSettings
from src.data_manager import DataValidationError, validate_data_columns, validate_data_not_empty


class TestPreprocessorSettings:
    """Test configuration validation using Pydantic Settings model."""

    @pytest.fixture
    def valid_config_dict(self):
        return {
            "keep_vars": ["var1", "var2"],
            "indicators": ["ind1"],
            "segment_filter": "test_segment",
            "octroi_bins": [0, 10, 100],
            "efx_bins": [0, 0.5, 1.0],
            "date_ini_book_obs": "2023-01-01",
            "date_fin_book_obs": "2023-12-31",
            "variables": ["v1", "v2"],
            "multiplier": 7.0,
            "z_threshold": 3.0,
            "date_ini_book_obs_mr": "2024-01-01",
            "date_fin_book_obs_mr": "2024-06-30",
        }

    def test_valid_config_passes(self, valid_config_dict):
        settings = PreprocessingSettings(**valid_config_dict)
        assert settings.segment_filter == "test_segment"
        assert len(settings.keep_vars) == 2

    def test_missing_required_key(self, valid_config_dict):
        del valid_config_dict["keep_vars"]
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "keep_vars" in str(excinfo.value)

    def test_empty_keep_vars(self, valid_config_dict):
        valid_config_dict["keep_vars"] = []
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "keep_vars" in str(excinfo.value)
        assert "must be a non-empty list" in str(excinfo.value)

    def test_variables_wrong_length(self, valid_config_dict):
        valid_config_dict["variables"] = ["v1"]
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "variables" in str(excinfo.value)
        assert "must contain exactly 2 elements" in str(excinfo.value)

    def test_bins_too_short(self, valid_config_dict):
        valid_config_dict["octroi_bins"] = [10.0]
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "octroi_bins" in str(excinfo.value)
        assert "must have at least 2 values" in str(excinfo.value)

    def test_negative_multiplier(self, valid_config_dict):
        valid_config_dict["multiplier"] = -1.0
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "multiplier" in str(excinfo.value)

    def test_invalid_date_format(self, valid_config_dict):
        valid_config_dict["date_ini_book_obs"] = "invalid-date"
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "date_ini_book_obs" in str(excinfo.value)

    def test_invalid_date_range(self, valid_config_dict):
        valid_config_dict["date_ini_book_obs"] = "2023-12-31"
        valid_config_dict["date_fin_book_obs"] = "2023-01-01"
        with pytest.raises(ValidationError) as excinfo:
            PreprocessingSettings(**valid_config_dict)
        assert "Invalid main observation period" in str(excinfo.value)


class TestDataValidation:
    """Test data validation functions."""

    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [10.0, 20.0, 30.0]})

    def test_all_columns_present(self, sample_data):
        assert validate_data_columns(sample_data, ["col1", "col3"]) == []

    def test_missing_columns(self, sample_data):
        with pytest.raises(DataValidationError) as excinfo:
            validate_data_columns(sample_data, ["col1", "missing_col"])
        assert "missing_col" in str(excinfo.value)

    def test_case_insensitive(self, sample_data):
        assert validate_data_columns(sample_data, ["COL1", "Col2"]) == []

    def test_validate_data_not_empty_success(self, sample_data):
        validate_data_not_empty(sample_data)

    def test_validate_data_not_empty_failure(self):
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError):
            validate_data_not_empty(empty_df)
