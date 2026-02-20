import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import PreprocessingSettings
from src.constants import RejectReason, StatusName
from src.preprocess_improved import (
    apply_binning_transformations,
    complete_preprocessing_pipeline,
    filter_by_date,
    preprocess_data,
    update_oa_amt_h0,
    update_status_and_reject_reason,
)


@pytest.fixture
def sample_data():
    """Create a sample dataframe for testing."""
    records = 100
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(np.random.choice(pd.date_range("2024-01-01", "2025-01-01"), records)),
            "status_name": np.random.choice(["booked", "rejected", "cancelled"], records),
            "risk_score_rf": np.random.uniform(0, 100, records),
            "se_decision_id": np.random.choice(["ok", "ko"], records),
            "reject_reason": np.random.choice(["08-other", "09-score", None], records),
            "score_rf": np.random.uniform(300, 500, records),
            "segment_cut_off": "test_segment",
            "early_bad": np.random.choice([0, 1], records),
            "acct_booked_h0": np.random.randint(0, 2, records),
            "oa_amt": np.random.uniform(1000, 50000, records),
            "todu_30ever_h6": np.random.randint(0, 10, records),
            "todu_amt_pile_h6": np.random.uniform(100, 1000, records),
            "oa_amt_h0": np.random.uniform(1000, 50000, records),
            "fuera_norma": "n",
            "fraud_flag": "n",
            "nature_holder": "physical",
            "m_ct_direct_sc_nov23": np.random.choice(["y", "n"], records),
        }
    )
    return df


@pytest.fixture
def config():
    """Create a sample configuration."""
    return PreprocessingSettings(
        keep_vars=["mis_date", "status_name", "risk_score_rf", "score_rf", "reject_reason"],
        indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        segment_filter="test_segment",
        octroi_bins=[-np.inf, 350, 400, 450, np.inf],
        efx_bins=[-np.inf, 20, 50, 80, np.inf],
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-12-31",
        variables=["sc_octroi_new_clus", "new_efx_clus"],
        score_measures=["m_ct_direct_sc_nov23"],
        log_level="WARNING",
    )


# =============================================================================
# Configuration Validation Tests
# =============================================================================


_REQUIRED_FIELDS = dict(
    keep_vars=["a"],
    indicators=["b"],
    octroi_bins=[1.0, 2.0],
    efx_bins=[1.0, 2.0],
    date_ini_book_obs="2024-01-01",
    date_fin_book_obs="2024-12-31",
    variables=["v0", "v1"],
)


def test_config_validation_empty_keep_vars():
    """Test that empty keep_vars raises error."""
    with pytest.raises(ValidationError, match="keep_vars"):
        PreprocessingSettings(**{**_REQUIRED_FIELDS, "keep_vars": []})


def test_config_validation_empty_indicators():
    """Test that empty indicators raises error."""
    with pytest.raises(ValidationError, match="indicators"):
        PreprocessingSettings(**{**_REQUIRED_FIELDS, "indicators": []})


def test_config_validation_invalid_bins():
    """Test that bins with less than 2 values raise error."""
    with pytest.raises(ValidationError, match="octroi_bins"):
        PreprocessingSettings(**{**_REQUIRED_FIELDS, "octroi_bins": [1.0]})

    with pytest.raises(ValidationError, match="efx_bins"):
        PreprocessingSettings(**{**_REQUIRED_FIELDS, "efx_bins": [1.0]})


def test_config_validation_valid():
    """Test that valid config passes validation."""
    PreprocessingSettings(**{**_REQUIRED_FIELDS, "octroi_bins": [1.0, 2.0, 3.0], "efx_bins": [1.0, 2.0, 3.0]})


# =============================================================================
# apply_binning_transformations Tests
# =============================================================================


@pytest.fixture
def binning_data():
    """Create data specifically for binning tests."""
    return pd.DataFrame(
        {
            "score_rf": [320, 360, 380, 410, 440],
            "risk_score_rf": [10, 30, 55, 75, 95],
        }
    )


def test_apply_binning_basic(binning_data):
    """Test basic binning transformation."""
    octroi_bins = [-np.inf, 350, 400, 450, np.inf]
    efx_bins = [-np.inf, 25, 50, 75, np.inf]

    result = apply_binning_transformations(binning_data, octroi_bins, efx_bins)

    assert "sc_octroi_new_clus" in result.columns
    assert "new_efx_clus" in result.columns

    # Check bins are 1-indexed
    assert result["sc_octroi_new_clus"].min() >= 1
    assert result["new_efx_clus"].min() >= 1


def test_apply_binning_preserves_data(binning_data):
    """Test that binning preserves original data."""
    octroi_bins = [-np.inf, 350, 400, 450, np.inf]
    efx_bins = [-np.inf, 25, 50, 75, np.inf]

    result = apply_binning_transformations(binning_data, octroi_bins, efx_bins)

    # Original columns should be preserved
    assert "score_rf" in result.columns
    assert "risk_score_rf" in result.columns
    pd.testing.assert_series_equal(result["score_rf"], binning_data["score_rf"])


def test_apply_binning_handles_edge_values():
    """Test binning handles edge values correctly."""
    data = pd.DataFrame(
        {
            "score_rf": [350, 400],  # Exact bin edges
            "risk_score_rf": [25, 50],
        }
    )
    octroi_bins = [-np.inf, 350, 400, 450, np.inf]
    efx_bins = [-np.inf, 25, 50, 75, np.inf]

    result = apply_binning_transformations(data, octroi_bins, efx_bins)

    # Should not raise and should produce valid bins
    assert not result["sc_octroi_new_clus"].isna().any()
    assert not result["new_efx_clus"].isna().any()


def test_apply_binning_missing_columns():
    """Test binning raises error for missing columns."""
    data = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        apply_binning_transformations(data, [0, 1], [0, 1])


# =============================================================================
# update_oa_amt_h0 Tests
# =============================================================================


def test_update_oa_amt_h0_basic():
    """Test oa_amt_h0 update for non-booked records."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "rejected", "cancelled"],
            "oa_amt": [1000, 2000, 3000],
            "oa_amt_h0": [1000, 0, 0],
        }
    )

    result = update_oa_amt_h0(data)

    # Booked should remain unchanged
    assert result.loc[0, "oa_amt_h0"] == 1000

    # Non-booked should be updated to oa_amt
    assert result.loc[1, "oa_amt_h0"] == 2000
    assert result.loc[2, "oa_amt_h0"] == 3000


def test_update_oa_amt_h0_preserves_booked():
    """Test that booked records are not modified."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "oa_amt": [5000, 6000],
            "oa_amt_h0": [1000, 2000],
        }
    )

    result = update_oa_amt_h0(data)

    # Booked values should be unchanged
    assert result.loc[0, "oa_amt_h0"] == 1000
    assert result.loc[1, "oa_amt_h0"] == 2000


def test_update_oa_amt_h0_missing_columns():
    """Test error handling for missing columns."""
    data = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        update_oa_amt_h0(data)


# =============================================================================
# filter_by_date Tests
# =============================================================================


def test_filter_by_date_basic():
    """Test basic date filtering."""
    data = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2024-01-01", "2024-06-15", "2024-12-31", "2025-01-01"]),
            "value": [1, 2, 3, 4],
        }
    )

    result = filter_by_date(data, "date_col", "2024-01-01", "2024-12-31")

    assert len(result) == 3
    assert result["value"].tolist() == [1, 2, 3]


def test_filter_by_date_inclusive():
    """Test that date filtering is inclusive on both ends."""
    data = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2024-01-01", "2024-12-31"]),
            "value": [1, 2],
        }
    )

    result = filter_by_date(data, "date_col", "2024-01-01", "2024-12-31")

    assert len(result) == 2


def test_filter_by_date_string_conversion():
    """Test that string dates in data are converted."""
    data = pd.DataFrame(
        {
            "date_col": ["2024-01-01", "2024-06-15", "2024-12-31"],
            "value": [1, 2, 3],
        }
    )

    result = filter_by_date(data, "date_col", "2024-01-01", "2024-06-30")

    assert len(result) == 2


def test_filter_by_date_invalid_range():
    """Test error for invalid date range (start > end)."""
    data = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2024-01-01"]),
            "value": [1],
        }
    )

    with pytest.raises(ValueError, match="start_date.*must be <= end_date"):
        filter_by_date(data, "date_col", "2024-12-31", "2024-01-01")


def test_filter_by_date_missing_column():
    """Test error for missing date column."""
    data = pd.DataFrame({"other_col": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing required columns"):
        filter_by_date(data, "date_col", "2024-01-01", "2024-12-31")


def test_filter_by_date_empty_result():
    """Test filtering that results in empty DataFrame."""
    data = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2023-01-01", "2023-06-15"]),
            "value": [1, 2],
        }
    )

    result = filter_by_date(data, "date_col", "2024-01-01", "2024-12-31")

    assert len(result) == 0


# =============================================================================
# update_status_and_reject_reason Tests
# =============================================================================


def test_update_status_basic():
    """Test status update based on measures."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked", "rejected"],
            "reject_reason": [None, None, "08-other"],
            "m_ct_direct_test": ["y", "n", "n"],
        }
    )

    result = update_status_and_reject_reason(data)

    # First row should be updated (has 'y' in measure)
    assert result.loc[0, "status_name"] == StatusName.REJECTED.value
    assert result.loc[0, "reject_reason"] == RejectReason.OTHER.value

    # Second row should remain unchanged
    assert result.loc[1, "status_name"] == "booked"

    # Third row should remain unchanged
    assert result.loc[2, "status_name"] == "rejected"


def test_update_status_with_score_measures():
    """Test status update with score measures."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "reject_reason": [None, None],
            "m_ct_direct_test": ["y", "y"],
            "score_measure": ["y", "n"],
        }
    )

    result = update_status_and_reject_reason(data, score_measures=["score_measure"])

    # First row has 'y' in score measure
    assert result.loc[0, "reject_reason"] == RejectReason.SCORE.value

    # Second row has 'n' in score measure, should be OTHER
    assert result.loc[1, "reject_reason"] == RejectReason.OTHER.value


def test_update_status_no_measures():
    """Test status update when no m_ct_direct columns exist."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "reject_reason": [None, None],
            "other_col": [1, 2],
        }
    )

    result = update_status_and_reject_reason(data)

    # Should return unchanged since no m_ct_direct columns
    assert result.loc[0, "status_name"] == "booked"
    assert result.loc[1, "status_name"] == "booked"


# =============================================================================
# Edge Cases Tests
# =============================================================================


def test_preprocess_empty_dataframe(config):
    """Test handling of empty DataFrame."""
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        preprocess_data(empty_df, config.keep_vars, config.indicators, config.segment_filter)


def test_binning_with_nan_values_raises_when_above_threshold():
    """Test binning raises ValueError when NaN percentage exceeds 1%."""
    data = pd.DataFrame(
        {
            "score_rf": [320, np.nan, 380],
            "risk_score_rf": [10, 30, np.nan],
        }
    )
    octroi_bins = [-np.inf, 350, 400, np.inf]
    efx_bins = [-np.inf, 25, 50, np.inf]

    with pytest.raises(ValueError, match="exceeds 1% threshold"):
        apply_binning_transformations(data, octroi_bins, efx_bins)


def test_filter_by_date_with_nat():
    """Test date filtering with NaT values."""
    data = pd.DataFrame(
        {
            "date_col": pd.to_datetime(["2024-01-01", pd.NaT, "2024-12-31"]),
            "value": [1, 2, 3],
        }
    )

    result = filter_by_date(data, "date_col", "2024-01-01", "2024-12-31")

    # NaT should be excluded
    assert len(result) == 2


# =============================================================================
# Integration Tests
# =============================================================================


def test_preprocess_data_filtering(sample_data, config):
    """Test that data is filtered correctly."""
    sample_data.loc[0, "fuera_norma"] = "y"

    processed = preprocess_data(sample_data, config.keep_vars, config.indicators, config.segment_filter)

    assert len(processed) < len(sample_data)
    assert "risk_score_rf" in processed.columns


def test_complete_pipeline(sample_data, config):
    """Test the complete pipeline execution."""
    data_clean, data_booked, data_demand = complete_preprocessing_pipeline(sample_data, config)

    assert not data_clean.empty
    assert "sc_octroi_new_clus" in data_clean.columns
    assert "new_efx_clus" in data_clean.columns
    assert "status_name" in data_clean.columns

    # Check date filtering on booked
    if not data_booked.empty:
        assert data_booked["mis_date"].min() >= pd.to_datetime(config.date_ini_book_obs)
        assert data_booked["mis_date"].max() <= pd.to_datetime(config.date_fin_book_obs)


def test_complete_pipeline_returns_three_dataframes(sample_data, config):
    """Test that pipeline returns exactly 3 DataFrames."""
    result = complete_preprocessing_pipeline(sample_data, config)

    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(df, pd.DataFrame) for df in result)
