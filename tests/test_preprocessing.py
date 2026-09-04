import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import BinConfig, PreprocessingSettings
from src.constants import RejectReason, StatusName, SystemDecision
from src.preprocess_improved import (
    _apply_binning_from_config,
    _infer_direction_from_bins,
    _run_data_transformations,
    apply_binning_transformations,
    complete_preprocessing_pipeline,
    filter_by_date,
    learn_optimization_bins,
    learn_quantile_bins,
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


def test_z_threshold_zero_allowed():
    """#56: z_threshold=0 disables target winsorization and must be accepted
    (the validator was gt=0, which rejected the disable value)."""
    s = PreprocessingSettings(**{**_REQUIRED_FIELDS, "z_threshold": 0})
    assert s.z_threshold == 0


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


def test_filter_by_date_uses_explicit_month_first_parsing():
    """Ambiguous string dates should be parsed consistently with dayfirst=False."""
    data = pd.DataFrame(
        {
            "date_col": ["03/04/2024", "04/03/2024"],
            "value": [1, 2],
        }
    )

    result = filter_by_date(data, "date_col", "2024-03-01", "2024-03-31")

    assert result["value"].tolist() == [1]


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


def test_update_status_score_only_vs_dual_measure():
    """#53 part 2: only a SCORE-ONLY rejection is 09-score (repesca/swap-in-eligible).
    A row that also tripped a non-score direct measure (fraud/legal/policy KO) must
    stay 08-other — the non-score KO is absolute and a better score can't overturn
    it, so it must not be counted as swap-in production."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked", "booked"],
            "reject_reason": [None, None, None],
            "m_ct_direct_score": ["y", "y", "n"],  # the configured score gate
            "m_ct_direct_fraud": ["n", "y", "y"],  # a non-score direct measure
        }
    )

    result = update_status_and_reject_reason(data, score_measures=["m_ct_direct_score"])

    # row 0: score-only rejection → 09-score (swap-in-eligible)
    assert result.loc[0, "reject_reason"] == RejectReason.SCORE.value
    # row 1: score AND non-score → non-score wins → 08-other (NOT swap-in-eligible)
    assert result.loc[1, "reject_reason"] == RejectReason.OTHER.value
    # row 2: non-score only → 08-other
    assert result.loc[2, "reject_reason"] == RejectReason.OTHER.value
    # all three tripped a direct measure → all rejected
    assert (result["status_name"] == StatusName.REJECTED.value).all()


def test_update_status_all_score_measures_no_non_score():
    """When every direct measure IS a score measure, a 'y' still maps to 09-score
    (no non-score measure to override it)."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "reject_reason": [None, None],
            "m_ct_direct_score": ["y", "n"],
        }
    )
    result = update_status_and_reject_reason(data, score_measures=["m_ct_direct_score"])
    assert result.loc[0, "reject_reason"] == RejectReason.SCORE.value
    assert result.loc[1, "status_name"] == "booked"  # no measure tripped


def test_update_status_new_convention_auto_detects_score_by_m_ct_sc_prefix():
    """New dataset convention: normal measures start with 'm_ct_', score measures with
    'm_ct_sc'. With NO explicit score_measures, the m_ct_sc* columns are auto-detected as
    the score gate; other m_ct_* columns are non-score (and win a dual-measure conflict)."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked", "booked"],
            "reject_reason": [None, None, None],
            "m_ct_sc_ecom": ["y", "y", "n"],  # score measure (m_ct_sc prefix)
            "m_ct_fraud": ["n", "y", "y"],  # non-score measure (m_ct_ but not m_ct_sc)
        }
    )
    result = update_status_and_reject_reason(data)  # no explicit list -> auto-detect

    assert result.loc[0, "reject_reason"] == RejectReason.SCORE.value  # score-only -> 09-score
    assert result.loc[1, "reject_reason"] == RejectReason.OTHER.value  # dual -> non-score wins
    assert result.loc[2, "reject_reason"] == RejectReason.OTHER.value  # non-score only
    assert (result["status_name"] == StatusName.REJECTED.value).all()


def test_update_status_m_ct_prefix_detects_measures_without_direct():
    """Measures no longer require the legacy 'm_ct_direct' prefix — a plain 'm_ct_' column
    is recognized as a (non-score) measure."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "reject_reason": [None, None],
            "m_ct_norm": ["y", "n"],  # non-score measure, plain m_ct_ prefix (not m_ct_sc)
        }
    )
    result = update_status_and_reject_reason(data)
    assert result.loc[0, "status_name"] == StatusName.REJECTED.value
    assert result.loc[0, "reject_reason"] == RejectReason.OTHER.value  # non-score -> 08-other
    assert result.loc[1, "status_name"] == "booked"


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


def test_update_status_sets_se_decision_ko_on_measure():
    """A direct-measure rejection sets se_decision_id to KO (System Rejection Rate)."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked", "booked"],
            "reject_reason": [None, None, None],
            "se_decision_id": ["ok", "ok", "rv"],
            "m_ct_direct_test": ["y", "n", "y"],
        }
    )

    result = update_status_and_reject_reason(data)

    # Rows 0 & 2 hit a direct measure → se_decision_id becomes KO
    assert result.loc[0, "se_decision_id"] == SystemDecision.KO.value
    assert result.loc[2, "se_decision_id"] == SystemDecision.KO.value
    # Row 1 has no measure → unchanged
    assert result.loc[1, "se_decision_id"] == "ok"


def test_update_status_no_se_decision_id_column():
    """se_decision_id is optional: an absent column must not be created or error."""
    data = pd.DataFrame(
        {
            "status_name": ["booked", "booked"],
            "reject_reason": [None, None],
            "m_ct_direct_test": ["y", "n"],
        }
    )

    result = update_status_and_reject_reason(data)

    # Column must NOT be conjured (a half-NaN column would mislead the rejection-rate logic)
    assert "se_decision_id" not in result.columns
    assert result.loc[0, "status_name"] == StatusName.REJECTED.value


def test_run_data_transformations_updates_oa_amt_h0_after_status_relabel(config):
    """#53: a booked row flagged by a direct measure is relabeled REJECTED, and
    must then receive oa_amt_h0 = oa_amt (demanded). Requires the status relabel
    (Step 3) to run BEFORE the oa_amt_h0 update (Step 4) — the old order left the
    newly-rejected row with its booked (financed) H0, corrupting oa_amt_demand.
    """
    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "status_name": ["booked", "booked"],
            "risk_score_rf": [30.0, 60.0],
            "se_decision_id": ["ok", "ok"],
            "reject_reason": [None, None],
            "score_rf": [360.0, 420.0],
            "segment_cut_off": ["test_segment", "test_segment"],
            "early_bad": [0, 0],
            "acct_booked_h0": [1, 1],
            "oa_amt": [10000.0, 20000.0],
            "todu_30ever_h6": [1, 2],
            "todu_amt_pile_h6": [500.0, 600.0],
            "oa_amt_h0": [9000.0, 18000.0],  # financed H0 < demanded oa_amt
            "fuera_norma": ["n", "n"],
            "fraud_flag": ["n", "n"],
            "nature_holder": ["physical", "physical"],
            "m_ct_direct_sc_nov23": ["y", "n"],  # row 0 flagged, row 1 genuine booked
        }
    )

    result = _run_data_transformations(df, config)

    flagged = result[result["oa_amt"] == 10000.0].iloc[0]
    assert flagged["status_name"] == StatusName.REJECTED.value
    # demanded amount, NOT the 9000 financed H0 it carried while "booked"
    assert flagged["oa_amt_h0"] == 10000.0

    genuine = result[result["oa_amt"] == 20000.0].iloc[0]
    assert genuine["status_name"] == StatusName.BOOKED.value
    assert genuine["oa_amt_h0"] == 18000.0  # booked rows keep their financed H0


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


# audit #21: out-of-range handling with FINITE edges (unreachable on the standard ±inf config).
_FINITE_OCTROI = [0.0, 50.0, 100.0]  # 2 bins, no ±inf
_FINITE_EFX = [0.0, 25.0, 50.0]  # risk kept in-range


def _binning_frame(score_vals):
    n = len(score_vals)
    return pd.DataFrame({"score_rf": score_vals, "risk_score_rf": [10.0] * n})


def test_out_of_range_below_snapped_to_first_bin():
    # 199 in-range + 1 below-range (0.5% < 1%): the below row snaps to the first bin (1-indexed = 1), kept.
    result = apply_binning_transformations(_binning_frame([60.0] * 199 + [-10.0]), _FINITE_OCTROI, _FINITE_EFX)
    assert len(result) == 200  # snapped, not dropped
    assert not result["sc_octroi_new_clus"].isna().any()
    assert result["sc_octroi_new_clus"].iloc[-1] == 1  # below-range -> first bin


def test_out_of_range_above_snapped_to_last_bin():
    # 199 in-range + 1 above-range: snaps to the last bin (2 bins -> 1-indexed 2), kept.
    result = apply_binning_transformations(_binning_frame([60.0] * 199 + [200.0]), _FINITE_OCTROI, _FINITE_EFX)
    assert len(result) == 200
    assert result["sc_octroi_new_clus"].iloc[-1] == 2  # above-range -> last bin


def test_nan_source_dropped_separately_from_oob():
    # 199 in-range + 1 NaN-source: the NaN row is dropped (not snapped); OOD rows would be kept.
    result = apply_binning_transformations(_binning_frame([60.0] * 199 + [np.nan]), _FINITE_OCTROI, _FINITE_EFX)
    assert len(result) == 199  # NaN-source dropped
    assert not result["sc_octroi_new_clus"].isna().any()


def test_out_of_range_over_threshold_still_raises():
    # >1% out of range with finite edges -> hard fail (behavior preserved).
    with pytest.raises(ValueError, match="exceeds 1% threshold"):
        apply_binning_transformations(_binning_frame([60.0] * 90 + [-10.0] * 10), _FINITE_OCTROI, _FINITE_EFX)


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


def test_bin_edge_learning_uses_demand_population():
    """Edge learning must use the full demand population, not booked-only.

    Bins are applied to the demand population (booked + rejected + canceled)
    and rejects feed reject inference, so edges are learned on demand.  This
    is a regression for the prior bug where edges were learned on booked-only
    (a selected, risk-truncated subset), violating the equal-count guarantee
    on the graded population.
    """
    n_booked = 10
    n_rejected = 10

    # Booked and rejected values are intentionally separated so the demand
    # median (includes rejected) differs sharply from the booked-only median.
    booked_values = np.arange(0, n_booked, dtype=float)
    rejected_values = np.arange(100, 100 + n_rejected, dtype=float)

    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(["2024-06-01"] * (n_booked + n_rejected)),
            "fuera_norma": ["n"] * (n_booked + n_rejected),
            "fraud_flag": ["n"] * (n_booked + n_rejected),
            "nature_holder": ["physical"] * (n_booked + n_rejected),
            "segment_cut_off": ["test_segment"] * (n_booked + n_rejected),
            "status_name": [StatusName.BOOKED.value] * (n_booked + n_rejected),
            "reject_reason": [None] * (n_booked + n_rejected),
            "risk_score_rf": np.concatenate([booked_values, rejected_values]),
            "oa_amt": np.ones(n_booked + n_rejected, dtype=float) * 1000.0,
            "oa_amt_h0": np.ones(n_booked + n_rejected, dtype=float) * 1000.0,
            # Direct measures: rows with 'y' are relabeled as REJECTED
            "m_ct_direct_sc_nov23": ["n"] * n_booked + ["y"] * n_rejected,
        }
    )

    settings = PreprocessingSettings(
        keep_vars=[
            "mis_date",
            "status_name",
            "reject_reason",
            "risk_score_rf",
            "oa_amt",
            "oa_amt_h0",
            "fuera_norma",
            "fraud_flag",
            "nature_holder",
            "segment_cut_off",
        ],
        indicators=["oa_amt", "oa_amt_h0"],
        segment_filter="test_segment",
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-12-31",
        variables=["sc_octroi_new_clus"],
        bins={
            "sc_octroi_new_clus": BinConfig(
                source_col="risk_score_rf",
                output_col="sc_octroi_new_clus",
                bin_edges=[],
                max_bins=2,
                method="quantile",
            )
        },
        score_measures=["m_ct_direct_sc_nov23"],
        log_level="WARNING",
        # Legacy fields still required by schema but can be empty
        octroi_bins=[],
        efx_bins=[],
    )

    _run_data_transformations(df, settings)
    learned_edges = settings.bins["sc_octroi_new_clus"].bin_edges

    # Demand median (booked + rejected) — NOT the booked-only median.
    demand_median = pd.Series(np.concatenate([booked_values, rejected_values])).quantile(0.5)
    booked_only_median = pd.Series(booked_values).quantile(0.5)
    assert len(learned_edges) == 3
    assert np.isneginf(learned_edges[0])
    assert np.isposinf(learned_edges[-1])
    assert np.isclose(learned_edges[1], demand_median)
    assert not np.isclose(learned_edges[1], booked_only_median)


def test_bin_method_optimization_falls_back_to_quantile():
    """Deprecated method='optimization' must produce quantile edges (no leakage).

    The supervised optimization split is deprecated; the dispatcher should
    ignore it and learn unsupervised quantile edges on the demand population.
    """
    n = 40
    values = np.arange(0, n, dtype=float)

    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(["2024-06-01"] * n),
            "fuera_norma": ["n"] * n,
            "fraud_flag": ["n"] * n,
            "nature_holder": ["physical"] * n,
            "segment_cut_off": ["test_segment"] * n,
            "status_name": [StatusName.BOOKED.value] * n,
            "reject_reason": [None] * n,
            "risk_score_rf": values,
            "oa_amt": np.ones(n, dtype=float) * 1000.0,
            "oa_amt_h0": np.ones(n, dtype=float) * 1000.0,
            "m_ct_direct_sc_nov23": ["n"] * n,
        }
    )

    common = dict(
        keep_vars=[
            "mis_date",
            "status_name",
            "reject_reason",
            "risk_score_rf",
            "oa_amt",
            "oa_amt_h0",
            "fuera_norma",
            "fraud_flag",
            "nature_holder",
            "segment_cut_off",
        ],
        indicators=["oa_amt", "oa_amt_h0"],
        segment_filter="test_segment",
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-12-31",
        variables=["sc_octroi_new_clus"],
        score_measures=["m_ct_direct_sc_nov23"],
        log_level="WARNING",
        octroi_bins=[],
        efx_bins=[],
    )

    def _learn(method: str) -> list[float]:
        settings = PreprocessingSettings(
            bins={
                "sc_octroi_new_clus": BinConfig(
                    source_col="risk_score_rf",
                    output_col="sc_octroi_new_clus",
                    bin_edges=[],
                    max_bins=2,
                    method=method,
                )
            },
            **common,
        )
        _run_data_transformations(df.copy(), settings)
        return settings.bins["sc_octroi_new_clus"].bin_edges

    opt_edges = _learn("optimization")
    quantile_edges = _learn("quantile")

    # Fallback: optimization yields the same edges as quantile.
    assert opt_edges == quantile_edges
    assert np.isclose(opt_edges[1], pd.Series(values).quantile(0.5))


# =============================================================================
# learn_optimization_bins Tests
# =============================================================================


def test_learn_optimization_bins_basic():
    """Test that optimization bins finds a meaningful split."""
    rng = np.random.RandomState(42)
    n = 2000
    income = rng.uniform(1000, 10000, n)
    # Low-income group has higher risk rate
    risk = (income < 5000).astype(float)
    risk += rng.normal(0, 0.05, n)
    risk = np.clip(risk, 0, 1)
    production = rng.uniform(5000, 50000, n)

    df = pd.DataFrame(
        {
            "income_t1_m": income,
            "early_bad": risk,
            "oa_amt_h0": production,
        }
    )

    edges = learn_optimization_bins(df, source_col="income_t1_m", min_samples_leaf=100)

    assert edges[0] == -np.inf
    assert edges[-1] == np.inf
    assert len(edges) == 3  # [-inf, threshold, inf]
    # Threshold should be near 5000
    assert 3000 < edges[1] < 7000


def test_learn_optimization_bins_fallback_no_weight_col():
    """Test fallback when weight column is missing."""
    rng = np.random.RandomState(42)
    n = 2000
    income = rng.uniform(1000, 10000, n)
    risk = (income < 5000).astype(float)

    df = pd.DataFrame(
        {
            "income_t1_m": income,
            "early_bad": risk,
        }
    )

    edges = learn_optimization_bins(df, source_col="income_t1_m", min_samples_leaf=100)
    assert len(edges) == 3
    assert edges[0] == -np.inf
    assert edges[-1] == np.inf


def test_learn_optimization_bins_fallback_zero_weights():
    """Test fallback when weight column is all zeros."""
    rng = np.random.RandomState(42)
    n = 2000
    income = rng.uniform(1000, 10000, n)
    risk = (income < 5000).astype(float)

    df = pd.DataFrame(
        {
            "income_t1_m": income,
            "early_bad": risk,
            "oa_amt_h0": np.zeros(n),
        }
    )

    edges = learn_optimization_bins(df, source_col="income_t1_m", min_samples_leaf=100)
    assert len(edges) == 3


def test_learn_optimization_bins_too_few_records():
    """Test that too few records raises ValueError."""
    df = pd.DataFrame(
        {
            "income_t1_m": [1000, 2000],
            "early_bad": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="only 2 valid records"):
        learn_optimization_bins(df, source_col="income_t1_m", min_samples_leaf=500)


def test_learn_optimization_bins_missing_column():
    """Test that missing source column raises ValueError."""
    df = pd.DataFrame({"other_col": [1, 2, 3], "early_bad": [0, 1, 0]})

    with pytest.raises(ValueError, match="Missing required columns"):
        learn_optimization_bins(df, source_col="income_t1_m")


class TestInferDirectionFromBins:
    """Audit #16: direction = sign of the production-weighted rank correlation, always."""

    def test_clear_descending(self):
        # higher bin index = safer (risk falls) -> descending -> -1
        bins = [1, 2, 3, 4, 5, 6]
        risk = [5.0, 4.0, 3.0, 2.0, 1.5, 1.0]
        w = [1000, 2000, 3000, 2500, 1500, 800]
        direction, rho_w, p = _infer_direction_from_bins(bins, risk, w)
        assert direction == -1
        assert rho_w < 0
        assert p < 0.05  # strong monotone signal

    def test_clear_ascending(self):
        bins = [1, 2, 3, 4, 5, 6]
        risk = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
        w = [1000, 2000, 3000, 2500, 1500, 800]
        direction, rho_w, p = _infer_direction_from_bins(bins, risk, w)
        assert direction == 1
        assert rho_w > 0
        assert p < 0.05

    def test_noisy_descending_regression(self):
        """The headline fix: a descending-on-average but noisy profile that the OLD
        significance-gated rule would have forced to ascending (+1) is now correctly
        inferred descending (-1), because direction follows the empirical sign."""
        bins = [1, 2, 3, 4, 5]
        risk = [3.0, 3.2, 2.5, 2.8, 2.0]  # overall down, with inversions
        w = [500, 3000, 400, 3000, 600]

        direction, rho_w, p = _infer_direction_from_bins(bins, risk, w)
        assert direction == -1  # empirical sign (was forced +1 by the old default)
        assert rho_w < 0
        assert p > 0.10  # the (calibrated) permutation deems it weak — but that no longer flips the sign

    def test_flat_risk_defaults_ascending(self):
        bins = [1, 2, 3, 4, 5]
        risk = [2.0, 2.0, 2.0, 2.0, 2.0]
        w = [1000, 1000, 1000, 1000, 1000]
        direction, rho_w, p = _infer_direction_from_bins(bins, risk, w)
        assert direction == 1  # documented default — no monotone signal
        assert np.isnan(rho_w)
        assert p == 1.0

    def test_permutation_calibration_sanity(self):
        bins = [1, 2, 3, 4, 5, 6]
        w = [1000, 1500, 1200, 1300, 1100, 900]
        # strong monotone -> small p
        _, _, p_strong = _infer_direction_from_bins(bins, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], w)
        # near-random (no clear trend) -> larger p
        _, _, p_weak = _infer_direction_from_bins(bins, [3.0, 1.0, 4.0, 2.0, 5.0, 3.5], w)
        assert p_strong < 0.05
        assert p_weak > p_strong

    def test_deterministic(self):
        bins = [1, 2, 3, 4, 5]
        risk = [3.0, 3.2, 2.5, 2.8, 2.0]
        w = [500, 3000, 400, 3000, 600]
        a = _infer_direction_from_bins(bins, risk, w)
        b = _infer_direction_from_bins(bins, risk, w)
        assert a == b


# =============================================================================
# Binning honest-warnings (quantile tie-collapse + realized skew)
# =============================================================================


def _capture_warnings(fn):
    """Run fn() capturing loguru WARNING messages via a stdlib handler bridge."""
    import logging

    from src import preprocess_improved as pp

    recs = []

    class _Sink(logging.Handler):
        def emit(self, record):
            recs.append(record.getMessage())

    hid = pp.logger.add(_Sink(level=logging.WARNING), level="WARNING", format="{message}")
    try:
        out = fn()
    finally:
        pp.logger.remove(hid)
    return out, recs


def test_learn_quantile_bins_warns_on_tie_collapse():
    """A discrete/mass-pointed source whose quantiles tie yields fewer bins than max_bins —
    surfaced as a WARNING (was INFO-only), and the returned grid is genuinely coarser."""
    # 90% at value 1: both the 1/3 and 2/3 quantiles are 1 -> terciles collapse to 1 split.
    data = pd.DataFrame({"inc": [1.0] * 90 + [2.0] * 10})
    edges, warns = _capture_warnings(lambda: learn_quantile_bins(data, source_col="inc", max_bins=3))
    assert any("collapsed" in m for m in warns)
    assert len(edges) - 1 < 3  # fewer bins than requested max_bins=3


def test_apply_binning_warns_on_quantile_skew():
    """A mass point on a split edge + the right-closed cut makes the realized quantile bins
    materially unequal; that must be WARNED (value cutpoint kept, not rank-split)."""
    from src.config import BinConfig

    # 70 records at exactly the edge (2000) -> right-closed cut puts them ALL in the left bin.
    data = pd.DataFrame({"inc": [2000.0] * 70 + [3000.0] * 30})
    bc = BinConfig(
        source_col="inc",
        output_col="income_bin",
        max_bins=2,
        bin_edges=[float("-inf"), 2000.0, float("inf")],
        method="quantile",
    )
    out, warns = _capture_warnings(lambda: _apply_binning_from_config(data, {"income_bin": bc}))
    assert any("materially unequal" in m for m in warns)
    # value cutpoint is kept: everyone at 2000 shares bin 1 (1-indexed), 3000s in bin 2
    assert set(out["income_bin"].unique()) == {1, 2}


def test_apply_binning_no_skew_warning_when_balanced():
    """A quantile binning with balanced populations must NOT warn."""
    from src.config import BinConfig

    data = pd.DataFrame({"inc": list(range(100))})  # uniform -> ~50/50 at the median edge
    bc = BinConfig(
        source_col="inc",
        output_col="income_bin",
        max_bins=2,
        bin_edges=[float("-inf"), 49.5, float("inf")],
        method="quantile",
    )
    _, warns = _capture_warnings(lambda: _apply_binning_from_config(data, {"income_bin": bc}))
    assert not any("materially unequal" in m for m in warns)
