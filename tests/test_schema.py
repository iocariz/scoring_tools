import numpy as np
import pandas as pd
import pytest

from src.data_manager import DataValidationError
from src.schema import validate_raw_data


def _make_valid_df(n: int = 20) -> pd.DataFrame:
    """Create a minimal valid DataFrame that passes schema validation."""
    rng = np.random.RandomState(42)
    return pd.DataFrame(
        {
            "status_name": pd.Categorical(rng.choice(["booked", "rejected"], size=n)),
            "segment_cut_off": pd.Categorical(["seg_a"] * n),
            "mis_date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "oa_amt_h0": rng.uniform(1000, 50000, size=n),
            "todu_30ever_h6": np.where(rng.random(n) > 0.5, rng.uniform(0, 10, size=n), np.nan),
            "todu_amt_pile_h6": np.where(rng.random(n) > 0.5, rng.uniform(0, 1000, size=n), np.nan),
        }
    )


class TestRawDataSchema:
    """Tests for the Pandera raw data schema."""

    def test_valid_data_passes(self):
        df = _make_valid_df()
        result = validate_raw_data(df, raise_on_error=False)
        assert result[0] is True
        assert result[1] == []

    def test_valid_data_returns_dataframe(self):
        df = _make_valid_df()
        validated = validate_raw_data(df, raise_on_error=True)
        assert isinstance(validated, pd.DataFrame)
        assert len(validated) == len(df)

    def test_invalid_status_fails(self):
        df = _make_valid_df()
        df["status_name"] = pd.Categorical(["invalid_status"] * len(df))
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is False
        assert any("status_name" in e for e in errors)

    def test_invalid_status_raises(self):
        df = _make_valid_df()
        df["status_name"] = pd.Categorical(["invalid_status"] * len(df))
        with pytest.raises(DataValidationError, match="status_name"):
            validate_raw_data(df, raise_on_error=True)

    def test_negative_oa_amt_h0_fails(self):
        df = _make_valid_df()
        df.loc[0, "oa_amt_h0"] = -100.0
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is False
        assert any("oa_amt_h0" in e for e in errors)

    def test_negative_todu_30ever_fails(self):
        df = _make_valid_df()
        df["todu_30ever_h6"] = df["todu_30ever_h6"].astype("float64")
        df.loc[0, "todu_30ever_h6"] = -5.0
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is False
        assert any("todu_30ever_h6" in e for e in errors)

    def test_negative_todu_amt_pile_fails(self):
        df = _make_valid_df()
        df["todu_amt_pile_h6"] = df["todu_amt_pile_h6"].astype("float64")
        df.loc[0, "todu_amt_pile_h6"] = -1.0
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is False
        assert any("todu_amt_pile_h6" in e for e in errors)

    def test_nullable_risk_columns_ok(self):
        """Risk columns can be all NaN (e.g., demand-only data)."""
        df = _make_valid_df()
        df["todu_30ever_h6"] = np.nan
        df["todu_amt_pile_h6"] = np.nan
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is True

    def test_extra_columns_allowed(self):
        """Schema is non-strict: extra columns should not cause failure."""
        df = _make_valid_df()
        df["extra_col"] = 999
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is True

    def test_missing_optional_columns_ok(self):
        """Optional risk columns (required=False) can be absent entirely."""
        df = _make_valid_df()
        df = df.drop(columns=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"])
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is True

    def test_null_segment_cut_off_fails(self):
        df = _make_valid_df()
        df.loc[0, "segment_cut_off"] = None
        is_valid, errors = validate_raw_data(df, raise_on_error=False)
        assert is_valid is False
