"""Coverage tests for src.data_manager.

Mostly validation helpers + the prepare-from-preloaded-data path. We mock
``load_data`` rather than supplying a SAS fixture.
"""

import pandas as pd
import pytest

from src import data_manager as dm


def make_settings(**overrides):
    from src.config import PreprocessingSettings

    values = {
        "keep_vars": ["mis_date"],
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        "segment_filter": "seg_a",
        "date_ini_book_obs": "2024-01-01",
        "date_fin_book_obs": "2024-12-31",
        "variables": ["sc_octroi_new_clus", "new_efx_clus"],
        "octroi_bins": [0.0, 1.0, 2.0],
        "efx_bins": [0.0, 1.0, 2.0],
    }
    values.update(overrides)
    return PreprocessingSettings(**values)


class TestValidateDataColumns:
    def test_all_present_returns_empty(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        assert dm.validate_data_columns(df, ["a", "b"]) == []

    def test_missing_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(dm.DataValidationError, match="Missing required columns"):
            dm.validate_data_columns(df, ["a", "z"])


class TestValidateDataNotEmpty:
    def test_non_empty_passes(self):
        dm.validate_data_not_empty(pd.DataFrame({"a": [1]}))

    def test_empty_raises(self):
        with pytest.raises(dm.DataValidationError, match="empty"):
            dm.validate_data_not_empty(pd.DataFrame())


class TestLoadAndPrepareData:
    def _minimal_data(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
                "oa_amt": [1000.0, 2000.0],
                "oa_amt_h0": [900.0, 1800.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
                "income_t1t2_m": [1500.0, 2500.0],
                "se_decision_id": ["ok", "ok"],
                "status_name": ["booked", "booked"],
                "reject_reason": ["", ""],
                "sc_octroi_new_clus": [1.0, 2.0],
                "new_efx_clus": [1.0, 2.0],
                "score_rf": [0.5, 0.6],
                "risk_score_rf": [0.4, 0.5],
                "authorization_id": [1, 2],
            }
        )

    def test_preloaded_data_short_circuits_load(self, monkeypatch):
        settings = make_settings()
        called = []

        def fake_load(_):
            called.append("load")
            return pd.DataFrame()

        monkeypatch.setattr(dm, "load_data", fake_load)
        # Avoid schema validation surprises by stubbing it
        monkeypatch.setattr("src.schema.validate_raw_data", lambda d, raise_on_error=True: None)

        data, out_settings = dm.load_and_prepare_data(settings, preloaded_data=self._minimal_data())
        assert called == []  # load_data was not invoked
        assert len(data) == 2
        assert out_settings is settings  # no H3 cols → settings unchanged

    def test_h3_columns_optional_warns_and_strips_settings(self, monkeypatch):
        settings = make_settings(
            keep_vars=["mis_date", "todu_30ever_h3"],
            indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "todu_amt_pile_h3"],
        )

        # Preloaded data does NOT include the H3 columns
        df = self._minimal_data()
        monkeypatch.setattr("src.schema.validate_raw_data", lambda d, raise_on_error=True: None)

        _, out_settings = dm.load_and_prepare_data(settings, preloaded_data=df)
        # Settings should have H3 columns stripped, no exception raised
        assert "todu_30ever_h3" not in out_settings.keep_vars
        assert "todu_amt_pile_h3" not in out_settings.indicators

    def test_load_path_invokes_load_and_standardizes_columns(self, monkeypatch):
        settings = make_settings()
        raw = self._minimal_data()
        # Inject a column with mixed-case and a space → should be standardized
        raw["MisCased_col"] = ["A B", "C D"]

        monkeypatch.setattr(dm, "load_data", lambda _: raw)
        monkeypatch.setattr("src.schema.validate_raw_data", lambda d, raise_on_error=True: None)

        data, _ = dm.load_and_prepare_data(settings, preloaded_data=None)
        # Columns lowercased + spaces replaced
        assert "miscased_col" in data.columns
        # Categorical values lowercased + spaces replaced
        assert data["miscased_col"].iloc[0] == "a_b"
