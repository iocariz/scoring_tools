"""Tests for the out-of-time backtest core (M4, src/backtest.py). Synthetic data only."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.backtest import (
    BacktestResult,
    _acceptance_rate,
    _realized_metrics,
    apply_policy,
    derive_holdout_window,
    write_backtest_report,
    write_consolidated_report,
)
from src.utils import calculate_b2_ever_h6


def _settings(date_fin="2025-05-01"):
    # Only the fields backtest functions read.
    return SimpleNamespace(
        segment_filter="seg_a",
        variables=["a", "b"],
        multiplier=7.0,
        date_ini_book_obs="2024-06-01",
        date_fin_book_obs=date_fin,
        bins={},
    )


# ------------------------------- apply_policy -------------------------------


def test_apply_policy_membership_and_fail_closed():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 10.0, 9.0]})
    acc = apply_policy(df, ["a", "b"], {(1.0, 10.0), (3.0, 9.0)})
    assert list(acc) == [True, False, True]


def test_apply_policy_empty():
    df = pd.DataFrame({"a": [], "b": []})
    assert apply_policy(df, ["a", "b"], {(1.0, 1.0)}).empty


def test_apply_policy_nan_rejected():
    df = pd.DataFrame({"a": [1.0, None], "b": [10.0, 10.0]})
    acc = apply_policy(df, ["a", "b"], {(1.0, 10.0)})
    assert list(acc) == [True, False]


# ----------------------------- realized metrics -----------------------------


def test_realized_metrics_match_formula():
    booked = pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0],
            "b": [10.0, 10.0, 10.0],
            "oa_amt_h0": [100.0, 200.0, 500.0],
            "todu_30ever_h6": [1.0, 2.0, 9.0],
            "todu_amt_pile_h6": [100.0, 100.0, 100.0],
        }
    )
    accepted = apply_policy(booked, ["a", "b"], {(1.0, 10.0)})  # first two rows
    m = _realized_metrics(booked, accepted, multiplier=7.0)
    assert m["production"] == 300.0
    assert m["n_accepted_booked"] == 2
    expected = float(calculate_b2_ever_h6(3.0, 200.0, multiplier=7.0, as_percentage=True))  # 7*3/200*100 = 10.5
    assert m["risk"] == pytest.approx(expected)
    assert m["risk"] == pytest.approx(10.5)


def test_realized_metrics_zero_denominator_returns_none():
    booked = pd.DataFrame(
        {"a": [1.0], "b": [1.0], "oa_amt_h0": [0.0], "todu_30ever_h6": [0.0], "todu_amt_pile_h6": [0.0]}
    )
    m = _realized_metrics(booked, apply_policy(booked, ["a", "b"], {(1.0, 1.0)}), multiplier=7.0)
    assert m["risk"] is None


def test_acceptance_rate():
    demand = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 1.0, 1.0]})
    acc = apply_policy(demand, ["a", "b"], {(1.0, 1.0), (2.0, 1.0)})  # 2 of 4
    assert _acceptance_rate(demand, acc) == pytest.approx(0.5)
    assert _acceptance_rate(pd.DataFrame({"a": [], "b": []}), pd.Series([], dtype=bool)) is None


# --------------------------- derive_holdout_window ---------------------------


def test_derive_holdout_window_picks_mature_post_training():
    # Data extends to 2026-01; training ends 2025-05; maturity 6mo → cutoff 2025-07.
    data = pd.DataFrame({"mis_date": pd.to_datetime(["2025-06-15", "2025-10-15", "2026-01-15"])})
    w = derive_holdout_window(data, _settings("2025-05-01"), maturity_months=6)
    assert w["sufficient"] is True
    assert w["start"] == pd.Timestamp("2025-05-01")
    assert w["end"] == pd.Timestamp("2025-07-15")  # 2026-01-15 minus 6 months
    assert w["reference_date"] == pd.Timestamp("2026-01-15")


def test_derive_holdout_window_insufficient_when_no_mature_post_training():
    # All data within ~6 months of training end → no mature post-training cohort.
    data = pd.DataFrame({"mis_date": pd.to_datetime(["2025-06-01", "2025-07-01"])})
    w = derive_holdout_window(data, _settings("2025-05-01"), maturity_months=6)
    assert w["sufficient"] is False  # cutoff 2025-01 is BEFORE training end 2025-05


# ------------------------------- reporting ----------------------------------


def test_write_reports_roundtrip(tmp_path):
    result = BacktestResult(
        segment="seg_a",
        sufficient=True,
        window={"start": pd.Timestamp("2025-05-01"), "end": pd.Timestamp("2025-07-01")},
        coverage={"n_booked_oot": 940, "pct_mature": 0.26},
        predicted={"risk": 1.389, "production": 5.1e7},
        in_sample={"risk": 0.97, "acceptance_rate": 0.456},
        out_of_time={"risk": 2.11, "acceptance_rate": 0.45, "production": 9.2e6},
        calibration=pd.DataFrame(
            {"a": [1.0], "b": [10.0], "predicted_risk_pct": [1.0], "oot_realized_risk_pct": [2.0], "n_booked_oot": [12]}
        ),
    )
    paths = write_backtest_report(result, tmp_path, suffix="_base")
    assert paths["aggregate"].exists() and paths["calibration"].exists()
    agg = pd.read_csv(paths["aggregate"]).iloc[0]
    assert agg["oot_minus_insample_pp"] == pytest.approx(2.11 - 0.97)
    assert agg["oot_minus_predicted_pp"] == pytest.approx(2.11 - 1.389)

    cons = write_consolidated_report([result], tmp_path, suffix="_base")
    assert cons["consolidated"].exists() and cons["summary"].exists()
    md = cons["summary"].read_text()
    assert "DRIFT" in md  # |2.11 - 0.97| = 1.14 pp > 0.5 threshold
    assert "RI+stress" in md  # the predicted-caveat note


def test_insufficient_result_reports_gracefully(tmp_path):
    result = BacktestResult(segment="seg_b", sufficient=False, message="no mature out-of-time cohort", window={})
    paths = write_backtest_report(result, tmp_path, suffix="_base")
    assert paths["aggregate"].exists()
    assert "calibration" not in paths  # no calibration when insufficient
    cons = write_consolidated_report([result], tmp_path, suffix="_base")
    assert "skipped" in cons["summary"].read_text()
