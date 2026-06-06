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


def _result(ins_ci, oot_ci, n_def, ins_risk=0.97, oot_risk=2.11):
    return BacktestResult(
        segment="seg_a",
        sufficient=True,
        window={"start": pd.Timestamp("2025-05-01"), "end": pd.Timestamp("2025-07-01")},
        coverage={"n_booked_oot": 940, "pct_mature": 0.26},
        predicted={"risk": 1.389, "production": 5.1e7},
        in_sample={"risk": ins_risk, "acceptance_rate": 0.456, "risk_ci_lower": ins_ci[0], "risk_ci_upper": ins_ci[1]},
        out_of_time={
            "risk": oot_risk,
            "acceptance_rate": 0.45,
            "production": 9.2e6,
            "n_defaults": n_def,
            "risk_ci_lower": oot_ci[0],
            "risk_ci_upper": oot_ci[1],
        },
        calibration=pd.DataFrame(
            {"a": [1.0], "b": [10.0], "predicted_risk_pct": [1.0], "oot_realized_risk_pct": [2.0], "n_booked_oot": [12]}
        ),
    )


class TestDriftFlag:
    def test_overlapping_cis_with_enough_defaults_is_ok(self):
        # cd-like: 12 defaults but OOT CI [0.73,3.94] overlaps in-sample [0.63,1.39] → within noise.
        assert _result((0.63, 1.39), (0.73, 3.94), 12).drift_flag() == "OK"

    def test_separated_cis_with_enough_defaults_is_drift(self):
        assert _result((0.6, 1.0), (1.5, 3.0), 20).drift_flag() == "DRIFT"

    def test_too_few_defaults_is_inconclusive(self):
        # ef-like: only 2 OOT defaults → inconclusive regardless of the point estimate.
        assert _result((0.27, 2.12), (0.0, 7.38), 2).drift_flag() == "INCONCLUSIVE"

    def test_insufficient_result_is_dash(self):
        assert BacktestResult(segment="s", sufficient=False).drift_flag() == "—"


def test_bootstrap_risk_ci_brackets_point_estimate():
    from src.backtest import _bootstrap_risk_ci

    # 100 loans, ~5% with a default; CI should bracket the point b2 and be a (lo<=hi) tuple.
    rng = pd.Series(range(100))
    acc = pd.DataFrame(
        {
            "todu_30ever_h6": [50.0 if i < 5 else 0.0 for i in rng],
            "todu_amt_pile_h6": [1000.0] * 100,
        }
    )
    lo, hi = _bootstrap_risk_ci(acc, multiplier=7.0, n_boot=500)
    assert lo is not None and hi is not None and lo <= hi
    assert _bootstrap_risk_ci(pd.DataFrame({"todu_30ever_h6": [], "todu_amt_pile_h6": []}), 7.0) == (None, None)


def test_write_reports_roundtrip(tmp_path):
    result = _result((0.63, 1.39), (0.73, 3.94), 12)  # cd-like → OK (within noise)
    paths = write_backtest_report(result, tmp_path, suffix="_base")
    assert paths["aggregate"].exists() and paths["calibration"].exists()
    agg = pd.read_csv(paths["aggregate"]).iloc[0]
    assert agg["oot_minus_insample_pp"] == pytest.approx(2.11 - 0.97)
    assert agg["oot_minus_predicted_pp"] == pytest.approx(2.11 - 1.389)
    assert agg["drift_flag"] == "OK"
    assert agg["n_defaults_oot"] == 12

    cons = write_consolidated_report([result], tmp_path, suffix="_base")
    assert cons["consolidated"].exists() and cons["summary"].exists()
    md = cons["summary"].read_text()
    assert "[OK]" in md  # overlapping CIs → not DRIFT
    assert "12 defaults" in md  # default count surfaced
    assert "RI+stress" in md  # the predicted-caveat note


def test_insufficient_result_reports_gracefully(tmp_path):
    result = BacktestResult(segment="seg_b", sufficient=False, message="no mature out-of-time cohort", window={})
    paths = write_backtest_report(result, tmp_path, suffix="_base")
    assert paths["aggregate"].exists()
    assert "calibration" not in paths  # no calibration when insufficient
    cons = write_consolidated_report([result], tmp_path, suffix="_base")
    assert "skipped" in cons["summary"].read_text()
