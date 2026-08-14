"""Tests for run_score_metrics — score-discriminance reporting.

Regression coverage for datasets that carry only a subset of the configured
scores (e.g. a 1-variable run with score_rf but no risk_score_rf), which used to
raise KeyError in dropna(subset=...).
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

import run_score_metrics as rsm


def _booked_df(n: int, *, with_risk_score: bool, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    data = {
        "status_name": ["booked"] * n,
        "mis_date": pd.to_datetime("2024-06-15") + pd.to_timedelta(rng.randint(0, 300, n), unit="D"),
        "early_bad": rng.randint(0, 2, n),
        "score_rf": rng.rand(n),
    }
    if with_risk_score:
        data["risk_score_rf"] = rng.rand(n)
    return pd.DataFrame(data)


class TestScoresPresent:
    def test_filters_to_present_score_columns(self):
        df = _booked_df(50, with_risk_score=False)
        score_cols, combined_cols = rsm._scores_present(df)
        assert list(score_cols.keys()) == ["Score RF"]  # risk_score_rf dropped
        assert combined_cols == {}  # Combined needs both columns

    def test_keeps_both_when_present(self):
        df = _booked_df(50, with_risk_score=True)
        score_cols, combined_cols = rsm._scores_present(df)
        assert set(score_cols.keys()) == {"Score RF", "Risk Score RF"}
        assert "Combined" in combined_cols


class TestComputeForPeriodSingleScore:
    def test_no_risk_score_does_not_raise(self):
        """Regression: a df without risk_score_rf must not KeyError in dropna(subset=...)."""
        df = _booked_df(200, with_risk_score=False)
        res = rsm._compute_for_period(df, "2024-06-01", "2025-05-01", "main", "segment", "known", "ss")
        assert res is not None
        # Only the present score is reported; no risk_score_rf, no Combined.
        assert res["discriminance_df"]["score"].tolist() == ["Score RF"]
        assert list(res["scores_dict"].keys()) == ["Score RF"]

    def test_both_scores_still_include_combined(self):
        df = _booked_df(300, with_risk_score=True, seed=0)
        res = rsm._compute_for_period(df, "2024-06-01", "2025-05-01", "main", "segment", "known", "ss")
        assert res is not None
        assert res["discriminance_df"]["score"].tolist() == ["Score RF", "Risk Score RF", "Combined"]


class TestPrepareScores:
    def test_skips_absent_score_column(self):
        df = _booked_df(50, with_risk_score=False)
        scores = rsm._prepare_scores(df, include_combined=True)
        assert list(scores.keys()) == ["Score RF"]


class TestDedupLoans:
    """#64: pooled supersegment/total metrics must not double-count loans that
    overlapping segment filters place in more than one member."""

    def test_dedup_removes_overlapping_loans(self):
        seg_a = pd.DataFrame({"authorization_id": [1, 2, 3], "score_rf": [10.0, 20.0, 30.0]})
        seg_b = pd.DataFrame({"authorization_id": [3, 4], "score_rf": [30.0, 40.0]})  # loan 3 overlaps
        combined = rsm._dedup_loans(pd.concat([seg_a, seg_b], ignore_index=True))
        assert sorted(combined["authorization_id"]) == [1, 2, 3, 4]  # loan 3 counted once, not twice

    def test_dedup_noop_without_id_column(self):
        df = pd.DataFrame({"score_rf": [1.0, 1.0, 2.0]})
        assert len(rsm._dedup_loans(df)) == 3  # no id → no dedup

    def test_dedup_noop_when_disjoint(self):
        df = pd.DataFrame({"authorization_id": [1, 2, 3], "score_rf": [1.0, 2.0, 3.0]})
        assert len(rsm._dedup_loans(df)) == 3
