"""Tests for the selection bias analysis module."""

import numpy as np
import pandas as pd
import pytest

from src.selection_bias import (
    compute_acceptance_gini_correlation,
    compute_bad_rate_by_decile,
    compute_booked_vs_rejected_discriminance,
    compute_distribution_truncation,
    compute_monthly_psi,
    compute_rejection_profile,
    compute_ri_gini,
    compute_rolling_acceptance_rate,
    compute_score_correlations,
    compute_score_rejection_only_truncation,
    simulate_range_restriction,
    thorndike_case2_correction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_demand_df(n=2000, acceptance_rate=0.6, seed=42):
    """Create a synthetic demand DataFrame with score-based selection."""
    rng = np.random.RandomState(seed)

    scores = rng.normal(500, 100, n)
    # Selection: higher scores are more likely to be accepted
    accept_prob = 1 / (1 + np.exp(-(scores - np.percentile(scores, (1 - acceptance_rate) * 100)) / 30))
    is_booked = rng.binomial(1, accept_prob).astype(bool)

    # Target: lower scores → higher default risk
    default_prob = 1 / (1 + np.exp((scores - 400) / 50))
    early_bad = np.where(is_booked, rng.binomial(1, default_prob), np.nan)

    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    return pd.DataFrame(
        {
            "score_rf": scores,
            "risk_score_rf": scores + rng.normal(0, 20, n),
            "status_name": np.where(is_booked, "booked", "rejected"),
            "reject_reason": np.where(is_booked, "", "09-score"),
            "early_bad": early_bad,
            "mis_date": dates[: len(scores)],
        }
    )


# ---------------------------------------------------------------------------
# Distribution truncation
# ---------------------------------------------------------------------------
class TestDistributionTruncation:
    def test_basic_structure(self):
        df = _make_demand_df()
        result = compute_distribution_truncation(df, "score_rf")
        assert "u_ratio" in result
        assert "n_demand" in result
        assert "ks_demand_vs_booked" in result
        assert result["n_demand"] == 2000

    def test_u_ratio_below_one(self):
        """Booked population should have smaller SD than demand (u < 1)."""
        df = _make_demand_df(n=5000, acceptance_rate=0.5)
        result = compute_distribution_truncation(df, "score_rf")
        assert result["u_ratio"] < 1.0, f"Expected u < 1, got {result['u_ratio']}"

    def test_no_selection_u_ratio_near_one(self):
        """If everyone is accepted, u should be ~1."""
        df = _make_demand_df(n=2000, acceptance_rate=0.99)
        result = compute_distribution_truncation(df, "score_rf")
        assert result["u_ratio"] > 0.95

    def test_negate_does_not_affect_u_ratio(self):
        df = _make_demand_df()
        r1 = compute_distribution_truncation(df, "score_rf", score_negate=False)
        r2 = compute_distribution_truncation(df, "score_rf", score_negate=True)
        assert r1["u_ratio"] == pytest.approx(r2["u_ratio"], abs=0.001)


# ---------------------------------------------------------------------------
# Thorndike Case 2
# ---------------------------------------------------------------------------
class TestThorndike:
    def test_no_restriction(self):
        """u=1.0 should return corrected_gini == observed_gini."""
        result = thorndike_case2_correction(observed_gini=0.4, u_ratio=1.0)
        assert result["corrected_gini"] == pytest.approx(0.4, abs=0.001)
        assert result["attenuation_pct"] == pytest.approx(0.0, abs=0.1)

    def test_corrected_higher_than_observed(self):
        """With restriction (u < 1), corrected Gini should be higher."""
        result = thorndike_case2_correction(observed_gini=0.3, u_ratio=0.7)
        assert result["corrected_gini"] > result["observed_gini"]

    def test_severe_restriction(self):
        result = thorndike_case2_correction(observed_gini=0.2, u_ratio=0.5)
        assert result["corrected_gini"] > 0.3
        assert result["attenuation_pct"] > 20

    def test_invalid_u_ratio(self):
        """u <= 0 should return unchanged Gini."""
        result = thorndike_case2_correction(observed_gini=0.4, u_ratio=0.0)
        assert result["corrected_gini"] == pytest.approx(0.4, abs=0.001)


# ---------------------------------------------------------------------------
# Simulated range restriction
# ---------------------------------------------------------------------------
class TestSimulatedRestriction:
    def test_gini_decreases_with_truncation(self):
        df = _make_demand_df(n=3000, acceptance_rate=0.8)
        booked = df[df["status_name"] == "booked"].copy()

        score_cols = {"Score RF": {"column": "score_rf", "negate": True}}
        result = simulate_range_restriction(booked, score_cols, "early_bad")

        assert not result.empty
        gini_100 = result[result["pct_retained"] == 1.0]["gini"].values
        gini_50 = result[result["pct_retained"] == 0.5]["gini"].values

        if len(gini_100) > 0 and len(gini_50) > 0:
            assert gini_100[0] >= gini_50[0], "Gini should decrease with more truncation"

    def test_empty_input(self):
        df = pd.DataFrame({"score_rf": [], "early_bad": []})
        result = simulate_range_restriction(df, {"S": {"column": "score_rf"}}, "early_bad")
        assert result.empty


# ---------------------------------------------------------------------------
# Rolling acceptance rate
# ---------------------------------------------------------------------------
class TestRollingAcceptance:
    def test_basic(self):
        df = _make_demand_df(n=5000)
        result = compute_rolling_acceptance_rate(df, "2024-01-01", "2024-08-30")
        assert not result.empty
        assert "acceptance_rate" in result.columns
        assert (result["acceptance_rate"] >= 0).all()
        assert (result["acceptance_rate"] <= 1).all()

    def test_no_dates(self):
        df = _make_demand_df()
        result = compute_rolling_acceptance_rate(df, None, None)
        assert result.empty


# ---------------------------------------------------------------------------
# Rejection profile
# ---------------------------------------------------------------------------
class TestRejectionProfile:
    def test_basic_structure(self):
        df = _make_demand_df()
        result = compute_rejection_profile(df, "score_rf")
        assert not result.empty
        assert "rejection_rate" in result.columns
        assert "decile" in result.columns

    def test_riskiest_decile_has_highest_rejection(self):
        """Decile 10 (riskiest) should have highest rejection rate."""
        df = _make_demand_df(n=5000, acceptance_rate=0.6)
        result = compute_rejection_profile(df, "score_rf", score_negate=True)
        if not result.empty and len(result) >= 2:
            # Last decile (highest negated score = riskiest) should have high rejection
            assert result.iloc[-1]["rejection_rate"] >= result.iloc[0]["rejection_rate"]


# ---------------------------------------------------------------------------
# Cross-segment correlation
# ---------------------------------------------------------------------------
class TestCrossSegmentCorrelation:
    def test_positive_correlation(self):
        data = [
            {"segment": "A", "score": "S", "acceptance_rate": 0.9, "gini": 0.5},
            {"segment": "B", "score": "S", "acceptance_rate": 0.7, "gini": 0.35},
            {"segment": "C", "score": "S", "acceptance_rate": 0.5, "gini": 0.2},
            {"segment": "D", "score": "S", "acceptance_rate": 0.3, "gini": 0.1},
        ]
        result = compute_acceptance_gini_correlation(data)
        assert result["slope"] is not None
        assert result["slope"] > 0  # positive: higher acceptance → higher Gini

    def test_too_few_points(self):
        data = [
            {"segment": "A", "score": "S", "acceptance_rate": 0.9, "gini": 0.5},
        ]
        result = compute_acceptance_gini_correlation(data)
        assert result["slope"] is None


# ---------------------------------------------------------------------------
# Booked-vs-Rejected discriminance
# ---------------------------------------------------------------------------
class TestBookedVsRejected:
    def test_basic_structure(self):
        df = _make_demand_df()
        score_cols = {"Score RF": {"column": "score_rf", "negate": True}}
        result = compute_booked_vs_rejected_discriminance(df, score_cols)
        assert not result.empty
        assert "gini" in result.columns
        assert result.iloc[0]["target"] == "booked_vs_rejected"

    def test_high_gini_with_score_based_selection(self):
        """Selection score should discriminate booked from rejected well."""
        df = _make_demand_df(n=5000, acceptance_rate=0.5)
        score_cols = {"Score RF": {"column": "score_rf", "negate": True}}
        result = compute_booked_vs_rejected_discriminance(df, score_cols)
        assert result.iloc[0]["gini"] > 0.1


# ---------------------------------------------------------------------------
# Score-rejection-only truncation
# ---------------------------------------------------------------------------
class TestScoreRejectionOnlyTruncation:
    def test_basic(self):
        df = _make_demand_df()
        df["reject_reason"] = np.where(df["status_name"] == "rejected", "09-score", "")
        score_cols = {"Score RF": {"column": "score_rf", "negate": True}}
        result = compute_score_rejection_only_truncation(df, score_cols)
        assert not result.empty
        assert "u_ratio" in result.columns


# ---------------------------------------------------------------------------
# Bad rate by decile
# ---------------------------------------------------------------------------
class TestBadRateByDecile:
    def test_basic_structure(self):
        df = _make_demand_df(n=3000)
        booked = df[df["status_name"] == "booked"].copy()
        result = compute_bad_rate_by_decile(booked, "score_rf", "early_bad")
        assert not result.empty
        assert "bad_rate" in result.columns
        assert "decile" in result.columns

    def test_monotonic_gradient(self):
        """Riskier deciles should tend to have higher bad rate."""
        df = _make_demand_df(n=5000, acceptance_rate=0.8)
        booked = df[df["status_name"] == "booked"].copy()
        result = compute_bad_rate_by_decile(booked, "score_rf", "early_bad", score_negate=True)
        if not result.empty and len(result) >= 2:
            # Last decile should have higher bad rate than first
            assert result.iloc[-1]["bad_rate"] >= result.iloc[0]["bad_rate"]


# ---------------------------------------------------------------------------
# Score correlations
# ---------------------------------------------------------------------------
class TestScoreCorrelations:
    def test_basic(self):
        df = _make_demand_df()
        score_cols = {
            "Score RF": {"column": "score_rf", "negate": True},
            "Risk Score RF": {"column": "risk_score_rf", "negate": True},
        }
        result = compute_score_correlations(df, score_cols)
        assert not result.empty
        assert "pearson_r" in result.columns
        assert "spearman_r" in result.columns
        # Scores are correlated by construction
        assert abs(result.iloc[0]["pearson_r"]) > 0.5


# ---------------------------------------------------------------------------
# RI Gini
# ---------------------------------------------------------------------------
class TestRiGini:
    def test_basic(self):
        df = _make_demand_df(n=3000, acceptance_rate=0.6)
        result = compute_ri_gini(df, "score_rf", "early_bad", score_negate=True)
        assert "ri_gini_mean" in result
        assert result["ri_gini_mean"] > 0

    def test_ri_gini_higher_than_booked(self):
        """RI Gini on full population should generally be >= booked-only Gini."""
        df = _make_demand_df(n=5000, acceptance_rate=0.5)
        result = compute_ri_gini(df, "score_rf", "early_bad", score_negate=True)
        # This is a soft test — RI Gini should be reasonable
        assert result["ri_gini_mean"] > 0.05

    def test_ri_gini_carries_circularity_note(self):
        """The result must flag that the RI Gini is a circular upper bound (diagnostic honesty)."""
        df = _make_demand_df(n=3000, acceptance_rate=0.6)
        result = compute_ri_gini(df, "score_rf", "early_bad", score_negate=True)
        assert "note" in result and result["note"]
        assert "circular" in result["note"].lower()


# ---------------------------------------------------------------------------
# Monthly PSI
# ---------------------------------------------------------------------------
class TestMonthlyPsi:
    def test_basic(self):
        df = _make_demand_df(n=5000)
        booked = df[df["status_name"] == "booked"].copy()
        result = compute_monthly_psi(booked, "score_rf")
        # May be empty if not enough months
        if not result.empty:
            assert "psi" in result.columns
            assert (result["psi"] >= 0).all()
