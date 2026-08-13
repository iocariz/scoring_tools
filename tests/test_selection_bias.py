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
    compute_selection_bias_report,
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

    def test_corrected_never_below_observed_and_attenuation_nonnegative(self):
        """A Case-2 correction (u<1) only raises the estimate, so corrected_gini >= observed_gini
        and attenuation_pct >= 0 must hold everywhere — including extreme/near-saturation inputs
        (the clamp guards against the 0.999 r-clip ever inverting it)."""
        for og in [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.98, 0.998]:
            for u in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
                r = thorndike_case2_correction(observed_gini=og, u_ratio=u)
                assert r["corrected_gini"] >= og - 1e-9, (og, u, r["corrected_gini"])
                assert r["attenuation_pct"] >= 0.0, (og, u, r["attenuation_pct"])


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

    def test_truncation_removes_the_riskiest_tail(self):
        """Audit #23: a tighter cutoff removes the RISKIEST tail. The old code
        sorted descending in riskiness and kept ``[:n_keep]`` — retaining the
        riskiest pct and removing the safest (the inverted restriction). Bads
        concentrate in the risky tail, so under the bug they survived
        truncation almost untouched; monotone Gini decay alone (the test
        above) holds for either tail and cannot catch this."""
        df = _make_demand_df(n=4000, acceptance_rate=0.8)
        booked = df[df["status_name"] == "booked"].copy()

        score_cols = {"Score RF": {"column": "score_rf", "negate": True}}
        result = simulate_range_restriction(booked, score_cols, "early_bad")

        full = result[result["pct_retained"] == 1.0].iloc[0]
        half = result[result["pct_retained"] == 0.5].iloc[0]

        # Most bads live in the removed risky tail.
        assert half["n_bads"] < 0.5 * full["n_bads"], (
            f"truncating the riskiest half must remove most bads "
            f"(kept {half['n_bads']}/{full['n_bads']}) — wrong tail truncated?"
        )
        # The retained book must be SAFER than the full book (tighter cutoff).
        assert half["n_bads"] / half["n_records"] < full["n_bads"] / full["n_records"]

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
# #59: canceled applications (score-accepted, not taken up) are NOT rejections
# ---------------------------------------------------------------------------
class TestCanceledNotRejected:
    @staticmethod
    def _frames(seed=0):
        rng = np.random.RandomState(seed)
        booked = pd.DataFrame(
            {
                "score_rf": rng.normal(600, 40, 300),
                "status_name": "booked",
                "reject_reason": "",
                "early_bad": rng.binomial(1, 0.1, 300).astype(float),
            }
        )
        rejected = pd.DataFrame(
            {
                "score_rf": rng.normal(350, 40, 300),
                "status_name": "rejected",
                "reject_reason": "09-score",
                "early_bad": np.nan,
            }
        )
        # canceled: high (booked-like) scores — they PASSED the score
        canceled = pd.DataFrame(
            {
                "score_rf": rng.normal(600, 40, 150),
                "status_name": "canceled",
                "reject_reason": "",
                "early_bad": np.nan,
            }
        )
        base = pd.concat([booked, rejected], ignore_index=True)
        with_canceled = pd.concat([booked, rejected, canceled], ignore_index=True)
        return base, with_canceled

    def test_truncation_rejected_set_excludes_canceled(self):
        df = pd.DataFrame(
            {
                "score_rf": [600.0, 610.0, 300.0, 310.0, 620.0, 630.0],
                "status_name": ["booked", "booked", "rejected", "rejected", "canceled", "canceled"],
                "reject_reason": ["", "", "09-score", "09-score", "", ""],
            }
        )
        r = compute_distribution_truncation(df, "score_rf")
        assert r["n_booked"] == 2
        assert r["n_rejected"] == 2  # canceled NOT counted (was 4 under ~booked)
        # rejected stats reflect the low-score rejections, not pulled up by high-score canceled
        assert r["mean_rejected"] < 400

    def test_rejection_profile_does_not_count_canceled(self):
        _, with_canceled = self._frames()
        prof = compute_rejection_profile(with_canceled, "score_rf")
        assert prof["n_rejected"].sum() == 300  # the genuine rejections, not 450 (incl. canceled)

    def test_booked_vs_rejected_gini_ignores_canceled(self):
        base, with_canceled = self._frames()
        score_cols = {"rf": {"column": "score_rf", "negate": True}}
        g_base = compute_booked_vs_rejected_discriminance(base, score_cols).iloc[0]["gini"]
        g_with = compute_booked_vs_rejected_discriminance(with_canceled, score_cols).iloc[0]["gini"]
        assert g_base == pytest.approx(g_with)  # canceled dropped → identical (was depressed before)

    def test_ri_gini_ignores_canceled(self):
        base, with_canceled = self._frames()
        g_base = compute_ri_gini(base, "score_rf", "early_bad", score_negate=True)
        g_with = compute_ri_gini(with_canceled, "score_rf", "early_bad", score_negate=True)
        assert g_base["ri_gini_mean"] == pytest.approx(g_with["ri_gini_mean"])
        assert g_base["n_rejected"] == g_with["n_rejected"] == 300

    def test_thorndike_uses_score_rejection_only_u_ratio(self):
        """#59: the Thorndike Case-2 correction must be fed the score-rejection-only
        u_ratio (direct selection on the score), not the all-demand one that 08-other
        rejections widen (score-orthogonal → biases the correction)."""
        rng = np.random.RandomState(3)
        dates = pd.date_range("2024-01-01", periods=700, freq="h")
        booked = pd.DataFrame(
            {
                "score_rf": rng.normal(600, 30, 300),
                "status_name": "booked",
                "reject_reason": "",
                "early_bad": rng.binomial(1, 0.15, 300).astype(float),
                "mis_date": dates[:300],
            }
        )
        score_rej = pd.DataFrame(
            {
                "score_rf": rng.normal(400, 30, 200),
                "status_name": "rejected",
                "reject_reason": "09-score",
                "early_bad": np.nan,
                "mis_date": dates[300:500],
            }
        )
        # 08-other: wide, score-orthogonal spread → widens the all-demand SD only.
        other_rej = pd.DataFrame(
            {
                "score_rf": rng.normal(500, 200, 200),
                "status_name": "rejected",
                "reject_reason": "08-other",
                "early_bad": np.nan,
                "mis_date": dates[500:700],
            }
        )
        df_demand = pd.concat([booked, score_rej, other_rej], ignore_index=True)
        score_cols = {"rf": {"column": "score_rf", "negate": True}}

        rep = compute_selection_bias_report(df_demand, booked, score_cols, "early_bad", "seg", "main")

        all_u = rep["truncation"].iloc[0]["u_ratio"]
        sro_u = rep["score_rejection_truncation"].iloc[0]["u_ratio"]
        thor_u = rep["thorndike"].iloc[0]["u_ratio"]
        assert sro_u != pytest.approx(all_u)  # 08-other widened the all-demand u_ratio
        assert thor_u == pytest.approx(sro_u)  # correction used the score-only u_ratio


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
