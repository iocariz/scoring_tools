import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.reject_inference import (
    _enforce_multiplier_monotonicity,
    _fix_partial_order_violations,
    apply_parceling_adjustment,
    apply_reject_inference,
    compute_acceptance_rates,
    compute_ri_confidence,
)


def _count_partial_order_violations(df, variables, inv_set=None):
    """Count strictly-dominating pairs (a riskier than b in all dims) with mult[a] < mult[b]."""
    inv_set = inv_set or set()
    signs = np.array([(-1 if v in inv_set else 1) for v in variables])
    oriented = df[variables].to_numpy() * signs
    vals = df["reject_risk_multiplier"].to_numpy()
    n = len(df)
    c = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = oriented[i] - oriented[j]
            if np.all(d >= 0) and np.any(d > 0) and vals[i] < vals[j] - 1e-9:
                c += 1
    return c


# =============================================================================
# Helper to build demand DataFrames
# =============================================================================


def _make_demand(rows):
    """Build a demand DataFrame from a list of (var0, var1, status, reject_reason) tuples."""
    return pd.DataFrame(rows, columns=["var0", "var1", "status_name", "reject_reason"])


def _make_demand_with_dates(rows):
    """Build a demand DataFrame including `mis_date`.

    rows are tuples: (var0, var1, mis_date, status_name, reject_reason)
    """
    return pd.DataFrame(rows, columns=["var0", "var1", "mis_date", "status_name", "reject_reason"])


VARIABLES = ["var0", "var1"]


# =============================================================================
# compute_acceptance_rates Tests
# =============================================================================


class TestComputeAcceptanceRates:
    def test_basic_rates(self):
        """50/50 booked vs score-rejected in a single bin."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
                (1, 1, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        assert len(rates) == 1
        assert rates.iloc[0]["acceptance_rate"] == pytest.approx(0.5)
        assert rates.iloc[0]["n_booked"] == 2
        assert rates.iloc[0]["n_score_rejected"] == 2

    def test_all_booked(self):
        """100% acceptance when no score rejections exist."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "booked", None),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        assert rates.iloc[0]["acceptance_rate"] == pytest.approx(1.0)
        assert rates.iloc[0]["n_score_rejected"] == 0

    def test_all_rejected(self):
        """0% acceptance when no bookings exist."""
        demand = _make_demand(
            [
                (1, 1, "rejected", "09-score"),
                (1, 1, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        assert rates.iloc[0]["acceptance_rate"] == pytest.approx(0.0)
        assert rates.iloc[0]["n_booked"] == 0

    def test_ignores_08_other(self):
        """08-other rejections should not be counted."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "08-other"),
                (1, 1, "rejected", "08-other"),
                (1, 1, "rejected", "08-other"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        # Only booked count, 08-other is ignored → acceptance = 1.0
        assert rates.iloc[0]["acceptance_rate"] == pytest.approx(1.0)
        assert rates.iloc[0]["n_score_rejected"] == 0

    def test_include_all_rejections_deprecated_and_ignored(self):
        """include_all_rejections=True is deprecated: 08-other still excluded.

        Regression for #3: the swap-in (repesca) population is solely
        score-rejected, so non-score rejections must never enter the
        acceptance-rate denominator regardless of the deprecated flag.
        """
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
                (1, 1, "rejected", "08-other"),
                (1, 1, "rejected", "08-other"),
            ]
        )
        rates_default = compute_acceptance_rates(demand, VARIABLES)
        rates_flag = compute_acceptance_rates(demand, VARIABLES, include_all_rejections=True)

        # Score-only regardless of the flag: 1 booked + 1 score-rejected → 0.5,
        # the two 08-other rejections excluded.
        for rates in (rates_default, rates_flag):
            assert rates.iloc[0]["n_score_rejected"] == 1
            assert rates.iloc[0]["acceptance_rate"] == pytest.approx(0.5)
        assert rates_flag.iloc[0]["acceptance_rate"] == pytest.approx(rates_default.iloc[0]["acceptance_rate"])

    def test_multiple_bins(self):
        """Rates computed independently per bin."""
        demand = _make_demand(
            [
                # Bin (1,1): 1 booked, 3 rejected → 25%
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
                (1, 1, "rejected", "09-score"),
                (1, 1, "rejected", "09-score"),
                # Bin (2,2): 3 booked, 1 rejected → 75%
                (2, 2, "booked", None),
                (2, 2, "booked", None),
                (2, 2, "booked", None),
                (2, 2, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        assert len(rates) == 2

        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        bin_22 = rates[(rates["var0"] == 2) & (rates["var1"] == 2)].iloc[0]

        assert bin_11["acceptance_rate"] == pytest.approx(0.25)
        assert bin_22["acceptance_rate"] == pytest.approx(0.75)

    def test_canceled_ignored(self):
        """Canceled applications should not affect rates."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "canceled", None),
                (1, 1, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES)
        assert rates.iloc[0]["acceptance_rate"] == pytest.approx(0.5)


# =============================================================================
# Bayesian Smoothing Tests
# =============================================================================


class TestBayesianSmoothing:
    def test_smoothing_produces_column(self):
        """Bayesian smoothing adds smoothed_acceptance_rate column."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES, bayesian_smoothing=True)
        assert "smoothed_acceptance_rate" in rates.columns

    def test_smoothing_shrinks_toward_global(self):
        """Smoothed rate should be pulled toward global rate for small bins."""
        demand = _make_demand(
            [
                # Bin (1,1): 1 booked, 9 rejected → raw 10%
                *[(1, 1, "booked", None)] * 1,
                *[(1, 1, "rejected", "09-score")] * 9,
                # Bin (2,2): 90 booked, 10 rejected → raw 90%
                *[(2, 2, "booked", None)] * 90,
                *[(2, 2, "rejected", "09-score")] * 10,
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES, bayesian_smoothing=True, bayesian_prior_strength=10.0)

        bin_11 = rates[rates["var0"] == 1].iloc[0]
        bin_22 = rates[rates["var0"] == 2].iloc[0]

        # Global rate = 91/110 ≈ 0.827
        # Smoothed rate for small bin (1,1) should be pulled up from 0.1 toward global
        assert bin_11["smoothed_acceptance_rate"] > bin_11["acceptance_rate"]
        # Smoothed rate for large bin (2,2) should barely change
        assert bin_22["smoothed_acceptance_rate"] < bin_22["acceptance_rate"]


# =============================================================================
# Time-aware acceptance rate Tests
# =============================================================================


class TestTimeAwareAcceptanceRates:
    def test_recent_window_filters_acceptance_rates(self):
        """Recent-window mode should exclude older booked/rejected records."""
        # max_date is 2024-03-15
        old_date = pd.Timestamp("2024-01-15")
        recent_date = pd.Timestamp("2024-03-15")

        demand = _make_demand_with_dates(
            [
                # Old bin: 1 booked, 9 rejected → 10% acceptance (should be excluded)
                *[(1, 1, old_date, "booked", None)] * 1,
                *[(1, 1, old_date, "rejected", "09-score")] * 9,
                # Recent bin: 9 booked, 1 rejected → 90% acceptance (should dominate)
                *[(1, 1, recent_date, "booked", None)] * 9,
                *[(1, 1, recent_date, "rejected", "09-score")] * 1,
            ]
        )

        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            recent_months=1,
            decay_half_life_months=None,
        )

        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        assert bin_11["acceptance_rate"] == pytest.approx(0.9)

    def test_exp_decay_weights_more_recent_records(self):
        """Exponential decay mode should up-weight recent booked and down-weight older rejects."""
        old_date = pd.Timestamp("2024-02-15")
        recent_date = pd.Timestamp("2024-03-15")  # max date anchor

        half_life = 0.5  # months
        demand = _make_demand_with_dates(
            [
                (1, 1, recent_date, "booked", None),
                (1, 1, old_date, "rejected", "09-score"),
            ]
        )

        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            recent_months=None,
            decay_half_life_months=half_life,
        )

        max_date = recent_date
        age_days = (max_date - old_date).total_seconds() / (3600 * 24)
        age_months = age_days / 30.437
        w_old = 2.0 ** (-age_months / half_life)
        expected = 1.0 / (1.0 + w_old)

        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        assert bin_11["acceptance_rate"] == pytest.approx(expected, rel=1e-3)

    def test_decay_with_unparseable_dates_falls_back_to_counts_no_crash(self):
        """Decay requested but every date is unparseable -> max_date is NaT. The old code set
        decay=None but never computed the counts-based aggregates, so the merge NameErrored.
        Now it falls back to unweighted counts and returns valid rates."""
        demand = _make_demand_with_dates(
            [
                (1, 1, "not-a-date", "booked", None),
                (1, 1, "also-bad", "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            recent_months=None,
            decay_half_life_months=0.5,  # decay requested
        )
        # Unweighted fallback: 1 booked + 1 score-rejected -> acceptance rate 0.5
        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        assert bin_11["acceptance_rate"] == pytest.approx(0.5)

    def test_empirical_bayes_adjusts_effective_prior_strength(self):
        """Empirical-Bayes should adapt prior strength based on cross-bin variance.

        Regression: avoid stale fixed `bayesian_prior_strength` by adapting
        shrinkage from observed cross-bin acceptance dispersion.
        """
        demand = _make_demand(
            [
                # Bin (1,1): 4 booked, 6 rejected → 0.4 (small n)
                *[(1, 1, "booked", None)] * 4,
                *[(1, 1, "rejected", "09-score")] * 6,
                # Bin (2,2): 60 booked, 40 rejected → 0.6 (large n)
                *[(2, 2, "booked", None)] * 60,
                *[(2, 2, "rejected", "09-score")] * 40,
            ]
        )

        bayesian_prior_strength = 10.0
        rates = compute_acceptance_rates(
            demand, VARIABLES, bayesian_smoothing=True, bayesian_prior_strength=bayesian_prior_strength
        )
        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        bin_22 = rates[(rates["var0"] == 2) & (rates["var1"] == 2)].iloc[0]

        # Baseline "fixed prior" posterior mean using cfg prior_strength.
        global_rate = (bin_11["n_booked"] + bin_22["n_booked"]) / (
            bin_11["n_booked"] + bin_11["n_score_rejected"] + bin_22["n_booked"] + bin_22["n_score_rejected"]
        )
        alpha_cfg = max(bayesian_prior_strength * global_rate, 0.5)
        beta_cfg = max(bayesian_prior_strength * (1 - global_rate), 0.5)

        total_11 = float(bin_11["n_booked"] + bin_11["n_score_rejected"])
        total_22 = float(bin_22["n_booked"] + bin_22["n_score_rejected"])
        smoothed_cfg_11 = (float(bin_11["n_booked"]) + alpha_cfg) / (total_11 + alpha_cfg + beta_cfg)
        smoothed_cfg_22 = (float(bin_22["n_booked"]) + alpha_cfg) / (total_22 + alpha_cfg + beta_cfg)

        # Effective prior strength should differ from cfg, producing stronger
        # shrinkage for the small bin and weaker pull against large bin.
        assert bin_11["smoothed_acceptance_rate"] > smoothed_cfg_11
        assert bin_22["smoothed_acceptance_rate"] < smoothed_cfg_22

    def test_no_smoothing_no_column(self):
        """Without bayesian_smoothing, no smoothed column should be present."""
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES, bayesian_smoothing=False)
        assert "smoothed_acceptance_rate" not in rates.columns

    def test_smoothed_rate_used_in_parceling(self):
        """When smoothed rates are available, parceling should use them."""
        repesca = pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [100.0], "todu_amt_pile_h6": [500.0]})
        rates = pd.DataFrame(
            {
                "var0": [1],
                "var1": [1],
                "n_booked": [1],
                "n_score_rejected": [9],
                "acceptance_rate": [0.1],
                "smoothed_acceptance_rate": [0.4],
            }
        )

        # With smoothed rate, effective rate is 0.4 → reject_ratio = 0.6 → mult = 1 + 1.5*0.6 = 1.9
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(1.9)


# =============================================================================
# Bayesian smoothing under time-decay (Kish effective sample size)
# =============================================================================


def _decay_weight(date, max_date, half_life):
    """Replicate the decay weight used in compute_acceptance_rates."""
    age_days = (max_date - date).total_seconds() / (3600 * 24)
    age_months = age_days / 30.437
    return 2.0 ** (-age_months / half_life)


class TestSmoothingWithDecay:
    def test_smoothing_with_decay_matches_count_path_at_unit_weights(self):
        """With all records at the same date, weights = 1 ⇒ Σw = Σw² = n_raw ⇒ n_eff = n_raw,
        so the decay posterior reduces to the count-*scale* posterior on raw counts.

        Note: this checks the decay posterior against the *configured-strength* formula
        ``(rate·n_raw + α)/(n_raw + α + β)`` — NOT against the no-decay code path's output.
        At unit weights n_eff = n_raw, but decay-on and decay-off still diverge when there
        are ≥2 bins, because the decay path uses the configured prior strength while the
        no-decay path auto-tunes it (Option-2's deliberate asymmetry). So the comparison is
        deliberately to the closed-form configured-strength posterior, not to a decay-off run.
        """
        d = pd.Timestamp("2024-03-15")
        demand = _make_demand_with_dates(
            [
                *[(1, 1, d, "booked", None)] * 2,
                *[(1, 1, d, "rejected", "09-score")] * 8,
                *[(2, 2, d, "booked", None)] * 80,
                *[(2, 2, d, "rejected", "09-score")] * 20,
            ]
        )
        strength = 10.0
        rates = compute_acceptance_rates(
            demand, VARIABLES, bayesian_smoothing=True, bayesian_prior_strength=strength, decay_half_life_months=240.0
        )

        # All weights are exactly 1 (age 0) ⇒ Σw = Σw² = n_raw ⇒ n_eff = n_raw.
        global_rate = rates["n_booked"].sum() / (rates["n_booked"] + rates["n_score_rejected"]).sum()
        alpha = max(strength * global_rate, 0.5)
        beta = max(strength * (1 - global_rate), 0.5)
        for _, row in rates.iterrows():
            n_raw = row["n_booked_raw"] + row["n_score_rejected_raw"]
            expected = (row["acceptance_rate"] * n_raw + alpha) / (n_raw + alpha + beta)
            assert row["smoothed_acceptance_rate"] == pytest.approx(expected)

    def test_smoothing_with_decay_less_shrinkage_than_sumw_scale(self):
        """The Kish n_eff posterior shrinks less toward the global rate than the old
        Σw-scale posterior, because Σw ≤ n_eff."""
        recent = pd.Timestamp("2024-03-15")  # max-date anchor, weight ≈ 1
        old = pd.Timestamp("2022-03-15")  # ~24 months back, small weight
        half_life = 6.0
        demand = _make_demand_with_dates(
            [
                # Bin (1,1): recent booked dominate the weighted rate; some old rejects.
                *[(1, 1, recent, "booked", None)] * 8,
                *[(1, 1, old, "rejected", "09-score")] * 8,
                # Bin (2,2): anchors the global rate away from bin (1,1)'s rate.
                *[(2, 2, recent, "booked", None)] * 10,
                *[(2, 2, recent, "rejected", "09-score")] * 90,
            ]
        )
        strength = 10.0
        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            bayesian_smoothing=True,
            bayesian_prior_strength=strength,
            decay_half_life_months=half_life,
        )

        # Recompute Σw, Σw², n_eff for bin (1,1) from first principles.
        w_recent = _decay_weight(recent, recent, half_life)  # = 1.0
        w_old = _decay_weight(old, recent, half_life)
        sumw = 8 * w_recent + 8 * w_old
        sumw2 = 8 * w_recent**2 + 8 * w_old**2
        n_eff = sumw**2 / sumw2
        assert sumw < n_eff < 16  # Σw ≤ n_eff ≤ n_raw, strict here (mixed weights)

        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        rate = bin_11["acceptance_rate"]

        global_rate = rates["n_booked"].sum() / (rates["n_booked"] + rates["n_score_rejected"]).sum()
        alpha = max(strength * global_rate, 0.5)
        beta = max(strength * (1 - global_rate), 0.5)

        new_smoothed = (rate * n_eff + alpha) / (n_eff + alpha + beta)
        old_smoothed = (rate * sumw + alpha) / (sumw + alpha + beta)  # buggy Σw-scale posterior

        assert bin_11["smoothed_acceptance_rate"] == pytest.approx(new_smoothed)
        # Core fix: the n_eff posterior stays closer to the observed rate (less shrinkage).
        assert abs(new_smoothed - rate) < abs(old_smoothed - rate)

    def test_smoothing_with_decay_uses_configured_prior_strength(self):
        """Under decay the posterior uses the configured prior strength directly
        (no empirical-Bayes auto-tuning)."""
        recent = pd.Timestamp("2024-03-15")
        old = pd.Timestamp("2023-09-15")
        half_life = 9.0
        demand = _make_demand_with_dates(
            [
                *[(1, 1, recent, "booked", None)] * 4,
                *[(1, 1, old, "rejected", "09-score")] * 6,
                *[(2, 2, recent, "booked", None)] * 60,
                *[(2, 2, old, "rejected", "09-score")] * 40,
            ]
        )
        strength = 10.0
        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            bayesian_smoothing=True,
            bayesian_prior_strength=strength,
            decay_half_life_months=half_life,
        )

        global_rate = rates["n_booked"].sum() / (rates["n_booked"] + rates["n_score_rejected"]).sum()
        alpha = max(strength * global_rate, 0.5)
        beta = max(strength * (1 - global_rate), 0.5)
        for _, row in rates.iterrows():
            total = row["n_booked"] + row["n_score_rejected"]
            # n_eff is not returned; reconstruct it from Σw and Σw² for this bin's rows.
            if row["var0"] == 1:
                w_b, w_r, nb, nr = (
                    _decay_weight(recent, recent, half_life),
                    _decay_weight(old, recent, half_life),
                    4,
                    6,
                )
            else:
                w_b, w_r, nb, nr = (
                    _decay_weight(recent, recent, half_life),
                    _decay_weight(old, recent, half_life),
                    60,
                    40,
                )
            sumw = nb * w_b + nr * w_r
            sumw2 = nb * w_b**2 + nr * w_r**2
            n_eff = sumw**2 / sumw2
            assert total == pytest.approx(sumw)
            expected = (row["acceptance_rate"] * n_eff + alpha) / (n_eff + alpha + beta)
            assert row["smoothed_acceptance_rate"] == pytest.approx(expected)

    def test_smoothing_no_decay_regression(self):
        """Golden values pinning the untouched count-based smoothing path, *including* the
        empirical-Bayes auto-tune.

        Uses two bins with rates either side of the global rate so the posterior is
        sensitive to the prior strength and the denominator. A single-bin case is
        degenerate — ``global_rate == bin_rate`` makes smoothing a no-op for any α/β/n, so
        it cannot detect a wrong strength or denominator and gives false confidence.
        """
        demand = _make_demand(
            [
                *[(1, 1, "booked", None)] * 3,
                *[(1, 1, "rejected", "09-score")] * 7,  # bin (1,1): rate 0.3, n=10
                *[(2, 2, "booked", None)] * 70,
                *[(2, 2, "rejected", "09-score")] * 30,  # bin (2,2): rate 0.7, n=100
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES, bayesian_smoothing=True, bayesian_prior_strength=10.0)
        bin_11 = rates[(rates["var0"] == 1) & (rates["var1"] == 1)].iloc[0]
        bin_22 = rates[(rates["var0"] == 2) & (rates["var1"] == 2)].iloc[0]

        # global_rate = 73/110 ≈ 0.6636. The EB auto-tune blends the configured strength
        # (10) with the cross-bin moment estimate → effective strength ≈ 6.148
        # (α ≈ 4.08, β ≈ 2.07). These golden values are captured from a verified run and
        # drift if either the posterior formula or the EB blend changes.
        global_rate = 0.6636363636363637
        assert bin_11["smoothed_acceptance_rate"] == pytest.approx(0.4384475711581742)
        assert bin_22["smoothed_acceptance_rate"] == pytest.approx(0.697893828233913)
        # Sanity: each bin shrinks toward the global rate, and the smoothed value is
        # genuinely moved off the raw rate by the auto-tuned strength (non-degenerate).
        assert bin_11["acceptance_rate"] < bin_11["smoothed_acceptance_rate"] < global_rate
        assert global_rate < bin_22["smoothed_acceptance_rate"] < bin_22["acceptance_rate"]

    def test_smoothing_decay_edges(self):
        """Single bin and rate ∈ {0, 1} under decay yield finite rates in [0, 1]."""
        d = pd.Timestamp("2024-03-15")
        # Single bin, all booked → rate 1.0
        demand_all = _make_demand_with_dates([*[(1, 1, d, "booked", None)] * 5])
        rates_all = compute_acceptance_rates(
            demand_all, VARIABLES, bayesian_smoothing=True, decay_half_life_months=12.0
        )
        s_all = rates_all.iloc[0]["smoothed_acceptance_rate"]
        assert np.isfinite(s_all) and 0.0 <= s_all <= 1.0

        # Single bin, all rejected → rate 0.0
        demand_none = _make_demand_with_dates([*[(1, 1, d, "rejected", "09-score")] * 5])
        rates_none = compute_acceptance_rates(
            demand_none, VARIABLES, bayesian_smoothing=True, decay_half_life_months=12.0
        )
        s_none = rates_none.iloc[0]["smoothed_acceptance_rate"]
        assert np.isfinite(s_none) and 0.0 <= s_none <= 1.0

    def test_smoothing_decay_does_not_leak_internal_columns(self):
        """The Σw² temp columns must not appear in the returned schema."""
        d = pd.Timestamp("2024-03-15")
        demand = _make_demand_with_dates(
            [
                (1, 1, d, "booked", None),
                (1, 1, d, "rejected", "09-score"),
            ]
        )
        rates = compute_acceptance_rates(demand, VARIABLES, bayesian_smoothing=True, decay_half_life_months=12.0)
        assert "__sumw2_booked" not in rates.columns
        assert "__sumw2_rej" not in rates.columns


# =============================================================================
# apply_parceling_adjustment Tests
# =============================================================================


class TestApplyParcelingAdjustment:
    def _make_repesca(self, todu_val=100.0, amt_val=500.0):
        return pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [todu_val], "todu_amt_pile_h6": [amt_val]})

    def _make_rates(self, acceptance_rate=1.0):
        return pd.DataFrame(
            {
                "var0": [1],
                "var1": [1],
                "n_booked": [1000],
                "n_score_rejected": [0],
                "acceptance_rate": [acceptance_rate],
            }
        )

    def test_no_adjustment_at_full_acceptance(self):
        """100% acceptance → multiplier = 1.0, no change."""
        repesca = self._make_repesca(todu_val=100.0)
        rates = self._make_rates(acceptance_rate=1.0)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(100.0)
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(1.0)

    def test_full_uplift_at_zero_acceptance(self):
        """0% acceptance → reject_ratio=1.0 → multiplier = 1 + 1.5*1 = 2.5."""
        repesca = self._make_repesca(todu_val=100.0)
        rates = self._make_rates(acceptance_rate=0.0)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(2.5)
        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(250.0)

    def test_max_cap(self):
        """With high uplift factor, multiplier is capped at max_risk_multiplier."""
        repesca = self._make_repesca(todu_val=100.0)
        rates = self._make_rates(acceptance_rate=0.0)
        result = apply_parceling_adjustment(
            repesca, rates, VARIABLES, reject_uplift_factor=5.0, max_risk_multiplier=3.0
        )

        # Without cap: 1 + 5.0 * 1.0 = 6.0, but capped at 3.0
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(3.0)
        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(300.0)

    def test_partial_rate(self):
        """50% acceptance → reject_ratio=0.5 → multiplier = 1 + 1.5*0.5 = 1.75."""
        repesca = self._make_repesca(todu_val=200.0)
        rates = self._make_rates(acceptance_rate=0.5)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)
        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(350.0)

    def test_revenue_unchanged(self):
        """todu_amt_pile_h6 (revenue proxy) should not be modified."""
        repesca = self._make_repesca(todu_val=100.0, amt_val=500.0)
        rates = self._make_rates(acceptance_rate=0.0)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        assert result["todu_amt_pile_h6"].iloc[0] == pytest.approx(500.0)

    def test_missing_bins_get_conservative_anchor(self):
        """No-demand bins get a conservative LOW anchor (low percentile of observed rates),
        NOT the anti-conservative median — so they receive a HIGHER reject uplift (#5).

        Replaces the old test_missing_bins_get_median_adjustment, which pinned the bug
        (missing bin filled with the median ⇒ near-typical, anti-conservative multiplier).
        """
        repesca = pd.DataFrame(
            {
                "var0": [1, 2, 3, 9],
                "var1": [1, 2, 3, 9],
                "todu_30ever_h6": [100.0, 100.0, 100.0, 100.0],
                "todu_amt_pile_h6": [500.0, 500.0, 500.0, 500.0],
            }
        )
        # Three well-observed, high-acceptance bins; bin (9,9) has NO demand (absent).
        rates = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 2, 3],
                "n_booked": [700, 800, 900],
                "n_score_rejected": [300, 200, 100],
                "acceptance_rate": [0.7, 0.8, 0.9],
            }
        )
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, no_demand_anchor_percentile=0.10)

        observed = np.array([0.7, 0.8, 0.9])
        anchor = max(float(np.quantile(observed, 0.10)), 0.01)  # ≈ 0.72
        median = float(np.median(observed))  # 0.80

        missing = result[result["var0"] == 9].iloc[0]
        # No-demand bin shrinks fully to the conservative anchor, not the median.
        assert missing["ri_effective_acceptance_rate"] == pytest.approx(anchor)
        assert missing["ri_effective_acceptance_rate"] < median
        # → strictly higher reject multiplier than the buggy median fill would have produced.
        assert missing["reject_risk_multiplier"] == pytest.approx(1.0 + 1.5 * (1.0 - anchor))
        assert missing["reject_risk_multiplier"] > 1.0 + 1.5 * (1.0 - median)

    def test_multiple_bins(self):
        """Multiple bins with different acceptance rates."""
        repesca = pd.DataFrame(
            {"var0": [1, 2], "var1": [1, 2], "todu_30ever_h6": [100.0, 100.0], "todu_amt_pile_h6": [500.0, 500.0]}
        )
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                # Well-observed bins (large n) so confidence≈1 and the conservative shrinkage is a
                # no-op — this test isolates the per-method multiplier formula.
                "n_booked": [1000, 500],
                "n_score_rejected": [0, 500],
                "acceptance_rate": [1.0, 0.5],
            }
        )
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        # Bin (1,1): acceptance=1.0 → mult=1.0
        assert result[result["var0"] == 1]["todu_30ever_h6"].iloc[0] == pytest.approx(100.0)
        # Bin (2,2): acceptance=0.5 → mult=1.75
        assert result[result["var0"] == 2]["todu_30ever_h6"].iloc[0] == pytest.approx(175.0)


# =============================================================================
# Sigmoid Method Tests
# =============================================================================


class TestSigmoidMethod:
    def _make_repesca(self, todu_val=100.0):
        return pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [todu_val], "todu_amt_pile_h6": [500.0]})

    def _make_rates(self, acceptance_rate=1.0):
        return pd.DataFrame(
            {
                "var0": [1],
                "var1": [1],
                "n_booked": [1000],
                "n_score_rejected": [0],
                "acceptance_rate": [acceptance_rate],
            }
        )

    def test_sigmoid_low_acceptance(self):
        """Low acceptance rate produces high multiplier with sigmoid."""
        repesca = self._make_repesca()
        rates = self._make_rates(acceptance_rate=0.1)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, method="sigmoid")

        # At rate=0.1 with steepness=10, midpoint=0.5: exp(10*(0.1-0.5)) = exp(-4) ≈ 0.018
        # multiplier = 1 + 1.5 / (1 + 0.018) ≈ 1 + 1.47 = 2.47
        assert result["reject_risk_multiplier"].iloc[0] > 2.0

    def test_sigmoid_high_acceptance(self):
        """High acceptance rate produces low multiplier with sigmoid."""
        repesca = self._make_repesca()
        rates = self._make_rates(acceptance_rate=0.9)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, method="sigmoid")

        # At rate=0.9 with steepness=10, midpoint=0.5: exp(10*(0.9-0.5)) = exp(4) ≈ 54.6
        # multiplier = 1 + 1.5 / (1 + 54.6) ≈ 1 + 0.027 ≈ 1.027
        assert result["reject_risk_multiplier"].iloc[0] < 1.1

    def test_sigmoid_midpoint(self):
        """At midpoint (0.5), sigmoid produces moderate multiplier."""
        repesca = self._make_repesca()
        rates = self._make_rates(acceptance_rate=0.5)
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, method="sigmoid")

        # At rate=0.5: exp(0) = 1, multiplier = 1 + 1.5/2 = 1.75
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)

    def test_sigmoid_monotonic_with_rate(self):
        """Sigmoid multiplier should decrease as acceptance rate increases."""
        repesca = pd.DataFrame(
            {"var0": [1, 2, 3], "var1": [1, 2, 3], "todu_30ever_h6": [100.0] * 3, "todu_amt_pile_h6": [500.0] * 3}
        )
        rates = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 2, 3],
                "n_booked": [2, 5, 9],
                "n_score_rejected": [8, 5, 1],
                "acceptance_rate": [0.2, 0.5, 0.9],
            }
        )
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, method="sigmoid")
        multipliers = result.sort_values("var0")["reject_risk_multiplier"].values
        assert multipliers[0] > multipliers[1] > multipliers[2]


# =============================================================================
# Monotonicity Enforcement Tests
# =============================================================================


class TestMonotonicityEnforcement:
    def test_enforces_monotonicity(self):
        """Multipliers should be non-decreasing along each variable axis after enforcement."""
        result = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "reject_risk_multiplier": [2.0, 1.5, 2.5],  # Non-monotone: 2.0 > 1.5
            }
        )
        enforced = _enforce_multiplier_monotonicity(result.copy(), VARIABLES)
        # After isotonic regression along var0, should be non-decreasing
        mults = enforced.sort_values("var0")["reject_risk_multiplier"].values
        assert all(mults[i] <= mults[i + 1] for i in range(len(mults) - 1))

    def test_already_monotone_unchanged(self):
        """Already monotone multipliers should stay the same."""
        result = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "reject_risk_multiplier": [1.0, 1.5, 2.0],
            }
        )
        enforced = _enforce_multiplier_monotonicity(result.copy(), VARIABLES)
        mults = enforced.sort_values("var0")["reject_risk_multiplier"].values
        np.testing.assert_array_almost_equal(mults, [1.0, 1.5, 2.0])

    def test_monotonicity_in_parceling(self):
        """enforce_monotonicity param in apply_parceling_adjustment works."""
        repesca = pd.DataFrame(
            {"var0": [1, 2, 3], "var1": [1, 1, 1], "todu_30ever_h6": [100.0] * 3, "todu_amt_pile_h6": [500.0] * 3}
        )
        # Non-monotone acceptance rates: bin 2 has higher rate than bin 3
        rates = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "n_booked": [1, 8, 5],
                "n_score_rejected": [9, 2, 5],
                "acceptance_rate": [0.1, 0.8, 0.5],
            }
        )
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, enforce_monotonicity=True)
        mults = result.sort_values("var0")["reject_risk_multiplier"].values
        # Should be non-decreasing after enforcement
        assert all(mults[i] <= mults[i + 1] + 1e-9 for i in range(len(mults) - 1))


# =============================================================================
# Per-Bin Confidence Tests
# =============================================================================


class TestComputeRIConfidence:
    def test_confidence_columns(self):
        """compute_ri_confidence returns expected columns."""
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "n_booked": [50, 5],
                "n_score_rejected": [50, 5],
                "acceptance_rate": [0.5, 0.5],
            }
        )
        conf = compute_ri_confidence(rates, VARIABLES)
        assert "ri_confidence" in conf.columns
        assert "ri_bin_count" in conf.columns
        assert "ri_bin_count_effective" in conf.columns
        assert "ri_bin_count_raw" in conf.columns
        assert "var0" in conf.columns
        assert "var1" in conf.columns

    def test_higher_count_higher_confidence(self):
        """Bins with more observations should have higher confidence."""
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "n_booked": [5, 500],
                "n_score_rejected": [5, 500],
                "acceptance_rate": [0.5, 0.5],
            }
        )
        conf = compute_ri_confidence(rates, VARIABLES)
        assert conf[conf["var0"] == 2]["ri_confidence"].iloc[0] > conf[conf["var0"] == 1]["ri_confidence"].iloc[0]

    def test_confidence_range(self):
        """Confidence should be in [0, 1)."""
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "n_booked": [0, 10000],
                "n_score_rejected": [0, 10000],
                "acceptance_rate": [0.0, 0.5],
            }
        )
        conf = compute_ri_confidence(rates, VARIABLES)
        assert (conf["ri_confidence"] >= 0).all()
        assert (conf["ri_confidence"] <= 1).all()

    def test_decay_mode_keeps_raw_and_effective_counts(self):
        """Decay-weighted rates should preserve both effective and raw sample counts."""
        old_date = pd.Timestamp("2024-02-15")
        recent_date = pd.Timestamp("2024-03-15")
        half_life = 0.5
        demand = _make_demand_with_dates(
            [
                (1, 1, recent_date, "booked", None),
                (1, 1, old_date, "rejected", "09-score"),
            ]
        )

        rates = compute_acceptance_rates(
            demand,
            VARIABLES,
            recent_months=None,
            decay_half_life_months=half_life,
        )
        conf = compute_ri_confidence(rates, VARIABLES)
        row = conf.iloc[0]

        assert row["ri_bin_count_raw"] == 2
        assert row["ri_bin_count"] == 2  # backward-compatible raw count
        assert row["ri_bin_count_effective"] < 2.0

    def test_confidence_merged_in_parceling(self):
        """apply_reject_inference with parceling should include confidence columns."""
        repesca = pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [100.0], "todu_amt_pile_h6": [500.0]})
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
            ]
        )
        result = apply_reject_inference(repesca, demand, VARIABLES, method="parceling")
        assert "ri_confidence" in result.columns
        assert "ri_bin_count" in result.columns
        assert "ri_bin_count_effective" in result.columns
        assert "ri_bin_count_raw" in result.columns


# =============================================================================
# apply_reject_inference Tests (dispatcher)
# =============================================================================


class TestApplyRejectInference:
    def _make_data(self):
        repesca = pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [100.0], "todu_amt_pile_h6": [500.0]})
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                (1, 1, "rejected", "09-score"),
            ]
        )
        return repesca, demand

    def test_none_returns_unchanged(self):
        """method='none' returns repesca_summary without modification."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(repesca, demand, VARIABLES, method="none")
        pd.testing.assert_frame_equal(result, repesca)

    def test_parceling_delegates(self):
        """method='parceling' applies adjustment."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(repesca, demand, VARIABLES, method="parceling")

        # acceptance_rate = 0.5, reject_ratio = 0.5, mult = 1 + 1.5*0.5 = 1.75
        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(175.0)
        # Revenue unchanged
        assert result["todu_amt_pile_h6"].iloc[0] == pytest.approx(500.0)

    def test_invalid_method_raises(self):
        """Unknown method raises ValueError."""
        repesca, demand = self._make_data()
        with pytest.raises(ValueError, match="Unknown reject inference method"):
            apply_reject_inference(repesca, demand, VARIABLES, method="invalid")

    def test_custom_parameters(self):
        """Custom uplift factor and max multiplier are passed through."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(
            repesca, demand, VARIABLES, method="parceling", reject_uplift_factor=4.0, max_risk_multiplier=2.0
        )

        # acceptance=0.5, reject_ratio=0.5, raw_mult=1+4.0*0.5=3.0, capped at 2.0
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(2.0)
        assert result["todu_30ever_h6"].iloc[0] == pytest.approx(200.0)

    def test_parceling_method_power_forwarded(self):
        """parceling_method='power' should use power-law formula."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(
            repesca, demand, VARIABLES, method="parceling", parceling_method="power", reject_uplift_factor=1.0
        )
        # acceptance=0.5, power: (1/0.5)^1.0 = 2.0
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(2.0)

    def test_parceling_method_sigmoid_forwarded(self):
        """parceling_method='sigmoid' should use sigmoid formula."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(repesca, demand, VARIABLES, method="parceling", parceling_method="sigmoid")
        # At rate=0.5: sigmoid midpoint → mult = 1 + 1.5/2 = 1.75
        assert result["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)

    def test_bayesian_smoothing_forwarded(self):
        """bayesian_smoothing params are forwarded through dispatcher."""
        repesca, demand = self._make_data()
        result = apply_reject_inference(
            repesca,
            demand,
            VARIABLES,
            method="parceling",
            bayesian_smoothing=True,
            bayesian_prior_strength=10.0,
        )
        # Should have smoothed columns
        assert "smoothed_acceptance_rate" in result.columns

    def test_enforce_monotonicity_forwarded(self):
        """enforce_monotonicity param is forwarded through dispatcher."""
        repesca = pd.DataFrame(
            {"var0": [1, 2, 3], "var1": [1, 1, 1], "todu_30ever_h6": [100.0] * 3, "todu_amt_pile_h6": [500.0] * 3}
        )
        demand = _make_demand(
            [
                (1, 1, "booked", None),
                *[(1, 1, "rejected", "09-score")] * 9,
                *[(2, 1, "booked", None)] * 8,
                *[(2, 1, "rejected", "09-score")] * 2,
                *[(3, 1, "booked", None)] * 5,
                *[(3, 1, "rejected", "09-score")] * 5,
            ]
        )
        result = apply_reject_inference(repesca, demand, VARIABLES, method="parceling", enforce_monotonicity=True)
        mults = result.sort_values("var0")["reject_risk_multiplier"].values
        assert all(mults[i] <= mults[i + 1] + 1e-9 for i in range(len(mults) - 1))


# =============================================================================
# Config integration Tests
# =============================================================================


class TestConfigRejectInference:
    def test_default_values(self):
        """PreprocessingSettings has reject inference defaults."""
        from src.config import PreprocessingSettings

        settings = PreprocessingSettings(
            keep_vars=["authorization_id"],
            indicators=["oa_amt"],
            octroi_bins=[-float("inf"), float("inf")],
            efx_bins=[-float("inf"), float("inf")],
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-12-01",
            variables=["var0", "var1"],
        )
        assert settings.reject_inference_method == "none"
        assert settings.reject_uplift_factor == 1.5
        assert settings.reject_max_risk_multiplier == 3.0
        assert settings.reject_bayesian_smoothing is False
        assert settings.reject_bayesian_prior_strength == 10.0
        assert settings.reject_enforce_monotonicity is False
        assert settings.ri_calibration_gamma == 1.0
        assert settings.ri_optimizer_method == "grid"
        assert settings.ri_optuna_n_trials == 100

    def test_custom_values(self):
        """PreprocessingSettings accepts custom reject inference config."""
        from src.config import PreprocessingSettings

        settings = PreprocessingSettings(
            keep_vars=["authorization_id"],
            indicators=["oa_amt"],
            octroi_bins=[-float("inf"), float("inf")],
            efx_bins=[-float("inf"), float("inf")],
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-12-01",
            variables=["var0", "var1"],
            reject_inference_method="parceling",
            reject_uplift_factor=2.0,
            reject_max_risk_multiplier=5.0,
            reject_bayesian_smoothing=True,
            reject_bayesian_prior_strength=20.0,
            reject_enforce_monotonicity=True,
            reject_parceling_method="sigmoid",
            ri_calibration_gamma=0.8,
            ri_optimizer_method="optuna",
            ri_optuna_n_trials=200,
        )
        assert settings.reject_inference_method == "parceling"
        assert settings.reject_uplift_factor == 2.0
        assert settings.reject_max_risk_multiplier == 5.0
        assert settings.reject_bayesian_smoothing is True
        assert settings.reject_bayesian_prior_strength == 20.0
        assert settings.reject_enforce_monotonicity is True
        assert settings.reject_parceling_method == "sigmoid"
        assert settings.ri_calibration_gamma == 0.8
        assert settings.ri_optimizer_method == "optuna"
        assert settings.ri_optuna_n_trials == 200

    def test_invalid_method_rejected(self):
        """Invalid reject_inference_method is rejected by pydantic."""
        from src.config import PreprocessingSettings

        with pytest.raises(ValueError):
            PreprocessingSettings(
                keep_vars=["authorization_id"],
                indicators=["oa_amt"],
                octroi_bins=[-float("inf"), float("inf")],
                efx_bins=[-float("inf"), float("inf")],
                date_ini_book_obs="2024-01-01",
                date_fin_book_obs="2024-12-01",
                variables=["var0", "var1"],
                reject_inference_method="invalid",
            )

    def test_uplift_factor_bounds(self):
        """reject_uplift_factor respects ge=0 and le=10 bounds."""
        from src.config import PreprocessingSettings

        with pytest.raises(ValueError):
            PreprocessingSettings(
                keep_vars=["authorization_id"],
                indicators=["oa_amt"],
                octroi_bins=[-float("inf"), float("inf")],
                efx_bins=[-float("inf"), float("inf")],
                date_ini_book_obs="2024-01-01",
                date_fin_book_obs="2024-12-01",
                variables=["var0", "var1"],
                reject_uplift_factor=-1.0,
            )

    def test_max_multiplier_bounds(self):
        """reject_max_risk_multiplier respects ge=1 and le=10 bounds."""
        from src.config import PreprocessingSettings

        with pytest.raises(ValueError):
            PreprocessingSettings(
                keep_vars=["authorization_id"],
                indicators=["oa_amt"],
                octroi_bins=[-float("inf"), float("inf")],
                efx_bins=[-float("inf"), float("inf")],
                date_ini_book_obs="2024-01-01",
                date_fin_book_obs="2024-12-01",
                variables=["var0", "var1"],
                reject_max_risk_multiplier=0.5,
            )

    def test_sigmoid_parceling_method_accepted(self):
        """sigmoid is accepted as a valid parceling method."""
        from src.config import PreprocessingSettings

        settings = PreprocessingSettings(
            keep_vars=["authorization_id"],
            indicators=["oa_amt"],
            octroi_bins=[-float("inf"), float("inf")],
            efx_bins=[-float("inf"), float("inf")],
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-12-01",
            variables=["var0", "var1"],
            reject_parceling_method="sigmoid",
        )
        assert settings.reject_parceling_method == "sigmoid"

    def test_invalid_parceling_method_rejected(self):
        """Invalid parceling method is rejected by pydantic."""
        from src.config import PreprocessingSettings

        with pytest.raises(ValueError):
            PreprocessingSettings(
                keep_vars=["authorization_id"],
                indicators=["oa_amt"],
                octroi_bins=[-float("inf"), float("inf")],
                efx_bins=[-float("inf"), float("inf")],
                date_ini_book_obs="2024-01-01",
                date_fin_book_obs="2024-12-01",
                variables=["var0", "var1"],
                reject_parceling_method="invalid",
            )

    def test_calibration_gamma_bounds(self):
        """ri_calibration_gamma must be in (0, 1]."""
        from src.config import PreprocessingSettings

        with pytest.raises(ValueError):
            PreprocessingSettings(
                keep_vars=["authorization_id"],
                indicators=["oa_amt"],
                octroi_bins=[-float("inf"), float("inf")],
                efx_bins=[-float("inf"), float("inf")],
                date_ini_book_obs="2024-01-01",
                date_fin_book_obs="2024-12-01",
                variables=["var0", "var1"],
                ri_calibration_gamma=1.5,
            )


# =============================================================================
# Partial-order verification in N-D isotonic monotonicity
# =============================================================================


class TestPartialOrderVerification:
    """Tests that _enforce_multiplier_monotonicity fixes N-D partial-order violations."""

    def test_2d_partial_order_violation_fixed(self):
        """A cell that is riskier in both dims must have >= multiplier."""
        result = pd.DataFrame(
            {
                "var0": [1, 1, 2, 2],
                "var1": [1, 2, 1, 2],
                # (2,2) dominates (1,1) → mult should be >= (1,1)'s mult
                # But here (2,2) has lower mult → partial-order violation
                "reject_risk_multiplier": [1.0, 1.5, 1.5, 0.8],
            }
        )
        enforced = _enforce_multiplier_monotonicity(result.copy(), ["var0", "var1"])
        mults = enforced.set_index(["var0", "var1"])["reject_risk_multiplier"]
        # (2,2) should not be less than (1,1)
        assert mults[(2, 2)] >= mults[(1, 1)] - 1e-9

    def test_3d_partial_order_check(self):
        """3D grid: all domination relationships should hold after enforcement."""
        rows = []
        for v0 in [1, 2]:
            for v1 in [1, 2]:
                for v2 in [1, 2]:
                    # Deliberately non-monotone: lower-risk cells get higher mults
                    rows.append(
                        {"var0": v0, "var1": v1, "var2": v2, "reject_risk_multiplier": 3.0 - v0 - v1 + v2 * 0.1}
                    )
        result = pd.DataFrame(rows)
        variables = ["var0", "var1", "var2"]
        enforced = _enforce_multiplier_monotonicity(result.copy(), variables)
        mults = enforced.set_index(variables)["reject_risk_multiplier"]

        # Check all domination pairs: if a >= b in all coords, mult[a] >= mult[b]
        for idx_a in mults.index:
            for idx_b in mults.index:
                if all(a >= b for a, b in zip(idx_a, idx_b)) and any(a > b for a, b in zip(idx_a, idx_b)):
                    assert mults[idx_a] >= mults[idx_b] - 1e-6, (
                        f"Partial-order violation: {idx_a} dominates {idx_b} "
                        f"but mult {mults[idx_a]:.4f} < {mults[idx_b]:.4f}"
                    )


class TestPosetIsotonicProjection:
    """Audit #17: block-pooling generalized PAVA — a valid, terminating poset isotonic projection."""

    def test_zero_residual_violations_on_tangled_grid(self):
        """The core guarantee: after enforcement, ZERO partial-order violations remain."""
        rng = np.random.RandomState(7)
        rows = []
        for v0 in range(4):
            for v1 in range(4):
                # tangled (anti-monotone-ish + noise) multiplier surface
                rows.append({"var0": v0, "var1": v1, "reject_risk_multiplier": float(6 - v0 - v1) + rng.uniform(-1, 1)})
        result = pd.DataFrame(rows)
        enforced = _enforce_multiplier_monotonicity(result.copy(), ["var0", "var1"])
        assert _count_partial_order_violations(enforced, ["var0", "var1"]) == 0

    def test_pools_decreasing_chain_to_isotonic_mean(self):
        """A totally-ordered decreasing chain projects to its mean (the true isotonic solution),
        not the arbitrary monotone staircase the old pairwise-averaging produced."""
        chain = pd.DataFrame(
            {"var0": list(range(10)), "var1": list(range(10)), "reject_risk_multiplier": np.arange(10, 0, -1.0)}
        )
        out = chain.copy()
        _fix_partial_order_violations(out, ["var0", "var1"], set())
        vals = out["reject_risk_multiplier"].to_numpy()
        # block pooling collapses the whole violating chain to its mean (= 5.5)
        np.testing.assert_allclose(vals, 5.5)

    def test_idempotent_and_already_monotone_unchanged(self):
        monotone = pd.DataFrame(
            {"var0": [1, 1, 2, 2], "var1": [1, 2, 1, 2], "reject_risk_multiplier": [1.0, 1.5, 1.5, 2.0]}
        )
        out = monotone.copy()
        merges = _fix_partial_order_violations(out, ["var0", "var1"], set())
        assert merges == 0
        np.testing.assert_array_almost_equal(out["reject_risk_multiplier"].to_numpy(), [1.0, 1.5, 1.5, 2.0])
        # idempotent: a second pass changes nothing
        _fix_partial_order_violations(out, ["var0", "var1"], set())
        np.testing.assert_array_almost_equal(out["reject_risk_multiplier"].to_numpy(), [1.0, 1.5, 1.5, 2.0])

    def test_inverted_axis_direction(self):
        """For an inv_var (higher bin = safer), domination is oriented the other way."""
        # var0 inverted: higher var0 = safer, so LOWER var0 should carry the higher multiplier.
        df = pd.DataFrame({"var0": [1, 2, 3], "var1": [1, 1, 1], "reject_risk_multiplier": [1.0, 2.0, 3.0]})
        _fix_partial_order_violations(df, ["var0", "var1"], {"var0"})
        # with var0 inverted this [1,2,3] is a violation (riskier low bin has lower mult) -> pooled
        assert _count_partial_order_violations(df, ["var0", "var1"], {"var0"}) == 0

    def test_clip_then_enforce_bounded_and_monotone(self):
        """apply_parceling_adjustment: output is within [1, max_mult] AND monotone (clip-then-enforce)."""
        repesca = pd.DataFrame(
            {"var0": [1, 2, 3], "var1": [1, 1, 1], "todu_30ever_h6": [100.0] * 3, "todu_amt_pile_h6": [500.0] * 3}
        )
        rates = pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "n_booked": [1, 8, 5],
                "n_score_rejected": [9, 2, 5],
                "acceptance_rate": [0.05, 0.9, 0.4],
            }
        )
        result = apply_parceling_adjustment(
            repesca, rates, VARIABLES, enforce_monotonicity=True, max_risk_multiplier=2.5
        )
        mults = result.sort_values("var0")["reject_risk_multiplier"].to_numpy()
        assert (mults >= 1.0 - 1e-9).all() and (mults <= 2.5 + 1e-9).all()  # bounded
        assert all(mults[i] <= mults[i + 1] + 1e-9 for i in range(len(mults) - 1))  # monotone


# =============================================================================
# H1-new: reject_apply_h3_multiplier flag
# =============================================================================


class TestApplyH3MultiplierFlag:
    """Tests that apply_h3_multiplier controls H3 risk adjustment."""

    def _make_repesca_with_h3(self):
        return pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "todu_30ever_h6": [100.0, 200.0, 300.0],
                "todu_amt_pile_h6": [500.0, 500.0, 500.0],
                "todu_30ever_h3": [50.0, 100.0, 150.0],
                "todu_amt_pile_h3": [500.0, 500.0, 500.0],
            }
        )

    def _make_rates(self):
        return pd.DataFrame(
            {
                "var0": [1, 2, 3],
                "var1": [1, 1, 1],
                "n_booked": [8, 5, 2],
                "n_score_rejected": [2, 5, 8],
                "acceptance_rate": [0.8, 0.5, 0.2],
            }
        )

    def test_h3_multiplier_not_applied_by_default(self):
        """Default (apply_h3_multiplier=False): H3 is NOT scaled like H6."""
        repesca = self._make_repesca_with_h3()
        rates = self._make_rates()
        original_h3 = repesca["todu_30ever_h3"].copy()

        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        # H3 should be unchanged
        np.testing.assert_array_equal(result["todu_30ever_h3"].values, original_h3.values)
        # H6 should still be scaled by reject risk multiplier
        h6_ratio = result["todu_30ever_h6"].values / self._make_repesca_with_h3()["todu_30ever_h6"].values
        assert not np.allclose(h6_ratio, np.ones_like(h6_ratio))

    def test_h3_multiplier_not_applied(self):
        """apply_h3_multiplier=False: H3 is preserved unchanged."""
        repesca = self._make_repesca_with_h3()
        rates = self._make_rates()
        original_h3 = repesca["todu_30ever_h3"].values.copy()

        result = apply_parceling_adjustment(repesca, rates, VARIABLES, apply_h3_multiplier=False)

        # H6 should still be adjusted
        assert result["todu_30ever_h6"].iloc[2] > 300.0
        # H3 should be unchanged
        np.testing.assert_array_equal(result["todu_30ever_h3"].values, original_h3)

    def test_no_h3_column_no_error(self):
        """When todu_30ever_h3 is absent, no error regardless of flag."""
        repesca = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 1],
                "todu_30ever_h6": [100.0, 200.0],
                "todu_amt_pile_h6": [500.0, 500.0],
            }
        )
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 1],
                "n_booked": [8, 5],
                "n_score_rejected": [2, 5],
                "acceptance_rate": [0.8, 0.5],
            }
        )
        # Should not raise for either flag value
        apply_parceling_adjustment(repesca, rates, VARIABLES, apply_h3_multiplier=True)
        apply_parceling_adjustment(repesca.copy(), rates, VARIABLES, apply_h3_multiplier=False)


# =============================================================================
# Confidence-weighted conservative shrinkage of no/low-demand bins (#5)
# =============================================================================


class TestConfidenceShrinkage:
    """apply_parceling_adjustment shrinks no/low-demand acceptance rates toward a
    conservative low anchor (audit #5), exposed via ri_effective_acceptance_rate."""

    @staticmethod
    def _repesca(bins):
        return pd.DataFrame(
            {
                "var0": [b[0] for b in bins],
                "var1": [b[1] for b in bins],
                "todu_30ever_h6": [100.0] * len(bins),
                "todu_amt_pile_h6": [500.0] * len(bins),
            }
        )

    @staticmethod
    def _rates(rows):
        # rows: list of (var0, var1, rate, n_total) with n_score_rejected = n_total*(1-rate)
        return pd.DataFrame(
            {
                "var0": [r[0] for r in rows],
                "var1": [r[1] for r in rows],
                "n_booked": [int(round(r[3] * r[2])) for r in rows],
                "n_score_rejected": [int(round(r[3] * (1 - r[2]))) for r in rows],
                "acceptance_rate": [r[2] for r in rows],
            }
        )

    def test_no_demand_shrinks_fully_to_anchor(self):
        """Absent bin → conf 0 → effective rate exactly the conservative anchor."""
        repesca = self._repesca([(1, 1), (2, 2), (9, 9)])
        rates = self._rates([(1, 1, 0.6, 2000), (2, 2, 0.9, 2000)])
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, no_demand_anchor_percentile=0.10)
        anchor = max(float(np.quantile([0.6, 0.9], 0.10)), 0.01)
        missing = result[result["var0"] == 9].iloc[0]
        assert missing["ri_effective_acceptance_rate"] == pytest.approx(anchor)

    def test_low_demand_partial_shrink(self):
        """A sparse high-rate bin is pulled partway toward the low anchor (anchor < eff < raw)."""
        repesca = self._repesca([(1, 1), (2, 2), (3, 3)])
        rates = self._rates([(1, 1, 0.2, 2000), (2, 2, 0.3, 2000), (3, 3, 0.9, 4)])
        scale = 10.0
        result = apply_parceling_adjustment(
            repesca, rates, VARIABLES, no_demand_anchor_percentile=0.10, confidence_scale=scale
        )
        anchor = max(float(np.quantile([0.2, 0.3, 0.9], 0.10)), 0.01)
        conf = 1.0 - np.exp(-4 / scale)
        expected = conf * 0.9 + (1 - conf) * anchor
        sparse = result[result["var0"] == 3].iloc[0]
        assert 0.0 < conf < 1.0
        assert anchor < sparse["ri_effective_acceptance_rate"] < 0.9
        assert sparse["ri_effective_acceptance_rate"] == pytest.approx(expected)

    def test_well_observed_unchanged(self):
        """At scale=10, well-observed bins (large n ⇒ conf≈1) keep ~their raw rate."""
        repesca = self._repesca([(1, 1), (2, 2), (3, 3)])
        rates = self._rates([(1, 1, 0.2, 2000), (2, 2, 0.5, 2000), (3, 3, 0.9, 2000)])
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, confidence_scale=10.0)
        eff = result.set_index("var0")["ri_effective_acceptance_rate"]
        assert eff[1] == pytest.approx(0.2, abs=1e-3)
        assert eff[2] == pytest.approx(0.5, abs=1e-3)
        assert eff[3] == pytest.approx(0.9, abs=1e-3)

    def test_empty_rates_uses_fixed_fallback(self):
        """No observed demand anywhere → fixed conservative fallback (0.05), no crash."""
        repesca = self._repesca([(1, 1)])
        rates = pd.DataFrame({"var0": [], "var1": [], "n_booked": [], "n_score_rejected": [], "acceptance_rate": []})
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)
        assert result["ri_effective_acceptance_rate"].iloc[0] == pytest.approx(0.05)

    def test_single_observed_bin_degenerate_percentile(self):
        """One observed bin → percentile is that bin's rate; absent bin gets it, no crash."""
        repesca = self._repesca([(1, 1), (9, 9)])
        rates = self._rates([(1, 1, 0.6, 2000)])
        result = apply_parceling_adjustment(repesca, rates, VARIABLES)
        assert result[result["var0"] == 9].iloc[0]["ri_effective_acceptance_rate"] == pytest.approx(0.6)

    def test_anchor_floored_and_multiplier_capped(self):
        """All-low observed rates → anchor floored at 0.01; multiplier stays ≤ cap."""
        repesca = self._repesca([(1, 1), (2, 2), (9, 9)])
        rates = self._rates([(1, 1, 0.0, 2000), (2, 2, 0.005, 2000)])
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, max_risk_multiplier=3.0)
        missing = result[result["var0"] == 9].iloc[0]
        assert missing["ri_effective_acceptance_rate"] == pytest.approx(0.01)
        assert missing["reject_risk_multiplier"] <= 3.0 + 1e-9

    def test_anchor_percentile_knob_direction(self):
        """Higher anchor percentile ⇒ higher anchor ⇒ LOWER multiplier on a no-demand bin."""
        repesca = self._repesca([(1, 1), (2, 2), (3, 3), (9, 9)])
        rates = self._rates([(1, 1, 0.2, 2000), (2, 2, 0.5, 2000), (3, 3, 0.9, 2000)])
        low = apply_parceling_adjustment(repesca, rates, VARIABLES, no_demand_anchor_percentile=0.10)
        high = apply_parceling_adjustment(repesca, rates, VARIABLES, no_demand_anchor_percentile=0.50)
        m_low = low[low["var0"] == 9].iloc[0]["reject_risk_multiplier"]
        m_high = high[high["var0"] == 9].iloc[0]["reject_risk_multiplier"]
        assert m_low > m_high

    def test_smoothing_interaction_absent_bin(self):
        """With smoothing on, rate_col is the smoothed rate; absent bins still → anchor."""
        repesca = self._repesca([(1, 1), (2, 2), (9, 9)])
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "n_booked": [1400, 1800],
                "n_score_rejected": [600, 200],
                "acceptance_rate": [0.70, 0.90],
                "smoothed_acceptance_rate": [0.72, 0.88],
            }
        )
        result = apply_parceling_adjustment(repesca, rates, VARIABLES, no_demand_anchor_percentile=0.10)
        anchor = max(float(np.quantile([0.72, 0.88], 0.10)), 0.01)  # percentile of SMOOTHED rates
        assert result[result["var0"] == 9].iloc[0]["ri_effective_acceptance_rate"] == pytest.approx(anchor)
