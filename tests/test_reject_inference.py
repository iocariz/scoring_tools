import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.reject_inference import (
    _enforce_multiplier_monotonicity,
    apply_parceling_adjustment,
    apply_reject_inference,
    compute_acceptance_rates,
    compute_ri_confidence,
)

# =============================================================================
# Helper to build demand DataFrames
# =============================================================================


def _make_demand(rows):
    """Build a demand DataFrame from a list of (var0, var1, status, reject_reason) tuples."""
    return pd.DataFrame(rows, columns=["var0", "var1", "status_name", "reject_reason"])


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
# apply_parceling_adjustment Tests
# =============================================================================


class TestApplyParcelingAdjustment:
    def _make_repesca(self, todu_val=100.0, amt_val=500.0):
        return pd.DataFrame({"var0": [1], "var1": [1], "todu_30ever_h6": [todu_val], "todu_amt_pile_h6": [amt_val]})

    def _make_rates(self, acceptance_rate=1.0):
        return pd.DataFrame(
            {"var0": [1], "var1": [1], "n_booked": [10], "n_score_rejected": [0], "acceptance_rate": [acceptance_rate]}
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

    def test_missing_bins_get_median_adjustment(self):
        """Bins not in acceptance_rates get median acceptance_rate as conservative fallback."""
        repesca = pd.DataFrame(
            {"var0": [1, 2], "var1": [1, 2], "todu_30ever_h6": [100.0, 200.0], "todu_amt_pile_h6": [500.0, 600.0]}
        )
        # Only bin (1,1) has rates (median = 0.5)
        rates = self._make_rates(acceptance_rate=0.5)

        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        # Bin (1,1): adjusted with actual rate 0.5
        assert result[result["var0"] == 1]["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)
        # Bin (2,2): missing from rates → filled with median (0.5) → same multiplier
        assert result[result["var0"] == 2]["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)
        assert result[result["var0"] == 2]["todu_30ever_h6"].iloc[0] == pytest.approx(350.0)

    def test_multiple_bins(self):
        """Multiple bins with different acceptance rates."""
        repesca = pd.DataFrame(
            {"var0": [1, 2], "var1": [1, 2], "todu_30ever_h6": [100.0, 100.0], "todu_amt_pile_h6": [500.0, 500.0]}
        )
        rates = pd.DataFrame(
            {
                "var0": [1, 2],
                "var1": [1, 2],
                "n_booked": [10, 5],
                "n_score_rejected": [0, 5],
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
            {"var0": [1], "var1": [1], "n_booked": [10], "n_score_rejected": [0], "acceptance_rate": [acceptance_rate]}
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
                    rows.append({"var0": v0, "var1": v1, "var2": v2, "reject_risk_multiplier": 3.0 - v0 - v1 + v2 * 0.1})
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
