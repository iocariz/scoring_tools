import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.reject_inference import (
    apply_parceling_adjustment,
    apply_reject_inference,
    compute_acceptance_rates,
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

    def test_missing_bins_get_no_adjustment(self):
        """Bins not in acceptance_rates get acceptance_rate=1.0 (no adjustment)."""
        repesca = pd.DataFrame(
            {"var0": [1, 2], "var1": [1, 2], "todu_30ever_h6": [100.0, 200.0], "todu_amt_pile_h6": [500.0, 600.0]}
        )
        # Only bin (1,1) has rates
        rates = self._make_rates(acceptance_rate=0.5)

        result = apply_parceling_adjustment(repesca, rates, VARIABLES)

        # Bin (1,1): adjusted
        assert result[result["var0"] == 1]["reject_risk_multiplier"].iloc[0] == pytest.approx(1.75)
        # Bin (2,2): missing from rates → no adjustment
        assert result[result["var0"] == 2]["reject_risk_multiplier"].iloc[0] == pytest.approx(1.0)
        assert result[result["var0"] == 2]["todu_30ever_h6"].iloc[0] == pytest.approx(200.0)

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
        )
        assert settings.reject_inference_method == "parceling"
        assert settings.reject_uplift_factor == 2.0
        assert settings.reject_max_risk_multiplier == 5.0

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
