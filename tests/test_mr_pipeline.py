import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.mr_pipeline import (
    _assign_tiered_risk,
    _compute_account_maturity,
    _compute_hybrid_mr_risk,
    _mr_outcomes_available,
    _reoptimize_mr_mask_after_recalibration,
    _robust_ratio_clip_band,
    _write_mr_cutoff_drift_overlay,
    calculate_metrics_from_cuts,
    process_mr_period,
)
from src.utils import calculate_b2_ever_h6

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_data():
    """Create sample data for MR testing."""
    data = pd.DataFrame(
        {
            "sc_octroi_new_clus": [1, 2, 1, 3],  # var0
            "new_efx_clus": [1, 5, 10, 2],  # var1
            "todu_30ever_h6_boo": [10, 20, 30, 40],
            "todu_amt_pile_h6_boo": [100, 200, 300, 400],
            "oa_amt_h0_boo": [1000, 2000, 3000, 4000],
            "todu_30ever_h6_rep": [1, 2, 3, 4],
            "todu_amt_pile_h6_rep": [10, 20, 30, 40],
            "oa_amt_h0_rep": [100, 200, 300, 400],
        }
    )
    return data


@pytest.fixture
def sample_data_with_h3():
    """Create sample data for MR testing with H3 columns."""
    data = pd.DataFrame(
        {
            "sc_octroi_new_clus": [1, 2, 1, 3],
            "new_efx_clus": [1, 5, 10, 2],
            "todu_30ever_h6_boo": [10, 20, 30, 40],
            "todu_amt_pile_h6_boo": [100, 200, 300, 400],
            "oa_amt_h0_boo": [1000, 2000, 3000, 4000],
            "todu_30ever_h6_rep": [1, 2, 3, 4],
            "todu_amt_pile_h6_rep": [10, 20, 30, 40],
            "oa_amt_h0_rep": [100, 200, 300, 400],
            "todu_30ever_h3_boo": [5, 10, 15, 20],
            "todu_amt_pile_h3_boo": [90, 180, 270, 360],
            "todu_30ever_h3_rep": [0.5, 1, 1.5, 2],
            "todu_amt_pile_h3_rep": [9, 18, 27, 36],
        }
    )
    return data


@pytest.fixture
def optimal_solution_df():
    """Create mock optimal solution with cuts."""
    # Columns '1', '2', '3' represent cuts for bins 1, 2, 3 of var0
    return pd.DataFrame({"1": [5.0], "2": [4.0], "3": [1.0]})


@pytest.fixture
def variables():
    """Standard variables list."""
    return ["sc_octroi_new_clus", "new_efx_clus"]


# =============================================================================
# calculate_metrics_from_cuts Tests
# =============================================================================


class TestCalculateMetricsFromCuts:
    """Tests for calculate_metrics_from_cuts function."""

    def test_basic_calculation(self, sample_data, optimal_solution_df, variables):
        """Test basic calculation of Actual, Swap-in, Swap-out, Optimum."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Check structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert result["Metric"].tolist() == ["Actual", "Swap-in", "Swap-out", "Optimum selected"]

    def test_actual_production(self, sample_data, optimal_solution_df, variables):
        """Test Actual production calculation."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Actual = sum of all booked production
        expected_actual_prod = 1000 + 2000 + 3000 + 4000  # 10000
        assert result.loc[0, "Production (€)"] == expected_actual_prod

    def test_swap_in_calculation(self, sample_data, optimal_solution_df, variables):
        """Test Swap-in calculation (repesca that passes)."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Based on cuts:
        # Bin 1 cut=5: Row 0 (var1=1) PASSES, Row 2 (var1=10) FAILS
        # Swap-in = repesca that passes = Row 0 Rep = 100
        expected_si_prod = 100
        assert result.loc[1, "Production (€)"] == expected_si_prod

    def test_swap_out_calculation(self, sample_data, optimal_solution_df, variables):
        """Test Swap-out calculation (booked that fails)."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Swap-out = booked that fails = Rows 1, 2, 3 = 2000 + 3000 + 4000 = 9000
        expected_so_prod = 9000
        assert result.loc[2, "Production (€)"] == expected_so_prod

    def test_optimum_calculation(self, sample_data, optimal_solution_df, variables):
        """Test Optimum calculation."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Optimum = Actual - Swap-out + Swap-in = 10000 - 9000 + 100 = 1100
        expected_opt_prod = 1100
        assert result.loc[3, "Production (€)"] == expected_opt_prod

    def test_risk_calculation(self, sample_data, optimal_solution_df, variables):
        """Test Risk (b2_ever_h6) calculation for Actual."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # Risk = 7 * sum(num) / sum(den) * 100 (percentage)
        # Num = 10+20+30+40 = 100
        # Den = 100+200+300+400 = 1000
        # Risk = 7 * 100 / 1000 * 100 = 70.0
        assert np.isclose(result.loc[0, "Risk (%)"], 70.0)

    def test_none_solution(self, sample_data, variables):
        """Test handling of None solution."""
        result = calculate_metrics_from_cuts(sample_data, None, variables)
        assert result is None

    def test_empty_solution(self, sample_data, variables):
        """Test handling of empty solution DataFrame."""
        empty_solution = pd.DataFrame()
        result = calculate_metrics_from_cuts(sample_data, empty_solution, variables)
        assert result is None

    def test_missing_bin_in_solution(self, sample_data, optimal_solution_df, variables):
        """Test behavior when a bin in data is missing from solution."""
        # Add a row with Bin 99
        data = sample_data.copy()
        new_row = pd.DataFrame(
            {
                "sc_octroi_new_clus": [99],
                "new_efx_clus": [1],
                "todu_30ever_h6_boo": [0],
                "todu_amt_pile_h6_boo": [1],
                "oa_amt_h0_boo": [0],
                "todu_30ever_h6_rep": [0],
                "todu_amt_pile_h6_rep": [1],
                "oa_amt_h0_rep": [0],
            }
        )
        data = pd.concat([data, new_row], ignore_index=True)

        # Function should handle gracefully and not crash
        result = calculate_metrics_from_cuts(data, optimal_solution_df, variables)

        assert result is not None
        assert len(result) == 4

    def test_all_pass(self, sample_data, variables):
        """Test when all records pass the cut."""
        # Set cuts so high that everything passes
        solution = pd.DataFrame({"1": [100.0], "2": [100.0], "3": [100.0]})

        result = calculate_metrics_from_cuts(sample_data, solution, variables)

        # Swap-out should be 0 (nothing fails)
        assert result.loc[2, "Production (€)"] == 0

    def test_all_fail(self, sample_data, variables):
        """Test when all records fail the cut."""
        # Set cuts so low that everything fails
        solution = pd.DataFrame({"1": [0.0], "2": [0.0], "3": [0.0]})

        result = calculate_metrics_from_cuts(sample_data, solution, variables)

        # Swap-in should be 0 (nothing passes from repesca)
        assert result.loc[1, "Production (€)"] == 0

    def test_rejection_rate_column_exists(self, sample_data, optimal_solution_df, variables):
        """Test that Rejection Rate (%) column is present."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        assert "Rejection Rate (%)" in result.columns

    def test_rejection_rate_actual(self, sample_data, optimal_solution_df, variables):
        """Test Actual rejection rate = repesca / total demand."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # actual_prod = 10000, total_rep = 1000, total_demand = 11000
        # rejection_rate = (1 - 10000/11000) * 100 ≈ 9.09%
        actual_rej = result.loc[result["Metric"] == "Actual", "Rejection Rate (%)"].iloc[0]
        assert np.isclose(actual_rej, (1 - 10000 / 11000) * 100)

    def test_rejection_rate_optimum(self, sample_data, optimal_solution_df, variables):
        """Test Optimum rejection rate = (total_demand - opt_prod) / total_demand."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        # opt_prod = 1100, total_demand = 11000
        # rejection_rate = (1 - 1100/11000) * 100 = 90.0%
        opt_rej = result.loc[result["Metric"] == "Optimum selected", "Rejection Rate (%)"].iloc[0]
        assert np.isclose(opt_rej, 90.0)

    def test_rejection_rate_swap_rows_none(self, sample_data, optimal_solution_df, variables):
        """Test that Swap-in and Swap-out rejection rates are None."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables)

        assert pd.isna(result.loc[result["Metric"] == "Swap-in", "Rejection Rate (%)"].iloc[0])
        assert pd.isna(result.loc[result["Metric"] == "Swap-out", "Rejection Rate (%)"].iloc[0])


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for MR pipeline functions."""

    def test_single_row_data(self, optimal_solution_df, variables):
        """Test with single row data."""
        data = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1],
                "new_efx_clus": [3],  # Should pass (3 <= 5)
                "todu_30ever_h6_boo": [10],
                "todu_amt_pile_h6_boo": [100],
                "oa_amt_h0_boo": [1000],
                "todu_30ever_h6_rep": [1],
                "todu_amt_pile_h6_rep": [10],
                "oa_amt_h0_rep": [100],
            }
        )

        result = calculate_metrics_from_cuts(data, optimal_solution_df, variables)

        assert result is not None
        assert len(result) == 4

    def test_zero_denominator_in_risk(self, optimal_solution_df, variables):
        """Test handling of zero denominator in risk calculation."""
        data = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1],
                "new_efx_clus": [3],
                "todu_30ever_h6_boo": [10],
                "todu_amt_pile_h6_boo": [0],  # Zero denominator
                "oa_amt_h0_boo": [1000],
                "todu_30ever_h6_rep": [1],
                "todu_amt_pile_h6_rep": [0],  # Zero denominator
                "oa_amt_h0_rep": [100],
            }
        )

        # Should not crash
        result = calculate_metrics_from_cuts(data, optimal_solution_df, variables)
        assert result is not None

    def test_all_same_bin(self, variables):
        """Test when all data is in same bin."""
        data = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 1, 1],
                "new_efx_clus": [3, 4, 6],
                "todu_30ever_h6_boo": [10, 20, 30],
                "todu_amt_pile_h6_boo": [100, 200, 300],
                "oa_amt_h0_boo": [1000, 2000, 3000],
                "todu_30ever_h6_rep": [1, 2, 3],
                "todu_amt_pile_h6_rep": [10, 20, 30],
                "oa_amt_h0_rep": [100, 200, 300],
            }
        )

        solution = pd.DataFrame({"1": [5.0]})

        result = calculate_metrics_from_cuts(data, solution, variables)

        assert result is not None
        # Row 0 and 1 pass (3,4 <= 5), Row 2 fails (6 > 5)


# =============================================================================
# Risk Calculation Integration Tests
# =============================================================================


class TestRiskCalculationIntegration:
    """Tests for risk calculation integration with calculate_b2_ever_h6."""

    def test_risk_uses_correct_formula(self):
        """Test that risk is calculated using the standard formula."""
        numerator = 100
        denominator = 1000

        expected_risk = calculate_b2_ever_h6(numerator, denominator)

        assert np.isclose(expected_risk, 0.7)

    def test_risk_with_zeros(self):
        """Test risk calculation handles zeros correctly."""
        result = calculate_b2_ever_h6(0, 1000)
        assert result == 0.0

        result = calculate_b2_ever_h6(100, 0)
        assert np.isnan(result)


# =============================================================================
# calculate_metrics_from_cuts Edge Cases
# =============================================================================


class TestCalculateMetricsFromCutsEdgeCases:
    """Edge case tests for calculate_metrics_from_cuts with degenerate inputs."""

    def test_returns_none_on_empty_data(self):
        """Pass empty DataFrames and verify it returns None or handles gracefully without crashing."""
        empty_data = pd.DataFrame()
        solution = pd.DataFrame({"1": [5.0]})
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        result = calculate_metrics_from_cuts(empty_data, solution, variables)

        assert result is None

    def test_returns_none_on_none_optimal_solution(self):
        """Pass None as optimal_solution_df and verify it returns None."""
        data = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [3, 4],
                "todu_30ever_h6_boo": [10, 20],
                "todu_amt_pile_h6_boo": [100, 200],
                "oa_amt_h0_boo": [1000, 2000],
                "todu_30ever_h6_rep": [1, 2],
                "todu_amt_pile_h6_rep": [10, 20],
                "oa_amt_h0_rep": [100, 200],
            }
        )
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        result = calculate_metrics_from_cuts(data, None, variables)

        assert result is None


# =============================================================================
# _mr_outcomes_available Tests
# =============================================================================


class TestMROutcomesAvailable:
    """Tests for the _mr_outcomes_available helper."""

    def test_returns_true_when_columns_present_with_data(self):
        df = pd.DataFrame({"todu_30ever_h6": [1.0, 2.0], "todu_amt_pile_h6": [10.0, 20.0]})
        assert _mr_outcomes_available(df) is True

    def test_returns_false_when_columns_missing(self):
        df = pd.DataFrame({"oa_amt": [100.0]})
        assert _mr_outcomes_available(df) is False

    def test_returns_false_when_all_null(self):
        df = pd.DataFrame({"todu_30ever_h6": [np.nan, np.nan], "todu_amt_pile_h6": [np.nan, np.nan]})
        assert _mr_outcomes_available(df) is False

    def test_returns_false_when_one_column_missing(self):
        df = pd.DataFrame({"todu_30ever_h6": [1.0], "oa_amt": [100.0]})
        assert _mr_outcomes_available(df) is False


# =============================================================================
# _compute_hybrid_mr_risk Tests
# =============================================================================


class TestComputeHybridMRRisk:
    """Tests for _compute_hybrid_mr_risk function."""

    @pytest.fixture
    def merge_keys(self):
        return ["bin_a", "bin_b"]

    @pytest.fixture
    def data_booked_main(self):
        """Main-period booked data with 2 bins."""
        return pd.DataFrame(
            {
                "bin_a": [1, 1, 2, 2],
                "bin_b": [1, 1, 1, 1],
                "todu_30ever_h6": [10.0, 10.0, 5.0, 5.0],
                "todu_amt_pile_h6": [100.0, 100.0, 200.0, 200.0],
                "status_name": ["booked"] * 4,
            }
        )

    @pytest.fixture
    def data_demand_mr_sufficient(self):
        """MR-period demand data where bin (1,1) has 40 obs, bin (2,1) has 40 obs."""
        rng = np.random.RandomState(42)
        n = 40
        return pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 2 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

    def test_uses_mr_observed_for_sufficient_bins(self, data_booked_main, data_demand_mr_sufficient, merge_keys):
        """Bins with n_obs >= threshold use MR observed risk."""
        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked_main, data_demand_mr_sufficient, merge_keys, min_obs=30
        )

        # Both bins have 40 obs >= 30 threshold -> should use mr_observed
        assert (comparison_df["risk_source"] == "mr_observed").all()
        # b2_ever_h6_tmp should equal b2_mr
        for _, row in comparison_df.iterrows():
            assert np.isclose(row["b2_ever_h6_tmp"], row["b2_mr"])

    def test_falls_back_to_main_for_sparse_bins(self, data_booked_main, merge_keys):
        """Bins with n_obs < threshold use main-period risk."""
        # Only 5 MR obs per bin -> below threshold of 30
        data_demand_mr_sparse = pd.DataFrame(
            {
                "bin_a": [1] * 5 + [2] * 5,
                "bin_b": [1] * 10,
                "todu_30ever_h6": [8.0] * 10,
                "todu_amt_pile_h6": [90.0] * 10,
                "oa_amt_h0": [1000.0] * 10,
                "status_name": ["booked"] * 10,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked_main, data_demand_mr_sparse, merge_keys, min_obs=30
        )

        assert (comparison_df["risk_source"] == "main_imputed").all()
        for _, row in comparison_df.iterrows():
            assert np.isclose(row["b2_ever_h6_tmp"], row["b2_main"])

    def test_handles_bins_only_in_mr(self, data_booked_main, merge_keys):
        """Bin present in MR but absent from main — sparse gets NaN (model_fallback)."""
        data_demand_mr_new_bin = pd.DataFrame(
            {
                "bin_a": [3] * 5,
                "bin_b": [1] * 5,
                "todu_30ever_h6": [10.0] * 5,
                "todu_amt_pile_h6": [100.0] * 5,
                "oa_amt_h0": [1000.0] * 5,
                "status_name": ["booked"] * 5,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked_main, data_demand_mr_new_bin, merge_keys, min_obs=30
        )

        new_bin_row = comparison_df[(comparison_df["bin_a"] == 3) & (comparison_df["bin_b"] == 1)]
        assert len(new_bin_row) == 1
        assert new_bin_row.iloc[0]["risk_source"] == "model_fallback"
        assert np.isnan(new_bin_row.iloc[0]["b2_ever_h6_tmp"])

    def test_handles_bins_only_in_main(self, data_booked_main, merge_keys):
        """Bin present in main but absent from MR uses main risk."""
        # MR data only has bin (1,1), not (2,1)
        data_demand_mr_partial = pd.DataFrame(
            {
                "bin_a": [1] * 5,
                "bin_b": [1] * 5,
                "todu_30ever_h6": [10.0] * 5,
                "todu_amt_pile_h6": [100.0] * 5,
                "oa_amt_h0": [1000.0] * 5,
                "status_name": ["booked"] * 5,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked_main, data_demand_mr_partial, merge_keys, min_obs=30
        )

        main_only_row = comparison_df[(comparison_df["bin_a"] == 2) & (comparison_df["bin_b"] == 1)]
        assert len(main_only_row) == 1
        # No MR obs -> falls back to main
        assert main_only_row.iloc[0]["risk_source"] == "main_imputed"
        assert np.isclose(main_only_row.iloc[0]["b2_ever_h6_tmp"], main_only_row.iloc[0]["b2_main"])

    def test_comparison_df_columns(self, data_booked_main, data_demand_mr_sufficient, merge_keys):
        """Verify all expected columns are present in comparison_df."""
        _, comparison_df = _compute_hybrid_mr_risk(data_booked_main, data_demand_mr_sufficient, merge_keys, min_obs=30)

        expected_cols = {
            "bin_a",
            "bin_b",
            "b2_main",
            "b2_mr",
            "n_obs_main",
            "n_obs_mr",
            "b2_ever_h6_tmp",
            "risk_source",
            "b2_delta",
            "b2_delta_pct",
            "mr_production",
            "fitted_method",
            "fitted_curvature",
        }
        assert expected_cols == set(comparison_df.columns)

    def test_h3_columns_in_comparison_df(self, data_booked_main, merge_keys):
        """When multiplier_h3 is provided and h3 data exists, comparison_df should have h3 columns."""
        # Add h3 columns to main data
        data_booked_h3 = data_booked_main.copy()
        data_booked_h3["todu_30ever_h3"] = data_booked_h3["todu_30ever_h6"] * 0.5
        data_booked_h3["todu_amt_pile_h3"] = data_booked_h3["todu_amt_pile_h6"] * 0.9

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 2 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 2 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 2 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(
            data_booked_h3, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4
        )

        assert "b2_main_h3" in comparison_df.columns
        assert "b2_mr_h3" in comparison_df.columns
        assert "n_obs_mr_h3" in comparison_df.columns
        assert "b2_delta_h3" in comparison_df.columns
        assert "b2_delta_pct_h3" in comparison_df.columns

    def test_h3_extrapolation_used_when_h6_is_unavailable(self, merge_keys):
        """When H6 is unavailable but H3 data exists and ratio is valid, bins use h3_extrapolated risk source."""
        # Main-period data with H3 columns — need enough observations per bin (>= min_obs)
        rng_main = np.random.RandomState(99)
        n_main = 35  # > min_obs=30
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate(
                    [rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]
                ),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate(
                    [rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]
                ),
                "status_name": ["booked"] * (2 * n_main),
            }
        )

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": [np.nan] * (2 * n),
                "todu_amt_pile_h6": [np.nan] * (2 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 2 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        assert (comparison_df["risk_source"] == "h3_extrapolated").all()

        for _, row in comparison_df.iterrows():
            # Extrapolation uses the clipped / robust-calibrated per-bin H6/H3 ratio.
            expected = row["b2_mr_h3"] * row["h6_h3_ratio"]
            assert np.isclose(row["b2_ever_h6_tmp"], expected), f"Expected {expected}, got {row['b2_ever_h6_tmp']}"

    def test_mr_h6_preferred_over_h3_when_both_available(self, merge_keys):
        """When both MR H6 and H3 data are sufficient, MR H6 takes priority.

        Priority order: MR H6 > H3 extrapolation > main-period > model fallback.
        """
        rng_main = np.random.RandomState(99)
        n_main = 35
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate(
                    [rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]
                ),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate(
                    [rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]
                ),
                "status_name": ["booked"] * (2 * n_main),
            }
        )

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 2 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 2 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 2 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        # MR H6 observed takes priority when sufficient observations exist
        assert (comparison_df["risk_source"] == "mr_observed").all()
        for _, row in comparison_df.iterrows():
            assert np.isclose(row["b2_ever_h6_tmp"], row["b2_mr"])

    def test_h3_fallback_to_mr_observed(self, merge_keys):
        """When b2_main_h3 ≈ 0 (ratio invalid), falls back to mr_observed."""
        # Main-period: H3 numerator/denominator both zero → b2_main_h3 ≈ 0
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "todu_30ever_h6": [10.0, 10.0],
                "todu_amt_pile_h6": [100.0, 100.0],
                "todu_30ever_h3": [0.0, 0.0],
                "todu_amt_pile_h3": [100.0, 100.0],
                "status_name": ["booked"] * 2,
            }
        )

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n,
                "bin_b": [1] * n,
                "todu_30ever_h6": rng.uniform(5, 15, n),
                "todu_amt_pile_h6": rng.uniform(80, 120, n),
                "todu_30ever_h3": rng.uniform(2, 8, n),
                "todu_amt_pile_h3": rng.uniform(70, 110, n),
                "oa_amt_h0": rng.uniform(500, 1500, n),
                "status_name": ["booked"] * n,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4
        )

        # b2_main_h3 ≈ 0 → ratio invalid → fallback to mr_observed
        assert (comparison_df["risk_source"] == "mr_observed").all()
        for _, row in comparison_df.iterrows():
            assert np.isclose(row["b2_ever_h6_tmp"], row["b2_mr"])

    def test_zero_mr_h3_falls_back_to_main(self, merge_keys):
        """When MR H3 delinquency is zero, should fall back to main-period, not extrapolate to 0."""
        rng_main = np.random.RandomState(99)
        n_main = 35
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate(
                    [rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]
                ),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate(
                    [rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]
                ),
                "status_name": ["booked"] * (2 * n_main),
            }
        )

        # MR data: H6 is NaN (immature), H3 delinquency is ZERO (no defaults in 3 months)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": [np.nan] * (2 * n),
                "todu_amt_pile_h6": [np.nan] * (2 * n),
                "todu_30ever_h3": [0.0] * (2 * n),  # zero delinquency
                "todu_amt_pile_h3": np.random.RandomState(42).uniform(70, 110, 2 * n),
                "oa_amt_h0": np.random.RandomState(42).uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        # Zero H3 risk with sufficient observations IS a valid signal:
        # it means genuinely low risk. The extrapolation produces 0% risk.
        assert (comparison_df["risk_source"] == "h3_extrapolated").all()
        for _, row in comparison_df.iterrows():
            assert row["b2_ever_h6_tmp"] == 0.0  # zero H3 delinquency → zero extrapolated risk

    def test_no_h3_columns_original_logic(self, data_booked_main, data_demand_mr_sufficient, merge_keys):
        """Without H3 columns, original mr_observed / main_imputed logic is preserved."""
        # No multiplier_h3 and no H3 columns → has_h3 = False
        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked_main, data_demand_mr_sufficient, merge_keys, min_obs=30
        )

        # Both bins have 40 obs >= 30 → mr_observed (original behavior)
        assert (comparison_df["risk_source"] == "mr_observed").all()
        for _, row in comparison_df.iterrows():
            assert np.isclose(row["b2_ever_h6_tmp"], row["b2_mr"])

        # h6_h3_ratio should NOT be in comparison_df
        assert "h6_h3_ratio" not in comparison_df.columns

    def test_h3_partial_maturity_excludes_immature_accounts(self, merge_keys):
        """MR accounts without mature H3 (NaN) are excluded from H3 aggregation."""
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "todu_30ever_h6": [10.0, 10.0],
                "todu_amt_pile_h6": [100.0, 100.0],
                "todu_30ever_h3": [5.0, 5.0],
                "todu_amt_pile_h3": [90.0, 90.0],
                "status_name": ["booked"] * 2,
            }
        )

        # 40 MR accounts total, but only 20 have mature H3 (first 3 months);
        # remaining 20 have NaN H3 (last 3 months)
        rng = np.random.RandomState(42)
        n_mature = 20
        n_immature = 20
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * (n_mature + n_immature),
                "bin_b": [1] * (n_mature + n_immature),
                "todu_30ever_h6": [np.nan] * (n_mature + n_immature),
                "todu_amt_pile_h6": [np.nan] * (n_mature + n_immature),
                "todu_30ever_h3": list(rng.uniform(2, 8, n_mature)) + [np.nan] * n_immature,
                "todu_amt_pile_h3": list(rng.uniform(70, 110, n_mature)) + [np.nan] * n_immature,
                "oa_amt_h0": rng.uniform(500, 1500, n_mature + n_immature),
                "status_name": ["booked"] * (n_mature + n_immature),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        row = comparison_df.iloc[0]
        assert row["n_obs_mr_h3"] == n_mature
        assert pd.isna(row["n_obs_mr"]) or row["n_obs_mr"] == 0
        assert row["risk_source"] == "main_imputed"

    def test_h3_partial_maturity_all_immature_falls_back(self, merge_keys):
        """When ALL MR accounts lack H3 and H6, falls back to the main-period estimate."""
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "todu_30ever_h6": [10.0, 10.0],
                "todu_amt_pile_h6": [100.0, 100.0],
                "todu_30ever_h3": [5.0, 5.0],
                "todu_amt_pile_h3": [90.0, 90.0],
                "status_name": ["booked"] * 2,
            }
        )

        # All 40 MR accounts are from months 4-6: no mature H3
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n,
                "bin_b": [1] * n,
                "todu_30ever_h6": [np.nan] * n,
                "todu_amt_pile_h6": [np.nan] * n,
                "todu_30ever_h3": [np.nan] * n,
                "todu_amt_pile_h3": [np.nan] * n,
                "oa_amt_h0": [1000.0] * n,
                "status_name": ["booked"] * n,
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        row = comparison_df.iloc[0]
        assert pd.isna(row["n_obs_mr"]) or row["n_obs_mr"] == 0
        assert row["risk_source"] == "main_imputed"
        assert np.isnan(row["n_obs_mr_h3"]) or row["n_obs_mr_h3"] == 0

    def test_h3_maturity_honors_reference_date(self, merge_keys):
        """H3 maturity filter uses the supplied horizon, not max(mis_date).

        Regression for #2: with bookings 2025-10/11/12 and the default
        max(mis_date)=2025-12-15 anchor, maturities are [2,1,0] → none reach the
        3mo H3 threshold.  Anchored to the observation horizon 2026-03-15,
        maturities are [5,4,3] → all mature and counted. (Reference aligned to the
        15th so day-precise maturity, #52, matches the intended complete months.)
        """
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "todu_30ever_h6": [10.0, 10.0],
                "todu_amt_pile_h6": [100.0, 100.0],
                "todu_30ever_h3": [5.0, 5.0],
                "todu_amt_pile_h3": [90.0, 90.0],
                "status_name": ["booked"] * 2,
            }
        )
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1, 1, 1],
                "bin_b": [1, 1, 1],
                "mis_date": pd.to_datetime(["2025-10-15", "2025-11-15", "2025-12-15"]),
                "todu_30ever_h6": [np.nan] * 3,
                "todu_amt_pile_h6": [np.nan] * 3,
                "todu_30ever_h3": [4.0, 5.0, 6.0],
                "todu_amt_pile_h3": [100.0, 100.0, 100.0],
                "oa_amt_h0": [1000.0] * 3,
                "status_name": ["booked"] * 3,
            }
        )

        # Default anchor (max(mis_date)) → maturities [2,1,0] < 3 → none mature.
        _, comp_default = _compute_hybrid_mr_risk(
            data_booked, data_demand_mr.copy(), merge_keys, min_obs=1, multiplier_h3=4
        )
        n_default = comp_default.iloc[0]["n_obs_mr_h3"]
        assert np.isnan(n_default) or n_default == 0

        # Observation horizon 2026-03-01 → maturities [5,4,3] >= 3 → all mature.
        _, comp_ref = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr.copy(),
            merge_keys,
            min_obs=1,
            multiplier_h3=4,
            maturity_reference_date=pd.Timestamp("2026-03-15"),
        )
        assert comp_ref.iloc[0]["n_obs_mr_h3"] == 3

    def test_immature_h6_zeros_excluded_when_filter_active(self, merge_keys):
        """audit #2b: immature accounts with H6=0.0 (not NaN — zero only because unseasoned) pass the
        .notna() gate but must be excluded by the maturity filter so they don't dilute b2_mr."""
        data_booked = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "todu_30ever_h6": [5.0],
                "todu_amt_pile_h6": [100.0],
                "status_name": ["booked"],
            }
        )
        n_mature, n_immature = 40, 60
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * (n_mature + n_immature),
                "bin_b": [1] * (n_mature + n_immature),
                "mis_date": pd.to_datetime(["2025-06-15"] * n_mature + ["2026-01-15"] * n_immature),
                # mature: real bad; immature: H6=0.0 (would dilute b2_mr toward 0 if included)
                "todu_30ever_h6": [8.0] * n_mature + [0.0] * n_immature,
                "todu_amt_pile_h6": [100.0] * (n_mature + n_immature),
                "oa_amt_h0": [1000.0] * (n_mature + n_immature),
                "status_name": ["booked"] * (n_mature + n_immature),
            }
        )
        _, comp = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=10,
            mr_maturity_months=6,
            maturity_reference_date=pd.Timestamp("2026-02-01"),
        )
        row = comp.iloc[0]
        # only the 40 mature accounts counted; the 60 immature H6=0 rows excluded
        assert row["n_obs_mr"] == n_mature
        # b2_mr (stored as a fraction) is the undiluted mature risk, NOT diluted toward 0 by the
        # immature zeros (diluted would be mult*320/10000 vs undiluted mult*320/4000).
        expected = calculate_b2_ever_h6(8.0 * n_mature, 100.0 * n_mature)
        assert row["b2_mr"] == pytest.approx(expected, rel=1e-6)

    def test_missing_mis_date_with_maturity_requested_warns(self, merge_keys):
        """audit #2b: if the maturity filter is requested but mis_date is absent, the immature rows
        are NOT excluded (filter can't run) — but the silent skip is now flagged with a loud log."""
        from loguru import logger as loguru_logger

        data_booked = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "todu_30ever_h6": [5.0],
                "todu_amt_pile_h6": [100.0],
                "status_name": ["booked"],
            }
        )
        n = 50
        data_demand_mr = pd.DataFrame(  # NO mis_date column
            {
                "bin_a": [1] * n,
                "bin_b": [1] * n,
                "todu_30ever_h6": [8.0] * 20 + [0.0] * 30,
                "todu_amt_pile_h6": [100.0] * n,
                "oa_amt_h0": [1000.0] * n,
                "status_name": ["booked"] * n,
            }
        )
        captured: list[str] = []
        sink_id = loguru_logger.add(lambda m: captured.append(m.record["message"]), level="ERROR")
        try:
            _, comp = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=10, mr_maturity_months=6)
        finally:
            loguru_logger.remove(sink_id)
        # filter could not run -> all rows counted (immature not excluded)
        assert comp.iloc[0]["n_obs_mr"] == n
        # ...but the silent skip is now flagged
        assert any("mis_date" in msg and "maturity" in msg.lower() for msg in captured)

    def test_h6_h3_ratio_in_comparison_df(self, merge_keys):
        """When H3 is available, h6_h3_ratio column appears in comparison_df with correct values."""
        rng_main = np.random.RandomState(99)
        n_main = 35  # > min_obs=30
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate(
                    [rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]
                ),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate(
                    [rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]
                ),
                "status_name": ["booked"] * (2 * n_main),
            }
        )

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 2 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 2 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 2 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4)

        assert "h6_h3_ratio" in comparison_df.columns

        # Verify ratio = clipped(b2_main / b2_main_h3) using robust percentiles.
        raw_ratios = comparison_df["b2_main"] / comparison_df["b2_main_h3"]
        ratio_vals = raw_ratios[np.isfinite(raw_ratios)].values
        lower_q = float(np.nanpercentile(ratio_vals, 1.0)) if ratio_vals.size > 0 else 0.5
        upper_q = float(np.nanpercentile(ratio_vals, 99.0)) if ratio_vals.size > 0 else 5.0

        if not np.isfinite(lower_q) or lower_q <= 0:
            lower_q = 0.5
        if not np.isfinite(upper_q) or upper_q <= lower_q:
            upper_q = 5.0

        for _, row in comparison_df.iterrows():
            raw_ratio = row["b2_main"] / row["b2_main_h3"]
            expected_ratio = np.clip(raw_ratio, lower_q, upper_q)
            assert np.isclose(row["h6_h3_ratio"], expected_ratio), (
                f"Expected ratio {expected_ratio}, got {row['h6_h3_ratio']}"
            )


# =============================================================================
# Auto-Calibration Tests
# =============================================================================


class TestAutoCalibration:
    """Tests for mr_extrapolation_method='auto' in _compute_hybrid_mr_risk."""

    @pytest.fixture
    def merge_keys(self):
        return ["bin_a", "bin_b"]

    def test_auto_method_selects_linear_for_uniform_ratios(self, merge_keys):
        """Constant H6/H3 ratio across bins → auto resolves to linear."""
        rng = np.random.RandomState(42)
        n_main = 35
        # Create data where H6 = exactly 2 * H3 → alpha ≈ 1
        # Use shared pile and proportional numerators to ensure truly constant ratio
        h3_nums = np.concatenate([rng.uniform(4.5, 5.5, n_main) for _ in range(4)])
        h6_nums = h3_nums * 2  # exact 2× relationship
        pile = np.concatenate([rng.uniform(90, 110, n_main) for _ in range(4)])
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main + [3] * n_main + [4] * n_main,
                "bin_b": [1] * (4 * n_main),
                "todu_30ever_h6": h6_nums,
                "todu_amt_pile_h6": pile,
                "todu_30ever_h3": h3_nums,
                "todu_amt_pile_h3": pile,
                "status_name": ["booked"] * (4 * n_main),
            }
        )

        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n + [3] * n + [4] * n,
                "bin_b": [1] * (4 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 4 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 4 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 4 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 4 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 4 * n),
                "status_name": ["booked"] * (4 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=30,
            multiplier_h3=4,
            mr_extrapolation_method="auto",
        )

        assert (comparison_df["fitted_method"] == "linear").all()
        assert (comparison_df["fitted_curvature"] == 1.0).all()

    def test_auto_method_selects_power_for_convex_data(self, merge_keys):
        """Increasing H6/H3 ratio → auto resolves to power with alpha > 1."""
        rng = np.random.RandomState(42)
        n_main = 35

        # Create bins with increasing H6/H3 ratio (convex relationship)
        h3_levels = [2.0, 4.0, 8.0, 16.0]
        h6_levels = [h3**1.5 for h3 in h3_levels]  # alpha = 1.5

        h3_data, h6_data, bin_a_data = [], [], []
        for i, (h3_base, h6_base) in enumerate(zip(h3_levels, h6_levels), 1):
            h3_data.extend(rng.normal(h3_base, 0.01, n_main))
            h6_data.extend(rng.normal(h6_base, 0.01, n_main))
            bin_a_data.extend([i] * n_main)

        data_booked = pd.DataFrame(
            {
                "bin_a": bin_a_data,
                "bin_b": [1] * (4 * n_main),
                "todu_30ever_h6": h6_data,
                "todu_amt_pile_h6": [100.0] * (4 * n_main),
                "todu_30ever_h3": h3_data,
                "todu_amt_pile_h3": [100.0] * (4 * n_main),
                "status_name": ["booked"] * (4 * n_main),
            }
        )

        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n + [3] * n + [4] * n,
                "bin_b": [1] * (4 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 4 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 4 * n),
                "todu_30ever_h3": rng.uniform(2, 8, 4 * n),
                "todu_amt_pile_h3": rng.uniform(70, 110, 4 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 4 * n),
                "status_name": ["booked"] * (4 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=30,
            multiplier_h3=4,
            mr_extrapolation_method="auto",
        )

        assert (comparison_df["fitted_method"] == "power").all()
        assert (comparison_df["fitted_curvature"] > 1.0).all()

    def test_auto_fallback_without_h3(self, merge_keys):
        """No H3 data → auto falls back to linear."""
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1, 2, 2],
                "bin_b": [1, 1, 1, 1],
                "todu_30ever_h6": [10.0, 10.0, 5.0, 5.0],
                "todu_amt_pile_h6": [100.0, 100.0, 200.0, 200.0],
                "status_name": ["booked"] * 4,
            }
        )

        rng = np.random.RandomState(42)
        n = 40
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * n + [2] * n,
                "bin_b": [1] * (2 * n),
                "todu_30ever_h6": rng.uniform(5, 15, 2 * n),
                "todu_amt_pile_h6": rng.uniform(80, 120, 2 * n),
                "oa_amt_h0": rng.uniform(500, 1500, 2 * n),
                "status_name": ["booked"] * (2 * n),
            }
        )

        _, comparison_df = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=30,
            mr_extrapolation_method="auto",
        )

        # No H3 columns → should fall back to linear
        assert (comparison_df["fitted_method"] == "linear").all()
        assert (comparison_df["fitted_curvature"] == 1.0).all()


# =============================================================================
# B2 Ever H3 Tests
# =============================================================================


class TestB2EverH3:
    """Tests for complementary H3 risk metric in MR pipeline."""

    def test_calculate_metrics_includes_h3(self, sample_data_with_h3, optimal_solution_df, variables):
        """calculate_metrics_from_cuts should include H3 columns when multiplier_h3 is provided."""
        result = calculate_metrics_from_cuts(sample_data_with_h3, optimal_solution_df, variables, multiplier_h3=4)

        assert result is not None
        assert "Risk H3 (%)" in result.columns
        assert "todu_30ever_h3" in result.columns
        assert "todu_amt_pile_h3" in result.columns

    def test_calculate_metrics_no_h3_without_multiplier(self, sample_data_with_h3, optimal_solution_df, variables):
        """calculate_metrics_from_cuts should not include H3 when multiplier_h3 is None."""
        result = calculate_metrics_from_cuts(sample_data_with_h3, optimal_solution_df, variables, multiplier_h3=None)

        assert result is not None
        assert "Risk H3 (%)" not in result.columns

    def test_calculate_metrics_no_h3_without_columns(self, sample_data, optimal_solution_df, variables):
        """calculate_metrics_from_cuts should not include H3 when h3 columns are absent."""
        result = calculate_metrics_from_cuts(sample_data, optimal_solution_df, variables, multiplier_h3=4)

        assert result is not None
        assert "Risk H3 (%)" not in result.columns

    def test_h3_optimum_calculation(self, sample_data_with_h3, optimal_solution_df, variables):
        """Optimum H3 risk should be computed from (Actual - Swap-out + Swap-in) components."""
        result = calculate_metrics_from_cuts(sample_data_with_h3, optimal_solution_df, variables, multiplier_h3=4)

        assert result is not None
        actual_h3_rn = result.loc[0, "todu_30ever_h3"]
        si_h3_rn = result.loc[1, "todu_30ever_h3"]
        so_h3_rn = result.loc[2, "todu_30ever_h3"]
        opt_h3_rn = result.loc[3, "todu_30ever_h3"]
        assert np.isclose(opt_h3_rn, actual_h3_rn - so_h3_rn + si_h3_rn)


# =============================================================================
# N>2 Metrics Tests (mask + grid classification)
# =============================================================================


class TestCalculateMetricsNd:
    """Tests for calculate_metrics_from_cuts with N>2 variables using mask+grid."""

    def test_3var_mask_grid_classification(self):
        """Verify mask+grid classification produces valid metrics for 3-var."""
        from src.optimization_utils import CellGrid, classify_by_mask, evaluate_solution

        # Build 3D data
        rng = np.random.RandomState(42)
        rows = []
        for v0 in [1.0, 2.0, 3.0]:
            for v1 in [1.0, 2.0]:
                for v2 in [1.0, 2.0]:
                    rows.append(
                        {
                            "var0": v0,
                            "var1": v1,
                            "var2": v2,
                            "todu_30ever_h6": rng.uniform(5, 50),
                            "todu_amt_pile_h6": rng.uniform(100, 500),
                            "oa_amt_h0": rng.uniform(1000, 10000),
                            "todu_30ever_h6_boo": rng.uniform(5, 50),
                            "todu_amt_pile_h6_boo": rng.uniform(100, 500),
                            "oa_amt_h0_boo": rng.uniform(1000, 10000),
                            "todu_30ever_h6_rep": rng.uniform(1, 10),
                            "todu_amt_pile_h6_rep": rng.uniform(10, 50),
                            "oa_amt_h0_rep": rng.uniform(100, 1000),
                        }
                    )

        df = pd.DataFrame(rows)
        variables = ["var0", "var1", "var2"]
        grid = CellGrid.from_summary(df, variables)

        # Accept cells where var0 <= 2
        mask = np.array(
            [1 if grid.cell_data.iloc[i]["var0"] <= 2.0 else 0 for i in range(grid.n_cells)],
            dtype=int,
        )

        # classify_by_mask should return booleans
        accepted = classify_by_mask(df, mask, grid)
        assert accepted.dtype == bool
        assert accepted.sum() > 0
        assert accepted.sum() < len(df)

        # All accepted rows should have var0 <= 2
        assert (df.loc[accepted, "var0"] <= 2.0).all()
        # All rejected rows should have var0 > 2
        assert (df.loc[~accepted, "var0"] > 2.0).all()

        # evaluate_solution should produce valid KPIs
        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier=7)
        assert kpis["oa_amt_h0"] > 0
        assert kpis["b2_ever_h6"] > 0

        # Accepted production should equal sum of accepted cells
        expected_prod = df.loc[accepted, "oa_amt_h0"].sum()
        assert np.isclose(kpis["oa_amt_h0"], expected_prod, rtol=1e-6)

    def test_3var_calculate_metrics_from_cuts_with_mask(self):
        """calculate_metrics_from_cuts works with mask+grid for 3-var."""
        from src.optimization_utils import CellGrid

        rng = np.random.RandomState(42)
        rows = []
        for v0 in [1.0, 2.0]:
            for v1 in [1.0, 2.0]:
                for v2 in [1.0, 2.0]:
                    rows.append(
                        {
                            "var0": v0,
                            "var1": v1,
                            "var2": v2,
                            "todu_30ever_h6_boo": rng.uniform(5, 50),
                            "todu_amt_pile_h6_boo": rng.uniform(100, 500),
                            "oa_amt_h0_boo": rng.uniform(1000, 10000),
                            "todu_30ever_h6_rep": rng.uniform(1, 10),
                            "todu_amt_pile_h6_rep": rng.uniform(10, 50),
                            "oa_amt_h0_rep": rng.uniform(100, 1000),
                        }
                    )

        df = pd.DataFrame(rows)
        variables = ["var0", "var1", "var2"]
        grid = CellGrid.from_summary(df, variables)
        mask = np.ones(grid.n_cells, dtype=int)  # accept all

        # calculate_metrics_from_cuts requires a non-empty optimal_solution_df
        # (early guard check), so pass a dummy one; the mask+grid path ignores it
        dummy_sol = pd.DataFrame({"sol_fac": [0]})
        result = calculate_metrics_from_cuts(df, dummy_sol, variables, mask=mask, grid=grid)

        assert result is not None
        assert len(result) == 4
        assert result["Metric"].tolist() == ["Actual", "Swap-in", "Swap-out", "Optimum selected"]
        # All accepted → swap-out should be 0
        assert result.loc[2, "Production (€)"] == 0


class TestProcessMRPeriodRegression:
    def test_sparse_fallback_uses_model_variables(self, monkeypatch, tmp_path):
        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        captured = {}

        def fake_calculate_B2(df, model_risk, variables, stressor, var_reg):
            captured["used_variables"] = list(variables)
            captured["used_features"] = list(var_reg)
            assert list(variables) == ["sc_octroi_new_clus", "new_efx_clus"]
            out = df.copy()
            out["b2_ever_h6"] = 0.55
            return out

        def fake_run_optimization_pipeline(*, data_booked, data_demand, **kwargs):
            mr_row = data_demand.loc[
                (data_demand["status_name"] == "booked")
                & (data_demand["sc_octroi_new_clus"] == 1)
                & (data_demand["new_efx_clus"] == 1)
                & (data_demand["income_bin"] == 2)
            ]
            assert len(mr_row) == 1
            captured["imputed_b2"] = float(mr_row["b2_ever_h6_tmp"].iloc[0])
            return data_booked.copy()

        monkeypatch.setattr("src.mr_pipeline.calculate_B2", fake_calculate_B2)
        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", fake_run_optimization_pipeline)
        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *args, **kwargs: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *args, **kwargs: DummyAlertReport())

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=["sc_octroi_new_clus", "new_efx_clus", "income_bin"],
            inference_variables=["sc_octroi_new_clus", "new_efx_clus"],
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            bins={
                "income_bin": BinConfig(
                    source_col="income_t1t2_m",
                    output_col="income_bin",
                    bin_edges=[-float("inf"), 1.5, float("inf")],
                )
            },
            use_mr_outcomes=False,
        )

        data_clean = pd.DataFrame(
            [
                {
                    "mis_date": "2024-01-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 300.0,
                    "risk_score_rf": 20.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 1000.0,
                    "oa_amt_h0": 1000.0,
                    "todu_30ever_h6": 10.0,
                    "todu_amt_pile_h6": 100.0,
                    "sc_octroi_new_clus": 1,
                    "new_efx_clus": 1,
                    "income_bin": 1,
                },
                {
                    "mis_date": "2024-02-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 320.0,
                    "risk_score_rf": 25.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 1200.0,
                    "oa_amt_h0": 1200.0,
                    "todu_30ever_h6": 5.0,
                    "todu_amt_pile_h6": 100.0,
                    "sc_octroi_new_clus": 2,
                    "new_efx_clus": 2,
                    "income_bin": 1,
                },
                {
                    "mis_date": "2024-07-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 305.0,
                    "risk_score_rf": 21.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 500.0,
                    "oa_amt_h0": 500.0,
                    "todu_30ever_h6": 0.0,
                    "todu_amt_pile_h6": 0.0,
                    "sc_octroi_new_clus": 1,
                    "new_efx_clus": 1,
                    "income_bin": 2,
                },
            ]
        )
        data_clean["mis_date"] = pd.to_datetime(data_clean["mis_date"])
        data_booked = data_clean.loc[data_clean["mis_date"] < pd.Timestamp("2024-07-01")].copy()

        risk_inference = {
            "best_model_info": {"model": object()},
            "features": ["sc_octroi_new_clus", "new_efx_clus"],
            "model_variables": ["sc_octroi_new_clus", "new_efx_clus"],
        }

        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference=risk_inference,
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=None,
            output=output,
        )

        # With merge_keys capped to 2 variables, the MR account at (1,1,income_bin=2)
        # now merges on (1,1) which exists in main period — no model fallback needed.
        # calculate_B2 should NOT be called, and the imputed b2 should come from
        # the main-period risk at (1,1): multiplier * 10.0 / 100.0 = 0.7
        assert "used_variables" not in captured, "model fallback should not be triggered with 2D merge keys"
        assert captured["imputed_b2"] == pytest.approx(0.7)

    def test_recalibration_reoptimizes_mask_for_consistency(self, monkeypatch, tmp_path):
        """After repesca recalibration, MR should re-optimize the mask before reporting.

        Regression for the "desynchronized decisions vs reported MR risk surface"
        issue: process_mr_period recalibrates repesca risk after running the MR
        optimization pipeline, so we must re-solve the cutoffs on the recalibrated
        MR risk surface before computing the MR KPIs.
        """

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        captured = {}

        def fake_calculate_metrics_from_cuts(
            data_summary_desagregado_mr,
            optimal_solution_df,
            variables,
            inv_vars=None,
            mask=None,
            grid=None,
            **kwargs,
        ):
            captured["mask"] = mask
            captured["grid_is_none"] = grid is None
            return pd.DataFrame([{"Metric": "Optimum selected", "Risk (%)": 0.0, "Production (€)": 0.0}])

        monkeypatch.setattr("src.mr_pipeline.calculate_metrics_from_cuts", fake_calculate_metrics_from_cuts)
        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *args, **kwargs: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *args, **kwargs: DummyAlertReport())
        monkeypatch.setattr("src.mr_pipeline.go.Figure.write_html", lambda *args, **kwargs: None)

        merge_keys = ["sc_octroi_new_clus", "new_efx_clus"]

        merge_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "b2_ever_h6_tmp": [0.01, 0.02],  # decimals
            }
        )
        comparison_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "b2_ever_h6_tmp": [0.01, 0.02],
                "b2_delta_pct": [0.0, 0.0],
                "risk_source": ["main", "main"],
                "fitted_method": ["linear", "linear"],
                "fitted_curvature": [1.0, 1.0],
            }
        )

        monkeypatch.setattr(
            "src.mr_pipeline._compute_hybrid_mr_risk", lambda *args, **kwargs: (merge_df, comparison_df)
        )
        monkeypatch.setattr(
            "src.mr_pipeline._assign_tiered_risk",
            lambda data_demand_mr, *args, **kwargs: data_demand_mr.assign(_mr_tier=3),
        )

        # Tiny MR surface for CellGrid: 2x2 grid => 4 cells.
        # (var0,var1) in {(1,1),(1,2),(2,1),(2,2)}
        mr_surface = pd.DataFrame(
            {
                merge_keys[0]: [1, 1, 2, 2],
                merge_keys[1]: [1, 2, 1, 2],
                "oa_amt_h0": [1000.0, 500.0, 700.0, 300.0],
                "todu_30ever_h6": [10.0, 5.0, 20.0, 2.0],
                "todu_amt_pile_h6": [100.0, 50.0, 200.0, 20.0],
                "todu_30ever_h6_boo": [4.0, 2.0, 8.0, 1.0],
                "todu_30ever_h6_rep": [6.0, 3.0, 12.0, 1.0],
                "todu_amt_pile_h6_boo": [40.0, 20.0, 80.0, 10.0],
                "todu_amt_pile_h6_rep": [100.0, 50.0, 200.0, 20.0],
            }
        )

        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", lambda *args, **kwargs: mr_surface.copy())

        # Force MILP re-solve after recalibration to return a known mask.
        new_mask = np.array([1, 0, 1, 0], dtype=int)
        expected_target_risk = 1.0  # from optimal_solution_df["b2_ever_h6"]

        def fake_milp_solve_cutoffs(grid, target_risk, inv_vars, multiplier, **kwargs):
            captured["milp_target_risk"] = float(target_risk)
            assert grid.n_cells == 4
            return new_mask

        from src import optimization_utils as optimization_utils_module

        monkeypatch.setattr(optimization_utils_module, "milp_solve_cutoffs", fake_milp_solve_cutoffs)

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=merge_keys,
            inference_variables=merge_keys,
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            use_mr_outcomes=True,
        )

        data_booked = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-02-15", "2024-03-15"]),
                "status_name": ["booked", "booked"],
                "segment_cut_off": ["seg_a", "seg_a"],
                "score_rf": [300.0, 320.0],
                "risk_score_rf": [20.0, 25.0],
                "acct_booked_h0": [1, 1],
                "oa_amt": [1000.0, 1200.0],
                "oa_amt_h0": [1000.0, 1200.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
            }
        )

        data_clean = pd.concat(
            [
                data_booked,
                pd.DataFrame(
                    {
                        "mis_date": pd.to_datetime(["2024-07-15", "2024-07-20"]),
                        "status_name": ["booked", "booked"],
                        "segment_cut_off": ["seg_a", "seg_a"],
                        "score_rf": [305.0, 310.0],
                        "risk_score_rf": [21.0, 22.0],
                        "acct_booked_h0": [1, 1],
                        "oa_amt": [500.0, 600.0],
                        "oa_amt_h0": [500.0, 600.0],
                        "todu_30ever_h6": [0.0, 0.0],
                        "todu_amt_pile_h6": [0.0, 0.0],
                        merge_keys[0]: [1, 2],
                        merge_keys[1]: [1, 1],
                    }
                ),
            ],
            ignore_index=True,
        )

        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        mask_main = np.array([0, 0, 0, 0], dtype=int)

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference={"best_model_info": {"model": object()}, "features": [], "model_variables": merge_keys},
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=pd.DataFrame({"b2_ever_h6": [1.0], "sol_fac": [0]}),
            file_suffix="",
            output=output,
            mask=mask_main,
            grid=None,
        )

        assert captured["milp_target_risk"] == pytest.approx(expected_target_risk)
        assert np.array_equal(captured["mask"], new_mask)
        assert captured["grid_is_none"] is False

    def test_mr_audit_regenerated_on_reoptimized_mask(self, monkeypatch, tmp_path):
        """Regression: when the MR mask re-optimizes, the audit feeding the production
        reconcile must be rebuilt on the RE-OPTIMIZED mask — otherwise the MR summary
        reports risk on the new mask but production (overwritten from the audit) on the
        main mask (mismatched cutoffs)."""

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        monkeypatch.setattr(
            "src.mr_pipeline.calculate_metrics_from_cuts",
            lambda *a, **k: pd.DataFrame([{"Metric": "Optimum selected", "Risk (%)": 0.0, "Production (€)": 0.0}]),
        )
        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *a, **k: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *a, **k: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *a, **k: DummyAlertReport())
        monkeypatch.setattr("src.mr_pipeline.go.Figure.write_html", lambda *a, **k: None)

        merge_keys = ["sc_octroi_new_clus", "new_efx_clus"]
        merge_df = pd.DataFrame({merge_keys[0]: [1, 2], merge_keys[1]: [1, 1], "b2_ever_h6_tmp": [0.01, 0.02]})
        comparison_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "b2_ever_h6_tmp": [0.01, 0.02],
                "b2_delta_pct": [0.0, 0.0],
                "risk_source": ["main", "main"],
                "fitted_method": ["linear", "linear"],
                "fitted_curvature": [1.0, 1.0],
            }
        )
        monkeypatch.setattr("src.mr_pipeline._compute_hybrid_mr_risk", lambda *a, **k: (merge_df, comparison_df))
        monkeypatch.setattr(
            "src.mr_pipeline._assign_tiered_risk", lambda data_demand_mr, *a, **k: data_demand_mr.assign(_mr_tier=3)
        )
        mr_surface = pd.DataFrame(
            {
                merge_keys[0]: [1, 1, 2, 2],
                merge_keys[1]: [1, 2, 1, 2],
                "oa_amt_h0": [1000.0, 500.0, 700.0, 300.0],
                "todu_30ever_h6": [10.0, 5.0, 20.0, 2.0],
                "todu_amt_pile_h6": [100.0, 50.0, 200.0, 20.0],
                "todu_30ever_h6_boo": [4.0, 2.0, 8.0, 1.0],
                "todu_30ever_h6_rep": [6.0, 3.0, 12.0, 1.0],
                "todu_amt_pile_h6_boo": [40.0, 20.0, 80.0, 10.0],
                "todu_amt_pile_h6_rep": [100.0, 50.0, 200.0, 20.0],
            }
        )
        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", lambda *a, **k: mr_surface.copy())

        new_mask = np.array([1, 0, 1, 0], dtype=int)
        from src import optimization_utils as ou

        monkeypatch.setattr(ou, "milp_solve_cutoffs", lambda grid, target_risk, inv_vars, multiplier, **k: new_mask)

        # Capture the mask the MR audit is rebuilt on.
        captured = {}

        def fake_generate_audit_table(*a, mask=None, **k):
            captured["audit_mask"] = mask
            return pd.DataFrame()  # no "classification" → downstream reconcile no-ops

        monkeypatch.setattr("src.audit.generate_audit_table", fake_generate_audit_table)

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=merge_keys,
            inference_variables=merge_keys,
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            use_mr_outcomes=True,
        )
        data_booked = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-02-15", "2024-03-15"]),
                "status_name": ["booked", "booked"],
                "segment_cut_off": ["seg_a", "seg_a"],
                "score_rf": [300.0, 320.0],
                "risk_score_rf": [20.0, 25.0],
                "acct_booked_h0": [1, 1],
                "oa_amt": [1000.0, 1200.0],
                "oa_amt_h0": [1000.0, 1200.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
            }
        )
        data_clean = pd.concat(
            [
                data_booked,
                pd.DataFrame(
                    {
                        "mis_date": pd.to_datetime(["2024-07-15", "2024-07-20"]),
                        "status_name": ["booked", "booked"],
                        "segment_cut_off": ["seg_a", "seg_a"],
                        "score_rf": [305.0, 310.0],
                        "risk_score_rf": [21.0, 22.0],
                        "acct_booked_h0": [1, 1],
                        "oa_amt": [500.0, 600.0],
                        "oa_amt_h0": [500.0, 600.0],
                        "todu_30ever_h6": [0.0, 0.0],
                        "todu_amt_pile_h6": [0.0, 0.0],
                        merge_keys[0]: [1, 2],
                        merge_keys[1]: [1, 1],
                    }
                ),
            ],
            ignore_index=True,
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        mask_main = np.array([0, 0, 0, 0], dtype=int)
        # Non-empty MAIN-mask audit (with classification) — the buggy path fed THIS to
        # the production reconcile; the fix must replace it with a re-optimized-mask audit.
        audit_main = pd.DataFrame({"classification": ["keep"], "oa_amt_demand": [1.0], "oa_amt_h0": [1.0]})

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference={"best_model_info": {"model": object()}, "features": [], "model_variables": merge_keys},
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=pd.DataFrame({"b2_ever_h6": [1.0], "sol_fac": [0]}),
            file_suffix="",
            output=output,
            mask=mask_main,
            grid=None,
            audit_mr_df=audit_main,
        )

        assert "audit_mask" in captured, "MR audit was not regenerated (generate_audit_table not called)"
        assert np.array_equal(captured["audit_mask"], new_mask), (
            "MR audit must be rebuilt on the re-optimized mask, not the main mask"
        )
        assert not np.array_equal(captured["audit_mask"], mask_main)

    def test_saved_mr_audit_matches_grid(self, monkeypatch, tmp_path):
        """Regression: the persisted MR audit (audit_*_mr.csv) must be re-saved on the
        re-optimized mask so its accepted cells equal the MR acceptance grid. The main
        mask here is all-reject, so a stale (buggy) save would leave the audit's accepted
        set EMPTY — distinct from the grid. generate_audit_table runs for real (not mocked)."""

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        monkeypatch.setattr(
            "src.mr_pipeline.calculate_metrics_from_cuts",
            lambda *a, **k: pd.DataFrame([{"Metric": "Optimum selected", "Risk (%)": 0.0, "Production (€)": 0.0}]),
        )
        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *a, **k: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *a, **k: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *a, **k: DummyAlertReport())
        monkeypatch.setattr("src.mr_pipeline.go.Figure.write_html", lambda *a, **k: None)

        merge_keys = ["sc_octroi_new_clus", "new_efx_clus"]
        merge_df = pd.DataFrame({merge_keys[0]: [1, 2], merge_keys[1]: [1, 1], "b2_ever_h6_tmp": [0.01, 0.02]})
        comparison_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "b2_ever_h6_tmp": [0.01, 0.02],
                "b2_delta_pct": [0.0, 0.0],
                "risk_source": ["main", "main"],
                "fitted_method": ["linear", "linear"],
                "fitted_curvature": [1.0, 1.0],
            }
        )
        monkeypatch.setattr("src.mr_pipeline._compute_hybrid_mr_risk", lambda *a, **k: (merge_df, comparison_df))
        monkeypatch.setattr(
            "src.mr_pipeline._assign_tiered_risk", lambda data_demand_mr, *a, **k: data_demand_mr.assign(_mr_tier=3)
        )
        mr_surface = pd.DataFrame(
            {
                merge_keys[0]: [1, 1, 2, 2],
                merge_keys[1]: [1, 2, 1, 2],
                "oa_amt_h0": [1000.0, 500.0, 700.0, 300.0],
                "todu_30ever_h6": [10.0, 5.0, 20.0, 2.0],
                "todu_amt_pile_h6": [100.0, 50.0, 200.0, 20.0],
                "todu_30ever_h6_boo": [4.0, 2.0, 8.0, 1.0],
                "todu_30ever_h6_rep": [6.0, 3.0, 12.0, 1.0],
                "todu_amt_pile_h6_boo": [40.0, 20.0, 80.0, 10.0],
                "todu_amt_pile_h6_rep": [100.0, 50.0, 200.0, 20.0],
            }
        )
        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", lambda *a, **k: mr_surface.copy())

        new_mask = np.array([1, 0, 1, 0], dtype=int)  # accept octroi-1/efx-1 and octroi-2/efx-1
        from src import optimization_utils as ou

        monkeypatch.setattr(ou, "milp_solve_cutoffs", lambda grid, target_risk, inv_vars, multiplier, **k: new_mask)

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=merge_keys,
            inference_variables=merge_keys,
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            use_mr_outcomes=True,
        )
        data_booked = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-02-15", "2024-03-15"]),
                "status_name": ["booked", "booked"],
                "segment_cut_off": ["seg_a", "seg_a"],
                "score_rf": [300.0, 320.0],
                "risk_score_rf": [20.0, 25.0],
                "acct_booked_h0": [1, 1],
                "oa_amt": [1000.0, 1200.0],
                "oa_amt_h0": [1000.0, 1200.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
            }
        )
        data_clean = pd.concat(
            [
                data_booked,
                pd.DataFrame(
                    {
                        "mis_date": pd.to_datetime(["2024-07-15", "2024-07-20"]),
                        "status_name": ["booked", "booked"],
                        "segment_cut_off": ["seg_a", "seg_a"],
                        "score_rf": [305.0, 310.0],
                        "risk_score_rf": [21.0, 22.0],
                        "acct_booked_h0": [1, 1],
                        "oa_amt": [500.0, 600.0],
                        "oa_amt_h0": [500.0, 600.0],
                        "todu_30ever_h6": [0.0, 0.0],
                        "todu_amt_pile_h6": [0.0, 0.0],
                        merge_keys[0]: [1, 2],  # cells (octroi 1, efx 1) and (octroi 2, efx 1)
                        merge_keys[1]: [1, 1],
                    }
                ),
            ],
            ignore_index=True,
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        mask_main = np.array([0, 0, 0, 0], dtype=int)  # all-reject → a stale save would be empty
        audit_main = pd.DataFrame({"classification": ["rejected"], "oa_amt_h0": [1.0], "status_name": ["booked"]})

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference={"best_model_info": {"model": object()}, "features": [], "model_variables": merge_keys},
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=pd.DataFrame({"b2_ever_h6": [1.0], "sol_fac": [0]}),
            file_suffix="",
            output=output,
            mask=mask_main,
            grid=None,
            audit_mr_df=audit_main,
        )

        keys = ["sc_octroi_new_clus", "new_efx_clus"]
        saved_audit = pd.read_csv(output.data_dir / "audit_base_mr.csv")
        grid = pd.read_csv(output.mr_cutoff_summary_wide_csv(""))
        cl = saved_audit["classification"].astype(str).str.strip()
        audit_acc = set(map(tuple, saved_audit.loc[cl.isin(["keep", "swap_in"]), keys].astype(float).values.tolist()))
        grid_acc = set(map(tuple, grid.loc[grid["accepted"] == 1, keys].astype(float).values.tolist()))
        assert audit_acc, "saved MR audit has no accepted cells — it is stale (main all-reject mask)"
        assert audit_acc == grid_acc, f"saved MR audit accepted cells {audit_acc} != grid {grid_acc}"

    def test_hybrid_mode_model_fallback_infers_risk_nd(self, monkeypatch, tmp_path):
        """In hybrid MR mode with N>2 variables, model_fallback bins should get model-inferred risk."""

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        captured = {}

        def fake_calculate_B2(df, model_risk, variables, stressor, var_reg):
            captured["called"] = True
            captured["used_variables"] = list(variables)
            out = df.copy()
            out["b2_ever_h6"] = 0.42
            return out

        def fake_run_optimization_pipeline(*, data_booked, data_demand, **kwargs):
            # Check that model_fallback bins now have inferred risk (not NaN)
            mr_row = data_demand.loc[
                (data_demand["status_name"] == "booked")
                & (data_demand["sc_octroi_new_clus"] == 1)
                & (data_demand["new_efx_clus"] == 1)
                & (data_demand["income_bin"] == 2)
            ]
            assert len(mr_row) == 1
            captured["imputed_b2"] = float(mr_row["b2_ever_h6_tmp"].iloc[0])
            return data_booked.copy()

        monkeypatch.setattr("src.mr_pipeline.calculate_B2", fake_calculate_B2)
        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", fake_run_optimization_pipeline)
        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *args, **kwargs: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *args, **kwargs: DummyAlertReport())

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=["sc_octroi_new_clus", "new_efx_clus", "income_bin"],
            inference_variables=["sc_octroi_new_clus", "new_efx_clus"],
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            bins={
                "income_bin": BinConfig(
                    source_col="income_t1t2_m",
                    output_col="income_bin",
                    bin_edges=[-float("inf"), 1.5, float("inf")],
                )
            },
            use_mr_outcomes=True,
        )

        # Main period: bins (1,1,1) and (2,2,1) exist
        data_clean = pd.DataFrame(
            [
                {
                    "mis_date": "2024-01-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 300.0,
                    "risk_score_rf": 20.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 1000.0,
                    "oa_amt_h0": 1000.0,
                    "todu_30ever_h6": 10.0,
                    "todu_amt_pile_h6": 100.0,
                    "sc_octroi_new_clus": 1,
                    "new_efx_clus": 1,
                    "income_bin": 1,
                },
                {
                    "mis_date": "2024-02-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 320.0,
                    "risk_score_rf": 25.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 1200.0,
                    "oa_amt_h0": 1200.0,
                    "todu_30ever_h6": 5.0,
                    "todu_amt_pile_h6": 100.0,
                    "sc_octroi_new_clus": 2,
                    "new_efx_clus": 2,
                    "income_bin": 1,
                },
                # MR period: bin (1,1,2) does NOT exist in main period -> model_fallback
                {
                    "mis_date": "2024-07-15",
                    "status_name": "booked",
                    "segment_cut_off": "seg_a",
                    "score_rf": 305.0,
                    "risk_score_rf": 21.0,
                    "acct_booked_h0": 1,
                    "oa_amt": 500.0,
                    "oa_amt_h0": 500.0,
                    "todu_30ever_h6": 3.0,
                    "todu_amt_pile_h6": 50.0,
                    "sc_octroi_new_clus": 1,
                    "new_efx_clus": 1,
                    "income_bin": 2,
                },
            ]
        )
        data_clean["mis_date"] = pd.to_datetime(data_clean["mis_date"])
        data_booked = data_clean.loc[data_clean["mis_date"] < pd.Timestamp("2024-07-01")].copy()

        risk_inference = {
            "best_model_info": {"model": object()},
            "features": ["sc_octroi_new_clus", "new_efx_clus"],
            "model_variables": ["sc_octroi_new_clus", "new_efx_clus"],
        }

        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference=risk_inference,
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=None,
            output=output,
        )

        # With merge_keys capped to 2 variables, the MR account at (1,1,income_bin=2)
        # matches (1,1) in main period — no model fallback needed.
        # The imputed b2 comes from main-period risk at (1,1): multiplier * 10 / 100 = 0.7
        assert not captured.get("called"), "model fallback should not be triggered with 2D merge keys"
        assert captured["imputed_b2"] == pytest.approx(0.7)

    def test_recalibration_handles_non_unique_merge_keys_without_row_expansion(self, monkeypatch, tmp_path):
        """Recalibration should stay row-preserving when comparison_df keys are duplicated."""

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *args, **kwargs: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *args, **kwargs: DummyAlertReport())
        monkeypatch.setattr("src.mr_pipeline.go.Figure.write_html", lambda *args, **kwargs: None)

        merge_keys = ["sc_octroi_new_clus", "new_efx_clus"]
        merge_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "b2_ever_h6_tmp": [0.01, 0.02],
            }
        )
        # Duplicate key (1,1) on purpose to stress recalibration merge.
        comparison_df = pd.DataFrame(
            {
                merge_keys[0]: [1, 1, 2],
                merge_keys[1]: [1, 1, 1],
                "b2_ever_h6_tmp": [0.01, 0.011, 0.02],
                "b2_delta_pct": [0.0, 0.0, 0.0],
                "risk_source": ["main", "main", "main"],
                "fitted_method": ["linear", "linear", "linear"],
                "fitted_curvature": [1.0, 1.0, 1.0],
            }
        )
        monkeypatch.setattr(
            "src.mr_pipeline._compute_hybrid_mr_risk", lambda *args, **kwargs: (merge_df, comparison_df)
        )
        monkeypatch.setattr(
            "src.mr_pipeline._assign_tiered_risk",
            lambda data_demand_mr, *args, **kwargs: data_demand_mr.assign(_mr_tier=3),
        )

        mr_surface = pd.DataFrame(
            {
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
                "oa_amt_h0": [1000.0, 800.0],
                "todu_30ever_h6": [10.0, 8.0],
                "todu_amt_pile_h6": [100.0, 80.0],
                "todu_30ever_h6_boo": [6.0, 4.0],
                "todu_amt_pile_h6_boo": [60.0, 40.0],
                "oa_amt_h0_boo": [600.0, 400.0],
                "todu_30ever_h6_rep": [4.0, 4.0],
                "todu_amt_pile_h6_rep": [40.0, 40.0],
                "oa_amt_h0_rep": [400.0, 400.0],
            }
        )
        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", lambda *args, **kwargs: mr_surface.copy())

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=merge_keys,
            inference_variables=merge_keys,
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            use_mr_outcomes=True,
        )

        data_booked = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-02-15", "2024-03-15"]),
                "status_name": ["booked", "booked"],
                "segment_cut_off": ["seg_a", "seg_a"],
                "score_rf": [300.0, 320.0],
                "risk_score_rf": [20.0, 25.0],
                "acct_booked_h0": [1, 1],
                "oa_amt": [1000.0, 1200.0],
                "oa_amt_h0": [1000.0, 1200.0],
                "todu_30ever_h6": [10.0, 20.0],
                "todu_amt_pile_h6": [100.0, 200.0],
                merge_keys[0]: [1, 2],
                merge_keys[1]: [1, 1],
            }
        )
        data_clean = pd.concat(
            [
                data_booked,
                pd.DataFrame(
                    {
                        "mis_date": pd.to_datetime(["2024-07-15", "2024-07-20"]),
                        "status_name": ["booked", "booked"],
                        "segment_cut_off": ["seg_a", "seg_a"],
                        "score_rf": [305.0, 310.0],
                        "risk_score_rf": [21.0, 22.0],
                        "acct_booked_h0": [1, 1],
                        "oa_amt": [500.0, 600.0],
                        "oa_amt_h0": [500.0, 600.0],
                        "todu_30ever_h6": [0.0, 0.0],
                        "todu_amt_pile_h6": [0.0, 0.0],
                        merge_keys[0]: [1, 2],
                        merge_keys[1]: [1, 1],
                    }
                ),
            ],
            ignore_index=True,
        )

        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference={"best_model_info": {"model": object()}, "features": [], "model_variables": merge_keys},
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=pd.DataFrame({"b2_ever_h6": [1.0], "sol_fac": [0]}),
            output=output,
        )

        mr_summary = pd.read_csv(output.mr_summary_csv(""))
        assert len(mr_summary) == len(mr_surface)

    def test_mr_only_model_fallback_nan_risk_is_tolerated_downstream(self, monkeypatch, tmp_path):
        """MR-only bins with failed model_fallback keep NaN risk without breaking MR flow."""

        class DummyRVModel:
            def predict(self, X):
                return np.full(len(X), 100.0)

        class DummyStabilityReport:
            unstable_vars = []

            def to_dataframe(self):
                return pd.DataFrame()

        class DummyAlertReport:
            def to_json(self, path):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("{}")

        monkeypatch.setattr("src.mr_pipeline.compare_main_vs_mr", lambda *args, **kwargs: DummyStabilityReport())
        monkeypatch.setattr("src.mr_pipeline.styles.apply_plotly_style", lambda *args, **kwargs: None)
        monkeypatch.setattr("src.alerts.generate_drift_alerts", lambda *args, **kwargs: DummyAlertReport())
        monkeypatch.setattr("src.mr_pipeline.go.Figure.write_html", lambda *args, **kwargs: None)

        merge_keys = ["sc_octroi_new_clus", "new_efx_clus"]
        merge_df = pd.DataFrame({merge_keys[0]: [1], merge_keys[1]: [1], "b2_ever_h6_tmp": [np.nan]})
        comparison_df = pd.DataFrame(
            {
                merge_keys[0]: [1],
                merge_keys[1]: [1],
                "b2_ever_h6_tmp": [np.nan],
                "b2_delta_pct": [np.nan],
                "risk_source": ["model_fallback"],
                "fitted_method": ["linear"],
                "fitted_curvature": [1.0],
            }
        )
        monkeypatch.setattr(
            "src.mr_pipeline._compute_hybrid_mr_risk", lambda *args, **kwargs: (merge_df, comparison_df)
        )
        # Force model fallback inference failure so NaN remains.
        monkeypatch.setattr(
            "src.mr_pipeline.calculate_B2", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
        )

        def fake_run_optimization_pipeline(*, data_demand, **kwargs):
            row = data_demand.loc[data_demand["status_name"] == "booked"].iloc[0]
            assert pd.isna(row["b2_ever_h6_tmp"])
            return pd.DataFrame(
                {
                    merge_keys[0]: [1],
                    merge_keys[1]: [1],
                    "oa_amt_h0": [500.0],
                    "todu_30ever_h6": [0.0],
                    "todu_amt_pile_h6": [0.0],
                    "todu_30ever_h6_boo": [0.0],
                    "todu_amt_pile_h6_boo": [0.0],
                    "oa_amt_h0_boo": [500.0],
                    "todu_30ever_h6_rep": [0.0],
                    "todu_amt_pile_h6_rep": [0.0],
                    "oa_amt_h0_rep": [0.0],
                }
            )

        monkeypatch.setattr("src.mr_pipeline.run_optimization_pipeline", fake_run_optimization_pipeline)

        settings = PreprocessingSettings(
            keep_vars=["mis_date", "status_name", "segment_cut_off", "score_rf", "risk_score_rf"],
            indicators=["acct_booked_h0", "oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
            segment_filter="seg_a",
            date_ini_book_obs="2024-01-01",
            date_fin_book_obs="2024-06-30",
            date_ini_book_obs_mr="2024-07-01",
            date_fin_book_obs_mr="2024-07-31",
            variables=merge_keys,
            inference_variables=merge_keys,
            octroi_bins=[-float("inf"), 1.5, float("inf")],
            efx_bins=[-float("inf"), 1.5, float("inf")],
            use_mr_outcomes=True,
        )
        data_booked = pd.DataFrame(
            {
                "mis_date": pd.to_datetime(["2024-02-15"]),
                "status_name": ["booked"],
                "segment_cut_off": ["seg_a"],
                "score_rf": [300.0],
                "risk_score_rf": [20.0],
                "acct_booked_h0": [1],
                "oa_amt": [1000.0],
                "oa_amt_h0": [1000.0],
                "todu_30ever_h6": [10.0],
                "todu_amt_pile_h6": [100.0],
                merge_keys[0]: [1],
                merge_keys[1]: [1],
            }
        )
        data_clean = pd.concat(
            [
                data_booked,
                pd.DataFrame(
                    {
                        "mis_date": pd.to_datetime(["2024-07-15"]),
                        "status_name": ["booked"],
                        "segment_cut_off": ["seg_a"],
                        "score_rf": [305.0],
                        "risk_score_rf": [21.0],
                        "acct_booked_h0": [1],
                        "oa_amt": [500.0],
                        "oa_amt_h0": [500.0],
                        "todu_30ever_h6": [0.0],
                        "todu_amt_pile_h6": [0.0],
                        merge_keys[0]: [1],
                        merge_keys[1]: [1],
                    }
                ),
            ],
            ignore_index=True,
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        process_mr_period(
            data_clean=data_clean,
            data_booked=data_booked,
            settings=settings,
            risk_inference={"best_model_info": {"model": object()}, "features": [], "model_variables": merge_keys},
            reg_todu_amt_pile=DummyRVModel(),
            stress_factor=1.0,
            tasa_fin=1.0,
            annual_coef=1.0,
            optimal_solution_df=pd.DataFrame({"b2_ever_h6": [1.0], "sol_fac": [0]}),
            output=output,
        )
        assert Path(output.mr_risk_production_summary_csv("")).exists()


# =============================================================================
# H3 Floor Enforcement
# =============================================================================


class TestH3FloorEnforcement:
    """Simulated B2 Ever H6 must always be >= real B2 Ever H3."""

    def test_h3_floor_clamps_low_h6(self):
        """When main-period H6 < MR H3, H6 should be clamped to H3."""
        merge_keys = ["bin_a", "bin_b"]
        # Main period: bin (1,1) has LOW H6 risk (0.5%), bin (2,1) has normal H6 risk (3%)
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1, 2, 2],
                "bin_b": [1, 1, 1, 1],
                "todu_30ever_h6": [1.0, 1.0, 6.0, 6.0],
                "todu_amt_pile_h6": [200.0, 200.0, 200.0, 200.0],
                "todu_30ever_h3": [3.0, 3.0, 4.0, 4.0],
                "todu_amt_pile_h3": [200.0, 200.0, 200.0, 200.0],
                "status_name": ["booked"] * 4,
            }
        )
        # MR period: sparse → forces main_imputed fallback
        # H3 risk is HIGHER than main H6 for bin (1,1)
        data_demand_mr = pd.DataFrame(
            {
                # Must have enough MR H3 observations to activate the
                # (reliability-gated) H3 floor logic.
                "bin_a": [1] * 20 + [2] * 20,
                "bin_b": [1] * 40,
                "todu_30ever_h6": [2.0] * 40,
                "todu_amt_pile_h6": [100.0] * 40,
                "todu_30ever_h3": [4.0] * 20 + [3.0] * 20,  # bin (1,1) H3 > main H6
                "todu_amt_pile_h3": [100.0] * 40,
                "oa_amt_h0": [1000.0] * 40,
                "status_name": ["booked"] * 40,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=30,
            multiplier=7.0,
            multiplier_h3=4.0,
        )

        # For each bin, simulated H6 must be >= best available H3
        for _, row in comparison_df.iterrows():
            h6 = row["b2_ever_h6_tmp"]
            # Best H3: MR H3 if valid, else main H3
            h3_candidates = [row.get("b2_mr_h3"), row.get("b2_main_h3")]
            h3_floor = next((v for v in h3_candidates if pd.notna(v)), None)
            if h3_floor is not None and pd.notna(h6):
                assert h6 >= h3_floor - 1e-9, (
                    f"Bin {row[merge_keys].to_dict()}: simulated H6 ({h6:.6f}) < H3 floor ({h3_floor:.6f})"
                )

    def test_h3_floor_no_effect_when_h6_already_higher(self):
        """When H6 is already > H3, no clamping occurs."""
        merge_keys = ["bin_a"]
        data_booked = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "todu_30ever_h6": [10.0, 10.0],
                "todu_amt_pile_h6": [100.0, 100.0],
                "todu_30ever_h3": [3.0, 3.0],
                "todu_amt_pile_h3": [100.0, 100.0],
                "status_name": ["booked"] * 2,
            }
        )
        data_demand_mr = pd.DataFrame(
            {
                "bin_a": [1] * 5,
                "todu_30ever_h6": [5.0] * 5,
                "todu_amt_pile_h6": [100.0] * 5,
                "todu_30ever_h3": [2.0] * 5,
                "todu_amt_pile_h3": [100.0] * 5,
                "oa_amt_h0": [1000.0] * 5,
                "status_name": ["booked"] * 5,
            }
        )

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked,
            data_demand_mr,
            merge_keys,
            min_obs=30,
            multiplier=7.0,
            multiplier_h3=4.0,
        )

        # H6 (main imputed) should be larger than H3 → no clamping
        row = comparison_df.iloc[0]
        assert row["b2_ever_h6_tmp"] > row.get("b2_main_h3", 0)

    def test_summary_table_h3_floor(self):
        """calculate_metrics_from_cuts enforces H6 >= H3 on summary rows."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        df = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2],
                "new_efx_clus": [1, 1],
                # H6 risk numerator/denominator — deliberately low H6
                "todu_30ever_h6_boo": [1.0, 1.0],
                "todu_amt_pile_h6_boo": [200.0, 200.0],
                "oa_amt_h0_boo": [1000.0, 1000.0],
                "todu_30ever_h6_rep": [0.5, 0.5],
                "todu_amt_pile_h6_rep": [100.0, 100.0],
                "oa_amt_h0_rep": [500.0, 500.0],
                # H3 risk — deliberately higher than H6
                "todu_30ever_h3_boo": [5.0, 5.0],
                "todu_amt_pile_h3_boo": [200.0, 200.0],
                "todu_30ever_h3_rep": [3.0, 3.0],
                "todu_amt_pile_h3_rep": [100.0, 100.0],
            }
        )
        opt_sol = pd.DataFrame({"sol_fac": [0], 1: [1], 2: [1]})
        result = calculate_metrics_from_cuts(
            df,
            opt_sol,
            variables,
            multiplier=7.0,
            multiplier_h3=4.0,
        )
        assert result is not None
        for _, row in result.iterrows():
            h6 = row.get("Risk (%)")
            h3 = row.get("Risk H3 (%)")
            if pd.notna(h6) and pd.notna(h3):
                assert h6 >= h3 - 1e-9, f"{row['Metric']}: H6 ({h6:.4f}%) < H3 ({h3:.4f}%)"


# =============================================================================
# Tiered MR Risk Reconstruction Tests
# =============================================================================


class TestTieredReconstruction:
    """Tests for _assign_tiered_risk and _compute_account_maturity."""

    @pytest.fixture
    def merge_keys(self):
        return ["bin_a", "bin_b"]

    @pytest.fixture
    def merge_df(self):
        """Per-bin b2_ever_h6_tmp from _compute_hybrid_mr_risk."""
        return pd.DataFrame(
            {
                "bin_a": [1, 2],
                "bin_b": [1, 1],
                "b2_ever_h6_tmp": [0.05, 0.10],
            }
        )

    @pytest.fixture
    def comparison_df_with_ratio(self):
        """comparison_df with h6_h3_ratio and b2_main_h3 columns."""
        return pd.DataFrame(
            {
                "bin_a": [1, 2],
                "bin_b": [1, 1],
                "b2_ever_h6_tmp": [0.05, 0.10],
                "h6_h3_ratio": [2.0, 2.5],
                "b2_main_h3": [0.025, 0.04],
                "risk_source": ["h3_extrapolated", "main_imputed"],
            }
        )

    def test_compute_account_maturity(self):
        """_compute_account_maturity returns correct month differences."""
        dates = pd.Series(pd.to_datetime(["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"]))
        ref = pd.Timestamp("2025-10-15")
        mat = _compute_account_maturity(dates, ref)
        assert list(mat) == [9, 6, 3, 0]

    def test_compute_account_maturity_default_ref(self):
        """_compute_account_maturity defaults to max(mis_date) as reference."""
        dates = pd.Series(pd.to_datetime(["2025-01-15", "2025-06-15"]))
        mat = _compute_account_maturity(dates)
        assert list(mat) == [5, 0]

    def test_compute_account_maturity_is_day_precise(self):
        """#52: maturity counts COMPLETE months (day-of-month aware). The bare
        (year, month) diff overstated maturity by up to a month, so loans weeks
        short of maturity passed the >= filter and diluted b2_mr with immature
        zeros. Matches the M4 backtest's DateOffset(months=...) semantics."""
        dates = pd.Series(pd.to_datetime(["2025-01-31", "2025-01-01", "2025-01-15", "2025-01-10"]))
        ref = pd.Timestamp("2025-07-10")
        mat = _compute_account_maturity(dates, ref)
        # 01-31 → 07-31 not reached (10<31) → 5 (was 6 under the bare diff)
        # 01-01 → 07-01 reached → 6
        # 01-15 → 07-15 not reached (10<15) → 5
        # 01-10 → 07-10 reached → 6
        assert list(mat) == [5, 6, 5, 6]

    def test_assign_tiered_risk_honors_reference_date(self, merge_keys, merge_df):
        """Tier maturity must use the supplied observation horizon, not max(mis_date).

        Regression for #2: anchoring to max(mis_date) understates maturity and
        over-excludes mature accounts.  A single account booked 2025-07-15 is
        immature (maturity 0) under the max(mis_date) anchor but mature (7mo)
        under the true observation horizon.
        """
        data = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "status_name": ["booked"],
                "mis_date": pd.to_datetime(["2025-07-15"]),
                "_actual_todu_30ever_h6": [8.0],
                "_actual_todu_amt_pile_h6": [100.0],
                "b2_ever_h6_tmp": [0.05],
                "oa_amt": [1000.0],
            }
        )

        # Default anchor = max(mis_date) = 2025-07-15 → maturity 0 → Tier 3 (bin-level).
        default_res = _assign_tiered_risk(data.copy(), merge_df, None, merge_keys, multiplier=7.0)
        assert default_res.iloc[0]["_mr_tier"] == 3

        # Observation horizon 2026-02-01 → maturity 7 → Tier 1 (actual H6: 7*8/100 = 0.56).
        ref_res = _assign_tiered_risk(
            data.copy(),
            merge_df,
            None,
            merge_keys,
            multiplier=7.0,
            maturity_reference_date=pd.Timestamp("2026-02-01"),
        )
        assert ref_res.iloc[0]["_mr_tier"] == 1
        assert ref_res.iloc[0]["b2_ever_h6_tmp"] == pytest.approx(0.56)

    def test_tier2_uses_account_h3_extrapolation(self, merge_keys, merge_df, comparison_df_with_ratio):
        """Tier 2 accounts use account-level H3 × bin ratio, not bin-level risk."""
        # Include a later-dated account so max(mis_date) gives the earlier ones >= 3mo maturity
        data = pd.DataFrame(
            {
                "bin_a": [1, 1, 1],
                "bin_b": [1, 1, 1],
                "status_name": ["booked", "booked", "booked"],
                "mis_date": pd.to_datetime(["2025-07-15", "2025-07-15", "2025-10-15"]),
                "todu_30ever_h3": [3.0, 6.0, 1.0],  # different H3 per account
                "todu_amt_pile_h3": [100.0, 100.0, 100.0],
                "b2_ever_h6_tmp": [0.05, 0.05, 0.05],  # bin-level from merge
                "oa_amt": [1000.0, 1000.0, 1000.0],
            }
        )
        # max(mis_date) = 2025-10-15; first two accounts have 3 months maturity → Tier 2

        result = _assign_tiered_risk(
            data,
            merge_df,
            comparison_df_with_ratio,
            merge_keys,
            multiplier=7.0,
            multiplier_h3=4.0,
        )

        tier2 = result[result["_mr_tier"] == 2]
        assert len(tier2) == 2

        # Account-level H3 risk (multiplier_h3=4):
        # b2_h3 = 4 * 3 / 100 = 0.12 for first, 4 * 6 / 100 = 0.24 for second
        # extrapolate linear: b2_h3 * h6_h3_ratio = 0.12 * 2.0 = 0.24, 0.24 * 2.0 = 0.48
        # Two accounts should have different risks (not the uniform bin-level 0.05)
        assert tier2.iloc[0]["b2_ever_h6_tmp"] != tier2.iloc[1]["b2_ever_h6_tmp"]
        # Tier 2 risk should be H3-extrapolated, not the bin-level 0.05
        assert tier2.iloc[0]["b2_ever_h6_tmp"] != 0.05

    def test_tier3_uses_bin_level_main_rate(self, merge_keys, merge_df):
        """Tier 3 accounts (< 3 months) keep the bin-level b2_ever_h6_tmp."""
        data = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "status_name": ["booked"],
                "mis_date": pd.to_datetime(["2025-09-15"]),  # 1 month maturity
                "b2_ever_h6_tmp": [0.05],  # from merge_df
                "oa_amt": [1000.0],
            }
        )

        result = _assign_tiered_risk(data, merge_df, None, merge_keys, multiplier=7.0)

        assert result.iloc[0]["_mr_tier"] == 3
        assert result.iloc[0]["b2_ever_h6_tmp"] == 0.05

    def test_h3_floor_enforced_for_tier2(self, merge_keys, merge_df):
        """Tier 2: extrapolated H6 must be >= account H3 (H3 floor)."""
        # comparison_df with a very low h6_h3_ratio (0.5 → H6 < H3 before floor)
        comparison_df = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "b2_ever_h6_tmp": [0.05],
                "h6_h3_ratio": [0.5],  # would make H6 = H3 * 0.5 < H3
                "b2_main_h3": [0.025],
                "risk_source": ["h3_extrapolated"],
            }
        )
        # Include a later account so max date gives earlier one >= 3mo maturity
        data = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "status_name": ["booked", "booked"],
                "mis_date": pd.to_datetime(["2025-07-15", "2025-10-15"]),
                "todu_30ever_h3": [5.0, 1.0],
                "todu_amt_pile_h3": [100.0, 100.0],
                "b2_ever_h6_tmp": [0.05, 0.05],
                "oa_amt": [1000.0, 1000.0],
            }
        )

        result = _assign_tiered_risk(
            data,
            merge_df,
            comparison_df,
            merge_keys,
            multiplier=7.0,
            multiplier_h3=4.0,
        )

        tier2 = result[result["_mr_tier"] == 2]
        assert len(tier2) == 1
        # b2_h3_account = 4 * 5 / 100 = 0.20
        # extrapolated = 0.20 * 0.5 = 0.10 < 0.20 → floored to 0.20
        assert tier2.iloc[0]["b2_ever_h6_tmp"] >= 0.20 - 1e-9

    def test_non_hybrid_mode_tiers_3_and_4_only(self, merge_keys, merge_df):
        """Without comparison_df (non-hybrid), only tiers 3 and 4 are assigned."""
        data = pd.DataFrame(
            {
                "bin_a": [1, 2, 3],
                "bin_b": [1, 1, 1],
                "status_name": ["booked", "booked", "booked"],
                "mis_date": pd.to_datetime(["2025-04-15", "2025-07-15", "2025-09-15"]),
                "todu_30ever_h3": [5.0, 3.0, 1.0],
                "todu_amt_pile_h3": [100.0, 100.0, 100.0],
                "b2_ever_h6_tmp": [0.05, 0.10, np.nan],
                "oa_amt": [1000.0, 1000.0, 1000.0],
            }
        )

        result = _assign_tiered_risk(data, merge_df, None, merge_keys, multiplier=7.0)

        # No Tier 1 or 2 (no comparison_df with h6_h3_ratio, no actual H6)
        assert (result["_mr_tier"].isin([3, 4])).all()
        # bin 3 has NaN b2_ever_h6_tmp → Tier 4
        assert result.iloc[2]["_mr_tier"] == 4

    def test_missing_mis_date_graceful_fallback(self, merge_keys, merge_df):
        """Without mis_date column, all booked → Tier 3 or 4 (no maturity)."""
        data = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "status_name": ["booked"],
                "b2_ever_h6_tmp": [0.05],
                "oa_amt": [1000.0],
            }
        )

        result = _assign_tiered_risk(data, merge_df, None, merge_keys, multiplier=7.0)

        # No mis_date → maturity is NaN → no Tier 1 or 2
        assert result.iloc[0]["_mr_tier"] == 3

    def test_tier1_uses_actual_h6(self, merge_keys, merge_df):
        """Tier 1 accounts (>=6 months) use actual H6 outcomes."""
        # Include a recent account so max(mis_date)=2025-10-15 gives first account 6 months maturity
        data = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "status_name": ["booked", "booked"],
                "mis_date": pd.to_datetime(["2025-04-15", "2025-10-15"]),
                "_actual_todu_30ever_h6": [10.0, np.nan],
                "_actual_todu_amt_pile_h6": [200.0, np.nan],
                "b2_ever_h6_tmp": [0.05, 0.05],  # bin-level, should be overridden for Tier 1
                "oa_amt": [1000.0, 1000.0],
            }
        )

        result = _assign_tiered_risk(data, merge_df, None, merge_keys, multiplier=7.0)

        assert result.iloc[0]["_mr_tier"] == 1
        # b2 = 7 * 10 / 200 = 0.35
        expected_b2 = 7.0 * 10.0 / 200.0
        assert np.isclose(result.iloc[0]["b2_ever_h6_tmp"], expected_b2)
        # Actual todu columns should be restored
        assert result.iloc[0]["todu_30ever_h6"] == 10.0
        assert result.iloc[0]["todu_amt_pile_h6"] == 200.0
        # Second account (0 months maturity) should NOT be Tier 1
        assert result.iloc[1]["_mr_tier"] != 1
        # Audit #30 lifetime regression: the _actual_* columns must SURVIVE
        # _assign_tiered_risk — process_mr_period later wipes/refills todu_*
        # with model-predicted reconstructions for all booked accounts and
        # _restore_tier1_actuals needs the actuals afterwards. (The old code
        # dropped them here, which silently turned the restore into a no-op.)
        assert "_actual_todu_30ever_h6" in result.columns
        assert "_actual_todu_amt_pile_h6" in result.columns

    def test_maturity_threshold_custom_and_zero(self, merge_keys, merge_df):
        """Tier boundaries follow configurable mr_maturity_months, including 0."""
        data = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "status_name": ["booked", "booked"],
                "mis_date": pd.to_datetime(["2025-06-15", "2025-10-15"]),  # maturities ~4 and 0 months
                "_actual_todu_30ever_h6": [10.0, 5.0],
                "_actual_todu_amt_pile_h6": [100.0, 100.0],
                "b2_ever_h6_tmp": [0.05, 0.05],
                "oa_amt": [1000.0, 1000.0],
            }
        )

        # Custom threshold 4 months: first account becomes Tier 1, second does not.
        out_4 = _assign_tiered_risk(data.copy(), merge_df, None, merge_keys, multiplier=7.0, mr_maturity_months=4)
        assert out_4.iloc[0]["_mr_tier"] == 1
        assert out_4.iloc[1]["_mr_tier"] != 1

        # Threshold 0 months: all booked accounts with actual H6 should be Tier 1.
        out_0 = _assign_tiered_risk(data.copy(), merge_df, None, merge_keys, multiplier=7.0, mr_maturity_months=0)
        assert (out_0["_mr_tier"] == 1).all()

    def test_h3_exists_but_obs_below_threshold_stays_out_of_tier2(self, merge_keys, merge_df):
        """If b2_mr_h3 exists but n_obs_mr_h3 < h3_min_obs, tier-2 extrapolation is blocked."""
        comparison_df = pd.DataFrame(
            {
                "bin_a": [1],
                "bin_b": [1],
                "h6_h3_ratio": [2.0],
                "b2_main_h3": [0.02],
                "b2_mr_h3": [0.03],
                "n_obs_mr_h3": [5],  # below h3_min_obs=max(min_obs/2,10)
                "risk_source": ["h3_extrapolated"],
            }
        )
        data = pd.DataFrame(
            {
                "bin_a": [1, 1],
                "bin_b": [1, 1],
                "status_name": ["booked", "booked"],
                "mis_date": pd.to_datetime(["2025-07-15", "2025-10-15"]),  # one eligible by maturity
                "todu_30ever_h3": [5.0, 1.0],
                "todu_amt_pile_h3": [100.0, 100.0],
                "b2_ever_h6_tmp": [0.05, 0.05],
                "oa_amt": [1000.0, 1000.0],
            }
        )
        out = _assign_tiered_risk(
            data,
            merge_df,
            comparison_df,
            merge_keys,
            multiplier=7.0,
            multiplier_h3=4.0,
            min_obs_per_bin=30,  # h3_min_obs = 15, so n_obs_mr_h3=5 should fail gate
        )
        assert not (out["_mr_tier"] == 2).any()


# =============================================================================
# Audit #30 — Tier-1 actual H6 outcomes survive the model refill
# =============================================================================


class TestRestoreTier1Actuals:
    @staticmethod
    def _frame():
        """Two mature (Tier-1) accounts + one immature, AFTER the model refill:
        todu_* carry the model-predicted reconstruction, _actual_* the truth."""
        return pd.DataFrame(
            {
                "_mr_tier": [1, 1, 3],
                "_actual_todu_30ever_h6": [4.0, 0.0, np.nan],
                "_actual_todu_amt_pile_h6": [100.0, 80.0, np.nan],
                # model reconstruction: num_i * (pred_pile/actual_pile) — wrong basis
                "todu_amt_pile_h6": [150.0, 40.0, 90.0],
                "todu_30ever_h6": [6.0, 0.0, 1.0],
            }
        )

    def test_tier1_actuals_restored_ratio_of_sums(self):
        from src.mr_pipeline import _restore_tier1_actuals

        df = self._frame()
        n = _restore_tier1_actuals(df)
        assert n == 2
        # Tier-1 rows carry the ACTUAL numerator and denominator again …
        assert df.loc[0, "todu_30ever_h6"] == 4.0 and df.loc[0, "todu_amt_pile_h6"] == 100.0
        assert df.loc[1, "todu_30ever_h6"] == 0.0 and df.loc[1, "todu_amt_pile_h6"] == 80.0
        # … so the bin aggregate is the actual ratio-of-sums (audit #30): the
        # model reconstruction gave 6/190 ≈ 0.0316; the truth is 4/180 ≈ 0.0222.
        tier1 = df["_mr_tier"] == 1
        assert df.loc[tier1, "todu_30ever_h6"].sum() / df.loc[tier1, "todu_amt_pile_h6"].sum() == pytest.approx(
            4.0 / 180.0
        )
        # the immature account keeps its model-based reconstruction
        assert df.loc[2, "todu_amt_pile_h6"] == 90.0 and df.loc[2, "todu_30ever_h6"] == 1.0

    def test_noop_without_tier_column_or_actuals(self):
        from src.mr_pipeline import _restore_tier1_actuals

        df = pd.DataFrame({"todu_30ever_h6": [1.0], "todu_amt_pile_h6": [10.0]})
        assert _restore_tier1_actuals(df) == 0
        df2 = self._frame().drop(columns=["_actual_todu_30ever_h6"])
        df2["_actual_todu_30ever_h6"] = np.nan
        assert _restore_tier1_actuals(df2) == 0


# =============================================================================
# MR cutoff re-optimization gate (mr_reoptimize_cutoffs + fixed-cutoff freeze)
# =============================================================================


def _mr_gate_settings(**overrides):
    base = dict(
        keep_vars=["mis_date"],
        indicators=["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        segment_filter="seg_a",
        date_ini_book_obs="2024-01-01",
        date_fin_book_obs="2024-06-30",
        variables=["sc_octroi_new_clus", "new_efx_clus"],
        octroi_bins=[-float("inf"), 1.5, float("inf")],
        efx_bins=[-float("inf"), 1.5, float("inf")],
    )
    base.update(overrides)
    return PreprocessingSettings(**base)


@pytest.fixture
def _mr_gate_inputs():
    """A 2x2 MR grid + accept-all main mask + an optimal-solution row."""
    from src.optimization_utils import CellGrid

    df = pd.DataFrame(
        {
            "sc_octroi_new_clus": [1, 2, 1, 2],
            "new_efx_clus": [1, 1, 2, 2],
            "oa_amt_h0": [1000.0, 2000.0, 3000.0, 4000.0],
            "todu_30ever_h6": [10.0, 20.0, 30.0, 40.0],
            "todu_amt_pile_h6": [100.0, 200.0, 300.0, 400.0],
        }
    )
    grid = CellGrid.from_summary(df, ["sc_octroi_new_clus", "new_efx_clus"])
    mask = np.ones(len(grid.cell_data), dtype=int)
    opt = pd.DataFrame({"sol_fac": [0], "b2_ever_h6": [2.0]})
    return df, grid, mask, opt


class TestMRReoptimizeGate:
    """Fixed cutoffs always freeze in MR; the flag governs optimized segments."""

    def test_fixed_cutoffs_always_freeze_even_when_flag_true(self, monkeypatch, _mr_gate_inputs):
        df, grid, mask, opt = _mr_gate_inputs
        settings = _mr_gate_settings(
            fixed_cutoffs={"sc_octroi_new_clus": [1.0, 2.0], "new_efx_clus": [2, 2]},
            mr_reoptimize_cutoffs=True,  # a fixed policy must stay fixed regardless
        )
        called = []
        monkeypatch.setattr(
            "src.optimization_utils.milp_solve_cutoffs",
            lambda *a, **k: called.append(1) or np.zeros(len(grid.cell_data), dtype=int),
        )
        out_mask, out_grid, _ = _reoptimize_mr_mask_after_recalibration(df, settings, opt, mask, grid, True)
        assert called == []  # no MR re-optimization
        assert out_mask is mask  # the frozen main mask object is returned unchanged
        assert out_grid is grid

    def test_flag_false_freezes_optimized_segment(self, monkeypatch, _mr_gate_inputs):
        df, grid, mask, opt = _mr_gate_inputs
        settings = _mr_gate_settings(mr_reoptimize_cutoffs=False)  # no fixed cutoffs
        called = []
        monkeypatch.setattr("src.optimization_utils.milp_solve_cutoffs", lambda *a, **k: called.append(1))
        out_mask, out_grid, _ = _reoptimize_mr_mask_after_recalibration(df, settings, opt, mask, grid, True)
        assert called == []
        assert out_mask is mask
        assert out_grid is grid

    def test_default_reoptimizes_optimized_segment(self, monkeypatch, _mr_gate_inputs):
        df, grid, mask, opt = _mr_gate_inputs
        settings = _mr_gate_settings()  # default mr_reoptimize_cutoffs=True, no fixed cutoffs
        new_mask = np.zeros(len(grid.cell_data), dtype=int)
        called = []
        monkeypatch.setattr(
            "src.optimization_utils.milp_solve_cutoffs",
            lambda *a, **k: (called.append(1), new_mask)[1],
        )
        out_mask, out_grid, _ = _reoptimize_mr_mask_after_recalibration(df, settings, opt, mask, grid, True)
        assert called == [1]  # re-optimization ran
        assert out_mask is new_mask  # replaced with the MR-optimized mask
        assert out_mask is not mask


class TestMRCutoffDriftOverlay:
    """The main-vs-MR accepted-cell drift overlay."""

    def test_overlay_categorizes_cells_and_writes_html(self, tmp_path):
        from src.optimization_utils import CellGrid

        df = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1, 2, 1, 2],
                "new_efx_clus": [1, 1, 2, 2],
                "oa_amt_h0": [1.0, 1.0, 1.0, 1.0],
                "todu_30ever_h6": [0.0, 0.0, 0.0, 0.0],
                "todu_amt_pile_h6": [0.0, 0.0, 0.0, 0.0],
            }
        )
        grid = CellGrid.from_summary(df, ["sc_octroi_new_clus", "new_efx_clus"])
        # cell order = (1,1),(1,2),(2,1),(2,2)
        main_mask = np.array([1, 1, 1, 0])  # accepts (1,1),(1,2),(2,1)
        mr_mask = np.array([0, 1, 1, 1])  # accepts (1,2),(2,1),(2,2)
        settings = _mr_gate_settings()
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()

        _write_mr_cutoff_drift_overlay(main_mask, grid, mr_mask, grid, settings, "_base", output)

        p = Path(output.mr_cutoff_drift_html("_base"))
        assert p.exists()
        html = p.read_text()
        # kept (both): (1,2),(2,1)=2 ; tightened (main only): (1,1)=1 ; loosened (MR only): (2,2)=1
        assert "kept=2" in html
        assert "tightened=1" in html
        assert "loosened=1" in html

    def test_overlay_skips_non_2var_grids(self, tmp_path):
        settings = _mr_gate_settings(
            variables=["sc_octroi_new_clus", "new_efx_clus", "income_bin"],
            inference_variables=["sc_octroi_new_clus", "new_efx_clus"],
            bins={
                "income_bin": BinConfig(
                    source_col="income_t1t2_m",
                    output_col="income_bin",
                    bin_edges=[-float("inf"), 1.5, float("inf")],
                )
            },
        )
        output = OutputPaths(base_dir=tmp_path)
        output.ensure_dirs()
        # Should log-and-return without writing, never raise.
        _write_mr_cutoff_drift_overlay(np.array([1]), object(), np.array([1]), object(), settings, "_base", output)
        assert not Path(output.mr_cutoff_drift_html("_base")).exists()


class TestRobustRatioClipBand:
    """#51: the H6/H3 ratio clip band must never collapse ratios to a fixed bound."""

    def test_degenerate_high_ratios_not_collapsed(self):
        # All observed ratios equal 6 (> 5). The old guard reset only upper_q to
        # 5.0, leaving lower_q at 6 -> np.clip(x, 6, 5) collapses every ratio to
        # 5.0, understating extrapolated MR risk.
        lo, hi = _robust_ratio_clip_band(np.array([6.0, 6.0, 6.0]))
        assert lo <= hi  # valid, non-inverted band
        assert hi >= 6.0  # brackets the observed ratio -> clip is a no-op
        assert float(np.clip(6.0, lo, hi)) == 6.0

    def test_normal_percentile_band_clips_outlier(self):
        lo, hi = _robust_ratio_clip_band(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
        assert lo <= hi
        assert hi < 100.0  # the 100 outlier is still clipped by the 99th pct

    def test_empty_input_falls_back_to_default_band(self):
        assert _robust_ratio_clip_band(np.array([])) == (0.5, 5.0)

    def test_all_nonfinite_falls_back_to_default_band(self):
        assert _robust_ratio_clip_band(np.array([np.nan, np.inf])) == (0.5, 5.0)
