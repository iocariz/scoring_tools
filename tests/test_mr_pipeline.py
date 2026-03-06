import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.mr_pipeline import _compute_hybrid_mr_risk, _mr_outcomes_available, calculate_metrics_from_cuts, process_mr_period
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

    def test_h3_extrapolation_replaces_direct_mr(self, merge_keys):
        """When H3 data exists and ratio is valid, bins use h3_extrapolated risk source."""
        # Main-period data with H3 columns — need enough observations per bin (>= min_obs)
        rng_main = np.random.RandomState(99)
        n_main = 35  # > min_obs=30
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate([rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate([rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]),
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

        merge_df, comparison_df = _compute_hybrid_mr_risk(
            data_booked, data_demand_mr, merge_keys, min_obs=30, multiplier_h3=4
        )

        # All bins should use h3_extrapolated (sufficient obs, valid ratio)
        assert (comparison_df["risk_source"] == "h3_extrapolated").all()

        # Verify the extrapolation formula: b2_mr_h3 * (b2_main / b2_main_h3)
        for _, row in comparison_df.iterrows():
            expected = row["b2_mr_h3"] * (row["b2_main"] / row["b2_main_h3"])
            assert np.isclose(row["b2_ever_h6_tmp"], expected), f"Expected {expected}, got {row['b2_ever_h6_tmp']}"

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
        # n_obs_mr_h3 should count only the 20 mature accounts, not all 40
        assert row["n_obs_mr_h3"] == n_mature
        # With only 20 H3 obs < min_obs=30, can't extrapolate → falls back to mr_observed
        assert row["risk_source"] == "mr_observed"

    def test_h3_partial_maturity_all_immature_falls_back(self, merge_keys):
        """When ALL MR accounts lack H3 (all from months 4-6), falls back to mr_observed."""
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
        # No H3 data at all → b2_mr_h3 is NaN → can't extrapolate → mr_observed
        assert row["risk_source"] == "mr_observed"
        assert np.isnan(row["n_obs_mr_h3"]) or row["n_obs_mr_h3"] == 0

    def test_h6_h3_ratio_in_comparison_df(self, merge_keys):
        """When H3 is available, h6_h3_ratio column appears in comparison_df with correct values."""
        rng_main = np.random.RandomState(99)
        n_main = 35  # > min_obs=30
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main,
                "bin_b": [1] * (2 * n_main),
                "todu_30ever_h6": np.concatenate([rng_main.uniform(8, 12, n_main), rng_main.uniform(4, 6, n_main)]),
                "todu_amt_pile_h6": np.concatenate([rng_main.uniform(90, 110, n_main), rng_main.uniform(190, 210, n_main)]),
                "todu_30ever_h3": np.concatenate([rng_main.uniform(4, 6, n_main), rng_main.uniform(2, 4, n_main)]),
                "todu_amt_pile_h3": np.concatenate([rng_main.uniform(80, 100, n_main), rng_main.uniform(170, 190, n_main)]),
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

        # Verify ratio = clipped(b2_main / b2_main_h3) within [0.5, 5.0]
        for _, row in comparison_df.iterrows():
            raw_ratio = row["b2_main"] / row["b2_main_h3"]
            expected_ratio = np.clip(raw_ratio, 0.5, 5.0)
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
        # Create data where H6 = 2 * H3 consistently → alpha ≈ 1
        data_booked = pd.DataFrame(
            {
                "bin_a": [1] * n_main + [2] * n_main + [3] * n_main + [4] * n_main,
                "bin_b": [1] * (4 * n_main),
                "todu_30ever_h6": np.concatenate([rng.uniform(9, 11, n_main) for _ in range(4)]),
                "todu_amt_pile_h6": np.concatenate([rng.uniform(90, 110, n_main) for _ in range(4)]),
                "todu_30ever_h3": np.concatenate([rng.uniform(4.5, 5.5, n_main) for _ in range(4)]),
                "todu_amt_pile_h3": np.concatenate([rng.uniform(90, 110, n_main) for _ in range(4)]),
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
            data_booked, data_demand_mr, merge_keys, min_obs=30,
            multiplier_h3=4, mr_extrapolation_method="auto",
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
            data_booked, data_demand_mr, merge_keys, min_obs=30,
            multiplier_h3=4, mr_extrapolation_method="auto",
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
            data_booked, data_demand_mr, merge_keys, min_obs=30,
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
        kpis = evaluate_solution(
            mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], multiplier=7
        )
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

        assert captured["used_variables"] == ["sc_octroi_new_clus", "new_efx_clus"]
        assert captured["used_features"] == ["sc_octroi_new_clus", "new_efx_clus"]
        assert captured["imputed_b2"] == pytest.approx(0.55)
