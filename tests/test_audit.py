"""Tests for the audit module."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.audit import (
    generate_audit_summary,
    generate_audit_table,
    reconcile_risk_production_summary_with_audit,
)
from src.constants import StatusName


class TestGenerateAuditTable:
    """Tests for the generate_audit_table function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame(
            {
                "authorization_id": [1, 2, 3, 4, 5],
                "status_name": ["booked", "booked", "rejected", "rejected", "rejected"],
                "reject_reason": [None, None, "09-score", "09-score", "08-other"],
                "risk_score_rf": [10, 20, 30, 40, 50],
                "score_rf": [100, 200, 300, 400, 500],
                "sc_octroi_new_clus": [1.0, 1.0, 2.0, 2.0, 2.0],
                "new_efx_clus": [3, 7, 4, 8, 4],
                "oa_amt_h0": [1000, 2000, 3000, 4000, 5000],
                "income_t1t2_m": [1500.0, 1600.0, 1700.0, 1800.0, 1900.0],
            }
        )

    @pytest.fixture
    def optimal_solution(self):
        """Create sample optimal solution."""
        return pd.DataFrame(
            {
                "sol_fac": [0],
                1.0: [5],  # Cutoff for bin 1.0
                2.0: [6],  # Cutoff for bin 2.0
            }
        )

    def test_basic_audit_table(self, sample_data, optimal_solution):
        """Test basic audit table generation."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        audit = generate_audit_table(sample_data, optimal_solution, variables)

        assert len(audit) == 5
        assert "classification" in audit.columns
        assert "cut_limit" in audit.columns
        assert "decision_source" in audit.columns
        assert "cell_key" in audit.columns
        assert "passes_cut" in audit.columns
        assert "reject_reason" in audit.columns
        assert "oa_amt_h0" in audit.columns
        assert "oa_amt_adjusted" in audit.columns
        assert "income_t1t2_m" in audit.columns
        assert audit["decision_source"].eq("cutoff").all()

    def test_classifications_correct(self, sample_data, optimal_solution):
        """Test that classifications are correct."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        audit = generate_audit_table(sample_data, optimal_solution, variables)

        # Record 1: booked, bin 1.0, efx 3 <= 5 -> keep
        assert audit.iloc[0]["classification"] == "keep"

        # Record 2: booked, bin 1.0, efx 7 > 5 -> swap_out
        assert audit.iloc[1]["classification"] == "swap_out"

        # Record 3: rejected 09-score, bin 2.0, efx 4 <= 6 -> swap_in
        assert audit.iloc[2]["classification"] == "swap_in"

        # Record 4: rejected 09-score, bin 2.0, efx 8 > 6 -> rejected
        assert audit.iloc[3]["classification"] == "rejected"

        # Record 5: rejected 08-other, bin 2.0, efx 4 <= 6 -> rejected_other (not swap_in!)
        assert audit.iloc[4]["classification"] == "rejected_other"

    def test_financing_rate_applied(self, sample_data, optimal_solution):
        """Test that financing rate is applied to swap-in amounts."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        financing_rate = 0.5

        audit = generate_audit_table(sample_data, optimal_solution, variables, financing_rate=financing_rate)

        # Record 3 is swap_in with oa_amt=3000
        swap_in_row = audit[audit["classification"] == "swap_in"].iloc[0]
        assert swap_in_row["oa_amt_h0"] == 3000
        assert swap_in_row["oa_amt_adjusted"] == 3000 * financing_rate

        # Record 1 is keep, should not be adjusted
        keep_row = audit[audit["classification"] == "keep"].iloc[0]
        assert keep_row["oa_amt_h0"] == keep_row["oa_amt_adjusted"]

    def test_n_months_annualization(self, sample_data, optimal_solution):
        """Test that n_months annualization is applied correctly."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        n_months = 6  # 6 months -> annual_coef = 12/6 = 2.0

        audit = generate_audit_table(sample_data, optimal_solution, variables, n_months=n_months)

        # All amounts should be multiplied by 2.0 (12/6)
        # Record 1 is keep with oa_amt=1000
        keep_row = audit[audit["classification"] == "keep"].iloc[0]
        assert keep_row["oa_amt_h0"] == 1000
        assert keep_row["oa_amt_adjusted"] == 1000 * 2.0  # annualized

        # Record 2 is swap_out with oa_amt_h0=2000
        swap_out_row = audit[audit["classification"] == "swap_out"].iloc[0]
        assert swap_out_row["oa_amt_h0"] == 2000
        assert swap_out_row["oa_amt_adjusted"] == 2000 * 2.0  # annualized

    def test_n_months_with_financing_rate(self, sample_data, optimal_solution):
        """Test that both n_months and financing_rate are applied to swap-in."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        financing_rate = 0.5
        n_months = 6  # annual_coef = 2.0

        audit = generate_audit_table(
            sample_data, optimal_solution, variables, financing_rate=financing_rate, n_months=n_months
        )

        # Record 3 is swap_in with oa_amt=3000
        # Should be: 3000 * 0.5 (financing) * 2.0 (annualized) = 3000
        swap_in_row = audit[audit["classification"] == "swap_in"].iloc[0]
        assert swap_in_row["oa_amt_h0"] == 3000
        assert swap_in_row["oa_amt_adjusted"] == 3000 * financing_rate * 2.0

        # Record 1 is keep with oa_amt_h0=1000
        # Should be: 1000 * 2.0 (annualized only, no financing rate)
        keep_row = audit[audit["classification"] == "keep"].iloc[0]
        assert keep_row["oa_amt_h0"] == 1000
        assert keep_row["oa_amt_adjusted"] == 1000 * 2.0

    def test_cut_limits_correct(self, sample_data, optimal_solution):
        """Test that cut limits are correctly assigned."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        audit = generate_audit_table(sample_data, optimal_solution, variables)

        # Bin 1.0 records should have cut_limit 5
        assert audit.iloc[0]["cut_limit"] == 5
        assert audit.iloc[1]["cut_limit"] == 5

        # Bin 2.0 records should have cut_limit 6
        assert audit.iloc[2]["cut_limit"] == 6
        assert audit.iloc[3]["cut_limit"] == 6
        assert audit.iloc[4]["cut_limit"] == 6

    def test_cell_key_format(self, sample_data, optimal_solution):
        """Cell key should include all configured variable coordinates."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        audit = generate_audit_table(sample_data, optimal_solution, variables)
        assert audit.iloc[0]["cell_key"] == "1.0|3"

    def test_nd_mask_decision_source(self):
        """N-D mask path should label decisions as mask and keep cut_limit blank."""
        data = pd.DataFrame(
            {
                "authorization_id": [1, 2],
                "status_name": ["booked", "rejected"],
                "reject_reason": [None, "09-score"],
                "risk_score_rf": [10, 20],
                "score_rf": [100, 200],
                "sc_octroi_new_clus": [1.0, 1.0],
                "new_efx_clus": [3, 4],
                "income_bin": [1, 2],
                "oa_amt_h0": [1000, 2000],
            }
        )
        optimal_solution = pd.DataFrame({"sol_fac": [0], "acceptance_mask": ["1,0"]})
        variables = ["sc_octroi_new_clus", "new_efx_clus", "income_bin"]

        from src.optimization_utils import CellGrid

        grid_df = pd.DataFrame(
            {
                "sc_octroi_new_clus": [1.0, 1.0],
                "new_efx_clus": [3, 4],
                "income_bin": [1, 2],
                "todu_30ever_h6": [0.0, 0.0],
                "todu_amt_pile_h6": [0.0, 0.0],
                "oa_amt_h0": [0.0, 0.0],
            }
        )
        grid = CellGrid.from_summary(grid_df, variables)
        # Full grid has 4 cells (2 efx values x 2 income values) for fixed mask indexing.
        mask_arr = pd.Series([1, 0, 0, 0]).to_numpy(dtype=int)

        audit = generate_audit_table(
            data=data,
            optimal_solution_df=optimal_solution,
            variables=variables,
            mask=mask_arr,
            grid=grid,
        )
        assert audit["decision_source"].eq("mask").all()
        assert audit["cut_limit"].isna().all()
        assert audit.iloc[0]["cell_key"] == "1.0|3|1"


class TestReconcileRiskProductionSummary:
    """Tests for reconcile_risk_production_summary_with_audit."""

    def test_reconcile_overwrites_production_from_audit(self):
        """Summary Production (€) should match audit classification sums."""
        audit_df = pd.DataFrame(
            {
                "status_name": [
                    StatusName.BOOKED.value,
                    StatusName.BOOKED.value,
                    StatusName.REJECTED.value,
                    StatusName.REJECTED.value,
                ],
                "classification": ["keep", "swap_out", "swap_in", "rejected_other"],
                "oa_amt_adjusted": [1000.0, 3000.0, 500.0, 100.0],
            }
        )
        summary = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
                "Production (€)": [99999.0, 0.0, 0.0, 0.0, 0.0],
                "Production (%)": [1.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        out = reconcile_risk_production_summary_with_audit(summary, audit_df)
        actual = 4000.0  # keep + swap_out
        assert out.loc[out["Metric"] == "Actual", "Production (€)"].values[0] == actual
        assert out.loc[out["Metric"] == "Swap-out", "Production (€)"].values[0] == 3000.0
        assert out.loc[out["Metric"] == "Swap-in", "Production (€)"].values[0] == 500.0
        optimum = actual - 3000.0 + 500.0
        assert out.loc[out["Metric"] == "Optimum selected", "Production (€)"].values[0] == optimum
        assert out.loc[out["Metric"] == "Summary", "Production (€)"].values[0] == optimum - actual

    def test_reconcile_per_income_bin_sums_to_full_audit(self):
        """Splitting audit by income_bin and reconciling each slice sums to the full audit."""
        rows = []
        for inc in (1, 1, 2, 2):
            rows.append(
                {
                    "status_name": StatusName.BOOKED.value,
                    "classification": "keep",
                    "income_bin": inc,
                    "oa_amt_adjusted": 500.0,
                }
            )
        rows.append(
            {
                "status_name": StatusName.BOOKED.value,
                "classification": "swap_out",
                "income_bin": 1,
                "oa_amt_adjusted": 100.0,
            }
        )
        rows.append(
            {
                "status_name": StatusName.REJECTED.value,
                "classification": "swap_in",
                "income_bin": 2,
                "oa_amt_adjusted": 50.0,
            }
        )
        full = pd.DataFrame(rows)
        s1 = reconcile_risk_production_summary_with_audit(
            pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
                    "Production (€)": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Production (%)": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "Total Demand (€)": [2000.0, np.nan, np.nan, np.nan, np.nan],
                }
            ),
            full.loc[full["income_bin"] == 1],
            silent=True,
        )
        s2 = reconcile_risk_production_summary_with_audit(
            pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
                    "Production (€)": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Production (%)": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "Total Demand (€)": [2000.0, np.nan, np.nan, np.nan, np.nan],
                }
            ),
            full.loc[full["income_bin"] == 2],
            silent=True,
        )
        full_tbl = reconcile_risk_production_summary_with_audit(
            pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected", "Summary"],
                    "Production (€)": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "Production (%)": [1.0, 0.0, 0.0, 0.0, 0.0],
                    "Total Demand (€)": [2000.0, np.nan, np.nan, np.nan, np.nan],
                }
            ),
            full,
            silent=True,
        )
        so1 = float(s1.loc[s1["Metric"] == "Swap-out", "Production (€)"].values[0])
        so2 = float(s2.loc[s2["Metric"] == "Swap-out", "Production (€)"].values[0])
        sof = float(full_tbl.loc[full_tbl["Metric"] == "Swap-out", "Production (€)"].values[0])
        assert so1 + so2 == sof


class TestGenerateAuditSummary:
    """Tests for the generate_audit_summary function."""

    def test_basic_summary(self):
        """Test basic summary generation."""
        audit_df = pd.DataFrame(
            {
                "classification": ["keep", "keep", "swap_out", "swap_in", "rejected", "rejected_other"],
                "oa_amt_h0": [1000, 2000, 3000, 4000, 5000, 6000],
                "oa_amt_adjusted": [1000, 2000, 3000, 2000, 5000, 6000],  # swap_in adjusted
            }
        )

        summary = generate_audit_summary(audit_df)

        assert len(summary) == 5  # 5 unique classifications
        assert "count" in summary.columns
        assert "total_oa_amt" in summary.columns

        # Check counts
        keep_row = summary[summary["classification"] == "keep"]
        assert keep_row["count"].values[0] == 2

    def test_summary_uses_adjusted_amounts(self):
        """Test that summary uses adjusted amounts by default."""
        audit_df = pd.DataFrame(
            {
                "classification": ["keep", "swap_in"],
                "oa_amt_h0": [1000, 4000],
                "oa_amt_adjusted": [1000, 2000],  # swap_in adjusted by 0.5
            }
        )

        summary = generate_audit_summary(audit_df, use_adjusted=True)

        swap_in_row = summary[summary["classification"] == "swap_in"]
        assert swap_in_row["total_oa_amt"].values[0] == 2000  # Uses adjusted

    def test_summary_raw_amounts(self):
        """Test that summary can use raw amounts."""
        audit_df = pd.DataFrame(
            {
                "classification": ["keep", "swap_in"],
                "oa_amt_h0": [1000, 4000],
                "oa_amt_adjusted": [1000, 2000],
            }
        )

        summary = generate_audit_summary(audit_df, use_adjusted=False)

        swap_in_row = summary[summary["classification"] == "swap_in"]
        assert swap_in_row["total_oa_amt"].values[0] == 4000  # Uses oa_amt_h0
