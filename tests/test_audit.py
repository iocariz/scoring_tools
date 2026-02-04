"""Tests for the audit module."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.audit import classify_record, generate_audit_table, generate_audit_summary


class TestClassifyRecord:
    """Tests for the classify_record function."""

    def test_keep_booked_passes_cut(self):
        """Test booked record that passes cutoff is classified as keep."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,
            "status_name": "booked",
            "reject_reason": None,
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "keep"

    def test_swap_out_booked_fails_cut(self):
        """Test booked record that fails cutoff is classified as swap_out."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 7,  # Above cutoff of 5
            "status_name": "booked",
            "reject_reason": None,
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "swap_out"

    def test_swap_in_score_rejected_passes_cut(self):
        """Test score-rejected record that passes cutoff is classified as swap_in."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,  # Below cutoff of 5
            "status_name": "rejected",
            "reject_reason": "09-score",
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "swap_in"

    def test_rejected_score_rejected_fails_cut(self):
        """Test score-rejected record that fails cutoff is classified as rejected."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 7,  # Above cutoff of 5
            "status_name": "rejected",
            "reject_reason": "09-score",
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "rejected"

    def test_rejected_other_not_swap_in(self):
        """Test other-rejected record is classified as rejected_other even if passes cut."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,  # Below cutoff of 5
            "status_name": "rejected",
            "reject_reason": "08-other",  # Not score-rejected
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "rejected_other"

    def test_inv_var1_logic(self):
        """Test inverted var1 logic (>= instead of <=)."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,  # Below cutoff of 5, would fail with inv_var1
            "status_name": "booked",
            "reject_reason": None,
        })
        cut_map = {1.0: 5, 2.0: 6}

        # With inv_var1=True, 3 >= 5 is False, so booked becomes swap_out
        result = classify_record(
            row, "sc_octroi_new_clus", "new_efx_clus", cut_map, inv_var1=True
        )
        assert result == "swap_out"

    def test_exact_cutoff_passes(self):
        """Test record at exact cutoff passes."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 5,  # Exactly at cutoff
            "status_name": "booked",
            "reject_reason": None,
        })
        cut_map = {1.0: 5}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "keep"


class TestGenerateAuditTable:
    """Tests for the generate_audit_table function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            "authorization_id": [1, 2, 3, 4, 5],
            "status_name": ["booked", "booked", "rejected", "rejected", "rejected"],
            "reject_reason": [None, None, "09-score", "09-score", "08-other"],
            "risk_score_rf": [10, 20, 30, 40, 50],
            "score_rf": [100, 200, 300, 400, 500],
            "sc_octroi_new_clus": [1.0, 1.0, 2.0, 2.0, 2.0],
            "new_efx_clus": [3, 7, 4, 8, 4],
            "oa_amt": [1000, 2000, 3000, 4000, 5000],
        })

    @pytest.fixture
    def optimal_solution(self):
        """Create sample optimal solution."""
        return pd.DataFrame({
            "sol_fac": [0],
            1.0: [5],  # Cutoff for bin 1.0
            2.0: [6],  # Cutoff for bin 2.0
        })

    def test_basic_audit_table(self, sample_data, optimal_solution):
        """Test basic audit table generation."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        audit = generate_audit_table(sample_data, optimal_solution, variables)

        assert len(audit) == 5
        assert "classification" in audit.columns
        assert "cut_limit" in audit.columns
        assert "passes_cut" in audit.columns
        assert "reject_reason" in audit.columns
        assert "oa_amt_adjusted" in audit.columns

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

        audit = generate_audit_table(
            sample_data, optimal_solution, variables, financing_rate=financing_rate
        )

        # Record 3 is swap_in with oa_amt=3000
        swap_in_row = audit[audit["classification"] == "swap_in"].iloc[0]
        assert swap_in_row["oa_amt"] == 3000
        assert swap_in_row["oa_amt_adjusted"] == 3000 * financing_rate

        # Record 1 is keep, should not be adjusted
        keep_row = audit[audit["classification"] == "keep"].iloc[0]
        assert keep_row["oa_amt"] == keep_row["oa_amt_adjusted"]

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


class TestGenerateAuditSummary:
    """Tests for the generate_audit_summary function."""

    def test_basic_summary(self):
        """Test basic summary generation."""
        audit_df = pd.DataFrame({
            "classification": ["keep", "keep", "swap_out", "swap_in", "rejected", "rejected_other"],
            "oa_amt": [1000, 2000, 3000, 4000, 5000, 6000],
            "oa_amt_adjusted": [1000, 2000, 3000, 2000, 5000, 6000],  # swap_in adjusted
        })

        summary = generate_audit_summary(audit_df)

        assert len(summary) == 5  # 5 unique classifications
        assert "count" in summary.columns
        assert "total_oa_amt" in summary.columns

        # Check counts
        keep_row = summary[summary["classification"] == "keep"]
        assert keep_row["count"].values[0] == 2

    def test_summary_uses_adjusted_amounts(self):
        """Test that summary uses adjusted amounts by default."""
        audit_df = pd.DataFrame({
            "classification": ["keep", "swap_in"],
            "oa_amt": [1000, 4000],
            "oa_amt_adjusted": [1000, 2000],  # swap_in adjusted by 0.5
        })

        summary = generate_audit_summary(audit_df, use_adjusted=True)

        swap_in_row = summary[summary["classification"] == "swap_in"]
        assert swap_in_row["total_oa_amt"].values[0] == 2000  # Uses adjusted

    def test_summary_raw_amounts(self):
        """Test that summary can use raw amounts."""
        audit_df = pd.DataFrame({
            "classification": ["keep", "swap_in"],
            "oa_amt": [1000, 4000],
            "oa_amt_adjusted": [1000, 2000],
        })

        summary = generate_audit_summary(audit_df, use_adjusted=False)

        swap_in_row = summary[summary["classification"] == "swap_in"]
        assert swap_in_row["total_oa_amt"].values[0] == 4000  # Uses raw
