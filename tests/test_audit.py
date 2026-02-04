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
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "swap_out"

    def test_swap_in_rejected_passes_cut(self):
        """Test rejected record that passes cutoff is classified as swap_in."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,  # Below cutoff of 5
            "status_name": "rejected",
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "swap_in"

    def test_rejected_fails_cut(self):
        """Test rejected record that fails cutoff is classified as rejected."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 7,  # Above cutoff of 5
            "status_name": "rejected",
        })
        cut_map = {1.0: 5, 2.0: 6}

        result = classify_record(row, "sc_octroi_new_clus", "new_efx_clus", cut_map)
        assert result == "rejected"

    def test_inv_var1_logic(self):
        """Test inverted var1 logic (>= instead of <=)."""
        row = pd.Series({
            "sc_octroi_new_clus": 1.0,
            "new_efx_clus": 3,  # Below cutoff of 5, would fail with inv_var1
            "status_name": "booked",
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
            "authorization_id": [1, 2, 3, 4],
            "status_name": ["booked", "booked", "rejected", "rejected"],
            "risk_score_rf": [10, 20, 30, 40],
            "score_rf": [100, 200, 300, 400],
            "sc_octroi_new_clus": [1.0, 1.0, 2.0, 2.0],
            "new_efx_clus": [3, 7, 4, 8],
            "oa_amt": [1000, 2000, 3000, 4000],
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

        assert len(audit) == 4
        assert "classification" in audit.columns
        assert "cut_limit" in audit.columns
        assert "passes_cut" in audit.columns

    def test_classifications_correct(self, sample_data, optimal_solution):
        """Test that classifications are correct."""
        variables = ["sc_octroi_new_clus", "new_efx_clus"]

        audit = generate_audit_table(sample_data, optimal_solution, variables)

        # Record 1: booked, bin 1.0, efx 3 <= 5 -> keep
        assert audit.iloc[0]["classification"] == "keep"

        # Record 2: booked, bin 1.0, efx 7 > 5 -> swap_out
        assert audit.iloc[1]["classification"] == "swap_out"

        # Record 3: rejected, bin 2.0, efx 4 <= 6 -> swap_in
        assert audit.iloc[2]["classification"] == "swap_in"

        # Record 4: rejected, bin 2.0, efx 8 > 6 -> rejected
        assert audit.iloc[3]["classification"] == "rejected"

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


class TestGenerateAuditSummary:
    """Tests for the generate_audit_summary function."""

    def test_basic_summary(self):
        """Test basic summary generation."""
        audit_df = pd.DataFrame({
            "classification": ["keep", "keep", "swap_out", "swap_in", "rejected"],
            "oa_amt": [1000, 2000, 3000, 4000, 5000],
        })

        summary = generate_audit_summary(audit_df)

        assert len(summary) == 4  # 4 unique classifications
        assert "count" in summary.columns
        assert "total_oa_amt" in summary.columns

        # Check counts
        keep_row = summary[summary["classification"] == "keep"]
        assert keep_row["count"].values[0] == 2

    def test_summary_amounts(self):
        """Test that amounts are summed correctly."""
        audit_df = pd.DataFrame({
            "classification": ["keep", "keep", "swap_out"],
            "oa_amt": [1000, 2000, 3000],
        })

        summary = generate_audit_summary(audit_df)

        keep_row = summary[summary["classification"] == "keep"]
        assert keep_row["total_oa_amt"].values[0] == 3000

        swap_out_row = summary[summary["classification"] == "swap_out"]
        assert swap_out_row["total_oa_amt"].values[0] == 3000
