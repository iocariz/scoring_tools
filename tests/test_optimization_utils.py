import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from loguru import logger

from src.optimization_utils import (
    create_fixed_cutoff_solution,
    get_fact_sol,
)

# =============================================================================
# create_fixed_cutoff_solution Tests
# =============================================================================


class TestCreateFixedCutoffSolution:
    """Tests for the create_fixed_cutoff_solution utility function."""

    def test_basic_fixed_cutoff(self):
        """Test creating a fixed cutoff solution."""
        fixed_cutoffs = {
            "sc_octroi_new_clus": [1.0, 2.0, 3.0, 4.0],
            "new_efx_clus": [2, 2, 3, 4],
        }
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        values_var0 = [1.0, 2.0, 3.0, 4.0]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        # Check structure
        assert "sol_fac" in result.columns
        assert result["sol_fac"].iloc[0] == 0
        assert len(result) == 1

        # Check cutoff values
        assert result[1.0].iloc[0] == 2
        assert result[2.0].iloc[0] == 2
        assert result[3.0].iloc[0] == 3
        assert result[4.0].iloc[0] == 4

    def test_fixed_cutoff_with_integers(self):
        """Test with integer bin values."""
        fixed_cutoffs = {
            "var0": [1, 2, 3],
            "var1": [5, 6, 7],
        }
        variables = ["var0", "var1"]
        values_var0 = [1, 2, 3]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 5
        assert result[2.0].iloc[0] == 6
        assert result[3.0].iloc[0] == 7

    def test_missing_variable_raises_error(self):
        """Test that missing variable raises ValueError."""
        fixed_cutoffs = {
            "sc_octroi_new_clus": [1.0, 2.0, 3.0],
            # missing new_efx_clus
        }
        variables = ["sc_octroi_new_clus", "new_efx_clus"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="must contain both variables"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

    def test_length_mismatch_raises_error(self):
        """Test that mismatched lengths raise ValueError."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [2, 3],  # Wrong length
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Length mismatch"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

    def test_single_bin(self):
        """Test with single bin."""
        fixed_cutoffs = {
            "var0": [1.0],
            "var1": [5],
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0]

        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 5

    def test_monotonicity_warning(self):
        """Test that non-monotonic cutoffs trigger warning (captured via loguru sink)."""
        from io import StringIO

        # Add a temporary sink to capture logs
        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0, 4.0],
                "var1": [5, 3, 4, 6],  # Non-monotonic: 5 -> 3 decreases
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0, 4.0]

            result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

            # Should succeed but log warning
            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic cutoffs detected" in log_output
        finally:
            logger.remove(handler_id)

    def test_monotonicity_strict_raises_error(self):
        """Test that non-monotonic cutoffs raise error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [5, 3, 4],  # Non-monotonic
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Non-monotonic cutoffs"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0, strict_validation=True)

    def test_monotonicity_inverted(self):
        """Test monotonicity with inv_var1=True (cutoffs should decrease)."""
        from io import StringIO

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            # For inv_var1=True, cutoffs should be non-increasing
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [5, 6, 7],  # Increasing - wrong for inv_var1
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]

            result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0, inv_var1=True)

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic cutoffs detected" in log_output
            assert "non-increasing" in log_output
        finally:
            logger.remove(handler_id)

    def test_monotonicity_inverted_valid(self):
        """Test valid monotonicity with inv_var1=True."""
        from io import StringIO

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            # For inv_var1=True, cutoffs should be non-increasing (valid case)
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [7, 5, 3],  # Decreasing - correct for inv_var1
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]

            result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0, inv_var1=True)

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "Non-monotonic" not in log_output
        finally:
            logger.remove(handler_id)

    def test_data_bounds_warning(self):
        """Test warning when cutoffs are outside data range."""
        from io import StringIO

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")

        try:
            fixed_cutoffs = {
                "var0": [1.0, 2.0, 3.0],
                "var1": [5, 6, 15],  # 15 is outside range [1, 10]
            }
            variables = ["var0", "var1"]
            values_var0 = [1.0, 2.0, 3.0]
            values_var1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0, values_var1=values_var1)

            assert len(result) == 1
            log_output = log_capture.getvalue()
            assert "outside data range" in log_output
            assert "[15]" in log_output
        finally:
            logger.remove(handler_id)

    def test_data_bounds_strict_raises_error(self):
        """Test that out-of-bounds cutoffs raise error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [5, 20],  # 20 is outside range
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0]
        values_var1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        with pytest.raises(ValueError, match="outside data range"):
            create_fixed_cutoff_solution(
                fixed_cutoffs,
                variables,
                values_var0,
                values_var1=values_var1,
                strict_validation=True,
            )

    def test_bin_mismatch_strict_raises_error(self):
        """Test that bin mismatch raises error in strict mode."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [5, 6, 7],
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 4.0]  # 4.0 instead of 3.0

        with pytest.raises(ValueError, match="don't exactly match"):
            create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0, strict_validation=True)

    def test_valid_monotonic_cutoffs(self):
        """Test that valid monotonic cutoffs don't trigger warnings."""
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0, 4.0],
            "var1": [2, 3, 3, 5],  # Non-decreasing (equal is OK)
        }
        variables = ["var0", "var1"]
        values_var0 = [1.0, 2.0, 3.0, 4.0]

        # Should not raise or warn
        result = create_fixed_cutoff_solution(fixed_cutoffs, variables, values_var0)

        assert len(result) == 1
        assert result[1.0].iloc[0] == 2
        assert result[2.0].iloc[0] == 3
        assert result[3.0].iloc[0] == 3
        assert result[4.0].iloc[0] == 5


# =============================================================================
# get_fact_sol Tests
# =============================================================================


class TestGetFactSol:
    def test_basic_generation(self):
        values_var0 = [0.0, 1.0, 2.0]
        values_var1 = [1.0, 2.0, 3.0]
        result = get_fact_sol(values_var0, values_var1)
        assert "sol_fac" in result.columns
        assert len(result) > 0

    def test_includes_zero_cut(self):
        """0 should always be included as a possible cut value."""
        values_var0 = [0.0, 1.0]
        values_var1 = [1.0, 2.0]
        result = get_fact_sol(values_var0, values_var1)
        # Check that some solutions have 0 as a cut value
        bin_cols = [c for c in result.columns if c != "sol_fac"]
        has_zero = (result[bin_cols] == 0).any().any()
        assert has_zero

    def test_monotonicity_non_decreasing(self):
        """All solutions should be monotonically non-decreasing."""
        values_var0 = [0.0, 1.0, 2.0]
        values_var1 = [1.0, 2.0, 3.0]
        result = get_fact_sol(values_var0, values_var1, inv_var1=False)
        bin_cols = [c for c in result.columns if c != "sol_fac"]
        for _, row in result.iterrows():
            vals = [row[c] for c in bin_cols]
            assert vals == sorted(vals)

    def test_monotonicity_non_increasing_inverted(self):
        """With inv_var1=True, solutions should be non-increasing."""
        values_var0 = [0.0, 1.0, 2.0]
        values_var1 = [1.0, 2.0, 3.0]
        result = get_fact_sol(values_var0, values_var1, inv_var1=True)
        bin_cols = [c for c in result.columns if c != "sol_fac"]
        for _, row in result.iterrows():
            vals = [row[c] for c in bin_cols]
            assert vals == sorted(vals, reverse=True)

    def test_solution_count(self):
        """Number of solutions should be C(n+k-1, k) where n=cut_values, k=n_bins."""
        from math import comb

        values_var0 = [0.0, 1.0]  # 2 bins
        values_var1 = [1.0, 2.0]  # + 0 = 3 cut values
        result = get_fact_sol(values_var0, values_var1)
        # C(3+2-1, 2) = C(4,2) = 6
        assert len(result) == comb(3 + 2 - 1, 2)
