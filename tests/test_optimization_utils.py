import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from src.optimization_utils import (
    CellGrid,
    _build_monotonicity_constraints,
    _ga_pareto_fallback,
    _validate_nd_cutoff_structure,
    _warn_unenforceable_swapin_caps,
    add_bin_columns,
    classify_by_mask,
    create_fixed_cutoff_mask,
    create_fixed_cutoff_solution,
    decode_mask,
    evaluate_solution,
    get_fact_sol,
    kpi_of_fact_sol,
    mask_to_cutoffs,
    milp_solve_cutoffs,
    pareto_keep_indices,
    trace_pareto_frontier,
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


# =============================================================================
# Helper: create synthetic data_summary_desagregado
# =============================================================================


def _make_summary_2d(n_var0=3, n_var1=4):
    """Create a 2D summary DataFrame for testing MILP optimization."""
    rng = np.random.RandomState(42)
    rows = []
    for v0 in range(1, n_var0 + 1):
        for v1 in range(1, n_var1 + 1):
            risk = rng.uniform(0.01, 0.1)
            amt = rng.uniform(1000, 10000)
            production = rng.uniform(5000, 50000)
            # H3 values: scaled from H6 (shorter horizon → smaller numerator, similar denominator)
            risk_h3 = risk * 0.6
            amt_h3 = amt * 0.9
            rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    # Loan count (SE sample size, todo #58). Deterministic from `amt` so this does
                    # NOT consume from the RandomState stream and perturb other fixtures' draws.
                    "acct_booked_h0": max(round(amt / 100.0), 2),
                    "todu_30ever_h6": risk * amt / 7,
                    "todu_amt_pile_h6": amt,
                    "oa_amt_h0": production,
                    "todu_30ever_h6_boo": risk * amt / 7 * 0.8,
                    "todu_amt_pile_h6_boo": amt * 0.8,
                    "oa_amt_h0_boo": production * 0.8,
                    "todu_30ever_h6_rep": risk * amt / 7 * 0.2,
                    "todu_amt_pile_h6_rep": amt * 0.2,
                    "oa_amt_h0_rep": production * 0.2,
                    "todu_30ever_h3": risk_h3 * amt_h3 / 4,
                    "todu_amt_pile_h3": amt_h3,
                    "todu_30ever_h3_boo": risk_h3 * amt_h3 / 4 * 0.8,
                    "todu_amt_pile_h3_boo": amt_h3 * 0.8,
                    "todu_30ever_h3_rep": risk_h3 * amt_h3 / 4 * 0.2,
                    "todu_amt_pile_h3_rep": amt_h3 * 0.2,
                }
            )
    return pd.DataFrame(rows)


def _make_summary_3d(n_var0=2, n_var1=3, n_var2=2):
    """Create a 3D summary DataFrame for testing N-variable MILP."""
    rng = np.random.RandomState(42)
    rows = []
    for v0 in range(1, n_var0 + 1):
        for v1 in range(1, n_var1 + 1):
            for v2 in range(1, n_var2 + 1):
                risk = rng.uniform(0.01, 0.1)
                amt = rng.uniform(1000, 10000)
                production = rng.uniform(5000, 50000)
                risk_h3 = risk * 0.6
                amt_h3 = amt * 0.9
                rows.append(
                    {
                        "var0": v0,
                        "var1": v1,
                        "var2": v2,
                        "todu_30ever_h6": risk * amt / 7,
                        "todu_amt_pile_h6": amt,
                        "oa_amt_h0": production,
                        "todu_30ever_h6_boo": risk * amt / 7 * 0.8,
                        "todu_amt_pile_h6_boo": amt * 0.8,
                        "oa_amt_h0_boo": production * 0.8,
                        "todu_30ever_h6_rep": risk * amt / 7 * 0.2,
                        "todu_amt_pile_h6_rep": amt * 0.2,
                        "oa_amt_h0_rep": production * 0.2,
                        "todu_30ever_h3": risk_h3 * amt_h3 / 4,
                        "todu_amt_pile_h3": amt_h3,
                        "todu_30ever_h3_boo": risk_h3 * amt_h3 / 4 * 0.8,
                        "todu_amt_pile_h3_boo": amt_h3 * 0.8,
                        "todu_30ever_h3_rep": risk_h3 * amt_h3 / 4 * 0.2,
                        "todu_amt_pile_h3_rep": amt_h3 * 0.2,
                    }
                )
    return pd.DataFrame(rows)


# =============================================================================
# CellGrid Tests
# =============================================================================


class TestCellGrid:
    def test_from_summary_2d(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.shape == (3, 4)
        assert grid.n_cells == 12
        assert len(grid.cell_index) == 12
        assert set(grid.variables) == {"var0", "var1"}

    def test_from_summary_3d(self):
        df = _make_summary_3d(2, 3, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        assert grid.shape == (2, 3, 2)
        assert grid.n_cells == 12

    def test_values_per_var(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.values_per_var["var0"] == [1, 2, 3]
        assert grid.values_per_var["var1"] == [1, 2, 3, 4]

    def test_cell_data_aligned(self):
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Cell data should have same number of rows as n_cells
        assert len(grid.cell_data) == grid.n_cells

    def test_missing_cells_filled_zero(self):
        """If some combinations are missing from data, they should be filled with 0."""
        df = _make_summary_2d(2, 2)
        # Remove one row
        df = df.iloc[:-1]
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.n_cells == 4  # 2x2 grid, even though only 3 rows in data
        # The missing cell should have 0 values
        assert grid.cell_data["oa_amt_h0"].iloc[-1] == 0.0

    def test_observed_flag_full_grid(self):
        """All cells observed when data covers the full grid."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.observed.all()
        assert len(grid.observed) == grid.n_cells

    def test_observed_flag_sparse_grid(self):
        """Phantom cells are marked as unobserved."""
        df = _make_summary_2d(2, 2)
        df = df.iloc[:-1]  # remove one cell
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.observed.sum() == 3
        assert (~grid.observed).sum() == 1
        # The unobserved cell should be the last one
        assert not grid.observed[-1]


# =============================================================================
# Monotonicity Constraints Tests
# =============================================================================


class TestMonotonicityConstraints:
    def test_2d_no_inversion(self):
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        A = _build_monotonicity_constraints(grid, inv_vars=[])
        # Should have constraints for both dimensions
        assert A.shape[1] == grid.n_cells
        assert A.shape[0] > 0

    def test_2d_inverted_var1(self):
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        A_normal = _build_monotonicity_constraints(grid, inv_vars=[])
        A_inv = _build_monotonicity_constraints(grid, inv_vars=["var1"])
        # Different inversion should produce different constraints
        assert A_normal.shape == A_inv.shape
        # The constraint matrices should differ
        assert not np.allclose(A_normal.toarray(), A_inv.toarray())

    def test_3d_constraints(self):
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        A = _build_monotonicity_constraints(grid, inv_vars=[])
        assert A.shape[1] == grid.n_cells
        assert A.shape[0] > 0

    def test_monotonicity_enforced(self):
        """Test that a mask violating monotonicity fails the constraint check."""
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        A = _build_monotonicity_constraints(grid, inv_vars=[])

        # Create a mask that violates monotonicity:
        # Accept cell (1,3) but reject (1,2) — higher bin rejected, lower accepted
        mask = np.zeros(grid.n_cells)
        mask[grid.cell_index[(1, 3)]] = 1  # Accept (1,3)
        mask[grid.cell_index[(1, 2)]] = 0  # Reject (1,2)

        violations = A @ mask
        # With no inversion, higher bin = riskier, so x[riskier] <= x[safer]
        # Accepting (1,3) and rejecting (1,2): x[(1,3)]=1, x[(1,2)]=0
        # Constraint: x[(1,3)] - x[(1,2)] <= 0 → 1 - 0 = 1 > 0 → violation
        assert np.any(violations > 0)

    def test_uncertainty_relaxation_reduces_constraints(self):
        """Uncertainty-aware mode should drop ambiguous sparse adjacency constraints."""
        df = _make_summary_2d(2, 2)

        # Force very sparse and near-equal neighboring cells on var1 axis
        # so they become "ambiguous" under z-threshold logic. Few loans (small
        # acct_booked_h0) → large SE → the near-equal risks are indistinguishable.
        df.loc[(df["var0"] == 1) & (df["var1"] == 1), "todu_amt_pile_h6"] = 50.0
        df.loc[(df["var0"] == 1) & (df["var1"] == 2), "todu_amt_pile_h6"] = 50.0
        df.loc[(df["var0"] == 1) & (df["var1"] == 1), "todu_30ever_h6"] = 0.50
        df.loc[(df["var0"] == 1) & (df["var1"] == 2), "todu_30ever_h6"] = 0.51
        df.loc[(df["var0"] == 1) & (df["var1"] == 1), "acct_booked_h0"] = 3
        df.loc[(df["var0"] == 1) & (df["var1"] == 2), "acct_booked_h0"] = 3

        grid = CellGrid.from_summary(df, ["var0", "var1"])
        A_strict = _build_monotonicity_constraints(grid, inv_vars=[])
        A_relaxed = _build_monotonicity_constraints(
            grid,
            inv_vars=[],
            relax_on_uncertainty=True,
            uncertainty_min_exposure=100.0,
            uncertainty_z_threshold=2.0,
            multiplier=7.0,
        )

        assert A_relaxed.shape[0] < A_strict.shape[0]

    def test_relaxation_se_uses_loan_count_not_exposure(self):
        """The SE sample size is the loan count (acct_booked_h0), not € exposure (todo #58).

        Two adjacent cells with LARGE € exposure but a moderate risk gap: with few loans the gap is
        within noise (relax); with many loans the same gap is significant (enforce). Under the old
        exposure-as-n SE both would look significant (n ~ 9e5 → SE ~ 0) and neither would relax —
        i.e. the relaxation was inert. Holding exposure fixed and only changing the loan count proves
        the SE now responds to the count.
        """
        df = _make_summary_2d(2, 2)
        big_exp = 900_000.0
        for v1, b2 in ((1, 0.07), (2, 0.17)):  # moderate risk gap along var1 (q=0.01 vs ~0.024)
            sel = (df["var0"] == 1) & (df["var1"] == v1)
            df.loc[sel, "todu_amt_pile_h6"] = big_exp
            df.loc[sel, "todu_30ever_h6"] = (b2 / 7.0) * big_exp  # b2 = 7 * t30 / exposure

        def relaxed_count(loan_count):
            d = df.copy()
            d.loc[(d["var0"] == 1) & (d["var1"].isin([1, 2])), "acct_booked_h0"] = loan_count
            grid = CellGrid.from_summary(d, ["var0", "var1"])
            strict = _build_monotonicity_constraints(grid, inv_vars=[])
            relaxed = _build_monotonicity_constraints(
                grid,
                inv_vars=[],
                relax_on_uncertainty=True,
                uncertainty_min_exposure=1_000_000.0,  # 900k < 1M → the pair counts as sparse
                uncertainty_z_threshold=2.0,
                multiplier=7.0,
            )
            return strict.shape[0] - relaxed.shape[0]  # #constraints dropped

        # Only the two target cells' loan count changes between the calls, so the DIFFERENCE isolates
        # them. Few loans → large SE → the 0.10 b2 gap is within 2·SE → the target constraint is
        # dropped; many loans → small SE → the same gap is significant → that constraint is kept.
        # (More SE can only add ambiguous pairs, never remove them, so the count is monotone.)
        dropped_few = relaxed_count(5)
        dropped_many = relaxed_count(5000)
        assert dropped_few >= 1
        assert dropped_few > dropped_many  # the target pair flips from relaxed → enforced

    def test_relaxation_falls_back_to_exposure_when_count_absent(self):
        """No acct_booked_h0 → warn and fall back to €-exposure as n (backward-compat path)."""
        from io import StringIO

        df = _make_summary_2d(2, 2).drop(columns=["acct_booked_h0"])
        df.loc[(df["var0"] == 1) & (df["var1"] == 1), "todu_amt_pile_h6"] = 50.0
        df.loc[(df["var0"] == 1) & (df["var1"] == 2), "todu_amt_pile_h6"] = 50.0
        df.loc[(df["var0"] == 1) & (df["var1"] == 1), "todu_30ever_h6"] = 0.50
        df.loc[(df["var0"] == 1) & (df["var1"] == 2), "todu_30ever_h6"] = 0.51
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        log_capture = StringIO()
        handler_id = logger.add(log_capture, format="{message}", level="WARNING")
        try:
            A = _build_monotonicity_constraints(
                grid,
                inv_vars=[],
                relax_on_uncertainty=True,
                uncertainty_min_exposure=100.0,
                uncertainty_z_threshold=2.0,
                multiplier=7.0,
            )
        finally:
            logger.remove(handler_id)
        assert A.shape[1] == grid.n_cells  # still builds
        logs = log_capture.getvalue()
        assert "acct_booked_h0" in logs and "falling back" in logs


class TestMonotonicityPhantomBridging:
    """Audit #22: phantom (unobserved) cells must not participate in monotonicity
    constraints — a forced-zero phantom acting as the "safer" endpoint would
    force-reject every observed cell coordinatewise-riskier than it. Chains must
    bridge over phantom gaps instead."""

    def test_phantom_cells_have_no_constraint_entries(self):
        """No constraint row may touch a phantom cell's column."""
        df = _make_summary_2d(3, 3)
        df = df[~((df["var0"] == 2) & (df["var1"] == 2))]  # phantom in the middle
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        phantom_idx = int(np.where(~grid.observed)[0][0])

        A = _build_monotonicity_constraints(grid, inv_vars=[])
        assert A.shape[0] > 0
        assert np.all(A.toarray()[:, phantom_idx] == 0)

    def test_phantom_gap_is_bridged(self):
        """A phantom inside a 1-D line links its observed neighbors directly."""
        df = _make_summary_2d(1, 4)  # single line along var1: cells (1,1)..(1,4)
        df = df[~((df["var0"] == 1) & (df["var1"] == 2))]  # phantom at (1,2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        A = _build_monotonicity_constraints(grid, inv_vars=[]).toarray()
        # Consecutive observed pairs: (1,1)-(1,3) bridged, (1,3)-(1,4) adjacent
        assert A.shape[0] == 2
        idx = grid.cell_index
        bridged = np.zeros(grid.n_cells)
        bridged[idx[(1, 3)]] = 1.0  # riskier (higher bin, not inverted)
        bridged[idx[(1, 1)]] = -1.0  # safer
        assert any(np.array_equal(row, bridged) for row in A)

    def test_phantom_gap_bridged_inverted(self):
        """Bridging respects the inverted direction (higher bin = safer)."""
        df = _make_summary_2d(1, 4)
        df = df[~((df["var0"] == 1) & (df["var1"] == 2))]
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        A = _build_monotonicity_constraints(grid, inv_vars=["var1"]).toarray()
        assert A.shape[0] == 2
        idx = grid.cell_index
        bridged = np.zeros(grid.n_cells)
        bridged[idx[(1, 1)]] = 1.0  # riskier (lower bin under inversion)
        bridged[idx[(1, 3)]] = -1.0  # safer
        assert any(np.array_equal(row, bridged) for row in A)

    def test_full_grid_constraint_count_unchanged(self):
        """On a fully-observed grid the constraints are the classic adjacency set."""
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        A = _build_monotonicity_constraints(grid, inv_vars=[]).toarray()
        # var0: (2-1)*3 = 3 pairs; var1: 2*(3-1) = 4 pairs
        assert A.shape[0] == 7
        assert np.all(A.sum(axis=1) == 0)  # each row is one +1 and one -1

    def test_phantom_safe_corner_does_not_force_reject_grid(self):
        """Regression for audit #22: a phantom at the SAFEST corner previously
        propagated x=0 through the constraint chain to every cell (everything is
        coordinatewise-riskier than the safe corner), making every MILP target
        infeasible. With bridging, the solver must find a normal solution."""
        df = _make_summary_2d(3, 3)
        df = df[~((df["var0"] == 1) & (df["var1"] == 1))]  # safest cell unobserved
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        phantom_idx = int(np.where(~grid.observed)[0][0])

        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        assert mask is not None, "phantom at safe corner must not make the MILP infeasible"
        assert mask[phantom_idx] == 0  # still never accepted
        assert mask.sum() == int(grid.observed.sum())  # generous target: all observed accepted

    def test_milp_with_phantom_respects_monotonicity_over_observed(self):
        """Solutions on a sparse grid stay monotone across the bridged pairs."""
        df = _make_summary_2d(3, 4)
        df = df[~((df["var0"] == 2) & (df["var1"] == 2))]
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        mask = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        assert mask is not None
        A = _build_monotonicity_constraints(grid, inv_vars=[])
        assert np.all(A @ mask <= 1e-6)


# =============================================================================
# MILP Solver Tests
# =============================================================================


class TestMILPSolver:
    def test_basic_solve(self):
        """MILP should find a feasible solution with loose risk constraint."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        assert mask is not None
        assert mask.shape == (grid.n_cells,)
        assert set(mask).issubset({0, 1})

    def test_all_accepted_for_high_risk(self):
        """With very high risk target, all cells should be accepted."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=100.0, inv_vars=[], multiplier=7)
        assert mask is not None
        assert mask.sum() == grid.n_cells

    def test_tight_risk_rejects_some(self):
        """With tight risk target, some cells should be rejected."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=0.5, inv_vars=[], multiplier=7)
        # Either infeasible (None) or has some rejections
        if mask is not None:
            assert mask.sum() < grid.n_cells

    def test_monotonicity_respected(self):
        """Accepted mask should respect monotonicity constraints."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        assert mask is not None

        A = _build_monotonicity_constraints(grid, inv_vars=[])
        violations = A @ mask
        assert np.all(violations <= 1e-6)

    def test_infeasible_returns_none(self):
        """Very tight risk target should return None (infeasible)."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=0.001, inv_vars=[], multiplier=7)
        # Should be None or a mask with 0 accepted cells
        if mask is not None:
            assert mask.sum() == 0

    def test_timeout_with_incumbent_returns_it(self, monkeypatch):
        """scipy status==1 (time limit) WITH a feasible integer incumbent → return the incumbent
        (rounded), not None. Every constraint is hard, so the incumbent is risk-feasible/monotone by
        construction; recovering it beats dropping the frontier point."""
        from types import SimpleNamespace

        from src import optimization_utils as ou

        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        incumbent = np.array([1.0, 0.0, 0.9999, 0.0001])  # float noise → must be rounded to 0/1

        monkeypatch.setattr(ou, "milp", lambda **kw: SimpleNamespace(success=False, status=1, x=incumbent))
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        assert mask is not None
        assert mask.tolist() == [1, 0, 1, 0]
        assert set(mask).issubset({0, 1})

    def test_timeout_without_incumbent_returns_none(self, monkeypatch):
        """status==1 but no feasible incumbent (x is None) → None (point dropped)."""
        from types import SimpleNamespace

        from src import optimization_utils as ou

        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        monkeypatch.setattr(ou, "milp", lambda **kw: SimpleNamespace(success=False, status=1, x=None))
        assert milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7) is None

    def test_infeasible_status2_returns_none_even_with_x(self, monkeypatch):
        """status==2 (infeasible) → None regardless of any x attribute (only timeouts recover)."""
        from types import SimpleNamespace

        from src import optimization_utils as ou

        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        monkeypatch.setattr(ou, "milp", lambda **kw: SimpleNamespace(success=False, status=2, x=np.ones(grid.n_cells)))
        assert milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7) is None

    def test_phantom_cells_not_accepted_in_milp(self):
        """Phantom/unobserved cells must never be accepted by MILP.

        Regression for "phantom cells treated as free rejections" risk:
        if a CellGrid cell is unobserved, the solver must force x=0.
        """
        df = _make_summary_2d(2, 2)
        df = df.iloc[:-1]  # remove one cell to create exactly one phantom
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        phantom_idxs = np.where(~grid.observed)[0]
        assert phantom_idxs.size == 1
        phantom_idx = int(phantom_idxs[0])

        # With a very high risk target, MILP would normally accept all cells
        # if phantom cells were not properly excluded.
        mask = milp_solve_cutoffs(grid, target_risk=999.0, inv_vars=[], multiplier=7)
        assert mask is not None
        assert mask[phantom_idx] == 0
        assert mask.sum() <= int(grid.observed.sum())

    def test_phantom_cells_not_accepted_in_pareto_masks(self):
        """Pareto masks produced from MILP sweep must also exclude phantom cells."""
        df = _make_summary_2d(2, 2)
        df = df.iloc[:-1]  # create phantom cell

        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        pareto_df, grid_used, pareto_masks = trace_pareto_frontier(
            data_summary_desagregado=df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=indicators,
            n_points=3,
            milp_time_limit=1.0,
        )

        assert pareto_masks  # should find at least one feasible mask
        phantom_idxs = np.where(~grid_used.observed)[0]
        assert phantom_idxs.size == 1
        phantom_idx = int(phantom_idxs[0])

        for m in pareto_masks:
            assert m[phantom_idx] == 0


# =============================================================================
# GA fallback tests (mocked pymoo)
# =============================================================================


def _install_dummy_pymoo(monkeypatch):
    """Install a minimal mock pymoo (optional dep) into sys.modules.

    The dummy optimizer returns X=1 exactly where the upper variable bound
    allows it (xu>0) — i.e. it respects variable bounds but knows nothing about
    inequality constraints, leaving those to `_ga_pareto_fallback`'s post-hoc
    feasibility checks."""
    import types

    class DummyGA:
        def __init__(self, pop_size=100):
            self.pop_size = pop_size

    class DummyProblem:
        def __init__(self, n_var, n_obj, n_ieq_constr, xl, xu):
            self.n_var = n_var
            self.n_obj = n_obj
            self.n_ieq_constr = n_ieq_constr
            self.xl = np.asarray(xl, dtype=float)
            self.xu = np.asarray(xu, dtype=float)

    def dummy_get_termination(*args, **kwargs):
        return ("termination", args, kwargs)

    def dummy_minimize(problem, algorithm, termination, seed=None, verbose=False):
        X = (np.asarray(problem.xu) > 0).astype(float)

        class Res:
            pass

        res = Res()
        res.X = X
        return res

    monkeypatch.setitem(sys.modules, "pymoo", types.ModuleType("pymoo"))
    monkeypatch.setitem(sys.modules, "pymoo.algorithms", types.ModuleType("pymoo.algorithms"))
    monkeypatch.setitem(sys.modules, "pymoo.algorithms.soo", types.ModuleType("pymoo.algorithms.soo"))
    monkeypatch.setitem(
        sys.modules, "pymoo.algorithms.soo.nonconvex", types.ModuleType("pymoo.algorithms.soo.nonconvex")
    )

    ga_mod = types.ModuleType("pymoo.algorithms.soo.nonconvex.ga")
    ga_mod.GA = DummyGA
    monkeypatch.setitem(sys.modules, "pymoo.algorithms.soo.nonconvex.ga", ga_mod)

    core_mod = types.ModuleType("pymoo.core")
    monkeypatch.setitem(sys.modules, "pymoo.core", core_mod)

    problem_mod = types.ModuleType("pymoo.core.problem")
    problem_mod.Problem = DummyProblem
    monkeypatch.setitem(sys.modules, "pymoo.core.problem", problem_mod)

    optimize_mod = types.ModuleType("pymoo.optimize")
    optimize_mod.minimize = dummy_minimize
    monkeypatch.setitem(sys.modules, "pymoo.optimize", optimize_mod)

    termination_mod = types.ModuleType("pymoo.termination")
    termination_mod.get_termination = dummy_get_termination
    monkeypatch.setitem(sys.modules, "pymoo.termination", termination_mod)


class TestGAFallbackPhantomCells:
    def test_phantom_cells_not_accepted_in_ga_fallback(self, monkeypatch):
        """GA fallback must encode phantom/unobserved cells with xu=0.

        `pymoo` is an optional dependency. This test mocks the minimal pymoo
        surface so we can exercise `_ga_pareto_fallback` and ensure the returned
        masks never accept phantom cells.
        """
        _install_dummy_pymoo(monkeypatch)

        # --- Build a grid with exactly one phantom/unobserved cell ---
        df = _make_summary_2d(2, 2)
        df = df.iloc[:-1]
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        phantom_idxs = np.where(~grid.observed)[0]
        assert phantom_idxs.size == 1

        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        pareto_df, grid_used, pareto_masks = _ga_pareto_fallback(
            grid=grid,
            inv_vars=[],
            multiplier=7,
            indicators=indicators,
            n_points=3,
        )

        assert pareto_masks  # should find at least one feasible GA mask (mocked)
        phantom_idxs_used = np.where(~grid_used.observed)[0]
        assert phantom_idxs_used.size == 1
        phantom_idx_used = int(phantom_idxs_used[0])

        for m in pareto_masks:
            assert m[phantom_idx_used] == 0


class TestGAFallbackConstraints:
    """Audit #36: the GA fallback must honor the same side constraints as the
    MILP path — cell pins (fixed_cells) and swap-in caps — instead of silently
    dropping them."""

    def test_ga_fallback_respects_fixed_cell_pins(self, monkeypatch):
        _install_dummy_pymoo(monkeypatch)
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Pin the RISKIEST cell (2,2) to rejected — monotonicity-compatible with
        # accepting everything else, so feasible masks exist.
        pinned = grid.cell_index[(2, 2)]

        _, _, masks = _ga_pareto_fallback(
            grid=grid,
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=3,
            fixed_cells={pinned: 0},
        )

        assert masks
        for m in masks:
            assert m[pinned] == 0

    def test_ga_fallback_enforces_swapin_production_cap(self, monkeypatch):
        _install_dummy_pymoo(monkeypatch)
        df = _make_summary_2d(2, 2)  # oa_amt_h0_rep = 20% of oa_amt_h0 per cell
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        common = dict(
            grid=grid,
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=3,
        )

        # Cap below the 20% swap-in share → every candidate violates → no masks.
        _, _, masks_tight = _ga_pareto_fallback(**common, max_swapin_production_pct=10.0)
        assert masks_tight == []

        # Cap above the share → masks survive.
        _, _, masks_loose = _ga_pareto_fallback(**common, max_swapin_production_pct=50.0)
        assert masks_loose


class TestLastResortRespectsPins:
    def test_last_resort_upper_bound_respects_ceiling_pins(self):
        """Audit #36: when every MILP target is infeasible under ceiling
        (must-reject) pins, the last-resort 'accept-all upper bound' used to
        ignore the pins entirely — returning a mask that violates the ceiling."""
        df = _make_summary_2d(2, 2)
        grid_probe = CellGrid.from_summary(df, ["var0", "var1"])
        # Ceiling: ONLY the riskiest cell (2,2) is allowed. Accepting it alone
        # violates monotonicity, so every MILP target is infeasible → the sweep
        # falls through to the last-resort solutions.
        allowed = grid_probe.cell_index[(2, 2)]
        pins = {idx: 0 for idx in range(grid_probe.n_cells) if idx != allowed}

        pareto_df, grid, masks = trace_pareto_frontier(
            data_summary_desagregado=df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=4,
            show_progress=False,
            fixed_cells=pins,
        )

        assert masks, "last resort must still produce a pin-respecting solution"
        for m in masks:
            for idx, val in pins.items():
                assert m[idx] == val, "returned mask violates a must-reject (ceiling) pin"


# =============================================================================
# evaluate_solution Tests
# =============================================================================


class TestEvaluateSolution:
    def test_all_accepted(self):
        """Evaluating with all cells accepted should include all data."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        mask = np.ones(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7)

        assert "b2_ever_h6" in result
        assert "oa_amt_h0" in result
        assert result["oa_amt_h0"] == pytest.approx(df["oa_amt_h0"].sum())
        # _cut should be 0 when all accepted
        assert result.get("oa_amt_h0_cut", 0) == pytest.approx(0, abs=0.01)

    def test_none_accepted(self):
        """Evaluating with no cells accepted should have 0 production."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        mask = np.zeros(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7)
        assert result["oa_amt_h0"] == 0
        # _cut should equal total _boo
        total_boo = df["oa_amt_h0_boo"].sum()
        assert result["oa_amt_h0_cut"] == pytest.approx(total_boo)

    def test_partial_acceptance(self):
        """Partial acceptance should have _boo + _cut = total_boo."""
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        mask = np.array([1, 1, 0, 1, 0, 0])  # Accept first 2 of var0=1, first of var0=2

        result = evaluate_solution(mask, grid, indicators, multiplier=7)
        total_boo = df["oa_amt_h0_boo"].sum()
        accepted_boo = result["oa_amt_h0_boo"]
        cut_boo = result["oa_amt_h0_cut"]
        assert accepted_boo + cut_boo == pytest.approx(total_boo, rel=1e-6)

    def test_has_all_suffixes(self):
        """Result should have base, _boo, _rep, _cut for each indicator."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        mask = np.ones(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7)
        for ind in indicators:
            assert ind in result
            assert f"{ind}_boo" in result
            assert f"{ind}_rep" in result
            assert f"{ind}_cut" in result


# =============================================================================
# B2 Ever H3 Tests
# =============================================================================


class TestB2EverH3:
    """Tests for complementary H3 risk metric computation."""

    def test_evaluate_solution_computes_h3_when_multiplier_provided(self):
        """evaluate_solution should compute b2_ever_h3 when multiplier_h3 is given."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "todu_30ever_h3", "todu_amt_pile_h3"]
        mask = np.ones(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7, multiplier_h3=4)

        assert "b2_ever_h3" in result
        assert "b2_ever_h3_boo" in result
        assert "b2_ever_h3_rep" in result
        assert "b2_ever_h3_cut" in result
        # H3 risk should be a positive number
        assert result["b2_ever_h3"] > 0

    def test_evaluate_solution_no_h3_when_multiplier_none(self):
        """evaluate_solution should not compute b2_ever_h3 when multiplier_h3 is None."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "todu_30ever_h3", "todu_amt_pile_h3"]
        mask = np.ones(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7, multiplier_h3=None)

        assert "b2_ever_h3" not in result
        assert "b2_ever_h3_boo" not in result

    def test_h3_risk_lower_than_h6(self):
        """H3 risk should generally be lower than H6 (shorter horizon)."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "todu_30ever_h3", "todu_amt_pile_h3"]
        mask = np.ones(grid.n_cells, dtype=int)

        result = evaluate_solution(mask, grid, indicators, multiplier=7, multiplier_h3=4)

        # With our synthetic data (h3 risk = 0.6 * h6 risk), h3 should be lower
        assert result["b2_ever_h3"] < result["b2_ever_h6"]

    def test_pareto_frontier_includes_h3(self):
        """trace_pareto_frontier should include h3 columns when multiplier_h3 is provided."""
        df = _make_summary_2d(3, 4)
        pareto_df, _, _ = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "todu_30ever_h3", "todu_amt_pile_h3"],
            n_points=10,
            multiplier_h3=4,
        )
        assert not pareto_df.empty
        assert "b2_ever_h3" in pareto_df.columns


# =============================================================================
# Pareto Frontier Tests
# =============================================================================


class TestParetoFrontier:
    def test_returns_nonempty(self):
        df = _make_summary_2d(3, 4)
        pareto_df, grid, masks = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=10,
        )
        assert not pareto_df.empty
        assert len(masks) == len(pareto_df)
        assert grid.n_cells == 12

    def test_trace_pareto_filters_nan_risk_solutions(self, monkeypatch):
        """Pareto sweep must exclude NaN-risk solutions (zero denominator).

        Regression: when accepted bins have sum(todu_amt_pile_h6) == 0,
        `calculate_b2_ever_h6` returns NaN and can contaminate Pareto/scenario
        selection. trace_pareto_frontier must filter such solutions.
        """
        df = pd.DataFrame(
            [
                {
                    "var0": 1,
                    "var1": 1,
                    "todu_30ever_h6": 10.0,
                    "todu_amt_pile_h6": 0.0,  # <- zero denom => NaN b2_ever_h6
                    "oa_amt_h0": 100.0,
                    "todu_30ever_h6_boo": 8.0,
                    "todu_amt_pile_h6_boo": 0.0,
                    "oa_amt_h0_boo": 80.0,
                    "todu_30ever_h6_rep": 2.0,
                    "todu_amt_pile_h6_rep": 0.0,
                    "oa_amt_h0_rep": 20.0,
                },
                {
                    "var0": 2,
                    "var1": 1,
                    "todu_30ever_h6": 1.0,
                    "todu_amt_pile_h6": 10.0,  # <- positive denom => finite b2_ever_h6
                    "oa_amt_h0": 50.0,
                    "todu_30ever_h6_boo": 0.8,
                    "todu_amt_pile_h6_boo": 8.0,
                    "oa_amt_h0_boo": 40.0,
                    "todu_30ever_h6_rep": 0.2,
                    "todu_amt_pile_h6_rep": 2.0,
                    "oa_amt_h0_rep": 10.0,
                },
            ]
        )

        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.n_cells == 2
        idx_valid = int(np.where(grid.cell_data["todu_amt_pile_h6"].to_numpy() > 0)[0][0])
        expected_mask = np.zeros(grid.n_cells, dtype=int)
        expected_mask[idx_valid] = 1
        mask_nan = np.zeros(grid.n_cells, dtype=int)
        mask_nan[1 - idx_valid] = 1

        call_state = {"i": 0}

        import src.optimization_utils as optimization_utils_module

        def fake_milp_solve_cutoffs(_grid, target_risk, inv_vars, multiplier, **kwargs):
            # First call returns the zero-denominator mask (NaN b2_ever_h6),
            # subsequent calls return the positive-denominator mask.
            call_state["i"] += 1
            if call_state["i"] == 1:
                return mask_nan.copy()
            return expected_mask.copy()

        monkeypatch.setattr(optimization_utils_module, "milp_solve_cutoffs", fake_milp_solve_cutoffs)

        indicators = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]
        pareto_df, _, pareto_masks = trace_pareto_frontier(
            data_summary_desagregado=df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=indicators,
            n_points=2,
            milp_time_limit=1.0,
        )

        assert not pareto_df.empty
        assert pareto_df["b2_ever_h6"].notna().all()
        assert all(np.isfinite(pareto_df["b2_ever_h6"].to_numpy()))
        assert len(pareto_masks) == len(pareto_df)
        assert any(np.array_equal(m, expected_mask) for m in pareto_masks)

    def test_pareto_optimal(self):
        """For ascending risk, production should be non-decreasing on the frontier."""
        df = _make_summary_2d(3, 4)
        pareto_df, _, _ = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=20,
        )
        if len(pareto_df) > 1:
            sorted_df = pareto_df.sort_values("b2_ever_h6")
            diffs = sorted_df["oa_amt_h0"].diff().iloc[1:]
            assert (diffs >= -1e-6).all(), "Production should be non-decreasing on Pareto frontier"

    def test_masks_are_binary(self):
        df = _make_summary_2d(3, 4)
        _, _, masks = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=10,
        )
        for mask in masks:
            assert set(mask).issubset({0, 1})

    def test_3d_pareto(self):
        """Pareto frontier should work with 3 variables."""
        df = _make_summary_3d(2, 3, 2)
        pareto_df, grid, masks = trace_pareto_frontier(
            df,
            variables=["var0", "var1", "var2"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=10,
        )
        assert not pareto_df.empty
        assert grid.shape == (2, 3, 2)


# =============================================================================
# mask_to_cutoffs Tests
# =============================================================================


class TestMaskToCutoffs:
    def test_2d_all_accepted(self):
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        cutoffs = mask_to_cutoffs(mask, grid, inv_vars=[])
        assert "var0" in cutoffs
        # All bins of var1 accepted → cutoff should be max var1
        for v0 in grid.values_per_var["var0"]:
            assert cutoffs["var0"][float(v0)] == max(grid.values_per_var["var1"])

    def test_2d_none_accepted(self):
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.zeros(grid.n_cells, dtype=int)

        cutoffs = mask_to_cutoffs(mask, grid, inv_vars=[])
        assert "var0" in cutoffs
        for v0 in grid.values_per_var["var0"]:
            assert cutoffs["var0"][float(v0)] == -np.inf


# =============================================================================
# add_bin_columns Tests
# =============================================================================


class TestAddBinColumns:
    def test_2d_adds_bin_columns(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], 7)
        pareto_df = pd.DataFrame([kpis])

        result = add_bin_columns(pareto_df, [mask], grid, inv_vars=[])
        assert "sol_fac" in result.columns
        # Should have bin columns for each var0 value
        for v0 in grid.values_per_var["var0"]:
            assert float(v0) in result.columns

    def test_3d_adds_mask_column(self):
        """For N>2, add_bin_columns should add sol_fac and acceptance_mask."""
        df = _make_summary_3d(2, 3, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.ones(grid.n_cells, dtype=int)

        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], 7)
        pareto_df = pd.DataFrame([kpis])

        result = add_bin_columns(pareto_df, [mask], grid, inv_vars=[])
        assert "sol_fac" in result.columns
        assert "acceptance_mask" in result.columns
        # Decoded mask length equals grid.n_cells
        decoded = decode_mask(result["acceptance_mask"].iloc[0])
        assert len(decoded) == grid.n_cells
        # Round-trip: decoded mask matches original
        np.testing.assert_array_equal(decoded, mask)

    def test_2d_no_mask_column(self):
        """For 2-var case, acceptance_mask should NOT be added (backward compat)."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], 7)
        pareto_df = pd.DataFrame([kpis])

        result = add_bin_columns(pareto_df, [mask], grid, inv_vars=[])
        assert "acceptance_mask" not in result.columns

    def test_decode_mask(self):
        """decode_mask should convert comma-separated string to numpy array."""
        result = decode_mask("1,0,1")
        np.testing.assert_array_equal(result, np.array([1, 0, 1]))


# =============================================================================
# GA Fallback Tests
# =============================================================================


class TestGAFallback:
    def test_missing_pymoo_returns_empty(self):
        """Without pymoo, GA fallback should return empty DataFrame gracefully."""
        from src.optimization_utils import _ga_pareto_fallback

        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        try:
            import pymoo  # noqa: F401

            pytest.skip("pymoo is installed, cannot test missing-pymoo path")
        except ImportError:
            result_df, result_grid, result_masks = _ga_pareto_fallback(
                grid,
                inv_vars=[],
                multiplier=7,
                indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
                n_points=5,
            )
            assert result_df.empty
            assert result_masks == []


# =============================================================================
# MILP fixed_cells Tests
# =============================================================================


class TestMILPFixedCells:
    """Tests for milp_solve_cutoffs with fixed_cells parameter."""

    def test_fixed_accept_appears_in_result(self):
        """A cell fixed as accepted (1) must appear as 1 in the result."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Fix cell (1,1) = index 0 as accepted
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7, fixed_cells={0: 1})
        assert mask is not None
        assert mask[0] == 1

    def test_fixed_reject_appears_in_result(self):
        """A cell fixed as rejected (0) must appear as 0 in the result."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Fix the riskiest cell (last index = highest var0, var1) as rejected.
        # Rejecting the safest cell (index 0) would force all cells rejected
        # via monotonicity, making the problem infeasible.
        last_idx = grid.n_cells - 1
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7, fixed_cells={last_idx: 0})
        assert mask is not None
        assert mask[last_idx] == 0

    def test_empty_fixed_cells_same_as_none(self):
        """Empty dict should produce same result as no fixed_cells."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask_none = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        mask_empty = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7, fixed_cells={})
        assert mask_none is not None
        assert mask_empty is not None
        np.testing.assert_array_equal(mask_none, mask_empty)


# =============================================================================
# classify_by_mask Tests
# =============================================================================


class TestClassifyByMask:
    def test_all_accepted_2d(self):
        """All records should be accepted when mask is all-ones."""
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        result = classify_by_mask(df, mask, grid)
        assert result.all()

    def test_none_accepted_2d(self):
        """No records should be accepted when mask is all-zeros."""
        df = _make_summary_2d(2, 3)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.zeros(grid.n_cells, dtype=int)

        result = classify_by_mask(df, mask, grid)
        assert not result.any()

    def test_partial_acceptance(self):
        """classify_by_mask should accept only cells where mask=1."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Accept first 2 cells only
        mask = np.array([1, 1, 0, 0])

        result = classify_by_mask(df, mask, grid)
        assert result.sum() == 2

    def test_3d_all_accepted(self):
        """classify_by_mask should work with 3 variables."""
        df = _make_summary_3d(2, 3, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.ones(grid.n_cells, dtype=int)

        result = classify_by_mask(df, mask, grid)
        assert result.all()

    def test_3d_partial(self):
        """classify_by_mask should correctly classify 3D cells."""
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        # Accept first half, reject second half
        mask = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        result = classify_by_mask(df, mask, grid)
        assert result.sum() == 4
        assert (~result).sum() == 4

    def test_unmatched_records_rejected(self):
        """Records not matching any grid cell should be rejected."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        # Add a record with out-of-grid values
        extra = pd.DataFrame({"var0": [99], "var1": [99], "oa_amt_h0": [1000]})
        combined = pd.concat([df, extra], ignore_index=True)

        result = classify_by_mask(combined, mask, grid)
        # Original records accepted, extra record rejected
        assert result.iloc[:-1].all()
        assert not result.iloc[-1]

    def test_returns_boolean_series(self):
        """Result should be a boolean Series aligned with input index."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = np.ones(grid.n_cells, dtype=int)

        result = classify_by_mask(df, mask, grid)
        assert isinstance(result, pd.Series)
        assert result.dtype == bool
        assert list(result.index) == list(df.index)


# =============================================================================
# learn_quantile_bins Tests
# =============================================================================


class TestLearnQuantileBins:
    def test_basic_splitting_2_bins(self):
        """With max_bins=2, should produce edges at the median."""
        from src.preprocess_improved import learn_quantile_bins

        rng = np.random.RandomState(42)
        n = 2000
        income = rng.uniform(1000, 5000, n)
        data = pd.DataFrame({"income_t1_m": income})

        edges = learn_quantile_bins(data, max_bins=2)
        assert len(edges) == 3  # [-inf, median, inf]
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        # Threshold should be near median of uniform(1000, 5000) ≈ 3000
        assert 2500 < edges[1] < 3500

    def test_basic_splitting_3_bins(self):
        """With max_bins=3, should produce 4 edges at terciles."""
        from src.preprocess_improved import learn_quantile_bins

        rng = np.random.RandomState(42)
        n = 3000
        income = rng.uniform(0, 9000, n)
        data = pd.DataFrame({"income_t1_m": income})

        edges = learn_quantile_bins(data, max_bins=3)
        assert len(edges) == 4  # [-inf, q33, q67, inf]
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        # Thresholds should be near 3000 and 6000
        assert 2500 < edges[1] < 3500
        assert 5500 < edges[2] < 6500

    def test_missing_column_raises(self):
        """Should raise ValueError if required column is missing."""
        from src.preprocess_improved import learn_quantile_bins

        data = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            learn_quantile_bins(data)

    def test_too_few_records_raises(self):
        """Should raise ValueError when data is too small."""
        from src.preprocess_improved import learn_quantile_bins

        data = pd.DataFrame({"income_t1_m": [100]})
        with pytest.raises(ValueError, match="only 1 valid records"):
            learn_quantile_bins(data, max_bins=2)


# =============================================================================
# learn_income_bins Tests (deprecated, kept for backward compat)
# =============================================================================


class TestLearnIncomeBins:
    def test_basic_splitting(self):
        """Should produce bin edges with splits between classes."""
        from src.preprocess_improved import learn_income_bins

        rng = np.random.RandomState(42)
        n = 2000
        income = np.concatenate([rng.normal(2000, 500, n // 2), rng.normal(5000, 500, n // 2)])
        bad = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        data = pd.DataFrame({"income_t1_m": income, "early_bad": bad})

        edges = learn_income_bins(data, max_bins=2, min_samples_leaf=100)
        assert len(edges) == 3  # [-inf, threshold, inf]
        assert edges[0] == -np.inf
        assert edges[-1] == np.inf
        # Threshold should be somewhere between the two distributions
        assert 2500 < edges[1] < 4500

    def test_max_bins_3(self):
        """With max_bins=3, should produce 4 edges."""
        from src.preprocess_improved import learn_income_bins

        rng = np.random.RandomState(42)
        n = 3000
        income = np.concatenate(
            [
                rng.normal(1000, 200, n // 3),
                rng.normal(3000, 200, n // 3),
                rng.normal(6000, 200, n // 3),
            ]
        )
        bad = np.concatenate([np.ones(n // 3), np.zeros(n // 3), np.zeros(n // 3)])
        data = pd.DataFrame({"income_t1_m": income, "early_bad": bad})

        edges = learn_income_bins(data, max_bins=3, min_samples_leaf=100)
        assert len(edges) >= 3  # At least 3 edges (2 bins)
        assert len(edges) <= 4  # At most 4 edges (3 bins)

    def test_missing_column_raises(self):
        """Should raise ValueError if required columns are missing."""
        from src.preprocess_improved import learn_income_bins

        data = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing required columns"):
            learn_income_bins(data)


# =============================================================================
# assess_binning_gini Tests
# =============================================================================


class TestAssessBinningGini:
    def test_basic_gini(self):
        """Gini retention should be between 0 and 100%."""
        from src.preprocess_improved import assess_binning_gini

        rng = np.random.RandomState(42)
        n = 1000
        raw = rng.uniform(0, 100, n)
        target = (raw > 50).astype(int)
        binned = (raw > 50).astype(int) + 1

        data = pd.DataFrame({"raw": raw, "binned": binned, "target": target})
        result = assess_binning_gini(data, "raw", "binned", "target")

        assert "gini_raw" in result
        assert "gini_binned" in result
        assert "gini_retention_pct" in result
        assert result["gini_raw"] > 0
        assert result["gini_retention_pct"] > 0

    def test_single_class_target(self):
        """Should handle single-class target gracefully."""
        from src.preprocess_improved import assess_binning_gini

        data = pd.DataFrame({"raw": [1, 2, 3], "binned": [1, 1, 2], "target": [0, 0, 0]})
        result = assess_binning_gini(data, "raw", "binned", "target")
        assert result["gini_raw"] == 0.0


# =============================================================================
# Swap-In Constraint Tests
# =============================================================================


def _make_summary_2d_swapin(n_var0=3, n_var1=4):
    """Create a 2D summary with controlled swap-in data for constraint testing.

    Cells with higher var1 index have disproportionately high repesca share,
    so a production-share constraint should reject some of them.
    """
    rng = np.random.RandomState(42)
    rows = []
    for v0 in range(1, n_var0 + 1):
        for v1 in range(1, n_var1 + 1):
            amt = rng.uniform(5000, 10000)
            production = rng.uniform(20000, 50000)
            # Higher var1 → higher repesca share (up to 80%)
            rep_frac = 0.1 + 0.7 * (v1 - 1) / max(n_var1 - 1, 1)
            risk_base = rng.uniform(0.01, 0.05)
            risk_rep = risk_base * (1 + 0.5 * v1)  # repesca risk escalates with v1
            rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "todu_30ever_h6": risk_base * amt / 7,
                    "todu_amt_pile_h6": amt,
                    "oa_amt_h0": production,
                    "todu_30ever_h6_boo": risk_base * amt / 7 * (1 - rep_frac),
                    "todu_amt_pile_h6_boo": amt * (1 - rep_frac),
                    "oa_amt_h0_boo": production * (1 - rep_frac),
                    "todu_30ever_h6_rep": risk_rep * amt * rep_frac / 7,
                    "todu_amt_pile_h6_rep": amt * rep_frac,
                    "oa_amt_h0_rep": production * rep_frac,
                }
            )
    return pd.DataFrame(rows)


class TestSwapInConstraints:
    """Tests for max_swapin_production_pct and max_swapin_risk MILP constraints."""

    def test_swapin_production_constraint_limits_repesca_share(self):
        """Accepted cells should respect the swap-in production fraction cap."""
        df = _make_summary_2d_swapin(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        # Solve without constraint — should accept most/all cells
        mask_unconstrained = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        assert mask_unconstrained is not None

        # Solve with tight swap-in production cap (20%)
        mask_constrained = milp_solve_cutoffs(
            grid, target_risk=50.0, inv_vars=[], multiplier=7, max_swapin_production_pct=20.0
        )
        assert mask_constrained is not None

        # Verify the constraint is satisfied
        cell = grid.cell_data
        accepted = mask_constrained.astype(bool)
        total_prod = (cell.loc[accepted, "oa_amt_h0"]).sum()
        rep_prod = (cell.loc[accepted, "oa_amt_h0_rep"]).sum()
        if total_prod > 0:
            actual_pct = rep_prod / total_prod * 100
            assert actual_pct <= 20.0 + 1e-6, f"Swap-in production share {actual_pct:.1f}% exceeds 20% cap"

    def test_swapin_risk_constraint_limits_repesca_risk(self):
        """Accepted cells should respect the swap-in risk cap."""
        df = _make_summary_2d_swapin(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        # Solve with swap-in risk cap (e.g. 5%)
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7, max_swapin_risk=5.0)
        assert mask is not None

        # Verify the constraint: multiplier * sum(rep_t30) / sum(rep_tamt) <= max_risk/100
        cell = grid.cell_data
        accepted = mask.astype(bool)
        rep_t30 = cell.loc[accepted, "todu_30ever_h6_rep"].sum()
        rep_tamt = cell.loc[accepted, "todu_amt_pile_h6_rep"].sum()
        if rep_tamt > 0:
            actual_risk = 7.0 * rep_t30 / rep_tamt * 100
            assert actual_risk <= 5.0 + 1e-6, f"Swap-in risk {actual_risk:.2f}% exceeds 5% cap"

    def test_swapin_constraints_none_has_no_effect(self):
        """None constraints should produce same result as omitting them."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        mask_default = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        mask_none = milp_solve_cutoffs(
            grid, target_risk=20.0, inv_vars=[], multiplier=7, max_swapin_production_pct=None, max_swapin_risk=None
        )
        assert mask_default is not None
        assert mask_none is not None
        np.testing.assert_array_equal(mask_default, mask_none)

    def test_swapin_constraint_makes_infeasible(self):
        """Overly tight swap-in constraint should return None or all-zero mask."""
        df = _make_summary_2d_swapin(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        # Extremely tight: 0% swap-in production allowed
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7, max_swapin_production_pct=0.0)
        # Either infeasible (None) or no repesca production in accepted cells
        if mask is not None:
            cell = grid.cell_data
            accepted = mask.astype(bool)
            rep_prod = cell.loc[accepted, "oa_amt_h0_rep"].sum()
            assert rep_prod <= 1e-6

    def test_pareto_frontier_with_swapin_constraints(self):
        """Swap-in params should flow through to Pareto sweep."""
        df = _make_summary_2d_swapin(3, 4)

        # Unconstrained Pareto
        pareto_unc, _, _ = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=10,
        )

        # Constrained Pareto with tight swap-in production cap
        pareto_con, _, masks_con = trace_pareto_frontier(
            df,
            variables=["var0", "var1"],
            inv_vars=[],
            multiplier=7,
            indicators=["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"],
            n_points=10,
            max_swapin_production_pct=15.0,
        )

        assert not pareto_unc.empty
        assert not pareto_con.empty
        # Constrained frontier should have equal or less max production
        assert pareto_con["oa_amt_h0"].max() <= pareto_unc["oa_amt_h0"].max() + 1e-6


# =============================================================================
# N>2 Fixed Cutoff Mask Tests
# =============================================================================


def _make_summary_3d(n0=3, n1=3, n2=2):
    """Create a 3D aggregated summary DataFrame for testing."""
    rng = np.random.RandomState(42)
    rows = []
    for v0 in range(1, n0 + 1):
        for v1 in range(1, n1 + 1):
            for v2 in range(1, n2 + 1):
                rows.append(
                    {
                        "var0": float(v0),
                        "var1": float(v1),
                        "var2": float(v2),
                        "todu_30ever_h6": rng.uniform(5, 50),
                        "todu_amt_pile_h6": rng.uniform(100, 500),
                        "oa_amt_h0": rng.uniform(1000, 10000),
                        "todu_30ever_h6_boo": rng.uniform(5, 50),
                        "todu_amt_pile_h6_boo": rng.uniform(100, 500),
                        "oa_amt_h0_boo": rng.uniform(1000, 10000),
                    }
                )
    return pd.DataFrame(rows)


class TestCreateFixedCutoffMask:
    """Tests for create_fixed_cutoff_mask (N>2 fixed cutoffs, per-var0-bin paired cutoffs)."""

    def test_basic_3d_paired(self):
        """Per-var0-bin paired cutoffs: var0=[1,2], var1 cutoffs=[2,1], var2 cutoffs=[2,1]."""
        df = _make_summary_3d(3, 3, 2)
        # var0 bin 1 → accept var1<=2, var2<=2 (all var2 bins)
        # var0 bin 2 → accept var1<=1, var2<=1
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [2.0, 1.0],
            "var2": [2.0, 1.0],
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

        assert grid.n_cells == 3 * 3 * 2
        assert mask.shape == (grid.n_cells,)
        # var0=1: var1 in {1,2}, var2 in {1,2} → 2*2 = 4 cells
        # var0=2: var1 in {1}, var2 in {1} → 1 cell
        # var0=3: not listed → 0 cells
        assert mask.sum() == 5

        # Verify specific cells
        assert mask[grid.cell_index[(1.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(1.0, 2.0, 2.0)]] == 1
        assert mask[grid.cell_index[(2.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(2.0, 2.0, 1.0)]] == 0  # var1=2 > cutoff 1
        assert mask[grid.cell_index[(3.0, 1.0, 1.0)]] == 0  # var0=3 not listed

    def test_all_accepted(self):
        """Max cutoffs for all var0 bins yields all matching cells accepted."""
        df = _make_summary_3d(2, 2, 2)
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [2.0, 2.0],  # cutoff = max bin for both
            "var2": [2.0, 2.0],
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

        assert mask.sum() == grid.n_cells

    def test_none_accepted(self):
        """Empty lists yield all-zeros mask."""
        df = _make_summary_3d(2, 2, 2)
        fixed_cutoffs = {
            "var0": [],
            "var1": [],
            "var2": [],
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

        assert mask.sum() == 0

    def test_missing_variable_raises(self):
        """Missing variable key in fixed_cutoffs (no dict entries) raises ValueError."""
        df = _make_summary_3d(2, 2, 2)
        fixed_cutoffs = {
            "var0": [1.0],
            "var1": [1.0],
            # var2 missing
        }
        with pytest.raises(ValueError, match="Missing"):
            create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

    def test_unequal_lengths_raises(self):
        """Unequal-length cutoff lists raise ValueError."""
        df = _make_summary_3d(3, 3, 2)
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": [1.0, 2.0],  # length 2 != 3
            "var2": [1.0, 2.0, 1.0],
        }
        with pytest.raises(ValueError, match="equal length"):
            create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

    def test_evaluate_solution_integration(self):
        """Mask from create_fixed_cutoff_mask works with evaluate_solution."""
        df = _make_summary_3d(3, 3, 2)
        # var0=[1,2], var1 cutoff=[3,2], var2 cutoff=[2,2]
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [3.0, 2.0],
            "var2": [2.0, 2.0],
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)
        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], 7)

        assert "oa_amt_h0" in kpis
        assert "b2_ever_h6" in kpis
        assert kpis["oa_amt_h0"] > 0
        assert kpis["b2_ever_h6"] > 0

    def test_inv_vars_direction(self):
        """inv_vars flips acceptance to >= instead of <=."""
        df = _make_summary_3d(3, 3, 2)
        # var0=[1,2], var1 cutoff=[2,2]
        # Without inv: var1<=2 → accepts var1 in {1,2}
        # With inv on var1: var1>=2 → accepts var1 in {2,3}
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": [2.0, 2.0],
            "var2": [2.0, 2.0],
        }
        _, mask_no_inv = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df, inv_vars=[])
        _, mask_with_inv = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df, inv_vars=["var1"])

        # Without inv: var1 in {1,2} accepted; with inv: var1 in {2,3} accepted
        # They should differ (var1=1 vs var1=3 swap)
        assert not np.array_equal(mask_no_inv, mask_with_inv)

        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        # With inv_vars=["var1"], var1=1 should be rejected (1 < 2), var1=3 accepted (3 >= 2)
        assert mask_with_inv[grid.cell_index[(1.0, 1.0, 1.0)]] == 0
        assert mask_with_inv[grid.cell_index[(1.0, 3.0, 1.0)]] == 1

    def test_strict_validation_unknown_bins(self):
        """strict_validation=True raises on unknown var0 bin values."""
        df = _make_summary_3d(2, 2, 2)
        fixed_cutoffs = {
            "var0": [1.0, 99.0],  # 99.0 doesn't exist in data
            "var1": [1.0, 1.0],
            "var2": [1.0, 1.0],
        }
        with pytest.raises(ValueError, match="don't exist in data bins"):
            create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df, strict_validation=True)

    def test_meta_keys_skipped(self):
        """Meta keys like strict_validation in fixed_cutoffs dict are skipped."""
        df = _make_summary_3d(2, 2, 2)
        fixed_cutoffs = {
            "var0": [1.0],
            "var1": [1.0],
            "var2": [1.0],
            "strict_validation": False,
            "run_all_scenarios": False,
        }
        # Should not raise (meta keys should be ignored by _validate_nd_cutoff_structure)
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)
        assert mask.sum() == 1

    def test_single_var0_bin(self):
        """Single var0 bin with cutoffs for secondary vars."""
        df = _make_summary_3d(3, 3, 2)
        fixed_cutoffs = {
            "var0": [2.0],
            "var1": [2.0],
            "var2": [1.0],
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)
        # Only var0=2 cells, var1<=2, var2<=1 → var1 in {1,2}, var2 in {1} → 2 cells
        assert mask.sum() == 2
        assert mask[grid.cell_index[(2.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(2.0, 2.0, 1.0)]] == 1
        assert mask[grid.cell_index[(2.0, 3.0, 1.0)]] == 0

    def test_matrix_cutoff_basic(self):
        """Matrix cutoffs: different var1 schedules per conditioning var (var2) value."""
        df = _make_summary_3d(3, 3, 2)  # var0: 1-3, var1: 1-3, var2: 1-2
        # var2 is conditioning variable (absent from fixed_cutoffs).
        # When var2=1: accept var1 <= [3, 2, 1] for var0=[1,2,3]
        # When var2=2: accept var1 <= [2, 1, 1] for var0=[1,2,3]
        fixed_cutoffs = {
            "var0": [1.0, 2.0, 3.0],
            "var1": {
                "1": [3, 2, 1],  # when var2=1
                "2": [2, 1, 1],  # when var2=2
            },
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df)

        # var0=1, var2=1: var1 <= 3 → {1,2,3} → 3 cells
        assert mask[grid.cell_index[(1.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(1.0, 2.0, 1.0)]] == 1
        assert mask[grid.cell_index[(1.0, 3.0, 1.0)]] == 1
        # var0=1, var2=2: var1 <= 2 → {1,2} → 2 cells
        assert mask[grid.cell_index[(1.0, 1.0, 2.0)]] == 1
        assert mask[grid.cell_index[(1.0, 2.0, 2.0)]] == 1
        assert mask[grid.cell_index[(1.0, 3.0, 2.0)]] == 0
        # var0=2, var2=1: var1 <= 2 → {1,2} → 2 cells
        assert mask[grid.cell_index[(2.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(2.0, 2.0, 1.0)]] == 1
        assert mask[grid.cell_index[(2.0, 3.0, 1.0)]] == 0
        # var0=2, var2=2: var1 <= 1 → {1} → 1 cell
        assert mask[grid.cell_index[(2.0, 1.0, 2.0)]] == 1
        assert mask[grid.cell_index[(2.0, 2.0, 2.0)]] == 0
        # var0=3, var2=1: var1 <= 1 → {1} → 1 cell
        assert mask[grid.cell_index[(3.0, 1.0, 1.0)]] == 1
        assert mask[grid.cell_index[(3.0, 2.0, 1.0)]] == 0
        # var0=3, var2=2: var1 <= 1 → {1} → 1 cell
        assert mask[grid.cell_index[(3.0, 1.0, 2.0)]] == 1
        assert mask[grid.cell_index[(3.0, 2.0, 2.0)]] == 0
        # Total: 3+2+2+1+1+1 = 10
        assert mask.sum() == 10

    def test_matrix_cutoff_inv_vars(self):
        """Matrix cutoffs with inv_vars flips direction to >=."""
        df = _make_summary_3d(2, 3, 2)  # var0: 1-2, var1: 1-3, var2: 1-2
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": {
                "1": [2, 3],  # when var2=1: var1 >= 2 for var0=1, var1 >= 3 for var0=2
                "2": [1, 2],  # when var2=2: var1 >= 1 for var0=1, var1 >= 2 for var0=2
            },
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df, inv_vars=["var1"])

        # var0=1, var2=1: var1 >= 2 → {2,3}
        assert mask[grid.cell_index[(1.0, 1.0, 1.0)]] == 0
        assert mask[grid.cell_index[(1.0, 2.0, 1.0)]] == 1
        assert mask[grid.cell_index[(1.0, 3.0, 1.0)]] == 1
        # var0=1, var2=2: var1 >= 1 → {1,2,3}
        assert mask[grid.cell_index[(1.0, 1.0, 2.0)]] == 1
        assert mask[grid.cell_index[(1.0, 2.0, 2.0)]] == 1
        assert mask[grid.cell_index[(1.0, 3.0, 2.0)]] == 1
        # var0=2, var2=1: var1 >= 3 → {3}
        assert mask[grid.cell_index[(2.0, 1.0, 1.0)]] == 0
        assert mask[grid.cell_index[(2.0, 2.0, 1.0)]] == 0
        assert mask[grid.cell_index[(2.0, 3.0, 1.0)]] == 1
        # var0=2, var2=2: var1 >= 2 → {2,3}
        assert mask[grid.cell_index[(2.0, 1.0, 2.0)]] == 0
        assert mask[grid.cell_index[(2.0, 2.0, 2.0)]] == 1
        assert mask[grid.cell_index[(2.0, 3.0, 2.0)]] == 1

    def test_matrix_mixed_flat_and_dict(self):
        """4-var with mixed flat and dict cutoff entries."""
        # var0: 1-2, var1: 1-3, var2: 1-2, var3: 1-2
        rng = np.random.RandomState(42)
        rows = []
        for v0 in range(1, 3):
            for v1 in range(1, 4):
                for v2 in range(1, 3):
                    for v3 in range(1, 3):
                        rows.append(
                            {
                                "var0": float(v0),
                                "var1": float(v1),
                                "var2": float(v2),
                                "var3": float(v3),
                                "todu_30ever_h6": rng.uniform(5, 50),
                                "todu_amt_pile_h6": rng.uniform(100, 500),
                                "oa_amt_h0": rng.uniform(1000, 10000),
                                "todu_30ever_h6_boo": rng.uniform(5, 50),
                                "todu_amt_pile_h6_boo": rng.uniform(100, 500),
                                "oa_amt_h0_boo": rng.uniform(1000, 10000),
                            }
                        )
        df = pd.DataFrame(rows)

        # var3 is conditioning variable (absent), var1 is matrix, var2 is flat
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": {
                "1": [3, 2],  # when var3=1
                "2": [2, 1],  # when var3=2
            },
            "var2": [2, 1],  # flat: cutoff per var0 bin
        }
        grid, mask = create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2", "var3"], df)

        # var0=1, var3=1: var1<=3, var2<=2 → all var1, all var2 → 3*2=6
        # var0=1, var3=2: var1<=2, var2<=2 → var1 in {1,2}, all var2 → 2*2=4
        # var0=2, var3=1: var1<=2, var2<=1 → var1 in {1,2}, var2 in {1} → 2*1=2
        # var0=2, var3=2: var1<=1, var2<=1 → var1 in {1}, var2 in {1} → 1*1=1
        # Total: 6+4+2+1 = 13
        assert mask.sum() == 13

        # Spot check
        assert mask[grid.cell_index[(1.0, 3.0, 2.0, 1.0)]] == 1  # var3=1, var1=3 <=3 ok
        assert mask[grid.cell_index[(1.0, 3.0, 2.0, 2.0)]] == 0  # var3=2, var1=3 > 2 fail
        assert mask[grid.cell_index[(2.0, 1.0, 1.0, 1.0)]] == 1  # var3=1, var1=1<=2, var2=1<=1 ok
        assert mask[grid.cell_index[(2.0, 1.0, 2.0, 1.0)]] == 0  # var2=2 > 1 fail

    def test_matrix_conditioning_var_not_in_data(self):
        """Strict validation raises when conditioning var keys don't exist in data."""
        df = _make_summary_3d(2, 2, 2)  # var2 has values 1, 2
        fixed_cutoffs = {
            "var0": [1.0, 2.0],
            "var1": {
                "5": [2, 1],  # var2=5 doesn't exist in data
                "6": [2, 1],  # var2=6 doesn't exist in data
            },
        }
        with pytest.raises(ValueError, match="don't exist in data bins"):
            create_fixed_cutoff_mask(fixed_cutoffs, ["var0", "var1", "var2"], df, strict_validation=True)


class TestValidateNdCutoffStructure:
    """Tests for _validate_nd_cutoff_structure helper."""

    def test_basic(self):
        """Basic extraction of per-variable cutoff lists."""
        result, cond_var, anchor = _validate_nd_cutoff_structure(
            {"var0": [1, 2], "var1": [3, 4], "strict_validation": False},
            ["var0", "var1"],
        )
        assert result == {"var0": [1.0, 2.0], "var1": [3.0, 4.0]}
        assert cond_var is None
        assert anchor == "var0"

    def test_missing_variable_no_dict(self):
        """Missing variable without dict entries raises ValueError."""
        with pytest.raises(ValueError, match="Missing"):
            _validate_nd_cutoff_structure({"var0": [1]}, ["var0", "var1"])

    def test_non_list_raises(self):
        """Non-list/non-dict value raises ValueError."""
        with pytest.raises(ValueError, match="must be a list"):
            _validate_nd_cutoff_structure({"var0": 1.0, "var1": [1]}, ["var0", "var1"])

    def test_unequal_lengths_raises(self):
        """Unequal-length lists raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            _validate_nd_cutoff_structure(
                {"var0": [1, 2, 3], "var1": [1, 2], "var2": [1, 2, 3]},
                ["var0", "var1", "var2"],
            )

    def test_dict_entry_matrix(self):
        """Dict entry parsed as matrix cutoffs with conditioning variable detected."""
        result, cond_var, anchor = _validate_nd_cutoff_structure(
            {
                "var0": [1, 2, 3],
                "var1": {"1": [10, 8, 5], "2": [10, 6, 3]},
                # var2 missing → conditioning variable
            },
            ["var0", "var1", "var2"],
        )
        assert cond_var == "var2"
        assert anchor == "var0"
        assert isinstance(result["var1"], dict)
        assert result["var1"][1.0] == [10.0, 8.0, 5.0]
        assert result["var1"][2.0] == [10.0, 6.0, 3.0]
        # Conditioning var gets sorted union of keys
        assert result["var2"] == [1.0, 2.0]

    def test_dict_var0_auto_anchor(self):
        """Dict in variables[0] auto-detects a flat-list variable as anchor."""
        result, cond_var, anchor = _validate_nd_cutoff_structure(
            {
                "var0": {"1": [10, 8, 5], "2": [10, 6, 3]},
                "var1": [1, 2, 3],
                # var2 missing → conditioning variable
            },
            ["var0", "var1", "var2"],
        )
        assert anchor == "var1"
        assert cond_var == "var2"
        assert isinstance(result["var0"], dict)
        assert result["var0"][1.0] == [10.0, 8.0, 5.0]

    def test_dict_with_nested_meta_keys(self):
        """Meta keys inside nested dicts are filtered out."""
        result, cond_var, anchor = _validate_nd_cutoff_structure(
            {
                "var0": {"1": [10, 8], "2": [10, 6], "strict_validation": True, "run_all_scenarios": True},
                "var1": [1, 2],
                # var2 missing → conditioning variable
            },
            ["var0", "var1", "var2"],
        )
        assert anchor == "var1"
        assert cond_var == "var2"
        assert isinstance(result["var0"], dict)
        assert 1.0 in result["var0"]
        assert 2.0 in result["var0"]
        assert len(result["var0"]) == 2  # meta keys excluded

    def test_mixed_flat_and_dict(self):
        """Mix of flat lists and dict entries with one conditioning variable."""
        result, cond_var, anchor = _validate_nd_cutoff_structure(
            {
                "var0": [1, 2],
                "var1": {"1": [5, 3], "2": [4, 2]},
                "var2": [10, 8],
                # var3 missing → conditioning variable
            },
            ["var0", "var1", "var2", "var3"],
        )
        assert cond_var == "var3"
        assert anchor == "var0"
        assert isinstance(result["var1"], dict)
        assert result["var2"] == [10.0, 8.0]
        assert result["var3"] == [1.0, 2.0]

    def test_dict_no_missing_var_raises(self):
        """Dict entry with no missing conditioning variable raises ValueError."""
        with pytest.raises(ValueError, match="no conditioning variable is missing"):
            _validate_nd_cutoff_structure(
                {
                    "var0": [1, 2],
                    "var1": {"1": [5, 3]},
                    "var2": [10, 8],
                },
                ["var0", "var1", "var2"],
            )

    def test_dict_multiple_missing_vars_raises(self):
        """Dict entry with multiple missing variables raises ValueError."""
        with pytest.raises(ValueError, match="multiple variables are missing"):
            _validate_nd_cutoff_structure(
                {
                    "var0": [1, 2],
                    "var1": {"1": [5, 3]},
                    # var2 and var3 both missing
                },
                ["var0", "var1", "var2", "var3"],
            )

    def test_dict_row_length_mismatch_raises(self):
        """Dict row with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="has length 2.*expected 3"):
            _validate_nd_cutoff_structure(
                {
                    "var0": [1, 2, 3],
                    "var1": {"1": [5, 3]},  # length 2 != 3
                },
                ["var0", "var1", "var2"],
            )


# =============================================================================
# N>2 mask_to_cutoffs Tests
# =============================================================================


class TestMaskToCutoffsNd:
    """Tests for mask_to_cutoffs with N>2 variables."""

    def test_3d_cell_dict(self):
        """N>2 path returns _cells dict with correct entries."""
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.zeros(grid.n_cells, dtype=int)
        # Accept cells (1,1,1) and (1,2,1)
        mask[grid.cell_index[(1.0, 1.0, 1.0)]] = 1
        mask[grid.cell_index[(1.0, 2.0, 1.0)]] = 1

        result = mask_to_cutoffs(mask, grid, inv_vars=[])

        assert "_cells" in result
        cells = result["_cells"]
        assert cells[(1.0, 1.0, 1.0)] == 1
        assert cells[(1.0, 2.0, 1.0)] == 1
        assert cells[(2.0, 1.0, 1.0)] == 0

    def test_3d_marginals(self):
        """N>2 path returns per-dimension marginals."""
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.zeros(grid.n_cells, dtype=int)
        mask[grid.cell_index[(1.0, 1.0, 1.0)]] = 1

        result = mask_to_cutoffs(mask, grid, inv_vars=[])

        assert "_marginal_var0" in result
        assert result["_marginal_var0"][1.0] == 1.0
        assert result["_marginal_var0"][2.0] == 0.0

        assert "_marginal_var1" in result
        assert result["_marginal_var1"][1.0] == 1.0
        assert result["_marginal_var1"][2.0] == 0.0

        assert "_marginal_var2" in result
        assert result["_marginal_var2"][1.0] == 1.0
        assert result["_marginal_var2"][2.0] == 0.0

    def test_3d_conditional_cutoff_last_dim(self):
        """N>2 path returns conditional cutoff for last dimension."""
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.zeros(grid.n_cells, dtype=int)
        # Accept (1,1,1), (1,1,2), (1,2,1)
        mask[grid.cell_index[(1.0, 1.0, 1.0)]] = 1
        mask[grid.cell_index[(1.0, 1.0, 2.0)]] = 1
        mask[grid.cell_index[(1.0, 2.0, 1.0)]] = 1

        result = mask_to_cutoffs(mask, grid, inv_vars=[])

        # Conditional cutoff for var2 (last dim, non-inverted → max accepted)
        assert "var2" in result
        cond = result["var2"]
        # For prefix (1.0, 1.0): max accepted var2 = 2.0
        assert cond[(1.0, 1.0)] == 2.0
        # For prefix (1.0, 2.0): max accepted var2 = 1.0
        assert cond[(1.0, 2.0)] == 1.0
        # For prefix (2.0, 1.0): no accepted → -inf
        assert cond[(2.0, 1.0)] == float("-inf")

    def test_3d_conditional_cutoff_inverted(self):
        """N>2 conditional cutoff with inverted last dimension."""
        df = _make_summary_3d(2, 2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.ones(grid.n_cells, dtype=int)  # all accepted

        result = mask_to_cutoffs(mask, grid, inv_vars=["var2"])

        # Inverted → min accepted (which is 1.0 for all prefixes)
        for prefix in result["var2"]:
            assert result["var2"][prefix] == 1.0


# =============================================================================
# Phantom cell exclusion in MILP
# =============================================================================


class TestPhantomCellExclusion:
    """Tests that unobserved (phantom) cells are excluded from optimization."""

    def test_milp_rejects_phantom_cells(self):
        """MILP must not accept cells that were unobserved in data."""
        df = _make_summary_2d(3, 3)
        # Remove 2 cells to create phantoms
        df = df.iloc[:-2]
        grid = CellGrid.from_summary(df, ["var0", "var1"])

        assert (~grid.observed).sum() == 2  # 2 phantom cells

        # Solve with a generous risk budget (accept everything feasible)
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        if mask is not None:
            # Phantom cells must be rejected
            phantom_indices = np.where(~grid.observed)[0]
            for idx in phantom_indices:
                assert mask[idx] == 0, f"Phantom cell {idx} was accepted by MILP"

    def test_full_grid_no_phantom_exclusion(self):
        """When all cells are observed, none should be excluded."""
        df = _make_summary_2d(2, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        assert grid.observed.all()

        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7)
        if mask is not None:
            # All cells could potentially be accepted
            assert mask.sum() > 0


# =============================================================================
# Legacy enumeration limit
# =============================================================================


class TestLegacyEnumerationLimit:
    """Tests for the hard limit on legacy 2-var combinatorial enumeration."""

    def test_small_enumeration_succeeds(self):
        """Small problem should work fine."""
        values_var0 = [1.0, 2.0, 3.0]
        values_var1 = [1.0, 2.0]
        result = get_fact_sol(values_var0, values_var1)
        assert len(result) > 0

    def test_huge_enumeration_raises(self):
        """Problem exceeding limit should raise RuntimeError."""
        # 30 bins × 30 cutoff values → way over 500K
        values_var0 = list(range(1, 31))
        values_var1 = list(range(1, 31))
        with pytest.raises(RuntimeError, match="Legacy enumeration"):
            get_fact_sol(values_var0, values_var1)


# =============================================================================
# pareto_keep_indices Tests (audit #13 — shared canonical sweep)
# =============================================================================


class TestParetoKeepIndices:
    def test_strictly_increasing_keeps_all(self):
        assert pareto_keep_indices([10.0, 20.0, 30.0]) == [0, 1, 2]

    def test_non_increasing_dropped(self):
        # production sorted by ascending risk; flat/decreasing tail is dominated
        assert pareto_keep_indices([100.0, 90.0, 200.0, 200.0, 150.0]) == [0, 2]

    def test_equal_values_keep_first_only(self):
        assert pareto_keep_indices([50.0, 50.0, 50.0]) == [0]

    def test_strict_no_epsilon(self):
        # a sub-1e-4 gain is strictly greater -> kept (canonical sweep has no epsilon)
        assert pareto_keep_indices([100.0, 100.00005, 100.0001]) == [0, 1, 2]

    def test_empty(self):
        assert pareto_keep_indices([]) == []

    def test_matches_manual_sweep(self):
        rng = np.random.RandomState(0)
        prod = rng.uniform(0, 1000, 200).tolist()
        prev = float("-inf")
        expected = []
        for i, p in enumerate(prod):
            if p > prev:
                expected.append(i)
                prev = p
        assert pareto_keep_indices(prod) == expected


class TestParetoTieHandlingSort:
    """The frontier sweep sorts by (risk ASC, production DESC) before pareto_keep_indices.
    A risk-only sort left the within-tie order arbitrary, so a lower-production point at the
    SAME risk (dominated) could survive. This verifies the (risk, -prod) lexsort drops it
    regardless of input order — the exact composition the fixed code uses."""

    @staticmethod
    def _frontier(risk, prod):
        risk = np.asarray(risk, dtype=float)
        prod = np.asarray(prod, dtype=float)
        order = np.lexsort((-prod, risk))  # risk asc, prod desc — the fixed sort
        keep = pareto_keep_indices(prod[order])
        kept = order[keep]
        return sorted(zip(risk[kept].tolist(), prod[kept].tolist()))

    def test_dominated_same_risk_point_dropped_regardless_of_order(self):
        # Two points at risk 1.0: prod 50 (dominated) and prod 100. Only (1.0, 100) is efficient.
        a = self._frontier([1.0, 1.0, 1.1], [50.0, 100.0, 120.0])
        b = self._frontier([1.0, 1.0, 1.1], [100.0, 50.0, 120.0])  # reversed within the tie
        assert a == b == [(1.0, 100.0), (1.1, 120.0)]

    def test_equal_production_at_higher_risk_dropped(self):
        # (1.1, 100) is dominated by (1.0, 100): same production, higher risk.
        assert self._frontier([1.0, 1.1], [100.0, 100.0]) == [(1.0, 100.0)]

    def test_old_risk_only_sort_was_order_dependent(self):
        # Documents the bug: a risk-only sort leaves the within-tie order intact, so the
        # dominated (1.0, 50) point survives or not depending purely on input order.
        risk = np.array([1.0, 1.0, 1.1])

        def risk_only(prod):
            prod = np.asarray(prod, dtype=float)
            order = np.argsort(risk, kind="stable")  # risk-only (old behavior)
            return sorted(prod[order][pareto_keep_indices(prod[order])].tolist())

        r1 = risk_only([50.0, 100.0, 120.0])  # 50 before 100 in the tie -> 50 (dominated) survives
        r2 = risk_only([100.0, 50.0, 120.0])  # 100 before 50 -> 50 dropped
        assert r1 != r2 and 50.0 in r1  # order-dependent, and the dominated point can survive
        # The fixed (risk ASC, prod DESC) sort is order-invariant and always drops it:
        assert (
            self._frontier([1.0, 1.0, 1.1], [50.0, 100.0, 120.0])
            == self._frontier([1.0, 1.0, 1.1], [100.0, 50.0, 120.0])
            == [(1.0, 100.0), (1.1, 120.0)]
        )


class TestInvVar1KpiSide:
    """Audit #37: with variables[1] inverted (higher bin = safer), the KPI
    evaluation must sum the var1 >= cutoff side — the old call sites defaulted
    inv_var1=False and summed var1 <= cutoff, so the frontier KPIs described a
    different policy than the audit/bootstrap/MR classification."""

    def test_kpi_sums_geq_side_when_inverted(self):
        df = _make_summary_2d(2, 3)
        # one solution: cut var1 at 2 for both var0 bins
        df_v = pd.DataFrame({"sol_fac": [0], 1: [2.0], 2: [2.0]})

        kpi_inv = kpi_of_fact_sol(
            df_v=df_v,
            values_var0=[1, 2],
            data_sumary_desagregado=df,
            variables=["var0", "var1"],
            indicadores=["oa_amt_h0"],
            inv_var1=True,
            multiplier=7,
        )
        expected_geq = df.loc[df["var1"] >= 2.0, "oa_amt_h0"].sum()
        expected_leq = df.loc[df["var1"] <= 2.0, "oa_amt_h0"].sum()
        assert expected_geq != expected_leq  # the two sides genuinely differ
        assert kpi_inv["oa_amt_h0"].iloc[0] == pytest.approx(expected_geq)

        # and the two directions partition the total at adjacent cuts
        kpi_norm = kpi_of_fact_sol(
            df_v=pd.DataFrame({"sol_fac": [0], 1: [1.0], 2: [1.0]}),
            values_var0=[1, 2],
            data_sumary_desagregado=df,
            variables=["var0", "var1"],
            indicadores=["oa_amt_h0"],
            inv_var1=False,
            multiplier=7,
        )
        total = df["oa_amt_h0"].sum()
        assert kpi_norm["oa_amt_h0"].iloc[0] + kpi_inv["oa_amt_h0"].iloc[0] == pytest.approx(total)

    def test_get_fact_sol_inverted_has_reject_all_sentinel(self):
        """Under var1 >= cut, reject-the-whole-bin needs a sentinel ABOVE
        max(var1); the old [0] sentinel meant accept-all under inversion and
        reject-all was unrepresentable."""
        values_var1 = [1.0, 2.0, 3.0]
        df_v = get_fact_sol([1.0, 2.0], values_var1, inv_var1=True)
        bin_cols = [c for c in df_v.columns if c != "sol_fac"]
        assert df_v[bin_cols].values.max() == 4.0  # max(var1) + 1 sentinel present
        # a full reject-all solution exists
        assert (df_v[bin_cols] == 4.0).all(axis=1).any()
        # normal direction keeps the 0 sentinel
        df_n = get_fact_sol([1.0, 2.0], values_var1, inv_var1=False)
        assert (df_n[[c for c in df_n.columns if c != "sol_fac"]] == 0.0).all(axis=1).any()


# =============================================================================
# TestWarnUnenforceableSwapinCaps — silent-skip guard for swap-in caps
# =============================================================================


class TestWarnUnenforceableSwapinCaps:
    """Warn when a swap-in cap is configured but the required `_rep` columns are absent (the
    MILP/GA constraint is otherwise silently skipped and the cap never binds)."""

    def _warns(self, columns, prod_pct, risk):
        import logging

        records = []

        class _Sink(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        sink = _Sink(level=logging.WARNING)
        hid = logger.add(sink, level="WARNING", format="{message}")
        try:
            _warn_unenforceable_swapin_caps(columns, prod_pct, risk)
        finally:
            logger.remove(hid)
        return records

    def test_production_cap_without_rep_warns(self):
        msgs = self._warns(["oa_amt_h0", "todu_30ever_h6"], 5.0, None)
        assert any("swap-in production cap" in m for m in msgs)

    def test_risk_cap_without_rep_warns(self):
        msgs = self._warns(["oa_amt_h0_rep", "todu_30ever_h6"], None, 1.2)
        assert any("swap-in risk cap" in m for m in msgs)

    def test_no_warning_when_rep_present(self):
        cols = ["oa_amt_h0_rep", "todu_30ever_h6_rep", "todu_amt_pile_h6_rep"]
        assert self._warns(cols, 5.0, 1.2) == []

    def test_no_warning_when_caps_unset(self):
        assert self._warns(["oa_amt_h0"], None, None) == []


# =============================================================================
# TestGAFallbackH3 — GA fallback must carry H3 KPI columns when multiplier_h3 set
# =============================================================================


class TestGAFallbackH3:
    def _grid(self):
        return CellGrid.from_summary(_make_summary_2d(2, 2), ["var0", "var1"])

    # H3 raw columns are in `indicators` in the live config, so evaluate_solution sums them;
    # b2_ever_h3 then appears only when multiplier_h3 is also passed (the fix).
    _INDICATORS = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0", "todu_30ever_h3", "todu_amt_pile_h3"]

    def test_ga_fallback_includes_h3_when_multiplier_h3_set(self, monkeypatch):
        _install_dummy_pymoo(monkeypatch)
        pareto_df, _, masks = _ga_pareto_fallback(
            grid=self._grid(), inv_vars=[], multiplier=7, indicators=self._INDICATORS, n_points=3, multiplier_h3=4
        )
        assert masks
        assert "b2_ever_h3" in pareto_df.columns  # H3 KPI no longer dropped in the GA path

    def test_ga_fallback_omits_h3_without_multiplier_h3(self, monkeypatch):
        _install_dummy_pymoo(monkeypatch)
        pareto_df, _, _ = _ga_pareto_fallback(
            grid=self._grid(), inv_vars=[], multiplier=7, indicators=self._INDICATORS, n_points=3
        )
        assert "b2_ever_h3" not in pareto_df.columns  # unchanged when H3 not requested


# =============================================================================
# TestMilpTimeoutLogging — timeout (status 1) logged distinctly from infeasible
# =============================================================================


class TestMilpTimeoutLogging:
    def _grid(self):
        return CellGrid.from_summary(_make_summary_2d(2, 2), ["var0", "var1"])

    def _run(self, monkeypatch, status):
        import logging

        import src.optimization_utils as ou

        class _Res:
            success = False

            def __init__(self, st):
                self.status = st
                self.x = None

        monkeypatch.setattr(ou, "milp", lambda **kw: _Res(status))

        recs = []

        class _Sink(logging.Handler):
            def emit(self, record):
                recs.append(record.getMessage())

        hid = logger.add(_Sink(level=logging.WARNING), level="WARNING", format="{message}")
        try:
            mask = milp_solve_cutoffs(self._grid(), target_risk=5.0, inv_vars=[], multiplier=7.0)
        finally:
            logger.remove(hid)
        return mask, recs

    def test_timeout_status_logs_and_returns_none(self, monkeypatch):
        mask, recs = self._run(monkeypatch, status=1)
        assert mask is None
        assert any("time limit" in m for m in recs)

    def test_infeasible_status_no_timeout_warning(self, monkeypatch):
        mask, recs = self._run(monkeypatch, status=2)
        assert mask is None
        assert not any("time limit" in m for m in recs)
