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
    add_bin_columns,
    create_fixed_cutoff_solution,
    evaluate_solution,
    get_fact_sol,
    mask_to_cutoffs,
    milp_solve_cutoffs,
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
            rows.append(
                {
                    "var0": v0,
                    "var1": v1,
                    "todu_30ever_h6": risk * amt / 7,
                    "todu_amt_pile_h6": amt,
                    "oa_amt_h0": production,
                    "todu_30ever_h6_boo": risk * amt / 7 * 0.8,
                    "todu_amt_pile_h6_boo": amt * 0.8,
                    "oa_amt_h0_boo": production * 0.8,
                    "todu_30ever_h6_rep": risk * amt / 7 * 0.2,
                    "todu_amt_pile_h6_rep": amt * 0.2,
                    "oa_amt_h0_rep": production * 0.2,
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

    def test_3d_has_sol_fac_only(self):
        """For N>2, add_bin_columns should only add sol_fac."""
        df = _make_summary_3d(2, 3, 2)
        grid = CellGrid.from_summary(df, ["var0", "var1", "var2"])
        mask = np.ones(grid.n_cells, dtype=int)

        kpis = evaluate_solution(mask, grid, ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"], 7)
        pareto_df = pd.DataFrame([kpis])

        result = add_bin_columns(pareto_df, [mask], grid, inv_vars=[])
        assert "sol_fac" in result.columns


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
        # Fix cell (1,1) = index 0 as rejected
        mask = milp_solve_cutoffs(grid, target_risk=50.0, inv_vars=[], multiplier=7, fixed_cells={0: 0})
        assert mask is not None
        assert mask[0] == 0

    def test_empty_fixed_cells_same_as_none(self):
        """Empty dict should produce same result as no fixed_cells."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask_none = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        mask_empty = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7, fixed_cells={})
        assert mask_none is not None
        assert mask_empty is not None
        np.testing.assert_array_equal(mask_none, mask_empty)
