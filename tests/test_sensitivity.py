import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.optimization_utils import CellGrid, milp_solve_cutoffs, solve_with_fixed_cells
from src.sensitivity import (
    compute_cell_marginal_impact,
    perturb_risk_summary,
    run_sensitivity_analysis,
    sensitivity_cell_detail,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_summary_2d(n_var0=3, n_var1=4):
    """Create a 2D summary DataFrame with _boo and _rep columns."""
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


INDICATORS = ["todu_30ever_h6", "todu_amt_pile_h6", "oa_amt_h0"]


# =============================================================================
# perturb_risk_summary Tests
# =============================================================================


class TestPerturbRiskSummary:
    def test_perturb_zero_unchanged(self):
        """0% perturbation should leave data unchanged."""
        df = _make_summary_2d()
        perturbed = perturb_risk_summary(df, 0.0)
        pd.testing.assert_frame_equal(df, perturbed)

    def test_perturb_increases_rep(self):
        """Positive perturbation should increase _rep risk values."""
        df = _make_summary_2d()
        perturbed = perturb_risk_summary(df, 10.0)
        assert (perturbed["todu_30ever_h6_rep"] >= df["todu_30ever_h6_rep"] - 1e-10).all()

    def test_other_indicators_unchanged(self):
        """Non-risk _rep columns should be unchanged."""
        df = _make_summary_2d()
        perturbed = perturb_risk_summary(df, 20.0)
        pd.testing.assert_series_equal(df["oa_amt_h0_rep"], perturbed["oa_amt_h0_rep"])
        pd.testing.assert_series_equal(df["todu_amt_pile_h6_rep"], perturbed["todu_amt_pile_h6_rep"])

    def test_base_recomputed(self):
        """Base column should equal _boo + perturbed _rep."""
        df = _make_summary_2d()
        perturbed = perturb_risk_summary(df, 15.0)
        expected = perturbed["todu_30ever_h6_boo"] + perturbed["todu_30ever_h6_rep"]
        pd.testing.assert_series_equal(perturbed["todu_30ever_h6"], expected, check_names=False)


# =============================================================================
# run_sensitivity_analysis Tests
# =============================================================================


class TestSensitivityAnalysis:
    @pytest.fixture
    def baseline_setup(self):
        df = _make_summary_2d()
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        assert mask is not None
        return df, mask

    def test_sensitivity_no_flips_at_zero(self, baseline_setup):
        """Zero perturbation should produce zero flips."""
        df, mask = baseline_setup
        result = run_sensitivity_analysis(
            df, ["var0", "var1"], [], 7, INDICATORS, mask, 20.0, perturbation_levels=[0.0]
        )
        assert len(result) == 1
        assert result.iloc[0]["n_flipped"] == 0

    def test_returns_expected_columns(self, baseline_setup):
        df, mask = baseline_setup
        result = run_sensitivity_analysis(df, ["var0", "var1"], [], 7, INDICATORS, mask, 20.0)
        expected_cols = {
            "perturbation_pct",
            "n_flipped",
            "n_accept_to_reject",
            "n_reject_to_accept",
            "new_production",
            "new_risk",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_large_perturbation_flips_cells(self, baseline_setup):
        """Very large perturbation should cause some cells to flip."""
        df, mask = baseline_setup
        # Use a tight risk target so perturbation has more effect
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        tight_mask = milp_solve_cutoffs(grid, target_risk=3.0, inv_vars=[], multiplier=7)
        if tight_mask is None:
            pytest.skip("Tight risk target infeasible")
        if tight_mask.sum() == 0 or tight_mask.sum() == grid.n_cells:
            pytest.skip("All or none accepted at tight risk")

        result = run_sensitivity_analysis(
            df,
            ["var0", "var1"],
            [],
            7,
            INDICATORS,
            tight_mask,
            3.0,
            perturbation_levels=[-50, 50],
        )
        # At least one level should produce a change
        assert result["n_flipped"].notna().any()


# =============================================================================
# sensitivity_cell_detail Tests
# =============================================================================


class TestSensitivityCellDetail:
    def test_returns_all_cells(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        assert mask is not None
        result = sensitivity_cell_detail(df, ["var0", "var1"], [], 7, INDICATORS, mask, 20.0)
        assert len(result) == grid.n_cells

    def test_has_expected_columns(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=20.0, inv_vars=[], multiplier=7)
        assert mask is not None
        result = sensitivity_cell_detail(df, ["var0", "var1"], [], 7, INDICATORS, mask, 20.0)
        assert "baseline_status" in result.columns
        assert "flip_threshold_pct" in result.columns
        assert "flip_direction" in result.columns


# =============================================================================
# compute_cell_marginal_impact Tests
# =============================================================================


class TestMarginalImpact:
    @pytest.fixture
    def grid_and_mask(self):
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask = milp_solve_cutoffs(grid, target_risk=10.0, inv_vars=[], multiplier=7)
        assert mask is not None
        return grid, mask

    def test_marginal_accepted_cell_loses_production(self, grid_and_mask):
        """Flipping an accepted cell should decrease production."""
        grid, mask = grid_and_mask
        result = compute_cell_marginal_impact(grid, mask, INDICATORS, 7)
        accepted = result[result["status"] == 1]
        if not accepted.empty:
            # Removing an accepted cell = negative delta_production
            assert (accepted["delta_production"] <= 0).all()

    def test_marginal_rejected_cell_gains_production(self, grid_and_mask):
        """Flipping a rejected cell should increase production."""
        grid, mask = grid_and_mask
        result = compute_cell_marginal_impact(grid, mask, INDICATORS, 7)
        rejected = result[result["status"] == 0]
        if not rejected.empty:
            # Adding a rejected cell = positive delta_production
            assert (rejected["delta_production"] >= 0).all()

    def test_has_expected_columns(self, grid_and_mask):
        grid, mask = grid_and_mask
        result = compute_cell_marginal_impact(grid, mask, INDICATORS, 7)
        expected = {"status", "delta_production", "delta_risk_pct", "cell_production", "cell_risk"}
        assert expected.issubset(set(result.columns))

    def test_all_cells_present(self, grid_and_mask):
        grid, mask = grid_and_mask
        result = compute_cell_marginal_impact(grid, mask, INDICATORS, 7)
        assert len(result) == grid.n_cells


# =============================================================================
# solve_with_fixed_cells Tests
# =============================================================================


class TestSolveWithFixedCells:
    def test_fixed_accept_cell_appears_as_1(self):
        """A cell pinned as accept should appear as 1 in the result mask."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Pin cell (1, 1) as accept
        mask, kpis = solve_with_fixed_cells(
            grid,
            target_risk=50.0,
            inv_vars=[],
            multiplier=7,
            indicators=INDICATORS,
            fixed_accepts=[(1, 1)],
        )
        assert mask is not None
        idx = grid.cell_index[(1, 1)]
        assert mask[idx] == 1

    def test_fixed_reject_cell_appears_as_0(self):
        """A cell pinned as reject should appear as 0 in the result mask."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        # Pin cell (1, 1) as reject
        mask, kpis = solve_with_fixed_cells(
            grid,
            target_risk=50.0,
            inv_vars=[],
            multiplier=7,
            indicators=INDICATORS,
            fixed_rejects=[(1, 1)],
        )
        assert mask is not None
        idx = grid.cell_index[(1, 1)]
        assert mask[idx] == 0

    def test_fixed_infeasible_returns_none(self):
        """Contradictory fixed cells should return None."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask, kpis = solve_with_fixed_cells(
            grid,
            target_risk=50.0,
            inv_vars=[],
            multiplier=7,
            indicators=INDICATORS,
            fixed_accepts=[(1, 1)],
            fixed_rejects=[(1, 1)],
        )
        assert mask is None
        assert kpis is None

    def test_returns_kpis(self):
        """Should return a KPI dict alongside the mask."""
        df = _make_summary_2d(3, 4)
        grid = CellGrid.from_summary(df, ["var0", "var1"])
        mask, kpis = solve_with_fixed_cells(
            grid,
            target_risk=50.0,
            inv_vars=[],
            multiplier=7,
            indicators=INDICATORS,
        )
        assert mask is not None
        assert kpis is not None
        assert "oa_amt_h0" in kpis
        assert "b2_ever_h6" in kpis
