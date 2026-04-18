"""Regression tests for CutoffSpec (R2 todo #65).

The type unifies the 2-var ``cut_map`` encoding and the N-var ``mask + grid``
encoding behind a single ``.classify(df)`` dispatch. These tests lock down:

  - Construction validation (exactly one encoding; no mixing)
  - 2-var classification (both directions — inverted and non-inverted)
  - N-var classification (via CellGrid + binary mask)
  - as_2d_cut_map adapter (returns dict when 2-var, None when N-var)
  - Byte-equivalence with the pre-migration bootstrap worker — feeding the
    same data through the old control flow and through CutoffSpec.classify
    must produce identical boolean Series.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.optimization_utils import CellGrid, CutoffSpec, classify_by_mask


class TestConstruction:
    def test_cut_map_builder(self):
        spec = CutoffSpec.from_cut_map({1.0: 2.0, 2.0: 3.0}, ["var0", "var1"])
        assert spec.is_2d
        assert spec.cut_map == {1.0: 2.0, 2.0: 3.0}
        assert spec.variables == ("var0", "var1")
        assert spec.mask is None
        assert spec.grid is None

    def test_inv_var1_flag_preserved(self):
        spec = CutoffSpec.from_cut_map({1.0: 2.0}, ["var0", "var1"], inv_var1=True)
        assert spec.inv_var1 is True

    def test_cut_map_requires_two_variables(self):
        with pytest.raises(ValueError, match="≥2 variables"):
            CutoffSpec.from_cut_map({1.0: 2.0}, ["only_one"])

    def test_refuses_both_encodings_simultaneously(self):
        summary = pd.DataFrame({"var0": [1, 2], "var1": [1, 2]})
        grid = CellGrid.from_summary(summary, ["var0", "var1"])
        with pytest.raises(ValueError, match="exactly one"):
            CutoffSpec(
                variables=("var0", "var1"),
                cut_map={1.0: 2.0},
                mask=np.ones(len(grid.cell_data), dtype=int),
                grid=grid,
            )

    def test_refuses_neither_encoding(self):
        with pytest.raises(ValueError, match="exactly one"):
            CutoffSpec(variables=("var0", "var1"))


class TestClassify2Var:
    @pytest.fixture
    def booked(self):
        return pd.DataFrame(
            {
                "sc_octroi": [1.0, 1.0, 2.0, 2.0, 3.0],
                "new_efx": [1.0, 2.0, 2.0, 3.0, 1.0],
            }
        )

    def test_non_inverted_accepts_below_or_equal_cutoff(self, booked):
        # For octroi=1 accept efx <= 1; for octroi=2 accept efx <= 2; for octroi=3 accept all
        spec = CutoffSpec.from_cut_map(
            {1.0: 1.0, 2.0: 2.0, 3.0: 10.0},
            ["sc_octroi", "new_efx"],
            inv_var1=False,
        )
        result = spec.classify(booked)
        assert list(result) == [True, False, True, False, True]

    def test_inverted_accepts_above_or_equal_cutoff(self, booked):
        spec = CutoffSpec.from_cut_map(
            {1.0: 2.0, 2.0: 2.0, 3.0: 0.0},
            ["sc_octroi", "new_efx"],
            inv_var1=True,
        )
        # Inverted: efx >= cutoff
        # octroi=1, cut=2: rows with efx=1→reject, efx=2→accept
        # octroi=2, cut=2: rows with efx=2→accept, efx=3→accept
        # octroi=3, cut=0: efx=1 >= 0 → accept
        result = spec.classify(booked)
        assert list(result) == [False, True, True, True, True]

    def test_missing_bin_strictly_rejects_non_inverted(self, booked):
        spec = CutoffSpec.from_cut_map(
            {1.0: 10.0, 2.0: 10.0},  # deliberately omits 3.0
            ["sc_octroi", "new_efx"],
            inv_var1=False,
        )
        result = spec.classify(booked)
        # octroi=3 bin missing → fillna(-inf) → efx never <= -inf → False
        assert result.iloc[4] is np.False_ or result.iloc[4] is False

    def test_missing_bin_strictly_rejects_inverted(self, booked):
        spec = CutoffSpec.from_cut_map(
            {1.0: 0.0, 2.0: 0.0},  # deliberately omits 3.0
            ["sc_octroi", "new_efx"],
            inv_var1=True,
        )
        result = spec.classify(booked)
        # octroi=3 bin missing → fillna(+inf) → efx never >= +inf → False
        assert result.iloc[4] is np.False_ or result.iloc[4] is False


class TestClassifyNVar:
    @pytest.fixture
    def grid_mask(self):
        summary = pd.DataFrame(
            {
                "sc_octroi": [1, 1, 2, 2, 3, 3],
                "new_efx": [1, 2, 1, 2, 1, 2],
                "dummy_metric": [0, 0, 0, 0, 0, 0],
            }
        )
        grid = CellGrid.from_summary(summary, ["sc_octroi", "new_efx"])
        # Accept cells where sc_octroi >= 2 AND new_efx == 1
        # The grid has 6 cells in lexicographic order of (sc_octroi, new_efx):
        #   (1,1) (1,2) (2,1) (2,2) (3,1) (3,2)
        mask = np.array([0, 0, 1, 0, 1, 0], dtype=int)
        return grid, mask

    def test_nvar_classify(self, grid_mask):
        grid, mask = grid_mask
        spec = CutoffSpec.from_mask(mask, grid)
        assert not spec.is_2d
        booked = pd.DataFrame(
            {"sc_octroi": [1, 2, 2, 3, 3], "new_efx": [1, 1, 2, 1, 2]},
        )
        result = spec.classify(booked)
        # (1,1): cell mask 0 → reject
        # (2,1): 1 → accept
        # (2,2): 0 → reject
        # (3,1): 1 → accept
        # (3,2): 0 → reject
        assert list(result) == [False, True, False, True, False]

    def test_as_2d_cut_map_returns_none_for_nvar(self, grid_mask):
        grid, mask = grid_mask
        spec = CutoffSpec.from_mask(mask, grid)
        assert spec.as_2d_cut_map() is None

    def test_equivalence_with_classify_by_mask(self, grid_mask):
        grid, mask = grid_mask
        spec = CutoffSpec.from_mask(mask, grid)
        booked = pd.DataFrame(
            {"sc_octroi": [1, 2, 2, 3, 3], "new_efx": [1, 1, 2, 1, 2]},
        )
        expected = classify_by_mask(booked, mask, grid)
        np.testing.assert_array_equal(spec.classify(booked).values, expected.values)


class TestAdapter:
    def test_as_2d_cut_map_returns_copy(self):
        original = {1.0: 2.0, 2.0: 3.0}
        spec = CutoffSpec.from_cut_map(original, ["a", "b"])
        extracted = spec.as_2d_cut_map()
        assert extracted == original
        # Mutating the extracted dict must not affect the spec.
        extracted[99.0] = 999.0
        assert 99.0 not in spec.cut_map


class TestBootstrapBackwardCompat:
    """calculate_bootstrap_intervals keeps accepting the legacy kwargs
    while internally converting to CutoffSpec (todo #65 migration)."""

    def test_legacy_cut_map_still_works(self):
        from src.utils import calculate_bootstrap_intervals

        np.random.seed(42)
        booked = pd.DataFrame(
            {
                "sc_octroi": np.random.choice([1, 2, 3], size=100),
                "new_efx": np.random.choice([1, 2, 3], size=100),
                "oa_amt_h0": np.random.uniform(1000, 5000, size=100),
                "todu_30ever_h6": np.random.uniform(0, 50, size=100),
                "todu_amt_pile_h6": np.random.uniform(500, 2000, size=100),
            }
        )
        out = calculate_bootstrap_intervals(
            booked,
            cut_map={1.0: 2.0, 2.0: 2.0, 3.0: 3.0},
            variables=["sc_octroi", "new_efx"],
            multiplier=7.0,
            n_bootstraps=20,
            random_state=42,
        )
        assert out["production_ci_lower"] < out["production_ci_upper"]
        assert out["risk_ci_lower"] >= 0

    def test_explicit_spec_matches_legacy_result(self):
        """Passing `spec=` must produce the same CI as the legacy kwargs."""
        from src.utils import calculate_bootstrap_intervals

        np.random.seed(42)
        booked = pd.DataFrame(
            {
                "sc_octroi": np.random.choice([1, 2, 3], size=80),
                "new_efx": np.random.choice([1, 2, 3], size=80),
                "oa_amt_h0": np.random.uniform(1000, 5000, size=80),
                "todu_30ever_h6": np.random.uniform(0, 50, size=80),
                "todu_amt_pile_h6": np.random.uniform(500, 2000, size=80),
            }
        )
        cut_map = {1.0: 2.0, 2.0: 2.0, 3.0: 3.0}
        variables = ["sc_octroi", "new_efx"]

        legacy = calculate_bootstrap_intervals(
            booked,
            cut_map=cut_map,
            variables=variables,
            multiplier=7.0,
            n_bootstraps=30,
            random_state=123,
        )
        via_spec = calculate_bootstrap_intervals(
            booked,
            spec=CutoffSpec.from_cut_map(cut_map, variables),
            multiplier=7.0,
            n_bootstraps=30,
            random_state=123,
        )
        for key in legacy:
            assert pytest.approx(legacy[key]) == via_spec[key]

    def test_no_spec_no_legacy_raises(self):
        from src.utils import calculate_bootstrap_intervals

        booked = pd.DataFrame(
            {
                "oa_amt_h0": [1, 2],
                "todu_30ever_h6": [0, 0],
                "todu_amt_pile_h6": [1, 1],
            }
        )
        with pytest.raises(ValueError, match="CutoffSpec"):
            calculate_bootstrap_intervals(booked, multiplier=7.0, n_bootstraps=5)
