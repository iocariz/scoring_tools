"""Tests for the interactive allocator's lock plumbing (#50). Pure-helper level —
the Dash callback wiring is exercised by importing the module (callbacks register)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import interactive_allocator as ia


class TestBuildSegmentConstraints:
    def test_lock_inert_without_sol_fac(self):
        # "Yes" before any run → no locked_sol_fac yet → nothing to pin to.
        rows = [{"segment": "a", "locked": "Yes", "locked_sol_fac": None}]
        assert ia._build_segment_constraints(rows) == {}

    def test_lock_applied_with_sol_fac(self):
        rows = [{"segment": "a", "locked": "Yes", "locked_sol_fac": 3}]
        c = ia._build_segment_constraints(rows)
        assert c["a"].locked_sol_fac == 3

    def test_lock_ignored_when_locked_is_no(self):
        rows = [{"segment": "a", "locked": "No", "locked_sol_fac": 3}]
        assert ia._build_segment_constraints(rows) == {}

    def test_risk_and_production_bounds_preserved(self):
        rows = [{"segment": "a", "min_risk": 1.0, "max_risk": 2.0, "min_production": 500.0, "locked": "No"}]
        c = ia._build_segment_constraints(rows)
        assert c["a"].min_risk == 1.0 and c["a"].max_risk == 2.0 and c["a"].min_production == 500.0

    def test_empty_or_none_table(self):
        assert ia._build_segment_constraints(None) == {}
        assert ia._build_segment_constraints([]) == {}


class TestApplyAllocationToTable:
    def test_writes_sol_fac_preserving_edits_without_mutating_input(self):
        rows = [
            {"segment": "a", "min_risk": 1.0, "locked": "Yes", "locked_sol_fac": None},
            {"segment": "b", "locked": "No", "locked_sol_fac": None},
        ]
        out = ia._apply_allocation_to_table(rows, {"a": 5, "b": 2})
        assert out[0]["locked_sol_fac"] == 5 and out[1]["locked_sol_fac"] == 2
        assert out[0]["min_risk"] == 1.0 and out[0]["locked"] == "Yes"  # user edits preserved
        assert rows[0]["locked_sol_fac"] is None  # original untouched (copied)

    def test_lock_takes_effect_after_a_run(self):
        """The whole #50 loop: 'Locked = Yes' is inert on the first run, then pinned
        to the segment's chosen sol_fac once the run writes it back."""
        rows = [{"segment": "a", "locked": "Yes", "locked_sol_fac": None}]
        assert ia._build_segment_constraints(rows) == {}  # first run: nothing to pin
        rows_after = ia._apply_allocation_to_table(rows, {"a": 7})  # writeback from run 1
        assert ia._build_segment_constraints(rows_after)["a"].locked_sol_fac == 7  # run 2: pinned
