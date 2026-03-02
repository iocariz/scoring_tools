import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import pandas as pd
import pytest

from src.global_optimizer import AllocationResult, GlobalAllocator, SegmentConstraints


def _make_frontier(points: list[tuple[int, float, float]]) -> pd.DataFrame:
    """Helper: build a frontier DataFrame from (sol_fac, risk, production) tuples."""
    return pd.DataFrame(points, columns=["sol_fac", "b2_ever_h6", "oa_amt_h0"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def two_segment_allocator():
    """Two segments, 3 frontier points each.

    Segment A: low-risk, moderate production
        (0, 0.5%, 1000), (1, 1.0%, 2000), (2, 1.5%, 2500)
    Segment B: higher-risk, higher production
        (0, 0.8%, 1500), (1, 1.2%, 3000), (2, 2.0%, 4000)
    """
    alloc = GlobalAllocator()
    alloc.load_frontier(
        "A",
        _make_frontier(
            [
                (0, 0.5, 1000),
                (1, 1.0, 2000),
                (2, 1.5, 2500),
            ]
        ),
    )
    alloc.load_frontier(
        "B",
        _make_frontier(
            [
                (0, 0.8, 1500),
                (1, 1.2, 3000),
                (2, 2.0, 4000),
            ]
        ),
    )
    return alloc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExactSolver:
    def test_two_segments_exact_known_answer(self, two_segment_allocator):
        """With target=1.5%, the optimal is A=2 (1.5%, 2500) + B=1 (1.2%, 3000).
        Weighted risk = (1.5*2500 + 1.2*3000) / (2500+3000) = 7350/5500 ≈ 1.336%
        Production = 5500.

        Alternatively A=1 + B=2 gives risk = (1.0*2000 + 2.0*4000)/6000 = 10000/6000 ≈ 1.667% > 1.5%  → infeasible.
        A=2 + B=2 gives risk = (1.5*2500 + 2.0*4000)/6500 = 11750/6500 ≈ 1.808% > 1.5% → infeasible.
        So the best feasible is A=2 + B=1 = production 5500.
        """
        result = two_segment_allocator.optimize_exact(global_risk_target=1.5)
        assert result.allocations["A"] == 2
        assert result.allocations["B"] == 1
        assert result.global_production == pytest.approx(5500)
        assert result.global_risk <= 1.5 + 1e-9

    def test_respects_global_risk_target(self, two_segment_allocator):
        """Result weighted risk must not exceed the target."""
        result = two_segment_allocator.optimize_exact(global_risk_target=1.0)
        assert result.global_risk <= 1.0 + 1e-9

    def test_respects_per_segment_constraints(self, two_segment_allocator):
        """Force segment A risk between 0.9 and 1.1 (must pick sol_fac=1, risk=1.0)."""
        result = two_segment_allocator.optimize_exact(
            global_risk_target=2.0,
            risk_constraints={"A": (0.9, 1.1)},
        )
        assert result.segment_metrics["A"]["risk"] == pytest.approx(1.0)

    def test_single_segment(self):
        """Single segment: should pick the highest-production point within risk target."""
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "only",
            _make_frontier(
                [
                    (0, 0.5, 100),
                    (1, 1.0, 200),
                    (2, 2.0, 300),
                ]
            ),
        )
        result = alloc.optimize_exact(global_risk_target=1.5)
        assert result.allocations["only"] == 1
        assert result.global_production == pytest.approx(200)

    def test_single_point_frontiers(self):
        """Each segment has only one option — solver must pick it."""
        alloc = GlobalAllocator()
        alloc.load_frontier("X", _make_frontier([(10, 1.0, 500)]))
        alloc.load_frontier("Y", _make_frontier([(20, 0.5, 800)]))
        result = alloc.optimize_exact(global_risk_target=2.0)
        assert result.allocations["X"] == 10
        assert result.allocations["Y"] == 20
        assert result.global_production == pytest.approx(1300)

    def test_infeasible_target(self):
        """Target so low that no combination works → RuntimeError from solver,
        caught by optimize() which falls back to greedy."""
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "A",
            _make_frontier(
                [
                    (0, 2.0, 100),
                    (1, 3.0, 200),
                ]
            ),
        )
        # Direct call to optimize_exact should raise
        with pytest.raises(RuntimeError):
            alloc.optimize_exact(global_risk_target=0.01)

    def test_infeasible_falls_back_to_greedy(self):
        """optimize() with method='exact' falls back to greedy on infeasible."""
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "A",
            _make_frontier(
                [
                    (0, 2.0, 100),
                    (1, 3.0, 200),
                ]
            ),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = alloc.optimize(global_risk_target=0.01, method="exact")
            assert any("falling back to greedy" in str(x.message) for x in w)
        # Greedy will just return the index-0 point (lowest risk start)
        assert result.allocations["A"] == 0
        # Fell back to greedy, so method reflects that
        assert result.method == "greedy"

    def test_exact_beats_greedy_local_optimum(self):
        """Greedy gets trapped by a near-dominated intermediate point.

        Segment A has three points. Point 1 (risk=0.8, prod=110) barely
        improves on point 0 (risk=0.5, prod=100) but sits below the big
        payoff at point 2 (risk=0.82, prod=2000). Segment B offers a
        moderate step (risk=0.7, prod=800) that the greedy takes first
        due to higher efficiency, consuming the risk budget.

        Greedy path: B 0→1 (eff=0.73), A 0→1 (eff=0.26), stuck → prod=910
        Optimal:     A=2, B=0 → risk=0.716, prod=2500
        """
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "A",
            _make_frontier(
                [
                    (0, 0.5, 100),
                    (1, 0.8, 110),
                    (2, 0.82, 2000),
                ]
            ),
        )
        alloc.load_frontier(
            "B",
            _make_frontier(
                [
                    (0, 0.3, 500),
                    (1, 0.7, 800),
                ]
            ),
        )

        exact = alloc.optimize_exact(global_risk_target=0.75)
        greedy = alloc.optimize_greedy(global_risk_target=0.75)

        # MILP finds the global optimum
        assert exact.allocations["A"] == 2
        assert exact.allocations["B"] == 0
        assert exact.global_production == pytest.approx(2500)
        assert exact.global_risk <= 0.75 + 1e-9

        # Greedy gets stuck at a worse local optimum
        assert greedy.global_production < exact.global_production


class TestGreedySolver:
    def test_greedy_matches_exact_simple(self, two_segment_allocator):
        """On a simple case, greedy and exact should agree."""
        exact = two_segment_allocator.optimize_exact(global_risk_target=1.5)
        greedy = two_segment_allocator.optimize_greedy(global_risk_target=1.5)
        assert greedy.global_production == pytest.approx(exact.global_production)

    def test_greedy_max_iterations(self):
        """Greedy should stop after max_iterations and still return a result."""
        alloc = GlobalAllocator()
        # Many points so greedy needs many steps
        n = 200
        points = [(i, 0.5 + i * 0.01, 100 + i * 10) for i in range(n)]
        alloc.load_frontier("big", _make_frontier(points))
        result = alloc.optimize_greedy(global_risk_target=10.0, max_iterations=5)
        # Should have taken only 5 steps from index 0 → index 5
        assert result.allocations["big"] == 5


class TestEmptyAllocator:
    def test_empty_allocator_exact(self):
        alloc = GlobalAllocator()
        with pytest.raises(ValueError, match="No frontiers loaded"):
            alloc.optimize_exact(global_risk_target=1.0)

    def test_empty_allocator_greedy(self):
        alloc = GlobalAllocator()
        with pytest.raises(ValueError, match="No frontiers loaded"):
            alloc.optimize_greedy(global_risk_target=1.0)

    def test_empty_allocator_dispatch(self):
        alloc = GlobalAllocator()
        with pytest.raises(ValueError, match="No frontiers loaded"):
            alloc.optimize(global_risk_target=1.0)


class TestDominatedPruning:
    def test_dominated_points_pruned(self):
        """Points with higher risk but no production gain are removed."""
        alloc = GlobalAllocator()
        # Point (1, 1.0, 90) is dominated by point (0, 0.5, 100)
        alloc.load_frontier(
            "seg",
            _make_frontier(
                [
                    (0, 0.5, 100),
                    (1, 1.0, 90),
                    (2, 1.5, 200),
                ]
            ),
        )
        assert len(alloc.frontiers["seg"]) == 2
        assert list(alloc.frontiers["seg"]["sol_fac"]) == [0, 2]

    def test_equal_production_pruned(self):
        """Points with same production but higher risk are pruned."""
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "seg",
            _make_frontier(
                [
                    (0, 0.5, 100),
                    (1, 1.0, 100),
                    (2, 1.5, 200),
                ]
            ),
        )
        assert len(alloc.frontiers["seg"]) == 2
        assert list(alloc.frontiers["seg"]["sol_fac"]) == [0, 2]

    def test_strictly_increasing_not_pruned(self):
        """Frontiers with strictly increasing production keep all points."""
        alloc = GlobalAllocator()
        alloc.load_frontier(
            "seg",
            _make_frontier(
                [
                    (0, 0.5, 100),
                    (1, 1.0, 200),
                    (2, 1.5, 300),
                ]
            ),
        )
        assert len(alloc.frontiers["seg"]) == 3


class TestUnknownConstraintWarning:
    def test_warns_on_unknown_segment(self, two_segment_allocator):
        """Constraints referencing unknown segments produce a warning."""
        from loguru import logger

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)))
        try:
            two_segment_allocator.optimize_exact(
                global_risk_target=2.0,
                risk_constraints={"UNKNOWN_SEG": (0.5, 1.5)},
            )
        finally:
            logger.remove(handler_id)
        assert any("unknown segments ignored" in m for m in messages)

    def test_no_warning_for_known_segments(self, two_segment_allocator):
        """Constraints for known segments don't warn."""
        from loguru import logger

        messages = []
        handler_id = logger.add(lambda msg: messages.append(str(msg)))
        try:
            two_segment_allocator.optimize_exact(
                global_risk_target=2.0,
                risk_constraints={"A": (0.5, 1.5)},
            )
        finally:
            logger.remove(handler_id)
        assert not any("unknown segments ignored" in m for m in messages)


class TestAllocationResult:
    def test_to_dataframe(self, two_segment_allocator):
        result = two_segment_allocator.optimize(global_risk_target=1.5)
        df = result.to_dataframe()
        assert "segment" in df.columns
        assert "risk" in df.columns
        assert "production" in df.columns
        assert "sol_fac" in df.columns
        assert len(df) == 2
        assert set(df["segment"]) == {"A", "B"}

    def test_str_output(self, two_segment_allocator):
        result = two_segment_allocator.optimize(global_risk_target=1.5)
        text = str(result)
        assert "GLOBAL ALLOCATION RESULTS" in text
        assert "Global Risk:" in text
        assert "Global Production:" in text
        assert "Risk Target:" in text
        assert "Method:" in text

    def test_method_and_target_fields(self, two_segment_allocator):
        exact = two_segment_allocator.optimize(global_risk_target=1.5, method="exact")
        assert exact.method == "exact"
        assert exact.target == 1.5

        greedy = two_segment_allocator.optimize(global_risk_target=1.5, method="greedy")
        assert greedy.method == "greedy"
        assert greedy.target == 1.5

    def test_segment_details_populated(self, two_segment_allocator):
        result = two_segment_allocator.optimize(global_risk_target=1.5)
        assert set(result.segment_details.keys()) == {"A", "B"}
        for seg in ("A", "B"):
            d = result.segment_details[seg]
            assert "sol_fac" in d
            assert "b2_ever_h6" in d
            assert "oa_amt_h0" in d

    def test_str_shows_swap_details_when_present(self):
        """When frontier has swap columns, __str__ includes the swap table."""
        result = AllocationResult(
            global_risk=1.0,
            global_production=5000,
            allocations={"X": 1},
            segment_metrics={"X": {"risk": 1.0, "production": 5000}},
            segment_details={
                "X": {
                    "acct_booked_h0_rep": 50,
                    "acct_booked_h0_cut": 30,
                    "oa_amt_h0_rep": 1000,
                    "oa_amt_h0_cut": 500,
                }
            },
        )
        text = str(result)
        assert "Swap In" in text
        assert "Swap Out" in text

    def test_str_no_swap_section_without_details(self, two_segment_allocator):
        """When frontier has no swap columns, no swap section is shown."""
        result = AllocationResult(
            global_risk=1.0,
            global_production=5000,
            allocations={"X": 1},
            segment_metrics={"X": {"risk": 1.0, "production": 5000}},
        )
        text = str(result)
        assert "Swap In" not in text

    def test_to_dataframe_includes_swap_columns(self):
        result = AllocationResult(
            global_risk=1.0,
            global_production=5000,
            allocations={"X": 1},
            segment_metrics={"X": {"risk": 1.0, "production": 5000}},
            segment_details={
                "X": {
                    "acct_booked_h0_rep": 50,
                    "acct_booked_h0_cut": 30,
                    "oa_amt_h0_rep": 1000,
                    "oa_amt_h0_cut": 500,
                }
            },
        )
        df = result.to_dataframe()
        assert "acct_booked_h0_rep" in df.columns
        assert "acct_booked_h0_cut" in df.columns
        assert df.iloc[0]["acct_booked_h0_rep"] == 50

    def test_to_full_dataframe(self):
        result = AllocationResult(
            global_risk=1.0,
            global_production=5000,
            allocations={"X": 1},
            segment_metrics={"X": {"risk": 1.0, "production": 5000}},
            segment_details={
                "X": {
                    "sol_fac": 1,
                    "b2_ever_h6": 1.0,
                    "oa_amt_h0": 5000,
                    "extra_col": 42,
                }
            },
        )
        df = result.to_full_dataframe()
        assert df.columns[0] == "segment"
        assert "extra_col" in df.columns
        assert df.iloc[0]["extra_col"] == 42

    def test_unknown_method(self, two_segment_allocator):
        with pytest.raises(ValueError, match="Unknown method"):
            two_segment_allocator.optimize(global_risk_target=1.0, method="invalid")


# ---------------------------------------------------------------------------
# SegmentConstraints dataclass tests
# ---------------------------------------------------------------------------


class TestSegmentConstraints:
    def test_segment_constraints_defaults(self):
        """All fields default to None."""
        sc = SegmentConstraints()
        assert sc.min_risk is None
        assert sc.max_risk is None
        assert sc.min_production is None
        assert sc.locked_sol_fac is None

    def test_to_risk_tuple(self):
        """Conversion to legacy (min_risk, max_risk) format."""
        sc = SegmentConstraints(min_risk=0.5, max_risk=1.5)
        assert sc.to_risk_tuple() == (0.5, 1.5)

    def test_to_risk_tuple_partial_min(self):
        sc = SegmentConstraints(min_risk=0.5)
        assert sc.to_risk_tuple() == (0.5, float("inf"))

    def test_to_risk_tuple_partial_max(self):
        sc = SegmentConstraints(max_risk=1.5)
        assert sc.to_risk_tuple() == (0.0, 1.5)

    def test_to_risk_tuple_none(self):
        sc = SegmentConstraints()
        assert sc.to_risk_tuple() is None


# ---------------------------------------------------------------------------
# Production floor tests
# ---------------------------------------------------------------------------


class TestProductionFloor:
    def test_milp_respects_per_segment_production_floor(self):
        """Per-segment production floor forces a minimum production for that segment."""
        alloc = GlobalAllocator()
        # A: (0, 0.5%, 500), (1, 1.0%, 1500), (2, 1.5%, 3000)
        # B: (0, 0.5%, 500), (1, 1.0%, 1000), (2, 1.5%, 1500)
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 500), (1, 1.0, 1500), (2, 1.5, 3000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 500), (1, 1.0, 1000), (2, 1.5, 1500)]))

        # Without floor, solver might pick A=0 for lower risk
        # With floor A >= 1500, must pick at least sol_fac=1
        result = alloc.optimize_exact(
            global_risk_target=2.0,
            constraints={"A": SegmentConstraints(min_production=1500)},
        )
        assert result.segment_metrics["A"]["production"] >= 1500

    def test_milp_respects_global_production_floor(self):
        """Global production floor ensures total production meets minimum."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))

        result = alloc.optimize_exact(
            global_risk_target=2.0,
            global_production_floor=5000,
        )
        assert result.global_production >= 5000

    def test_infeasible_production_floor_raises(self):
        """Production floor impossible with risk target raises RuntimeError."""
        alloc = GlobalAllocator()
        # Max production is 200 + 200 = 400; floor of 1000 is impossible
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 100), (1, 1.0, 200)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 100), (1, 1.0, 200)]))

        with pytest.raises(RuntimeError):
            alloc.optimize_exact(
                global_risk_target=0.5,
                global_production_floor=1000,
            )


# ---------------------------------------------------------------------------
# Segment locking tests
# ---------------------------------------------------------------------------


class TestSegmentLocking:
    def test_milp_locked_segment_uses_fixed_point(self):
        """Lock segment A to sol_fac=1, verify the allocation respects it."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))

        result = alloc.optimize_exact(
            global_risk_target=2.0,
            constraints={"A": SegmentConstraints(locked_sol_fac=1)},
        )
        assert result.allocations["A"] == 1
        assert result.segment_metrics["A"]["risk"] == pytest.approx(1.0)
        assert result.segment_metrics["A"]["production"] == pytest.approx(2000)

    def test_greedy_locked_segment_skipped(self):
        """Greedy skips locked segment in hill-climbing."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 1.5, 3000)]))

        result = alloc.optimize_greedy(
            global_risk_target=2.0,
            constraints={"A": SegmentConstraints(locked_sol_fac=0)},
        )
        # A is locked to sol_fac=0, B should be free to optimize
        assert result.allocations["A"] == 0
        assert result.segment_metrics["A"]["production"] == pytest.approx(1000)
        # B should go higher since it has budget
        assert result.allocations["B"] >= 1

    def test_locked_segment_invalid_sol_fac_raises(self):
        """Non-existent sol_fac raises ValueError."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000)]))

        with pytest.raises(ValueError, match="Locked sol_fac=99 not found"):
            alloc.optimize_exact(
                global_risk_target=2.0,
                constraints={"A": SegmentConstraints(locked_sol_fac=99)},
            )

    def test_greedy_locked_invalid_sol_fac_raises(self):
        """Greedy also raises ValueError for non-existent locked sol_fac."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000)]))

        with pytest.raises(ValueError, match="Locked sol_fac=99 not found"):
            alloc.optimize_greedy(
                global_risk_target=2.0,
                constraints={"A": SegmentConstraints(locked_sol_fac=99)},
            )


# ---------------------------------------------------------------------------
# Binding constraints tests
# ---------------------------------------------------------------------------


class TestBindingConstraints:
    def test_binding_global_risk_detected(self):
        """Tight global risk target should appear in binding_constraints."""
        alloc = GlobalAllocator()
        # With target=1.0, the solver is forced to pick low-risk options
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 2.0, 5000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000), (2, 2.0, 5000)]))

        result = alloc.optimize_exact(global_risk_target=1.0)
        assert "global_risk" in result.binding_constraints

    def test_no_binding_when_slack(self):
        """Very loose target should not bind."""
        alloc = GlobalAllocator()
        alloc.load_frontier("A", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000)]))
        alloc.load_frontier("B", _make_frontier([(0, 0.5, 1000), (1, 1.0, 2000)]))

        # Target of 10% is very loose — both segments pick max production
        result = alloc.optimize_exact(global_risk_target=10.0)
        assert "global_risk" not in result.binding_constraints


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_legacy_risk_constraints_still_work(self, two_segment_allocator):
        """Old-style risk_constraints dict still works via optimize()."""
        result = two_segment_allocator.optimize(
            global_risk_target=2.0,
            risk_constraints={"A": (0.9, 1.1)},
        )
        assert result.segment_metrics["A"]["risk"] == pytest.approx(1.0)

    def test_legacy_risk_constraints_exact(self, two_segment_allocator):
        """Old-style risk_constraints dict works with optimize_exact directly."""
        result = two_segment_allocator.optimize_exact(
            global_risk_target=2.0,
            risk_constraints={"A": (0.9, 1.1)},
        )
        assert result.segment_metrics["A"]["risk"] == pytest.approx(1.0)

    def test_legacy_risk_constraints_greedy(self, two_segment_allocator):
        """Old-style risk_constraints dict works with optimize_greedy directly."""
        result = two_segment_allocator.optimize_greedy(
            global_risk_target=2.0,
            risk_constraints={"A": (0.9, 1.1)},
        )
        assert result.segment_metrics["A"]["risk"] >= 0.9

    def test_new_constraints_take_precedence(self, two_segment_allocator):
        """When both risk_constraints and constraints are given, constraints wins."""
        result = two_segment_allocator.optimize(
            global_risk_target=2.0,
            risk_constraints={"A": (1.4, 1.6)},  # Would force A to sol_fac=2
            constraints={"A": SegmentConstraints(min_risk=0.9, max_risk=1.1)},  # Forces A to sol_fac=1
        )
        assert result.segment_metrics["A"]["risk"] == pytest.approx(1.0)
