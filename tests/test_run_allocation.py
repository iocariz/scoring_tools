"""Tests for run_allocation frontier discovery (layout precedence / no double-count)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from run_allocation import _discover_segment_frontiers


def _write_frontier(base, rel, scenario="base"):
    d = base / rel / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"efficient_frontier_{scenario}.csv").write_text("sol_fac,b2_ever_h6,oa_amt_h0\n0,1.0,100\n")


class TestDiscoverSegmentFrontiers:
    def test_multi_segment_layout(self, tmp_path):
        _write_frontier(tmp_path, "seg_a")
        _write_frontier(tmp_path, "seg_b")
        found = _discover_segment_frontiers(tmp_path, "base")
        names = [n for n, _ in found]
        assert names == ["seg_a", "seg_b"]  # sorted, both segments

    def test_single_segment_layout(self, tmp_path):
        # output_base itself is the segment: <base>/data/efficient_frontier_base.csv
        d = tmp_path / "data"
        d.mkdir(parents=True)
        (d / "efficient_frontier_base.csv").write_text("sol_fac,b2_ever_h6,oa_amt_h0\n0,1.0,100\n")
        found = _discover_segment_frontiers(tmp_path, "base")
        assert [n for n, _ in found] == [tmp_path.name]

    def test_both_layouts_present_prefers_multi_no_double_count(self, tmp_path):
        # A batch root that ALSO has a stray top-level data/ frontier: must NOT count the root
        # as an extra segment on top of the real segments.
        _write_frontier(tmp_path, "seg_a")
        _write_frontier(tmp_path, "seg_b")
        stray = tmp_path / "data"
        stray.mkdir(parents=True)
        (stray / "efficient_frontier_base.csv").write_text("sol_fac,b2_ever_h6,oa_amt_h0\n0,9.9,999\n")
        found = _discover_segment_frontiers(tmp_path, "base")
        names = [n for n, _ in found]
        assert names == ["seg_a", "seg_b"]  # root NOT included
        assert tmp_path.name not in names

    def test_no_frontiers(self, tmp_path):
        assert _discover_segment_frontiers(tmp_path, "base") == []

    def test_scenario_specific(self, tmp_path):
        _write_frontier(tmp_path, "seg_a", scenario="pessimistic")
        assert _discover_segment_frontiers(tmp_path, "base") == []
        assert [n for n, _ in _discover_segment_frontiers(tmp_path, "pessimistic")] == ["seg_a"]
