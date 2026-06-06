"""Tests for the config-sensitivity harness (M3 tooling)."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from config_sensitivity import KNOBS, RunResult, select_knobs, summarize

# Tier-1 knobs whose grids were widened (M3 generality check) — must carry ≥4 values.
WIDENED_TIER1 = {
    "multiplier",
    "reject_uplift_factor",
    "reject_max_risk_multiplier",
    "ri_calibration_gamma",
}


class TestKnobRegistry:
    def test_names_unique(self):
        names = [k.name for k in KNOBS]
        assert len(names) == len(set(names)), "duplicate knob names in KNOBS"

    def test_all_values_non_empty(self):
        for k in KNOBS:
            assert isinstance(k.values, list) and k.values, f"{k.name} has empty values"

    def test_keys_are_strings(self):
        for k in KNOBS:
            assert isinstance(k.key, str) and k.key, f"{k.name} has a bad config key"

    def test_tier1_grids_widened(self):
        by_name = {k.name: k for k in KNOBS}
        for name in WIDENED_TIER1:
            assert name in by_name, f"expected Tier-1 knob {name} in registry"
            assert len(by_name[name].values) >= 4, f"{name} grid not widened (has {by_name[name].values})"

    def test_categorical_tier1_enumerate_all_values(self):
        by_name = {k.name: k for k in KNOBS}
        assert set(by_name["stress_mode"].values) == {"global", "per_bin", "disabled"}
        assert set(by_name["reject_parceling_method"].values) == {"linear", "power", "sigmoid"}


class TestSelectKnobs:
    def test_all_returns_full_registry(self):
        assert select_knobs("all") == KNOBS

    def test_subset_selection(self):
        chosen = select_knobs("multiplier,stress_mode")
        assert {k.name for k in chosen} == {"multiplier", "stress_mode"}

    def test_unknown_raises(self):
        with pytest.raises(SystemExit):
            select_knobs("not_a_real_knob")


class TestSummarize:
    def test_ranks_by_headline_impact(self):
        baseline = RunResult("baseline", "(baseline)", None, True, production=100.0, risk=1.0, n_accepted=10)
        # knob "big" moves production +50%; knob "small" moves it +5%.
        big = RunResult("big=1", "big", 1, True, production=150.0, risk=1.0, n_accepted=12, cutoff_diff=4)
        small = RunResult("small=1", "small", 1, True, production=105.0, risk=1.0, n_accepted=10, cutoff_diff=0)
        detail, summary = summarize([baseline, big, small])

        assert not summary.empty
        # Ranked by impact_score descending → "big" first.
        assert list(summary["knob"]) == ["big", "small"]
        big_row = summary[summary["knob"] == "big"].iloc[0]
        assert big_row["production_spread_pct"] == pytest.approx(50.0)
        assert big_row["max_cutoff_cells_changed"] == 4
        assert "big=1" in set(detail["label"])

    def test_empty_when_baseline_missing(self):
        # No successful baseline → no ranking produced.
        only = RunResult("multiplier=8", "multiplier", 8, True, production=100.0, risk=1.0, n_accepted=5)
        _detail, summary = summarize([only])
        assert summary.empty
