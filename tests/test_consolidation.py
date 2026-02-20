"""
Tests for consolidation module and scenario generation.

Tests cover:
- Scenario naming (pessimistic, base, optimistic)
- Scenario suffix detection from filenames
- Metrics extraction from summary tables
- Metrics aggregation across segments
- Risk calculation (sum(todu_30ever_h6) / sum(todu_amt_pile_h6) * 7)
- Percentage formatting
- Full consolidation workflow
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.consolidation import (
    ConsolidatedMetrics,
    aggregate_metrics,
    consolidate_segments,
    extract_metrics_from_table,
    find_scenario_suffix,
    map_scenario_names,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_summary_table():
    """Create a sample risk_production_summary_table."""
    return pd.DataFrame(
        {
            "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
            "Risk (%)": [1.5, 1.2, 2.0, 1.4],
            "Production (€)": [1000000, 200000, 150000, 1050000],
            "Production (%)": [1.0, 0.2, 0.15, 1.05],
            "todu_30ever_h6": [1000, 200, 300, 900],
            "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
        }
    )


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory with mock segment data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Scenario Suffix Detection Tests
# =============================================================================


class TestFindScenarioSuffix:
    """Tests for find_scenario_suffix function."""

    def test_named_pessimistic(self):
        """Test detection of pessimistic scenario."""
        result = find_scenario_suffix("risk_production_summary_table_pessimistic.csv")
        assert result == "_pessimistic"

    def test_named_base(self):
        """Test detection of base scenario."""
        result = find_scenario_suffix("risk_production_summary_table_base.csv")
        assert result == "_base"

    def test_named_optimistic(self):
        """Test detection of optimistic scenario."""
        result = find_scenario_suffix("risk_production_summary_table_optimistic.csv")
        assert result == "_optimistic"

    def test_legacy_numeric(self):
        """Test detection of legacy numeric scenarios."""
        assert find_scenario_suffix("risk_production_summary_table_1.0.csv") == "_1.0"
        assert find_scenario_suffix("risk_production_summary_table_0.9.csv") == "_0.9"
        assert find_scenario_suffix("risk_production_summary_table_1.1.csv") == "_1.1"

    def test_no_suffix(self):
        """Test file without scenario suffix."""
        result = find_scenario_suffix("risk_production_summary_table.csv")
        assert result == ""

    def test_mr_file_with_scenario(self):
        """Test MR file with scenario suffix."""
        result = find_scenario_suffix("risk_production_summary_table_mr_pessimistic.csv")
        assert result == "_pessimistic"

    def test_case_insensitive(self):
        """Test case insensitivity for named scenarios."""
        result = find_scenario_suffix("risk_production_summary_table_PESSIMISTIC.csv")
        assert result == "_pessimistic"


# =============================================================================
# Scenario Name Mapping Tests
# =============================================================================


class TestMapScenarioNames:
    """Tests for map_scenario_names function."""

    def test_named_scenarios(self):
        """Test mapping of named scenario suffixes."""
        suffixes = ["_pessimistic", "_base", "_optimistic"]
        mapping = map_scenario_names(suffixes)

        assert mapping["_pessimistic"] == "pessimistic"
        assert mapping["_base"] == "base"
        assert mapping["_optimistic"] == "optimistic"

    def test_legacy_numeric_three_scenarios(self):
        """Test mapping of three legacy numeric scenarios."""
        suffixes = ["_0.9", "_1.0", "_1.1"]
        mapping = map_scenario_names(suffixes)

        assert mapping["_0.9"] == "pessimistic"
        assert mapping["_1.0"] == "base"
        assert mapping["_1.1"] == "optimistic"

    def test_legacy_numeric_two_scenarios(self):
        """Test mapping of two legacy numeric scenarios."""
        suffixes = ["_0.9", "_1.1"]
        mapping = map_scenario_names(suffixes)

        assert mapping["_0.9"] == "pessimistic"
        assert mapping["_1.1"] == "optimistic"

    def test_empty_suffix_as_base(self):
        """Test that empty suffix maps to base."""
        suffixes = ["", "_pessimistic", "_optimistic"]
        mapping = map_scenario_names(suffixes)

        assert mapping[""] == "base"
        assert mapping["_pessimistic"] == "pessimistic"
        assert mapping["_optimistic"] == "optimistic"

    def test_single_scenario(self):
        """Test mapping of single scenario."""
        suffixes = ["_1.0"]
        mapping = map_scenario_names(suffixes)

        assert mapping["_1.0"] == "base"


# =============================================================================
# Metrics Extraction Tests
# =============================================================================


class TestExtractMetricsFromTable:
    """Tests for extract_metrics_from_table function."""

    def test_basic_extraction(self, sample_summary_table):
        """Test basic metrics extraction."""
        metrics = extract_metrics_from_table(sample_summary_table)

        assert "actual" in metrics
        assert "optimum" in metrics
        assert "swap_in" in metrics
        assert "swap_out" in metrics

    def test_production_values(self, sample_summary_table):
        """Test production value extraction."""
        metrics = extract_metrics_from_table(sample_summary_table)

        assert metrics["actual"]["production"] == 1000000
        assert metrics["optimum"]["production"] == 1050000
        assert metrics["swap_in"]["production"] == 200000
        assert metrics["swap_out"]["production"] == 150000

    def test_todu_values(self, sample_summary_table):
        """Test todu value extraction."""
        metrics = extract_metrics_from_table(sample_summary_table)

        assert metrics["actual"]["todu_30ever_h6"] == 1000
        assert metrics["actual"]["todu_amt_pile_h6"] == 50000
        assert metrics["optimum"]["todu_30ever_h6"] == 900
        assert metrics["optimum"]["todu_amt_pile_h6"] == 45000

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        metrics = extract_metrics_from_table(pd.DataFrame())

        assert metrics["actual"]["production"] == 0
        assert metrics["actual"]["todu_30ever_h6"] == 0

    def test_none_input(self):
        """Test handling of None input."""
        metrics = extract_metrics_from_table(None)

        assert metrics["actual"]["production"] == 0


# =============================================================================
# Metrics Aggregation Tests
# =============================================================================


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_single_segment(self):
        """Test aggregation of single segment."""
        metrics = [
            {
                "actual": {"production": 1000, "todu_30ever_h6": 100, "todu_amt_pile_h6": 5000},
                "optimum": {"production": 1100, "todu_30ever_h6": 90, "todu_amt_pile_h6": 5000},
                "swap_in": {"production": 200, "todu_30ever_h6": 10, "todu_amt_pile_h6": 500},
                "swap_out": {"production": 100, "todu_30ever_h6": 20, "todu_amt_pile_h6": 500},
            }
        ]

        result = aggregate_metrics(metrics)

        assert result["actual"]["production"] == 1000
        assert result["actual"]["todu_30ever_h6"] == 100

    def test_multiple_segments(self):
        """Test aggregation of multiple segments."""
        metrics = [
            {
                "actual": {"production": 1000, "todu_30ever_h6": 100, "todu_amt_pile_h6": 5000},
                "optimum": {"production": 1100, "todu_30ever_h6": 90, "todu_amt_pile_h6": 5000},
                "swap_in": {"production": 200, "todu_30ever_h6": 10, "todu_amt_pile_h6": 500},
                "swap_out": {"production": 100, "todu_30ever_h6": 20, "todu_amt_pile_h6": 500},
            },
            {
                "actual": {"production": 2000, "todu_30ever_h6": 200, "todu_amt_pile_h6": 10000},
                "optimum": {"production": 2200, "todu_30ever_h6": 180, "todu_amt_pile_h6": 10000},
                "swap_in": {"production": 400, "todu_30ever_h6": 20, "todu_amt_pile_h6": 1000},
                "swap_out": {"production": 200, "todu_30ever_h6": 40, "todu_amt_pile_h6": 1000},
            },
        ]

        result = aggregate_metrics(metrics)

        # Check sums
        assert result["actual"]["production"] == 3000
        assert result["actual"]["todu_30ever_h6"] == 300
        assert result["actual"]["todu_amt_pile_h6"] == 15000
        assert result["optimum"]["production"] == 3300

    def test_empty_list(self):
        """Test aggregation of empty list."""
        result = aggregate_metrics([])

        assert result["actual"]["production"] == 0
        assert result["actual"]["todu_30ever_h6"] == 0


# =============================================================================
# ConsolidatedMetrics Tests
# =============================================================================


class TestConsolidatedMetrics:
    """Tests for ConsolidatedMetrics dataclass."""

    def test_risk_calculation(self):
        """Test that risk is calculated correctly from todu values."""
        metrics = ConsolidatedMetrics(
            group_name="test",
            period="main",
            scenario="base",
            segments=["seg1"],
            actual_production=1000000,
            actual_todu_30ever_h6=100,
            actual_todu_amt_pile_h6=10000,
        )

        # Risk = todu_30ever_h6 / todu_amt_pile_h6 * 7 * 100 (percentage)
        # = 100 / 10000 * 7 * 100 = 7.0
        assert np.isclose(metrics.actual_risk, 7.0)

    def test_risk_percentage_in_dict(self):
        """Test that risk is converted to percentage in to_dict()."""
        metrics = ConsolidatedMetrics(
            group_name="test",
            period="main",
            scenario="base",
            segments=["seg1"],
            actual_production=1000000,
            actual_todu_30ever_h6=100,
            actual_todu_amt_pile_h6=10000,
        )

        result = metrics.to_dict()

        # actual_risk property returns percentage directly: 7.0%
        assert np.isclose(result["actual_risk_pct"], 7.0)

    def test_zero_denominator(self):
        """Test handling of zero denominator in risk calculation."""
        metrics = ConsolidatedMetrics(
            group_name="test",
            period="main",
            scenario="base",
            segments=["seg1"],
            actual_production=1000000,
            actual_todu_30ever_h6=100,
            actual_todu_amt_pile_h6=0,  # Zero denominator
        )

        assert metrics.actual_risk == 0.0

    def test_production_delta(self):
        """Test production delta calculation."""
        metrics = ConsolidatedMetrics(
            group_name="test",
            period="main",
            scenario="base",
            segments=["seg1"],
            actual_production=1000000,
            optimum_production=1100000,
        )

        assert metrics.production_delta == 100000
        assert np.isclose(metrics.production_delta_pct, 0.1)

    def test_production_delta_pct_in_dict(self):
        """Test production delta percentage is in correct format."""
        metrics = ConsolidatedMetrics(
            group_name="test",
            period="main",
            scenario="base",
            segments=["seg1"],
            actual_production=1000000,
            optimum_production=1100000,
        )

        result = metrics.to_dict()

        # 0.1 * 100 = 10%
        assert np.isclose(result["production_delta_pct"], 10.0)


# =============================================================================
# Full Consolidation Workflow Tests
# =============================================================================


class TestConsolidateSegments:
    """Tests for consolidate_segments function."""

    def test_all_scenarios_included(self, temp_output_dir):
        """Test that all three scenarios are included in output."""
        # Create mock segment with all scenarios
        seg_dir = temp_output_dir / "test-segment" / "data"
        seg_dir.mkdir(parents=True)

        for scenario in ["pessimistic", "base", "optimistic"]:
            df = pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                    "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                    "Production (€)": [1000000, 200000, 150000, 1050000],
                    "Production (%)": [1.0, 0.2, 0.15, 1.05],
                    "todu_30ever_h6": [1000, 200, 300, 900],
                    "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
                }
            )
            df.to_csv(seg_dir / f"risk_production_summary_table_{scenario}.csv", index=False)
            df.to_csv(seg_dir / f"risk_production_summary_table_mr_{scenario}.csv", index=False)

        segments = {"test-segment": {"name": "test-segment"}}
        result = consolidate_segments(temp_output_dir, segments, {})

        scenarios_found = result["scenario"].unique().tolist()
        assert "pessimistic" in scenarios_found
        assert "base" in scenarios_found
        assert "optimistic" in scenarios_found

    def test_both_periods_included(self, temp_output_dir):
        """Test that both main and MR periods are included."""
        seg_dir = temp_output_dir / "test-segment" / "data"
        seg_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                "Production (€)": [1000000, 200000, 150000, 1050000],
                "Production (%)": [1.0, 0.2, 0.15, 1.05],
                "todu_30ever_h6": [1000, 200, 300, 900],
                "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
            }
        )
        df.to_csv(seg_dir / "risk_production_summary_table_base.csv", index=False)
        df.to_csv(seg_dir / "risk_production_summary_table_mr_base.csv", index=False)

        segments = {"test-segment": {"name": "test-segment"}}
        result = consolidate_segments(temp_output_dir, segments, {})

        periods_found = result["period"].unique().tolist()
        assert "main" in periods_found
        assert "mr" in periods_found

    def test_supersegment_aggregation(self, temp_output_dir):
        """Test that segments in a supersegment are aggregated."""
        # Create two segments in same supersegment
        for seg_name in ["seg-a", "seg-b"]:
            seg_dir = temp_output_dir / seg_name / "data"
            seg_dir.mkdir(parents=True)

            df = pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                    "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                    "Production (€)": [1000000, 200000, 150000, 1050000],
                    "Production (%)": [1.0, 0.2, 0.15, 1.05],
                    "todu_30ever_h6": [1000, 200, 300, 900],
                    "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
                }
            )
            df.to_csv(seg_dir / "risk_production_summary_table_base.csv", index=False)

        segments = {
            "seg-a": {"name": "seg-a", "supersegment": "super1"},
            "seg-b": {"name": "seg-b", "supersegment": "super1"},
        }
        supersegments = {"super1": {}}

        result = consolidate_segments(temp_output_dir, segments, supersegments)

        # Should have supersegment total
        ss_rows = result[result["group"] == "supersegment_super1"]
        assert len(ss_rows) > 0

        # Supersegment production should be sum of both segments
        ss_main = ss_rows[ss_rows["period"] == "main"]
        assert ss_main["actual_production"].values[0] == 2000000

    def test_individual_segments_in_supersegment(self, temp_output_dir):
        """Test that individual segments are shown even when in supersegment."""
        for seg_name in ["seg-a", "seg-b"]:
            seg_dir = temp_output_dir / seg_name / "data"
            seg_dir.mkdir(parents=True)

            df = pd.DataFrame(
                {
                    "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                    "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                    "Production (€)": [1000000, 200000, 150000, 1050000],
                    "Production (%)": [1.0, 0.2, 0.15, 1.05],
                    "todu_30ever_h6": [1000, 200, 300, 900],
                    "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
                }
            )
            df.to_csv(seg_dir / "risk_production_summary_table_base.csv", index=False)

        segments = {
            "seg-a": {"name": "seg-a", "supersegment": "super1"},
            "seg-b": {"name": "seg-b", "supersegment": "super1"},
        }
        supersegments = {"super1": {}}

        result = consolidate_segments(temp_output_dir, segments, supersegments)

        # Should have individual segment rows with supersegment prefix
        groups = result["group"].unique().tolist()
        assert "super1/seg-a" in groups
        assert "super1/seg-b" in groups

    def test_total_row_included(self, temp_output_dir):
        """Test that TOTAL row is included."""
        seg_dir = temp_output_dir / "test-segment" / "data"
        seg_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                "Production (€)": [1000000, 200000, 150000, 1050000],
                "Production (%)": [1.0, 0.2, 0.15, 1.05],
                "todu_30ever_h6": [1000, 200, 300, 900],
                "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
            }
        )
        df.to_csv(seg_dir / "risk_production_summary_table_base.csv", index=False)

        segments = {"test-segment": {"name": "test-segment"}}
        result = consolidate_segments(temp_output_dir, segments, {})

        assert "TOTAL" in result["group"].values

    def test_risk_aggregation_formula(self, temp_output_dir):
        """Test that aggregated risk uses correct formula: sum(num)/sum(den)*7."""
        # Create two segments with different risk profiles
        seg1_dir = temp_output_dir / "seg1" / "data"
        seg1_dir.mkdir(parents=True)
        seg2_dir = temp_output_dir / "seg2" / "data"
        seg2_dir.mkdir(parents=True)

        # Segment 1: todu_30ever_h6=100, todu_amt_pile_h6=5000 -> risk = 0.14
        df1 = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Risk (%)": [0.14, 0.1, 0.2, 0.12],
                "Production (€)": [1000000, 200000, 150000, 1050000],
                "Production (%)": [1.0, 0.2, 0.15, 1.05],
                "todu_30ever_h6": [100, 20, 30, 90],
                "todu_amt_pile_h6": [5000, 1000, 1000, 4500],
            }
        )
        df1.to_csv(seg1_dir / "risk_production_summary_table_base.csv", index=False)

        # Segment 2: todu_30ever_h6=200, todu_amt_pile_h6=10000 -> risk = 0.14
        df2 = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Risk (%)": [0.14, 0.1, 0.2, 0.12],
                "Production (€)": [2000000, 400000, 300000, 2100000],
                "Production (%)": [1.0, 0.2, 0.15, 1.05],
                "todu_30ever_h6": [200, 40, 60, 180],
                "todu_amt_pile_h6": [10000, 2000, 2000, 9000],
            }
        )
        df2.to_csv(seg2_dir / "risk_production_summary_table_base.csv", index=False)

        segments = {
            "seg1": {"name": "seg1"},
            "seg2": {"name": "seg2"},
        }
        result = consolidate_segments(temp_output_dir, segments, {})

        # Get TOTAL row for main period
        total_row = result[(result["group"] == "TOTAL") & (result["period"] == "main")]

        # Aggregated: todu_30ever_h6 = 100+200=300, todu_amt_pile_h6 = 5000+10000=15000
        # Risk = 300/15000*7 = 0.14 -> 14% in output
        expected_risk_pct = (300 / 15000 * 7) * 100  # 14.0
        assert np.isclose(total_row["actual_risk_pct"].values[0], expected_risk_pct, rtol=0.01)


# =============================================================================
# Scenario Order Tests
# =============================================================================


class TestScenarioOrder:
    """Tests for scenario ordering (pessimistic -> base -> optimistic)."""

    def test_scenario_order_in_mapping(self):
        """Test that scenarios are mapped in correct order."""
        # The order should be: pessimistic (lowest), base (middle), optimistic (highest)
        suffixes = ["_1.1", "_0.9", "_1.0"]  # Out of order
        mapping = map_scenario_names(suffixes)

        assert mapping["_0.9"] == "pessimistic"
        assert mapping["_1.0"] == "base"
        assert mapping["_1.1"] == "optimistic"


class TestScenarioDeduplication:
    """Tests for scenario deduplication."""

    def test_no_duplicate_base_scenarios(self, temp_output_dir):
        """Test that '' and '_base' don't create duplicate entries."""
        # Create mock segment with both unsuffixed and _base files
        seg_dir = temp_output_dir / "test-segment" / "data"
        seg_dir.mkdir(parents=True)

        df = pd.DataFrame(
            {
                "Metric": ["Actual", "Swap-in", "Swap-out", "Optimum selected"],
                "Risk (%)": [1.5, 1.2, 2.0, 1.4],
                "Production (€)": [1000000, 200000, 150000, 1050000],
                "Production (%)": [1.0, 0.2, 0.15, 1.05],
                "todu_30ever_h6": [1000, 200, 300, 900],
                "todu_amt_pile_h6": [50000, 10000, 10000, 45000],
            }
        )

        # Create both unsuffixed (backward compat) and _base files
        df.to_csv(seg_dir / "risk_production_summary_table.csv", index=False)
        df.to_csv(seg_dir / "risk_production_summary_table_base.csv", index=False)
        df.to_csv(seg_dir / "risk_production_summary_table_pessimistic.csv", index=False)
        df.to_csv(seg_dir / "risk_production_summary_table_optimistic.csv", index=False)

        segments = {"test-segment": {"name": "test-segment"}}
        result = consolidate_segments(temp_output_dir, segments, {})

        # Count occurrences of 'base' scenario in main period
        main_base_count = ((result["scenario"] == "base") & (result["period"] == "main")).sum()

        # Should only have one 'base' entry per group/period combination, not duplicates
        # With one segment + TOTAL, we expect 2 rows for main period with base scenario
        assert main_base_count == 2  # segment + TOTAL

    def test_deduplication_prefers_named_suffix(self):
        """Test that deduplication prefers '_base' over '' suffix."""
        suffixes = ["", "_base", "_pessimistic", "_optimistic"]
        mapping = map_scenario_names(suffixes)

        # Both '' and '_base' map to 'base'
        assert mapping[""] == "base"
        assert mapping["_base"] == "base"

        # Deduplication logic
        seen_names = {}
        for suffix in suffixes:
            name = mapping.get(suffix, "base")
            if name not in seen_names or suffix:
                seen_names[name] = suffix

        # Should prefer '_base' over ''
        assert seen_names["base"] == "_base"
        assert len(seen_names) == 3  # base, pessimistic, optimistic


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
