"""Coverage tests for portfolio_owner functions not in test_portfolio_owner.py.

Targets ``write_portfolio_owner_pack``, ``policy_cutoff_table_to_markdown``,
``_explain_binding_label`` branches, and edge cases of `_is_bin_column_key`.
"""

import pandas as pd

from src import portfolio_owner as po
from src.global_optimizer import AllocationResult, SegmentConstraints


class TestIsBinColumnKey:
    def test_int_is_bin(self):
        assert po._is_bin_column_key(3) is True

    def test_bool_is_not_bin(self):
        assert po._is_bin_column_key(True) is False

    def test_known_kpi_is_not_bin(self):
        assert po._is_bin_column_key("sol_fac") is False
        assert po._is_bin_column_key("b2_ever_h6") is False

    def test_acct_prefix_is_not_bin(self):
        assert po._is_bin_column_key("acct_booked") is False

    def test_oa_amt_prefix_is_not_bin(self):
        assert po._is_bin_column_key("oa_amt_h0_rep") is False

    def test_numeric_string_is_bin(self):
        assert po._is_bin_column_key("3") is True
        assert po._is_bin_column_key("3.5") is True

    def test_alpha_string_is_not_bin(self):
        assert po._is_bin_column_key("foo") is False


class TestPolicyCutoffTableToMarkdown:
    def test_empty_returns_message(self):
        out = po.policy_cutoff_table_to_markdown(pd.DataFrame())
        assert "No policy" in out

    def test_non_empty_returns_table_string(self):
        df = pd.DataFrame({"segment": ["a", "b"], "production_eur": [100, 200]})
        out = po.policy_cutoff_table_to_markdown(df)
        assert "a" in out and "b" in out


class TestExplainBindingLabel:
    def test_global_risk_with_target(self):
        text = po._explain_binding_label("global_risk", None, None, 2.5)
        assert "2.5000" in text or "2.5" in text

    def test_global_production_floor_with_value(self):
        text = po._explain_binding_label("global_production_floor", None, 50000.0, None)
        assert "50,000" in text

    def test_global_production_floor_no_value(self):
        text = po._explain_binding_label("global_production_floor", None, None, None)
        assert "global production floor" in text

    def test_min_risk_with_constraint(self):
        constraints = {"seg_a": SegmentConstraints(min_risk=0.5)}
        text = po._explain_binding_label("seg_a_min_risk", constraints, None, None)
        assert "seg_a" in text
        assert "0.5" in text

    def test_max_risk_with_constraint(self):
        constraints = {"seg_a": SegmentConstraints(max_risk=5.0)}
        text = po._explain_binding_label("seg_a_max_risk", constraints, None, None)
        assert "seg_a" in text
        assert "5.0" in text

    def test_min_production_with_constraint(self):
        constraints = {"seg_a": SegmentConstraints(min_production=10000.0)}
        text = po._explain_binding_label("seg_a_min_production", constraints, None, None)
        assert "seg_a" in text
        assert "10,000" in text

    def test_unknown_label_returns_label_in_backticks(self):
        text = po._explain_binding_label("custom_label", None, None, None)
        assert "custom_label" in text


class TestAllocationConstraintNarrative:
    def test_no_target_no_floor_minimal_output(self):
        result = AllocationResult(
            global_risk=1.0,
            global_production=100.0,
            allocations={"A": 0},
            segment_metrics={"A": {"risk": 1.0, "production": 100.0}},
            method="exact",
            target=None,
            binding_constraints=[],
        )
        text = po.allocation_constraint_narrative(result, None, global_production_floor=None)
        assert "Allocation constraint narrative" in text
        assert "exact" in text

    def test_global_floor_met(self):
        result = AllocationResult(
            global_risk=2.0,
            global_production=200.0,
            allocations={"A": 1},
            segment_metrics={"A": {"risk": 2.0, "production": 200.0}},
            method="greedy",
            target=2.5,
            binding_constraints=[],
        )
        text = po.allocation_constraint_narrative(result, None, global_production_floor=100.0)
        assert "met" in text

    def test_global_floor_unmet(self):
        result = AllocationResult(
            global_risk=2.0,
            global_production=50.0,
            allocations={"A": 1},
            segment_metrics={"A": {"risk": 2.0, "production": 50.0}},
            method="greedy",
            target=2.5,
            binding_constraints=[],
        )
        text = po.allocation_constraint_narrative(result, None, global_production_floor=100.0)
        assert "below configured floor" in text

    def test_constraints_locked_sol_fac_appears(self):
        result = AllocationResult(
            global_risk=1.0,
            global_production=100.0,
            allocations={"X": 5},
            segment_metrics={"X": {"risk": 1.0, "production": 100.0}},
            method="exact",
            target=None,
            binding_constraints=[],
        )
        constraints = {"X": SegmentConstraints(locked_sol_fac=5)}
        text = po.allocation_constraint_narrative(result, constraints, global_production_floor=None)
        assert "locked" in text


class TestWritePortfolioOwnerPack:
    def test_writes_csv_and_md(self, tmp_path):
        result = AllocationResult(
            global_risk=2.0,
            global_production=1_000_000.0,
            allocations={"A": 0},
            segment_metrics={"A": {"risk": 2.0, "production": 1_000_000.0}},
            segment_details={
                "A": {
                    "sol_fac": 0,
                    "b2_ever_h6": 2.0,
                    "oa_amt_h0": 1_000_000.0,
                    "0": 3.0,
                }
            },
            method="exact",
            target=2.5,
            binding_constraints=[],
        )
        output = tmp_path / "alloc.csv"
        policy_path, narrative_path = po.write_portfolio_owner_pack(
            result,
            None,
            output_csv=output,
            scenario="base",
        )
        assert policy_path.exists()
        assert narrative_path.exists()
        # Round-trip: CSV is non-empty
        df = pd.read_csv(policy_path)
        assert len(df) >= 1
