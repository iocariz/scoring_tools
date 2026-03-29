"""Tests for portfolio-owner summaries (policy table + allocation narrative)."""

from src.global_optimizer import AllocationResult, SegmentConstraints
from src.portfolio_owner import (
    allocation_constraint_narrative,
    policy_cutoff_table_from_allocation,
    what_if_dataframe_row,
)


def test_policy_cutoff_table_2_var_bins():
    segment_details = {
        "seg_a": {
            "sol_fac": 3,
            "b2_ever_h6": 1.25,
            "oa_amt_h0": 1_000_000.0,
            "0": 2.0,
            "1": 4.0,
            "2": 5.0,
        },
    }
    df = policy_cutoff_table_from_allocation(segment_details, scenario="base", target_risk_pct=2.5)
    assert len(df) == 3
    assert df["policy_type"].eq("2_var_cutoff").all()
    assert df["segment"].eq("seg_a").all()
    assert set(df["score_bin_axis_0"]) == {0.0, 1.0, 2.0}


def test_policy_cutoff_table_n_var_mask():
    segment_details = {
        "seg_b": {
            "sol_fac": 0,
            "b2_ever_h6": 2.0,
            "oa_amt_h0": 500_000.0,
            "acceptance_mask": "1,0,1,0,1",
        },
    }
    df = policy_cutoff_table_from_allocation(segment_details)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["policy_type"] == "N_var_mask"
    assert row["accepted_cells"] == 3


def test_allocation_narrative_binding():
    result = AllocationResult(
        global_risk=2.45,
        global_production=10_000_000.0,
        allocations={"A": 1},
        segment_metrics={"A": {"risk": 2.45, "production": 10_000_000.0}},
        method="exact",
        target=2.5,
        binding_constraints=["global_risk"],
    )
    constraints = {"A": SegmentConstraints(min_risk=0.5, max_risk=5.0)}
    text = allocation_constraint_narrative(result, constraints, global_production_floor=None)
    assert "Global portfolio risk" in text or "global_risk" in text
    assert "min risk" in text or "≥" in text


def test_what_if_row():
    result = AllocationResult(
        global_risk=1.1,
        global_production=100.0,
        allocations={"A": 0},
        segment_metrics={"A": {"risk": 1.1, "production": 100.0}},
        method="greedy",
        target=1.0,
        binding_constraints=["global_risk"],
    )
    row = what_if_dataframe_row(result, 1.0)
    assert row["target_risk_pct"] == 1.0
    assert row["achieved_global_risk_pct"] == 1.1
    assert "global_risk" in row["binding_constraints"]


def test_policy_table_empty():
    assert policy_cutoff_table_from_allocation({}).empty


def test_policy_table_summary_only():
    """No bin columns → one summary row per segment."""
    segment_details = {"x": {"sol_fac": 0, "b2_ever_h6": 1.0, "oa_amt_h0": 100.0}}
    df = policy_cutoff_table_from_allocation(segment_details)
    assert len(df) == 1
    assert df.iloc[0]["policy_type"] == "summary_only"
