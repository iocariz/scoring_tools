"""Tests for the dashboard's "How We Got Here" tab (management story)."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import dashboard


@pytest.fixture
def story_output(tmp_path, monkeypatch):
    """A minimal segment output tree with everything the story tab reads."""
    seg = "seg_a"
    data_dir = tmp_path / seg / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"Metric": "Actual", "Risk (%)": 2.06, "Production (€)": 35_762_585.0, "Total Demand (€)": 374_733_272.0},
            {"Metric": "Swap-in", "Risk (%)": 2.60, "Production (€)": 4_125_086.0},
            {"Metric": "Swap-out", "Risk (%)": 5.27, "Production (€)": 11_759_903.0},
            {"Metric": "Optimum selected", "Risk (%)": 0.79, "Production (€)": 28_127_768.0},
        ]
    ).to_csv(data_dir / "risk_production_summary_table_base.csv", index=False)
    pd.DataFrame([{"Metric": "Optimum selected", "Risk (%)": 0.59}]).to_csv(
        data_dir / "risk_production_summary_table_mr_base.csv", index=False
    )
    pd.DataFrame({"cell": range(5)}).to_csv(data_dir / "data_summary_desagregado_base.csv", index=False)
    pd.DataFrame({"b2_ever_h6": [0.2, 0.79, 2.0, 5.0], "oa_amt_h0": [5e6, 28.1e6, 45e6, 60e6]}).to_csv(
        data_dir / "efficient_frontier_base.csv", index=False
    )
    pd.DataFrame([{"b2_ever_h6": 0.79, "oa_amt_h0": 28_127_768.0}]).to_csv(
        data_dir / "optimal_solution_base.csv", index=False
    )
    monkeypatch.setattr(dashboard, "OUTPUT_BASE", tmp_path)
    return seg


def test_load_story_data(story_output):
    story = dashboard._load_story_data("base", story_output)
    assert set(story["rows"]) == {"Actual", "Swap-in", "Swap-out", "Optimum selected"}
    assert story["n_cells"] == 5
    assert len(story["frontier"]) == 4
    assert story["chosen"] == pytest.approx((0.79, 28_127_768.0))
    assert story["mr_risk"] == pytest.approx(0.59)


def test_bridge_figure(story_output):
    story = dashboard._load_story_data("base", story_output)
    fig = dashboard._build_story_bridge_figure(story["rows"])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4  # four waterfall bars
    # the floating bars sit on the kept level (actual - swap_out)
    kept = 35_762_585.0 - 11_759_903.0
    assert fig.data[1].base[0] == pytest.approx(kept)
    assert fig.data[2].base[0] == pytest.approx(kept)


def test_bridge_figure_missing_rows_returns_none():
    assert dashboard._build_story_bridge_figure({}) is None


def test_frontier_figure(story_output):
    story = dashboard._load_story_data("base", story_output)
    fig = dashboard._build_story_frontier_figure(story["frontier"], story["chosen"], 0.8)
    names = [t.name for t in fig.data]
    assert "Efficient policies" in names and "Chosen policy" in names


def test_story_tab_content(story_output):
    content = dashboard.create_story_tab_content("base", story_output)
    text = str(content)
    for step in [
        "Start from every application",
        "Sort applications into score cells",
        "Price the risk of every cell",
        "Evaluate every sensible cutoff policy",
        "Choose the point on the frontier",
        "Prove it holds up",
    ]:
        assert step in text
    assert "4 efficient policies" in text  # run numbers flow into the narrative
    assert "Proposed production" in text


def test_story_tab_degrades_without_data(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "OUTPUT_BASE", tmp_path)
    content = dashboard.create_story_tab_content("base", None)
    assert "Story Not Available" in str(content)
