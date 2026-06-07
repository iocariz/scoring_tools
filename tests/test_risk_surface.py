"""Tests for the 3D risk-surface module (src/risk_surface.py). Synthetic data only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.risk_surface import (
    CATEGORY_ORDER,
    aggregate_classified,
    build_risk_surface_figure,
    classify_cells,
)


def _summary():
    # 3 cells on a 2-score grid (a, b); booked + repesca components.
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 1.0, 1.0],
            "todu_30ever_h6_boo": [1.0, 2.0, 0.0],
            "todu_amt_pile_h6_boo": [100.0, 100.0, 0.0],
            "acct_booked_h0_boo": [10.0, 10.0, 0.0],
            "todu_30ever_h6_rep": [3.0, 0.0, 9.0],
            "todu_amt_pile_h6_rep": [100.0, 0.0, 100.0],
            "acct_booked_h0_rep": [5.0, 0.0, 5.0],
        }
    )


# ------------------------------- classify_cells -------------------------------


def test_classify_assigns_four_categories_by_mask():
    accepted = {(1.0, 1.0)}  # only cell a=1 accepted
    long = classify_cells(_summary(), accepted, ["a", "b"])
    # booked component: a=1 -> keep, a=2/3 -> swap_out
    boo = long.iloc[:3]
    assert list(boo["category"]) == ["keep", "swap_out", "swap_out"]
    # repesca component: a=1 -> swap_in, a=2/3 -> rejected
    rep = long.iloc[3:]
    assert list(rep["category"]) == ["swap_in", "rejected", "rejected"]


def test_classify_carries_population_numerators():
    long = classify_cells(_summary(), {(1.0, 1.0)}, ["a", "b"])
    keep = long[long["category"] == "keep"].iloc[0]
    assert keep["t30"] == 1.0 and keep["pile"] == 100.0
    swap_in = long[long["category"] == "swap_in"].iloc[0]
    assert swap_in["t30"] == 3.0 and swap_in["pile"] == 100.0


# ----------------------------- aggregate_classified -----------------------------


def test_aggregate_computes_b2_and_drops_empty():
    long = classify_cells(_summary(), {(1.0, 1.0)}, ["a", "b"])
    agg = aggregate_classified(long, ["a", "b"], multiplier=7.0)
    # cells with pile<=0 are dropped (booked a=3 had pile 0; repesca a=2 had pile 0)
    assert (agg["pile"] > 0).all()
    keep = agg[(agg["a"] == 1.0) & (agg["category"] == "keep")].iloc[0]
    assert keep["b2_ever_h6"] == pytest.approx(7.0 * 1.0 / 100.0 * 100)  # 7%
    rej = agg[(agg["a"] == 3.0) & (agg["category"] == "rejected")].iloc[0]
    assert rej["b2_ever_h6"] == pytest.approx(7.0 * 9.0 / 100.0 * 100)  # 63%


def test_aggregate_sums_across_rows_supersegment_style():
    # Two "members" with the same cell + category aggregate (exposure-weighted).
    long = pd.DataFrame(
        {
            "a": [1.0, 1.0],
            "b": [1.0, 1.0],
            "category": ["keep", "keep"],
            "t30": [1.0, 3.0],
            "pile": [100.0, 100.0],
            "exposure": [10.0, 10.0],
        }
    )
    agg = aggregate_classified(long, ["a", "b"], multiplier=7.0)
    assert len(agg) == 1
    assert agg.iloc[0]["b2_ever_h6"] == pytest.approx(7.0 * 4.0 / 200.0 * 100)  # 14%


# --------------------------- build_risk_surface_figure ---------------------------


def test_build_figure_no_facet_returns_surfaces():
    agg = aggregate_classified(classify_cells(_summary(), {(1.0, 1.0)}, ["a", "b"]), ["a", "b"])
    fig = build_risk_surface_figure(agg, ["a", "b"], None, "title")
    assert isinstance(fig, go.Figure)
    surfaces = [t for t in fig.data if t.type == "surface"]
    assert len(surfaces) >= 1
    # legend names are drawn from the audit categories
    assert all(s.name for s in surfaces)


def test_build_figure_facets_per_income_bin():
    summary = _summary()
    summary["income_bin"] = [1.0, 1.0, 2.0]  # third variable
    long = classify_cells(summary, {(1.0, 1.0, 1.0)}, ["a", "b", "income_bin"])
    agg = aggregate_classified(long, ["a", "b", "income_bin"], multiplier=7.0)
    fig = build_risk_surface_figure(agg, ["a", "b"], "income_bin", "title")
    # grid: rows = present categories, cols = income bins (2) → multiple 3D scenes
    assert "scene" in fig.layout and "scene2" in fig.layout
    assert fig.layout.scene2 is not None


def test_build_figure_grid_has_no_overlap():
    """Each 3D scene holds at most one category surface → no overlapping surfaces."""
    from collections import Counter

    summary = _summary()
    summary["income_bin"] = [1.0, 1.0, 2.0]
    agg = aggregate_classified(
        classify_cells(summary, {(1.0, 1.0, 1.0)}, ["a", "b", "income_bin"]), ["a", "b", "income_bin"], 7.0
    )
    fig = build_risk_surface_figure(agg, ["a", "b"], "income_bin", "title")
    surfaces = [t for t in fig.data if t.type == "surface"]
    per_scene = Counter(t.scene for t in surfaces)
    assert per_scene  # at least one surface placed
    assert max(per_scene.values()) == 1  # no scene has two surfaces → nothing overlaps


def test_category_order_constant():
    assert CATEGORY_ORDER == ["keep", "swap_in", "swap_out", "rejected"]
