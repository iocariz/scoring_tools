"""Tests for the continuous 3D risk-surface module (src/risk_surface.py). Synthetic data only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import plotly.graph_objects as go
import pytest

from src.risk_surface import (
    CATEGORY_CODE,
    CATEGORY_ORDER,
    aggregate_to_cells,
    build_risk_surface_figure,
    classify_cells,
)


def _summary():
    # 4 cells on a 2-score grid (a, b). Combined risk + booked exposure (→ "booked before").
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [1.0, 1.0, 1.0, 1.0],
            "todu_30ever_h6": [1.0, 2.0, 3.0, 9.0],  # combined numerator
            "todu_amt_pile_h6": [100.0, 100.0, 100.0, 100.0],  # combined denominator
            "todu_amt_pile_h6_boo": [100.0, 100.0, 0.0, 0.0],  # a=1,2 had booked; a=3,4 repesca-only
        }
    )


# ------------------------------- classify_cells -------------------------------


def test_classify_carries_components_and_accept_weight():
    long = classify_cells(_summary(), {(1.0, 1.0), (3.0, 1.0)}, ["a", "b"])
    assert set(long.columns) == {"a", "b", "t30", "pile", "boo_mass", "acc_wt", "rej_wt"}
    # a=1 accepted -> weight on acc side; a=2 rejected -> weight on rej side
    a1 = long[long["a"] == 1.0].iloc[0]
    assert a1["acc_wt"] == 100.0 and a1["rej_wt"] == 0.0
    a2 = long[long["a"] == 2.0].iloc[0]
    assert a2["acc_wt"] == 0.0 and a2["rej_wt"] == 100.0


@pytest.mark.parametrize("missing", ["todu_30ever_h6", "todu_amt_pile_h6"])
def test_classify_raises_on_missing_risk_column(missing):
    """#49: a summary missing a risk column must fail loudly, not silently zero
    it into a fabricated flat-0 / empty surface."""
    summary = _summary().drop(columns=[missing])
    with pytest.raises(ValueError, match="missing required risk column"):
        classify_cells(summary, {(1.0, 1.0)}, ["a", "b"])


def test_classify_tolerates_missing_booked_before_column():
    """A missing booked-before column is optional: defaults to 'no prior bookings'
    (all cells swap_in/rejected), not an error."""
    summary = _summary().drop(columns=["todu_amt_pile_h6_boo"])
    cells = aggregate_to_cells(classify_cells(summary, {(1.0, 1.0)}, ["a", "b"]), ["a", "b"])
    assert set(cells["category"]) <= {"swap_in", "rejected"}  # no keep/swap_out without booked-before


# ----------------------------- aggregate_to_cells -----------------------------


def test_one_category_per_cell_no_overlap():
    """Single segment: every cell maps to exactly one category (no overlap possible)."""
    cells = aggregate_to_cells(classify_cells(_summary(), {(1.0, 1.0), (3.0, 1.0)}, ["a", "b"]), ["a", "b"])
    assert len(cells) == 4  # one row per cell
    assert cells.duplicated(subset=["a", "b"]).sum() == 0
    cat = {row.a: row.category for row in cells.itertuples()}
    assert cat == {1.0: "keep", 2.0: "swap_out", 3.0: "swap_in", 4.0: "rejected"}


def test_category_logic_old_x_new():
    cells = aggregate_to_cells(classify_cells(_summary(), {(1.0, 1.0), (3.0, 1.0)}, ["a", "b"]), ["a", "b"])
    # booked-before (a=1,2) -> keep/swap_out ; rejected-before (a=3,4) -> swap_in/rejected
    assert cells[cells["a"] == 1.0].iloc[0]["category"] == "keep"  # booked + accept
    assert cells[cells["a"] == 2.0].iloc[0]["category"] == "swap_out"  # booked + reject
    assert cells[cells["a"] == 3.0].iloc[0]["category"] == "swap_in"  # repesca + accept
    assert cells[cells["a"] == 4.0].iloc[0]["category"] == "rejected"  # repesca + reject


def test_b2_value():
    cells = aggregate_to_cells(classify_cells(_summary(), {(1.0, 1.0)}, ["a", "b"]), ["a", "b"], multiplier=7.0)
    assert cells[cells["a"] == 1.0].iloc[0]["b2_ever_h6"] == pytest.approx(7.0 * 1.0 / 100.0 * 100)  # 7%
    assert cells[cells["a"] == 4.0].iloc[0]["b2_ever_h6"] == pytest.approx(7.0 * 9.0 / 100.0 * 100)  # 63%


def test_supersegment_majority_resolves_one_category():
    """Two members disagree on a cell's accept; exposure-weighted majority picks one category."""
    m1 = classify_cells(
        pd.DataFrame(
            {
                "a": [1.0],
                "b": [1.0],
                "todu_30ever_h6": [1.0],
                "todu_amt_pile_h6": [300.0],
                "todu_amt_pile_h6_boo": [300.0],
            }
        ),
        {(1.0, 1.0)},  # member 1 accepts
        ["a", "b"],
    )
    m2 = classify_cells(
        pd.DataFrame(
            {
                "a": [1.0],
                "b": [1.0],
                "todu_30ever_h6": [1.0],
                "todu_amt_pile_h6": [100.0],
                "todu_amt_pile_h6_boo": [100.0],
            }
        ),
        set(),  # member 2 rejects
        ["a", "b"],
    )
    cells = aggregate_to_cells(pd.concat([m1, m2], ignore_index=True), ["a", "b"])
    assert len(cells) == 1  # one category per cell even when members disagree
    # accept exposure 300 > reject exposure 100, and booked-before → keep
    assert cells.iloc[0]["category"] == "keep"


def test_drops_empty_cells():
    summary = _summary()
    summary.loc[0, "todu_amt_pile_h6"] = 0.0  # empty denominator → dropped
    cells = aggregate_to_cells(classify_cells(summary, {(1.0, 1.0)}, ["a", "b"]), ["a", "b"])
    assert (cells["a"] == 1.0).sum() == 0
    assert len(cells) == 3


# --------------------------- build_risk_surface_figure ---------------------------


def test_build_single_surface_per_facet():
    cells = aggregate_to_cells(classify_cells(_summary(), {(1.0, 1.0), (3.0, 1.0)}, ["a", "b"]), ["a", "b"])
    fig = build_risk_surface_figure(cells, ["a", "b"], None, "title")
    assert isinstance(fig, go.Figure)
    surfaces = [t for t in fig.data if t.type == "surface"]
    assert len(surfaces) == 1  # ONE continuous surface (no facet)
    assert surfaces[0].surfacecolor is not None  # coloured by category code
    assert surfaces[0].colorbar.tickvals == (0, 1, 2, 3)


def test_build_facets_one_surface_each():
    summary = _summary()
    summary["income_bin"] = [1.0, 1.0, 2.0, 2.0]
    cells = aggregate_to_cells(
        classify_cells(summary, {(1.0, 1.0, 1.0), (3.0, 1.0, 2.0)}, ["a", "b", "income_bin"]),
        ["a", "b", "income_bin"],
    )
    fig = build_risk_surface_figure(cells, ["a", "b"], "income_bin", "title")
    surfaces = [t for t in fig.data if t.type == "surface"]
    assert len(surfaces) == 2  # one continuous surface per income bin, side by side
    assert "scene" in fig.layout and "scene2" in fig.layout


def test_category_code_order():
    assert CATEGORY_ORDER == ["keep", "swap_in", "swap_out", "rejected"]
    assert CATEGORY_CODE == {"keep": 0, "swap_in": 1, "swap_out": 2, "rejected": 3}


def test_summary_path_selects_period():
    from pathlib import Path

    from plot_risk_surface import _summary_path

    data = Path("/x/data")
    assert _summary_path(data, "_base", "main").name == "data_summary_desagregado_base.csv"
    assert _summary_path(data, "_base", "mr").name == "data_summary_desagregado_mr_base.csv"
