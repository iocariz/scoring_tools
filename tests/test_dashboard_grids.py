"""Tests for the cutoff-explorer grid rendering (three-state cells + staircase frontier)
and the summary-table number formatting."""

import numpy as np
import pandas as pd
import pytest

import dashboard

nan = float("nan")


# ---------------------------------------------------------------------------
# Frontier shapes
# ---------------------------------------------------------------------------


def test_frontier_shapes_trace_continuous_staircase():
    """Grey (NaN) cells inside the accept region must not fragment the frontier:
    the monotone imputation folds them in, so no shape edge touches them."""
    acc = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, nan, 1],  # grey inside the accept region
            [0, 1, 1, 1],
        ]
    )
    shapes = dashboard._frontier_shapes(acc)
    assert shapes, "expected frontier edges"
    # no edge may separate (1,2) [the imputed grey cell] from its accept neighbours:
    # its left edge at x=1.5 spanning y in [0.5, 1.5] would only exist if it were
    # treated as non-accept — assert the frontier at x=1.5 exists (boundary to the
    # reject cell on its left) but no box fully surrounds the grey cell.
    edges = {(s["x0"], s["x1"], s["y0"], s["y1"]) for s in shapes}
    assert (1.5, 2.5, 0.5, 0.5) not in edges  # no top edge isolating the grey cell
    assert (1.5, 2.5, 1.5, 1.5) not in edges  # no bottom edge isolating it


def test_frontier_shapes_all_reject_draws_nothing():
    assert dashboard._frontier_shapes(np.zeros((3, 3))) == []


def test_frontier_shapes_all_accept_is_grid_perimeter():
    shapes = dashboard._frontier_shapes(np.ones((2, 3)))
    # perimeter of a 2x3 grid: 2*3 horizontal + 2*2 vertical edges
    assert len(shapes) == 10
    for s in shapes:
        assert s["x0"] in (-0.5, 2.5) or s["y0"] in (-0.5, 1.5) or s["x0"] != s["x1"] or s["y0"] != s["y1"]


# ---------------------------------------------------------------------------
# Three-state slice grid
# ---------------------------------------------------------------------------


def _synthetic_3var_summary():
    """2x2x2 grid with one combo missing (unobserved)."""
    rows = []
    for a in (1, 2):
        for b in (1, 2):
            for c in (1, 2):
                if (a, b, c) == (2, 2, 1):
                    continue  # unobserved cell
                rows.append(
                    {
                        "v1": a,
                        "v2": b,
                        "v3": c,
                        "oa_amt_h0_boo": 1e6,
                        "oa_amt_h0_rep": 0.0,
                        "todu_30ever_h6_boo": 100.0,
                        "todu_amt_pile_h6_boo": 1e5,
                    }
                )
    return pd.DataFrame(rows)


def test_slice_grid_unobserved_cells_render_grey():
    sd = _synthetic_3var_summary()
    mask = [1] * 8  # accept everything
    fig = dashboard._build_nd_slice_grid_figure(sd, ["v1", "v2", "v3"], {"v3": 1}, "v1", "v2", mask, mask, {}, 7.0)
    z = np.array(fig.data[0].z, dtype=float)
    # cell (v2=2 row, v1=2 col) is unobserved with v3=1 -> grey 0.5, others accept 1.0
    assert z[1][1] == pytest.approx(0.5)
    assert z[0][0] == pytest.approx(1.0)
    assert "(no records)" in str(fig.data[0].customdata)
    assert len(fig.layout.shapes) > 0  # frontier drawn
    assert fig.data[0].colorscale is not None


# ---------------------------------------------------------------------------
# Summary-table formatting
# ---------------------------------------------------------------------------


def test_load_table_formats_percentage_points_and_counts(tmp_path):
    csv = tmp_path / "risk_production_summary_table_base.csv"
    pd.DataFrame(
        [
            {
                "Metric": "Actual",
                "Risk (%)": 2.06,
                "Production (€)": 35762585.0,
                "Production (%)": 100.0,
                "todu_30ever_h6": 678126.46,
                "risk_ci_lower": 0.409655,
            },
            {
                "Metric": "Optimum selected",
                "Risk (%)": 0.788784,
                "Production (€)": 28127768.467334513,
                "Production (%)": 78.65138514828979,
                "todu_30ever_h6": 202910.36702688594,
                "risk_ci_lower": 1.36873,
            },
        ]
    ).to_csv(csv, index=False)
    s = str(dashboard.load_table(csv))
    assert "78.65%" in s  # percentage points, not ratio x100
    assert "7865" not in s  # the old `:.2%` bug
    assert "100.00%" in s
    assert "€28,127,768" in s
    assert "202,910" in s  # counts grouped, no float tail
    assert "0.788784" not in s  # risk rounded


def test_load_table_drops_unnamed_empty_columns(tmp_path):
    csv = tmp_path / "t.csv"
    csv.write_text("Metric,Risk (%),Unnamed: 2,empty\nActual,2.06,,\n")
    s = str(dashboard.load_table(csv))
    assert "Unnamed" not in s
    assert "empty" not in s


def test_kpi_row_production_share_is_percentage_points(tmp_path):
    main = tmp_path / "main.csv"
    mr = tmp_path / "mr.csv"
    pd.DataFrame(
        [{"Metric": "Optimum selected", "Risk (%)": 0.79, "Production (%)": 78.65, "Production (€)": 28e6}]
    ).to_csv(main, index=False)
    pd.DataFrame([{"Metric": "Optimum selected", "Risk (%)": 0.59, "Production (%)": 80.0}]).to_csv(mr, index=False)
    s = str(dashboard.create_kpi_row(main, mr))
    assert "78.7%" in s
    assert "7865" not in s
    assert "-1.3pp" in s  # delta in points, not ratio (78.65-80.0 floats to -1.3499...)
