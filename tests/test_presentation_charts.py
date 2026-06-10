"""Tests for the results-presentation charts + deck builder. Synthetic data only."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import presentation_charts as pc


def _consolidated():
    rows = [
        # group, period, scenario, n_segments, actual/optimum prod, deltas, risks, swaps, rejection rates
        ("segment_seg_a", "main", "base", 1, 100.0, 90.0, -10.0, 2.0, 1.5, 30.0, 20.0, 60.0, 65.0),
        ("segment_seg_b", "main", "base", 1, 50.0, 55.0, 10.0, 1.0, 1.2, 10.0, 5.0, 40.0, 38.0),
        ("TOTAL", "main", "base", 2, 150.0, 145.0, -3.33, 1.67, 1.4, 40.0, 25.0, 52.0, 54.0),
    ]
    cols = [
        "group",
        "period",
        "scenario",
        "n_segments",
        "actual_production",
        "optimum_production",
        "production_delta_pct",
        "actual_risk_pct",
        "optimum_risk_pct",
        "swap_in_production",
        "swap_out_production",
        "actual_rejection_rate_pct",
        "optimum_rejection_rate_pct",
    ]
    return pd.DataFrame(rows, columns=cols)


def _backtest():
    return pd.DataFrame(
        {
            "segment": ["seg_a", "seg_b"],
            "predicted_risk_pct": [1.6, 1.3],
            "in_sample_realized_risk_pct": [1.5, 1.2],
            "oot_realized_risk_pct": [1.7, 1.1],
            "oot_risk_ci": ["[1.2, 2.2]", "[0.8, 1.5]"],
            "drift_flag": ["OK", "INCONCLUSIVE"],
        }
    )


# ------------------------------- data helpers -------------------------------


def test_seg_label():
    assert pc._seg_label("segment_no_premium_cd") == "no_premium_cd"
    assert pc._seg_label("TOTAL") == "TOTAL"


def test_segments_frame_excludes_total_and_sorts():
    df = pc._segments_frame(_consolidated())
    assert "TOTAL" not in df["group"].tolist()
    assert df.iloc[0]["segment"] == "seg_a"  # higher optimum production first


def test_segments_frame_excludes_supersegment_rows():
    """Audit #24: supersegment aggregate rows duplicate their members' euros —
    keeping them double-counts the by-segment charts and breaks the sum-to-TOTAL."""
    cons = _consolidated()
    ss_row = cons.iloc[[0]].copy()
    ss_row["group"] = "supersegment_pl_known"
    ss_row["actual_production"] = 150.0  # = seg_a + seg_b (aggregate of the members)
    ss_row["optimum_production"] = 145.0
    cons = pd.concat([ss_row, cons], ignore_index=True)  # aggregate row first, worst case

    df = pc._segments_frame(cons)
    assert not df["group"].str.startswith("supersegment_").any()
    assert sorted(df["segment"]) == ["seg_a", "seg_b"]
    # by-segment bars must reconcile with the TOTAL row again
    total = pc._total_row(cons)
    assert df["actual_production"].sum() == total["actual_production"]


def test_charts_use_requested_scenario():
    """Audit #44: the deck chart builders accept a scenario and must plot THAT
    scenario's rows, not silently fall back to base."""
    cons = _consolidated()
    pess = cons.copy()
    pess["scenario"] = "pessimistic"
    pess.loc[pess["group"] == "segment_seg_a", "optimum_production"] = 999e6
    cons = pd.concat([cons, pess], ignore_index=True)

    fig = pc.chart_production_by_segment(cons, scenario="pessimistic")
    heights = [round(p.get_height(), 6) for p in fig.axes[0].patches]
    assert 999.0 in heights  # pessimistic value plotted
    plt.close(fig)

    fig = pc.chart_production_by_segment(cons)  # default stays base
    heights = [round(p.get_height(), 6) for p in fig.axes[0].patches]
    assert 999.0 not in heights
    plt.close(fig)


def test_total_row():
    tr = pc._total_row(_consolidated())
    assert tr is not None and tr["group"] == "TOTAL"


# ------------------------------- charts -------------------------------


def test_summary_charts_return_figures():
    cons = _consolidated()
    for fn in (
        pc.chart_production_by_segment,
        pc.chart_risk_by_segment,
        pc.chart_swap_waterfall,
        pc.chart_acceptance_by_segment,
    ):
        fig = fn(cons)
        assert isinstance(fig, plt.Figure) and len(fig.axes) >= 1
        plt.close(fig)


def test_backtest_chart():
    fig = pc.chart_backtest_validation(_backtest())
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_category_heatmap_facets():
    cells = pd.DataFrame(
        {
            "a": [1.0, 2.0, 1.0, 2.0],
            "b": [1.0, 1.0, 1.0, 1.0],
            "income_bin": [1.0, 1.0, 2.0, 2.0],
            "b2_ever_h6": [5.0, 30.0, 8.0, 25.0],
            "category": ["keep", "swap_out", "swap_in", "rejected"],
        }
    )
    fig = pc.chart_category_heatmap(cells, ["a", "b"], "income_bin", "title")
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2  # one panel per income bin
    plt.close(fig)


def test_save_chart_writes_png(tmp_path):
    fig = pc.chart_production_by_segment(_consolidated())
    p = pc.save_chart(fig, tmp_path / "x.png")
    assert p.exists() and p.stat().st_size > 0


def test_export_surface_png_graceful_on_error(tmp_path):
    # A non-figure has no write_image → the helper catches and returns None (no kaleido needed).
    assert pc.export_surface_png(object(), tmp_path / "x.png") is None


# ------------------------------- deck smoke -------------------------------


def _build_output_tree(tmp_path):
    _consolidated().to_csv(tmp_path / "consolidated_risk_production.csv", index=False)
    data = tmp_path / "seg_a" / "data"
    data.mkdir(parents=True)
    pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 1.0]}).to_csv(data / "accepted_cells_base.csv", index=False)
    pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 1.0, 1.0],
            "todu_30ever_h6": [1.0, 2.0, 9.0],
            "todu_amt_pile_h6": [100.0, 100.0, 100.0],
            "todu_amt_pile_h6_boo": [100.0, 0.0, 0.0],
        }
    ).to_csv(data / "data_summary_desagregado_base.csv", index=False)
    return tmp_path


def test_build_deck_smoke(tmp_path, monkeypatch):
    # Skip the (heavy) 3D kaleido render in the test; the deck still assembles with the 2D heatmap.
    monkeypatch.setattr(pc, "export_surface_png", lambda *a, **k: None)
    import generate_results_presentation as grp

    tree = _build_output_tree(tmp_path)
    out = grp.build_results_presentation(
        data_dir=tree, out_dir=tree / "deck", scenario="base", segments_config=tree / "no_such.toml"
    )
    assert out.exists() and out.suffix == ".pptx"
    from pptx import Presentation

    # title + exec + methodology + 4 charts + >=1 surface + next steps
    assert len(Presentation(str(out)).slides) >= 8


def test_build_deck_unknown_scenario_raises(tmp_path):
    """Audit #44: a scenario absent from the consolidated CSV must fail loudly,
    not produce a deck of base numbers under a non-base label."""
    import pytest

    import generate_results_presentation as grp

    tree = _build_output_tree(tmp_path)
    with pytest.raises(ValueError, match="pessimistic.*base|base.*pessimistic"):
        grp.build_results_presentation(
            data_dir=tree, out_dir=tree / "deck", scenario="pessimistic", segments_config=tree / "no_such.toml"
        )


def test_all_surface_units_enumerates_segments(tmp_path):
    import generate_results_presentation as grp

    tree = _build_output_tree(tmp_path)
    units = grp._all_surface_units(tree, tree / "no_such.toml", "_base")
    names = [u[0] for u in units]
    assert "seg_a" in names
    cells = dict(zip(names, [u[1] for u in units]))["seg_a"]
    assert set(cells["category"]).issubset({"keep", "swap_in", "swap_out", "rejected"})
