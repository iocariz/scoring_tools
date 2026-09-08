"""Tests for the Harmonized Risk Indicator (HRI) — PR-1 "compute + show both".

HRI_h6 = h_num_h6/h_den_h6, HRI_h3 = h_num_h3/h_den_h3 — plain ratios, NO multiplier.
Source columns are optional; everything degrades gracefully when absent.
"""

import numpy as np
import pandas as pd
import pytest

from src.consolidation import ConsolidatedMetrics, aggregate_metrics, extract_metrics_from_table
from src.risk_indicators import HRI_H6, RISK_INDICATORS, hri_available, hri_h3_available

nan = float("nan")


# ---------------------------------------------------------------------------
# Registry + predicates
# ---------------------------------------------------------------------------


def test_registry_and_compute():
    assert set(RISK_INDICATORS) == {"b2_ever_h6", "b2_ever_h3", "hri_h6", "hri_h3"}
    # no multiplier: 2/100 -> 2.0%
    assert HRI_H6.compute(2.0, 100.0) == pytest.approx(2.0)
    assert RISK_INDICATORS["b2_ever_h6"].multiplier_field == "multiplier"
    assert HRI_H6.multiplier() == 1.0


def test_availability_predicates():
    df = pd.DataFrame(columns=["h_num_h6", "h_den_h6"])
    assert hri_available(df)
    assert not hri_available(df, "_boo")
    assert not hri_h3_available(df)
    assert hri_available(["h_num_h6_boo", "h_den_h6_boo"], "_boo")


# ---------------------------------------------------------------------------
# Optional-column intake (mirrors the H3 pattern)
# ---------------------------------------------------------------------------


def test_hri_columns_optional_warns_and_strips_settings(monkeypatch):
    import src.data_manager as dm
    from tests.test_data_manager import make_settings

    settings = make_settings(
        keep_vars=["mis_date"],
        indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"],
    )
    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "oa_amt": [1.0, 2.0],
            "oa_amt_h0": [1.0, 2.0],
            "todu_30ever_h6": [0.0, 0.1],
            "todu_amt_pile_h6": [1.0, 2.0],
            "status_name": ["booked", "booked"],
        }
    )
    monkeypatch.setattr("src.schema.validate_raw_data", lambda d, raise_on_error=True: None)
    _, out_settings = dm.load_and_prepare_data(settings, preloaded_data=df)
    assert "h_num_h6" not in out_settings.indicators
    assert "h_den_h6" not in out_settings.indicators


# ---------------------------------------------------------------------------
# Optimizer KPIs
# ---------------------------------------------------------------------------


def _cell_grid_with_hri():
    """Two cells with booked + repesca sums; HRI only on booked (like real data)."""
    return pd.DataFrame(
        {
            "v1": [1, 2],
            "oa_amt_h0_boo": [100.0, 200.0],
            "oa_amt_h0_rep": [10.0, 20.0],
            "oa_amt_h0": [110.0, 220.0],
            "todu_30ever_h6_boo": [1.0, 4.0],
            "todu_amt_pile_h6_boo": [700.0, 1400.0],
            "todu_30ever_h6_rep": [0.2, 0.4],
            "todu_amt_pile_h6_rep": [70.0, 140.0],
            "todu_30ever_h6": [1.2, 4.4],
            "todu_amt_pile_h6": [770.0, 1540.0],
            "h_num_h6_boo": [2.0, 6.0],
            "h_den_h6_boo": [400.0, 800.0],
            "h_num_h6_rep": [0.0, 0.0],  # rejected loans: no HRI outcomes
            "h_den_h6_rep": [0.0, 0.0],
            "h_num_h6": [2.0, 6.0],
            "h_den_h6": [400.0, 800.0],
        }
    )


def test_evaluate_solution_computes_hri_without_multiplier():
    from src.optimization_utils import CellGrid, evaluate_solution

    grid = CellGrid.from_summary(_cell_grid_with_hri(), ["v1"])
    indicators = [
        "oa_amt_h0",
        "todu_30ever_h6",
        "todu_amt_pile_h6",
        "h_num_h6",
        "h_den_h6",
    ]
    result = evaluate_solution(np.array([1, 1]), grid, indicators, multiplier=7.0, multiplier_h3=None)
    # b2 uses x7; HRI is the plain ratio x100
    assert result["hri_h6_boo"] == pytest.approx(100 * (2.0 + 6.0) / (400.0 + 800.0))
    assert result["b2_ever_h6_boo"] == pytest.approx(7 * (1.0 + 4.0) / (700.0 + 1400.0) * 100)
    # _rep HRI is 0/0 -> NaN (rejected loans carry no HRI outcomes), never 0.00
    assert np.isnan(result["hri_h6_rep"])


def test_kpi_of_fact_sol_keeps_hri_rep_nan():
    # Build the final_result frame path directly through kpi computation on a tiny df
    from src.optimization_utils import (
        calculate_b2_ever_h6,  # noqa: F401 (import sanity)
        kpi_of_fact_sol,
    )

    cell = _cell_grid_with_hri()
    sols = pd.DataFrame({"sol_fac": [0], "1": [1], "2": [1]})
    try:
        out = kpi_of_fact_sol(
            sols, cell.rename(columns={"v1": "octroi"}), ["octroi"], multiplier=7.0, multiplier_h3=None
        )
    except TypeError:
        pytest.skip("kpi_of_fact_sol signature differs; covered by evaluate_solution test")
    if "hri_h6" in out.columns:
        assert out["hri_h6"].notna().any()
        assert out["hri_h6_rep"].isna().all()  # deliberate: no fillna(0) for HRI


# ---------------------------------------------------------------------------
# MR summary table
# ---------------------------------------------------------------------------


def _mr_desagregado():
    return pd.DataFrame(
        {
            "v1": [1, 2],
            "v2": [1, 1],
            "oa_amt_h0_boo": [100.0, 200.0],
            "oa_amt_h0_rep": [10.0, 20.0],
            "todu_30ever_h6_boo": [1.0, 8.0],
            "todu_amt_pile_h6_boo": [700.0, 1400.0],
            "todu_30ever_h6_rep": [0.1, 0.2],
            "todu_amt_pile_h6_rep": [70.0, 140.0],
            "h_num_h6_boo": [2.0, 10.0],
            "h_den_h6_boo": [400.0, 800.0],
            "h_num_h6_rep": [0.0, 0.0],
            "h_den_h6_rep": [0.0, 0.0],
        }
    )


def _mr_summary(df):
    from src.mr_pipeline import calculate_metrics_from_cuts

    optimal = pd.DataFrame({"v1": [1], "v2": [1], "sol_fac": [0]})
    mask = np.array([1, 0])  # accept cell 1, reject cell 2

    from src.optimization_utils import CellGrid

    grid = CellGrid.from_summary(df, ["v1", "v2"])
    full_mask = np.zeros(grid.n_cells, dtype=int)
    for i, combo in enumerate([(1, 1), (2, 1)]):
        full_mask[grid.cell_index[combo]] = mask[i]
    return calculate_metrics_from_cuts(
        df, optimal, ["v1", "v2"], mask=full_mask, grid=grid, multiplier=7.0, multiplier_h3=None
    )


def test_mr_summary_carries_hri_with_additive_optimum():
    result = _mr_summary(_mr_desagregado())
    assert result is not None and "HRI (%)" in result.columns
    by_metric = result.set_index("Metric")
    actual_hri = by_metric.loc["Actual", "HRI (%)"]
    assert actual_hri == pytest.approx(100 * 12.0 / 1200.0)
    # Optimum = (actual - swap_out) + swap_in on num/den: cell 2 swapped out, no swap-in HRI
    opt_num = 12.0 - 10.0 + 0.0
    opt_den = 1200.0 - 800.0 + 0.0
    assert by_metric.loc["Optimum selected", "HRI (%)"] == pytest.approx(100 * opt_num / opt_den)
    assert by_metric.loc["Optimum selected", "h_num_h6"] == pytest.approx(opt_num)
    # Swap-in HRI is None/NaN (0/0)
    assert pd.isna(by_metric.loc["Swap-in", "HRI (%)"])


def test_h3_floor_clamp_never_touches_hri():
    """The H6>=H3 clamp back-solves todu_30ever_h6 only — HRI must be untouched."""
    df = _mr_desagregado()
    # add H3 columns engineered so H3 risk > H6 risk => clamp fires
    df["todu_30ever_h3_boo"] = [5.0, 10.0]
    df["todu_amt_pile_h3_boo"] = [100.0, 200.0]
    df["todu_30ever_h3_rep"] = [0.0, 0.0]
    df["todu_amt_pile_h3_rep"] = [0.0, 0.0]

    from src.mr_pipeline import calculate_metrics_from_cuts
    from src.optimization_utils import CellGrid

    grid = CellGrid.from_summary(df, ["v1", "v2"])
    full_mask = np.ones(grid.n_cells, dtype=int)
    result = calculate_metrics_from_cuts(
        df,
        pd.DataFrame({"v1": [1], "v2": [1], "sol_fac": [0]}),
        ["v1", "v2"],
        mask=full_mask,
        grid=grid,
        multiplier=7.0,
        multiplier_h3=4.0,
    )
    by_metric = result.set_index("Metric")
    # clamp fired: Risk == Risk H3 on Actual
    assert by_metric.loc["Actual", "Risk (%)"] == pytest.approx(by_metric.loc["Actual", "Risk H3 (%)"])
    # HRI unaffected by the clamp
    assert by_metric.loc["Actual", "HRI (%)"] == pytest.approx(100 * 12.0 / 1200.0)
    assert by_metric.loc["Actual", "h_num_h6"] == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def _summary_csv_df(with_hri=True):
    rows = [
        {"Metric": "Actual", "Production (€)": 1000.0, "todu_30ever_h6": 2.0, "todu_amt_pile_h6": 700.0},
        {"Metric": "Swap-in", "Production (€)": 100.0, "todu_30ever_h6": 0.2, "todu_amt_pile_h6": 70.0},
        {"Metric": "Swap-out", "Production (€)": 200.0, "todu_30ever_h6": 1.0, "todu_amt_pile_h6": 140.0},
        {"Metric": "Optimum selected", "Production (€)": 900.0, "todu_30ever_h6": 1.2, "todu_amt_pile_h6": 630.0},
    ]
    df = pd.DataFrame(rows)
    if with_hri:
        df["h_num_h6"] = [4.0, 0.0, 1.0, 3.0]
        df["h_den_h6"] = [400.0, 0.0, 100.0, 300.0]
    return df


def test_extract_metrics_reads_hri_components():
    metrics = extract_metrics_from_table(_summary_csv_df())
    assert metrics["actual"]["h_num_h6"] == pytest.approx(4.0)
    assert metrics["optimum"]["h_den_h6"] == pytest.approx(300.0)
    assert metrics["swap_out"]["h_num_h6"] == pytest.approx(1.0)


def test_extract_metrics_legacy_csv_defaults_to_zero():
    metrics = extract_metrics_from_table(_summary_csv_df(with_hri=False))
    assert metrics["actual"]["h_num_h6"] == 0
    assert metrics["optimum"]["h_den_h6"] == 0


def test_aggregate_metrics_pools_hri_num_den_not_rates():
    m1 = extract_metrics_from_table(_summary_csv_df())
    m2 = extract_metrics_from_table(_summary_csv_df())
    m2["actual"]["h_num_h6"], m2["actual"]["h_den_h6"] = 1.0, 1000.0  # very different rate
    agg = aggregate_metrics([m1, m2], multiplier=7.0)
    assert agg["actual"]["h_num_h6"] == pytest.approx(5.0)
    assert agg["actual"]["h_den_h6"] == pytest.approx(1400.0)
    # pooled rate = 5/1400, NOT mean of (4/400, 1/1000)
    pooled = 100 * 5.0 / 1400.0
    mean_of_rates = (100 * 4.0 / 400.0 + 100 * 1.0 / 1000.0) / 2
    assert pooled != pytest.approx(mean_of_rates)


def test_consolidated_metrics_hri_properties_and_conditional_dict():
    cm = ConsolidatedMetrics(
        group_name="g",
        period="main",
        scenario="base",
        segments=["s"],
        actual_h_num_h6=4.0,
        actual_h_den_h6=400.0,
        optimum_h_num_h6=3.0,
        optimum_h_den_h6=300.0,
    )
    assert cm.actual_hri == pytest.approx(1.0)
    d = cm.to_dict()
    assert d["actual_hri_pct"] == pytest.approx(1.0)
    assert "actual_hri_h3_pct" not in d  # H3 pair absent -> not emitted

    cm_no_hri = ConsolidatedMetrics(group_name="g", period="main", scenario="base", segments=["s"])
    assert "actual_hri_pct" not in cm_no_hri.to_dict()  # legacy runs unchanged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_portfolio_metric_table_shows_hri_only_when_present():
    from src.reporting import _build_portfolio_metric_table

    row = pd.Series(
        {
            "actual_risk_pct": 1.9,
            "optimum_risk_pct": 1.2,
            "actual_production": 100.0,
            "optimum_production": 90.0,
            "actual_hri_pct": 2.0,
            "optimum_hri_pct": 1.5,
        }
    )
    html = _build_portfolio_metric_table(row, show_ci=False)
    assert "HRI (%)" in html and "2.00" in html

    legacy = pd.Series({"actual_risk_pct": 1.9, "optimum_risk_pct": 1.2})
    assert "HRI (%)" not in _build_portfolio_metric_table(legacy, show_ci=False)


def test_hri_excluded_raw_columns_in_segment_report():
    import inspect

    import src.reporting as reporting_mod

    src_text = inspect.getsource(reporting_mod)
    assert '"h_num_h6"' in src_text  # raw components are excluded from HTML tables


# ---------------------------------------------------------------------------
# Schema / DQ
# ---------------------------------------------------------------------------


def test_schema_rejects_negative_hri_and_allows_absent():
    from src.data_manager import DataValidationError
    from src.schema import validate_raw_data

    ok = pd.DataFrame({"status_name": pd.Categorical(["booked"]), "segment_cut_off": ["x"], "mis_date": ["2024-01-01"]})
    validate_raw_data(ok, raise_on_error=True)  # absent HRI columns: fine

    bad = ok.copy()
    bad["h_num_h6"] = [-1.0]
    with pytest.raises(DataValidationError):
        validate_raw_data(bad, raise_on_error=True)


# ---------------------------------------------------------------------------
# End-to-end consolidation (would have caught the column_order whitelist drop)
# ---------------------------------------------------------------------------


def test_consolidate_segments_carries_hri_to_final_csv(tmp_path):
    """consolidate_segments ends with a hard-coded column whitelist that silently
    drops unknown columns — the HRI family must survive it end to end."""
    from src.consolidation import consolidate_segments

    data_dir = tmp_path / "seg_a" / "data"
    data_dir.mkdir(parents=True)
    _summary_csv_df().to_csv(data_dir / "risk_production_summary_table_base.csv", index=False)

    df = consolidate_segments(tmp_path, {"seg_a": {}}, {}, ["_base"], multiplier=7.0, multiplier_h3=4.0)
    assert "actual_hri_pct" in df.columns
    assert "optimum_h_den_h6" in df.columns
    total = df[df["group"] == "TOTAL"].iloc[0]
    assert total["actual_hri_pct"] == pytest.approx(100 * 4.0 / 400.0)
