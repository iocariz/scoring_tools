"""Tests for HRI phase 2 — the selectable optimization target (risk_indicator / optimum_hri)."""

import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessingSettings

BASE_KW = dict(
    keep_vars=["mis_date", "status_name"],
    indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
    variables=["v1", "v2"],
    octroi_bins=[-np.inf, 1, 2, np.inf],
    efx_bins=[-np.inf, 1, 2, np.inf],
    date_ini_book_obs="2024-01-01",
    date_fin_book_obs="2024-06-01",
)


def _settings(**over):
    kw = {**BASE_KW, **over}
    return PreprocessingSettings(**kw)


# ---------------------------------------------------------------------------
# Config validator + accessors
# ---------------------------------------------------------------------------


def test_default_is_b2_and_selected_target_is_optimum_risk():
    s = _settings(optimum_risk=1.4)
    assert s.risk_indicator == "b2_ever_h6"
    assert s.selected_target == pytest.approx(1.4)
    assert s.selected_indicator.output_col == "b2_ever_h6"
    assert s.selected_indicator.multiplier(s) == pytest.approx(s.multiplier)


def test_hri_requires_optimum_hri():
    with pytest.raises(ValueError, match="optimum_hri"):
        _settings(
            risk_indicator="hri_h6",
            indicators=BASE_KW["indicators"] + ["h_num_h6", "h_den_h6"],
        )


def test_hri_requires_source_columns_in_indicators():
    with pytest.raises(ValueError, match="h_num_h6"):
        _settings(risk_indicator="hri_h6", optimum_hri=1.0)


def test_hri_mode_selected_target_and_multiplier():
    s = _settings(
        risk_indicator="hri_h6",
        optimum_hri=0.9,
        indicators=BASE_KW["indicators"] + ["h_num_h6", "h_den_h6"],
    )
    assert s.selected_target == pytest.approx(0.9)
    assert s.selected_indicator.output_col == "hri_h6"
    assert s.selected_indicator.multiplier(s) == 1.0


# ---------------------------------------------------------------------------
# Hard data guard
# ---------------------------------------------------------------------------


def test_data_manager_fails_loudly_when_hri_target_columns_absent(monkeypatch):
    import src.data_manager as dm

    s = _settings(
        risk_indicator="hri_h6",
        optimum_hri=1.0,
        indicators=BASE_KW["indicators"] + ["h_num_h6", "h_den_h6"],
    )
    df = pd.DataFrame(
        {
            "mis_date": pd.to_datetime(["2024-01-01"]),
            "status_name": ["booked"],
            "oa_amt": [1.0],
            "oa_amt_h0": [1.0],
            "todu_30ever_h6": [0.0],
            "todu_amt_pile_h6": [1.0],
        }
    )
    monkeypatch.setattr("src.schema.validate_raw_data", lambda d, raise_on_error=True: None)
    with pytest.raises(dm.DataValidationError, match="hri_h6"):
        dm.load_and_prepare_data(s, preloaded_data=df)


# ---------------------------------------------------------------------------
# MILP: the mask flips with the target indicator
# ---------------------------------------------------------------------------


def _flip_grid():
    """Two bins where b2 and HRI DISAGREE about which bin is risky.

    Monotone accept sets along v1 (higher bin riskier under b2): {}, {1}, {1,2}.
    bin1: b2-safe (low todu ratio) but HRI-risky (high h ratio).
    bin2: b2-risky but HRI-safe.
    """
    return pd.DataFrame(
        {
            "v1": [1, 2],
            "oa_amt_h0": [100.0, 100.0],
            "todu_30ever_h6": [0.1, 2.0],  # b2: bin1 0.07%, bin1+2 ~1.47%
            "todu_amt_pile_h6": [1000.0, 1000.0],
            "h_num_h6": [30.0, 1.0],  # HRI: bin1 3.0%, bin1+2 ~1.55%
            "h_den_h6": [1000.0, 1000.0],
            "acct_booked_h0": [50.0, 50.0],
        }
    )


def test_milp_mask_flips_with_indicator():
    from src.optimization_utils import CellGrid, milp_solve_cutoffs

    grid = CellGrid.from_summary(_flip_grid(), ["v1"])

    # b2 target 0.5%: only bin1 fits (bin1 b2=0.07%; both = 0.735% > 0.5%)
    mask_b2 = milp_solve_cutoffs(grid, 0.5, [], multiplier=7.0)
    assert mask_b2 is not None and mask_b2.tolist() == [1, 0]

    # HRI target 2.0%: bin1 alone has HRI 3.0% > 2.0%, but both bins = 1.55% <= 2.0%.
    mask_hri = milp_solve_cutoffs(grid, 2.0, [], multiplier=1.0, risk_num_col="h_num_h6", risk_den_col="h_den_h6")
    assert mask_hri is not None and mask_hri.tolist() == [1, 1]
    assert mask_hri.tolist() != mask_b2.tolist()


def test_trace_pareto_frontier_orders_by_hri():
    from src.optimization_utils import trace_pareto_frontier

    df, grid, masks = trace_pareto_frontier(
        _flip_grid(),
        ["v1"],
        [],
        multiplier=7.0,
        indicators=["oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6"],
        n_points=10,
        show_progress=False,
        risk_num_col="h_num_h6",
        risk_den_col="h_den_h6",
        risk_col="hri_h6",
        target_multiplier=1.0,
    )
    assert not df.empty
    assert "hri_h6" in df.columns and "b2_ever_h6" in df.columns  # both indicators on every row
    assert df["hri_h6"].is_monotonic_increasing  # frontier ordered on the target indicator


# ---------------------------------------------------------------------------
# Selection on the HRI column
# ---------------------------------------------------------------------------


def test_selection_honors_hri_target_and_feasible_flag():
    from src.plots import RiskProductionVisualizer

    vis = RiskProductionVisualizer.__new__(RiskProductionVisualizer)
    vis.data_summary = pd.DataFrame(
        {
            "sol_fac": [0, 1, 2],
            "b2_ever_h6": [0.5, 1.0, 1.5],
            "hri_h6": [2.5, 1.2, 0.9],  # deliberately anti-correlated with b2
            "oa_amt_h0": [300.0, 200.0, 100.0],
        }
    )
    vis.target_sol_fac = None
    vis.risk_col = "hri_h6"
    vis.optimum_risk = 1.3  # HRI units

    row = vis._get_selected_solution_row()
    # candidates with hri <= 1.3: sol 1 (prod 200) and sol 2 (prod 100) -> max production = sol 1
    assert int(row.iloc[0]["sol_fac"]) == 1
    assert vis._selection_feasible is True

    vis.optimum_risk = 0.5  # unattainable in HRI units
    row = vis._get_selected_solution_row()
    assert vis._selection_feasible is False
    assert int(row.iloc[0]["sol_fac"]) == 2  # min-HRI fallback


# ---------------------------------------------------------------------------
# Model layer: target metric + RI gating
# ---------------------------------------------------------------------------


def test_process_dataset_hri_target_uses_h_columns_without_multiplier():
    from src.inference_optimized import process_dataset

    data = pd.DataFrame(
        {
            "v1": [1, 1, 2, 2],
            "v2": [1, 1, 1, 1],
            "todu_30ever_h6": [1.0, 1.0, 4.0, 4.0],
            "todu_amt_pile_h6": [100.0, 100.0, 100.0, 100.0],
            "h_num_h6": [2.0, 2.0, 8.0, 8.0],
            "h_den_h6": [100.0, 100.0, 100.0, 100.0],
            "oa_amt_h0": [10.0, 10.0, 10.0, 10.0],
        }
    )
    out = process_dataset(
        data,
        bins=([-np.inf, 1.5, np.inf], [-np.inf, np.inf]),
        variables=["v1", "v2"],
        indicators=["todu_30ever_h6", "todu_amt_pile_h6", "h_num_h6", "h_den_h6", "oa_amt_h0"],
        target_var="hri_h6",
        multiplier=1.0,
        var_reg=["v1", "v2"],
        z_threshold=0.0,
    )
    by_v1 = out.set_index("v1")["hri_h6"]
    assert by_v1.loc[1] == pytest.approx(4.0 / 200.0)  # plain ratio, no x7, not as %
    assert by_v1.loc[2] == pytest.approx(16.0 / 200.0)


def test_ri_uplift_scales_h_num_only_when_enabled():
    from src.reject_inference import apply_parceling_adjustment

    repesca = pd.DataFrame(
        {
            "v1": [1, 2],
            "v2": [1, 1],
            "todu_30ever_h6": [1.0, 1.0],
            "h_num_h6": [5.0, 5.0],
            "oa_amt": [10.0, 10.0],
        }
    )
    rates = pd.DataFrame({"v1": [1, 2], "v2": [1, 1], "acceptance_rate": [0.5, 0.5], "n_total": [100, 100]})

    off = apply_parceling_adjustment(
        repesca.copy(), rates, ["v1", "v2"], reject_uplift_factor=1.0, apply_hri_multiplier=False, quiet=True
    )
    on = apply_parceling_adjustment(
        repesca.copy(), rates, ["v1", "v2"], reject_uplift_factor=1.0, apply_hri_multiplier=True, quiet=True
    )
    assert (off["h_num_h6"] == 5.0).all()  # untouched when display-only
    assert (on["h_num_h6"] > 5.0).all()  # uplifted in hri-target mode
    assert (on["todu_30ever_h6"] == off["todu_30ever_h6"]).all()  # b2 uplift identical either way


# ---------------------------------------------------------------------------
# Reproducibility: additive fields, legacy references untouched
# ---------------------------------------------------------------------------


def test_compare_headline_hri_fields_additive():
    from src.reproducibility import Headline, compare_headline

    actual = Headline(
        segment="s",
        scenario="base",
        risk_pct=1.0,
        production_eur=100.0,
        n_accepted_cells=5,
        accepted_set_hash="abc",
        data_sha256="sha",
        config_hash="cfg",
        risk_indicator="hri_h6",
        hri_pct=0.9,
    )
    legacy_ref = {
        "risk_pct": 1.0,
        "production_eur": 100.0,
        "n_accepted_cells": 5,
        "accepted_set_hash": "abc",
        "data_sha256": "sha",
        "config_hash": "cfg",
    }
    assert compare_headline(actual, legacy_ref)["passed"] is True  # legacy refs compare as before

    ref_mismatch = {**legacy_ref, "risk_indicator": "b2_ever_h6"}
    res = compare_headline(actual, ref_mismatch)
    assert res["passed"] is False
    assert any("risk_indicator" in r for r in res["reasons"])

    ref_hri_drift = {**legacy_ref, "risk_indicator": "hri_h6", "hri_pct": 0.5}
    assert compare_headline(actual, ref_hri_drift)["passed"] is False
