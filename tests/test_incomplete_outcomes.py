"""Audit F1: partially-realized booked outcome pairs must never dilute risk.

One joint completeness rule (the backtest's #41 rule, applied upstream): a loan whose
H6 (or H3) numerator OR denominator is NaN contributes to NEITHER of that pair's sums —
production and every other indicator still count the loan. HRI pairs are exempt
(their nulls are sparse by design, not unrealized outcomes).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.utils import mask_incomplete_outcome_pairs

nan = float("nan")


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def test_mask_blanks_both_columns_of_a_partial_pair():
    df = pd.DataFrame(
        {
            "todu_30ever_h6": [1.0, nan, 2.0, nan],
            "todu_amt_pile_h6": [100.0, 900.0, nan, nan],
            "oa_amt_h0": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out, n_masked = mask_incomplete_outcome_pairs(df)
    assert n_masked == 2  # rows 1 (NaN num) and 2 (NaN den); row 3 (both NaN) needs no masking
    assert out["todu_30ever_h6"].sum() == pytest.approx(1.0)
    assert out["todu_amt_pile_h6"].sum() == pytest.approx(100.0)
    assert out["oa_amt_h0"].sum() == pytest.approx(100.0)  # production untouched
    assert df["todu_amt_pile_h6"].sum() == pytest.approx(1000.0)  # input frame not mutated


def test_mask_pairs_are_independent_and_noop_returns_original():
    df = pd.DataFrame(
        {
            "todu_30ever_h6": [1.0, nan],
            "todu_amt_pile_h6": [100.0, 900.0],
            "todu_30ever_h3": [0.5, 0.7],  # H3 pair complete — must survive the H6 masking
            "todu_amt_pile_h3": [50.0, 60.0],
        }
    )
    out, n = mask_incomplete_outcome_pairs(df)
    assert n == 1
    assert out["todu_30ever_h3"].sum() == pytest.approx(1.2)
    assert out["todu_amt_pile_h3"].sum() == pytest.approx(110.0)

    clean = pd.DataFrame({"todu_30ever_h6": [1.0], "todu_amt_pile_h6": [2.0]})
    out2, n2 = mask_incomplete_outcome_pairs(clean)
    assert n2 == 0 and out2 is clean  # nothing to mask -> no copy


def test_hri_pairs_are_exempt():
    """HRI nulls are sparse by design (no default event), NOT unrealized outcomes."""
    df = pd.DataFrame({"h_num_h6": [nan, 5.0], "h_den_h6": [100.0, 200.0]})
    out, n = mask_incomplete_outcome_pairs(df)
    assert n == 0
    assert out["h_den_h6"].sum() == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Optimizer aggregation (the audited defect)
# ---------------------------------------------------------------------------


def _booked_with_partial_h6():
    return pd.DataFrame(
        {
            "a": [0, 0],
            "status_name": ["booked", "booked"],
            "reject_reason": ["", ""],
            "oa_amt": [100.0, 900.0],
            "oa_amt_h0": [100.0, 900.0],
            "todu_30ever_h6": [1.0, nan],  # loan 2: unrealized numerator...
            "todu_amt_pile_h6": [100.0, 900.0],  # ...but real exposure
        }
    )


def test_aggregation_excludes_partial_h6_from_risk_but_not_production():
    from src.inference_optimized import compute_pre_reject_inference_data

    data = _booked_with_partial_h6()
    inds = ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"]
    risk_model = {"best_model_info": {"model": object()}, "features": ["a"]}
    with patch("src.models.calculate_risk_values", side_effect=lambda df, *a, **k: df):
        booked, _ = compute_pre_reject_inference_data(
            data, data, risk_model, None, 1.0, indicators=inds, variables=["a"], annual_coef=1
        )
    row = booked.iloc[0]
    assert row["oa_amt_h0_boo"] == pytest.approx(1000.0)  # production keeps both loans
    assert row["todu_30ever_h6_boo"] == pytest.approx(1.0)
    assert row["todu_amt_pile_h6_boo"] == pytest.approx(100.0)  # partial loan's exposure excluded
    # cell risk on the booked basis = 7 * 1/100 = 7.0%, not the diluted 0.7%
    assert 7 * row["todu_30ever_h6_boo"] / row["todu_amt_pile_h6_boo"] * 100 == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Bootstraps (same rule, per the audit's "apply to selection and fixed-policy bootstraps")
# ---------------------------------------------------------------------------


def test_bootstrap_worker_uses_joint_completeness():
    from src.optimization_utils import CellGrid, CutoffSpec
    from src.utils import _bootstrap_worker

    df = _booked_with_partial_h6()
    cell = pd.DataFrame({"a": [0], "oa_amt_h0": [1000.0], "todu_30ever_h6": [1.0], "todu_amt_pile_h6": [100.0]})
    grid = CellGrid.from_summary(cell, ["a"])
    spec = CutoffSpec.from_mask(np.array([1]), grid)
    # accept-everything spec; identity resample via frac=1 with fixed state
    production, risk_blended, risk_booked = _bootstrap_worker(df, spec, 7.0, random_state=0)
    # risk_booked is a fraction here (as_percentage=False): must reflect ONLY the complete loan
    sample = df.sample(frac=1.0, replace=True, random_state=0)
    complete = sample["todu_30ever_h6"].notna() & sample["todu_amt_pile_h6"].notna()
    expected = 7.0 * sample.loc[complete, "todu_30ever_h6"].sum() / sample.loc[complete, "todu_amt_pile_h6"].sum()
    assert risk_booked == pytest.approx(expected)


def test_selection_bootstrap_zeroes_risk_pair_jointly():
    from src.optimization_utils import CellGrid
    from src.selection_uncertainty import selection_aware_bootstrap

    df = _booked_with_partial_h6()
    cell = pd.DataFrame(
        {
            "a": [0],
            "oa_amt_h0": [1000.0],
            "todu_30ever_h6": [1.0],
            "todu_amt_pile_h6": [100.0],
        }
    )
    grid = CellGrid.from_summary(cell, ["a"])
    res = selection_aware_bootstrap(
        data_booked=df,
        grid=grid,
        pareto_masks=[np.array([1])],
        variables=["a"],
        threshold=50.0,
        multiplier=7.0,
        annual_coef=1.0,
        n_bootstraps=25,
    )
    if res is None:
        pytest.skip("selection_aware_bootstrap unavailable for this fixture shape")
    # With joint masking, every replicate's risk is built from the complete loan only —
    # the original point risk is 7*1/100*100 = 7.0%; the diluted-basis 0.7% must not appear.
    assert res.candidate_risk_orig[0] == pytest.approx(7.0, abs=1e-6)
