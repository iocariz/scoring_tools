"""Batch bin-edge resolution tests (audit #27 — fixed supersegment edges are
configuration and must survive a data-preload failure)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from run_batch import learn_supersegment_bin_edges

SS_FIXED = {
    "pl_known": {
        "segment_filters": ["premium", "precon"],
        "bin_edges": {"income_bin": [-np.inf, 2000.0, np.inf]},
    }
}

BASE_CFG_LEARNABLE = {
    "bins": {
        "income_bin": {"source_col": "income", "output_col": "income_bin", "max_bins": 2},
    },
    "date_ini_book_obs": None,
    "date_fin_book_obs": None,
}


def _demand_frame():
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame(
        {
            "income": rng.uniform(500, 5000, n),
            "segment_cut_off": ["premium"] * n,
            "fuera_norma": ["n"] * n,
            "fraud_flag": ["n"] * n,
            "nature_holder": ["person"] * n,
        }
    )


def test_fixed_edges_survive_preload_failure():
    """data=None (preload failed) must still yield the configured fixed edges —
    the old gate dropped them entirely, silently degrading every segment to
    per-segment quantile learning."""
    result = learn_supersegment_bin_edges(None, BASE_CFG_LEARNABLE, SS_FIXED)
    assert result == {"pl_known": {"income_bin": [-np.inf, 2000.0, np.inf]}}


def test_fixed_edges_extracted_even_when_nothing_learnable():
    """Fixed supersegment edges are configuration — they must flow through even
    when no global bin needs learning (the old early-return dropped them)."""
    cfg_all_fixed = {
        "bins": {
            "income_bin": {
                "source_col": "income",
                "output_col": "income_bin",
                "bin_edges": [-np.inf, 1000.0, np.inf],
            },
        },
    }
    result = learn_supersegment_bin_edges(_demand_frame(), cfg_all_fixed, SS_FIXED)
    assert result == {"pl_known": {"income_bin": [-np.inf, 2000.0, np.inf]}}


def test_learn_own_edges_skipped_without_data():
    """learn_own_bin_edges needs data; with data=None it must skip (the batch
    gate halts the run) — not crash, and not invent edges."""
    ss = {"others": {"segment_filters": ["open_banking"], "learn_own_bin_edges": True}}
    result = learn_supersegment_bin_edges(None, BASE_CFG_LEARNABLE, ss)
    assert result == {}


def test_learn_own_edges_still_works_with_data():
    ss = {"others": {"segment_filters": ["premium"], "learn_own_bin_edges": True}}
    result = learn_supersegment_bin_edges(_demand_frame(), BASE_CFG_LEARNABLE, ss)
    assert "others" in result
    assert "income_bin" in result["others"]
    assert len(result["others"]["income_bin"]) >= 2
