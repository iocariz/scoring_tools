"""Batch bin-edge resolution tests (audit #27 — fixed supersegment edges are
configuration and must survive a data-preload failure)."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from run_batch import _inject_learned_bin_edges, _segment_pins_bin_edges, learn_supersegment_bin_edges

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


# =============================================================================
# #60 — segment-pinned bin_edges must survive global/supersegment injection
# =============================================================================

_PINNED = [float("-inf"), 2000.0, float("inf")]
_GLOBAL = [float("-inf"), 5000.0, float("inf")]
_SS = [float("-inf"), 7000.0, float("inf")]


def _merged_with(bin_edges=None):
    """A merged_config whose income_bin either pins bin_edges (segment pin) or only has max_bins."""
    bc = {"source_col": "income", "output_col": "income_bin", "max_bins": 2}
    if bin_edges is not None:
        bc["bin_edges"] = bin_edges
    return {"bins": {"income_bin": bc}}


class TestSegmentPinnedBinEdgesRespected:
    def test_detects_segment_pin(self):
        assert _segment_pins_bin_edges({"bins": {"income_bin": {"bin_edges": _PINNED}}}, "income_bin")
        assert not _segment_pins_bin_edges({"bins": {"income_bin": {"max_bins": 2}}}, "income_bin")
        assert not _segment_pins_bin_edges({}, "income_bin")

    def test_global_injection_does_not_clobber_segment_pin(self):
        seg = {"bins": {"income_bin": {"bin_edges": _PINNED}}}
        merged = _merged_with(_PINNED)  # merge_configs already folded the pin in
        out = _inject_learned_bin_edges(merged, seg, "seg", {"income_bin": _GLOBAL}, None)
        assert out["bins"]["income_bin"]["bin_edges"] == _PINNED  # kept, NOT the global _GLOBAL

    def test_supersegment_injection_does_not_clobber_segment_pin(self):
        seg = {"bins": {"income_bin": {"bin_edges": _PINNED}}, "reporting_supersegment": "ss"}
        merged = _merged_with(_PINNED)
        out = _inject_learned_bin_edges(merged, seg, "seg", None, {"ss": {"income_bin": _SS}})
        assert out["bins"]["income_bin"]["bin_edges"] == _PINNED  # kept, NOT the supersegment _SS

    def test_global_injected_when_segment_does_not_pin(self):
        seg = {"bins": {"income_bin": {"max_bins": 2}}}
        merged = _merged_with(None)
        out = _inject_learned_bin_edges(merged, seg, "seg", {"income_bin": _GLOBAL}, None)
        assert out["bins"]["income_bin"]["bin_edges"] == _GLOBAL  # learned edges applied as before

    def test_supersegment_overrides_global_when_no_pin(self):
        seg = {"bins": {"income_bin": {"max_bins": 2}}, "reporting_supersegment": "ss"}
        merged = _merged_with(None)
        out = _inject_learned_bin_edges(merged, seg, "seg", {"income_bin": _GLOBAL}, {"ss": {"income_bin": _SS}})
        assert out["bins"]["income_bin"]["bin_edges"] == _SS  # supersegment wins over global (unchanged behavior)
