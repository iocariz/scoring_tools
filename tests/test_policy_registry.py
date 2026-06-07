"""Tests for the policy registry + champion/challenger comparison (phase 1). Synthetic data only."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import pytest

from src.policy_registry import (
    PolicyComparison,
    PolicyEntry,
    _risk_verdict,
    build_policy_entry,
    compare_policies,
    get_champion,
    load_registry,
    register_policy,
    registry_path,
    write_comparison_consolidated,
    write_comparison_report,
)

# --------------------------------------------------------------------------- #
# Fixtures / factories
# --------------------------------------------------------------------------- #


def _entry(segment="seg_a", policy_id="seg_a-aaaaaaaa", cells=((1.0, 10.0), (2.0, 10.0))) -> PolicyEntry:
    return PolicyEntry(
        policy_id=policy_id,
        segment=segment,
        scenario="base",
        variables=("a", "b"),
        bin_edges={"a": [0.0, 1.0, 2.0]},
        accepted_cells=tuple(cells),
        accepted_set_hash=policy_id.split("-")[-1],
        headline={"risk_pct": 1.0, "production_eur": 1000.0, "n_accepted_cells": len(cells)},
        provenance={"data_sha256": "abc", "git_commit": "def", "config_hash": "ghi", "created": "2026-06-06T00:00:00Z"},
        evidence={"lineage": "x", "headline_reference": "y", "mrm_signoff": "z"},
    )


def _build_run_tree(tmp_path, seg="seg_a"):
    """Materialize a completed run's per-segment data dir that build_policy_entry reads."""
    data_dir = tmp_path / seg / "data"
    data_dir.mkdir(parents=True)
    pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 10.0]}).to_csv(data_dir / "accepted_cells_base.csv", index=False)
    pd.DataFrame({"b2_ever_h6": [1.5], "oa_amt_h0": [1000.0]}).to_csv(
        data_dir / "optimal_solution_base.csv", index=False
    )
    (data_dir / "run_lineage.json").write_text(
        json.dumps({"data": {"sha256": "SHA123"}, "git": {"commit": "COMMIT9"}, "config": {"hash": "CFG7"}})
    )
    return tmp_path / seg


def _settings(seg="seg_a"):
    return SimpleNamespace(
        segment_filter=seg,
        variables=["a", "b"],
        multiplier=7.0,
        bins={"a": SimpleNamespace(bin_edges=[0.0, 1.0, 2.0])},
    )


# --------------------------------------------------------------------------- #
# PolicyEntry roundtrip
# --------------------------------------------------------------------------- #


def test_policy_entry_roundtrip():
    e = _entry()
    d = e.to_dict()
    assert isinstance(d["accepted_cells"], list) and isinstance(d["variables"], list)
    e2 = PolicyEntry.from_dict(d)
    assert e2 == e
    assert e2.accepted_set() == {(1.0, 10.0), (2.0, 10.0)}


def test_policy_entry_accepted_set_is_tuples():
    e = PolicyEntry.from_dict(_entry().to_dict())
    assert all(isinstance(c, tuple) for c in e.accepted_set())


# --------------------------------------------------------------------------- #
# build_policy_entry from a synthetic run tree
# --------------------------------------------------------------------------- #


def test_build_policy_entry(tmp_path):
    seg_dir = _build_run_tree(tmp_path, "seg_a")
    entry = build_policy_entry(seg_dir, _settings("seg_a"), "_base", created="2026-06-06T00:00:00Z")
    assert entry.segment == "seg_a"
    assert entry.scenario == "base"
    assert entry.policy_id.startswith("seg_a-") and len(entry.policy_id.split("-")[-1]) == 8
    assert entry.accepted_set() == {(1.0, 10.0), (2.0, 10.0)}
    assert entry.headline == {"risk_pct": 1.5, "production_eur": 1000.0, "n_accepted_cells": 2}
    assert entry.provenance["data_sha256"] == "SHA123"
    assert entry.provenance["git_commit"] == "COMMIT9"
    assert entry.bin_edges == {"a": [0.0, 1.0, 2.0]}


# --------------------------------------------------------------------------- #
# Registry I/O
# --------------------------------------------------------------------------- #


def test_load_registry_missing_returns_skeleton(tmp_path):
    reg = load_registry("seg_a", tmp_path)
    assert reg == {"segment": "seg_a", "champion_policy_id": None, "policies": []}
    assert get_champion("seg_a", tmp_path) is None


def test_register_bootstraps_champion(tmp_path):
    reg = register_policy(_entry(policy_id="seg_a-1111"), registry_dir=tmp_path)
    assert reg["champion_policy_id"] == "seg_a-1111"
    assert registry_path("seg_a", tmp_path).exists()
    champ = get_champion("seg_a", tmp_path)
    assert champ is not None and champ.policy_id == "seg_a-1111" and champ.status == "champion"


def test_register_second_is_challenger_then_promote(tmp_path):
    register_policy(_entry(policy_id="seg_a-1111"), registry_dir=tmp_path)
    register_policy(_entry(policy_id="seg_a-2222"), registry_dir=tmp_path)
    reg = load_registry("seg_a", tmp_path)
    assert reg["champion_policy_id"] == "seg_a-1111"  # bootstrap champion unchanged
    assert {p["policy_id"]: p["status"] for p in reg["policies"]} == {
        "seg_a-1111": "champion",
        "seg_a-2222": "challenger",
    }
    # Promote the second.
    register_policy(_entry(policy_id="seg_a-2222"), make_champion=True, registry_dir=tmp_path)
    reg = load_registry("seg_a", tmp_path)
    assert reg["champion_policy_id"] == "seg_a-2222"
    assert len(reg["policies"]) == 2  # idempotent: no duplicate appended


def test_register_idempotent(tmp_path):
    register_policy(_entry(policy_id="seg_a-1111"), registry_dir=tmp_path)
    register_policy(_entry(policy_id="seg_a-1111"), registry_dir=tmp_path)
    assert len(load_registry("seg_a", tmp_path)["policies"]) == 1


# --------------------------------------------------------------------------- #
# Risk verdict (noise-aware)
# --------------------------------------------------------------------------- #


def test_verdict_inconclusive_when_few_defaults():
    champ = {"n_defaults": 5, "risk_ci_lower": 2.0, "risk_ci_upper": 3.0}
    chal = {"n_defaults": 50, "risk_ci_lower": 0.1, "risk_ci_upper": 0.5}
    assert _risk_verdict(champ, chal) == "INCONCLUSIVE"  # min defaults 5 < 10


def test_verdict_better_when_challenger_ci_below():
    champ = {"n_defaults": 30, "risk_ci_lower": 2.0, "risk_ci_upper": 3.0}
    chal = {"n_defaults": 30, "risk_ci_lower": 0.5, "risk_ci_upper": 1.0}
    assert _risk_verdict(champ, chal) == "BETTER"


def test_verdict_worse_when_challenger_ci_above():
    champ = {"n_defaults": 30, "risk_ci_lower": 0.5, "risk_ci_upper": 1.0}
    chal = {"n_defaults": 30, "risk_ci_lower": 2.0, "risk_ci_upper": 3.0}
    assert _risk_verdict(champ, chal) == "WORSE"


def test_verdict_inconclusive_when_ci_overlap():
    champ = {"n_defaults": 30, "risk_ci_lower": 1.0, "risk_ci_upper": 3.0}
    chal = {"n_defaults": 30, "risk_ci_lower": 2.0, "risk_ci_upper": 4.0}
    assert _risk_verdict(champ, chal) == "INCONCLUSIVE"


def test_verdict_inconclusive_when_ci_missing():
    champ = {"n_defaults": 30, "risk_ci_lower": None, "risk_ci_upper": None}
    chal = {"n_defaults": 30, "risk_ci_lower": 0.5, "risk_ci_upper": 1.0}
    assert _risk_verdict(champ, chal) == "INCONCLUSIVE"


# --------------------------------------------------------------------------- #
# compare_policies
# --------------------------------------------------------------------------- #


def _cohort():
    # 4 cells; varying risk. Each row is one booked loan.
    return pd.DataFrame(
        {
            "a": [1.0, 1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 10.0, 10.0, 10.0, 10.0],
            "oa_amt_h0": [100.0, 100.0, 200.0, 300.0, 400.0],
            "todu_30ever_h6": [0.0, 1.0, 2.0, 5.0, 9.0],
            "todu_amt_pile_h6": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )


def test_compare_policies_cell_diff():
    champ = {(1.0, 10.0), (2.0, 10.0)}
    chal = {(2.0, 10.0), (3.0, 10.0), (4.0, 10.0)}
    cmp = compare_policies(champ, chal, _cohort(), ["a", "b"], 7.0, segment="seg_a")
    assert cmp.added_cells == [(3.0, 10.0), (4.0, 10.0)]  # in challenger, not champion
    assert cmp.removed_cells == [(1.0, 10.0)]  # in champion, not challenger
    assert cmp.n_champion_cells == 2 and cmp.n_challenger_cells == 3


def test_compare_policies_metrics_and_deltas():
    champ = {(1.0, 10.0)}  # rows 0,1 → oa 200, t30 1, pile 200
    chal = {(4.0, 10.0)}  # row 4 → oa 400, t30 9, pile 100
    cmp = compare_policies(champ, chal, _cohort(), ["a", "b"], 7.0, segment="seg_a")
    assert cmp.champion["production"] == 200.0
    assert cmp.challenger["production"] == 400.0
    assert cmp.production_delta_pct == pytest.approx((400.0 - 200.0) / 200.0 * 100)
    # risk = mult * t30 / pile * 100
    assert cmp.champion["risk"] == pytest.approx(7.0 * 1.0 / 200.0 * 100)
    assert cmp.challenger["risk"] == pytest.approx(7.0 * 9.0 / 100.0 * 100)
    assert cmp.risk_delta_pp == pytest.approx(cmp.challenger["risk"] - cmp.champion["risk"])
    assert cmp.verdict == "INCONCLUSIVE"  # tiny cohort → few defaults


def test_compare_policies_variables_recorded():
    cmp = compare_policies({(1.0, 10.0)}, {(2.0, 10.0)}, _cohort(), ["a", "b"], 7.0, segment="seg_a")
    assert cmp.variables == ["a", "b"]


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #


def _comparison():
    return PolicyComparison(
        segment="seg_a",
        basis="realized_oot",
        variables=["a", "b"],
        champion_policy_id="seg_a-1111",
        challenger_policy_id="seg_a-2222",
        n_champion_cells=2,
        n_challenger_cells=3,
        added_cells=[(3.0, 10.0)],
        removed_cells=[(1.0, 10.0)],
        champion={"risk": 2.0, "production": 200.0, "n_defaults": 12, "risk_ci_lower": 1.5, "risk_ci_upper": 2.5},
        challenger={"risk": 1.0, "production": 300.0, "n_defaults": 12, "risk_ci_lower": 0.5, "risk_ci_upper": 1.4},
        risk_delta_pp=-1.0,
        production_delta_pct=50.0,
        verdict="BETTER",
    )


def test_write_comparison_report(tmp_path):
    paths = write_comparison_report(_comparison(), tmp_path)
    assert paths["aggregate"].exists()
    assert paths["celldiff"].exists()
    agg = pd.read_csv(paths["aggregate"])
    assert agg.loc[0, "verdict"] == "BETTER"
    assert agg.loc[0, "n_added_cells"] == 1 and agg.loc[0, "n_removed_cells"] == 1
    diff = pd.read_csv(paths["celldiff"])
    assert set(diff["change"]) == {"added", "removed"}
    assert {"a", "b"}.issubset(diff.columns)


def test_write_comparison_report_no_celldiff_when_identical(tmp_path):
    c = _comparison()
    c.added_cells, c.removed_cells = [], []
    paths = write_comparison_report(c, tmp_path)
    assert "celldiff" not in paths


def test_write_comparison_consolidated(tmp_path):
    skipped = PolicyComparison(segment="seg_b", basis="realized_oot", sufficient=False, message="no champion")
    paths = write_comparison_consolidated([_comparison(), skipped], tmp_path)
    assert paths["consolidated"].exists() and paths["summary"].exists()
    rows = pd.read_csv(paths["consolidated"])
    assert set(rows["segment"]) == {"seg_a", "seg_b"}
    md = paths["summary"].read_text()
    assert "seg_a" in md and "no champion" in md
