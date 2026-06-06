"""Tests for the golden-numbers reproducibility core (M5, src/reproducibility.py). Synthetic only."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from src.reproducibility import (
    Headline,
    _accepted_set_hash,
    compare_headline,
    extract_headline,
    load_reference,
    render_report,
    write_reference,
)


def _ref(**over):
    base = {
        "risk_pct": 1.0,
        "production_eur": 1000.0,
        "n_accepted_cells": 5,
        "accepted_set_hash": "abc123",
        "data_sha256": "deadbeef",
    }
    base.update(over)
    return base


def _headline(**over):
    base = dict(
        segment="seg_a",
        scenario="base",
        risk_pct=1.0,
        production_eur=1000.0,
        n_accepted_cells=5,
        accepted_set_hash="abc123",
        data_sha256="deadbeef",
    )
    base.update(over)
    return Headline(**base)


# ------------------------------ accepted-set hash ------------------------------


def test_accepted_set_hash_order_independent():
    a = _accepted_set_hash({(1.0, 2.0), (3.0, 4.0)})
    b = _accepted_set_hash({(3.0, 4.0), (1.0, 2.0)})
    assert a == b
    assert a != _accepted_set_hash({(1.0, 2.0)})


# ------------------------------- compare_headline ------------------------------


def test_compare_within_tolerance_passes():
    res = compare_headline(_headline(risk_pct=1.005, production_eur=1000.5), _ref(), risk_tol_pp=0.01, prod_tol_pct=0.1)
    assert res["passed"] is True
    assert res["snapshot_match"] is True
    assert res["reasons"] == []


def test_risk_drift_fails():
    res = compare_headline(_headline(risk_pct=1.2), _ref(), risk_tol_pp=0.01)
    assert res["passed"] is False
    assert any("risk drift" in r for r in res["reasons"])


def test_production_drift_fails():
    res = compare_headline(_headline(production_eur=1100.0), _ref(), prod_tol_pct=0.1)
    assert res["passed"] is False
    assert any("production drift" in r for r in res["reasons"])


def test_cell_count_mismatch_fails():
    res = compare_headline(_headline(n_accepted_cells=6), _ref())
    assert res["passed"] is False
    assert res["cells_match"] is False


def test_accepted_set_hash_mismatch_fails():
    res = compare_headline(_headline(accepted_set_hash="different"), _ref())
    assert res["passed"] is False
    assert res["accepted_set_match"] is False


def test_snapshot_drift_fails_even_when_numbers_match():
    # Identical numbers, different data SHA → snapshot drift forces FAIL (re-validate).
    res = compare_headline(_headline(data_sha256="CHANGED"), _ref())
    assert res["numbers_ok"] is True
    assert res["snapshot_match"] is False
    assert res["passed"] is False
    assert any("SNAPSHOT" in r.upper() for r in res["reasons"])


def test_missing_sha_does_not_block():
    # No SHA recorded on either side → snapshot pin not enforced.
    res = compare_headline(_headline(data_sha256=None), _ref(data_sha256=None))
    assert res["snapshot_match"] is True
    assert res["passed"] is True


# --------------------------------- reference IO --------------------------------


def test_reference_round_trip(tmp_path):
    h = _headline()
    path = write_reference(tmp_path / "ref.json", h)
    assert path.exists()
    loaded = load_reference(path)
    assert loaded["risk_pct"] == 1.0
    assert loaded["accepted_set_hash"] == "abc123"
    # A freshly-loaded reference compares equal to its own headline.
    assert compare_headline(h, loaded)["passed"] is True


# -------------------------------- extract_headline -----------------------------


def test_extract_headline_from_frozen_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 10.0]}).to_csv(data_dir / "accepted_cells_base.csv", index=False)
    pd.DataFrame([{"b2_ever_h6": 1.234, "oa_amt_h0": 5.0e7, "sol_fac": 3}]).to_csv(
        data_dir / "optimal_solution_base.csv", index=False
    )
    (data_dir / "run_lineage.json").write_text(
        json.dumps({"data": {"sha256": "abc123def456"}, "git": {"commit": "c0ffee"}, "config": {"hash": "cfg123"}})
    )
    settings = SimpleNamespace(segment_filter="seg_a", variables=["a", "b"])

    h = extract_headline(tmp_path, settings, suffix="_base")
    assert h.risk_pct == 1.234
    assert h.production_eur == 5.0e7
    assert h.n_accepted_cells == 2
    assert h.data_sha256 == "abc123def456"
    assert h.git_commit == "c0ffee"
    assert h.config_hash == "cfg123"


def test_render_report_contains_status():
    res = compare_headline(_headline(risk_pct=1.2), _ref())
    md = render_report(res, "seg_a")
    assert "FAIL" in md and "seg_a" in md
    md_pass = render_report(compare_headline(_headline(), _ref()), "seg_a")
    assert "PASS" in md_pass
