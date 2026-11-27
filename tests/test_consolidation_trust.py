"""Tests for the consolidated-report trust layer (CIs, Validation & Governance, Out-of-time sheets)."""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from openpyxl import Workbook

from src.report.consolidation import (
    _backtest_flags_by_segment,
    _ci_eur,
    _ci_pct,
    _fmt_pct_or_na,
    _read_backtest_consolidated,
    _segment_status,
    _write_exec_recommendation,
    _write_sheet_oot_validation,
    _write_sheet_validation_governance,
)

# ------------------------------- CI formatters -------------------------------


def test_ci_pct_and_eur():
    row = {"risk_ci_lower": 0.42, "risk_ci_upper": 1.24, "production_ci_lower": 39.4e6, "production_ci_upper": 42.0e6}
    assert _ci_pct(row, "risk_ci_lower", "risk_ci_upper") == "95% CI [0.42–1.24]"
    assert _ci_eur(row, "production_ci_lower", "production_ci_upper") == "95% CI [€39.4M–€42.0M]"


def test_ci_none_when_missing():
    assert _ci_pct({"risk_ci_lower": None, "risk_ci_upper": 1.0}, "risk_ci_lower", "risk_ci_upper") is None
    assert _ci_eur({}, "production_ci_lower", "production_ci_upper") is None


# ------------------------------- status mapping ------------------------------


def test_segment_status_traffic_light():
    assert _segment_status("OK", {})[0] == "OK"
    assert _segment_status("INCONCLUSIVE", {})[0] == "WATCH"
    assert _segment_status("DRIFT", {})[0] == "REVIEW"
    assert _segment_status("OK", {"critical": 1})[0] == "REVIEW"  # stability overrides up
    assert _segment_status("OK", {"warning": 2})[0] == "WATCH"
    assert _segment_status(None, {"worst_status": "moderate"})[0] == "WATCH"


# --------------------------- synthetic output tree ---------------------------


def _build_tree(tmp_path, seg="no_premium_cd"):
    data = tmp_path / seg / "data"
    data.mkdir(parents=True)
    (data / "run_lineage.json").write_text(
        json.dumps(
            {
                "run_id": "20260606T120000Z_abc123",
                "data": {
                    "path": "data/x.sas7bdat",
                    "sha256": "deadbeefcafe0000",
                    "mtime": "2026-03-11",
                    "rows_loaded": 1444802,
                },
                "git": {"short": "abc123", "dirty": False},
                "config": {"hash": "cfg123"},
                "assumptions": {"multiplier": 7.0, "stress_mode": "disabled", "use_mr_outcomes": True},
            }
        )
    )
    (data / "drift_alerts_base.json").write_text(json.dumps({"summary": {"info": 4, "warning": 0, "critical": 0}}))
    pd.DataFrame({"variable": ["oa_amt"], "psi": [0.08], "status": ["stable"], "n_bins": [9]}).to_csv(
        data / "stability_psi_base.csv", index=False
    )
    bt = tmp_path / "backtest"
    bt.mkdir()
    pd.DataFrame(
        [
            {
                "segment": seg,
                "drift_flag": "OK",
                "n_defaults_oot": 12,
                "predicted_risk_pct": 1.389,
                "in_sample_realized_risk_pct": 0.97,
                "in_sample_risk_ci": "[0.62, 1.37]",
                "oot_realized_risk_pct": 2.11,
                "oot_risk_ci": "[0.73, 3.93]",
                "in_sample_acceptance_rate": 0.46,
                "oot_acceptance_rate": 0.45,
                "holdout_start": "2025-05-01",
                "holdout_end": "2025-07-01",
                "pct_mature": 0.26,
            }
        ]
    ).to_csv(bt / "backtest_consolidated_base.csv", index=False)
    return seg


def test_read_backtest_consolidated(tmp_path):
    seg = _build_tree(tmp_path)
    df = _read_backtest_consolidated(tmp_path, "_base")
    assert df is not None and df.iloc[0]["segment"] == seg and df.iloc[0]["drift_flag"] == "OK"
    assert _read_backtest_consolidated(tmp_path / "nope", "_base") is None


def test_validation_governance_sheet(tmp_path):
    seg = _build_tree(tmp_path)
    wb = Workbook()
    _write_sheet_validation_governance(wb, tmp_path, {seg: {}}, pd.DataFrame(), "_base")
    assert "Validation & Governance" in wb.sheetnames
    text = "\n".join(
        str(c.value) for rowcells in wb["Validation & Governance"].iter_rows() for c in rowcells if c.value
    )
    assert "deadbeefcafe0000" in text  # data SHA pinned
    assert "multiplier" in text and "FIXED" in text  # assumptions + tier
    assert seg in text  # per-segment reproducibility/stability row


def test_oot_validation_sheet(tmp_path):
    seg = _build_tree(tmp_path)
    wb = Workbook()
    _write_sheet_oot_validation(wb, tmp_path, "_base")
    assert "Out-of-time Validation" in wb.sheetnames
    text = "\n".join(str(c.value) for rowcells in wb["Out-of-time Validation"].iter_rows() for c in rowcells if c.value)
    assert seg in text and "OK" in text and "noise-aware" in text


def test_oot_sheet_graceful_when_missing(tmp_path):
    (tmp_path / "empty").mkdir()
    wb = Workbook()
    _write_sheet_oot_validation(wb, tmp_path / "empty", "_base")
    text = "\n".join(str(c.value) for rowcells in wb["Out-of-time Validation"].iter_rows() for c in rowcells if c.value)
    assert "not available" in text and "run_backtest" in text


# ----------------- audit 2026-11-27 A11 / A12: no evidence is not "OK" -----------------


def test_segment_status_does_not_call_missing_evidence_ok():
    """No backtest row, an insufficient backtest ("—") or a failed one is not a pass."""
    stable = {"worst_status": "stable", "info": 3}
    for flag in (None, "—", "FAILED", float("nan")):
        assert _segment_status(flag, {})[0] == "UNVERIFIED"
        assert _segment_status(flag, stable)[0] == "UNVERIFIED"
    assert _segment_status("BELOW", stable)[0] == "OK"  # realized risk clearly lower: a pass


def test_segment_status_unstable_psi_is_review_even_without_alert_counts():
    """worst_status comes from stability_psi; the alert counts from drift_alerts, which may be absent."""
    assert _segment_status("OK", {"worst_status": "unstable"})[0] == "REVIEW"
    assert _segment_status(None, {"worst_status": "unstable"})[0] == "REVIEW"
    assert _segment_status(None, {"warning": 1})[0] == "WATCH"  # PSI evidence still counts without a backtest


def test_backtest_flags_are_matched_by_segment_filter():
    """The backtest writes ``settings.segment_filter`` ("direct/pl/new") in its ``segment`` column,
    while the report iterates segment NAMES ("direct-pl-new"). Looking the flag up by name found
    nothing on the live naming, so a DRIFT never reached the status column."""
    bt = pd.DataFrame({"segment": ["direct/pl/new", "legacy_name"], "drift_flag": ["DRIFT", "OK"]})
    segments = {"direct-pl-new": {"segment_filter": "direct/pl/new"}, "legacy_name": {}, "not-backtested": {}}

    flags = _backtest_flags_by_segment(bt, segments)

    assert flags == {"direct-pl-new": "DRIFT", "legacy_name": "OK"}
    assert _backtest_flags_by_segment(None, segments) == {}


def _recommendation_text(tmp_path, backtest_rows: list[dict] | None, segments: dict) -> str:
    if backtest_rows is not None:
        (tmp_path / "backtest").mkdir(exist_ok=True)
        pd.DataFrame(backtest_rows).to_csv(tmp_path / "backtest" / "backtest_consolidated_base.csv", index=False)
    wb = Workbook()
    no_rows = pd.DataFrame(columns=["group", "period", "scenario"])
    _write_exec_recommendation(wb.active, 1, no_rows, tmp_path, segments, "_base")
    return "\n".join(str(c.value) for rowcells in wb.active.iter_rows() for c in rowcells if c.value)


def test_recommendation_does_not_claim_noise_when_a_segment_drifts(tmp_path):
    rows = [
        {"segment": "a", "drift_flag": "DRIFT"},
        {"segment": "b", "drift_flag": "BELOW"},
        {"segment": "c", "drift_flag": "—"},
        {"segment": "d", "drift_flag": "FAILED"},
    ]
    text = _recommendation_text(tmp_path, rows, {name: {} for name in "abcd"})

    assert "within sampling noise" not in text
    assert "OUT-OF-TIME DRIFT" in text and " a" in text
    # every flag is counted: nothing disappears from "of 4"
    assert "0 OK, 1 below, 0 inconclusive, 1 drift, 1 insufficient, 1 failed (of 4 segments backtested)" in text
    assert "1 OK · 0 watch · 1 review · 2 unverified" in text  # a review, b OK, c and d unverified


def test_recommendation_counts_statuses_per_segment(tmp_path):
    rows = [{"segment": "a", "drift_flag": "OK"}, {"segment": "b", "drift_flag": "INCONCLUSIVE"}]
    text = _recommendation_text(tmp_path, rows, {"a": {}, "b": {}, "c": {}})

    assert "1 OK · 1 watch · 0 review · 1 unverified" in text
    assert "within sampling noise" in text  # no drift: the residual sentence stands


def test_recommendation_without_backtest_makes_no_out_of_time_claim(tmp_path):
    text = _recommendation_text(tmp_path, None, {"a": {}})

    assert "Out-of-time validation: not available" in text
    assert "within sampling noise" not in text
    assert "0 OK · 0 watch · 0 review · 1 unverified" in text


def test_oot_sheet_shows_failed_and_below_rows(tmp_path):
    (tmp_path / "backtest").mkdir()
    pd.DataFrame(
        [
            {"segment": "a", "drift_flag": "FAILED", "message": "backtest failed: no mature holdout"},
            {"segment": "b", "drift_flag": "BELOW"},
        ]
    ).to_csv(tmp_path / "backtest" / "backtest_consolidated_base.csv", index=False)
    wb = Workbook()
    _write_sheet_oot_validation(wb, tmp_path, "_base")
    ws = wb["Out-of-time Validation"]
    fills = {str(c.value): c.fill.start_color.rgb for rowcells in ws.iter_rows() for c in rowcells if c.value}
    assert fills["FAILED"] not in (None, "00000000") and fills["BELOW"] not in (None, "00000000")
    assert "backtest failed: no mature holdout" in fills  # the reason is on the sheet


def test_kpi_rate_formatter_treats_nan_as_missing():
    """A None rate becomes NaN once another row of the column holds a float; ``is not None`` let it
    through and the KPI card read "nan%"."""
    assert _fmt_pct_or_na(12.345) == "12.3%"
    assert _fmt_pct_or_na(0.0) == "0.0%"
    assert _fmt_pct_or_na(None) == "n/a"
    assert _fmt_pct_or_na(float("nan")) == "n/a"
