"""Tests for per-run data lineage capture (M2) and the --allow-dq-warnings wiring."""

import hashlib
import inspect
import json
import sys
from pathlib import Path

from src import lineage
from src.config import OutputPaths, PreprocessingSettings


def _make_settings(**overrides):
    values = {
        "keep_vars": ["mis_date"],
        "indicators": ["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
        "segment_filter": "seg_a",
        "date_ini_book_obs": "2024-01-01",
        "date_fin_book_obs": "2024-12-31",
        "variables": ["sc_octroi_new_clus", "new_efx_clus"],
        "octroi_bins": [0.0, 1.0, 2.0],
        "efx_bins": [0.0, 1.0, 2.0],
    }
    values.update(overrides)
    return PreprocessingSettings(**values)


# ----------------------------- file_provenance -----------------------------


def test_file_provenance_matches_manual_sha(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world\n" * 100)
    prov = lineage.file_provenance(str(f))
    assert prov["exists"] is True
    assert prov["size_bytes"] == f.stat().st_size
    assert prov["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest()
    assert "mtime" in prov


def test_file_provenance_memoized_returns_same_object(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"abc")
    first = lineage.file_provenance(str(f))
    second = lineage.file_provenance(str(f))
    assert first is second  # cached on (path, size, mtime) — hashed once


def test_file_provenance_missing_or_empty_no_raise():
    assert lineage.file_provenance("/no/such/file_xyz.sas7bdat")["exists"] is False
    assert lineage.file_provenance(None)["exists"] is False
    assert lineage.file_provenance("")["exists"] is False


# ----------------------------- git_provenance ------------------------------


def test_git_provenance_never_raises(monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(lineage.subprocess, "run", boom)
    assert lineage.git_provenance() == {"commit": None, "short": None, "dirty": None}


def test_git_provenance_keys():
    # In or out of a repo, the shape is stable.
    assert set(lineage.git_provenance()) == {"commit", "short", "dirty"}


# ------------------------------ build_lineage ------------------------------


def test_build_lineage_has_expected_keys():
    settings = _make_settings()
    lin = lineage.build_lineage(settings, config_path="config.toml", n_rows=123)
    assert set(lin) >= {
        "run_id",
        "run_timestamp",
        "segment",
        "date_range",
        "data",
        "git",
        "config",
        "assumptions",
    }
    assert lin["data"]["rows_loaded"] == 123
    assert lin["config"]["path"] == "config.toml"
    assert lin["assumptions"]["multiplier"] == settings.multiplier
    assert lin["segment"] == "seg_a"
    assert lin["run_id"]  # generated when not supplied


def test_build_lineage_uses_provided_run_context():
    settings = _make_settings()
    lin = lineage.build_lineage(
        settings,
        "config.toml",
        1,
        run_id="BATCH123",
        run_ts_iso="2026-01-01T00:00:00+00:00",
    )
    assert lin["run_id"] == "BATCH123"
    assert lin["run_timestamp"] == "2026-01-01T00:00:00+00:00"


# ------------------------------ write_lineage ------------------------------


def test_write_lineage_round_trips(tmp_path):
    settings = _make_settings()
    output = OutputPaths(base_dir=tmp_path)
    lin = lineage.build_lineage(settings, "config.toml", 5, run_id="RID")
    path = lineage.write_lineage(output, lin)
    assert Path(path) == Path(output.run_lineage_json)
    loaded = json.loads(Path(path).read_text())
    assert loaded["run_id"] == "RID"
    assert loaded["data"]["rows_loaded"] == 5


def test_log_run_banner_handles_minimal_dict():
    # Must not raise even on a sparse lineage dict.
    lineage.log_run_banner({})
    lineage.log_run_banner({"run_id": "x", "data": {}, "git": {}, "config": {}, "assumptions": {}})


# ------------------------- CLI / threading wiring --------------------------


def test_main_registers_allow_dq_warnings(monkeypatch):
    from main import parse_args

    monkeypatch.setattr(sys, "argv", ["main.py", "--allow-dq-warnings"])
    assert parse_args().allow_dq_warnings is True

    monkeypatch.setattr(sys, "argv", ["main.py"])
    assert parse_args().allow_dq_warnings is False


def test_run_batch_threads_allow_dq_warnings_and_run_id():
    import run_batch

    for fn in (
        run_batch.run_segment_pipeline,
        run_batch.run_supersegment_training,
        run_batch.run_segments_sequential,
        run_batch.run_segments_parallel,
    ):
        params = set(inspect.signature(fn).parameters)
        assert {"allow_dq_warnings", "run_id", "run_ts_iso"} <= params, fn.__name__


def test_main_threads_run_context():
    import main

    params = set(inspect.signature(main.main).parameters)
    assert {"allow_dq_warnings", "run_id", "run_ts_iso"} <= params
