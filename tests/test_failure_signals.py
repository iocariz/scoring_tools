"""Audit 2026-11-27 (PR 1): a failed step must never be reported as a success.

``run_pipeline`` returns the ``False`` sentinel on failure (#61). Batch only tested ``is None``, so a
failed segment was logged as "completed successfully", its previous-run artifacts were consolidated as
current, and a failed supersegment training handed an OLD model to its members. A DQ halt returned
``None``, which the single-segment CLI reads as (resimulation) success. And a skipped/failed backtest
left the previous run's consolidated out-of-time evidence in place for the workbook to read.
"""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import src.governance.backtest_runner as backtest_runner
import src.pipeline.batch as batch
import src.pipeline.segment_runner as runner
from src.core.config import OutputPaths, PreprocessingSettings
from src.governance.backtest import BacktestError


def _settings(**overrides) -> PreprocessingSettings:
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


@pytest.fixture
def stub_segment_config(monkeypatch, tmp_path):
    """Skip the TOML write: the pipeline itself is mocked in these tests."""
    monkeypatch.setattr(batch, "write_segment_config", lambda cfg, root: tmp_path / "config_segment.toml")


# --------------------------------------------------------------------------- #
# Batch: the False sentinel is a failure
# --------------------------------------------------------------------------- #


class TestSegmentFailureSentinel:
    @pytest.mark.parametrize("sentinel", [False, None])
    def test_failed_pipeline_is_not_a_success(self, monkeypatch, tmp_path, stub_segment_config, sentinel):
        monkeypatch.setattr(batch, "run_pipeline", lambda **kwargs: sentinel)
        assert batch.run_segment_pipeline("seg_a", {}, {}, str(tmp_path)) is False

    def test_result_tuple_is_a_success(self, monkeypatch, tmp_path, stub_segment_config):
        frame = pd.DataFrame({"a": [1]})
        monkeypatch.setattr(batch, "run_pipeline", lambda **kwargs: (frame, frame, frame, frame, frame))
        assert batch.run_segment_pipeline("seg_a", {}, {}, str(tmp_path)) is True


class TestSupersegmentTrainingFailure:
    SS_CONFIG = {"segment_filters": ["seg_a", "seg_b"]}

    def _stale_model(self, tmp_path: Path) -> Path:
        stale = tmp_path / "_supersegment_ss" / "models" / "model_20250101_000000"
        stale.mkdir(parents=True)
        return stale

    def test_failed_training_does_not_return_a_previous_model(self, monkeypatch, tmp_path, stub_segment_config):
        self._stale_model(tmp_path)
        monkeypatch.setattr(batch, "run_pipeline", lambda **kwargs: False)
        assert batch.run_supersegment_training("ss", self.SS_CONFIG, {}, str(tmp_path)) is None

    def test_training_that_writes_no_new_model_fails(self, monkeypatch, tmp_path, stub_segment_config):
        """A "successful" return with only an older model on disk must not pass that model on."""
        self._stale_model(tmp_path)
        monkeypatch.setattr(batch, "run_pipeline", lambda **kwargs: (None,))
        assert batch.run_supersegment_training("ss", self.SS_CONFIG, {}, str(tmp_path)) is None

    def test_returns_the_model_written_by_this_run(self, monkeypatch, tmp_path, stub_segment_config):
        self._stale_model(tmp_path)
        # Sorts BEFORE the stale dir on purpose: the newest-by-name glob would pick the stale one.
        fresh = tmp_path / "_supersegment_ss" / "models" / "model_20240101_000000"

        def train(**kwargs):
            fresh.mkdir(parents=True)
            return (None,)

        monkeypatch.setattr(batch, "run_pipeline", train)
        assert batch.run_supersegment_training("ss", self.SS_CONFIG, {}, str(tmp_path)) == str(fresh.resolve())


# --------------------------------------------------------------------------- #
# Single segment: a DQ halt is a failure, not a None "success"
# --------------------------------------------------------------------------- #


def test_dq_halt_returns_the_failure_sentinel(monkeypatch, tmp_path):
    settings = _settings()
    monkeypatch.setattr(runner, "load_and_validate_config", lambda p: (settings, None, None, 1.0))
    monkeypatch.setattr(runner, "load_and_prepare_data", lambda s, preloaded: (pd.DataFrame({"a": [1]}), s))
    monkeypatch.setattr(runner, "run_preprocessing_phase", lambda *args, **kwargs: None)

    def must_not_run(*args, **kwargs):
        raise AssertionError("inference must not run after a DQ halt")

    monkeypatch.setattr(runner, "run_inference_phase", must_not_run)

    result = runner.run_pipeline("cfg.toml", output=OutputPaths(base_dir=tmp_path))

    assert result is False


def test_dq_halt_is_a_pipeline_execution_error():
    assert issubclass(runner.DataQualityHaltError, runner.PipelineExecutionError)


# --------------------------------------------------------------------------- #
# Backtest: no stale out-of-time evidence
# --------------------------------------------------------------------------- #


def _stale_backtest(out_dir: Path, suffix: str = "_base") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"backtest_consolidated{suffix}.csv"
    md_path = out_dir / f"backtest_summary{suffix}.md"
    csv_path.write_text("segment,flag\nseg_a,OK\n", encoding="utf-8")
    md_path.write_text("# previous run\n", encoding="utf-8")
    return csv_path, md_path


def _frozen_segment(data_dir: Path, name: str) -> Path:
    seg_dir = data_dir / name
    (seg_dir / "data").mkdir(parents=True)
    (seg_dir / "config_segment.toml").write_text("", encoding="utf-8")
    (seg_dir / "data" / "accepted_cells_base.csv").write_text("a\n1\n", encoding="utf-8")
    return seg_dir


class TestBacktestStaleEvidence:
    def test_invalidate_removes_consolidated_outputs(self, tmp_path):
        csv_path, md_path = _stale_backtest(tmp_path / "backtest")
        other_scenario, _ = _stale_backtest(tmp_path / "backtest", suffix="_pessimistic")

        removed = backtest_runner.invalidate_consolidated_backtest(tmp_path / "backtest", "base")

        assert not csv_path.exists() and not md_path.exists()
        assert sorted(removed) == sorted([csv_path, md_path])
        assert other_scenario.exists()

    def test_invalidate_is_a_noop_without_previous_outputs(self, tmp_path):
        assert backtest_runner.invalidate_consolidated_backtest(tmp_path / "missing", "base") == []

    def test_no_frozen_segments_drops_previous_evidence(self, tmp_path):
        csv_path, _ = _stale_backtest(tmp_path / "backtest")
        (tmp_path / "output").mkdir()

        results = backtest_runner.backtest_all(None, data_dir=tmp_path / "output", out_dir=tmp_path / "backtest")

        assert results == []
        assert not csv_path.exists()

    @staticmethod
    def _batch_args(tmp_path: Path, **overrides) -> SimpleNamespace:
        values = {
            "output": str(tmp_path),
            "no_backtest": True,
            "training_only": False,
            "no_consolidation": True,
            "no_report": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_batch_no_backtest_drops_evidence_for_previous_cutoffs(self, tmp_path):
        import run_batch

        csv_path, _ = _stale_backtest(tmp_path / "backtest")

        run_batch._post_run_outputs(self._batch_args(tmp_path), {}, {}, {}, None, {"seg_a": True})

        assert not csv_path.exists()

    @pytest.mark.parametrize(
        ("overrides", "results"),
        [
            ({"training_only": True}, {"seg_a": True}),  # no cutoffs were derived
            ({}, {"seg_a": False}),  # nothing on disk changed
        ],
    )
    def test_batch_keeps_evidence_when_cutoffs_did_not_change(self, tmp_path, overrides, results):
        import run_batch

        csv_path, _ = _stale_backtest(tmp_path / "backtest")

        run_batch._post_run_outputs(self._batch_args(tmp_path, **overrides), {}, {}, {}, None, results)

        assert csv_path.exists()

    def test_every_segment_failing_drops_previous_evidence(self, monkeypatch, tmp_path):
        csv_path, md_path = _stale_backtest(tmp_path / "backtest")
        _frozen_segment(tmp_path / "output", "seg_a")
        monkeypatch.setattr(
            backtest_runner.PreprocessingSettings,
            "from_toml",
            classmethod(lambda cls, path: SimpleNamespace(mr_maturity_months=6)),
        )

        def fail(*args, **kwargs):
            raise BacktestError("no mature holdout")

        monkeypatch.setattr(backtest_runner, "backtest_segment", fail)

        results = backtest_runner.backtest_all(None, data_dir=tmp_path / "output", out_dir=tmp_path / "backtest")

        assert results == []
        assert not csv_path.exists() and not md_path.exists()
