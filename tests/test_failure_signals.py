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

    def test_failed_backtest_replaces_previous_evidence_with_a_failed_row(self, monkeypatch, tmp_path):
        """A3 (rest): a segment whose backtest fails used to vanish from the consolidated file. It
        now has a row that says so, and the previous run's verdict for it is gone."""
        import pandas as pd

        csv_path, _ = _stale_backtest(tmp_path / "backtest")
        _frozen_segment(tmp_path / "output", "seg_a")
        monkeypatch.setattr(
            backtest_runner.PreprocessingSettings,
            "from_toml",
            classmethod(lambda cls, path: SimpleNamespace(mr_maturity_months=6, segment_filter="direct/seg/a")),
        )

        def fail(*args, **kwargs):
            raise BacktestError("no mature\n  holdout")  # multi-line reasons are flattened

        monkeypatch.setattr(backtest_runner, "backtest_segment", fail)

        results = backtest_runner.backtest_all(None, data_dir=tmp_path / "output", out_dir=tmp_path / "backtest")

        assert results == []
        rows = pd.read_csv(csv_path)
        assert rows["segment"].tolist() == ["direct/seg/a"]
        assert rows["drift_flag"].tolist() == ["FAILED"]
        assert "no mature holdout" in rows["message"].iloc[0]
        assert "FAILED" in (tmp_path / "backtest" / "backtest_summary_base.md").read_text(encoding="utf-8")

    def test_unreadable_segment_config_is_reported_under_the_directory_name(self, monkeypatch, tmp_path):
        import pandas as pd

        _frozen_segment(tmp_path / "output", "seg_a")

        def broken(cls, path):
            raise ValueError("bad toml")

        monkeypatch.setattr(backtest_runner.PreprocessingSettings, "from_toml", classmethod(broken))

        backtest_runner.backtest_all(None, data_dir=tmp_path / "output", out_dir=tmp_path / "backtest")

        rows = pd.read_csv(tmp_path / "backtest" / "backtest_consolidated_base.csv")
        assert rows["segment"].tolist() == ["seg_a"] and rows["drift_flag"].tolist() == ["FAILED"]


# --------------------------------------------------------------------------- #
# Audit 2026-11-27 A7: standalone governance CLIs
# --------------------------------------------------------------------------- #


def test_backtest_all_reports_failed_segments_to_the_caller(monkeypatch, tmp_path):
    _frozen_segment(tmp_path / "output", "seg_a")
    monkeypatch.setattr(
        backtest_runner.PreprocessingSettings,
        "from_toml",
        classmethod(lambda cls, path: SimpleNamespace(mr_maturity_months=6, segment_filter="seg/a")),
    )

    def fail(*args, **kwargs):
        raise BacktestError("no mature holdout")

    monkeypatch.setattr(backtest_runner, "backtest_segment", fail)
    failures: list[tuple[str, str]] = []

    backtest_runner.backtest_all(None, data_dir=tmp_path / "output", out_dir=tmp_path / "backtest", failures=failures)

    assert failures == [("seg/a", "no mature holdout")]


def test_run_backtest_exits_nonzero_when_a_segment_fails(monkeypatch, tmp_path):
    """Some segments succeeding used to be enough for exit code 0."""
    import run_backtest
    import src.data.data_manager as data_manager
    import src.pipeline.batch_config as batch_config

    monkeypatch.setattr(batch_config, "load_base_config", lambda path: {})
    monkeypatch.setattr(data_manager, "load_configured_data", lambda cfg: pd.DataFrame({"a": [1]}))

    def partly_failing(*args, failures=None, **kwargs):
        failures.append(("seg/b", "boom"))
        return ["a result"]

    monkeypatch.setattr(run_backtest, "backtest_all", partly_failing)

    assert run_backtest.main(["--data-dir", str(tmp_path), "--output", str(tmp_path / "bt")]) == 1


@pytest.mark.parametrize("segment", ["..", ".", "a/../..", "../elsewhere"])
def test_reproducibility_refuses_a_segment_whose_run_dir_escapes_the_output_dir(tmp_path, segment):
    """`-s ..` reached shutil.rmtree(output/..) — the whole output tree — before anything else."""
    import run_reproducibility

    victim = tmp_path / "output" / "keep_me.txt"
    victim.parent.mkdir(parents=True)
    victim.write_text("x", encoding="utf-8")

    assert run_reproducibility.main(["-s", segment, "--output", str(tmp_path / "output" / "_repro")]) == 1
    assert victim.exists()


def test_reproducibility_still_accepts_a_literal_filter(monkeypatch, tmp_path):
    """`-s direct/pl/new` is the documented fallback when the value is not a segments.toml name;
    its run directory nests under the output dir, so it is cleaned and used as before."""
    import run_reproducibility

    stale = tmp_path / "_repro" / "direct" / "pl" / "new" / "old_artifact.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("x", encoding="utf-8")

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(run_reproducibility, "_resolve_segment_filter", stop)

    with pytest.raises(Stop):
        run_reproducibility.main(["-s", "direct/pl/new", "--output", str(tmp_path / "_repro")])
    assert not stale.exists() and stale.parent.is_dir()  # cleaned, recreated, and the run went on


def test_baseline_flag_refuses_a_segment_with_fixed_cutoffs(monkeypatch, tmp_path):
    """The config validator rejects baseline_mode + fixed_cutoffs; --baseline is applied after
    validation and silently won."""
    settings = _settings(fixed_cutoffs={"sc_octroi_new_clus": [1], "new_efx_clus": [1]})
    monkeypatch.setattr(runner, "load_and_validate_config", lambda p: (settings, None, None, 1.0))

    loaded = []
    monkeypatch.setattr(runner, "load_and_prepare_data", lambda s, preloaded: loaded.append(1) or (None, s))

    assert runner.run_pipeline("cfg.toml", baseline_mode=True, output=OutputPaths(base_dir=tmp_path)) is False
    assert not loaded  # refused before any data is loaded


@pytest.mark.parametrize("name", [".", "..", "...", "a/b", "", "a b"])
def test_segment_names_that_are_not_a_plain_directory_are_rejected(name):
    from src.pipeline.batch_config import validate_segment_name

    with pytest.raises(ValueError, match="Invalid segment name"):
        validate_segment_name(name)


@pytest.mark.parametrize("name", ["direct-pl-new", "seg_a", "v1.2", ".hidden"])
def test_plain_segment_names_are_accepted(name):
    from src.pipeline.batch_config import validate_segment_name

    validate_segment_name(name)
