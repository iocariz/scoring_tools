"""Audit F3: the exposure (todu) model must be the one paired with the selected risk model.

Risk models are versioned per training run; the exposure model used to be saved
unversioned at the models root and overwritten by every run — selecting an older
model_<ts> silently paired it with the NEWEST exposure model.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import BinConfig, OutputPaths, PreprocessingSettings
from src.persistence import save_model_with_metadata
from src.pipeline.inference import run_inference_phase


def _settings():
    return PreprocessingSettings(
        variables=["a"],
        inference_variables=["a"],
        segment_filter="pairing_segment",
        keep_vars=["status_name", "reject_reason", "mis_date", "old_score"],
        date_ini_book_obs="2023-01-01",
        date_fin_book_obs="2023-12-01",
        bins={"a": BinConfig(source_col="old_score", output_col="a", bin_edges=[-np.inf, 50, np.inf])},
        indicators=["oa_amt", "oa_amt_h0", "todu_30ever_h6", "todu_amt_pile_h6"],
    )


@pytest.fixture
def model_root(tmp_path):
    old_dir = tmp_path / "model_output/models/model_20240101_000000"
    new_dir = tmp_path / "model_output/models/model_20240202_000000"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    (old_dir.parent / "todu_model.joblib").touch()  # legacy shared root copy
    return tmp_path, old_dir, new_dir


def _run(model_dir, tmp_path):
    metadata = {"multiplier": 7, "model_variables": ["a"], "bin_edges": {"a": [-np.inf, 50, np.inf]}}
    with (
        patch("src.pipeline.inference.load_model_for_prediction", return_value=(object(), metadata, ["a"])),
        patch("src.pipeline.inference.safe_joblib_load", return_value=object()) as loader,
    ):
        run_inference_phase(
            pd.DataFrame(), _settings(), str(model_dir), OutputPaths(base_dir=tmp_path / "model_output")
        )
    return loader


def test_older_dir_with_only_root_companion_is_refused(model_root):
    tmp_path, old_dir, _ = model_root
    with pytest.raises(RuntimeError, match="pairing cannot be verified"):
        _run(old_dir, tmp_path)


def test_newest_dir_may_use_root_companion(model_root):
    tmp_path, old_dir, new_dir = model_root
    loader = _run(new_dir, tmp_path)
    assert loader.call_args.args[0] == old_dir.parent / "todu_model.joblib"


def test_in_dir_companion_is_the_verified_pair(model_root):
    tmp_path, old_dir, _ = model_root
    (old_dir / "todu_model.joblib").touch()
    loader = _run(old_dir, tmp_path)
    assert loader.call_args.args[0] == old_dir / "todu_model.joblib"  # not the newer root copy


def test_two_training_saves_reuse_original_exposure_pair(tmp_path, monkeypatch):
    """F7: exercise the actual directory return contract and trusted save/load roundtrip."""
    from pathlib import Path

    import joblib
    from sklearn.linear_model import LinearRegression

    from src.persistence import write_integrity_sidecar

    output = OutputPaths(base_dir=tmp_path / "run")
    monkeypatch.setenv("SCORING_TRUSTED_MODEL_ROOTS", str(tmp_path))
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    risk_model = LinearRegression().fit(X, [1.0, 2.0, 3.0])
    exposure_models = [LinearRegression().fit(X, [v, 2 * v, 3 * v]) for v in (100.0, 900.0)]
    saved_dirs = []

    def train_risk(**kwargs):
        metadata = {
            "multiplier": kwargs["multiplier"],
            "model_variables": kwargs["variables"],
            "bin_edges": dict(zip(kwargs["variables"], kwargs["bins"], strict=True)),
            "bin_sources": kwargs["bin_sources"],
        }
        directory = save_model_with_metadata(risk_model, ["a"], metadata, kwargs["model_base_path"])
        saved_dirs.append(Path(directory))
        return {
            "model_path": directory,
            "features": ["a"],
            "best_model_info": {"model": risk_model, "name": "LinearRegression"},
        }

    def train_exposure(**kwargs):
        model = exposure_models[len(saved_dirs) - 1]
        joblib.dump(model, kwargs["model_output_path"])
        write_integrity_sidecar(kwargs["model_output_path"])
        return None, model, None

    monkeypatch.setattr("src.pipeline.inference.inference_pipeline", train_risk)
    monkeypatch.setattr("src.pipeline.inference.todu_average_inference", train_exposure)
    for _ in range(2):
        run_inference_phase(pd.DataFrame(), _settings(), output=output)
    assert saved_dirs[0] != saved_dirs[1]
    for directory in saved_dirs:
        assert (directory / "todu_model.joblib.sha256").is_file()
    _, old_exposure = run_inference_phase(pd.DataFrame(), _settings(), str(saved_dirs[0]), output)
    np.testing.assert_allclose(old_exposure.predict(X), [100.0, 200.0, 300.0])
    np.testing.assert_allclose(joblib.load(output.todu_model_joblib).predict(X), [900.0, 1800.0, 2700.0])
