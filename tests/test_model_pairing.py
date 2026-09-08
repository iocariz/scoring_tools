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
