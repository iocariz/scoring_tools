"""Unit tests for src/dashboard_data.py (R2 todo #64).

Covers the shared data-loading helpers that dashboard.py and
gradio_dashboard.py both import. Each helper is exercised against a
synthetic output tree so behaviour is pinned regardless of which
dashboard runs it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.dashboard_data import (
    SCENARIO_ORDER,
    get_available_segments,
    get_data_dir,
    get_images_dir,
    get_models_dir,
    get_scenario_paths,
    get_scenarios,
    parse_coefficients,
    resolve_supersegment,
)


@pytest.fixture
def synthetic_output(tmp_path):
    """Build a minimal output/ tree with two segments and one supersegment."""
    base = tmp_path / "output"
    for seg in ["seg_a", "seg_b"]:
        (base / seg / "data").mkdir(parents=True)
        (base / seg / "images").mkdir(parents=True)
        (base / seg / "models").mkdir(parents=True)
    # Supersegment models dir — no per-segment models locally.
    (base / "_supersegment_shared" / "models").mkdir(parents=True)
    (base / "_supersegment_shared" / "models" / "model_20260101").mkdir()
    # Clean segment that only has data+images (no models locally)
    (base / "seg_clean" / "data").mkdir(parents=True)
    (base / "seg_clean" / "images").mkdir(parents=True)
    # Hidden/private directories (should be ignored by get_available_segments)
    (base / "_private" / "data").mkdir(parents=True)
    (base / ".cache" / "data").mkdir(parents=True)
    return base


class TestGetDataDir:
    def test_segment_mode(self, synthetic_output):
        assert get_data_dir(synthetic_output, "seg_a") == synthetic_output / "seg_a" / "data"

    def test_single_run_mode_returns_root_data(self, synthetic_output):
        (synthetic_output / "data").mkdir()
        assert get_data_dir(synthetic_output) == synthetic_output / "data"

    def test_fallback_to_cwd_relative_when_no_data(self, synthetic_output):
        # No `output/data/`; returns the relative Path("data").
        assert get_data_dir(synthetic_output) == __import__("pathlib").Path("data")


class TestGetImagesDir:
    def test_segment_mode(self, synthetic_output):
        assert get_images_dir(synthetic_output, "seg_a") == synthetic_output / "seg_a" / "images"

    def test_single_run_mode(self, synthetic_output):
        (synthetic_output / "images").mkdir()
        assert get_images_dir(synthetic_output) == synthetic_output / "images"


class TestGetModelsDir:
    def test_uses_local_models_when_present_and_nonempty(self, synthetic_output):
        (synthetic_output / "seg_a" / "models" / "model_x").mkdir()
        got = get_models_dir(synthetic_output, "seg_a")
        assert got == synthetic_output / "seg_a" / "models"

    def test_falls_back_to_supersegment_when_local_empty(self, synthetic_output, tmp_path):
        # Wire a segments.toml that maps seg_clean → _supersegment_shared.
        # Must be in CWD because resolve_supersegment looks for ./segments.toml
        # first. Tests use monkeypatch.chdir() to fix this.
        (synthetic_output / "segments.toml").write_text('[segments.seg_clean]\nmodelling_supersegment = "shared"\n')
        got = get_models_dir(synthetic_output, "seg_clean")
        assert got == synthetic_output / "_supersegment_shared" / "models"

    def test_returns_local_even_if_empty_when_no_supersegment(self, synthetic_output):
        # seg_a has an empty models dir and no supersegment mapping.
        got = get_models_dir(synthetic_output, "seg_a")
        assert got == synthetic_output / "seg_a" / "models"


class TestGetAvailableSegments:
    def test_excludes_private_and_hidden(self, synthetic_output):
        got = get_available_segments(synthetic_output)
        assert set(got) == {"seg_a", "seg_b", "seg_clean"}

    def test_requires_data_subdir(self, tmp_path):
        base = tmp_path / "output"
        (base / "seg_with_data" / "data").mkdir(parents=True)
        (base / "seg_no_data").mkdir(parents=True)  # no `data/` subdir
        assert get_available_segments(base) == ["seg_with_data"]

    def test_nonexistent_base_returns_empty(self, tmp_path):
        assert get_available_segments(tmp_path / "does_not_exist") == []


class TestGetScenarios:
    def test_canonical_scenario_discovery(self, synthetic_output):
        data_dir = synthetic_output / "seg_a" / "data"
        (data_dir / "risk_production_summary_table_pessimistic.csv").touch()
        (data_dir / "risk_production_summary_table_base.csv").touch()
        (data_dir / "risk_production_summary_table_optimistic.csv").touch()
        got = get_scenarios(synthetic_output, "seg_a")
        assert got == ["pessimistic", "base", "optimistic"]

    def test_legacy_numeric_scenarios(self, synthetic_output):
        data_dir = synthetic_output / "seg_a" / "data"
        (data_dir / "risk_production_summary_table_1.2.csv").touch()
        (data_dir / "risk_production_summary_table_2.5.csv").touch()
        got = get_scenarios(synthetic_output, "seg_a")
        assert got == ["1.2", "2.5"]

    def test_mr_files_excluded(self, synthetic_output):
        data_dir = synthetic_output / "seg_a" / "data"
        (data_dir / "risk_production_summary_table_base.csv").touch()
        (data_dir / "risk_production_summary_table_mr_base.csv").touch()
        got = get_scenarios(synthetic_output, "seg_a")
        assert got == ["base"]

    def test_unsuffixed_base_fallback(self, synthetic_output):
        data_dir = synthetic_output / "seg_a" / "data"
        (data_dir / "risk_production_summary_table.csv").touch()
        got = get_scenarios(synthetic_output, "seg_a")
        assert got == ["base"]

    def test_scenario_order_constant(self):
        # Guard the ordering contract so downstream sort doesn't drift.
        assert SCENARIO_ORDER == {"pessimistic": 0, "base": 1, "optimistic": 2}


class TestGetScenarioPaths:
    def test_all_four_artifacts_returned(self, synthetic_output):
        paths = get_scenario_paths(synthetic_output, "base", "seg_a")
        assert set(paths.keys()) == {"main_csv", "mr_csv", "cutoffs", "summary_data"}
        assert paths["main_csv"].name == "risk_production_summary_table_base.csv"

    def test_base_scenario_fallback_to_unsuffixed(self, synthetic_output):
        data_dir = synthetic_output / "seg_a" / "data"
        (data_dir / "risk_production_summary_table.csv").touch()
        paths = get_scenario_paths(synthetic_output, "base", "seg_a")
        # Suffixed file doesn't exist, unsuffixed does → fallback wins.
        assert paths["main_csv"] == data_dir / "risk_production_summary_table.csv"

    def test_no_static_url_keys(self, synthetic_output):
        """Static URLs (/static/...) are dashboard-framework-specific; this
        module stays framework-agnostic and must not emit them."""
        paths = get_scenario_paths(synthetic_output, "base", "seg_a")
        for value in paths.values():
            assert not str(value).startswith("/static")


class TestResolveSupersegment:
    def test_modelling_supersegment_preferred(self, synthetic_output, monkeypatch):
        (synthetic_output / "segments.toml").write_text(
            '[segments.seg_a]\nmodelling_supersegment = "ms"\nsupersegment = "legacy"\n'
        )
        monkeypatch.chdir(synthetic_output)
        assert resolve_supersegment("seg_a") == "ms"

    def test_falls_back_to_plain_supersegment(self, synthetic_output, monkeypatch):
        (synthetic_output / "segments.toml").write_text('[segments.seg_a]\nsupersegment = "legacy"\n')
        monkeypatch.chdir(synthetic_output)
        assert resolve_supersegment("seg_a") == "legacy"

    def test_unknown_segment(self, synthetic_output, monkeypatch):
        (synthetic_output / "segments.toml").write_text('[segments.other]\nsupersegment = "s"\n')
        monkeypatch.chdir(synthetic_output)
        assert resolve_supersegment("seg_a") is None

    def test_output_base_fallback(self, synthetic_output, tmp_path, monkeypatch):
        """If the cwd has no segments.toml but output_base does, use output_base's."""
        # Move to an empty directory.
        empty = tmp_path / "elsewhere"
        empty.mkdir()
        monkeypatch.chdir(empty)
        (synthetic_output / "segments.toml").write_text(
            '[segments.seg_a]\nmodelling_supersegment = "via_output_base"\n'
        )
        assert resolve_supersegment("seg_a", output_base=synthetic_output) == "via_output_base"

    def test_malformed_toml_returns_none(self, synthetic_output, monkeypatch):
        (synthetic_output / "segments.toml").write_text("this = = = not valid toml")
        monkeypatch.chdir(synthetic_output)
        assert resolve_supersegment("seg_a") is None

    def test_empty_segment_name(self, synthetic_output, monkeypatch):
        monkeypatch.chdir(synthetic_output)
        assert resolve_supersegment(None) is None
        assert resolve_supersegment("") is None


class TestParseCoefficients:
    def test_extracts_features_and_values(self, tmp_path):
        model_dir = tmp_path / "model_20260101"
        model_dir.mkdir()
        (model_dir / "model_summary.txt").write_text(
            "HEADER\n"
            "---\n"
            "Intercept: 0.42\n"
            "\n"
            "MODEL COEFFICIENTS:\n"
            "------------------\n"
            "feat_a: 1.234\n"
            "feat_b: -0.56\n"
            "feat_c: 0.0\n"
            "\n"
            "NEXT SECTION\n"
            "other_thing: 99\n"
        )
        got = parse_coefficients(model_dir)
        assert got == [
            {"feature": "feat_a", "coefficient": 1.234},
            {"feature": "feat_b", "coefficient": -0.56},
            {"feature": "feat_c", "coefficient": 0.0},
        ]
        # "NEXT SECTION" all-caps header terminates parsing.
        assert all(c["feature"] != "other_thing" for c in got)

    def test_missing_file_returns_empty(self, tmp_path):
        assert parse_coefficients(tmp_path / "does_not_exist") == []

    def test_no_coef_section(self, tmp_path):
        (tmp_path / "model_summary.txt").write_text("HEADER\nSOME OTHER CONTENT\nno colons\n")
        assert parse_coefficients(tmp_path) == []

    def test_non_numeric_coefficient_skipped(self, tmp_path):
        (tmp_path / "model_summary.txt").write_text("MODEL COEFFICIENTS:\n---\nfeat_a: not_a_number\nfeat_b: 3.14\n")
        got = parse_coefficients(tmp_path)
        assert got == [{"feature": "feat_b", "coefficient": 3.14}]
