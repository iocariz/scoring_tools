"""Regression tests for todo #66: deprecation of legacy ``octroi_bins`` /
``efx_bins`` config fields in favour of ``[preprocessing.bins.*]``.
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.config import BinConfig, PreprocessingSettings

_BASE_KWARGS: dict = {
    "variables": ["sc_octroi_new_clus", "new_efx_clus"],
    "keep_vars": ["sc_octroi_new_clus", "new_efx_clus"],
    "indicators": ["sc_octroi_new_clus", "new_efx_clus"],
    "date_ini_book_obs": "2024-01-01",
    "date_fin_book_obs": "2024-12-31",
}

_LEGACY_EDGES = [float("-inf"), 100.0, float("inf")]


class TestLegacyBinsDeprecation:
    def test_legacy_octroi_efx_bins_emit_deprecation(self):
        """Legacy-only config (no ``bins`` dict) must emit DeprecationWarning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PreprocessingSettings(
                **_BASE_KWARGS,
                octroi_bins=_LEGACY_EDGES,
                efx_bins=_LEGACY_EDGES,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1, f"expected 1 DeprecationWarning, got {len(dep)}"
        msg = str(dep[0].message)
        assert "octroi_bins" in msg
        assert "efx_bins" in msg
        assert "scripts/migrate_legacy_bins.py" in msg
        assert "sc_octroi_new_clus" in msg
        assert "new_efx_clus" in msg

    def test_modern_bins_dict_emits_no_deprecation(self):
        """Explicit [preprocessing.bins.*] config must not trigger the warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            PreprocessingSettings(
                **_BASE_KWARGS,
                bins={
                    "sc_octroi_new_clus": BinConfig(
                        source_col="score_rf",
                        output_col="sc_octroi_new_clus",
                        bin_edges=_LEGACY_EDGES,
                    ),
                    "new_efx_clus": BinConfig(
                        source_col="risk_score_rf",
                        output_col="new_efx_clus",
                        bin_edges=_LEGACY_EDGES,
                    ),
                },
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep == [], f"modern config should not emit DeprecationWarning, got: {[str(w.message) for w in dep]}"

    def test_legacy_config_still_produces_valid_settings(self):
        """Back-compat: legacy config must still yield a working bins dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            settings = PreprocessingSettings(
                **_BASE_KWARGS,
                octroi_bins=_LEGACY_EDGES,
                efx_bins=[float("-inf"), 200.0, float("inf")],
            )
        # Both variables should have been populated via the legacy fallback.
        assert "sc_octroi_new_clus" in settings.bins
        assert "new_efx_clus" in settings.bins
        assert settings.bins["sc_octroi_new_clus"].source_col == "score_rf"
        assert settings.bins["new_efx_clus"].source_col == "risk_score_rf"

    def test_mixed_config_prefers_explicit_bins(self):
        """If a variable has an explicit bins entry AND legacy fields exist,
        the explicit entry wins. Warning should still fire if the legacy
        field is used for any other variable; but here both vars have
        explicit bins, so no warning."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            settings = PreprocessingSettings(
                **_BASE_KWARGS,
                bins={
                    "sc_octroi_new_clus": BinConfig(
                        source_col="score_rf",
                        output_col="sc_octroi_new_clus",
                        bin_edges=_LEGACY_EDGES,
                    ),
                    "new_efx_clus": BinConfig(
                        source_col="risk_score_rf",
                        output_col="new_efx_clus",
                        bin_edges=_LEGACY_EDGES,
                    ),
                },
                # Legacy fields provided but unused (already covered by bins).
                octroi_bins=_LEGACY_EDGES,
                efx_bins=_LEGACY_EDGES,
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep == [], "warning should not fire when explicit bins cover all variables"
        # And the bins dict reflects the explicit entries, not legacy.
        assert settings.bins["sc_octroi_new_clus"].source_col == "score_rf"


@pytest.mark.skipif(
    not (os.path.exists("scripts/migrate_legacy_bins.py")),
    reason="migration script not present",
)
class TestMigrationScript:
    def test_dry_run_reports_changes(self, tmp_path):
        """Dry-run mode describes the rewrite without touching disk."""
        import subprocess

        cfg = tmp_path / "legacy.toml"
        cfg.write_text(
            "[preprocessing]\n"
            'variables = ["sc_octroi_new_clus", "new_efx_clus"]\n'
            "octroi_bins = [-1e18, 100, 1e18]\n"
            "efx_bins = [-1e18, 200, 1e18]\n"
            'date_ini_book_obs = "2024-01-01"\n'
            'date_fin_book_obs = "2024-12-31"\n'
        )
        result = subprocess.run(
            ["uv", "run", "python", "scripts/migrate_legacy_bins.py", str(cfg), "--dry-run"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "preprocessing.bins.sc_octroi_new_clus" in result.stdout
        assert "preprocessing.bins.new_efx_clus" in result.stdout
        assert "Dry run" in result.stdout
        # File must not have been modified.
        assert "octroi_bins" in cfg.read_text()
        assert "efx_bins" in cfg.read_text()

    def test_apply_rewrites_file_and_writes_backup(self, tmp_path):
        """Non-dry-run writes the new file and leaves a .bak."""
        import subprocess

        cfg = tmp_path / "legacy.toml"
        cfg.write_text(
            "[preprocessing]\n"
            'variables = ["sc_octroi_new_clus", "new_efx_clus"]\n'
            "octroi_bins = [-1e18, 100, 1e18]\n"
            "efx_bins = [-1e18, 200, 1e18]\n"
            'date_ini_book_obs = "2024-01-01"\n'
            'date_fin_book_obs = "2024-12-31"\n'
        )
        result = subprocess.run(
            ["uv", "run", "python", "scripts/migrate_legacy_bins.py", str(cfg)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        rewritten = cfg.read_text()
        # Legacy keys gone
        assert "octroi_bins" not in rewritten
        assert "efx_bins" not in rewritten
        # Modern blocks present
        assert "[preprocessing.bins.sc_octroi_new_clus]" in rewritten
        assert 'source_col = "score_rf"' in rewritten
        assert "[preprocessing.bins.new_efx_clus]" in rewritten
        assert 'source_col = "risk_score_rf"' in rewritten
        # Backup exists with the original content
        backup = cfg.with_suffix(cfg.suffix + ".bak")
        assert backup.is_file()
        assert "octroi_bins" in backup.read_text()
