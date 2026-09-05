"""Regression tests for todo #63: PreprocessingSettings must WARN on unrecognized config keys
(a misspelled governance key otherwise silently falls back to its default) while staying quiet on
keys that legitimately belong to other components (batch orchestration / SegmentConstraints)."""

from __future__ import annotations

import os
import sys
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger

from src.config import PreprocessingSettings

_BASE_KWARGS: dict = {
    "variables": ["sc_octroi_new_clus", "new_efx_clus"],
    "keep_vars": ["sc_octroi_new_clus", "new_efx_clus"],
    "indicators": ["sc_octroi_new_clus", "new_efx_clus"],
    "date_ini_book_obs": "2024-01-01",
    "date_fin_book_obs": "2024-12-31",
    "octroi_bins": [float("-inf"), 100.0, float("inf")],
    "efx_bins": [float("-inf"), 100.0, float("inf")],
}


def _construct_and_capture(**extra) -> str:
    buf = StringIO()
    hid = logger.add(buf, format="{message}", level="WARNING")
    try:
        PreprocessingSettings(**_BASE_KWARGS, **extra)
    finally:
        logger.remove(hid)
    return buf.getvalue()


class TestUnknownConfigKeyGuard:
    def test_typo_key_is_flagged(self):
        # A misspelled governance key (mr_reoptimize_cutoff -> should be ...cutoffs) is surfaced.
        logs = _construct_and_capture(mr_reoptimize_cutoff=True)
        assert "unrecognized key" in logs
        assert "mr_reoptimize_cutoff" in logs

    def test_ri_optimizer_methods_plural_is_flagged(self):
        # The exact live typo: field is `ri_optimizer_method` (singular); the plural is ignored.
        logs = _construct_and_capture(ri_optimizer_methods="optuna")
        assert "ri_optimizer_methods" in logs and "unrecognized key" in logs

    def test_known_external_keys_do_not_warn(self):
        # Keys consumed by batch/SegmentConstraints legitimately share the [preprocessing] section
        # (they land here when a frozen segment config is loaded) — must NOT be flagged as typos.
        logs = _construct_and_capture(
            cutoff_ordering_mode="bottom_up",
            min_risk=0.5,
            max_risk=2.0,
            modelling_supersegment="direct",
            reporting_supersegment="direct-consolidation",
        )
        assert "unrecognized key" not in logs

    def test_all_valid_fields_produce_no_warning(self):
        logs = _construct_and_capture(multiplier=7, optimum_risk=1.1, run_ri_optimizer=True)
        assert "unrecognized key" not in logs

    def test_unknown_key_is_still_ignored_backcompat(self):
        # Warning-only: the misspelled key is still dropped (extra="ignore"), the model builds.
        settings = PreprocessingSettings(**_BASE_KWARGS, totally_made_up_key=123)
        assert not hasattr(settings, "totally_made_up_key")
