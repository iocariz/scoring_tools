import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dashboard


def test_static_segment_allowlist_regular_segment(tmp_path, monkeypatch):
    base = tmp_path / "output"
    (base / "seg_a" / "data").mkdir(parents=True)
    (base / "seg_a" / "images").mkdir(parents=True)
    monkeypatch.setattr(dashboard, "OUTPUT_BASE", base)
    assert dashboard._is_allowed_static_segment("seg_a")


def test_static_segment_allowlist_blocks_traversal(tmp_path, monkeypatch):
    base = tmp_path / "output"
    base.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "OUTPUT_BASE", base)
    assert not dashboard._is_allowed_static_segment("../etc")
    assert not dashboard._is_allowed_static_segment("a/../../b")
    assert not dashboard._is_allowed_static_segment(r"..\windows")


def test_static_segment_allowlist_supersegment_exists(tmp_path, monkeypatch):
    base = tmp_path / "output"
    (base / "_supersegment_shared" / "images").mkdir(parents=True)
    monkeypatch.setattr(dashboard, "OUTPUT_BASE", base)
    assert dashboard._is_allowed_static_segment("_supersegment_shared")

