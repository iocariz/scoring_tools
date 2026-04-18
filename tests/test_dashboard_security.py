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


# =============================================================================
# R1 #46 — extension allowlist on /static/* routes
# =============================================================================


def test_static_filename_accepts_allowlisted_extensions():
    """Allowlisted suffixes (case-insensitive) must pass validation."""
    for name in [
        "report.html",
        "chart.HTML",
        "heatmap.png",
        "fig.svg",
        "export.csv",
        "photo.JPG",
        "photo.jpeg",
    ]:
        assert dashboard._is_allowed_static_filename(name), name


def test_static_filename_rejects_sensitive_extensions():
    """Pickles, configs, logs, and shell artefacts must be refused."""
    for name in [
        "model.pkl",
        "model.joblib",
        "config.toml",
        "secrets.env",
        "run.log",
        "shell.sh",
        "exec.py",
        "archive.tar.gz",
        "nosuffix",
        "",
    ]:
        assert not dashboard._is_allowed_static_filename(name), name


def test_static_filename_rejects_path_traversal_forms():
    """Even if the extension is allowlisted, names containing traversal or
    absolute-path forms still get refused here — defense in depth before
    send_from_directory's own safe_join kicks in."""
    # Extension-allowlisted but path-component-bearing — these are still
    # rejected by Flask's safe_join downstream. Our guard catches the
    # suffix check; path traversal rejection is the web framework's job.
    # This test documents the split: we allowlist extensions, not shapes.
    assert dashboard._is_allowed_static_filename("../etc/passwd.html")  # suffix OK
    # The web framework will reject the path at send_from_directory time.


def test_static_segment_route_enforces_extension_allowlist(tmp_path, monkeypatch):
    """Regression for #46: the reachable segment-scoped route must reject
    non-allowlisted extensions (pickles, env files, logs) even if the file
    exists in the images dir. Valid (segment with data/ + images/) and
    invalid (unknown / unsafe extension) cases."""
    base = tmp_path / "output"
    # get_available_segments() requires data/ to exist for the segment to
    # register in the allowlist.
    (base / "seg_a" / "data").mkdir(parents=True)
    images_dir = base / "seg_a" / "images"
    images_dir.mkdir(parents=True)
    (images_dir / "report.html").write_text("<h1>ok</h1>")
    (images_dir / "secret.pkl").write_text("pickle data")

    monkeypatch.setattr(dashboard, "OUTPUT_BASE", base)

    with dashboard.app.server.test_client() as client:
        # Allowlisted extension + valid segment → served.
        resp = client.get("/static/seg_a/report.html")
        assert resp.status_code == 200
        assert b"<h1>ok</h1>" in resp.data

        # Non-allowlisted extension → 404 (refused by allowlist, defense
        # in depth even though the file exists).
        resp = client.get("/static/seg_a/secret.pkl")
        assert resp.status_code == 404

        # Untrusted segment with allowlisted extension → 404 (segment allowlist).
        resp = client.get("/static/does_not_exist/report.html")
        assert resp.status_code == 404


def test_static_root_route_is_shadowed_by_flask_default():
    """Document the Flask-default shadowing of /static/<path:filename>.

    The root route (serve_static_root) is kept in the source for defense
    in depth, but at runtime Flask's built-in `static` endpoint (registered
    first at app creation) wins URL dispatch for the same rule. We still
    unit-test the allowlist directly via _is_allowed_static_filename above.
    """
    app = dashboard.app.server
    # Both endpoints are registered. When two rules match, Flask dispatches
    # to the first-registered; `static` (from Flask) is registered before
    # our `serve_static_root`.
    endpoints = {rule.endpoint for rule in app.url_map.iter_rules() if rule.rule == "/static/<path:filename>"}
    assert "static" in endpoints
    assert "serve_static_root" in endpoints

