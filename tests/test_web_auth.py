"""Tests for src/web_auth.py — R1 todo #50 auth policy.

Hard-gate policy: non-localhost binds require DASHBOARD_AUTH_USER and
DASHBOARD_AUTH_PASS; loopback binds allow optional auth.
"""

from __future__ import annotations

import base64
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from flask import Flask

from src.web_auth import (
    DashboardAuthError,
    enforce_bind_auth_policy,
    get_auth_credentials,
    install_basic_auth,
    is_localhost_host,
)


class TestIsLocalhost:
    @pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "[::1]", "LOCALHOST"])
    def test_loopback_forms(self, host):
        assert is_localhost_host(host)

    @pytest.mark.parametrize(
        "host",
        [
            "0.0.0.0",  # all interfaces — NOT loopback
            "192.168.1.10",
            "10.0.0.5",
            "dashboard.example.com",
            "",  # empty is treated as non-loopback (fail closed)
        ],
    )
    def test_non_loopback(self, host):
        assert not is_localhost_host(host)


class TestGetAuthCredentials:
    def test_both_set(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_USER", "alice")
        monkeypatch.setenv("DASHBOARD_AUTH_PASS", "s3cret")
        assert get_auth_credentials() == ("alice", "s3cret")

    def test_missing_user(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_AUTH_USER", raising=False)
        monkeypatch.setenv("DASHBOARD_AUTH_PASS", "s3cret")
        assert get_auth_credentials() is None

    def test_missing_pass(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_USER", "alice")
        monkeypatch.delenv("DASHBOARD_AUTH_PASS", raising=False)
        assert get_auth_credentials() is None

    def test_empty_user_treated_as_missing(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_USER", "")
        monkeypatch.setenv("DASHBOARD_AUTH_PASS", "s3cret")
        assert get_auth_credentials() is None


class TestEnforceBindAuthPolicy:
    def test_localhost_without_creds_allowed(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_AUTH_USER", raising=False)
        monkeypatch.delenv("DASHBOARD_AUTH_PASS", raising=False)
        assert enforce_bind_auth_policy("127.0.0.1") is None

    def test_localhost_with_creds_returns_creds(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_USER", "alice")
        monkeypatch.setenv("DASHBOARD_AUTH_PASS", "s3cret")
        assert enforce_bind_auth_policy("localhost") == ("alice", "s3cret")

    def test_non_localhost_without_creds_raises(self, monkeypatch):
        monkeypatch.delenv("DASHBOARD_AUTH_USER", raising=False)
        monkeypatch.delenv("DASHBOARD_AUTH_PASS", raising=False)
        with pytest.raises(DashboardAuthError, match="non-localhost bind requires"):
            enforce_bind_auth_policy("0.0.0.0")

    def test_non_localhost_with_creds_allowed(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_AUTH_USER", "alice")
        monkeypatch.setenv("DASHBOARD_AUTH_PASS", "s3cret")
        assert enforce_bind_auth_policy("192.168.1.10") == ("alice", "s3cret")


class TestInstallBasicAuth:
    @pytest.fixture
    def app_with_auth(self):
        app = Flask(__name__)

        @app.route("/")
        def index():
            return "secret data", 200

        install_basic_auth(app, ("alice", "s3cret"))
        return app

    def test_missing_header_returns_401(self, app_with_auth):
        client = app_with_auth.test_client()
        resp = client.get("/")
        assert resp.status_code == 401
        assert "Basic" in resp.headers.get("WWW-Authenticate", "")

    def test_wrong_password_returns_401(self, app_with_auth):
        client = app_with_auth.test_client()
        token = base64.b64encode(b"alice:wrong").decode()
        resp = client.get("/", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 401

    def test_wrong_user_returns_401(self, app_with_auth):
        client = app_with_auth.test_client()
        token = base64.b64encode(b"bob:s3cret").decode()
        resp = client.get("/", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 401

    def test_correct_credentials_allow_request(self, app_with_auth):
        client = app_with_auth.test_client()
        token = base64.b64encode(b"alice:s3cret").decode()
        resp = client.get("/", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 200
        assert resp.data == b"secret data"

    def test_empty_credentials_rejected(self, app_with_auth):
        client = app_with_auth.test_client()
        token = base64.b64encode(b":").decode()
        resp = client.get("/", headers={"Authorization": f"Basic {token}"})
        assert resp.status_code == 401
