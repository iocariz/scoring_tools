"""Authentication helpers for the dashboard web apps (R1 todo #50).

Policy (hard-gate):

- Binding to a loopback address (``127.0.0.1`` / ``::1`` / ``localhost``) is
  always allowed, with or without credentials. This preserves the
  developer-on-localhost workflow.
- Binding to any other interface (including ``0.0.0.0``) **requires**
  ``DASHBOARD_AUTH_USER`` and ``DASHBOARD_AUTH_PASS`` env vars. If either
  is missing, the application refuses to start.

The mirror of R0's RCE hardening (#44, #47, #48): security checks are
hard-gated. No silent degradation to insecure defaults when a human
forgot to set a variable — that is exactly how dashboards end up on the
public internet leaking loan KPIs.

Usage (Flask / Dash):

    import os
    from src.web_auth import (
        enforce_bind_auth_policy,
        install_basic_auth,
        LOCALHOST_HOSTS,
    )

    host = args.host  # from argparse; default 127.0.0.1
    creds = enforce_bind_auth_policy(host)   # raises if non-localhost w/o creds
    if creds is not None:
        install_basic_auth(app.server, creds)
    app.run(host=host, port=args.port)

Usage (Gradio):

    creds = enforce_bind_auth_policy(host)
    app.launch(server_name=host, server_port=port, auth=creds)  # None → no auth
"""

from __future__ import annotations

import hmac
import os
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover — typing only
    from flask import Flask


# Hosts that are treated as loopback-only and therefore not auth-required.
# ``0.0.0.0`` is deliberately EXCLUDED — it binds to all interfaces, so a
# misconfigured firewall or container exposure would leak it.
LOCALHOST_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "::1", "localhost"})

_USER_ENV = "DASHBOARD_AUTH_USER"
_PASS_ENV = "DASHBOARD_AUTH_PASS"


class DashboardAuthError(RuntimeError):
    """Raised when the auth policy refuses to let the dashboard start."""


def is_localhost_host(host: str) -> bool:
    """Return True if *host* refers to loopback only.

    Matches only the literal loopback forms. Bracketed IPv6 (``[::1]``) is
    allowed. Any other host — including ``0.0.0.0``, any LAN IP, or a
    DNS name — is treated as a non-localhost bind.
    """
    if not host:
        return False
    normalised = host.strip().lower()
    # Strip IPv6 brackets if present.
    if normalised.startswith("[") and normalised.endswith("]"):
        normalised = normalised[1:-1]
    return normalised in LOCALHOST_HOSTS


def get_auth_credentials() -> tuple[str, str] | None:
    """Return ``(user, pass)`` if both env vars are set, else ``None``."""
    user = os.environ.get(_USER_ENV)
    password = os.environ.get(_PASS_ENV)
    if user and password:
        return (user, password)
    return None


def enforce_bind_auth_policy(host: str) -> tuple[str, str] | None:
    """Enforce the auth policy for the bind target.

    Returns the credential tuple if auth is required AND configured.
    Returns ``None`` when auth is not required (localhost bind with no
    creds) OR when creds are set but the bind is loopback (auth still
    enabled as a courtesy, still returns the creds).

    Raises ``DashboardAuthError`` when the bind is non-localhost and no
    credentials are set. Refuses to let the app start in that case.
    """
    creds = get_auth_credentials()
    if is_localhost_host(host):
        if creds is None:
            logger.info(f"Dashboard binding to loopback {host!r}; auth disabled (no {_USER_ENV}/{_PASS_ENV} set).")
        else:
            logger.info(f"Dashboard binding to loopback {host!r}; Basic Auth enabled via {_USER_ENV}.")
        return creds

    # Non-localhost bind → creds MUST be set.
    if creds is None:
        raise DashboardAuthError(
            f"Refusing to bind dashboard to {host!r}: non-localhost bind requires "
            f"both {_USER_ENV} and {_PASS_ENV} to be set in the environment. "
            f"This is a hard gate — see src/web_auth.py and todo #50."
        )
    logger.warning(
        f"Dashboard binding to non-localhost host {host!r}; Basic Auth is enforced. "
        f"Ensure {_USER_ENV}/{_PASS_ENV} are set via a secret manager, not committed to git."
    )
    return creds


def install_basic_auth(flask_app: Flask, creds: tuple[str, str]) -> None:
    """Install a ``before_request`` hook on *flask_app* that enforces Basic Auth.

    Timing-safe credential comparison via :func:`hmac.compare_digest` —
    avoids leaking user/password length via string-comparison side channels.
    """
    expected_user, expected_pass = creds
    expected_user_b = expected_user.encode()
    expected_pass_b = expected_pass.encode()

    @flask_app.before_request
    def _require_auth():  # pragma: no cover — exercised via test_client
        from flask import request

        auth = request.authorization
        if auth is None:
            return _challenge()
        submitted_user = (auth.username or "").encode()
        submitted_pass = (auth.password or "").encode()
        # hmac.compare_digest requires equal-length inputs; pad mismatches.
        user_ok = hmac.compare_digest(
            submitted_user.ljust(len(expected_user_b), b"\x00"),
            expected_user_b.ljust(len(submitted_user), b"\x00"),
        ) and len(submitted_user) == len(expected_user_b)
        pass_ok = hmac.compare_digest(
            submitted_pass.ljust(len(expected_pass_b), b"\x00"),
            expected_pass_b.ljust(len(submitted_pass), b"\x00"),
        ) and len(submitted_pass) == len(expected_pass_b)
        if not (user_ok and pass_ok):
            return _challenge()

        return None  # proceed

    # Name used in tests to check installation; kept on the function for introspection.
    _require_auth.__ecc_auth_installed__ = True  # type: ignore[attr-defined]


def _challenge():  # pragma: no cover — exercised via test_client
    from flask import Response

    return Response(
        "Authentication required",
        status=401,
        headers={"WWW-Authenticate": 'Basic realm="scoring-tools dashboard"'},
    )
