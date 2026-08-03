"""Unit tests for the loopback ASGI security policy."""

from __future__ import annotations

from urllib.parse import urlencode

import pytest

from backend.security import (
    CORE_ACCESS_TOKEN_QUERY,
    STARTUP_TOKEN_HEADER,
    STARTUP_TOKEN_QUERY,
    SecurityPolicy,
    is_loopback_host,
)


TOKEN = "t" * 32
ORIGIN = "tauri://localhost"


def _scope(
    *,
    scope_type: str = "http",
    path: str = "/health",
    method: str = "GET",
    client_host: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
    query: dict[str, str] | None = None,
) -> dict[str, object]:
    """Build a minimal HTTP or WebSocket ASGI scope."""

    return {
        "type": scope_type,
        "path": path,
        "method": method,
        "client": (client_host, 49152),
        "headers": [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in (headers or {}).items()
        ],
        "query_string": urlencode(query or {}).encode("utf-8"),
    }


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("127.12.34.56", True),
        ("::1", True),
        ("localhost", True),
        ("192.0.2.1", False),
        ("example.test", False),
        (None, False),
    ],
)
def test_is_loopback_host(host: str | None, expected: bool) -> None:
    """Recognize only local loopback peers."""

    assert is_loopback_host(host) is expected


def test_policy_rejects_remote_peer_before_other_checks() -> None:
    """Block requests whose ASGI peer is not loopback."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    denial = policy.evaluate(_scope(client_host="192.0.2.10"))

    assert denial is not None
    assert denial.status_code == 403


def test_policy_requires_token_for_native_health_without_origin() -> None:
    """Protect native probes while allowing them to omit a browser Origin."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    assert policy.evaluate(_scope(path="/health")) is not None
    assert (
        policy.evaluate(
            _scope(
                path="/health",
                headers={STARTUP_TOKEN_HEADER: TOKEN},
            )
        )
        is None
    )
    assert (
        policy.evaluate(
            _scope(
                path="/ready",
                query={STARTUP_TOKEN_QUERY: TOKEN},
            )
        )
        is None
    )


def test_policy_requires_an_explicit_allowed_origin_when_present() -> None:
    """Reject browser requests from origins not supplied by the parent."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    assert (
        policy.evaluate(
            _scope(
                path="/health",
                headers={
                    "Origin": ORIGIN,
                    STARTUP_TOKEN_HEADER: TOKEN,
                },
            )
        )
        is None
    )
    denial = policy.evaluate(
        _scope(path="/health", headers={"Origin": "https://attacker.test"})
    )
    assert denial is not None
    assert denial.status_code == 403


@pytest.mark.parametrize(
    "metadata",
    [
        {"headers": {STARTUP_TOKEN_HEADER: TOKEN}},
        {"query": {STARTUP_TOKEN_QUERY: TOKEN}},
        {"headers": {"Authorization": f"Bearer {TOKEN}"}},
    ],
)
def test_app_routes_accept_supported_launch_token_transports(
    metadata: dict[str, dict[str, str]],
) -> None:
    """Accept header, query, and app-only bearer launch authentication."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    assert policy.evaluate(_scope(path="/app/api/shutdown", **metadata)) is None


def test_core_http_routes_preserve_authorization_for_core_jwt() -> None:
    """Require the dedicated launch token alongside the core JWT header."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))
    jwt_headers = {"Authorization": "Bearer core-jwt"}

    denial = policy.evaluate(_scope(path="/api/sessions", headers=jwt_headers))
    assert denial is not None
    assert denial.status_code == 401

    jwt_headers[STARTUP_TOKEN_HEADER] = TOKEN
    assert (
        policy.evaluate(_scope(path="/api/sessions", headers=jwt_headers))
        is None
    )


def test_websocket_allows_launch_token_or_delegates_access_token_to_core() -> None:
    """Support the public SDK URL while leaving JWT validation to XTalk."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    assert (
        policy.evaluate(
            _scope(
                scope_type="websocket",
                path="/ws",
                query={STARTUP_TOKEN_QUERY: TOKEN},
            )
        )
        is None
    )
    assert (
        policy.evaluate(
            _scope(
                scope_type="websocket",
                path="/ws",
                query={CORE_ACCESS_TOKEN_QUERY: "core-jwt"},
            )
        )
        is None
    )
    denial = policy.evaluate(_scope(scope_type="websocket", path="/ws"))
    assert denial is not None
    assert denial.status_code == 401


def test_wrong_launch_token_is_rejected() -> None:
    """Use constant-value authorization rather than token presence alone."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    denial = policy.evaluate(
        _scope(
            path="/app/api/shutdown",
            headers={STARTUP_TOKEN_HEADER: "wrong"},
        )
    )

    assert denial is not None
    assert denial.status_code == 401


def test_similar_unprotected_http_prefix_is_not_misclassified() -> None:
    """Allow app-style bearer auth outside the reserved core API segment."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))
    headers = {"Authorization": f"Bearer {TOKEN}"}

    assert (
        policy.evaluate(
            _scope(path="/application-info", headers=headers)
        )
        is None
    )
    assert policy.evaluate(_scope(path="/apiary", headers=headers)) is None


def test_non_core_websocket_requires_the_launch_token() -> None:
    """Default future WebSocket routes to app launch authentication."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    denial = policy.evaluate(
        _scope(scope_type="websocket", path="/future-events")
    )
    assert denial is not None
    assert denial.status_code == 401
    assert (
        policy.evaluate(
            _scope(
                scope_type="websocket",
                path="/future-events",
                query={STARTUP_TOKEN_QUERY: TOKEN},
            )
        )
        is None
    )


def test_tool_ui_frame_defers_to_one_time_ticket_authentication() -> None:
    """Keep the launch token out of the sandboxed iframe document URL."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    assert policy.evaluate(_scope(path="/tool-ui-frame/random-ticket")) is None
    denial = policy.evaluate(
        _scope(
            path="/tool-ui-frame/random-ticket",
            client_host="192.0.2.10",
        )
    )
    assert denial is not None
    assert denial.status_code == 403


def test_options_requires_an_allowed_origin_but_not_a_token() -> None:
    """Permit only exact-origin CORS preflight requests."""

    policy = SecurityPolicy(token=TOKEN, origins=(ORIGIN,))

    missing_origin = policy.evaluate(
        _scope(method="OPTIONS", path="/api/sessions")
    )
    assert missing_origin is not None
    assert missing_origin.status_code == 403
    assert (
        policy.evaluate(
            _scope(
                method="OPTIONS",
                path="/api/sessions",
                headers={"Origin": ORIGIN},
            )
        )
        is None
    )
