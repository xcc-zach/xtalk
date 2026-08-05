"""Unit tests for FastAPI sidecar application composition."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

from backend.config import StartupConfig
from backend.runtime import create_application
from backend.security import STARTUP_TOKEN_HEADER
from backend.tool_ui import ToolUIBinding, ToolUIBroker


TOKEN = "t" * 32
ORIGIN = "tauri://localhost"


class _FakeRuntime:
    """Mount one representative core HTTP route."""

    def mount_routes(self, app: Any) -> None:
        """Register a route standing in for XTalk's public mounted routes."""

        @app.get("/api/sessions")
        async def _sessions() -> dict[str, list[Any]]:
            """Return an empty fake session list."""

            return {"sessions": []}


def _startup(tmp_path: Path) -> StartupConfig:
    """Create an application launch configuration."""

    return StartupConfig.from_mapping(
        {
            "protocol_version": 1,
            "token": TOKEN,
            "config_path": str(tmp_path / "unused.json"),
            "data_dir": str(tmp_path / "data"),
            "origins": [ORIGIN],
            "config_overlay": {},
        }
    )


async def _request(
    app: Any,
    method: str,
    path: str,
    *,
    client_host: str = "127.0.0.1",
    headers: dict[str, str] | None = None,
    json_body: Any = None,
) -> httpx.Response:
    """Issue one in-process ASGI request from an explicit peer address."""

    transport = httpx.ASGITransport(
        app=app,
        client=(client_host, 49152),
    )
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://sidecar.test",
    ) as client:
        return await client.request(
            method,
            path,
            headers=headers,
            json=json_body,
        )


def test_health_ready_and_core_route_security(tmp_path: Path) -> None:
    """Expose probes while requiring a launch token for mounted core HTTP."""

    app = create_application(
        startup=_startup(tmp_path),
        xtalk_runtime=_FakeRuntime(),
        shutdown_callback=lambda: None,
    )

    unauthorized_health = asyncio.run(_request(app, "GET", "/health"))
    health = asyncio.run(
        _request(
            app,
            "GET",
            "/health",
            headers={STARTUP_TOKEN_HEADER: TOKEN},
        )
    )
    ready = asyncio.run(
        _request(
            app,
            "GET",
            "/ready",
            headers={STARTUP_TOKEN_HEADER: TOKEN},
        )
    )
    unauthorized = asyncio.run(_request(app, "GET", "/api/sessions"))
    authorized = asyncio.run(
        _request(
            app,
            "GET",
            "/api/sessions",
            headers={
                STARTUP_TOKEN_HEADER: TOKEN,
                "Authorization": "Bearer core-jwt",
            },
        )
    )

    assert unauthorized_health.status_code == 401
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "protocol_version": 1}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "protocol_version": 1}
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert authorized.json() == {"sessions": []}


def test_shutdown_requires_token_and_invokes_callback(tmp_path: Path) -> None:
    """Protect controlled shutdown and invoke the lifecycle callback once."""

    shutdown_requests: list[bool] = []
    app = create_application(
        startup=_startup(tmp_path),
        xtalk_runtime=_FakeRuntime(),
        shutdown_callback=lambda: shutdown_requests.append(True),
    )

    unauthorized = asyncio.run(
        _request(app, "POST", "/app/api/shutdown")
    )
    authorized = asyncio.run(
        _request(
            app,
            "POST",
            "/app/api/shutdown",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
    )

    assert unauthorized.status_code == 401
    assert shutdown_requests == [True]
    assert authorized.status_code == 200
    assert authorized.json() == {"status": "shutting_down"}


def test_application_rejects_non_loopback_and_unlisted_origin(
    tmp_path: Path,
) -> None:
    """Enforce both network and browser-origin boundaries."""

    app = create_application(
        startup=_startup(tmp_path),
        xtalk_runtime=_FakeRuntime(),
        shutdown_callback=lambda: None,
    )

    remote = asyncio.run(
        _request(app, "GET", "/health", client_host="192.0.2.10")
    )
    bad_origin = asyncio.run(
        _request(
            app,
            "GET",
            "/health",
            headers={"Origin": "https://attacker.test"},
        )
    )
    allowed_origin = asyncio.run(
        _request(
            app,
            "GET",
            "/health",
            headers={
                "Origin": ORIGIN,
                STARTUP_TOKEN_HEADER: TOKEN,
            },
        )
    )

    assert remote.status_code == 403
    assert bad_origin.status_code == 403
    assert allowed_origin.status_code == 200
    assert allowed_origin.headers["access-control-allow-origin"] == ORIGIN


def test_cors_preflight_requires_an_exact_origin_but_not_a_token(
    tmp_path: Path,
) -> None:
    """Run security checks before handing valid preflight to CORS."""

    app = create_application(
        startup=_startup(tmp_path),
        xtalk_runtime=_FakeRuntime(),
        shutdown_callback=lambda: None,
    )
    preflight_headers = {
        "Origin": ORIGIN,
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": STARTUP_TOKEN_HEADER,
    }

    missing_origin = asyncio.run(
        _request(app, "OPTIONS", "/api/sessions")
    )
    allowed = asyncio.run(
        _request(
            app,
            "OPTIONS",
            "/api/sessions",
            headers=preflight_headers,
        )
    )
    blocked = asyncio.run(
        _request(
            app,
            "OPTIONS",
            "/api/sessions",
            headers={
                **preflight_headers,
                "Origin": "https://attacker.test",
            },
        )
    )

    assert missing_origin.status_code == 403
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == ORIGIN
    assert blocked.status_code == 403


def test_tool_ui_frame_uses_authenticated_runtime_scoped_reload_ticket(
    tmp_path: Path,
) -> None:
    """Serve runtime-scoped reloads without exposing the launch token."""

    app = create_application(
        startup=_startup(tmp_path),
        xtalk_runtime=_FakeRuntime(),
        shutdown_callback=lambda: None,
        tool_ui_broker=ToolUIBroker(),
    )
    source = "<!doctype html><script>document.body.textContent='ok'</script>"
    created = asyncio.run(
        _request(
            app,
            "POST",
            "/app/api/tool-ui/frames",
            headers={STARTUP_TOKEN_HEADER: TOKEN},
            json_body={"source": source},
        )
    )
    ticket = created.json()["ticket"]
    first = asyncio.run(
        _request(app, "GET", f"/tool-ui-frame/{ticket}")
    )
    second = asyncio.run(
        _request(app, "GET", f"/tool-ui-frame/{ticket}")
    )
    third = asyncio.run(_request(app, "GET", f"/tool-ui-frame/{ticket}"))
    repeated = [
        asyncio.run(_request(app, "GET", f"/tool-ui-frame/{ticket}"))
        for _ in range(10)
    ]

    assert created.status_code == 200
    assert first.status_code == 200
    assert first.text == source
    assert first.headers["cache-control"] == "no-store"
    assert "script-src 'unsafe-inline'" in first.headers[
        "content-security-policy"
    ]
    assert second.text == source
    assert third.text == source
    assert all(response.status_code == 200 for response in repeated)
    assert all(response.text == source for response in repeated)


def test_tool_ui_event_snapshot_requires_token_and_replays_session_events(
    tmp_path: Path,
) -> None:
    """Deliver Tool UI events over the WebKit-compatible HTTP channel."""

    async def scenario() -> tuple[httpx.Response, httpx.Response]:
        broker = ToolUIBroker()
        await broker.publish_status(
            binding=ToolUIBinding(
                tool_id="builtin:timer",
                update_every_s=0.5,
            ),
            tool_name="timer",
            call_id="timer-call",
            status="5 of 10 seconds",
            running=True,
        )
        await broker.publish_emit(
            binding=ToolUIBinding(
                tool_id="builtin:timer",
                update_every_s=0.5,
            ),
            tool_name="timer",
            call_id="timer-call",
            message="Timer started",
            status="5 of 10 seconds",
            running=True,
        )
        app = create_application(
            startup=_startup(tmp_path),
            xtalk_runtime=_FakeRuntime(),
            shutdown_callback=lambda: None,
            tool_ui_broker=broker,
        )
        unauthorized = await _request(
            app,
            "GET",
            "/app/api/tool-ui/events?session_id=session-1",
        )
        authorized = await _request(
            app,
            "GET",
            "/app/api/tool-ui/events?session_id=session-1",
            headers={STARTUP_TOKEN_HEADER: TOKEN},
        )
        return unauthorized, authorized

    unauthorized, authorized = asyncio.run(scenario())

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    events = authorized.json()["events"]
    assert [event["type"] for event in events] == [
        "tool_ui.emit",
        "tool_ui.status",
    ]
    assert all(event["sessionId"] == "session-1" for event in events)


def test_tool_ui_event_snapshot_retains_structured_payload(
    tmp_path: Path,
) -> None:
    """Deliver structured whiteboard content over the event channel."""

    snapshot = {
        "version": 1,
        "call_id": "board-call",
        "title": "计划",
        "revision": 1,
        "notes": [{"id": "n1", "text": "便签", "color": "yellow"}],
        "updated_at": "now",
    }

    async def scenario() -> httpx.Response:
        broker = ToolUIBroker()
        await broker.publish_emit(
            binding=ToolUIBinding(
                tool_id="builtin:whiteboard",
                update_every_s=-1,
            ),
            tool_name="whiteboard_update",
            call_id="board-call",
            message=json.dumps(snapshot, ensure_ascii=False),
            status="complete",
            running=False,
            payload=snapshot,
        )
        app = create_application(
            startup=_startup(tmp_path),
            xtalk_runtime=_FakeRuntime(),
            shutdown_callback=lambda: None,
            tool_ui_broker=broker,
        )
        return await _request(
            app,
            "GET",
            "/app/api/tool-ui/events?session_id=session-1",
            headers={STARTUP_TOKEN_HEADER: TOKEN},
        )

    response = asyncio.run(scenario())

    assert response.status_code == 200
    events = response.json()["events"]
    assert len(events) == 1
    assert events[0]["payload"] == snapshot
    assert events[0]["payload"]["notes"][0]["text"] == "便签"
