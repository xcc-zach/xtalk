"""FastAPI application composition for the desktop sidecar."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from .config import PROTOCOL_VERSION, StartupConfig, build_effective_config
from .desktop_tool_bridge import DesktopToolCallBridge
from .frontend_text_log import FrontendTextLogMiddleware
from .security import STARTUP_TOKEN_HEADER, SidecarSecurityMiddleware
from .tool_ui import ToolUIBroker
from .whiteboard_store import get_whiteboard_store
from .xtalk_adapter import build_xtalk_runtime, mount_xtalk_routes


ShutdownCallback = Callable[[], Awaitable[None] | None]


def create_application(
    *,
    startup: StartupConfig,
    xtalk_runtime: Any,
    shutdown_callback: ShutdownCallback,
    tool_ui_broker: ToolUIBroker | None = None,
) -> FastAPI:
    """Create the secured app wrapper and mount a prebuilt XTalk runtime.

    Parameters
    ----------
    startup : StartupConfig
        Validated launch configuration.
    xtalk_runtime : Any
        Public XTalk runtime wrapper.
    shutdown_callback : ShutdownCallback
        Callback requesting graceful Uvicorn termination.
    tool_ui_broker : ToolUIBroker | None, optional
        Read-only developer-tool UI event broker.

    Returns
    -------
    fastapi.FastAPI
        Fully mounted sidecar application.
    """

    app = FastAPI(
        title="XTalk Desktop Sidecar",
        docs_url=None,
        redoc_url=None,
    )
    app.state.sidecar_ready = False

    app.add_middleware(
        FrontendTextLogMiddleware,
        log_path=startup.data_dir / "frontend-text.log",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(startup.origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            STARTUP_TOKEN_HEADER,
        ],
    )
    # Added after CORS so Starlette makes this the outer middleware. Loopback
    # and exact-Origin checks therefore run before CORS handles a valid
    # preflight request.
    app.add_middleware(
        SidecarSecurityMiddleware,
        token=startup.token,
        origins=startup.origins,
    )

    @app.get("/health")
    async def _health() -> dict[str, Any]:
        """Return process liveness without disclosing configuration."""

        return {
            "status": "ok",
            "protocol_version": PROTOCOL_VERSION,
        }

    @app.get("/ready")
    async def _ready() -> JSONResponse:
        """Return whether XTalk routes finished mounting."""

        ready = bool(app.state.sidecar_ready)
        return JSONResponse(
            status_code=200 if ready else 503,
            content={
                "status": "ready" if ready else "starting",
                "protocol_version": PROTOCOL_VERSION,
            },
        )

    @app.post("/app/api/shutdown")
    async def _shutdown() -> dict[str, str]:
        """Request a graceful sidecar shutdown."""

        callback_result = shutdown_callback()
        if inspect.isawaitable(callback_result):
            await callback_result
        return {"status": "shutting_down"}

    @app.delete("/app/api/sessions/{session_id}")
    async def _delete_session(session_id: str) -> dict[str, str]:
        """Delete one persisted chat session owned by the desktop identity.

        The desktop UI talks to a single anonymous identity, so the delete is
        scoped by that identity. Any live service for the session is stopped
        before its SQLite rows are removed, letting the ``chat_messages``
        foreign key cascade clean up the history.

        Parameters
        ----------
        session_id : str
            Identifier of the persisted session to delete.

        Returns
        -------
        dict[str, str]
            Status acknowledgment once the session and its rows are removed.

        Raises
        ------
        fastapi.HTTPException
            Raised with 404 when persistence or the desktop identity is
            unavailable, or the session is not owned by that identity.
        """

        persistence = xtalk_runtime._persistence
        user_id = xtalk_runtime._anonymous_user_id
        if (
            persistence is None
            or user_id is None
            or not persistence.user_owns_session(user_id, session_id)
        ):
            raise HTTPException(
                status_code=404,
                detail="Session not found",
            )
        # Stop the in-memory service first so any final message flush happens
        # before the rows it targets are removed from the SQLite store.
        service_manager = xtalk_runtime._service_manager
        if service_manager is not None:
            await service_manager.remove_service(session_id)
        # The persistence store lives in the public ``xtalk`` package, so the
        # App reuses its connection and lock directly for the deletion.
        with persistence._lock, persistence._connect() as conn:
            conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
        return {"status": "ok"}

    @app.get("/app/api/whiteboard")
    async def _whiteboard(session_id: str) -> dict[str, Any]:
        """Return one conversation's whiteboard text snapshot for the App.

        Each conversation owns an independent Markdown document. The trusted
        whiteboard window polls this read-only endpoint with the active
        conversation's session id and renders the returned text as Markdown.

        Parameters
        ----------
        session_id : str
            Persisted chat session owning the requested board.

        Returns
        -------
        dict[str, Any]
            Normalized ``{version, text, revision, updated_at}`` snapshot.
        """

        return get_whiteboard_store(session_id).snapshot()

    if tool_ui_broker is not None:

        @app.post("/app/api/tool-ui/frames")
        async def _create_tool_ui_frame(payload: dict[str, Any]) -> dict[str, str]:
            """Create one runtime-scoped sandbox-frame document ticket."""

            if set(payload) != {"source"} or not isinstance(
                payload["source"],
                str,
            ):
                raise HTTPException(
                    status_code=400,
                    detail="A tool UI frame source is required",
                )
            try:
                ticket = await tool_ui_broker.create_frame_ticket(
                    payload["source"]
                )
            except ValueError as exc:
                raise HTTPException(status_code=413, detail=str(exc)) from exc
            return {"ticket": ticket}

        @app.get("/app/api/tool-ui/events")
        async def _tool_ui_events(session_id: str) -> dict[str, Any]:
            """Return replayable Tool UI events for the active App session."""

            try:
                events = await tool_ui_broker.snapshot(session_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {"events": events}

        @app.get("/tool-ui-frame/{ticket}")
        async def _tool_ui_frame(ticket: str) -> HTMLResponse:
            """Consume one ticket and return a sandboxed frame document."""

            source = await tool_ui_broker.consume_frame_ticket(ticket)
            if source is None:
                raise HTTPException(
                    status_code=404,
                    detail="Tool UI frame is unavailable",
                )
            return HTMLResponse(
                source,
                headers={
                    "Cache-Control": "no-store",
                    "Content-Security-Policy": (
                        "default-src 'none'; base-uri 'none'; "
                        "connect-src 'none'; font-src 'none'; "
                        "form-action 'none'; frame-src 'none'; "
                        "img-src data: blob:; media-src 'none'; "
                        "object-src 'none'; style-src 'unsafe-inline'; "
                        "script-src 'unsafe-inline'"
                    ),
                    "Referrer-Policy": "no-referrer",
                    "X-Content-Type-Options": "nosniff",
                },
            )

        @app.websocket("/app/tool-ui/ws")
        async def _tool_ui(websocket: Any) -> None:
            """Stream read-only developer-tool UI events to the App."""

            await tool_ui_broker.serve(websocket)

    mount_xtalk_routes(xtalk_runtime, app)
    app.state.sidecar_ready = True
    return app


def build_application(
    *,
    startup: StartupConfig,
    shutdown_callback: ShutdownCallback,
) -> FastAPI:
    """Compose configuration, build XTalk, and create the FastAPI wrapper.

    Parameters
    ----------
    startup : StartupConfig
        Validated launch configuration.
    shutdown_callback : ShutdownCallback
        Callback requesting graceful Uvicorn termination.

    Returns
    -------
    fastapi.FastAPI
        Ready-to-serve application.
    """

    effective_config = build_effective_config(startup)
    tool_call_bridge = DesktopToolCallBridge()
    effective_config.setdefault("service_config", {})[
        "_desktop_tool_call_bridge"
    ] = tool_call_bridge
    tool_ui_broker = ToolUIBroker(bridge=tool_call_bridge)
    xtalk_runtime = build_xtalk_runtime(
        effective_config,
        tools_root=startup.data_dir / "tools",
        builtin_tools_root=startup.builtin_tools_root,
        anonymous_user_id=startup.anonymous_user_id,
        tool_ui_broker=tool_ui_broker,
    )
    return create_application(
        startup=startup,
        xtalk_runtime=xtalk_runtime,
        shutdown_callback=shutdown_callback,
        tool_ui_broker=tool_ui_broker,
    )
