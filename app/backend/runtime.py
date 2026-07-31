"""FastAPI application composition for the desktop sidecar."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import PROTOCOL_VERSION, StartupConfig, build_effective_config
from .security import STARTUP_TOKEN_HEADER, SidecarSecurityMiddleware
from .xtalk_adapter import build_xtalk_runtime, mount_xtalk_routes


ShutdownCallback = Callable[[], Awaitable[None] | None]


def create_application(
    *,
    startup: StartupConfig,
    xtalk_runtime: Any,
    shutdown_callback: ShutdownCallback,
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
    xtalk_runtime = build_xtalk_runtime(
        effective_config,
        tools_root=startup.data_dir / "tools",
        anonymous_user_id=startup.anonymous_user_id,
    )
    return create_application(
        startup=startup,
        xtalk_runtime=xtalk_runtime,
        shutdown_callback=shutdown_callback,
    )
