"""Loopback Uvicorn lifecycle for the desktop sidecar."""

from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import dataclass
from typing import Any, TextIO

import uvicorn

from .config import PROTOCOL_VERSION, StartupConfig
from .runtime import build_application


LOOPBACK_HOST = "127.0.0.1"
STARTUP_TIMEOUT_SECONDS = 30.0


@dataclass
class ShutdownController:
    """Bridge the HTTP shutdown route to a Uvicorn server instance."""

    server: uvicorn.Server | None = None

    def attach(self, server: uvicorn.Server) -> None:
        """Attach the Uvicorn server after application construction.

        Parameters
        ----------
        server : uvicorn.Server
            Server whose graceful-exit flag should be controlled.
        """

        self.server = server

    def request(self) -> None:
        """Request graceful server termination."""

        if self.server is not None:
            self.server.should_exit = True


def bind_loopback_socket() -> socket.socket:
    """Bind and listen on an OS-assigned IPv4 loopback port.

    Returns
    -------
    socket.socket
        Pre-bound listening socket owned by the caller.
    """

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server_socket.bind((LOOPBACK_HOST, 0))
        server_socket.listen(2048)
        server_socket.set_inheritable(False)
    except OSError:
        server_socket.close()
        raise
    return server_socket


def create_uvicorn_config(app: Any) -> uvicorn.Config:
    """Create a quiet, non-proxy-aware Uvicorn configuration.

    Parameters
    ----------
    app : Any
        ASGI application.

    Returns
    -------
    uvicorn.Config
        Server configuration with access logging disabled.
    """

    return uvicorn.Config(
        app,
        host=LOOPBACK_HOST,
        port=0,
        log_level="warning",
        access_log=False,
        proxy_headers=False,
        lifespan="on",
    )


def write_ready_message(stream: TextIO, *, port: int) -> None:
    """Write and flush the first sidecar protocol response.

    Parameters
    ----------
    stream : TextIO
        Original process stdout reserved for protocol messages.
    port : int
        Bound loopback TCP port.
    """

    payload = {
        "type": "ready",
        "protocol_version": PROTOCOL_VERSION,
        "port": port,
    }
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


async def _wait_until_started(
    server: uvicorn.Server,
    server_task: asyncio.Task[None],
) -> None:
    """Wait until Uvicorn reports readiness or fails."""

    loop = asyncio.get_running_loop()
    deadline = loop.time() + STARTUP_TIMEOUT_SECONDS
    while not server.started:
        if server_task.done():
            await server_task
            raise RuntimeError("Uvicorn exited before reporting readiness")
        if loop.time() >= deadline:
            server.should_exit = True
            await server_task
            raise TimeoutError("Uvicorn did not become ready in time")
        await asyncio.sleep(0.01)


async def serve_sidecar(
    startup: StartupConfig,
    *,
    protocol_output: TextIO,
) -> None:
    """Build and serve the sidecar until graceful shutdown.

    Parameters
    ----------
    startup : StartupConfig
        Validated launch configuration.
    protocol_output : TextIO
        Original stdout stream used only for protocol messages.
    """

    shutdown_controller = ShutdownController()
    app = build_application(
        startup=startup,
        shutdown_callback=shutdown_controller.request,
    )
    server_socket = bind_loopback_socket()
    port = int(server_socket.getsockname()[1])
    server = uvicorn.Server(create_uvicorn_config(app))
    shutdown_controller.attach(server)
    server_task = asyncio.create_task(
        server.serve(sockets=[server_socket]),
        name="xtalk-app-sidecar",
    )

    try:
        await _wait_until_started(server, server_task)
        write_ready_message(protocol_output, port=port)
        await server_task
    finally:
        if not server_task.done():
            server.should_exit = True
            await server_task
        server_socket.close()
