"""Drive the Xtalk ASGI application in-process without a network server."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class Closed:
    """A closed WebSocket, carrying the close code sent by the server."""

    code: int
    reason: str = ""


class AsgiWebSocket:
    """Client end of one in-process ASGI WebSocket connection."""

    def __init__(
        self,
        to_app: asyncio.Queue[dict[str, Any]],
        from_app: asyncio.Queue[dict[str, Any]],
        task: asyncio.Task[None],
    ) -> None:
        self._to_app = to_app
        self._from_app = from_app
        self._task = task
        self._closed = False

    async def send_json(self, message: dict[str, Any]) -> None:
        """Send one JSON text message."""
        await self._to_app.put({"type": "websocket.receive", "text": json.dumps(message)})

    async def send_bytes(self, data: bytes) -> None:
        """Send one binary message."""
        await self._to_app.put({"type": "websocket.receive", "bytes": data})

    async def receive(self) -> dict[str, Any] | bytes | Closed:
        """Return the next server message, decoded JSON for text frames."""
        while True:
            message = await self._from_app.get()
            if message["type"] == "websocket.close":
                return Closed(int(message.get("code", 1000)), str(message.get("reason", "")))
            if message["type"] != "websocket.send":
                continue
            if message.get("bytes") is not None:
                return message["bytes"]
            return json.loads(message["text"])

    async def close(self, code: int = 1000) -> None:
        """Disconnect and wait for the server side task to finish."""
        if self._closed:
            return
        self._closed = True
        await self._to_app.put({"type": "websocket.disconnect", "code": code})
        try:
            await asyncio.wait_for(self._task, timeout=10.0)
        except asyncio.TimeoutError:
            self._task.cancel()


class InProcessServer:
    """Run one ASGI application through its lifespan and accept connections.

    Parameters
    ----------
    app : callable
        ASGI application.
    """

    def __init__(self, app: Callable[..., Any]) -> None:
        self._app = app
        self._lifespan_to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lifespan_from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._lifespan_task: asyncio.Task[None] | None = None
        self._connection_tasks: set[asyncio.Task[None]] = set()

    async def __aenter__(self) -> "InProcessServer":
        scope = {"type": "lifespan", "asgi": {"version": "3.0"}}
        self._lifespan_task = asyncio.create_task(
            self._app(scope, self._lifespan_to_app.get, self._lifespan_from_app.put)
        )
        await self._lifespan_to_app.put({"type": "lifespan.startup"})
        message = await self._lifespan_from_app.get()
        if message["type"] != "lifespan.startup.complete":
            raise RuntimeError(f"lifespan startup failed: {message}")
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for task in tuple(self._connection_tasks):
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._connection_tasks, return_exceptions=True)
        await self._lifespan_to_app.put({"type": "lifespan.shutdown"})
        await self._lifespan_from_app.get()
        if self._lifespan_task is not None:
            await self._lifespan_task

    async def connect(
        self,
        path: str = "/v1/realtime",
        *,
        query: str = "model=xtalk",
        headers: list[tuple[bytes, bytes]] | None = None,
        subprotocols: list[str] | None = None,
    ) -> AsgiWebSocket | Closed:
        """Open a WebSocket and return it, or the close sent during handshake."""
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": query.encode(),
            "headers": headers or [],
            "subprotocols": subprotocols or [],
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", 0),
        }
        to_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        from_app: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        task = asyncio.create_task(self._app(scope, to_app.get, from_app.put))
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)
        await to_app.put({"type": "websocket.connect"})
        message = await from_app.get()
        if message["type"] == "websocket.close":
            return Closed(int(message.get("code", 1000)), str(message.get("reason", "")))
        if message["type"] != "websocket.accept":
            raise RuntimeError(f"unexpected handshake message: {message}")
        return AsgiWebSocket(to_app, from_app, task)

    async def http(
        self,
        method: str,
        path: str,
        headers: list[tuple[bytes, bytes]] | None = None,
    ) -> tuple[int, list[tuple[bytes, bytes]], bytes]:
        """Perform one HTTP request and return status, headers and body."""
        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": headers or [],
            "client": ("127.0.0.1", 0),
            "server": ("127.0.0.1", 0),
        }
        requests: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        await requests.put({"type": "http.request", "body": b"", "more_body": False})
        await self._app(scope, requests.get, responses.put)
        status = 0
        response_headers: list[tuple[bytes, bytes]] = []
        body = b""
        while not responses.empty():
            message = responses.get_nowait()
            if message["type"] == "http.response.start":
                status = int(message["status"])
                response_headers = list(message.get("headers", []))
            elif message["type"] == "http.response.body":
                body += message.get("body", b"")
        return status, response_headers, body
