"""Fixtures shared by the backend conversation experience tests."""

from __future__ import annotations

from typing import Any

import pytest

from tests.support.asgi import Closed, InProcessServer
from tests.support.client import RealtimeTestClient
from tests.support.server import build_mock_xtalk


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def server(request: pytest.FixtureRequest):
    """Build the mock application and drive its lifespan for one test.

    ``@pytest.mark.mock_config(**kwargs)`` overrides :func:`mock_config`.
    """
    marker = request.node.get_closest_marker("mock_config")
    overrides = dict(marker.kwargs) if marker else {}
    async with InProcessServer(build_mock_xtalk(**overrides).create_asgi_app()) as running:
        yield running


@pytest.fixture
async def connect(server: InProcessServer):
    """Return a factory that opens ready ``RealtimeTestClient`` sessions."""
    clients: list[RealtimeTestClient] = []
    extra_servers: list[InProcessServer] = []

    async def _connect(
        session_update: dict[str, Any] | None = None,
        *,
        config: dict[str, Any] | None = None,
        query: str = "model=xtalk",
        headers: list[tuple[bytes, bytes]] | None = None,
        subprotocols: list[str] | None = None,
        **client_options: Any,
    ) -> RealtimeTestClient:
        target = server
        if config is not None:
            target = InProcessServer(build_mock_xtalk(**config).create_asgi_app())
            await target.__aenter__()
            extra_servers.append(target)
        websocket = await target.connect(query=query, headers=headers, subprotocols=subprotocols)
        if isinstance(websocket, Closed):
            raise AssertionError(f"handshake rejected with {websocket.code}")
        client = RealtimeTestClient(websocket, **client_options)
        clients.append(client)
        await client.open(session_update)
        return client

    yield _connect
    for client in clients:
        await client.close()
    for extra in extra_servers:
        await extra.__aexit__(None, None, None)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "mock_config(**kwargs): override the mock server config")
