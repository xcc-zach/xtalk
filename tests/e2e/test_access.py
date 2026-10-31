"""Connecting to the service with and without credentials."""

from __future__ import annotations

import json

import pytest

from tests.support.asgi import AsgiWebSocket, Closed, InProcessServer
from tests.support.client import RealtimeTestClient

pytestmark = pytest.mark.anyio

KEY = "secret-key"
BEARER = [(b"authorization", f"Bearer {KEY}".encode())]
SUBPROTOCOLS = ["realtime", f"openai-insecure-api-key.{KEY}"]


async def _session_ready(websocket: AsgiWebSocket | Closed) -> bool:
    if isinstance(websocket, Closed):
        return False
    client = RealtimeTestClient(websocket)
    await client.open()
    ready = client.session is not None
    await client.close()
    return ready


async def test_connects_without_credentials_by_default(server: InProcessServer):
    assert await _session_ready(await server.connect())


@pytest.mark.mock_config(api_keys=(KEY,))
async def test_missing_or_wrong_credentials_are_rejected(server: InProcessServer):
    assert isinstance(await server.connect(), Closed)
    assert isinstance(
        await server.connect(headers=[(b"authorization", b"Bearer wrong")]), Closed
    )


@pytest.mark.mock_config(api_keys=(KEY,))
async def test_bearer_and_subprotocol_credentials_both_work(server: InProcessServer):
    assert await _session_ready(await server.connect(headers=BEARER))
    assert await _session_ready(await server.connect(subprotocols=SUBPROTOCOLS))


async def test_models_endpoint_lists_the_instance_model(server: InProcessServer):
    status, _, body = await server.http("GET", "/v1/models")
    assert status == 200
    assert [entry["id"] for entry in json.loads(body)["data"]] == ["xtalk"]


@pytest.mark.mock_config(api_keys=(KEY,))
async def test_models_endpoint_follows_the_credential_rule(server: InProcessServer):
    status, _, _ = await server.http("GET", "/v1/models")
    assert status == 401
    status, _, body = await server.http("GET", "/v1/models", headers=BEARER)
    assert status == 200
    assert [entry["id"] for entry in json.loads(body)["data"]] == ["xtalk"]


async def test_unknown_model_cannot_open_a_session(server: InProcessServer):
    assert isinstance(await server.connect(query="model=unknown"), Closed)
