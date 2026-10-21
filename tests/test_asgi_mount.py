"""Exercise prefix mounting and resource ownership with a plain ASGI host."""

import asyncio
import json
import pytest
from xtalk import Models, Service, Xtalk


@pytest.mark.parametrize("prefix", ["", "/speech"])
def test_framework_free_mount_requires_lifespan_and_closes_active_connections(prefix):
    """Resolve root_path correctly and release live sessions during shutdown."""

    async def run():
        runtime = Xtalk(service_prototype=Service(models=Models()))
        app = runtime.create_asgi_app()

        async def host(scope, receive, send):
            await app({**scope, "root_path": prefix}, receive, send)

        async def http(path):
            messages = []

            async def send(message):
                messages.append(message)

            await host(
                {"type": "http", "path": prefix + path, "method": "GET", "headers": []},
                asyncio.Queue().get,
                send,
            )
            return messages

        assert (await http("/v1/models"))[0]["status"] == 503
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        lifecycle = asyncio.create_task(
            host({"type": "lifespan"}, incoming.get, outgoing.put)
        )
        await incoming.put({"type": "lifespan.startup"})
        assert (await outgoing.get())["type"] == "lifespan.startup.complete"
        assert (
            json.loads((await http("/v1/models"))[1]["body"])["data"][0]["id"]
            == "xtalk"
        )
        assert (await http("/api/sessions"))[0]["status"] == 404
        ws_in, ws_out = asyncio.Queue(), asyncio.Queue()
        await ws_in.put({"type": "websocket.connect"})
        ws = asyncio.create_task(
            host(
                {
                    "type": "websocket",
                    "path": prefix + "/v1/realtime",
                    "query_string": b"",
                    "headers": [],
                },
                ws_in.get,
                ws_out.put,
            )
        )
        assert (await ws_out.get())["type"] == "websocket.accept"
        assert json.loads((await ws_out.get())["text"])["type"] == "session.created"
        assert runtime._service_manager.get_service_count() == 1
        await incoming.put({"type": "lifespan.shutdown"})
        assert (await asyncio.wait_for(outgoing.get(), 2))[
            "type"
        ] == "lifespan.shutdown.complete"
        await lifecycle
        await asyncio.gather(ws, return_exceptions=True)
        assert runtime._service_manager.get_service_count() == 0
        assert (await http("/v1/models"))[0]["status"] == 503
        restart = asyncio.create_task(
            app({"type": "lifespan"}, incoming.get, outgoing.put)
        )
        await incoming.put({"type": "lifespan.startup"})
        assert (await outgoing.get())["type"] == "lifespan.startup.failed"
        await restart

    asyncio.run(run())
