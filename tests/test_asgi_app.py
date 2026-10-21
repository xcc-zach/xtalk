"""Verify the public native ASGI entry point rejects obsolete surfaces."""

import asyncio
import pytest
from xtalk import Models, Service, Xtalk


def test_fixed_paths_models_and_single_lifespan_owner():
    """Reject old routes, unknown aliases and concurrent lifecycle ownership."""

    async def run():
        runtime = Xtalk(service_prototype=Service(models=Models()))
        app = runtime.create_asgi_app()
        assert app is runtime.create_asgi_app()
        with pytest.raises(TypeError):
            runtime.create_asgi_app(ws_path="/custom")
        assert not hasattr(runtime, "connect")
        assert not hasattr(runtime, "run")
        assert not hasattr(runtime, "mount_routes")
        assert not hasattr(runtime, "asgi_lifespan")
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        owner = asyncio.create_task(
            app({"type": "lifespan"}, incoming.get, outgoing.put)
        )
        await incoming.put({"type": "lifespan.startup"})
        assert (await outgoing.get())["type"] == "lifespan.startup.complete"
        second_in, second_out = asyncio.Queue(), asyncio.Queue()
        second = asyncio.create_task(
            app({"type": "lifespan"}, second_in.get, second_out.put)
        )
        await second_in.put({"type": "lifespan.startup"})
        assert (await second_out.get())["type"] == "lifespan.startup.failed"
        await second
        for path, query in [("/ws", b""), ("/v1/realtime", b"model=unknown")]:
            received = []

            async def send(message):
                received.append(message)

            async def receive():
                return {"type": "websocket.connect"}

            await app(
                {
                    "type": "websocket",
                    "path": path,
                    "query_string": query,
                    "headers": [],
                },
                receive,
                send,
            )
            assert received == [{"type": "websocket.close", "code": 1008}]
            assert runtime._service_manager.get_service_count() == 0
        await incoming.put({"type": "lifespan.shutdown"})
        await outgoing.get()
        await owner

    asyncio.run(run())
