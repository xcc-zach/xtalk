"""Exercise native ASGI and temporary sessions without model dependencies."""

import asyncio
import json
import pytest
from xtalk import Xtalk, Models, Service


def test_models_auth_and_lifecycle():
    async def scenario():
        runtime = Xtalk(
            service_prototype=Service(
                models=Models(), service_config={"realtime_api": {"api_keys": ["test"]}}
            )
        )
        app = runtime.create_asgi_app()
        assert app is runtime.create_asgi_app()
        incoming = asyncio.Queue()
        outgoing = asyncio.Queue()
        task = asyncio.create_task(
            app({"type": "lifespan"}, incoming.get, outgoing.put)
        )
        await incoming.put({"type": "lifespan.startup"})
        assert (await outgoing.get())["type"] == "lifespan.startup.complete"
        for header, status in [(b"", 401), (b"Bearer test", 200)]:
            output = []

            async def send(message):
                output.append(message)

            await app(
                {
                    "type": "http",
                    "method": "GET",
                    "path": "/v1/models",
                    "headers": [(b"authorization", header)],
                },
                incoming.get,
                send,
            )
            assert output[0]["status"] == status
            if status == 200:
                assert json.loads(output[1]["body"])["data"][0]["id"] == "xtalk"
        await incoming.put({"type": "lifespan.shutdown"})
        assert (await outgoing.get())["type"] == "lifespan.shutdown.complete"
        await task

    asyncio.run(scenario())


def test_websocket_protocol_and_custom_registration():
    async def scenario():
        prototype = Service(models=Models())

        async def handler(gateway, message):
            await gateway.websocket.send_json(
                {"type": "custom.reply", "text": message["text"]}
            )

        prototype.register_input_handler(type="custom.request", handler=handler)
        app = Xtalk(service_prototype=prototype).create_asgi_app()
        lifecycle = asyncio.Queue()
        lifecycle_out = asyncio.Queue()
        task = asyncio.create_task(
            app({"type": "lifespan"}, lifecycle.get, lifecycle_out.put)
        )
        await lifecycle.put({"type": "lifespan.startup"})
        await lifecycle_out.get()
        incoming, outgoing = asyncio.Queue(), asyncio.Queue()
        await incoming.put({"type": "websocket.connect"})
        session = asyncio.create_task(
            app(
                {
                    "type": "websocket",
                    "path": "/v1/realtime",
                    "headers": [],
                    "query_string": b"",
                },
                incoming.get,
                outgoing.put,
            )
        )
        assert (await outgoing.get())["type"] == "websocket.accept"
        created = json.loads((await outgoing.get())["text"])
        assert created["type"] == "session.created"
        for payload, expected in [
            ({"type": "response.create", "event_id": "bad"}, "error"),
            ({"type": "custom.request", "text": "hello"}, "custom.reply"),
        ]:
            await incoming.put(
                {"type": "websocket.receive", "text": json.dumps(payload)}
            )
            result = json.loads((await asyncio.wait_for(outgoing.get(), 2))["text"])
            assert result["type"] == expected
        await incoming.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(session, 2)
        await lifecycle.put({"type": "lifespan.shutdown"})
        await lifecycle_out.get()
        await task

    asyncio.run(scenario())
