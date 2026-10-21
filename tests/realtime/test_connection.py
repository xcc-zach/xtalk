"""Verify transport failure cancels a blocked input loop without a second close."""

import asyncio
from xtalk.realtime.connection import RealtimeConnection


def test_sender_failure_wakes_receiver_and_skips_duplicate_close():
    async def run():
        incoming = asyncio.Queue()
        sent = []

        async def send(message):
            sent.append(message)
            if message["type"] == "websocket.send":
                raise OSError("peer disconnected")

        connection = RealtimeConnection(incoming.get, send)
        await connection.accept()
        await connection.send_json({"type": "example"})
        disconnected = await asyncio.wait_for(connection.receive(), 1)
        assert disconnected == {"type": "websocket.disconnect", "code": 1006}
        await connection.close()
        assert [message["type"] for message in sent] == [
            "websocket.accept",
            "websocket.send",
        ]

    asyncio.run(run())


def test_json_and_binary_share_order_and_budget():
    """Preserve message types and ordering, counting raw binary bytes on overflow."""
    async def run():
        incoming = asyncio.Queue()
        sent = []
        blocked = asyncio.Event()

        async def send(message):
            sent.append(message)
            if message.get("text") == '{"type": "blocked"}':
                await blocked.wait()

        connection = RealtimeConnection(incoming.get, send)
        await connection.accept()
        assert not hasattr(connection, "send_text")
        await connection.send_json({"type": "first"})
        await connection.send_bytes(b"\x00\x01")
        await connection.send_json({"type": "last"})
        await connection._queue.join()
        assert [message.get("bytes", message.get("text")) for message in sent[1:]] == [
            '{"type": "first"}', b"\x00\x01", '{"type": "last"}',
        ]
        await connection.send_json({"type": "blocked"})
        await asyncio.sleep(0)
        try:
            await connection.send_bytes(b"0" * (8 * 1024 * 1024))
        except ConnectionError:
            pass
        else:
            raise AssertionError("Shared byte budget must include the pending text")
        assert sent[-1] == {"type": "websocket.close", "code": 1013, "reason": ""}
        assert connection._sender.done()
        await connection.close()
    asyncio.run(run())
