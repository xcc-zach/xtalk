"""Native ASGI WebSocket transport with bounded ordered output."""

import asyncio
import json


class RealtimeConnection:
    """Adapt one ASGI WebSocket to the runtime transport contract.

    Parameters
    ----------
    receive, send : callable
        ASGI receive and send functions.
    """

    def __init__(self, receive, send):
        """Initialize session-scoped transport and protocol state."""
        self._receive = receive
        self._send = send
        self._queue = asyncio.Queue()
        self._queued_bytes = 0
        self._closed = False
        self._sender = None
        self._disconnected = False

    async def accept(self):
        """Accept the connection and start its ordered output task."""
        await self._send({"type": "websocket.accept"})
        self._sender = asyncio.create_task(self._drain())

    async def receive(self):
        """Receive one ASGI WebSocket message."""
        receiving = asyncio.create_task(self._receive())
        try:
            done, _ = await asyncio.wait(
                (receiving, self._sender), return_when=asyncio.FIRST_COMPLETED
            )
            if self._sender in done:
                try:
                    self._sender.result()
                except ConnectionError:
                    return {"type": "websocket.disconnect", "code": 1006}
                raise ConnectionError("Realtime output sender stopped")
            message = receiving.result()
        finally:
            if not receiving.done():
                receiving.cancel()
                await asyncio.gather(receiving, return_exceptions=True)
        if message["type"] == "websocket.disconnect":
            self._disconnected = True
        return message

    async def send_json(self, message):
        """Queue JSON data within an eight MiB connection budget.

        Parameters
        ----------
        message : dict
            Outbound JSON event.
        """
        await self._enqueue(json.dumps(message))

    async def send_bytes(self, data: bytes) -> None:
        """Queue binary data within the shared connection budget.

        Parameters
        ----------
        data : bytes
            Complete binary protocol message.
        """
        await self._enqueue(data)

    async def _enqueue(self, payload: str | bytes) -> None:
        """Queue one complete message in transport order."""
        size = len(payload.encode()) if isinstance(payload, str) else len(payload)
        if self._closed:
            raise ConnectionError("Realtime connection is closed")
        if self._sender.done():
            self._sender.result()
            raise ConnectionError("Realtime output sender stopped")
        if self._queued_bytes + size > 8 * 1024 * 1024:
            await self.close(code=1013)
            raise ConnectionError("Realtime output queue exceeded its byte budget")
        self._queued_bytes += size
        self._queue.put_nowait((payload, size))

    async def _drain(self):
        """Send queued output in order and propagate transport failure."""
        while True:
            payload, size = await self._queue.get()
            try:
                key = "text" if isinstance(payload, str) else "bytes"
                await self._send({"type": "websocket.send", key: payload})
            except (OSError, RuntimeError) as exc:
                self._disconnected = True
                raise ConnectionError("Realtime output transport closed") from exc
            finally:
                self._queued_bytes -= size
                self._queue.task_done()

    async def close(self, code=1000, reason=""):
        """Close the transport and stop its sender.

        Parameters
        ----------
        code : int
            WebSocket close code.
        reason : str
            Public reason without credentials.
        """
        if self._closed:
            return
        self._closed = True
        if self._sender:
            self._sender.cancel()
            await asyncio.gather(self._sender, return_exceptions=True)
        if not self._disconnected:
            await self._send(
                {"type": "websocket.close", "code": code, "reason": reason}
            )
