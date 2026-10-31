"""A Realtime client that behaves like a conversation participant.

The client speaks test speech, sends text, listens to replies and acknowledges
playback the way the frontend SDK does. Tests read the ``Reply`` summaries and
the event log instead of handling protocol messages themselves.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

from . import speech
from .asgi import AsgiWebSocket, Closed

WIRE_SAMPLE_RATE = 24000
WIRE_BYTES_PER_MS = WIRE_SAMPLE_RATE * 2 // 1000
MESSAGE_MS = 20


class ConnectionClosedError(RuntimeError):
    """Raised when waiting on a connection the server has closed."""


@dataclass
class Reply:
    """What the participant observed for one server response."""

    response_id: str
    created_index: int
    status: str | None = None
    transcript: str = ""
    """Spoken text: the final transcript, or the text spoken so far when cut short."""
    audio_ms: float = 0.0
    heard_text: str | None = None
    heard_updates: list[str] = field(default_factory=list)
    transcript_final: bool = False
    stop_requested: bool = False
    done_index: int | None = None
    heard_index: int | None = None

    @property
    def done(self) -> bool:
        """Whether the response reached a terminal state."""
        return self.status is not None


@dataclass
class _Playback:
    queue: deque[float] = field(default_factory=deque)
    heard_ms: float = 0.0
    limit_ms: float | None = None
    audio_done: bool = False
    finished_sent: bool = False
    stopped: bool = False


class RealtimeTestClient:
    """Drive one Realtime session over an in-process or network transport.

    Parameters
    ----------
    websocket : AsgiWebSocket
        Connected transport.
    auto_play : bool, optional
        Treat received audio as heard immediately. ``False`` holds playback
        until :meth:`hear_up_to` or :meth:`resume`.
    respond_to_stop : bool, optional
        Report the unconfirmed heard duration when the server stops playback.
    """

    def __init__(
        self,
        websocket: AsgiWebSocket,
        *,
        auto_play: bool = True,
        respond_to_stop: bool = True,
    ) -> None:
        self._ws = websocket
        self._auto_play = auto_play
        self._respond_to_stop = respond_to_stop
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []
        self.session: dict[str, Any] | None = None
        self.closed: Closed | None = None
        self._cond = asyncio.Condition()
        self._cursor = 0
        self._replies: dict[str, Reply] = {}
        self._reply_order: list[Reply] = []
        self._reply_cursor = 0
        self._transcripts: list[tuple[int, str]] = []
        self._transcript_cursor = -1
        self._say_mark = -1
        self._playback: dict[str, _Playback] = {}
        self._default_limit: float | None = None if auto_play else 0.0
        self._pending_limit: float | None = None
        self._latest_response: str | None = None
        self._receiver: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------
    async def open(self, session_update: dict[str, Any] | None = None) -> None:
        """Wait for session readiness and optionally apply a session update."""
        self._receiver = asyncio.create_task(self._receive_loop())
        await self.wait_for("session.created")
        if session_update is not None:
            await self._ws.send_json({"type": "session.update", "session": session_update})
            await self.wait_for("session.updated")

    async def close(self) -> None:
        """Disconnect and stop listening."""
        if self._receiver is not None:
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
            self._receiver = None
        await self._ws.close()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    async def say(
        self,
        sentence_id: int,
        *,
        voiced_ms: float = 600.0,
        trailing_silence_ms: float = 700.0,
        binary: bool = False,
        realtime: bool = True,
    ) -> None:
        """Send one test sentence as 24 kHz audio.

        The trailing silence exceeds the server's end-of-speech window so the
        turn closes without further input.
        """
        self._say_mark = self.mark
        pcm = speech.encode(
            sentence_id,
            voiced_ms=voiced_ms,
            trailing_silence_ms=trailing_silence_ms,
            sample_rate=WIRE_SAMPLE_RATE,
        )
        await self._send_audio(pcm, binary=binary, realtime=realtime)

    async def silence(self, ms: float, *, realtime: bool = True) -> None:
        """Send only silence."""
        await self._send_audio(speech.silence(ms, sample_rate=WIRE_SAMPLE_RATE), binary=False, realtime=realtime)

    async def _send_audio(self, pcm: bytes, *, binary: bool, realtime: bool) -> None:
        step = WIRE_BYTES_PER_MS * MESSAGE_MS
        started = time.monotonic()
        for index, start in enumerate(range(0, len(pcm), step)):
            chunk = pcm[start : start + step]
            if binary:
                await self._ws.send_bytes(chunk)
            else:
                await self._ws.send_json(
                    {"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")}
                )
            if realtime:
                target = started + (index + 1) * MESSAGE_MS / 1000.0
                await asyncio.sleep(max(0.0, target - time.monotonic()))

    async def commit(self) -> None:
        """Commit the buffered audio turn."""
        await self._ws.send_json({"type": "input_audio_buffer.commit"})

    async def clear(self) -> None:
        """Clear the buffered audio."""
        await self._ws.send_json({"type": "input_audio_buffer.clear"})

    async def speech_start(self) -> None:
        """Notify the server that the client detected speech onset."""
        await self._ws.send_json({"type": "input_audio_buffer.speech_start"})

    async def send_text(
        self,
        text: str,
        *,
        generate_response: bool = True,
        item_id: str | None = None,
    ) -> str:
        """Send a user text item and return its item id."""
        item_id = item_id or f"item_client_{len(self.events)}_{id(self)}"
        message: dict[str, Any] = {
            "type": "conversation.item.create",
            "event_id": item_id,
            "item": {
                "id": item_id,
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }
        if generate_response:
            message["xtalk"] = {"generate_response": True}
        await self._ws.send_json(message)
        return item_id

    async def cancel(self, response_id: str | None = None) -> None:
        """Cancel the identified or active response."""
        message: dict[str, Any] = {"type": "response.cancel"}
        if response_id is not None:
            message["response_id"] = response_id
        await self._ws.send_json(message)

    async def send_raw(self, message: dict[str, Any]) -> None:
        """Send an arbitrary client event."""
        await self._ws.send_json(message)

    # ------------------------------------------------------------------
    # playback simulation
    # ------------------------------------------------------------------
    def hear_up_to(self, ms: float) -> None:
        """Pause playback once about ``ms`` milliseconds were heard.

        The limit applies to the current reply when one is active, otherwise
        to the next reply that starts.
        """
        if self._latest_response is not None and not self._playback_for(self._latest_response).audio_done:
            self._playback_for(self._latest_response).limit_ms = ms
            return
        self._pending_limit = ms

    async def resume(self) -> None:
        """Resume playback of the current reply without limit."""
        self._pending_limit = None
        if self._latest_response is not None:
            self._playback_for(self._latest_response).limit_ms = None
            await self._drain(self._latest_response)

    def heard_ms(self, response_id: str) -> float:
        """Return how much audio of one response was heard."""
        return self._playback_for(response_id).heard_ms

    async def wait_until_heard(self, response_id: str, ms: float, *, timeout: float = 5.0) -> None:
        """Wait until at least ``ms`` milliseconds of one response were heard."""
        state = self._playback_for(response_id)

        def _reached() -> bool:
            if state.heard_ms + 1e-6 >= ms:
                return True
            # Playback paused on a chunk boundary just below the limit.
            return bool(
                state.queue
                and state.limit_ms is not None
                and state.heard_ms + state.queue[0] > state.limit_ms + 1e-6
            )

        await self._wait_until(_reached, timeout, f"{ms} ms heard")

    def _playback_for(self, response_id: str) -> _Playback:
        state = self._playback.get(response_id)
        if state is None:
            limit = self._default_limit if self._pending_limit is None else self._pending_limit
            self._pending_limit = None
            state = _Playback(limit_ms=limit)
            self._playback[response_id] = state
        return state

    async def _drain(self, response_id: str) -> None:
        state = self._playback_for(response_id)
        while state.queue and not state.stopped:
            chunk_ms = state.queue[0]
            if state.limit_ms is not None and state.heard_ms + chunk_ms > state.limit_ms + 1e-6:
                break
            state.queue.popleft()
            state.heard_ms += chunk_ms
            await self._ws.send_json(
                {"type": "response.output_audio.playback.chunk_played", "response_id": response_id}
            )
        if state.audio_done and not state.queue and not state.finished_sent and not state.stopped:
            state.finished_sent = True
            await self._ws.send_json(
                {"type": "response.output_audio.playback.finished", "response_id": response_id}
            )

    async def _on_stop(self, response_id: str) -> None:
        state = self._playback_for(response_id)
        state.stopped = True
        unconfirmed = 0.0
        if state.queue and state.limit_ms is not None:
            unconfirmed = min(state.queue[0], max(0.0, state.limit_ms - state.heard_ms))
        state.heard_ms += unconfirmed
        state.queue.clear()
        if self._respond_to_stop:
            await self._ws.send_json(
                {
                    "type": "response.output_audio.playback.stopped",
                    "response_id": response_id,
                    "played_audio_ms": unconfirmed,
                }
            )

    # ------------------------------------------------------------------
    # receiving
    # ------------------------------------------------------------------
    async def _receive_loop(self) -> None:
        while True:
            message = await self._ws.receive()
            if isinstance(message, Closed):
                async with self._cond:
                    self.closed = message
                    self._cond.notify_all()
                return
            event = self._parse_binary(message) if isinstance(message, bytes) else message
            await self._handle(event)
            # Let waiting test code observe each event before the next one.
            await asyncio.sleep(0)

    @staticmethod
    def _parse_binary(message: bytes) -> dict[str, Any]:
        length = int.from_bytes(message[:4], "big")
        header = json.loads(message[4 : 4 + length].decode("utf-8"))
        return {**header, "pcm": message[4 + length :]}

    async def _handle(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        async with self._cond:
            event["_i"] = len(self.events)
            if kind == "response.created":
                response_id = event["response"]["id"]
                reply = Reply(response_id=response_id, created_index=event["_i"])
                self._replies[response_id] = reply
                self._reply_order.append(reply)
                self._latest_response = response_id
                self._playback_for(response_id)
            reply = self._replies.get(str(event.get("response_id") or ""))
            if kind in {"response.output_audio.delta", "xtalk.output_audio.delta"} and reply:
                pcm = event["pcm"] if "pcm" in event else base64.b64decode(event["delta"])
                reply.audio_ms += len(pcm) / WIRE_BYTES_PER_MS
                self._playback_for(reply.response_id).queue.append(len(pcm) / WIRE_BYTES_PER_MS)
            elif kind == "response.output_audio.playback.updated" and reply:
                event["_heard_ms"] = self._playback_for(reply.response_id).heard_ms
                reply.heard_updates.append(str(event.get("text", "")))
            elif kind == "response.output_audio.playback.done" and reply:
                reply.heard_text = str(event.get("text", ""))
                reply.heard_index = event["_i"]
            elif kind == "response.output_audio.playback.stop" and reply:
                reply.stop_requested = True
            elif kind == "response.output_audio_transcript.delta" and reply and not reply.transcript_final:
                reply.transcript += str(event.get("delta", ""))
            elif kind == "response.output_audio_transcript.done" and reply:
                reply.transcript = str(event.get("transcript", ""))
                reply.transcript_final = True
            elif kind == "response.output_text.delta" and reply and not reply.transcript_final:
                reply.transcript += str(event.get("delta", ""))
            elif kind == "response.output_text.done" and reply:
                reply.transcript = str(event.get("text", ""))
                reply.transcript_final = True
            elif kind == "response.done":
                target = self._replies.get(event["response"]["id"])
                if target:
                    target.status = event["response"]["status"]
                    target.done_index = event["_i"]
            elif kind == "conversation.item.input_audio_transcription.completed":
                self._transcripts.append((event["_i"], str(event.get("transcript", ""))))
            elif kind == "session.created":
                self.session = event["session"]
            elif kind == "error":
                self.errors.append(event)
            self.events.append(event)
            self._cond.notify_all()
        if reply and kind in {"response.output_audio.delta", "xtalk.output_audio.delta"}:
            await self._drain(reply.response_id)
        elif reply and kind == "response.output_audio.done":
            self._playback_for(reply.response_id).audio_done = True
            await self._drain(reply.response_id)
        elif reply and kind == "response.output_audio.playback.stop":
            await self._on_stop(reply.response_id)

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def find(
        self,
        kind: str,
        *,
        where: Callable[[dict[str, Any]], bool] | None = None,
        after: int = -1,
        before: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return logged events of ``kind`` within an index window."""
        end = len(self.events) if before is None else before
        return [
            event
            for event in self.events[after + 1 : end]
            if event.get("type") == kind and (where is None or where(event))
        ]

    async def wait_for(
        self,
        kind: str,
        *,
        where: Callable[[dict[str, Any]], bool] | None = None,
        after: int | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        """Wait for the first matching event after the cursor and return it."""
        start = self._cursor if after is None else after + 1
        deadline = time.monotonic() + timeout
        async with self._cond:
            while True:
                for index in range(start, len(self.events)):
                    event = self.events[index]
                    if event.get("type") == kind and (where is None or where(event)):
                        self._cursor = index + 1
                        return event
                start = len(self.events)
                if self.closed is not None:
                    raise ConnectionClosedError(f"connection closed with {self.closed.code}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no {kind} event within {timeout}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"no {kind} event within {timeout}s") from None

    async def wait_reply(self, *, after: int | None = None, timeout: float = 10.0) -> Reply:
        """Wait for the next response to reach a terminal state."""
        deadline = time.monotonic() + timeout
        async with self._cond:
            while True:
                candidates = [
                    reply
                    for reply in self._reply_order[self._reply_cursor :]
                    if after is None or reply.created_index > after
                ]
                if candidates and candidates[0].done:
                    reply = candidates[0]
                    self._reply_cursor = self._reply_order.index(reply) + 1
                    return reply
                if self.closed is not None:
                    raise ConnectionClosedError(f"connection closed with {self.closed.code}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no finished reply within {timeout}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"no finished reply within {timeout}s") from None

    async def wait_heard(self, reply: Reply, *, timeout: float = 5.0) -> str:
        """Wait until the playback settlement text of ``reply`` is known."""
        await self._wait_until(lambda: reply.heard_text is not None, timeout, "playback settlement")
        return reply.heard_text or ""

    async def wait_transcript(self, *, timeout: float = 5.0) -> str:
        """Wait for the final transcript of the most recent ``say``."""

        def _pending() -> bool:
            return any(
                index > self._say_mark and index > self._transcript_cursor
                for index, _ in self._transcripts
            )

        await self._wait_until(_pending, timeout, "transcript")
        index, transcript = next(
            item
            for item in self._transcripts
            if item[0] > self._say_mark and item[0] > self._transcript_cursor
        )
        self._transcript_cursor = index
        return transcript

    async def _wait_until(self, predicate: Callable[[], bool], timeout: float, what: str) -> None:
        deadline = time.monotonic() + timeout
        async with self._cond:
            while not predicate():
                if self.closed is not None:
                    raise ConnectionClosedError(f"connection closed with {self.closed.code}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"no {what} within {timeout}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), remaining)
                except asyncio.TimeoutError:
                    raise TimeoutError(f"no {what} within {timeout}s") from None

    @property
    def replies(self) -> list[Reply]:
        """Replies that reached a terminal state, in creation order."""
        return [reply for reply in self._reply_order if reply.done]

    @property
    def mark(self) -> int:
        """Index of the last logged event, for ``after=`` arguments."""
        return len(self.events) - 1
