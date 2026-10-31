"""OpenAI Live primary-WebSocket adapter."""

from __future__ import annotations

import asyncio
import base64
import binascii
import copy
import json
import math
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from ..registry import model
from .interfaces import AudioProfile, EndToEnd, EndToEndCloseResult, EndToEndModelEvent


@model
class GPTLive(EndToEnd):
    """Connect one X-Talk session to OpenAI's continuous Live protocol.

    Parameters
    ----------
    model : str
        Live voice model identifier.
    api_key : str or None
        Key supplied by ``end_to_end.params.api_key``; required when opening.
    base_url : str
        Complete primary WebSocket endpoint.
    voice : str
        Startup voice identifier.
    instructions : str
        Startup conversation instructions.
    delegation : dict
        Explicit Responses backend configuration, without application tools.
    connect_timeout_s, command_timeout_s, close_timeout_s : float
        Bounds for startup, command acknowledgments, and finalization.
    sample_rate : int
        Shared PCM sample rate, 16000 or 24000.
    max_pending_audio_ms : float
        Maximum decoded audio awaiting the normalized event consumer.
    max_pending_events : int
        Maximum pending normalized events and outbound messages.
    """

    def __init__(
        self,
        model: str = "gpt-live-1",
        api_key: str | None = None,
        base_url: str = "wss://api.openai.com/v1/live/sessions",
        voice: str = "marin",
        instructions: str = "Speak naturally and concisely.",
        delegation: dict[str, Any] | None = None,
        connect_timeout_s: float = 15,
        command_timeout_s: float = 15,
        close_timeout_s: float = 15,
        max_pending_audio_ms: float = 5000,
        sample_rate: int = 24000,
        max_pending_events: int = 256,
    ) -> None:
        if sample_rate not in {16000, 24000}:
            raise ValueError("Live sample_rate must be 16000 or 24000")
        endpoint = urlsplit(base_url)
        if endpoint.scheme != "wss" and not (
            endpoint.scheme == "ws"
            and endpoint.hostname in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError(
                "Live base_url requires wss, or ws on a loopback test server"
            )
        if (
            endpoint.query
            or endpoint.fragment
            or endpoint.username
            or not endpoint.hostname
        ):
            raise ValueError(
                "Live base_url must be a complete endpoint without query or credentials"
            )
        if not delegation or delegation.get("type") != "responses":
            raise ValueError(
                "GPTLive requires explicit Responses delegation in this release"
            )
        backend = delegation.get("responses", {})
        if (
            not backend.get("model")
            or backend.get("tools", [])
            or backend.get("tool_choice") != "none"
        ):
            raise ValueError(
                "Configure a Responses model, tools: [], and tool_choice: none"
            )
        for value in (
            connect_timeout_s,
            command_timeout_s,
            close_timeout_s,
            max_pending_audio_ms,
            max_pending_events,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(
                    "Live timeout and buffer limits must be finite and positive"
                )
        if type(max_pending_events) is not int:
            raise ValueError("max_pending_events must be a positive integer")
        if (
            not isinstance(instructions, str)
            or len(instructions.encode("utf-8")) > 16384
        ):
            raise ValueError(
                "Live instructions exceed the conservative 16 KiB local limit"
            )
        self._config = {
            "sample_rate": sample_rate,
            "model": model,
            "api_key": api_key,
            "base_url": base_url,
            "voice": voice,
            "instructions": instructions,
            "delegation": copy.deepcopy(delegation),
            "connect_timeout_s": connect_timeout_s,
            "command_timeout_s": command_timeout_s,
            "close_timeout_s": close_timeout_s,
            "max_pending_audio_ms": max_pending_audio_ms,
            "max_pending_events": int(max_pending_events),
        }
        self._session: aiohttp.ClientSession | None = None
        self._socket: aiohttp.ClientWebSocketResponse | None = None
        self._reader: asyncio.Task | None = None
        self._writer: asyncio.Task | None = None
        self._queue: asyncio.Queue[EndToEndModelEvent] = asyncio.Queue(
            int(max_pending_events)
        )
        self._writes: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            int(max_pending_events)
        )
        self._started = asyncio.Event()
        self._was_started = False
        self._closed = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._commands: dict[str, tuple[str, Any, asyncio.TimerHandle]] = {}
        self._audio_bytes = 0
        self._write_audio_bytes = 0
        self._pending_byte = b""
        self._failure = ""
        self._closing = False
        self._consuming = False
        self._voice_usage: dict[str, Any] = {}
        self._backend_usage: list[dict[str, Any]] = []
        self._result: EndToEndCloseResult | None = None

    def clone(self) -> GPTLive:
        """Create a fresh runtime, excluding history and network resources."""
        return GPTLive(**copy.deepcopy(self._config))

    @property
    def metadata(self) -> dict[str, str]:
        """Return the model and fixed voice while excluding credentials."""
        return {
            "provider": "openai",
            "model": self._config["model"],
            "voice": self._config["voice"],
        }

    @property
    def audio_profile(self) -> AudioProfile:
        """Return the fixed PCM format used in both directions."""
        return AudioProfile(sample_rate=self._config["sample_rate"])

    async def open(self) -> None:
        """Start the sole reader/writer and await a bounded startup acknowledgment."""
        if self._session is not None or self._closing:
            raise RuntimeError("Create a new GPTLive instance for a new connection")
        key = self._config["api_key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError(
                "GPTLive api_key is required; set end_to_end.params.api_key in config"
            )
        self._session = aiohttp.ClientSession(trust_env=True)
        try:
            async with _deadline(self._config["connect_timeout_s"]):
                self._socket = await self._session.ws_connect(
                    self._config["base_url"],
                    headers={"Authorization": f"Bearer {key}"},
                    max_msg_size=1024 * 1024,
                    heartbeat=20,
                )
                self._reader = asyncio.create_task(self._read(), name="gpt-live-reader")
                self._writer = asyncio.create_task(
                    self._write(), name="gpt-live-writer"
                )
                self._enqueue(
                    {
                        "type": "session.start",
                        "event_id": uuid.uuid4().hex,
                        "session": {
                            "model": self._config["model"],
                            "instructions": self._config["instructions"],
                            "audio": {
                                "format": {
                                    "type": "audio/pcm",
                                    "rate": self.audio_profile.sample_rate,
                                },
                                "output": {"voice": self._config["voice"]},
                            },
                            "delegation": self._config["delegation"],
                        },
                    }
                )
                await self._started.wait()
                if self._failure:
                    raise RuntimeError(self._failure)
        except BaseException:
            await self.close()
            raise

    def _enqueue(self, message: dict[str, Any]) -> None:
        self._writes.put_nowait(message)

    async def append_audio(self, pcm: bytes) -> None:
        """Queue ordered PCM with a finite unsent-audio budget."""
        if not self._started.is_set() or self._closing or self._failure:
            raise RuntimeError("Live session is not active")
        if len(pcm) % 2:
            raise ValueError("PCM16 input must contain an even number of bytes")
        if not pcm:
            return
        if (
            self._write_audio_bytes + len(pcm)
            > self.audio_profile.sample_rate
            * 2
            * self._config["max_pending_audio_ms"]
            / 1000
        ):
            raise BufferError("Live input queue exceeded its audio budget")
        self._enqueue(
            {
                "type": "session.input_audio.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        self._write_audio_bytes += len(pcm)

    async def set_input_muted(self, muted: bool) -> None:
        """Enqueue mute/unmute; expose its acknowledgment as a control event."""
        await self._command(
            "session.input_audio.mute" if muted else "session.input_audio.unmute",
            "input_muted",
            muted,
        )

    async def _command(
        self, event_type: str, kind: str, value: Any, **fields: Any
    ) -> None:
        if not self._started.is_set() or self._closing:
            raise RuntimeError("Live session is not active")
        if len(self._commands) >= 128:
            raise BufferError("Live command queue is full")
        event_id = uuid.uuid4().hex
        timer = asyncio.get_running_loop().call_later(
            self._config["command_timeout_s"], self._command_timeout, event_id
        )
        try:
            self._enqueue({"type": event_type, "event_id": event_id, **fields})
        except asyncio.QueueFull:
            timer.cancel()
            raise BufferError("Live command queue is full") from None
        self._commands[event_id] = (kind, value, timer)

    def _command_timeout(self, event_id: str) -> None:
        if self._commands.pop(event_id, None):
            self._emit(
                EndToEndModelEvent(
                    "error",
                    {
                        "code": "command_timeout",
                        "message": "Live control acknowledgment timed out",
                        "fatal": False,
                    },
                )
            )

    def _emit(self, event: EndToEndModelEvent) -> None:
        if event.kind == "audio" and (
            self._audio_bytes + len(event.audio)
            > self.audio_profile.sample_rate
            * 2
            * self._config["max_pending_audio_ms"]
            / 1000
        ):
            self._fail("backpressure", "Live output consumer exceeded its audio budget")
            return
        try:
            self._queue.put_nowait(event)
            self._audio_bytes += len(event.audio)
        except asyncio.QueueFull:
            self._fail("backpressure", "Live event consumer exceeded its queue budget")

    def _fail(self, code: str, message: str) -> None:
        if self._failure:
            return
        self._failure = message
        self._started.set()
        if self._queue.full():
            discarded = self._queue.get_nowait()
            self._audio_bytes -= len(discarded.audio)
        self._queue.put_nowait(
            EndToEndModelEvent(
                "error", {"code": code, "message": message, "fatal": True}
            )
        )
        self._closed.set()

    async def _write(self) -> None:
        audio_deadline = time.monotonic()
        try:
            while True:
                message = await self._writes.get()
                if self._socket is None:
                    return
                if message["type"] == "session.input_audio.append":
                    size = len(base64.b64decode(message["audio"]))
                    self._write_audio_bytes -= size
                    if self._closing:
                        continue
                    await asyncio.sleep(max(0, audio_deadline - time.monotonic()))
                    if self._closing:
                        continue
                    await self._socket.send_json(message)
                    audio_deadline = max(audio_deadline, time.monotonic()) + size / (
                        self.audio_profile.sample_rate * 2
                    )
                else:
                    await self._socket.send_json(message)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - owned task boundary must normalize failures
            self._fail("transport", "Live upstream write failed")

    async def _read(self) -> None:
        try:
            assert self._socket is not None
            async for packet in self._socket:
                if packet.type == aiohttp.WSMsgType.TEXT:
                    self._receive(json.loads(packet.data))
                    if self._result is not None:
                        break
                elif packet.type == aiohttp.WSMsgType.ERROR:
                    self._fail("transport", "Live upstream transport failed")
                    break
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError, binascii.Error):
            self._fail("protocol", "Malformed Live event or PCM payload")
        except Exception:  # noqa: BLE001 - owned task boundary must normalize failures
            self._fail("transport", "Live upstream receive failed")
        finally:
            if self._result is None and not self._failure:
                self._fail("transport", "Live connection ended without session.closed")
            self._closed.set()

    def _receive(self, event: dict[str, Any]) -> None:
        kind = event.get("type", "")
        if kind == "session.started":
            self._was_started = True
            self._started.set()
        elif kind == "session.output_audio.delta":
            pcm = self._pending_byte + base64.b64decode(event["delta"], validate=True)
            complete = len(pcm) - len(pcm) % 2
            self._pending_byte = pcm[complete:]
            if complete:
                self._emit(EndToEndModelEvent("audio", audio=pcm[:complete]))
        elif kind in {
            "session.input_transcript.delta",
            "session.output_transcript.delta",
        }:
            text, start, end = event["delta"], event["start_ms"], event["end_ms"]
            if not isinstance(text, str) or not (0 <= start <= end):
                raise ValueError("Invalid transcript")
            self._emit(
                EndToEndModelEvent(
                    "transcript",
                    {
                        "speaker": "user"
                        if kind == "session.input_transcript.delta"
                        else "assistant",
                        "delta": text,
                        "start_ms": start,
                        "end_ms": end,
                        "provider_event_id": event.get("event_id") or None,
                    },
                )
            )
        elif kind == "session.closed":
            if self._pending_byte:
                raise ValueError("Odd terminal PCM payload")
            self._voice_usage = event.get("usage", self._voice_usage)
            self._result = EndToEndCloseResult(
                "complete",
                str(event.get("reason", ""))[:100],
                self._voice_usage,
                self._backend_usage,
            )
            self._closed.set()
            self._emit(
                EndToEndModelEvent(
                    "closed",
                    {
                        "finalization": "complete",
                        "usage": self._voice_usage,
                        "reason": self._result.reason,
                    },
                )
            )
        elif kind == "error":
            error = event.get("error", event)
            code = str(error.get("code", "provider_error"))[:100]
            command_id = event.get("client_event_id") or error.get("client_event_id")
            command = self._commands.pop(command_id, None)
            if command:
                command[2].cancel()
            # Commands and speech moderation can be rejected within a healthy session.
            fatal = not self._was_started and not command
            message = (
                "Live startup was rejected"
                if fatal
                else "Live command was rejected"
                if command
                else "Live reported an audio or backend error"
            )
            if fatal:
                self._fail(code, message)
            else:
                self._emit(
                    EndToEndModelEvent(
                        "error",
                        {
                            "code": code,
                            "message": message,
                            "fatal": False,
                        },
                    )
                )
        elif event.get("client_event_id") in self._commands:
            event_id = event["client_event_id"]
            control, value, timer = self._commands.pop(event_id)
            timer.cancel()
            self._emit(
                EndToEndModelEvent(
                    "control_ack",
                    {"command_id": event_id, "control": control, "value": value},
                )
            )
        elif kind in {"response.event", "session.delegation.created"}:
            nested = event.get("event", event)
            nested_type = nested.get("type", "")
            self._emit(
                EndToEndModelEvent(
                    "backend_activity",
                    {
                        "status": "idle"
                        if nested_type
                        in {
                            "response.completed",
                            "response.failed",
                            "response.cancelled",
                        }
                        else "working",
                        "delegation_id": event.get("delegation_id"),
                        "response_id": nested.get("response_id"),
                    },
                )
            )
            response = nested.get("response", {})
            items = [nested.get("item", {})] + response.get("output", [])
            if any(item.get("type") == "function_call" for item in items):
                self._emit(
                    EndToEndModelEvent(
                        "error",
                        {
                            "code": "unsupported_capability",
                            "message": "Application tools are disabled for this Live session",
                            "fatal": False,
                        },
                    )
                )
            if response.get("usage"):
                usage = {"response_id": response.get("id"), "usage": response["usage"]}
                self._backend_usage.append(usage)
                self._backend_usage = self._backend_usage[-256:]
                self._emit(EndToEndModelEvent("usage", {"scope": "backend", **usage}))
        elif "usage" in event:
            self._voice_usage = event["usage"]
            self._emit(
                EndToEndModelEvent(
                    "usage", {"scope": "voice", "usage": self._voice_usage}
                )
            )

    async def events(self) -> AsyncIterator[EndToEndModelEvent]:
        """Yield events to one consumer while independently receiving controls."""
        if self._consuming:
            raise RuntimeError("Live events allow only one consumer")
        self._consuming = True
        while True:
            event = await self._queue.get()
            self._audio_bytes -= len(event.audio)
            yield event
            if event.kind == "closed" or (
                event.kind == "error" and event.data.get("fatal")
            ):
                break

    async def close(self) -> EndToEndCloseResult:
        """Keep receiving until terminal usage or a bounded close deadline."""
        async with self._close_lock:
            if self._result is not None and self._session is None:
                return self._result
            self._closing = True
            started = self._was_started
            while not self._writes.empty():
                self._writes.get_nowait()
            self._write_audio_bytes = 0
            if (
                started
                and not self._failure
                and self._socket is not None
                and not self._socket.closed
                and self._result is None
            ):
                try:
                    self._enqueue({"type": "session.close"})
                    await asyncio.wait_for(
                        self._closed.wait(), self._config["close_timeout_s"]
                    )
                except asyncio.TimeoutError:
                    self._failure = "Live finalization timed out"
                except asyncio.QueueFull:
                    self._failure = "Live close queue is full"
            for _, _, timer in self._commands.values():
                timer.cancel()
            self._commands.clear()
            tasks = [task for task in (self._reader, self._writer) if task is not None]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self._socket is not None:
                await self._socket.close()
            if self._session is not None:
                await self._session.close()
                self._session = None
            if self._result is None:
                self._result = EndToEndCloseResult(
                    "incomplete" if started else "not_started",
                    self._failure,
                    self._voice_usage,
                    self._backend_usage,
                )
            return self._result


class _deadline:
    """Bound an async block on Python 3.10 without third-party helpers."""

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    async def __aenter__(self):
        self.task = asyncio.current_task()
        self.expired = False
        self.timer = asyncio.get_running_loop().call_later(self.timeout, self._expire)

    def _expire(self) -> None:
        self.expired = True
        self.task.cancel()

    async def __aexit__(self, exc_type, exc, traceback):
        self.timer.cancel()
        if self.expired and exc_type is asyncio.CancelledError:
            raise asyncio.TimeoutError("Live startup timed out") from None
