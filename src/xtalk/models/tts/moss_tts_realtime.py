"""Client adapter for the MOSS-TTS-Realtime WebSocket service."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from ..registry import model
from .interfaces import StreamingTextTTS, TTS

__all__ = ["MossTTSRealtime"]

_AUDIO_STREAM_END = object()

logger = logging.getLogger(__name__)


def _websocket_url(base_url: str) -> str:
    """Build the MOSS-TTS-Realtime WebSocket endpoint from a base URL."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must not be empty")

    parsed = urlsplit(normalized)
    schemes = {
        "http": "ws",
        "https": "wss",
        "ws": "ws",
        "wss": "wss",
    }
    websocket_scheme = schemes.get(parsed.scheme.lower())
    if websocket_scheme is None or not parsed.netloc:
        raise ValueError(
            "base_url must be an absolute http, https, ws, or wss URL"
        )

    path = parsed.path.rstrip("/")
    if not path.endswith("/tts/ws"):
        path = f"{path}/tts/ws"
    return urlunsplit(
        (websocket_scheme, parsed.netloc, path, parsed.query, "")
    )


def _decode_control_event(payload: str) -> dict[str, Any]:
    """Decode and validate a JSON control event from the service."""
    event = json.loads(payload)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RuntimeError(f"Invalid MOSS-TTS control event: {event!r}")
    if event["type"] == "error":
        detail = event.get("detail", "unknown service error")
        raise RuntimeError(f"MOSS-TTS-Realtime service error: {detail}")
    return event


@model
class MossTTSRealtime(TTS, StreamingTextTTS):
    """Adapt the MOSS-TTS-Realtime WebSocket service to xtalk TTS APIs.

    Parameters
    ----------
    base_url : str, optional
        Service base URL, such as ``http://127.0.0.1:8000``. A complete
        ``ws://`` or ``wss://`` URL ending in ``/tts/ws`` is also accepted.
        Defaults to ``http://127.0.0.1:8000``.
    voices : list[dict[str, str]] | None, optional
        Reference voice configurations containing ``name`` and ``path``.
        The first voice is selected by default. An empty list leaves reference
        audio selection to the service.

    Notes
    -----
    The service emits mono signed 16-bit little-endian PCM at 48 kHz. The
    ``output_sample_rate`` attribute is updated from the server's ``started``
    event so xtalk publishes audio with the correct rate.
    """

    output_sample_rate = 48000

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        voices: list[dict[str, str]] | None = None,
    ) -> None:
        self.base_url = base_url
        self._websocket_endpoint = _websocket_url(base_url)
        self._voices = [voice.copy() for voice in voices or []]
        self._voice_path_map = self._build_voice_path_map(self._voices)
        self._active_voice_name = (
            self._voices[0]["name"] if self._voices else None
        )
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._audio_queue: asyncio.Queue[bytes | object] | None = None
        self._receiver_error: BaseException | None = None
        self._finalized = False
        self._send_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()

    async def start(self) -> None:
        """Open a new live synthesis session and validate its audio format."""
        async with self._lifecycle_lock:
            if self._websocket is not None:
                raise RuntimeError("MOSS-TTS-Realtime session is already active")

            self._audio_queue = asyncio.Queue()
            self._receiver_error = None
            self._finalized = False
            self._session = aiohttp.ClientSession()
            try:
                self._websocket = await self._session.ws_connect(
                    self._websocket_endpoint,
                    autoping=True,
                    heartbeat=20.0,
                    max_msg_size=0,
                )
                start_event: dict[str, Any] = {
                    "type": "start",
                    "session_id": str(uuid.uuid4()),
                    "text": "",
                }
                prompt_audio = self._selected_prompt_audio()
                if prompt_audio is not None:
                    start_event["prompt_audio"] = prompt_audio
                await self._websocket.send_json(start_event)
                message = await self._websocket.receive()
                if message.type is not aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(
                        "Expected a started JSON event from MOSS-TTS-Realtime"
                    )
                event = _decode_control_event(message.data)
                if event["type"] != "started":
                    raise RuntimeError(
                        f"Expected a started event, received {event!r}"
                    )
                self._validate_audio_format(event)
                logger.debug(
                    "[realtime-tts-race] stage=moss_started "
                    "client=%x sample_rate=%d",
                    id(self),
                    self.output_sample_rate,
                )
                self._receiver_task = asyncio.create_task(
                    self._receive_audio(),
                    name="moss-tts-realtime-receiver",
                )
            except BaseException:
                await self._close_connection()
                raise

    async def append_text(self, text: str) -> None:
        """Append a text fragment to the active synthesis session.

        Parameters
        ----------
        text : str
            Incremental text to synthesize. CR and LF characters are removed
            before the text is sent to the service.
        """
        text = text.replace("\r", "").replace("\n", "")
        if not text:
            return
        websocket = self._require_active_session()
        if self._finalized:
            raise RuntimeError("Cannot append text after the session is flushed")
        async with self._send_lock:
            await websocket.send_json(
                {"type": "push", "text": text, "is_final": False}
            )
        logger.debug(
            "[realtime-tts-race] stage=moss_append_sent "
            "client=%x chunk_chars=%d finalized=%s",
            id(self),
            len(text),
            self._finalized,
        )

    async def flush(self) -> None:
        """Finalize incremental text and request all remaining audio."""
        websocket = self._require_active_session()
        if self._finalized:
            return
        async with self._send_lock:
            if self._finalized:
                return
            await websocket.send_json(
                {"type": "push", "text": "", "is_final": True}
            )
            self._finalized = True
            logger.debug(
                "[realtime-tts-race] stage=moss_flush_sent client=%x",
                id(self),
            )

    async def stop(self) -> None:
        """Finish or abort the active session and release network resources."""
        async with self._lifecycle_lock:
            websocket = self._websocket
            receiver_task = self._receiver_task
            if websocket is None:
                return

            logger.debug(
                "[realtime-tts-race] stage=moss_stop_begin "
                "client=%x finalized=%s receiver_present=%s",
                id(self),
                self._finalized,
                receiver_task is not None,
            )
            if self._finalized and receiver_task is not None:
                await receiver_task
            else:
                async with self._send_lock:
                    if not websocket.closed:
                        await websocket.send_json({"type": "close"})
            await self._close_connection()

    async def audio_stream(self) -> AsyncIterator[bytes]:
        """Yield raw PCM chunks until the service reports completion.

        Yields
        ------
        bytes
            Mono PCM S16LE chunks at ``output_sample_rate``.
        """
        queue = self._audio_queue
        if queue is None:
            raise RuntimeError("MOSS-TTS-Realtime session has not been started")

        while True:
            item = await queue.get()
            if item is _AUDIO_STREAM_END:
                break
            if isinstance(item, bytes):
                yield item

        if self._receiver_error is not None:
            raise RuntimeError("MOSS-TTS-Realtime audio stream failed") from (
                self._receiver_error
            )

    def synthesize(self, text: str) -> bytes:
        """Synchronously synthesize complete text through a temporary session.

        Parameters
        ----------
        text : str
            Complete text to synthesize.

        Returns
        -------
        bytes
            Concatenated mono PCM S16LE audio at 48 kHz.

        Raises
        ------
        RuntimeError
            If called from a running event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.async_synthesize(text))
        raise RuntimeError(
            "synthesize() cannot run inside an event loop; "
            "use async_synthesize() instead"
        )

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Asynchronously synthesize complete text through a temporary session.

        Parameters
        ----------
        text : str
            Complete text to synthesize.
        **kwargs
            Reserved for compatibility with the base TTS API.

        Returns
        -------
        bytes
            Concatenated mono PCM S16LE audio at 48 kHz.
        """
        del kwargs
        client = self.clone()
        await client.start()

        async def collect_audio() -> bytes:
            """Collect all chunks emitted by the temporary session."""
            return b"".join([chunk async for chunk in client.audio_stream()])

        collector = asyncio.create_task(collect_audio())
        try:
            await client.append_text(text)
            await client.flush()
            await client.stop()
            return await collector
        finally:
            if not collector.done():
                collector.cancel()
                await asyncio.gather(collector, return_exceptions=True)
            await client.stop()

    def clone(self) -> "MossTTSRealtime":
        """Create an independent client with the same service URL.

        Returns
        -------
        MossTTSRealtime
            A client with fresh connection and streaming state.
        """
        clone = MossTTSRealtime(
            base_url=self.base_url,
            voices=[voice.copy() for voice in self._voices],
        )
        clone._active_voice_name = self._active_voice_name
        return clone

    def set_voice(self, voice_names: list[str]) -> None:
        """Select the reference voice used for subsequent sessions.

        Parameters
        ----------
        voice_names : list[str]
            A list containing exactly one configured voice name.

        Raises
        ------
        ValueError
            If the selection is empty, contains multiple names, or names an
            unknown voice.
        """
        if len(voice_names) != 1:
            raise ValueError(
                "MossTTSRealtime accepts exactly one reference voice"
            )
        voice_name = voice_names[0]
        if voice_name not in self._voice_path_map:
            raise ValueError(f"Unknown voice name: {voice_name!r}")
        self._active_voice_name = voice_name


    @staticmethod
    def _build_voice_path_map(
        voices: list[dict[str, str]],
    ) -> dict[str, str]:
        """Validate voice configurations and map names to audio paths."""
        voice_path_map: dict[str, str] = {}
        for index, voice in enumerate(voices):
            name = voice.get("name")
            path = voice.get("path")
            if not isinstance(name, str) or not name:
                raise ValueError(
                    f"voices[{index}].name must be a non-empty string"
                )
            if not isinstance(path, str) or not path:
                raise ValueError(
                    f"voices[{index}].path must be a non-empty string"
                )
            if name in voice_path_map:
                raise ValueError(f"Duplicate voice name: {name!r}")
            voice_path_map[name] = path
        return voice_path_map

    def _selected_prompt_audio(self) -> str | None:
        """Return the selected server-local reference audio path."""
        if self._active_voice_name is None:
            return None
        return self._voice_path_map[self._active_voice_name]

    def _require_active_session(self) -> aiohttp.ClientWebSocketResponse:
        """Return the active WebSocket or raise a lifecycle error."""
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise RuntimeError("MOSS-TTS-Realtime session is not active")
        return websocket

    def _validate_audio_format(self, event: dict[str, Any]) -> None:
        """Validate server audio metadata and retain its sample rate."""
        codec = event.get("audio_codec")
        channels = event.get("channels")
        sample_rate = event.get("sample_rate")
        if codec != "pcm_s16le" or channels != 1:
            raise RuntimeError(
                "MOSS-TTS-Realtime must return mono pcm_s16le audio; "
                f"received codec={codec!r}, channels={channels!r}"
            )
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise RuntimeError(
                f"Invalid MOSS-TTS-Realtime sample rate: {sample_rate!r}"
            )
        self.output_sample_rate = sample_rate

    async def _receive_audio(self) -> None:
        """Receive binary audio and terminal control events in the background."""
        websocket = self._require_active_session()
        queue = self._audio_queue
        if queue is None:
            raise RuntimeError("MOSS-TTS-Realtime audio queue is unavailable")

        try:
            async for message in websocket:
                if message.type is aiohttp.WSMsgType.BINARY:
                    await queue.put(bytes(message.data))
                    continue
                if message.type is aiohttp.WSMsgType.TEXT:
                    event = _decode_control_event(message.data)
                    if event["type"] == "accepted":
                        continue
                    if event["type"] == "completed":
                        logger.debug(
                            "[realtime-tts-race] stage=moss_completed client=%x",
                            id(self),
                        )
                        return
                    raise RuntimeError(
                        f"Unexpected MOSS-TTS-Realtime event: {event!r}"
                    )
                if message.type in {
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                }:
                    break
            raise RuntimeError(
                "MOSS-TTS-Realtime connection closed before completion"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            self._receiver_error = error
        finally:
            await queue.put(_AUDIO_STREAM_END)

    async def _close_connection(self) -> None:
        """Close the receiver, WebSocket, and HTTP client session."""
        receiver_task = self._receiver_task
        current_task = asyncio.current_task()
        if (
            receiver_task is not None
            and receiver_task is not current_task
            and not receiver_task.done()
        ):
            receiver_task.cancel()
            await asyncio.gather(receiver_task, return_exceptions=True)

        websocket = self._websocket
        if websocket is not None and not websocket.closed:
            await websocket.close()

        session = self._session
        if session is not None and not session.closed:
            await session.close()

        self._receiver_task = None
        self._websocket = None
        self._session = None
        self._finalized = False
