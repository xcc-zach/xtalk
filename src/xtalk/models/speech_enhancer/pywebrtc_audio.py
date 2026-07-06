"""Remote speech enhancer backed by the pywebrtc-audio service."""

from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .interfaces import SpeechEnhancer
from ..registry import model


_STREAM_PATH = "/v1/stream"
_REMOTE_TIMEOUT_SECONDS = 10.0
_DEFAULT_STREAM_QUERY: dict[str, str] = {
    "sample_rate": "16000",
    "num_channels": "1",
    "echo_cancellation": "true",
    "noise_suppression": "true",
    "high_pass_filter": "false",
    "auto_gain_control": "false",
    "dtype": "int16",
}

try:
    import websockets
except ImportError:  # pragma: no cover - optional dependency.
    websockets = None


def _resolve_stream_url(base_url: str) -> tuple[str, str]:
    """Resolve a pywebrtc-audio base URL into a streaming WebSocket URL.

    Parameters
    ----------
    base_url : str
        HTTP(S) or WS(S) base URL of the pywebrtc-audio service. The URL may
        already include ``/v1/stream`` and query parameters.

    Returns
    -------
    tuple[str, str]
        Normalized base URL and WebSocket stream URL.
    """
    normalized_base_url = base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("base_url must not be empty")

    parts = urlsplit(normalized_base_url)
    if parts.scheme in {"http", "https"}:
        scheme = "ws" if parts.scheme == "http" else "wss"
    elif parts.scheme in {"ws", "wss"}:
        scheme = parts.scheme
    else:
        raise ValueError("pywebrtc-audio base_url must use http(s) or ws(s)")

    path = parts.path.rstrip("/") or _STREAM_PATH
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    for key, value in _DEFAULT_STREAM_QUERY.items():
        query.setdefault(key, value)

    stream_url = urlunsplit(
        (
            scheme,
            parts.netloc,
            path,
            urlencode(query),
            "",
        )
    )
    return normalized_base_url, stream_url


def _json_payload(message: object) -> dict[str, object]:
    """Decode one pywebrtc-audio WebSocket JSON message."""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if not isinstance(message, str):
        raise ValueError(
            f"pywebrtc-audio returned unsupported message type: {type(message)}"
        )
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError("pywebrtc-audio JSON message must be an object")
    if payload.get("type") == "error":
        detail = str(payload.get("detail") or payload.get("message") or payload)
        raise RuntimeError(f"pywebrtc-audio error: {detail}")
    return payload


@model(aliases=["PyWebRTCAudio", "pywebrtc_audio"])
class PyWebRTCAudio(SpeechEnhancer):
    """Speech enhancer adapter for a remote pywebrtc-audio service."""

    def __init__(self, base_url: str):
        """Initialize the remote pywebrtc-audio adapter.

        Parameters
        ----------
        base_url : str
            HTTP(S) or WS(S) base URL of the pywebrtc-audio service.
        """
        self.base_url, self._stream_url = _resolve_stream_url(base_url)
        self._ws: object | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    def enhance(self, audio: bytes, far: bytes) -> bytes:
        """Enhance one PCM frame synchronously.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.
        far : bytes
            Far-end reference PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bytes
            Enhanced PCM 16-bit mono audio bytes at 16 kHz.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._enhance_once(audio, far))
        raise RuntimeError(
            "PyWebRTCAudio.enhance() cannot run inside an active event loop; "
            "use async_enhance() instead"
        )

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes:
        """Enhance one PCM frame asynchronously.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.
        far : bytes
            Far-end reference PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bytes
            Enhanced PCM 16-bit mono audio bytes at 16 kHz.
        """
        if not audio:
            return b""
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote pywebrtc-audio inference"
            )

        lock = self._get_lock()
        async with lock:
            return await self._send_frame_with_retry(audio, far)

    def flush(self) -> bytes:
        """Return pending enhanced audio.

        Returns
        -------
        bytes
            Empty bytes because pywebrtc-audio emits one result per request.
        """
        return b""

    async def async_flush(self) -> bytes:
        """Return pending enhanced audio asynchronously.

        Returns
        -------
        bytes
            Empty bytes because pywebrtc-audio emits one result per request.
        """
        return b""

    def reset(self) -> None:
        """Reset the remote stream state."""
        self._schedule_close()

    def clone(self) -> "PyWebRTCAudio":
        """Clone the enhancer for a new session.

        Returns
        -------
        PyWebRTCAudio
            Clone with an independent WebSocket connection.
        """
        return PyWebRTCAudio(base_url=self.base_url)

    async def _enhance_once(self, audio: bytes, far: bytes) -> bytes:
        """Enhance one synchronous request using a temporary connection."""
        try:
            return await self.async_enhance(audio, far)
        finally:
            await self._close_connection()

    async def _send_frame(self, websocket: object, audio: bytes, far: bytes) -> bytes:
        """Send one JSON process frame and return enhanced PCM bytes."""
        await websocket.send(
            json.dumps(
                {
                    "type": "process",
                    "audio": base64.b64encode(audio).decode("ascii"),
                    "far": base64.b64encode(far).decode("ascii"),
                }
            )
        )
        while True:
            message = await asyncio.wait_for(
                websocket.recv(),
                timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            payload = _json_payload(message)
            if payload.get("type") == "result":
                result_audio = payload.get("audio")
                if isinstance(result_audio, str):
                    return base64.b64decode(result_audio)
                raise ValueError("pywebrtc-audio result missing base64 audio")

    async def _send_frame_with_retry(self, audio: bytes, far: bytes) -> bytes:
        """Send one process frame and retry once after reconnecting."""
        for attempt in range(2):
            try:
                websocket = await self._ensure_connection()
                return await self._send_frame(websocket, audio, far)
            except Exception:
                await self._close_connection()
                if attempt == 0:
                    continue
                raise
        return b""

    def _get_lock(self) -> asyncio.Lock:
        """Return the WebSocket lock for the current event loop."""
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._loop = loop
            self._lock = asyncio.Lock()
            self._ws = None
        return self._lock

    async def _ensure_connection(self) -> object:
        """Open a pywebrtc-audio WebSocket connection when needed."""
        if self._ws is not None:
            return self._ws
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote pywebrtc-audio inference"
            )
        try:
            self._ws = await websockets.connect(
                self._stream_url,
                open_timeout=_REMOTE_TIMEOUT_SECONDS,
                close_timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            return self._ws
        except Exception:
            await self._close_connection()
            raise

    async def _close_connection(self) -> None:
        """Close the active WebSocket connection."""
        websocket = self._ws
        self._ws = None
        if websocket is not None:
            await websocket.close()

    def _schedule_close(self) -> None:
        """Schedule a best-effort close for the active WebSocket connection."""
        websocket = self._ws
        self._ws = None
        if websocket is None:
            return
        loop = self._loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop is loop:
            running_loop.create_task(websocket.close())
            return
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.create_task, websocket.close())
