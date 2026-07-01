"""Silero VAD backed by a shared ONNX Runtime session.

Each process creates a single ONNX session that is shared by all
``SileroVAD`` instances. Every instance keeps its own streaming model state,
context window, and pending audio buffer so concurrent sessions do not leak
state into each other.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, field
import json
import os
import socket
from pathlib import Path
import sys
import tempfile
import threading
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import websockets
except ImportError:
    websockets = None

from .interfaces import VAD
from ..registry import model


SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
CONTEXT_SAMPLES = 64

SILERO_VAD_ONNX_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/"
    "silero_vad/data/silero_vad.onnx"
)
_CACHE_SUBDIR = "xtalk/models"
_CACHE_FILENAME = "silero_vad.onnx"
_REMOTE_TIMEOUT_SECONDS = 10.0
_REMOTE_WS_PATH = "/ws/vad"

_MODEL_FILE_LOCK = threading.Lock()
_SESSION_LOCK = threading.Lock()
_SHARED_SESSION: Any | None = None
_SHARED_MODEL_PATH: str | None = None


def _user_cache_dir() -> Path:
    """Return the per-user cache directory for xtalk assets."""
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / _CACHE_SUBDIR


def _ensure_cached_model() -> Path:
    """Download the Silero VAD ONNX model into the user cache if needed."""
    cache_dir = _user_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / _CACHE_FILENAME
    if model_path.exists():
        return model_path

    with _MODEL_FILE_LOCK:
        if model_path.exists():
            return model_path

        temp_path: Path | None = None
        try:
            with urlopen(SILERO_VAD_ONNX_URL) as response:
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix="silero_vad.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(response.read())

            assert temp_path is not None
            temp_path.replace(model_path)
            return model_path
        except Exception:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise


def _resolve_model_path(model_path: str | None) -> Path:
    """Resolve the Silero VAD ONNX model path."""
    if model_path is not None:
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Silero VAD model not found: {resolved}")
        return resolved

    return _ensure_cached_model().resolve()


def _create_session(model_path: Path) -> ort.InferenceSession:
    """Create the shared ONNX Runtime session."""
    if ort is None:
        raise RuntimeError(
            "onnxruntime is required for local Silero VAD inference; "
            "install xtalk[silero-vad] or pass base_url to use a remote service"
        )

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def _get_shared_session(model_path: str | None) -> tuple[ort.InferenceSession, str]:
    """Return the process-wide ONNX session for Silero VAD."""
    global _SHARED_MODEL_PATH, _SHARED_SESSION

    resolved_path = str(_resolve_model_path(model_path))
    with _SESSION_LOCK:
        if (
            _SHARED_SESSION is None
            or _SHARED_MODEL_PATH is None
            or _SHARED_MODEL_PATH != resolved_path
        ):
            _SHARED_SESSION = _create_session(Path(resolved_path))
            _SHARED_MODEL_PATH = resolved_path
        assert _SHARED_SESSION is not None
        return _SHARED_SESSION, resolved_path


def _resolve_remote_vad_url(base_url: str) -> tuple[str, str]:
    """Resolve the remote Silero VAD base URL and WebSocket URL."""
    normalized_base_url = base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("base_url must not be empty")

    parts = urlsplit(normalized_base_url)
    if parts.scheme in {"http", "https"}:
        scheme = "ws" if parts.scheme == "http" else "wss"
    elif parts.scheme in {"ws", "wss"}:
        scheme = parts.scheme
    else:
        raise ValueError("remote Silero VAD base_url must use http(s) or ws(s)")

    path = parts.path.rstrip("/") or _REMOTE_WS_PATH
    websocket_url = urlunsplit((scheme, parts.netloc, path, parts.query, ""))
    return normalized_base_url, websocket_url


def _remote_json_payload(message: Any) -> dict[str, Any]:
    """Decode one remote VAD JSON message."""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if not isinstance(message, str):
        raise ValueError(f"remote VAD returned unsupported message type: {type(message)}")
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError("remote VAD JSON message must be an object")
    message_type = payload.get("type")
    if message_type == "error":
        detail = str(payload.get("message") or payload)
        raise RuntimeError(f"remote VAD error: {detail}")
    return payload


def _remote_frame_has_speech(payload: dict[str, Any], threshold: float) -> bool:
    """Return the boolean speech decision from one remote VAD frame payload."""
    if payload.get("type") != "frame":
        raise ValueError(f"remote VAD expected frame message, got {payload.get('type')!r}")
    if "is_speech" in payload:
        return bool(payload["is_speech"])
    speech_prob = payload.get("speech_prob")
    if isinstance(speech_prob, (int, float)):
        return float(speech_prob) >= threshold
    raise ValueError("remote VAD frame missing 'is_speech' or numeric 'speech_prob'")


@dataclass
class _SileroStreamState:
    """Per-instance streaming state for the Silero VAD model."""

    model_state: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 1, 128), dtype=np.float32)
    )
    context: np.ndarray = field(
        default_factory=lambda: np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
    )
    pending_audio: np.ndarray = field(
        default_factory=lambda: np.zeros((0,), dtype=np.float32)
    )

    def reset(self) -> None:
        """Reset per-stream model memory and buffered audio."""
        self.model_state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)
        self.pending_audio = np.zeros((0,), dtype=np.float32)


@model
class SileroVAD(VAD):
    """Silero VAD implementation with shared model session and per-instance state."""

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: str | None = None,
        frame_samples: int = FRAME_SAMPLES,
        context_samples: int = CONTEXT_SAMPLES,
        base_url: str | None = None,
    ) -> None:
        """Initialize the VAD instance.

        Parameters
        ----------
        base_url : str | None, optional
            Optional base URL of an external Silero VAD WebSocket service. When
            set, this instance sends PCM frames over a persistent WebSocket and
            ignores local inference parameters.
        threshold : float, optional
            Speech probability threshold used to convert the model output into a
            boolean decision.
        model_path : str | None, optional
            Optional path to the Silero VAD ONNX model. When omitted, several
            common repository-local paths are tried.
        frame_samples : int, optional
            Expected frame size in samples for model inference. Current Silero
            ONNX models typically expect 512 samples at 16 kHz.
        context_samples : int, optional
            Number of trailing samples from the previous frame to prepend as
            model context on the next inference call.
        """
        self.threshold = float(threshold)
        self.frame_samples = int(frame_samples)
        self.context_samples = int(context_samples)
        self.base_url: str | None = None
        self._remote_vad_url: str | None = None
        self._remote_ws: Any | None = None
        self._remote_loop: asyncio.AbstractEventLoop | None = None
        self._remote_lock: asyncio.Lock | None = None
        if base_url is not None:
            self.base_url, self._remote_vad_url = _resolve_remote_vad_url(base_url)
            return

        self.session, self.model_path = _get_shared_session(model_path)
        self._inference_lock = threading.Lock()
        self._state = _SileroStreamState()
        self._state.context = np.zeros(
            (1, self.context_samples),
            dtype=np.float32,
        )

    def clone(self) -> "SileroVAD":
        """Clone the VAD while reusing the shared ONNX session."""
        if self.base_url is not None:
            return SileroVAD(
                threshold=self.threshold,
                frame_samples=self.frame_samples,
                base_url=self.base_url,
            )

        return SileroVAD(
            threshold=self.threshold,
            model_path=self.model_path,
            frame_samples=self.frame_samples,
            context_samples=self.context_samples,
        )

    def reset(self) -> None:
        """Reset this instance's streaming state."""
        if self.base_url is not None:
            self._schedule_remote_close()
            return

        self._state.reset()
        if self.context_samples != CONTEXT_SAMPLES:
            self._state.context = np.zeros(
                (1, self.context_samples),
                dtype=np.float32,
            )

    def is_speech(self, frame: bytes) -> bool:
        """Determine whether the latest complete frame contains speech.

        Parameters
        ----------
        frame : bytes
            PCM 16-bit mono audio bytes at 16 kHz. Input may contain any number
            of samples; the instance buffers partial data internally and only
            runs inference on complete ``frame_samples`` windows.

        Returns
        -------
        bool
            ``True`` when the most recent complete frame has speech
            probability greater than or equal to ``threshold``. If not enough
            audio is available to form a complete frame, returns ``False``.
        """
        if not frame:
            return False

        if self.base_url is not None:
            return self._is_speech_remote_sync(frame)

        audio = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return False

        self._state.pending_audio = np.concatenate([self._state.pending_audio, audio])
        last_prob: float | None = None

        while self._state.pending_audio.shape[0] >= self.frame_samples:
            chunk = self._state.pending_audio[: self.frame_samples]
            self._state.pending_audio = self._state.pending_audio[self.frame_samples :]
            last_prob = self._infer_one_frame(chunk)

        if last_prob is None:
            return False
        return last_prob >= self.threshold

    async def async_is_speech(self, frame: bytes) -> bool:
        """Asynchronously determine whether the latest frame contains speech."""
        if self.base_url is not None:
            return await self._is_speech_remote_async(frame)
        return await VAD.async_is_speech(self, frame)

    def _infer_one_frame(self, frame: np.ndarray) -> float:
        """Run ONNX inference for one complete frame."""
        if frame.shape[0] != self.frame_samples:
            raise ValueError(f"frame must have {self.frame_samples} samples")

        x = frame.reshape(1, -1).astype(np.float32, copy=False)
        x = np.concatenate([self._state.context, x], axis=1)
        ort_inputs = {
            "input": x,
            "state": self._state.model_state,
            "sr": np.array(SAMPLE_RATE, dtype=np.int64),
        }

        with self._inference_lock:
            out, new_state = self.session.run(None, ort_inputs)

        self._state.model_state = new_state
        self._state.context = x[:, -self.context_samples :].astype(
            np.float32,
            copy=False,
        )
        return float(np.asarray(out).squeeze())

    def _is_speech_remote_sync(self, frame: bytes) -> bool:
        """Synchronously send a PCM frame through a short-lived WebSocket."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._is_speech_remote_once(frame))
        raise RuntimeError(
            "remote SileroVAD.is_speech() cannot run inside an active event loop; "
            "use async_is_speech() instead"
        )

    async def _is_speech_remote_once(self, frame: bytes) -> bool:
        """Run one synchronous compatibility request with a temporary connection."""
        try:
            return await self._is_speech_remote_async(frame)
        finally:
            await self._close_remote_connection()

    async def _is_speech_remote_async(self, frame: bytes) -> bool:
        """Send a PCM frame to the remote Silero VAD WebSocket service."""
        if self._remote_vad_url is None:
            raise RuntimeError("remote VAD URL is not configured")
        if not frame:
            return False
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote Silero VAD inference; "
                "install xtalk[silero-vad]"
            )

        lock = self._get_remote_lock()
        async with lock:
            for attempt in range(2):
                try:
                    websocket = await self._ensure_remote_connection()
                    await websocket.send(frame)
                    while True:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=_REMOTE_TIMEOUT_SECONDS,
                        )
                        payload = _remote_json_payload(message)
                        if payload.get("type") == "frame":
                            return _remote_frame_has_speech(payload, self.threshold)
                except Exception:
                    await self._close_remote_connection()
                    if attempt == 0:
                        continue
                    raise
            return False

    def _get_remote_lock(self) -> asyncio.Lock:
        """Return the remote WebSocket lock for the current event loop."""
        loop = asyncio.get_running_loop()
        if self._remote_lock is None or self._remote_loop is not loop:
            self._remote_loop = loop
            self._remote_lock = asyncio.Lock()
            self._remote_ws = None
        return self._remote_lock

    async def _ensure_remote_connection(self) -> Any:
        """Open and initialize the remote VAD WebSocket when needed."""
        if self._remote_vad_url is None:
            raise RuntimeError("remote VAD URL is not configured")
        if self._remote_ws is not None:
            return self._remote_ws
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote Silero VAD inference; "
                "install xtalk[silero-vad]"
            )

        try:
            websocket = await websockets.connect(
                self._remote_vad_url,
                open_timeout=_REMOTE_TIMEOUT_SECONDS,
                close_timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            self._remote_ws = websocket
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": SAMPLE_RATE,
                        "frame_samples": self.frame_samples,
                        "encoding": "pcm_s16le",
                        "channels": 1,
                        "positive_speech_threshold": self.threshold,
                    }
                )
            )
            message = await asyncio.wait_for(
                websocket.recv(),
                timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            payload = _remote_json_payload(message)
            if payload.get("type") != "start_ack":
                raise ValueError(
                    f"remote VAD expected start_ack, got {payload.get('type')!r}"
                )
            return websocket
        except Exception:
            await self._close_remote_connection()
            raise

    async def _close_remote_connection(self) -> None:
        """Close the active remote VAD WebSocket connection."""
        websocket = self._remote_ws
        self._remote_ws = None
        if websocket is not None:
            await websocket.close()

    def _schedule_remote_close(self) -> None:
        """Best-effort close for the active remote VAD connection."""
        websocket = self._remote_ws
        self._remote_ws = None
        if websocket is None:
            return
        loop = self._remote_loop
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is not None and running_loop is loop:
            running_loop.create_task(websocket.close())
            return
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(loop.create_task, websocket.close())

def _is_tcp_port_open(host: str, port: int, timeout: float) -> bool:
    """Return whether a TCP listener accepts connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _build_test_pcm(sample_rate: int, duration_seconds: float) -> bytes:
    """Build a small PCM s16le test tone for the remote VAD smoke test."""
    sample_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * time_axis)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _run_remote_smoke_test() -> int:
    """Run a simple client request against a running Silero VAD service."""
    parser = argparse.ArgumentParser(
        description="Smoke test the Silero VAD WebSocket client."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--duration-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not _is_tcp_port_open(args.host, args.port, timeout=2.0):
        print(
            "No service is listening on "
            f"{args.host}:{args.port}. Please start silerovad first.",
            file=sys.stderr,
        )
        return 2

    base_url = f"ws://{args.host}:{args.port}{_REMOTE_WS_PATH}"
    vad = SileroVAD(base_url=base_url)
    pcm_bytes = _build_test_pcm(SAMPLE_RATE, args.duration_seconds)
    try:
        is_speech = vad.is_speech(pcm_bytes)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"Silero VAD client request failed for {base_url}: {exc}", file=sys.stderr)
        return 1

    print(
        "Silero VAD client request succeeded: "
        f"input_bytes={len(pcm_bytes)}, is_speech={is_speech}, base_url={base_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_remote_smoke_test())
