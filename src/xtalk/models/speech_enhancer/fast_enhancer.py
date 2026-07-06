"""FastEnhancer speech enhancement module.

Implements streaming enhancement based on the FastEnhancer-S ONNX model:
- Input: 16 kHz PCM s16le audio frames
- Output: enhanced 16 kHz PCM s16le audio frames
- Maintains ONNX cache state for streaming processing
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np

try:
    import onnxruntime
except ImportError:  # pragma: no cover - remote mode does not need ONNX Runtime.
    onnxruntime = None  # type: ignore[assignment]

try:
    import websockets
except ImportError:  # pragma: no cover - local mode does not need websockets.
    websockets = None

from .interfaces import SpeechEnhancer
from ..registry import model


_REMOTE_WS_PATH = "/ws/fastenhancer/realtime"
_REMOTE_TIMEOUT_SECONDS = 10.0
_REMOTE_FRAME_SAMPLES = 512


def _resolve_remote_enhancer_url(base_url: str) -> tuple[str, str]:
    """Resolve the remote FastEnhancer base URL and WebSocket URL."""
    normalized_base_url = base_url.strip().rstrip("/")
    if not normalized_base_url:
        raise ValueError("base_url must not be empty")

    parts = urlsplit(normalized_base_url)
    if parts.scheme in {"http", "https"}:
        scheme = "ws" if parts.scheme == "http" else "wss"
    elif parts.scheme in {"ws", "wss"}:
        scheme = parts.scheme
    else:
        raise ValueError("remote FastEnhancer base_url must use http(s) or ws(s)")

    path = parts.path.rstrip("/") or _REMOTE_WS_PATH
    websocket_url = urlunsplit((scheme, parts.netloc, path, parts.query, ""))
    return normalized_base_url, websocket_url


def _remote_json_payload(message: object) -> dict[str, object]:
    """Decode one remote FastEnhancer JSON message."""
    if isinstance(message, bytes):
        message = message.decode("utf-8")
    if not isinstance(message, str):
        raise ValueError(
            f"remote FastEnhancer returned unsupported message type: {type(message)}"
        )
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError("remote FastEnhancer JSON message must be an object")
    if payload.get("type") == "error":
        detail = str(payload.get("message") or payload)
        raise RuntimeError(f"remote FastEnhancer error: {detail}")
    return payload


@model(aliases=["FastEnhancerS", "speech_enhancer"])
class FastEnhancer(SpeechEnhancer):
    """Streaming or remote speech enhancer using FastEnhancer.

    Notes
    -----
    When ``base_url`` is provided, all other initialization parameters are
    ignored and audio is enhanced by a FastEnhancer WebSocket service.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_fft: int = 512,
        hop_size: int = 256,
        _shared_session: Optional[object] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the enhancer.

        Parameters
        ----------
        model_path : str | None, optional
            Local ONNX model path for in-process enhancement.
        n_fft : int, optional
            FFT size used for delay compensation in local mode.
        hop_size : int, optional
            Number of samples processed per local ONNX step.
        _shared_session : object | None, optional
            Existing ONNX Runtime session reused by local clones.
        base_url : str | None, optional
            Base URL of a running FastEnhancer WebSocket service. When set,
            ``model_path``, ``n_fft``, ``hop_size``, and ``_shared_session`` are
            ignored and no local ONNX model is loaded.
        """
        self.sample_rate = 16000
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.base_url: str | None = None
        self._remote_enhancer_url: str | None = None
        self._remote_frame_samples = _REMOTE_FRAME_SAMPLES
        self._remote_pending_audio = bytearray()
        self._remote_ws: object | None = None
        self._remote_loop: asyncio.AbstractEventLoop | None = None
        self._remote_lock: asyncio.Lock | None = None
        if base_url is not None:
            self.base_url, self._remote_enhancer_url = _resolve_remote_enhancer_url(
                base_url
            )
        else:
            self.base_url = None
        if self.base_url is not None:
            self.model_path = None
            return

        self.model_path = model_path

        # Reuse shared session if provided
        if _shared_session is not None:
            self.session = _shared_session
        else:
            # Otherwise create a new session
            self._init_session(model_path)

        # Cache state per instance
        self._init_cache()

        # Input/output buffers per instance
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)

        # Flag for first frame (requires special padding)
        self.is_first_frame = True

        # Track total samples for tail padding/alignment
        self._total_input_samples = 0
        self._total_output_samples = 0

    def _init_session(self, model_path: Optional[str]) -> None:
        """Initialize ONNX Runtime session (only during first creation)."""
        if onnxruntime is None:
            raise ImportError("onnxruntime is required for local FastEnhancer mode")

        # Resolve default model paths relative to this file
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidates = [
                os.path.join(base_dir, "fastenhancer_s.onnx"),
                os.path.join(base_dir, "model", "fastenhancer_s.onnx"),
                os.path.normpath(
                    os.path.join(
                        base_dir,
                        "..",
                        "..",
                        "..",
                        "..",
                        "frontend",
                        "src",
                        "fastenhancer_s.onnx",
                    )
                ),
            ]
            model_path = next(
                (p for p in candidates if os.path.exists(p)), candidates[0]
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Speech enhancer model not found: {model_path}")

        self.model_path = model_path

        # Create ONNX Runtime session
        sess_options = onnxruntime.SessionOptions()
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1

        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    def _init_cache(self) -> None:
        """Initialize per-instance cache state."""
        self.cache_inputs = {
            x.name: np.zeros(x.shape, dtype=np.float32)
            for x in self.session.get_inputs()
            if x.name.startswith("cache_in_")
        }

    def reset(self) -> None:
        """Reset enhancer state."""
        if self.base_url is not None:
            self._remote_pending_audio.clear()
            self._schedule_remote_close()
            return

        self._init_cache()
        self.input_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)
        self.is_first_frame = True
        # Reset sample counters for alignment
        self._total_input_samples = 0
        self._total_output_samples = 0

    def clone(self) -> "FastEnhancer":
        """Clone enhancer sharing local weights or remote service settings.

        Returns
        -------
        FastEnhancer
            Clone with isolated runtime state.
        """
        if self.base_url is not None:
            return FastEnhancer(base_url=self.base_url)

        return FastEnhancer(
            model_path=self.model_path,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            _shared_session=self.session,  # Shared ONNX session
        )

    def enhance(self, pcm_bytes: bytes, far: bytes) -> bytes:
        """Enhance audio frames in streaming mode.

        Parameters
        ----------
        pcm_bytes : bytes
            PCM 16-bit mono audio bytes at 16 kHz.
        far : bytes
            Ignored by FastEnhancer.

        Returns
        -------
        bytes
            Enhanced PCM 16-bit mono audio bytes at 16 kHz.
        """
        if not pcm_bytes:
            return b""
        if self.base_url is not None:
            return self._enhance_remote_sync(pcm_bytes)

        # Convert to float32 [-1, 1]
        pcm_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        pcm_float = pcm_int16.astype(np.float32) / 32768.0
        pcm_float = np.clip(pcm_float, a_min=-1.0, a_max=1.0)

        input_len = len(pcm_int16)
        self._total_input_samples += input_len

        # Append to input buffer
        self.input_buffer = np.concatenate([self.input_buffer, pcm_float])

        # Process frames by hop size
        while len(self.input_buffer) >= self.hop_size:
            frame_in = self.input_buffer[: self.hop_size].reshape(1, -1)
            self.input_buffer = self.input_buffer[self.hop_size :]

            # ONNX inference
            self.cache_inputs["wav_in"] = frame_in
            outputs = self.session.run(None, self.cache_inputs)

            # Update cache state
            for j in range(len(outputs) - 1):
                self.cache_inputs[f"cache_in_{j}"] = outputs[j + 1]

            # Append output frame
            frame_out = outputs[0][0]
            self.output_buffer = np.concatenate([self.output_buffer, frame_out])

        # Drop first (n_fft - hop_size) samples for delay compensation
        if self.is_first_frame and len(self.output_buffer) >= (
            self.n_fft - self.hop_size
        ):
            self.output_buffer = self.output_buffer[self.n_fft - self.hop_size :]
            self.is_first_frame = False

        # Extract same-length audio from output buffer
        if len(self.output_buffer) < input_len:
            # Zero-pad when there is not enough output to maintain length
            output_samples = np.concatenate(
                [
                    self.output_buffer,
                    np.zeros(input_len - len(self.output_buffer), dtype=np.float32),
                ]
            )
            self.output_buffer = np.array([], dtype=np.float32)
        else:
            output_samples = self.output_buffer[:input_len]
            self.output_buffer = self.output_buffer[input_len:]

        self._total_output_samples += len(output_samples)

        # Convert back to s16le
        output_samples = np.clip(output_samples, a_min=-1.0, a_max=1.0)
        output_int16 = (output_samples * 32768.0).astype(np.int16)
        return output_int16.tobytes()

    def flush(self) -> bytes:
        """Flush remaining buffers at the end of the stream.

        Returns
        -------
        bytes
            Remaining enhanced PCM audio bytes.
        """
        if self.base_url is not None:
            return self._flush_remote_sync()

        # Pad silence to process leftover input (similar to official pad-right)
        padding_needed = self.n_fft
        padding = np.zeros(padding_needed, dtype=np.float32)
        self.input_buffer = np.concatenate([self.input_buffer, padding])

        # Process any remaining frames
        while len(self.input_buffer) >= self.hop_size:
            frame_in = self.input_buffer[: self.hop_size].reshape(1, -1)
            self.input_buffer = self.input_buffer[self.hop_size :]

            self.cache_inputs["wav_in"] = frame_in
            outputs = self.session.run(None, self.cache_inputs)

            for j in range(len(outputs) - 1):
                self.cache_inputs[f"cache_in_{j}"] = outputs[j + 1]

            frame_out = outputs[0][0]
            self.output_buffer = np.concatenate([self.output_buffer, frame_out])

        # Apply first-frame compensation if it didn't trigger before
        if self.is_first_frame and len(self.output_buffer) >= (
            self.n_fft - self.hop_size
        ):
            self.output_buffer = self.output_buffer[self.n_fft - self.hop_size :]
            self.is_first_frame = False

        # Drain remaining output without exceeding input length
        remaining_needed = self._total_input_samples - self._total_output_samples
        if remaining_needed <= 0:
            return b""

        output_len = min(len(self.output_buffer), remaining_needed)
        if output_len <= 0:
            return b""

        output_samples = self.output_buffer[:output_len]
        self.output_buffer = self.output_buffer[output_len:]
        self._total_output_samples += len(output_samples)

        output_samples = np.clip(output_samples, a_min=-1.0, a_max=1.0)
        output_int16 = (output_samples * 32768.0).astype(np.int16)
        return output_int16.tobytes()

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes:
        """Asynchronously enhance audio."""
        if self.base_url is not None:
            return await self._enhance_remote_async(audio)
        return await SpeechEnhancer.async_enhance(self, audio, far)

    async def async_flush(self) -> bytes:
        """Asynchronously flush buffered remote or local audio."""
        if self.base_url is not None:
            return await self._flush_remote()
        return await SpeechEnhancer.async_flush(self)

    def _enhance_remote_sync(self, pcm_bytes: bytes) -> bytes:
        """Synchronously enhance PCM through a short-lived WebSocket."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._enhance_remote_once(pcm_bytes))
        raise RuntimeError(
            "remote FastEnhancer.enhance() cannot run inside an active event loop; "
            "use async_enhance() instead"
        )

    async def _enhance_remote_once(self, pcm_bytes: bytes) -> bytes:
        """Run one synchronous compatibility request with a temporary connection."""
        try:
            return await self._enhance_remote_async(pcm_bytes)
        finally:
            await self._close_remote_connection()

    async def _enhance_remote_async(self, pcm_bytes: bytes) -> bytes:
        """Enhance PCM through the remote FastEnhancer WebSocket."""
        if self._remote_enhancer_url is None:
            raise RuntimeError("remote FastEnhancer URL is not configured")
        if not pcm_bytes:
            return b""
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote FastEnhancer inference; "
                "install xtalk[fast-enhancer]"
            )

        lock = self._get_remote_lock()
        async with lock:
            self._remote_pending_audio.extend(pcm_bytes)
            frame_bytes = self._remote_frame_samples * 2
            chunks: list[bytes] = []
            while len(self._remote_pending_audio) >= frame_bytes:
                frame = bytes(self._remote_pending_audio[:frame_bytes])
                output = await self._send_remote_frame_with_retry(frame)
                del self._remote_pending_audio[:frame_bytes]
                chunks.append(output)
            return b"".join(chunks)

    def _flush_remote_sync(self) -> bytes:
        """Synchronously flush remote audio."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._flush_remote())
        raise RuntimeError(
            "remote FastEnhancer.flush() cannot run inside an active event loop; "
            "use async_flush() instead"
        )

    async def _flush_remote(self) -> bytes:
        """Send a best-effort remote flush command and return tail audio."""
        lock = self._get_remote_lock()
        async with lock:
            if self._remote_enhancer_url is None:
                return b""
            chunks: list[bytes] = []
            try:
                websocket = await self._ensure_remote_connection()
                if self._remote_pending_audio:
                    pending_len = len(self._remote_pending_audio)
                    frame_bytes = self._remote_frame_samples * 2
                    padding_len = frame_bytes - pending_len
                    frame = bytes(self._remote_pending_audio) + bytes(padding_len)
                    output = await self._send_remote_frame(websocket, frame)
                    chunks.append(output[:pending_len])
                    self._remote_pending_audio.clear()
                await websocket.send(json.dumps({"type": "flush"}))
                while True:
                    message = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=_REMOTE_TIMEOUT_SECONDS,
                    )
                    if isinstance(message, bytes):
                        chunks.append(message)
                        continue
                    payload = _remote_json_payload(message)
                    if payload.get("type") == "flush_ack":
                        return b"".join(chunks)
            except Exception:
                await self._close_remote_connection()
                raise

    async def _send_remote_frame_with_retry(self, pcm_bytes: bytes) -> bytes:
        """Send one complete remote frame and retry once after reconnecting."""
        for attempt in range(2):
            try:
                websocket = await self._ensure_remote_connection()
                return await self._send_remote_frame(websocket, pcm_bytes)
            except Exception:
                await self._close_remote_connection()
                if attempt == 0:
                    continue
                raise
        return b""

    async def _send_remote_frame(self, websocket: object, pcm_bytes: bytes) -> bytes:
        """Send one complete remote frame and return enhanced PCM bytes."""
        await websocket.send(pcm_bytes)
        while True:
            message = await asyncio.wait_for(
                websocket.recv(),
                timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            if isinstance(message, bytes):
                return message
            payload = _remote_json_payload(message)
            if payload.get("type") == "audio":
                audio = payload.get("audio")
                if isinstance(audio, str):
                    import base64

                    return base64.b64decode(audio)
                raise ValueError(
                    "remote FastEnhancer audio message missing base64 audio"
                )

    def _get_remote_lock(self) -> asyncio.Lock:
        """Return the remote WebSocket lock for the current event loop."""
        loop = asyncio.get_running_loop()
        if self._remote_lock is None or self._remote_loop is not loop:
            self._remote_loop = loop
            self._remote_lock = asyncio.Lock()
            self._remote_ws = None
        return self._remote_lock

    async def _ensure_remote_connection(self) -> object:
        """Open and initialize the remote FastEnhancer WebSocket when needed."""
        if self._remote_enhancer_url is None:
            raise RuntimeError("remote FastEnhancer URL is not configured")
        if self._remote_ws is not None:
            return self._remote_ws
        if websockets is None:
            raise RuntimeError(
                "websockets is required for remote FastEnhancer inference; "
                "install xtalk[fast-enhancer]"
            )

        try:
            websocket = await websockets.connect(
                self._remote_enhancer_url,
                open_timeout=_REMOTE_TIMEOUT_SECONDS,
                close_timeout=_REMOTE_TIMEOUT_SECONDS,
            )
            self._remote_ws = websocket
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "sample_rate": self.sample_rate,
                        "frame_samples": self._remote_frame_samples,
                        "encoding": "pcm_s16le",
                        "channels": 1,
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
                    "remote FastEnhancer expected start_ack, "
                    f"got {payload.get('type')!r}"
                )
            return websocket
        except Exception:
            await self._close_remote_connection()
            raise

    async def _close_remote_connection(self) -> None:
        """Close the active remote FastEnhancer WebSocket connection."""
        websocket = self._remote_ws
        self._remote_ws = None
        if websocket is not None:
            await websocket.close()

    def _schedule_remote_close(self) -> None:
        """Best-effort close for the active remote FastEnhancer connection."""
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
    """Build a small PCM s16le test tone for the remote smoke test."""
    sample_count = int(sample_rate * duration_seconds)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    samples = 0.2 * np.sin(2.0 * np.pi * 440.0 * time_axis)
    return (samples * 32767.0).astype(np.int16).tobytes()


def _run_remote_smoke_test() -> int:
    """Run a simple client request against a running FastEnhancer service."""
    parser = argparse.ArgumentParser(
        description="Smoke test the FastEnhancer WebSocket client."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--duration-seconds", type=float, default=0.5)
    args = parser.parse_args()

    if not _is_tcp_port_open(args.host, args.port, timeout=2.0):
        print(
            "No service is listening on "
            f"{args.host}:{args.port}. Please start fastenhancer first.",
            file=sys.stderr,
        )
        return 2

    base_url = f"ws://{args.host}:{args.port}{_REMOTE_WS_PATH}"
    enhancer = FastEnhancer(base_url=base_url)
    pcm_bytes = _build_test_pcm(enhancer.sample_rate, args.duration_seconds)
    try:
        enhanced_bytes = enhancer.enhance(pcm_bytes, bytes(len(pcm_bytes)))
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(
            f"FastEnhancer client request failed for {base_url}: {exc}",
            file=sys.stderr,
        )
        return 1

    if not enhanced_bytes or len(enhanced_bytes) % 2 != 0:
        print(
            "FastEnhancer client request failed: invalid PCM response "
            f"({len(enhanced_bytes)} bytes).",
            file=sys.stderr,
        )
        return 1

    print(
        "FastEnhancer client request succeeded: "
        f"input_bytes={len(pcm_bytes)}, output_bytes={len(enhanced_bytes)}, "
        f"base_url={base_url}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_remote_smoke_test())
