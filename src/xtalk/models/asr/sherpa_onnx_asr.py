from __future__ import annotations

"""
Remote ASR implementation based on sherpa-onnx WebSocket services.

Features:
- Supports two modes:
  - streaming: connect to sherpa-onnx-online-websocket-server.
  - offline: connect to sherpa-onnx-offline-websocket-server and simulate
    streaming via MockStreamRecognizer.

Input: PCM 16-bit mono 16 kHz raw bytes.
"""

import asyncio
from typing import Optional
from urllib.parse import urlparse

import numpy as np

from .interfaces import ASR
from ...log_utils import logger
from ..audio_utils import MockStreamRecognizer
from ..registry import model

try:  # websockets dependency
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover - early failure when dependency missing
    websockets = None  # type: ignore
    _import_error = e
else:
    _import_error = None


def _resolve_sherpa_base_url(base_url: str) -> str:
    """Resolve the Sherpa-ONNX WebSocket base URL."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must not be empty")
    parsed = urlparse(normalized)
    if not parsed.scheme:
        normalized = f"ws://{normalized}"
        parsed = urlparse(normalized)
    if parsed.scheme not in {"ws", "wss"}:
        raise ValueError("SherpaOnnxASR base_url must use ws:// or wss://")
    return normalized


@model
class SherpaOnnxASR(ASR):
    """Sherpa-ONNX WebSocket ASR wrapper."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        base_url: str = "ws://localhost:6006",
        mode: str = "streaming",
        # streaming-mode parameters
        samples_per_message: int = 8000,
        seconds_per_message: float = 0.0,
        # offline-mode parameters
        offline_payload_len: int = 10240,
        mock_window_size: int = 5,
    ) -> None:
        """Initialize Sherpa-ONNX WebSocket ASR.

        Parameters
        ----------
        base_url : str, optional
            WebSocket server URL.
        mode : str, optional
            Recognition mode, either ``"streaming"`` or ``"offline"``.
        samples_per_message : int, optional
            Number of samples per send when streaming.
        seconds_per_message : float, optional
            Optional delay after each streaming payload. Zero disables delays.
        offline_payload_len : int, optional
            Maximum payload bytes per send for offline mode.
        mock_window_size : int, optional
            Window size used by ``MockStreamRecognizer`` in offline mode.
        """

        if websockets is None:  # pragma: no cover - dependency missing
            raise ImportError(
                f"websockets is required for SherpaOnnxASR: {_import_error}"
            )

        self.base_url = _resolve_sherpa_base_url(base_url)

        mode_norm = mode.strip().lower()
        if mode_norm in ("streaming", "online"):
            self.mode = "streaming"
        elif mode_norm in ("offline", "non_streaming", "non-streaming"):
            self.mode = "offline"
        else:
            raise ValueError(f"Unsupported mode for SherpaOnnxASR: {mode!r}")

        self.samples_per_message = int(samples_per_message)
        self.seconds_per_message = float(seconds_per_message)
        self.offline_payload_len = int(offline_payload_len)
        self.mock_window_size = int(mock_window_size)

        # Offline mode uses MockStreamRecognizer to simulate streaming via async recognition
        self._mock_recognizer: Optional[MockStreamRecognizer]
        if self.mode == "offline":
            self._mock_recognizer = MockStreamRecognizer(
                self._recognize_offline_async, window_size=self.mock_window_size
            )
        else:
            self._mock_recognizer = None

        # Streaming buffers
        self._stream_audio = bytearray()
        self._stream_text: str = ""

    # ------------------------------------------------------------------
    # Public ASR interface
    # ------------------------------------------------------------------
    def recognize(self, audio: bytes) -> str:
        """Recognize the entire audio buffer at once."""
        if not audio:
            return ""

        speech = self._pcm_to_float(audio)

        if self.mode == "offline":
            return self._run_coro(self._offline_decode(speech, self.TARGET_SAMPLE_RATE))
        return self._run_coro(self._streaming_decode(speech))

    async def async_recognize(self, audio: bytes) -> str:
        """Asynchronously recognize the entire audio buffer."""
        if not audio:
            return ""

        speech = self._pcm_to_float(audio)

        if self.mode == "offline":
            return await self._offline_decode(speech, self.TARGET_SAMPLE_RATE)
        return await self._streaming_decode(speech)

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """
        Streaming recognition interface.

        - offline mode: MockStreamRecognizer performs incremental calls to the
          offline service, manages stable prefixes, and returns accumulated text.
        - streaming mode: each call decodes the buffered audio via the streaming
          service and returns the server's current transcript (which may update).
        """
        del chat_history
        if self.mode == "offline":
            if not self._mock_recognizer:
                return ""
            if not audio:
                return self._mock_recognizer.recognized_text
            return self._mock_recognizer.recognize(audio, is_final=is_final)

        # streaming mode
        if audio:
            self._stream_audio.extend(audio)
        if not self._stream_audio:
            return self._stream_text

        speech = self._pcm_to_float(bytes(self._stream_audio))
        try:
            self._stream_text = self._run_coro(self._streaming_decode(speech))
            return self._stream_text
        except Exception as e:  # pragma: no cover - runtime guard
            logger.error("SherpaOnnxASR streaming recognize_stream failed: %s", e)
            raise RuntimeError(f"SherpaOnnxASR streaming recognize_stream failed: {e}")

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Asynchronous streaming recognition."""
        del chat_history
        if self.mode == "offline":
            if not self._mock_recognizer:
                return ""
            if not audio:
                return self._mock_recognizer.recognized_text
            # Use MockStreamRecognizer's async API directly
            return await self._mock_recognizer.async_recognize(audio, is_final=is_final)  # type: ignore[union-attr]

        # streaming mode
        if audio:
            self._stream_audio.extend(audio)
        if not self._stream_audio:
            return self._stream_text

        speech = self._pcm_to_float(bytes(self._stream_audio))
        try:
            self._stream_text = await self._streaming_decode(speech)
            return self._stream_text
        except Exception as e:  # pragma: no cover - runtime guard
            logger.error("SherpaOnnxASR async streaming recognize_stream failed: %s", e)
            raise RuntimeError(
                f"SherpaOnnxASR async streaming recognize_stream failed: {e}"
            )

    def stream_chunk_bytes_hint(self) -> int | None:
        """
        Return a suggested trigger size for streaming (bytes).

        Defaults to roughly 0.6s per trigger (about 19200 bytes).
        """
        secs = 0.6
        return int(self.TARGET_SAMPLE_RATE * 2 * secs)

    def reset(self) -> None:
        """Reset streaming buffers."""
        if self.mode == "offline":
            if self._mock_recognizer is not None:
                self._mock_recognizer.reset()
        else:
            self._stream_audio.clear()
            self._stream_text = ""

    def clone(self) -> "SherpaOnnxASR":
        """Create a clone that reuses the remote config but keeps separate state."""
        return SherpaOnnxASR(
            base_url=self.base_url,
            mode=self.mode,
            samples_per_message=self.samples_per_message,
            seconds_per_message=self.seconds_per_message,
            offline_payload_len=self.offline_payload_len,
            mock_window_size=self.mock_window_size,
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _pcm_to_float(self, pcm: bytes) -> np.ndarray:
        """Convert PCM int16 bytes to float32 samples (-1~1)."""
        if not pcm:
            return np.zeros((0,), dtype=np.float32)
        x = np.frombuffer(pcm, dtype=np.int16)
        return x.astype(np.float32) / 32768.0

    def _run_coro(self, coro: "asyncio.Future[str]") -> str:
        """Synchronously run a coroutine inside a new event loop."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    # -------------------- offline websocket --------------------
    async def _offline_decode(self, samples: np.ndarray, sample_rate: int) -> str:
        """Call sherpa-onnx-offline-websocket-server for a decode request."""
        uri = self.base_url
        async with websockets.connect(uri) as websocket:  # type: ignore[arg-type]
            assert samples.dtype == np.float32
            assert samples.ndim == 1

            buf = bytearray()
            buf.extend(sample_rate.to_bytes(4, byteorder="little", signed=False))
            buf.extend((samples.size * 4).to_bytes(4, byteorder="little", signed=False))
            buf.extend(samples.tobytes())

            payload_len = self.offline_payload_len
            while len(buf) > payload_len:
                await websocket.send(bytes(buf[:payload_len]))
                del buf[:payload_len]
            if buf:
                await websocket.send(bytes(buf))

            decoding_results = await websocket.recv()
            # Tell the server this session is over
            await websocket.send("Done")

        if isinstance(decoding_results, bytes):
            return decoding_results.decode("utf-8", errors="ignore")
        return str(decoding_results)

    async def _recognize_offline_async(self, audio: bytes) -> str:
        """One-shot async recognition used by MockStreamRecognizer."""
        if not audio:
            return ""
        speech = self._pcm_to_float(audio)
        return await self._offline_decode(speech, self.TARGET_SAMPLE_RATE)

    # -------------------- streaming websocket --------------------
    async def _streaming_decode(self, samples: np.ndarray) -> str:
        """Call sherpa-onnx-online-websocket-server for streaming decode."""
        uri = self.base_url
        async with websockets.connect(uri) as websocket:  # type: ignore[arg-type]

            async def _receiver() -> str:
                last_message = ""
                async for message in websocket:
                    if message == "Done!":
                        break
                    last_message = (
                        message.decode("utf-8", errors="ignore")
                        if isinstance(message, (bytes, bytearray))
                        else str(message)
                    )
                return last_message

            receive_task = asyncio.create_task(_receiver())

            start = 0
            total = samples.shape[0]
            step = max(1, self.samples_per_message)

            while start < total:
                end = min(start + step, total)
                chunk = samples[start:end]
                await websocket.send(chunk.astype(np.float32).tobytes())
                if self.seconds_per_message > 0:
                    await asyncio.sleep(self.seconds_per_message)
                start = end

            # Notify the server that the upload is complete
            await websocket.send("Done")

            decoding_results = await receive_task

        return decoding_results
