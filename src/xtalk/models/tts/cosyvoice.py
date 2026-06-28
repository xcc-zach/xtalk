from __future__ import annotations

import os
import queue
import threading
from typing import Any, Dict, Iterable, Optional

import certifi
import dashscope
from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

from .interfaces import TTS
from ..registry import model


def _ensure_ssl_cert_file() -> None:
    """Ensure WebSocket TLS verification can find a CA bundle."""

    if not os.getenv("SSL_CERT_FILE"):
        os.environ["SSL_CERT_FILE"] = certifi.where()


class _QueueingCallback(ResultCallback):
    """Pushes streaming PCM chunks into a queue; signals completion via sentinel."""

    def __init__(self, chunk_queue: queue.Queue[object], sentinel: object) -> None:
        super().__init__()
        self._queue = chunk_queue
        self._sentinel = sentinel

    def on_data(self, data: bytes) -> None:
        if data:
            self._queue.put(data)

    def on_error(self, message: str) -> None:
        self._queue.put(RuntimeError(f"CosyVoice streaming error: {message}"))
        self._queue.put(self._sentinel)

    def on_complete(self) -> None:
        self._queue.put(self._sentinel)


@model
class CosyVoice(TTS):
    """
    DashScope CosyVoice TTS implementation using SpeechSynthesizer.
    - Forces PCM 48kHz mono s16: AudioFormat.PCM_48000HZ_MONO_16BIT
    - synthesize(): true synchronous call (no callback) returning full bytes
    - synthesize_stream(): callback-driven streaming generator
    """

    DEFAULT_MODEL = "cosyvoice-v3-flash"
    DEFAULT_VOICE = "longanyang"
    DEFAULT_SAMPLE_RATE = 48000

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        stream_timeout: float = 30.0,
        extra_request_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY or pass api_key."
            )

        _ensure_ssl_cert_file()
        dashscope.api_key = self.api_key

        self.model = model
        self.voice = voice
        self._audio_format = AudioFormat.PCM_48000HZ_MONO_16BIT  # force PCM 48k
        self._sample_rate = self.DEFAULT_SAMPLE_RATE
        self._stream_timeout = float(stream_timeout)
        self._extra_request_params: Dict[str, Any] = (
            extra_request_params.copy() if extra_request_params else {}
        )

    def clone(self) -> "CosyVoice":
        return CosyVoice(
            api_key=self.api_key,
            model=self.model,
            voice=self.voice,
            stream_timeout=self._stream_timeout,
            extra_request_params=self._extra_request_params.copy(),
        )

    # -------------------------
    # True synchronous synthesis
    # -------------------------
    def synthesize(self, text: str) -> bytes:
        if not text:
            raise ValueError("Text for CosyVoice synthesis cannot be empty.")

        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=self._audio_format,
            **self._extra_request_params,
        )

        try:
            audio = synthesizer.call(text)  # blocking; returns bytes in sync mode
        except Exception as exc:
            raise RuntimeError(f"CosyVoice synthesis failed: {exc}") from exc

        if not audio:
            raise RuntimeError("CosyVoice returned no audio data.")

        return audio

    # -------------------------
    # Callback-driven streaming
    # -------------------------
    def synthesize_stream(
        self, text: str, chunk_timeout: Optional[float] = None
    ) -> Iterable[bytes]:
        if not text:
            raise ValueError("Text for CosyVoice synthesis cannot be empty.")

        timeout = (
            self._stream_timeout if chunk_timeout is None else float(chunk_timeout)
        )
        chunk_queue: queue.Queue[object] = queue.Queue()
        sentinel = object()

        def _worker() -> None:
            callback = _QueueingCallback(chunk_queue, sentinel)
            synthesizer = SpeechSynthesizer(
                model=self.model,
                voice=self.voice,
                format=self._audio_format,
                callback=callback,
                **self._extra_request_params,
            )
            try:
                synthesizer.call(text)  # returns quickly; callback keeps pushing data
            except Exception as exc:
                chunk_queue.put(RuntimeError(f"CosyVoice streaming failed: {exc}"))
                chunk_queue.put(sentinel)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            try:
                item = chunk_queue.get(timeout=timeout)
            except queue.Empty:
                raise TimeoutError("Timed out while waiting for CosyVoice audio chunk.")

            if item is sentinel:
                break

            if isinstance(item, Exception):
                raise item

            if item:
                yield item
