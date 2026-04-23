# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import os
import random
import threading
import time
from dataclasses import dataclass
from typing import Optional, Literal

import dashscope
from dashscope.audio.qwen_omni import (
    OmniRealtimeConversation,
    OmniRealtimeCallback,
    MultiModality,
)
from dashscope.audio.qwen_omni.omni_realtime import TranscriptionParams

from ..interfaces import ASR


@dataclass(frozen=True)
class Qwen3ASRFlashConfig:
    model: str = "qwen3-asr-flash-realtime"
    url: str = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    language: str = "zh"
    sample_rate: int = 16000
    input_audio_format: Literal["pcm"] = "pcm"

    enable_turn_detection: bool = True

    stream_chunk_bytes_hint: int = 3200

    tail_silence_cycles: int = 30
    tail_silence_bytes_per_cycle: int = 1024
    tail_silence_delay_sec: float = 0.01

    send_delay_sec: float = 0.0
    final_wait_timeout_sec: float = 15.0

    connect_retry_max_attempts: int = 5
    connect_retry_base_delay_sec: float = 0.6
    connect_retry_max_delay_sec: float = 6.0
    connect_retry_jitter_ratio: float = 0.2

    reconnect_on_send_error_max_attempts: int = 2


class _RealtimeASRCallback(OmniRealtimeCallback):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._final_event = threading.Event()
        self._last_partial: str = ""
        self._final_text: str = ""
        self._session_id: str = ""

        self.handlers = {
            "session.created": self._handle_session_created,
            "conversation.item.input_audio_transcription.text": self._handle_partial,
            "conversation.item.input_audio_transcription.completed": self._handle_final,
            "input_audio_buffer.speech_started": lambda r: None,
            "input_audio_buffer.speech_stopped": lambda r: None,
        }

    def on_open(self):
        return

    def on_close(self, code, msg):
        return

    def on_event(self, response):
        try:
            handler = self.handlers.get(response.get("type"))
            if handler:
                handler(response)
        except Exception:
            return

    def _handle_session_created(self, response):
        try:
            sid = response["session"]["id"]
        except Exception:
            sid = ""
        with self._lock:
            self._session_id = sid

    def _handle_partial(self, response):
        partial = (
            response.get("stash")
            or response.get("transcript")
            or response.get("text")
            or ""
        )
        with self._lock:
            self._last_partial = partial

    def _handle_final(self, response):
        final = response.get("transcript") or response.get("text") or ""
        with self._lock:
            self._final_text = final
            self._final_event.set()

    def clear_turn(self) -> None:
        with self._lock:
            self._last_partial = ""
            self._final_text = ""
        self._final_event.clear()

    def get_last_partial(self) -> str:
        with self._lock:
            return self._last_partial

    def wait_final(self, timeout: float) -> Optional[str]:
        ok = self._final_event.wait(timeout=timeout)
        if not ok:
            return None
        with self._lock:
            return self._final_text


class Qwen3ASRFlashRealtime(ASR):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        config: Optional[Qwen3ASRFlashConfig] = None,
    ) -> None:
        self._config = config or Qwen3ASRFlashConfig()
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY") or ""
        if not self._api_key:
            raise ValueError("DashScope API key is required.")
        dashscope.api_key = self._api_key

        self._lock = threading.Lock()
        self._callback: Optional[_RealtimeASRCallback] = None
        self._conv: Optional[OmniRealtimeConversation] = None
        self._connected: bool = False

    def recognize(self, audio: bytes) -> str:
        conv, cb = self._create_conversation()
        connect_ok = False
        try:
            self._connect_and_configure_with_retry(conv)
            connect_ok = True
            cb.clear_turn()

            self._append_audio_bytes(conv, audio)
            self._send_tail_silence(conv)

            if not self._config.enable_turn_detection:
                conv.commit()

            final = cb.wait_final(timeout=self._config.final_wait_timeout_sec)
            return final if final is not None else cb.get_last_partial()
        finally:
            self._safe_close_if_connected(conv, connect_ok)

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        self._ensure_session()

        assert self._conv is not None
        assert self._callback is not None

        if audio:
            self._append_audio_bytes_with_reconnect(self._conv, audio)

        if not is_final:
            return self._callback.get_last_partial()

        self._send_tail_silence_with_reconnect(self._conv)

        if not self._config.enable_turn_detection:
            try:
                self._conv.commit()
            except Exception:
                self.reset()
                self._ensure_session()
                assert self._conv is not None
                self._conv.commit()

        final = self._callback.wait_final(timeout=self._config.final_wait_timeout_sec)
        return final if final is not None else self._callback.get_last_partial()

    def stream_chunk_bytes_hint(self) -> int | None:
        return self._config.stream_chunk_bytes_hint

    def reset(self) -> None:
        with self._lock:
            conv = self._conv
            was_connected = self._connected
            self._conv = None
            self._callback = None
            self._connected = False
        if conv is not None:
            self._safe_close_if_connected(conv, was_connected)

    def clone(self) -> "ASR":
        return Qwen3ASRFlashRealtime(api_key=self._api_key, config=self._config)

    def _ensure_session(self) -> None:
        with self._lock:
            if self._conv is not None and self._connected:
                return
            cb = _RealtimeASRCallback()
            conv = OmniRealtimeConversation(
                model=self._config.model,
                url=self._config.url,
                callback=cb,
            )
            self._callback = cb
            self._conv = conv
            self._connected = False

        try:
            assert self._conv is not None
            self._connect_and_configure_with_retry(self._conv)
            assert self._callback is not None
            self._callback.clear_turn()
            with self._lock:
                self._connected = True
        except Exception:
            self.reset()
            raise

    def _create_conversation(
        self,
    ) -> tuple[OmniRealtimeConversation, _RealtimeASRCallback]:
        cb = _RealtimeASRCallback()
        conv = OmniRealtimeConversation(
            model=self._config.model,
            url=self._config.url,
            callback=cb,
        )
        return conv, cb

    def _connect_and_configure_with_retry(self, conv: OmniRealtimeConversation) -> None:
        last_err: Optional[BaseException] = None

        for attempt in range(1, self._config.connect_retry_max_attempts + 1):
            try:
                self._connect_and_configure(conv)
                return
            except Exception as e:
                last_err = e

                if attempt >= self._config.connect_retry_max_attempts:
                    break

                delay = min(
                    self._config.connect_retry_base_delay_sec * (2 ** (attempt - 1)),
                    self._config.connect_retry_max_delay_sec,
                )
                jitter = delay * self._config.connect_retry_jitter_ratio
                time.sleep(max(0.0, delay + random.uniform(-jitter, jitter)))

                conv = OmniRealtimeConversation(
                    model=self._config.model,
                    url=self._config.url,
                    callback=getattr(conv, "callback", None),
                )

        msg = str(last_err) if last_err is not None else ""
        raise RuntimeError(
            f"websocket connect failed after retries: {msg}"
        ) from last_err

    def _connect_and_configure(self, conv: OmniRealtimeConversation) -> None:
        conv.connect()
        tp = TranscriptionParams(
            language=self._config.language,
            sample_rate=self._config.sample_rate,
            input_audio_format=self._config.input_audio_format,
        )
        conv.update_session(
            output_modalities=[MultiModality.TEXT],
            enable_input_audio_transcription=True,
            transcription_params=tp,
        )

    def _safe_close_if_connected(
        self, conv: OmniRealtimeConversation, connected: bool
    ) -> None:
        if not connected:
            return
        try:
            conv.close()
        except Exception:
            return

    def _append_audio_bytes_with_reconnect(
        self, conv: OmniRealtimeConversation, audio: bytes
    ) -> None:
        last_err: Optional[BaseException] = None
        for attempt in range(self._config.reconnect_on_send_error_max_attempts + 1):
            try:
                self._append_audio_bytes(conv, audio)
                return
            except Exception as e:
                last_err = e
                if attempt >= self._config.reconnect_on_send_error_max_attempts:
                    break
                self.reset()
                self._ensure_session()
                assert self._conv is not None
                conv = self._conv
        msg = str(last_err) if last_err is not None else ""
        raise RuntimeError(
            f"append_audio failed after reconnect attempts: {msg}"
        ) from last_err

    def _send_tail_silence_with_reconnect(self, conv: OmniRealtimeConversation) -> None:
        last_err: Optional[BaseException] = None
        for attempt in range(self._config.reconnect_on_send_error_max_attempts + 1):
            try:
                self._send_tail_silence(conv)
                return
            except Exception as e:
                last_err = e
                if attempt >= self._config.reconnect_on_send_error_max_attempts:
                    break
                self.reset()
                self._ensure_session()
                assert self._conv is not None
                conv = self._conv
        msg = str(last_err) if last_err is not None else ""
        raise RuntimeError(
            f"send_tail_silence failed after reconnect attempts: {msg}"
        ) from last_err

    def _append_audio_bytes(self, conv: OmniRealtimeConversation, audio: bytes) -> None:
        if not audio:
            return
        b64 = base64.b64encode(audio).decode("ascii")
        conv.append_audio(b64)
        if self._config.send_delay_sec > 0:
            time.sleep(self._config.send_delay_sec)

    def _send_tail_silence(self, conv: OmniRealtimeConversation) -> None:
        silence = bytes(self._config.tail_silence_bytes_per_cycle)
        for _ in range(self._config.tail_silence_cycles):
            b64 = base64.b64encode(silence).decode("ascii")
            conv.append_audio(b64)
            if self._config.tail_silence_delay_sec > 0:
                time.sleep(self._config.tail_silence_delay_sec)
