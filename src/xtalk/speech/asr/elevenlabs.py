from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlencode

import aiohttp
import requests

from ..interfaces import ASR
from ...log_utils import logger
from ..utils import MockStreamRecognizer

try:  # Optional dependency for realtime streaming
    import websockets  # type: ignore
except Exception as exc:  # pragma: no cover - dependency missing
    websockets = None  # type: ignore
    _ws_import_error = exc
else:  # pragma: no cover
    _ws_import_error = None


class ElevenLabsASR(ASR):
    """ElevenLabs speech-to-text wrapper with offline and realtime modes."""

    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        mode: Literal["offline", "streaming"] = "offline",
        model_id: str = "scribe_v1",
        language_code: Optional[str] = None,
        base_url: str = "https://api.elevenlabs.io",
        realtime_url: str = "wss://api.elevenlabs.io",
        commit_strategy: Literal["manual", "vad"] = "manual",
        include_timestamps: Optional[bool] = None,
        include_language_detection: Optional[bool] = None,
        enable_logging: Optional[bool] = None,
        mock_window_size: int = 5,
        request_timeout: float = 30.0,
        always_commit_stream: bool = True,
    ) -> None:
        """Initialize the ElevenLabs ASR client."""

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY or pass api_key explicitly."
            )

        mode_normalized = mode.strip().lower()
        if mode_normalized not in {"offline", "streaming"}:
            raise ValueError(f"Unsupported ElevenLabs ASR mode: {mode!r}")
        if mode_normalized == "streaming" and websockets is None:
            raise ImportError(
                f"websockets is required for realtime ElevenLabs ASR: {_ws_import_error}"
            )

        self.mode = mode_normalized
        self.model_id = model_id
        self.language_code = language_code
        self.base_url = base_url.rstrip("/")
        self.realtime_url = realtime_url.rstrip("/")
        self.audio_format = "pcm_16000"
        self.file_format = "pcm_s16le_16"
        self.commit_strategy = commit_strategy
        self.include_timestamps = include_timestamps
        self.include_language_detection = include_language_detection
        self.enable_logging = enable_logging
        self.request_timeout = float(request_timeout)
        self.always_commit_stream = always_commit_stream
        self._mock_window_size = mock_window_size

        # Offline streaming uses MockStreamRecognizer to simulate incremental results
        self._mock_recognizer: Optional[MockStreamRecognizer]
        if self.mode == "offline":
            self._mock_recognizer = MockStreamRecognizer(
                self._async_offline_transcribe, window_size=mock_window_size
            )
        else:
            self._mock_recognizer = None

        # Streaming buffers for the manual re-upload strategy
        self._stream_audio = bytearray()
        self._stream_text = ""

    # ------------------------------------------------------------------
    # Public ASR interface
    # ------------------------------------------------------------------
    def recognize(self, audio: bytes) -> str:
        if not audio:
            return ""
        if self.mode == "offline":
            return self._offline_transcribe(audio)
        return self._run_coro(self._realtime_transcribe(audio, commit=True))

    async def async_recognize(self, audio: bytes) -> str:
        if not audio:
            return ""
        if self.mode == "offline":
            return await self._async_offline_transcribe(audio)
        return await self._realtime_transcribe(audio, commit=True)

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        if self.mode == "offline":
            if not self._mock_recognizer:
                return ""
            if not audio:
                return self._mock_recognizer.recognized_text
            return self._mock_recognizer.recognize(audio, is_final=is_final)

        if audio:
            self._stream_audio.extend(audio)
        if not self._stream_audio:
            return self._stream_text

        commit_flag = self.always_commit_stream or is_final
        try:
            self._stream_text = self._run_coro(
                self._realtime_transcribe(bytes(self._stream_audio), commit=commit_flag)
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.error("ElevenLabs streaming recognize_stream failed: %s", exc)
            raise

        if is_final:
            self._stream_audio.clear()
        return self._stream_text

    async def async_recognize_stream(
        self, audio: bytes, *, is_final: bool = False
    ) -> str:
        if self.mode == "offline":
            if not self._mock_recognizer:
                return ""
            if not audio:
                return self._mock_recognizer.recognized_text
            return await self._mock_recognizer.async_recognize(audio, is_final=is_final)  # type: ignore[union-attr]

        if audio:
            self._stream_audio.extend(audio)
        if not self._stream_audio:
            return self._stream_text

        commit_flag = self.always_commit_stream or is_final
        self._stream_text = await self._realtime_transcribe(
            bytes(self._stream_audio), commit=commit_flag
        )
        if is_final:
            self._stream_audio.clear()
        return self._stream_text

    def stream_chunk_bytes_hint(self) -> int | None:
        secs = 0.6
        return int(self.TARGET_SAMPLE_RATE * 2 * secs)

    def reset(self) -> None:
        if self.mode == "offline":
            if self._mock_recognizer is not None:
                self._mock_recognizer.reset()
        else:
            self._stream_audio.clear()
            self._stream_text = ""

    def clone(self) -> "ElevenLabsASR":
        return ElevenLabsASR(
            api_key=self.api_key,
            mode=self.mode,  # type: ignore[arg-type]
            model_id=self.model_id,
            language_code=self.language_code,
            base_url=self.base_url,
            realtime_url=self.realtime_url,
            commit_strategy=self.commit_strategy,
            include_timestamps=self.include_timestamps,
            include_language_detection=self.include_language_detection,
            enable_logging=self.enable_logging,
            mock_window_size=(
                self._mock_recognizer.window_size
                if self._mock_recognizer
                else self._mock_window_size
            ),
            request_timeout=self.request_timeout,
            always_commit_stream=self.always_commit_stream,
        )

    # ------------------------------------------------------------------
    # Offline REST helpers
    # ------------------------------------------------------------------
    def _offline_transcribe(self, audio: bytes) -> str:
        url = f"{self.base_url}/v1/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        data = self._build_form_fields()
        files = {
            "file": (
                "audio.pcm",
                audio,
                "application/octet-stream",
            )
        }
        params = self._build_query_params()
        try:
            resp = requests.post(
                url,
                headers=headers,
                data=data,
                files=files,
                params=params,
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network guard
            raise RuntimeError(
                f"ElevenLabs offline transcription failed: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs offline transcription failed: HTTP {resp.status_code} {resp.text[:200]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid ElevenLabs transcription response: {resp.text}"
            ) from exc

        return self._extract_text(payload)

    async def _async_offline_transcribe(self, audio: bytes) -> str:
        url = f"{self.base_url}/v1/speech-to-text"
        headers = {"xi-api-key": self.api_key}
        params = self._build_query_params()
        form = aiohttp.FormData()
        for key, value in self._build_form_fields().items():
            if value is None:
                continue
            form.add_field(key, str(value))
        form.add_field(
            "file",
            audio,
            content_type="application/octet-stream",
            filename="audio.pcm",
        )
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                async with session.post(
                    url, headers=headers, params=params, data=form
                ) as resp:
                    body = await resp.text()
                    if resp.status != 200:
                        raise RuntimeError(
                            f"ElevenLabs offline transcription failed: HTTP {resp.status} {body[:200]}"
                        )
                    try:
                        payload = json.loads(body)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Invalid ElevenLabs transcription response: {body}"
                        ) from exc
                    return self._extract_text(payload)
            except aiohttp.ClientError as exc:  # pragma: no cover - network guard
                raise RuntimeError(
                    f"ElevenLabs offline transcription request failed: {exc}"
                ) from exc

    def _build_form_fields(self) -> Dict[str, Any]:
        fields: Dict[str, Any] = {
            "model_id": self.model_id,
            "file_format": self.file_format,
        }
        if self.language_code:
            fields["language_code"] = self.language_code
        return fields

    def _build_query_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self.enable_logging is not None:
            params["enable_logging"] = "true" if self.enable_logging else "false"
        return params

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        if "text" in payload and isinstance(payload["text"], str):
            return payload["text"]
        transcripts = payload.get("transcripts")
        if isinstance(transcripts, list):
            parts = [
                item.get("text", "") for item in transcripts if isinstance(item, dict)
            ]
            return "\n".join(filter(None, parts))
        message = payload.get("message")
        if isinstance(message, str):
            return message
        return json.dumps(payload, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Realtime WebSocket helpers
    # ------------------------------------------------------------------
    async def _realtime_transcribe(self, audio: bytes, *, commit: bool) -> str:
        if websockets is None:
            raise RuntimeError(
                "websockets dependency missing for realtime ElevenLabs ASR"
            )
        if not audio:
            return ""

        uri = self._build_realtime_uri()
        headers = [("xi-api-key", self.api_key)]
        try:
            async with websockets.connect(uri, extra_headers=headers) as websocket:  # type: ignore[arg-type]
                await self._send_audio_chunk(websocket, audio, commit=commit)
                return await self._consume_transcripts(websocket, expect_commit=commit)
        except Exception as exc:  # pragma: no cover - network guard
            raise RuntimeError(
                f"ElevenLabs realtime transcription failed: {exc}"
            ) from exc

    async def _send_audio_chunk(
        self, websocket: Any, audio: bytes, *, commit: bool
    ) -> None:
        payload = {
            "message_type": "input_audio_chunk",
            "audio_base_64": base64.b64encode(audio).decode("ascii"),
            "commit": commit,
            "sample_rate": self.TARGET_SAMPLE_RATE,
        }
        await websocket.send(json.dumps(payload))

    async def _consume_transcripts(self, websocket: Any, *, expect_commit: bool) -> str:
        transcript = ""
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.recv(), timeout=self.request_timeout
                )
            except asyncio.TimeoutError:
                break
            data = self._decode_ws_message(message)
            msg_type = data.get("message_type")
            if msg_type == "partial_transcript":
                transcript = data.get("text", "")
                if not expect_commit:
                    break
            elif msg_type in {
                "committed_transcript",
                "committed_transcript_with_timestamps",
            }:
                transcript = data.get("text", "")
                break
            elif msg_type and msg_type.endswith("error"):
                raise RuntimeError(
                    f"ElevenLabs realtime error: {data.get('error', data)}"
                )
            elif msg_type == "error":
                raise RuntimeError(
                    f"ElevenLabs realtime error: {data.get('error', data)}"
                )
        return transcript

    def _decode_ws_message(self, message: Any) -> Dict[str, Any]:
        if isinstance(message, (bytes, bytearray)):
            text = message.decode("utf-8", errors="ignore")
        else:
            text = str(message)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"message_type": "unknown", "text": text}

    def _build_realtime_uri(self) -> str:
        query: Dict[str, Any] = {
            "model_id": self.model_id,
            "audio_format": self.audio_format,
            "commit_strategy": self.commit_strategy,
        }
        if self.language_code:
            query["language_code"] = self.language_code
        if self.include_timestamps is not None:
            query["include_timestamps"] = str(self.include_timestamps).lower()
        if self.include_language_detection is not None:
            query["include_language_detection"] = str(
                self.include_language_detection
            ).lower()
        if self.enable_logging is not None:
            query["enable_logging"] = str(self.enable_logging).lower()
        encoded = urlencode(query)
        return f"{self.realtime_url}/v1/speech-to-text/realtime?{encoded}"

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _run_coro(self, coro: "asyncio.Future[str]") -> str:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
