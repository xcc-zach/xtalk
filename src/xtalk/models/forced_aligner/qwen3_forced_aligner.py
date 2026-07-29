from __future__ import annotations

import base64
import io
import unicodedata
import wave
from urllib.parse import urlsplit

import numpy as np
import requests

from .interfaces import ForcedAligner, ForcedAlignmentUnit
from ..registry import model


_DEFAULT_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
_FORCE_ALIGNMENT_SAMPLE_RATE = 48_000
_TIMESTAMP_TOKEN_ID = 151_705
_TIMESTAMP_SEGMENT_MS = 80.0
_AUDIO_PAD_TOKEN_ID = 151_676
_AUDIO_START_TOKEN = "<|audio_start|>"
_AUDIO_PAD_TOKEN = "<|audio_pad|>"
_AUDIO_END_TOKEN = "<|audio_end|>"
_TIMESTAMP_TOKEN = "<timestamp>"
_RAW_CONTENT_CHAT_TEMPLATE = "{{ messages[0]['content'] }}"
_REQUEST_TIMEOUT_SECONDS = 120.0


def _normalize_base_url(base_url: str) -> str:
    """Normalize and validate a vLLM server base URL."""

    normalized = base_url.strip().rstrip("/")
    parts = urlsplit(normalized)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    return normalized


@model
class Qwen3ForcedAligner(ForcedAligner):
    """Client for a vLLM-hosted Qwen3 forced-alignment model.

    The client sends in-memory 48 kHz PCM WAV data to vLLM's pooling API and
    converts token-classification logits into millisecond alignment units.
    Model protocol constants are fixed for Qwen3-ForcedAligner-0.6B so the
    client does not need local model files or internet access.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str = _DEFAULT_MODEL,
        language: str | None = None,
    ) -> None:
        """Initialize the vLLM forced-aligner client.

        Parameters
        ----------
        base_url : str
            Root URL of the vLLM server, for example
            ``http://127.0.0.1:8001``.
        model : str, optional
            Served model name for the Qwen3-ForcedAligner-0.6B checkpoint.
        language : str | None, optional
            Default language hint used for client-side alignment-unit
            segmentation. ``None`` selects generic segmentation.
        """

        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.language = language
        self._tokenize_url = f"{self.base_url}/tokenize"
        self._pooling_url = f"{self.base_url}/pooling"

    def align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """Align reference text against 48 kHz PCM audio through vLLM.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 48 kHz.
        text : str
            Reference text spoken by the audio.
        language : str | None, optional
            Per-request language hint overriding the configured default.

        Returns
        -------
        list[ForcedAlignmentUnit]
            Character- or word-level alignment units in milliseconds.
        """

        if not audio or not text.strip():
            return []

        resolved_language = language or self.language
        units = self._split_text_units(text, resolved_language)
        if not units:
            return []

        prompt = self._build_prompt(units)
        prompt_token_ids = self._tokenize_prompt(prompt)
        logits = self._request_logits(
            prompt=prompt,
            audio_uri=self._encode_audio_uri(audio),
        )
        predictions = logits.argmax(axis=-1)
        timestamp_ms = self._extract_timestamp_ms(
            prompt_token_ids=prompt_token_ids,
            predictions=predictions,
            unit_count=len(units),
        )
        return [
            ForcedAlignmentUnit(
                text=unit,
                start_ms=timestamp_ms[index * 2],
                end_ms=timestamp_ms[index * 2 + 1],
            )
            for index, unit in enumerate(units)
        ]

    def clone(self) -> "Qwen3ForcedAligner":
        """Clone the vLLM client configuration for a new service session."""

        return Qwen3ForcedAligner(
            base_url=self.base_url,
            model=self.model,
            language=self.language,
        )

    def _tokenize_prompt(self, prompt: str) -> list[int]:
        response = requests.post(
            self._tokenize_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": False,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise RuntimeError("vLLM /tokenize returned invalid token IDs")
        if _AUDIO_PAD_TOKEN_ID not in tokens or _TIMESTAMP_TOKEN_ID not in tokens:
            raise RuntimeError(
                "vLLM tokenizer is incompatible with Qwen3-ForcedAligner-0.6B"
            )
        return tokens

    def _request_logits(self, *, prompt: str, audio_uri: str) -> np.ndarray:
        response = requests.post(
            self._pooling_url,
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "audio_url",
                                "audio_url": {"url": audio_uri},
                            },
                        ],
                    }
                ],
                "task": "token_classify",
                "chat_template": _RAW_CONTENT_CHAT_TEMPLATE,
            },
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or len(data) != 1:
            raise RuntimeError("vLLM /pooling returned invalid output data")
        output = data[0]
        logits_data = output.get("data") if isinstance(output, dict) else None
        if not isinstance(logits_data, list):
            raise RuntimeError("vLLM /pooling response does not contain logits")

        logits = np.asarray(logits_data, dtype=np.float32)
        if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
            raise RuntimeError("vLLM /pooling returned an invalid logits matrix")
        return logits

    @staticmethod
    def _build_prompt(units: list[str]) -> str:
        body = f"{_TIMESTAMP_TOKEN}{_TIMESTAMP_TOKEN}".join(units)
        body += f"{_TIMESTAMP_TOKEN}{_TIMESTAMP_TOKEN}"
        return f"{_AUDIO_START_TOKEN}{_AUDIO_PAD_TOKEN}{_AUDIO_END_TOKEN}{body}"

    @staticmethod
    def _encode_audio_uri(audio: bytes) -> str:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(_FORCE_ALIGNMENT_SAMPLE_RATE)
            wav_file.writeframes(audio)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    @classmethod
    def _extract_timestamp_ms(
        cls,
        *,
        prompt_token_ids: list[int],
        predictions: np.ndarray,
        unit_count: int,
    ) -> list[float]:
        audio_pad_index = prompt_token_ids.index(_AUDIO_PAD_TOKEN_ID)
        audio_token_shift = len(predictions) - len(prompt_token_ids)
        if audio_token_shift < 0:
            raise RuntimeError(
                "vLLM pooling output is shorter than the tokenized prompt"
            )

        timestamp_ms: list[float] = []
        for index, token_id in enumerate(prompt_token_ids):
            if token_id != _TIMESTAMP_TOKEN_ID:
                continue
            prediction_index = (
                index + audio_token_shift if index > audio_pad_index else index
            )
            if prediction_index >= len(predictions):
                raise RuntimeError(
                    "vLLM timestamp prediction index exceeds pooling output"
                )
            timestamp_ms.append(
                float(predictions[prediction_index]) * _TIMESTAMP_SEGMENT_MS
            )

        expected_count = unit_count * 2
        if len(timestamp_ms) != expected_count:
            raise RuntimeError(
                "vLLM returned "
                f"{len(timestamp_ms)} timestamps for {unit_count} alignment units"
            )
        return cls._repair_timestamps(timestamp_ms)

    @staticmethod
    def _repair_timestamps(timestamp_ms: list[float]) -> list[float]:
        if len(timestamp_ms) < 2:
            return timestamp_ms

        count = len(timestamp_ms)
        lengths = [1] * count
        parents = [-1] * count
        for index in range(1, count):
            for previous in range(index):
                if (
                    timestamp_ms[previous] <= timestamp_ms[index]
                    and lengths[previous] + 1 > lengths[index]
                ):
                    lengths[index] = lengths[previous] + 1
                    parents[index] = previous

        longest_index = max(range(count), key=lengths.__getitem__)
        normal_indices: set[int] = set()
        while longest_index != -1:
            normal_indices.add(longest_index)
            longest_index = parents[longest_index]

        repaired = list(timestamp_ms)
        index = 0
        while index < count:
            if index in normal_indices:
                index += 1
                continue

            end = index
            while end < count and end not in normal_indices:
                end += 1
            anomaly_count = end - index
            left = repaired[index - 1] if index > 0 else None
            right = repaired[end] if end < count else None

            if anomaly_count <= 2:
                for anomaly_index in range(index, end):
                    if left is None:
                        assert right is not None
                        repaired[anomaly_index] = right
                    elif right is None:
                        repaired[anomaly_index] = left
                    else:
                        left_distance = anomaly_index - index + 1
                        right_distance = end - anomaly_index
                        repaired[anomaly_index] = (
                            left if left_distance <= right_distance else right
                        )
            elif left is not None and right is not None:
                step = (right - left) / (anomaly_count + 1)
                for anomaly_index in range(index, end):
                    repaired[anomaly_index] = left + step * (anomaly_index - index + 1)
            elif left is not None:
                for anomaly_index in range(index, end):
                    repaired[anomaly_index] = left
            elif right is not None:
                for anomaly_index in range(index, end):
                    repaired[anomaly_index] = right
            index = end

        return repaired

    @classmethod
    def _split_text_units(
        cls,
        text: str,
        language: str | None,
    ) -> list[str]:
        if language and language.strip().lower() == "japanese":
            return cls._split_japanese_units(text)

        units: list[str] = []
        for segment in text.split():
            cleaned = "".join(
                character
                for character in segment
                if cls._is_alignment_character(character)
            )
            units.extend(cls._split_cjk_segment(cleaned))
        return units

    @classmethod
    def _split_cjk_segment(cls, segment: str) -> list[str]:
        units: list[str] = []
        buffer: list[str] = []
        for character in segment:
            if cls._is_cjk_character(character):
                if buffer:
                    units.append("".join(buffer))
                    buffer.clear()
                units.append(character)
            else:
                buffer.append(character)
        if buffer:
            units.append("".join(buffer))
        return units

    @classmethod
    def _split_japanese_units(cls, text: str) -> list[str]:
        units: list[str] = []
        buffer: list[str] = []
        for character in text:
            if not cls._is_alignment_character(character):
                if buffer:
                    units.append("".join(buffer))
                    buffer.clear()
                continue
            if cls._is_japanese_character(character):
                if buffer:
                    units.append("".join(buffer))
                    buffer.clear()
                units.append(character)
            else:
                buffer.append(character)
        if buffer:
            units.append("".join(buffer))
        return units

    @staticmethod
    def _is_alignment_character(character: str) -> bool:
        return character == "'" or unicodedata.category(character)[0] in {"L", "N"}

    @staticmethod
    def _is_cjk_character(character: str) -> bool:
        codepoint = ord(character)
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2CEAF
        )

    @classmethod
    def _is_japanese_character(cls, character: str) -> bool:
        codepoint = ord(character)
        return (
            cls._is_cjk_character(character)
            or 0x3040 <= codepoint <= 0x30FF
            or 0xFF66 <= codepoint <= 0xFF9D
        )
