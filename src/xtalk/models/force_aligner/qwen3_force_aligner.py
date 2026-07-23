from __future__ import annotations

import os
import tempfile
import threading
import wave
from typing import Any, Literal

from .interfaces import ForceAligner, ForceAlignmentUnit
from ..registry import model


@model
class Qwen3ForceAligner(ForceAligner):
    """Qwen3 forced-aligner adapter.

    The adapter accepts X-Talk's PCM bytes, writes a temporary WAV file, calls
    Qwen3ForcedAligner, then normalizes the returned unit timestamps to
    milliseconds.
    """

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        language: str = "Chinese",
        time_unit: Literal["seconds", "milliseconds"] = "seconds",
        device_map: str | None = "auto",
        dtype: str | None = None,
        torch_dtype: str | None = None,
        trust_remote_code: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        eager_load: bool = False,
    ) -> None:
        if time_unit not in {"seconds", "milliseconds"}:
            raise ValueError(f"Unsupported alignment time unit: {time_unit}")
        self.model = model
        self.language = language
        self.time_unit = time_unit
        self.device_map = device_map
        self.dtype = dtype or torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model_kwargs = dict(model_kwargs or {})
        self.eager_load = eager_load

        self._aligner: Any | None = None
        self._lock = threading.Lock()
        if self.eager_load:
            self._ensure_aligner()

    def align(
        self,
        *,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str | None = None,
    ) -> list[ForceAlignmentUnit]:
        if not audio or not text or sample_rate <= 0:
            return []

        aligner = self._ensure_aligner()
        wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = wav_file.name
        wav_file.close()
        try:
            self._write_pcm_wav(
                path=wav_path,
                audio=audio,
                sample_rate=sample_rate,
            )
            with self._lock:
                raw_result = aligner.align(
                    audio=wav_path,
                    text=text,
                    language=language or self.language,
                )
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass
        return self._parse_alignment_result(raw_result)

    def clone(self) -> "Qwen3ForceAligner":
        cloned = Qwen3ForceAligner(
            model=self.model,
            language=self.language,
            time_unit=self.time_unit,
            device_map=self.device_map,
            dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
            model_kwargs=self.model_kwargs,
            eager_load=False,
        )
        cloned._aligner = self._aligner
        cloned.eager_load = self.eager_load
        cloned._lock = self._lock
        return cloned

    def _ensure_aligner(self) -> Any:
        if self._aligner is not None:
            return self._aligner
        with self._lock:
            if self._aligner is not None:
                return self._aligner

            try:
                from qwen_asr import Qwen3ForcedAligner
            except ImportError as exc:
                raise ImportError(
                    "qwen_asr is required for Qwen3ForceAligner. "
                    "Install the qwen3-force-aligner extra."
                ) from exc

            kwargs: dict[str, Any] = dict(self.model_kwargs)
            if self.device_map is not None:
                kwargs.setdefault("device_map", self.device_map)
            if self.trust_remote_code:
                kwargs.setdefault("trust_remote_code", True)
            if self.dtype is not None:
                kwargs.setdefault("dtype", self._resolve_torch_dtype())

            self._aligner = Qwen3ForcedAligner.from_pretrained(
                self.model,
                **kwargs,
            )
            return self._aligner

    def _resolve_torch_dtype(self) -> Any:
        import torch

        dtype = self.dtype
        if dtype is None:
            return None
        if hasattr(torch, dtype):
            return getattr(torch, dtype)
        raise ValueError(f"Unsupported torch dtype: {dtype}")

    @staticmethod
    def _write_pcm_wav(*, path: str, audio: bytes, sample_rate: int) -> None:
        with wave.open(path, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio)

    def _parse_alignment_result(self, raw_result: Any) -> list[ForceAlignmentUnit]:
        items = self._extract_items(raw_result)
        units: list[ForceAlignmentUnit] = []
        for item in items:
            unit_text = str(
                self._read_value(item, "text", "word", "token", "unit")
                or ""
            )
            start = self._read_time(item, "start_time", "start", "begin")
            end = self._read_time(item, "end_time", "end", "stop")
            if start is None or end is None:
                continue
            units.append(
                ForceAlignmentUnit(
                    text=unit_text,
                    start_ms=self._to_ms(start),
                    end_ms=self._to_ms(end),
                )
            )
        return units

    @staticmethod
    def _extract_items(raw_result: Any) -> list[Any]:
        if isinstance(raw_result, list):
            if not raw_result:
                return []
            first_result = raw_result[0]
            if isinstance(first_result, list):
                return first_result
            first_items = getattr(first_result, "items", None)
            if isinstance(first_items, list):
                return first_items
            return raw_result
        if isinstance(raw_result, dict):
            for key in ("words", "segments", "alignment", "timestamps", "result"):
                value = raw_result.get(key)
                if isinstance(value, list):
                    return value
            return []
        result_items = getattr(raw_result, "items", None)
        if isinstance(result_items, list):
            return result_items
        return []

    @staticmethod
    def _read_value(item: Any, *keys: str) -> Any:
        if isinstance(item, dict):
            for key in keys:
                value = item.get(key)
                if value is not None:
                    return value
            return None
        for key in keys:
            if hasattr(item, key):
                return getattr(item, key)
        return None

    @classmethod
    def _read_time(cls, item: Any, *keys: str) -> float | None:
        for key in keys:
            value = cls._read_value(item, key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _to_ms(self, value: float) -> float:
        if self.time_unit == "milliseconds":
            return value
        return value * 1000.0
