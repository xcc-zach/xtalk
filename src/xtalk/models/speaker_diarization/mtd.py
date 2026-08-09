"""Unified HTTP client for official-vLLM and SGLang-Omni MTD runtimes."""

from __future__ import annotations

import asyncio
import io
import math
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping
from urllib.parse import quote

import aiohttp
import numpy as np

from ..registry import model
from .interfaces import DiarizationResult, DiarizationSegment, SpeakerDiarization


_TIMESTAMP_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*"
    r"\[*\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\s*\[*\[S\d+\]|$))",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_SPEAKER_RE = re.compile(r"^\s*\[*\[S\d+\]", re.IGNORECASE)
_TRAILING_TIMESTAMP_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*$")

_Backend = Literal["official", "sglang_omni"]

_INTER_EXEMPLAR_SILENCE_S = 0.5
_EXEMPLAR_TO_CURRENT_SILENCE_S = 1.0
_DEFAULT_POOL_CONFIG: dict[str, Any] = {
    "max_speakers": 16,
    "min_register_duration_s": 0.70,
    "min_update_duration_s": 0.45,
    "preferred_min_duration_s": 1.0,
    "preferred_max_duration_s": 7.0,
    "min_rms_dbfs": -42.0,
    "max_clipping_ratio": 0.01,
    "min_boundary_margin_s": 0.08,
    "replace_score_margin": 0.08,
    "score_weights": {
        "duration": 0.35,
        "rms": 0.25,
        "non_overlap": 0.25,
        "boundary": 0.10,
        "unclipped": 0.05,
    },
}


@dataclass(frozen=True)
class _TimedSpeakerSegment:
    """One request-global segment parsed from an MTD response."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


@dataclass(frozen=True)
class _ParsedDiarizationSegment:
    """Internal segment representation used for exemplar selection."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str

    @property
    def duration_s(self) -> float:
        """Return the non-negative segment duration."""

        return max(0.0, self.end_s - self.start_s)


@dataclass
class _SpeakerExemplar:
    """Best registered audio and text exemplar for one session speaker."""

    speaker_id: str
    audio: np.ndarray
    text: str
    score: float
    quality: dict[str, Any]
    source_start_s: float
    source_end_s: float

    @property
    def duration_s(self) -> float:
        """Return exemplar duration at the MTD sample rate."""

        return len(self.audio) / 16000.0


@dataclass(frozen=True)
class _ExemplarCandidate:
    """One complete MTD segment considered for speaker registration."""

    segment: _ParsedDiarizationSegment
    audio: np.ndarray
    overlap_ratio: float
    overlap_class: Literal["non_overlap", "partial_overlap", "full_overlap"]
    score: float
    quality: dict[str, Any]
    eligibility_reason: str


@dataclass(frozen=True)
class _MtdAudioLayout:
    """Fully assembled MTD request input."""

    pcm16: bytes
    decoder_prefix: str
    context_seconds: float


def _pcm16_bytes_to_float32(pcm16: bytes) -> np.ndarray:
    """Convert little-endian PCM16 bytes to mono float32 samples."""

    if not pcm16:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm16, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def _float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert normalized float audio to little-endian PCM16 bytes."""

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


def _render_decoder_prefix(
    segments: Iterable[Mapping[str, Any]],
    *,
    precision: int = 2,
) -> str:
    """Render registered exemplars in the MTD fixed-prefix format."""

    return " ".join(
        f"[{float(item['start_s']):.{precision}f}]"
        f"[{str(item['speaker_id'])}]"
        f"{str(item.get('text') or '')}"
        f"[{float(item['end_s']):.{precision}f}]"
        for item in segments
    )


def _build_audio_layout(
    *,
    exemplars: list[_SpeakerExemplar],
    current_pcm16: bytes,
    sample_rate: int,
) -> _MtdAudioLayout:
    """Assemble registered exemplars, silence, and the current snapshot."""

    chunks: list[np.ndarray] = []
    prefix_segments: list[DiarizationSegment] = []
    cursor_s = 0.0
    for index, item in enumerate(exemplars):
        audio = np.asarray(item.audio, dtype=np.float32)
        start_s = cursor_s
        end_s = start_s + len(audio) / sample_rate
        chunks.append(audio)
        prefix_segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "speaker_id": item.speaker_id,
                "text": item.text,
            }
        )
        cursor_s = end_s
        if index + 1 < len(exemplars) and _INTER_EXEMPLAR_SILENCE_S > 0:
            chunks.append(
                np.zeros(
                    round(_INTER_EXEMPLAR_SILENCE_S * sample_rate),
                    dtype=np.float32,
                )
            )
            cursor_s += _INTER_EXEMPLAR_SILENCE_S
    if exemplars and _EXEMPLAR_TO_CURRENT_SILENCE_S > 0:
        chunks.append(
            np.zeros(
                round(_EXEMPLAR_TO_CURRENT_SILENCE_S * sample_rate),
                dtype=np.float32,
            )
        )
        cursor_s += _EXEMPLAR_TO_CURRENT_SILENCE_S
    chunks.append(_pcm16_bytes_to_float32(current_pcm16))
    request_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return _MtdAudioLayout(
        pcm16=_float32_to_pcm16_bytes(request_audio),
        decoder_prefix=_render_decoder_prefix(prefix_segments),
        context_seconds=cursor_s,
    )


def _dbfs(value: float) -> float:
    """Convert a linear amplitude to dBFS."""

    return 20.0 * math.log10(max(value, 1e-8))


def _covered_duration(
    start: float,
    end: float,
    blockers: Iterable[tuple[float, float]],
) -> float:
    """Return the union-covered duration of one interval by blockers."""

    clipped = sorted(
        (max(start, left), min(end, right))
        for left, right in blockers
        if right > start and left < end
    )
    covered_s = 0.0
    merged_start: float | None = None
    merged_end: float | None = None
    for left, right in clipped:
        if right <= left:
            continue
        if merged_start is None:
            merged_start, merged_end = left, right
            continue
        assert merged_end is not None
        if left <= merged_end:
            merged_end = max(merged_end, right)
            continue
        covered_s += merged_end - merged_start
        merged_start, merged_end = left, right
    if merged_start is not None and merged_end is not None:
        covered_s += merged_end - merged_start
    return covered_s


class _SpeakerExemplarPool:
    """Maintain one quality-ranked exemplar per global MTD speaker label."""

    SAMPLE_RATE = 16000
    _OVERLAP_PRIORITY = {
        "non_overlap": 0,
        "partial_overlap": 1,
        "full_overlap": 2,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or _DEFAULT_POOL_CONFIG)
        self.items: dict[str, _SpeakerExemplar] = {}
        self.version = 0

    def ordered(self) -> list[_SpeakerExemplar]:
        """Return exemplars in deterministic label order."""

        return sorted(self.items.values(), key=lambda item: item.speaker_id)

    def update_from_final(
        self,
        current_pcm16: bytes,
        segments: list[DiarizationSegment],
    ) -> list[dict[str, Any]]:
        """Select and register final-only exemplar candidates."""

        current_audio = _pcm16_bytes_to_float32(current_pcm16)
        parsed = [
            _ParsedDiarizationSegment(
                float(item["start_s"]),
                float(item["end_s"]),
                str(item["speaker_id"]),
                str(item.get("text") or "").strip(),
            )
            for item in segments
            if float(item["end_s"]) > float(item["start_s"])
        ]
        by_speaker: dict[str, list[_ParsedDiarizationSegment]] = {}
        for item in parsed:
            by_speaker.setdefault(item.speaker_id, []).append(item)

        decisions: list[dict[str, Any]] = []
        for speaker_id, occurrences in by_speaker.items():
            decisions.extend(
                self._select_speaker_candidate(
                    current_audio,
                    speaker_id=speaker_id,
                    occurrences=occurrences,
                    all_segments=parsed,
                )
            )
        return decisions

    def _select_speaker_candidate(
        self,
        current_audio: np.ndarray,
        *,
        speaker_id: str,
        occurrences: list[_ParsedDiarizationSegment],
        all_segments: list[_ParsedDiarizationSegment],
    ) -> list[dict[str, Any]]:
        """Choose one complete segment in overlap-priority order."""

        previous = self.items.get(speaker_id)
        candidates = [
            self._evaluate_complete_candidate(
                current_audio,
                item,
                all_segments,
                previous_exists=previous is not None,
            )
            for item in occurrences
        ]
        selected_class: str | None = None
        selected: _ExemplarCandidate | None = None
        for overlap_class in self._OVERLAP_PRIORITY:
            eligible = [
                candidate
                for candidate in candidates
                if candidate.overlap_class == overlap_class
                and candidate.eligibility_reason == "accepted"
            ]
            if eligible:
                selected_class = overlap_class
                selected = max(
                    eligible,
                    key=lambda candidate: (
                        candidate.score,
                        candidate.segment.duration_s,
                        -candidate.segment.start_s,
                    ),
                )
                break

        actions: list[dict[str, Any]] = []
        if selected is None:
            for candidate in candidates:
                actions.append(
                    self._candidate_action(
                        candidate,
                        action="reject",
                        reason=candidate.eligibility_reason,
                        previous=previous,
                    )
                )
            return actions

        action, reason = self._decide_pool_mutation(selected, previous)
        if action in {"register", "replace"}:
            self.items[speaker_id] = _SpeakerExemplar(
                speaker_id=speaker_id,
                audio=selected.audio.copy(),
                text=selected.segment.text,
                score=selected.score,
                quality=selected.quality,
                source_start_s=selected.segment.start_s,
                source_end_s=selected.segment.end_s,
            )
            self.version += 1

        for candidate in candidates:
            if candidate is selected:
                actions.append(
                    self._candidate_action(
                        candidate,
                        action=action,
                        reason=reason,
                        previous=previous,
                    )
                )
                continue
            if candidate.eligibility_reason != "accepted":
                rejection = candidate.eligibility_reason
            elif candidate.overlap_class != selected_class:
                rejection = "not_selected_overlap_priority"
            else:
                rejection = "not_selected_lower_score"
            actions.append(
                self._candidate_action(
                    candidate,
                    action="reject",
                    reason=rejection,
                    previous=previous,
                )
            )
        return actions

    def _evaluate_complete_candidate(
        self,
        current_audio: np.ndarray,
        item: _ParsedDiarizationSegment,
        all_segments: list[_ParsedDiarizationSegment],
        *,
        previous_exists: bool,
    ) -> _ExemplarCandidate:
        """Score one unmodified model segment and determine eligibility."""

        blockers = [
            (other.start_s, other.end_s)
            for other in all_segments
            if other.speaker_id != item.speaker_id
        ]
        overlap_s = _covered_duration(item.start_s, item.end_s, blockers)
        overlap_ratio = min(1.0, overlap_s / max(item.duration_s, 1e-6))
        if overlap_ratio <= 1e-9:
            overlap_class: Literal["non_overlap", "partial_overlap", "full_overlap"] = (
                "non_overlap"
            )
        elif overlap_ratio >= 1.0 - 1e-9:
            overlap_class = "full_overlap"
        else:
            overlap_class = "partial_overlap"

        start_sample = max(0, round(item.start_s * self.SAMPLE_RATE))
        end_sample = min(len(current_audio), round(item.end_s * self.SAMPLE_RATE))
        candidate_audio = np.asarray(
            current_audio[start_sample:end_sample],
            dtype=np.float32,
        )
        boundary_margin_s = min(
            item.start_s,
            max(0.0, len(current_audio) / self.SAMPLE_RATE - item.end_s),
        )
        score, quality = self._quality(
            candidate_audio,
            duration_s=item.duration_s,
            overlap_ratio=overlap_ratio,
            boundary_margin_s=boundary_margin_s,
            used_non_overlap=overlap_class == "non_overlap",
        )
        min_duration_s = float(
            self.config["min_update_duration_s"]
            if previous_exists
            else self.config["min_register_duration_s"]
        )
        reason = "accepted"
        if item.duration_s < min_duration_s:
            reason = "too_short"
        elif quality["rms_dbfs"] < float(self.config["min_rms_dbfs"]):
            reason = "too_quiet"
        elif quality["clipping_ratio"] > float(self.config["max_clipping_ratio"]):
            reason = "clipped"
        quality["overlap_class"] = overlap_class
        return _ExemplarCandidate(
            segment=item,
            audio=candidate_audio,
            overlap_ratio=overlap_ratio,
            overlap_class=overlap_class,
            score=score,
            quality=quality,
            eligibility_reason=reason,
        )

    def _decide_pool_mutation(
        self,
        candidate: _ExemplarCandidate,
        previous: _SpeakerExemplar | None,
    ) -> tuple[Literal["register", "replace", "reject"], str]:
        """Return the mutation permitted for the chosen candidate."""

        if previous is None:
            if len(self.items) < int(self.config["max_speakers"]):
                return "register", "accepted"
            return "reject", "pool_full"

        candidate_rank = self._OVERLAP_PRIORITY[candidate.overlap_class]
        previous_class = self._stored_overlap_class(previous)
        previous_rank = self._OVERLAP_PRIORITY[previous_class]
        if candidate_rank < previous_rank:
            return "replace", "better_overlap_class"
        if candidate_rank > previous_rank:
            return "reject", "worse_overlap_class"
        if candidate.score >= previous.score + float(
            self.config["replace_score_margin"]
        ):
            return "replace", "accepted"
        if (
            previous.duration_s < float(self.config["preferred_min_duration_s"])
            and candidate.segment.duration_s
            >= float(self.config["preferred_min_duration_s"])
            and candidate.score
            >= previous.score - float(self.config["replace_score_margin"])
        ):
            return "replace", "accepted"
        return "reject", "not_better"

    @classmethod
    def _stored_overlap_class(cls, exemplar: _SpeakerExemplar) -> str:
        """Recover overlap class from current or legacy quality fields."""

        value = str(exemplar.quality.get("overlap_class") or "")
        if value in cls._OVERLAP_PRIORITY:
            return value
        overlap_ratio = float(exemplar.quality.get("overlap_ratio") or 0.0)
        if overlap_ratio <= 1e-9:
            return "non_overlap"
        if overlap_ratio >= 1.0 - 1e-9:
            return "full_overlap"
        return "partial_overlap"

    @staticmethod
    def _candidate_action(
        candidate: _ExemplarCandidate,
        *,
        action: str,
        reason: str,
        previous: _SpeakerExemplar | None,
    ) -> dict[str, Any]:
        """Render one observable pool decision without mutating the pool."""

        return {
            "speaker_id": candidate.segment.speaker_id,
            "action": action,
            "reason": reason,
            "candidate_start_s": candidate.segment.start_s,
            "candidate_end_s": candidate.segment.end_s,
            "text": candidate.segment.text,
            "overlap_class": candidate.overlap_class,
            "quality": candidate.quality,
            "previous_score": previous.score if previous is not None else None,
        }

    def _quality(
        self,
        audio: np.ndarray,
        *,
        duration_s: float,
        overlap_ratio: float,
        boundary_margin_s: float,
        used_non_overlap: bool,
    ) -> tuple[float, dict[str, Any]]:
        """Compute the configured duration, audio, and overlap quality score."""

        rms = (
            float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
            if audio.size
            else 0.0
        )
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 1.0
        preferred_min = float(self.config["preferred_min_duration_s"])
        preferred_max = float(self.config["preferred_max_duration_s"])
        if duration_s < preferred_min:
            duration_score = duration_s / max(preferred_min, 1e-6)
        elif duration_s <= preferred_max:
            duration_score = 1.0
        else:
            duration_score = max(
                0.0,
                1.0 - (duration_s - preferred_max) / preferred_max,
            )
        rms_dbfs = _dbfs(rms)
        rms_score = min(
            1.0,
            max(
                0.0,
                (rms_dbfs - float(self.config["min_rms_dbfs"])) / 24.0,
            ),
        )
        boundary_score = min(
            1.0,
            boundary_margin_s / max(float(self.config["min_boundary_margin_s"]), 1e-6),
        )
        unclipped_score = max(
            0.0,
            1.0 - clipping_ratio / max(float(self.config["max_clipping_ratio"]), 1e-8),
        )
        weights = self.config["score_weights"]
        score = (
            float(weights["duration"]) * duration_score
            + float(weights["rms"]) * rms_score
            + float(weights["non_overlap"]) * max(0.0, 1.0 - overlap_ratio)
            + float(weights["boundary"]) * boundary_score
            + float(weights["unclipped"]) * unclipped_score
        )
        return score, {
            "duration_s": duration_s,
            "rms_dbfs": rms_dbfs,
            "peak_dbfs": _dbfs(peak),
            "clipping_ratio": clipping_ratio,
            "overlap_ratio": overlap_ratio,
            "boundary_margin_s": boundary_margin_s,
            "used_non_overlap": used_non_overlap,
            "score": score,
        }


class MtdRequestCancelled(RuntimeError):
    """Signal that an obsolete locally-cancellable MTD request was cancelled."""


@model
class OfficialMtdClient(SpeakerDiarization):
    """Call an official-vLLM or SGLang-Omni MTD HTTP runtime.

    The client discovers the backend once per session-local clone. A successful
    OpenAI-compatible ``GET /v1/models`` response selects SGLang-Omni and
    provides the model name submitted to ``/v1/audio/transcriptions``. A 404
    response selects the headless official-vLLM ``/v1/mtd/decode`` protocol.

    Parameters
    ----------
    base_url : str
        Official-vLLM or SGLang-Omni server root URL.
    request_timeout_s : float, optional
        Total HTTP timeout for one snapshot request.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum number of generated text tokens.
    """

    _INSTRUCTION = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
    )
    _RAW_PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>\n"
        "{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _RESPONSE_FORMAT = "verbose_json"

    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._session: aiohttp.ClientSession | None = None
        self._backend: _Backend | None = None
        self._model_name: str | None = None
        self._discovery_lock = asyncio.Lock()
        self._requests: dict[str, asyncio.Task[tuple[dict[str, Any], str]]] = {}
        self._exemplar_pool = _SpeakerExemplarPool()

    def clone(self) -> OfficialMtdClient:
        """Create a session-local client with the same public configuration."""

        return OfficialMtdClient(
            base_url=self.base_url,
            request_timeout_s=self.request_timeout_s,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def decode_snapshot(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        is_final: bool,
    ) -> DiarizationResult:
        """Apply MTD registration context and decode one current snapshot."""

        layout = _build_audio_layout(
            exemplars=self._exemplar_pool.ordered(),
            current_pcm16=pcm16,
            sample_rate=sample_rate,
        )
        current_audio_seconds = len(pcm16) / (sample_rate * 2)
        backend = await self._discover_backend()
        if backend == "official":
            result = await self._decode_official(
                request_id=request_id,
                pcm16=layout.pcm16,
                sample_rate=sample_rate,
                decoder_prefix=layout.decoder_prefix,
                context_seconds=layout.context_seconds,
                current_audio_seconds=current_audio_seconds,
                is_final=is_final,
            )
        else:
            result = await self._decode_sglang(
                request_id=request_id,
                pcm16=layout.pcm16,
                sample_rate=sample_rate,
                decoder_prefix=layout.decoder_prefix,
                context_seconds=layout.context_seconds,
                current_audio_seconds=current_audio_seconds,
            )
        pool_actions = (
            self._exemplar_pool.update_from_final(pcm16, result.segments)
            if is_final
            else []
        )
        return DiarizationResult(
            raw_text=result.raw_text,
            segments=result.segments,
            latency_ms=result.latency_ms,
            metrics={
                **result.metrics,
                "pool_version": self._exemplar_pool.version,
                "pool_actions": pool_actions,
            },
        )

    async def _discover_backend(self) -> _Backend:
        """Discover the protocol and SGLang model name from the server."""

        if self._backend is not None:
            return self._backend
        async with self._discovery_lock:
            if self._backend is not None:
                return self._backend
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 404:
                    self._backend = "official"
                    return self._backend
                response.raise_for_status()
                payload: dict[str, Any] = await response.json()
            self._model_name = _extract_model_name(payload)
            self._backend = "sglang_omni"
            return self._backend

    async def _decode_official(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
        context_seconds: float,
        current_audio_seconds: float,
        is_final: bool,
    ) -> DiarizationResult:
        """Send one PCM snapshot to the headless official-vLLM runtime."""

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("request_id", request_id)
        form.add_field("sample_rate", str(sample_rate))
        form.add_field("decoder_prefix", decoder_prefix)
        form.add_field("context_seconds", str(context_seconds))
        form.add_field("current_audio_seconds", str(current_audio_seconds))
        form.add_field("is_final", "true" if is_final else "false")
        form.add_field("instruction", self._INSTRUCTION)
        form.add_field("temperature", str(self.temperature))
        form.add_field("max_tokens", str(self.max_tokens))
        form.add_field(
            "audio",
            pcm16,
            filename="snapshot.pcm",
            content_type="audio/pcm",
        )
        started = time.perf_counter()
        async with session.post(
            f"{self.base_url}/v1/mtd/decode",
            data=form,
        ) as response:
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()
        latency_ms = float(payload.get("latency_ms") or 0.0)
        if latency_ms <= 0:
            latency_ms = (time.perf_counter() - started) * 1000.0
        return DiarizationResult(
            raw_text=str(payload.get("raw_text") or ""),
            segments=_normalize_segments(payload.get("current_segments")),
            latency_ms=latency_ms,
            metrics=dict(payload.get("metrics") or {}),
        )

    async def _decode_sglang(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
        context_seconds: float,
        current_audio_seconds: float,
    ) -> DiarizationResult:
        """Decode one snapshot through the native SGLang-Omni endpoint."""

        started = time.perf_counter()
        request_task = asyncio.create_task(
            self._post_transcription(
                pcm16=pcm16,
                sample_rate=sample_rate,
                decoder_prefix=decoder_prefix,
            )
        )
        previous = self._requests.setdefault(request_id, request_task)
        if previous is not request_task:
            request_task.cancel()
            raise ValueError(f"duplicate SGLang-Omni request_id: {request_id}")
        try:
            payload, remote_request_id = await request_task
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise MtdRequestCancelled(
                f"SGLang-Omni request was cancelled: {request_id}"
            ) from exc
        finally:
            if self._requests.get(request_id) is request_task:
                self._requests.pop(request_id, None)

        generated_suffix = str(payload.get("text") or "").strip()
        raw_text = _join_decoder_prefix_and_suffix(decoder_prefix, generated_suffix)
        raw_segments = _parse_timestamped_text(raw_text)
        current_segments = _crop_current_segments(
            raw_segments,
            context_seconds=max(0.0, float(context_seconds)),
            current_audio_seconds=max(0.0, float(current_audio_seconds)),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return DiarizationResult(
            raw_text=raw_text,
            segments=current_segments,
            latency_ms=latency_ms,
            metrics={
                "backend": "sglang_omni",
                "request_id": request_id,
                "remote_request_id": remote_request_id,
                "server_duration_s": payload.get("duration"),
                "usage": payload.get("usage"),
                "registration_mode": "fixed_decoder_prefix",
                "decoder_prefix_chars": len(decoder_prefix),
                "generated_suffix": generated_suffix,
            },
        )

    async def cancel(self, request_id: str) -> None:
        """Cancel an official request remotely or an SGLang request locally."""

        backend = await self._discover_backend()
        if backend == "official":
            session = await self._get_session()
            url = f"{self.base_url}/v1/mtd/requests/{quote(request_id, safe='')}"
            async with session.delete(url) as response:
                if response.status not in {200, 202, 204, 404}:
                    response.raise_for_status()
            return
        request_task = self._requests.get(request_id)
        if request_task is not None and not request_task.done():
            request_task.cancel()

    async def close(self) -> None:
        """Cancel outstanding requests and close the HTTP session."""

        for request_task in tuple(self._requests.values()):
            request_task.cancel()
        if self._requests:
            await asyncio.gather(*self._requests.values(), return_exceptions=True)
        self._requests.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post_transcription(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
    ) -> tuple[dict[str, Any], str]:
        """Post one WAV snapshot with the automatically discovered model name."""

        if self._model_name is None:
            raise RuntimeError("SGLang-Omni model discovery has not completed")
        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field(
            "file",
            _pcm16_to_wav(pcm16, sample_rate),
            filename="snapshot.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self._model_name)
        form.add_field("prompt", self._build_forced_prefix_prompt(decoder_prefix))
        form.add_field("response_format", self._RESPONSE_FORMAT)
        form.add_field("temperature", str(self.temperature))
        form.add_field("max_new_tokens", str(self.max_tokens))
        async with session.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data=form,
        ) as response:
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()
            remote_request_id = response.headers.get("X-Request-Id", "")
        return payload, remote_request_id

    def _build_forced_prefix_prompt(self, decoder_prefix: str) -> str:
        """Build the complete SGLang prompt with a fixed decoder prefix."""

        return (
            self._RAW_PROMPT_TEMPLATE.format(instruction=self._INSTRUCTION)
            + decoder_prefix
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a live HTTP session for the current event loop."""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


def _extract_model_name(payload: object) -> str:
    """Extract the first non-empty model ID from an OpenAI model-list response."""

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("id") or "").strip()
                if model_name:
                    return model_name
    raise RuntimeError("GET /v1/models returned no usable model ID")


def _normalize_segments(value: object) -> list[DiarizationSegment]:
    """Normalize official-runtime JSON into the public segment contract."""

    if not isinstance(value, list):
        return []
    segments: list[DiarizationSegment] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
            speaker_id = str(item["speaker_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "speaker_id": speaker_id,
                "text": str(item.get("text") or ""),
            }
        )
    return segments


def _pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap mono little-endian PCM16 in a standard WAV container."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(pcm16) % 2:
        raise ValueError("PCM16 payload must contain complete two-byte samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return output.getvalue()


def _parse_timestamped_text(text: str) -> list[_TimedSpeakerSegment]:
    """Parse native MTD timestamp-plus-speaker text."""

    result: list[_TimedSpeakerSegment] = []
    for match in _TIMESTAMP_SEGMENT_RE.finditer(text):
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        if end_s <= start_s:
            continue
        result.append(
            _TimedSpeakerSegment(
                start_s=max(0.0, start_s),
                end_s=end_s,
                speaker_id=match.group("speaker").upper(),
                text=re.sub(r"\s+", " ", match.group("text")).strip(),
            )
        )
    return result


def _join_decoder_prefix_and_suffix(decoder_prefix: str, suffix: str) -> str:
    """Reconstruct a parseable full timeline at a continuation boundary."""

    prefix = decoder_prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if _LEADING_SPEAKER_RE.match(suffix):
        boundary = _TRAILING_TIMESTAMP_RE.search(prefix)
        if boundary is not None:
            suffix = f"[{boundary.group(1)}]{suffix}"
    return f"{prefix} {suffix}"


def _crop_current_segments(
    segments: list[_TimedSpeakerSegment],
    *,
    context_seconds: float,
    current_audio_seconds: float,
) -> list[DiarizationSegment]:
    """Remove registered-audio time and retain decoder-produced speaker IDs."""

    current_end_s = context_seconds + current_audio_seconds
    result: list[DiarizationSegment] = []
    for segment in segments:
        start_s = max(segment.start_s, context_seconds)
        end_s = min(segment.end_s, current_end_s)
        if end_s <= start_s:
            continue
        result.append(
            {
                "start_s": round(max(0.0, start_s - context_seconds), 6),
                "end_s": round(max(0.0, end_s - context_seconds), 6),
                "speaker_id": segment.speaker_id,
                "text": segment.text,
            }
        )
    return result
