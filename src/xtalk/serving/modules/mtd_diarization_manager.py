"""Session manager for VAD-internal MTD full-snapshot decoding."""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping

import numpy as np

from ...models import Models, SpeakerDiarization
from ...models.speaker_diarization.interfaces import (
    DiarizationResult,
    DiarizationSegment,
)
from ..event_bus import EventBus
from ..events import (
    EnhancedAudioFrameReceived,
    SpeakerDiarizationPartial,
    SpeakerDiarizationSegmentFinal,
    SpeakerDiarizationTurnFinal,
    TurnASREndRequested,
    TurnASRPauseRequested,
    TurnASRStartRequested,
    VADSpeechEnd,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


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
    source_segment_id: int
    source_start_s: float
    source_end_s: float

    @property
    def duration_s(self) -> float:
        """Return exemplar duration at 16 kHz."""

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
    slots: list[dict[str, object]]


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


def _render_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    precision: int = 2,
) -> str:
    """Render segments in the MTD timestamp-plus-speaker format."""

    return " ".join(
        f"[{float(item['start_s']):.{precision}f}]"
        f"[{str(item['speaker_id'])}]"
        f"{str(item.get('text') or '')}"
        f"[{float(item['end_s']):.{precision}f}]"
        for item in segments
    )


def _offset_segments(
    segments: Iterable[Mapping[str, Any]],
    *,
    offset_s: float,
    turn_id: int,
    segment_id: int,
) -> list[dict[str, Any]]:
    """Convert current-local segments to a session-source timeline."""

    return [
        {
            **item,
            "start_s": float(item["start_s"]) + offset_s,
            "end_s": float(item["end_s"]) + offset_s,
            "turn_id": turn_id,
            "segment_id": segment_id,
        }
        for item in segments
    ]


def _build_audio_layout(
    *,
    exemplars: list[_SpeakerExemplar],
    current_pcm16: bytes,
    sample_rate: int,
    inter_exemplar_silence_s: float,
    exemplar_to_current_silence_s: float,
) -> _MtdAudioLayout:
    """Assemble exemplar audio, configurable silence, and current PCM."""

    chunks: list[np.ndarray] = []
    prefix_segments: list[DiarizationSegment] = []
    slots: list[dict[str, object]] = []
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
        slots.append(
            {
                "speaker_id": item.speaker_id,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
            }
        )
        cursor_s = end_s
        if index + 1 < len(exemplars) and inter_exemplar_silence_s > 0:
            chunks.append(
                np.zeros(
                    round(inter_exemplar_silence_s * sample_rate), dtype=np.float32
                )
            )
            cursor_s += inter_exemplar_silence_s
    if exemplars and exemplar_to_current_silence_s > 0:
        chunks.append(
            np.zeros(
                round(exemplar_to_current_silence_s * sample_rate),
                dtype=np.float32,
            )
        )
        cursor_s += exemplar_to_current_silence_s
    chunks.append(_pcm16_bytes_to_float32(current_pcm16))
    request_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return _MtdAudioLayout(
        pcm16=_float32_to_pcm16_bytes(request_audio),
        decoder_prefix=_render_segments(prefix_segments),
        context_seconds=cursor_s,
        slots=slots,
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

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.items: dict[str, _SpeakerExemplar] = {}
        self.version = 0

    def ordered(self) -> list[_SpeakerExemplar]:
        """Return exemplars in deterministic label order."""

        return sorted(self.items.values(), key=lambda item: item.speaker_id)

    def update_from_final(
        self,
        current_pcm16: bytes,
        segments: list[DiarizationSegment],
        *,
        source_segment_id: int,
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
                    source_segment_id=source_segment_id,
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
        source_segment_id: int,
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
                source_segment_id=source_segment_id,
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


@dataclass
class _SegmentState:
    """Mutable PCM and scheduling state for one VAD segment."""

    turn_id: int
    segment_id: int
    source_start_sample: int
    pcm: bytearray
    revision: int = 0
    next_partial_s: float = 0.8
    final_requested: bool = False
    last_partial_text: str = ""
    published_partial_revision: int = 0


@dataclass(frozen=True)
class _SnapshotRequest:
    """Immutable snapshot queued for the single per-session worker."""

    request_id: str
    turn_id: int
    segment_id: int
    revision: int
    source_start_sample: int
    current_pcm16: bytes
    is_final: bool


@dataclass
class _TurnState:
    """Track VAD segment terminals for one ASR hard turn."""

    segment_ids: list[int] = field(default_factory=list)
    finals: dict[int, SpeakerDiarizationSegmentFinal] = field(default_factory=dict)
    hard_closed: bool = False
    turn_final_published: bool = False


class MtdDiarizationManager(Manager):
    """Run MTD snapshots without blocking Xtalk's serialized audio frame chain."""

    BYTES_PER_SAMPLE = 2

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config = config or {}
        multi_config = dict(self.config.get("multi_speaker") or {})
        self.enabled = bool(multi_config.get("enabled", False))
        self.model = models.get(SpeakerDiarization) if self.enabled else None
        mtd_config = dict(self.config.get("mtd") or {})
        self.sample_rate = int(mtd_config.get("sample_rate", 16000))
        self.pre_buffer_s = float(mtd_config.get("pre_buffer_s", 1.0))
        partial_config = dict(mtd_config.get("partial") or {})
        self.partial_interval_s = float(partial_config.get("interval_s", 1.0))
        self.first_partial_min_s = float(partial_config.get("first_partial_min_s", 0.8))
        self.publish_unchanged = bool(partial_config.get("publish_unchanged", False))
        self.abort_on_vad_end = bool(partial_config.get("abort_on_vad_end", True))
        layout_config = dict(mtd_config.get("audio_layout") or {})
        self.inter_exemplar_silence_s = float(
            layout_config.get("inter_exemplar_silence_s", 0.5)
        )
        self.exemplar_to_current_silence_s = float(
            layout_config.get("exemplar_to_current_silence_s", 1.0)
        )
        pool_config = _merge_nested_dict(
            _DEFAULT_POOL_CONFIG,
            dict(mtd_config.get("pool") or {}),
        )
        self.pool = _SpeakerExemplarPool(pool_config)
        self._sample_cursor = 0
        self._pre_buffer = bytearray()
        self._pre_buffer_max_bytes = round(
            self.pre_buffer_s * self.sample_rate * self.BYTES_PER_SAMPLE
        )
        self._active_segment: _SegmentState | None = None
        self._segments: dict[int, _SegmentState] = {}
        self._turns: dict[int, _TurnState] = {}
        self._pending_partial: _SnapshotRequest | None = None
        self._final_queue: deque[_SnapshotRequest] = deque()
        self._pending_event = asyncio.Event()
        self._in_flight: _SnapshotRequest | None = None
        self._worker_task = (
            asyncio.create_task(self._worker()) if self.enabled else None
        )
        if self.enabled and self.model is None:
            logger.warning(
                "MTD multi-speaker mode enabled without a SpeakerDiarization model - session: %s",
                session_id,
            )

    @Manager.event_handler(
        EnhancedAudioFrameReceived,
        priority=-20,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Copy enhanced PCM into a bounded pre-buffer or active segment."""

        if not self.enabled or not event.audio_data:
            return
        if event.sample_rate != self.sample_rate:
            logger.warning(
                "MTD ignored PCM with sample rate %s; expected %s - session: %s",
                event.sample_rate,
                self.sample_rate,
                self.session_id,
            )
            return
        pcm = event.audio_data
        self._sample_cursor += len(pcm) // self.BYTES_PER_SAMPLE
        if self._active_segment is None:
            self._pre_buffer.extend(pcm)
            if len(self._pre_buffer) > self._pre_buffer_max_bytes:
                del self._pre_buffer[
                    : len(self._pre_buffer) - self._pre_buffer_max_bytes
                ]
            return
        self._active_segment.pcm.extend(pcm)
        self._maybe_schedule_partial(self._active_segment)

    @Manager.event_handler(
        TurnASRStartRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_segment_start(self, event: TurnASRStartRequested) -> None:
        """Open a VAD segment using the same ID and pre-buffer as ASR."""

        if not self.enabled:
            return
        if self._active_segment is not None:
            logger.warning(
                "MTD received a new segment before the previous segment closed - session: %s",
                self.session_id,
            )
            await self._freeze_and_schedule_final(self._active_segment)
        pre_buffer = bytes(self._pre_buffer)
        source_start = max(
            0,
            self._sample_cursor - len(pre_buffer) // self.BYTES_PER_SAMPLE,
        )
        state = _SegmentState(
            turn_id=event.turn_id,
            segment_id=event.segment_id,
            source_start_sample=source_start,
            pcm=bytearray(pre_buffer),
            next_partial_s=self.first_partial_min_s,
        )
        self._active_segment = state
        self._segments[event.segment_id] = state
        turn = self._turns.setdefault(event.turn_id, _TurnState())
        if event.segment_id not in turn.segment_ids:
            turn.segment_ids.append(event.segment_id)
        self._pre_buffer.clear()
        self._maybe_schedule_partial(state)

    @Manager.event_handler(
        VADSpeechEnd,
        priority=100,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_vad_end(self, _event: VADSpeechEnd) -> None:
        """Freeze the current VAD segment before hard-turn control arrives."""

        if not self.enabled or self._active_segment is None:
            return
        state = self._active_segment
        self._active_segment = None
        await self._freeze_and_schedule_final(state)

    @Manager.event_handler(
        TurnASRPauseRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_segment_pause(self, event: TurnASRPauseRequested) -> None:
        """Freeze one VAD segment while leaving its ASR turn open."""

        if not self.enabled:
            return
        await self._close_matching_segment(event.turn_id, event.segment_id)

    @Manager.event_handler(
        TurnASREndRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_turn_end(self, event: TurnASREndRequested) -> None:
        """Freeze the final VAD segment and mark the ASR turn hard-closed."""

        if not self.enabled:
            return
        await self._close_matching_segment(event.turn_id, event.segment_id)
        turn = self._turns.setdefault(event.turn_id, _TurnState())
        turn.hard_closed = True
        await self._maybe_publish_turn_final(event.turn_id)

    async def _close_matching_segment(self, turn_id: int, segment_id: int) -> None:
        """Freeze the active matching segment and schedule a final snapshot."""

        state = self._active_segment
        if state is None or state.turn_id != turn_id or state.segment_id != segment_id:
            return
        self._active_segment = None
        await self._freeze_and_schedule_final(state)

    async def _freeze_and_schedule_final(self, state: _SegmentState) -> None:
        """Promote one immutable segment snapshot above all queued partials."""

        if state.final_requested:
            return
        state.final_requested = True
        state.revision += 1
        request = self._make_request(state, is_final=True)
        self._pending_partial = None
        self._final_queue.append(request)
        self._pending_event.set()
        if (
            self.abort_on_vad_end
            and self._in_flight is not None
            and not self._in_flight.is_final
            and self.model is not None
        ):
            asyncio.create_task(self._best_effort_cancel(self._in_flight.request_id))

    def _maybe_schedule_partial(self, state: _SegmentState) -> None:
        """Queue the latest complete VAD snapshot at the configured cadence."""

        if state.final_requested:
            return
        duration_s = len(state.pcm) / (self.sample_rate * self.BYTES_PER_SAMPLE)
        if duration_s + 1e-9 < state.next_partial_s:
            return
        state.revision += 1
        self._pending_partial = self._make_request(state, is_final=False)
        self._pending_event.set()
        while state.next_partial_s <= duration_s + 1e-9:
            state.next_partial_s += self.partial_interval_s

    def _make_request(
        self,
        state: _SegmentState,
        *,
        is_final: bool,
    ) -> _SnapshotRequest:
        """Copy mutable segment PCM into an immutable worker request."""

        kind = "final" if is_final else "partial"
        request_id = (
            f"{self.session_id}/{state.turn_id}/{state.segment_id}/"
            f"{state.revision}/{kind}"
        )
        return _SnapshotRequest(
            request_id=request_id,
            turn_id=state.turn_id,
            segment_id=state.segment_id,
            revision=state.revision,
            source_start_sample=state.source_start_sample,
            current_pcm16=bytes(state.pcm),
            is_final=is_final,
        )

    async def _worker(self) -> None:
        """Process at most one MTD request at a time with latest-only pending."""

        try:
            while True:
                await self._pending_event.wait()
                request: _SnapshotRequest | None
                if self._final_queue:
                    request = self._final_queue.popleft()
                else:
                    request = self._pending_partial
                    self._pending_partial = None
                if not self._final_queue and self._pending_partial is None:
                    self._pending_event.clear()
                if request is None:
                    continue
                self._in_flight = request
                try:
                    await self._decode_and_publish(request)
                finally:
                    self._in_flight = None
                    if self._final_queue or self._pending_partial is not None:
                        self._pending_event.set()
        except asyncio.CancelledError:
            raise

    async def _decode_and_publish(self, request: _SnapshotRequest) -> None:
        """Build the registered-speaker prefix, decode, and publish one result."""

        if self.model is None:
            if request.is_final:
                await self._publish_degraded_final(request, "model_not_configured")
            return
        layout = _build_audio_layout(
            exemplars=self.pool.ordered(),
            current_pcm16=request.current_pcm16,
            sample_rate=self.sample_rate,
            inter_exemplar_silence_s=self.inter_exemplar_silence_s,
            exemplar_to_current_silence_s=self.exemplar_to_current_silence_s,
        )
        current_duration_s = len(request.current_pcm16) / (
            self.sample_rate * self.BYTES_PER_SAMPLE
        )
        try:
            result = await self.model.decode_snapshot(
                request_id=request.request_id,
                pcm16=layout.pcm16,
                sample_rate=self.sample_rate,
                decoder_prefix=layout.decoder_prefix,
                context_seconds=layout.context_seconds,
                current_audio_seconds=current_duration_s,
                is_final=request.is_final,
            )
        except Exception as exc:
            logger.warning(
                "MTD snapshot failed - session: %s, request: %s, error: %s",
                self.session_id,
                request.request_id,
                exc,
            )
            if request.is_final:
                await self._publish_degraded_final(request, type(exc).__name__)
            return
        state = self._segments.get(request.segment_id)
        if state is None:
            return
        if not request.is_final and state.final_requested:
            return
        if request.is_final:
            await self._publish_segment_final(request, result)
            return
        if request.revision <= state.published_partial_revision:
            return
        diarization_text = _render_segments(result.current_segments)
        if not self.publish_unchanged and diarization_text == state.last_partial_text:
            return
        state.last_partial_text = diarization_text
        state.published_partial_revision = request.revision
        await self.event_bus.publish(
            SpeakerDiarizationPartial(
                session_id=self.session_id,
                turn_id=request.turn_id,
                segment_id=request.segment_id,
                revision=request.revision,
                source_start_sample=request.source_start_sample,
                source_end_sample=request.source_start_sample
                + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
                sample_rate=self.sample_rate,
                raw_text=result.raw_text,
                diarization_text=diarization_text,
                segments=result.current_segments,
                latency_ms=result.latency_ms,
            )
        )

    async def _publish_segment_final(
        self,
        request: _SnapshotRequest,
        result: DiarizationResult,
    ) -> None:
        """Update the pool and publish one successful terminal segment."""

        pool_actions = self.pool.update_from_final(
            request.current_pcm16,
            result.current_segments,
            source_segment_id=request.segment_id,
        )
        event = SpeakerDiarizationSegmentFinal(
            session_id=self.session_id,
            turn_id=request.turn_id,
            segment_id=request.segment_id,
            source_start_sample=request.source_start_sample,
            source_end_sample=request.source_start_sample
            + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
            sample_rate=self.sample_rate,
            raw_text=result.raw_text,
            diarization_text=_render_segments(result.current_segments),
            segments=[dict(item) for item in result.current_segments],
            pool_version=self.pool.version,
            pool_actions=pool_actions,
            latency_ms=result.latency_ms,
        )
        await self._record_segment_final(event)

    async def _publish_degraded_final(
        self,
        request: _SnapshotRequest,
        reason: str,
    ) -> None:
        """Publish a terminal empty result so the turn join cannot deadlock."""

        event = SpeakerDiarizationSegmentFinal(
            session_id=self.session_id,
            turn_id=request.turn_id,
            segment_id=request.segment_id,
            source_start_sample=request.source_start_sample,
            source_end_sample=request.source_start_sample
            + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
            sample_rate=self.sample_rate,
            pool_version=self.pool.version,
            degraded=True,
            degraded_reason=reason,
        )
        await self._record_segment_final(event)

    async def _record_segment_final(
        self,
        event: SpeakerDiarizationSegmentFinal,
    ) -> None:
        """Record and publish a segment terminal before testing the turn barrier."""

        turn = self._turns.setdefault(event.turn_id, _TurnState())
        turn.finals[event.segment_id] = event
        await self.event_bus.publish(event)
        await self._maybe_publish_turn_final(event.turn_id)

    async def _maybe_publish_turn_final(self, turn_id: int) -> None:
        """Publish exactly one ordered turn timeline after its barrier is complete."""

        turn = self._turns.get(turn_id)
        if turn is None or not turn.hard_closed or turn.turn_final_published:
            return
        if any(segment_id not in turn.finals for segment_id in turn.segment_ids):
            return
        timeline: list[dict[str, Any]] = []
        degraded_reasons: list[str] = []
        for segment_id in turn.segment_ids:
            final = turn.finals[segment_id]
            timeline.extend(
                _offset_segments(
                    final.segments,
                    offset_s=final.source_start_sample / final.sample_rate,
                    turn_id=turn_id,
                    segment_id=segment_id,
                )
            )
            if final.degraded:
                degraded_reasons.append(final.degraded_reason)
        timeline.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"])))
        active_speaker_id = next(
            (
                str(item["speaker_id"])
                for item in reversed(timeline)
                if str(item.get("text") or "").strip()
            ),
            None,
        )
        turn.turn_final_published = True
        await self.event_bus.publish(
            SpeakerDiarizationTurnFinal(
                session_id=self.session_id,
                turn_id=turn_id,
                segment_ids=list(turn.segment_ids),
                segments=timeline,
                diarization_text=_render_segments(timeline),
                active_speaker_id=active_speaker_id,
                degraded=bool(degraded_reasons),
                degraded_reason=",".join(dict.fromkeys(degraded_reasons)),
            )
        )

    async def _best_effort_cancel(self, request_id: str) -> None:
        """Cancel an obsolete partial without propagating runtime failures."""

        if self.model is None:
            return
        try:
            await self.model.cancel(request_id)
        except Exception as exc:
            logger.debug("MTD cancel failed for %s: %s", request_id, exc)

    async def shutdown(self) -> None:
        """Cancel session tasks and release the cloned MTD client."""

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self.model is not None:
            await self.model.close()


def _merge_nested_dict(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a shallow configuration merge with nested dict preservation."""

    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
