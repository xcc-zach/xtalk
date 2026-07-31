"""Session-local global-speaker exemplar selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np

from .audio_layout import pcm16_bytes_to_float32
from .transcript import DiarizationSegment


@dataclass
class SpeakerExemplar:
    """Best registered audio/text exemplar for one session speaker label."""

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
    """One complete MTD segment considered for speaker registration.

    Audio and text always refer to the exact same model-produced MTD segment.
    In particular, overlap handling must never cut a subrange from ``segment``:
    MTD does not provide word-level alignments that would allow its text to be
    cut reliably as well.
    """

    segment: DiarizationSegment
    audio: np.ndarray
    overlap_ratio: float
    overlap_class: Literal["non_overlap", "partial_overlap", "full_overlap"]
    score: float
    quality: dict[str, Any]
    eligibility_reason: str


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))


def _covered_duration(
    start: float,
    end: float,
    blockers: Iterable[tuple[float, float]],
) -> float:
    """Return the union-covered duration of ``[start, end]`` by blockers."""

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


class SpeakerExemplarPool:
    """Maintain one quality-ranked exemplar per global MTD speaker label."""

    SAMPLE_RATE = 16000
    _OVERLAP_PRIORITY = {
        "non_overlap": 0,
        "partial_overlap": 1,
        "full_overlap": 2,
    }

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.items: dict[str, SpeakerExemplar] = {}
        self.version = 0

    def ordered(self) -> list[SpeakerExemplar]:
        """Return exemplars in deterministic label order."""

        return sorted(self.items.values(), key=lambda item: item.speaker_id)

    def update_from_final(
        self,
        current_pcm16: bytes,
        segments: list[dict[str, Any]],
        *,
        source_segment_id: int,
    ) -> list[dict[str, Any]]:
        """Select and register final-only exemplar candidates."""

        current_audio = pcm16_bytes_to_float32(current_pcm16)
        parsed = [
            DiarizationSegment(
                float(item["start_s"]),
                float(item["end_s"]),
                str(item["speaker_id"]),
                str(item.get("text") or "").strip(),
            )
            for item in segments
            if float(item["end_s"]) > float(item["start_s"])
        ]
        by_speaker: dict[str, list[DiarizationSegment]] = {}
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
        occurrences: list[DiarizationSegment],
        all_segments: list[DiarizationSegment],
        source_segment_id: int,
    ) -> list[dict[str, Any]]:
        """Choose one complete segment in overlap-priority order.

        Candidates are deliberately grouped before quality ranking:

        ``non_overlap`` > ``partial_overlap`` > ``full_overlap``.

        The quality score ranks candidates *within* one group only. This keeps
        a short but usable clean MTD segment ahead of a higher-scoring segment
        that contains another speaker, while preserving exact audio/text pairs.
        """

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
        priority = tuple(self._OVERLAP_PRIORITY)
        selected_class: str | None = None
        selected: _ExemplarCandidate | None = None
        for overlap_class in priority:
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
            self.items[speaker_id] = SpeakerExemplar(
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
        item: DiarizationSegment,
        all_segments: list[DiarizationSegment],
        *,
        previous_exists: bool,
    ) -> _ExemplarCandidate:
        """Score one unmodified model segment and determine basic eligibility."""

        cfg = self.config
        blockers = [
            (other.start_s, other.end_s)
            for other in all_segments
            if other.speaker_id != item.speaker_id
        ]
        overlap_s = _covered_duration(item.start_s, item.end_s, blockers)
        overlap_ratio = min(1.0, overlap_s / max(item.duration_s, 1e-6))
        if overlap_ratio <= 1e-9:
            overlap_class: Literal[
                "non_overlap", "partial_overlap", "full_overlap"
            ] = "non_overlap"
        elif overlap_ratio >= 1.0 - 1e-9:
            overlap_class = "full_overlap"
        else:
            overlap_class = "partial_overlap"

        # Preserve the exact MTD segment boundaries. Splitting audio to remove
        # an overlap would require splitting text without word timestamps.
        start_sample = max(0, round(item.start_s * self.SAMPLE_RATE))
        end_sample = min(len(current_audio), round(item.end_s * self.SAMPLE_RATE))
        candidate_audio = np.asarray(current_audio[start_sample:end_sample], dtype=np.float32)
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
            cfg["min_update_duration_s"]
            if previous_exists
            else cfg["min_register_duration_s"]
        )
        reason = "accepted"
        if item.duration_s < min_duration_s:
            reason = "too_short"
        elif quality["rms_dbfs"] < float(cfg["min_rms_dbfs"]):
            reason = "too_quiet"
        elif quality["clipping_ratio"] > float(cfg["max_clipping_ratio"]):
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
        previous: SpeakerExemplar | None,
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

        if candidate.score >= previous.score + float(self.config["replace_score_margin"]):
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
    def _stored_overlap_class(cls, exemplar: SpeakerExemplar) -> str:
        """Recover overlap class from current or legacy pool quality fields."""

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
        previous: SpeakerExemplar | None,
    ) -> dict[str, Any]:
        """Render one observable pool decision without altering its candidate."""

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
        """Compute the configured duration/audio/overlap quality score."""

        cfg = self.config
        rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        clipping_ratio = float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 1.0
        preferred_min = float(cfg["preferred_min_duration_s"])
        preferred_max = float(cfg["preferred_max_duration_s"])
        if duration_s < preferred_min:
            duration_score = duration_s / max(preferred_min, 1e-6)
        elif duration_s <= preferred_max:
            duration_score = 1.0
        else:
            duration_score = max(0.0, 1.0 - (duration_s - preferred_max) / preferred_max)
        rms_dbfs = _dbfs(rms)
        rms_score = min(1.0, max(0.0, (rms_dbfs - float(cfg["min_rms_dbfs"])) / 24.0))
        boundary_score = min(
            1.0,
            boundary_margin_s / max(float(cfg["min_boundary_margin_s"]), 1e-6),
        )
        unclipped_score = max(
            0.0,
            1.0 - clipping_ratio / max(float(cfg["max_clipping_ratio"]), 1e-8),
        )
        weights = cfg["score_weights"]
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
