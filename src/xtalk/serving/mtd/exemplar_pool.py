"""Session-local global-speaker exemplar selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

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


def _dbfs(value: float) -> float:
    return 20.0 * math.log10(max(value, 1e-8))


def _interval_overlap(left: float, right: float, start: float, end: float) -> float:
    return max(0.0, min(right, end) - max(left, start))


def _subtract_intervals(
    start: float,
    end: float,
    blockers: Iterable[tuple[float, float]],
) -> list[tuple[float, float]]:
    pieces = [(start, end)]
    for left, right in sorted(blockers):
        updated: list[tuple[float, float]] = []
        for piece_start, piece_end in pieces:
            if right <= piece_start or left >= piece_end:
                updated.append((piece_start, piece_end))
                continue
            if left > piece_start:
                updated.append((piece_start, min(left, piece_end)))
            if right < piece_end:
                updated.append((max(right, piece_start), piece_end))
        pieces = updated
    return [(left, right) for left, right in pieces if right > left]


def _crop_text_by_time(
    text: str,
    *,
    source_start_s: float,
    source_end_s: float,
    selected_start_s: float,
    selected_end_s: float,
) -> str:
    if not text or source_end_s <= source_start_s:
        return text
    duration_s = source_end_s - source_start_s
    left_ratio = max(0.0, (selected_start_s - source_start_s) / duration_s)
    right_ratio = min(1.0, (selected_end_s - source_start_s) / duration_s)
    left_char = min(len(text), round(len(text) * left_ratio))
    right_char = min(len(text), max(left_char + 1, round(len(text) * right_ratio)))
    return text[left_char:right_char].strip()


class SpeakerExemplarPool:
    """Maintain one quality-ranked exemplar per global MTD speaker label."""

    SAMPLE_RATE = 16000

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
        decisions: list[dict[str, Any]] = []
        for item in parsed:
            decision = self._evaluate_candidate(
                current_audio,
                item,
                parsed,
                source_segment_id=source_segment_id,
            )
            decisions.append(decision)
        return decisions

    def _evaluate_candidate(
        self,
        current_audio: np.ndarray,
        item: DiarizationSegment,
        all_segments: list[DiarizationSegment],
        *,
        source_segment_id: int,
    ) -> dict[str, Any]:
        """Evaluate one speaker occurrence and optionally mutate the pool."""

        cfg = self.config
        blockers = [
            (other.start_s, other.end_s)
            for other in all_segments
            if other.speaker_id != item.speaker_id
        ]
        overlap_s = sum(
            _interval_overlap(item.start_s, item.end_s, left, right)
            for left, right in blockers
        )
        overlap_ratio = min(1.0, overlap_s / max(item.duration_s, 1e-6))
        exclusive = sorted(
            _subtract_intervals(item.start_s, item.end_s, blockers),
            key=lambda pair: pair[1] - pair[0],
            reverse=True,
        )
        use_non_overlap = bool(cfg["prefer_non_overlap"]) and bool(exclusive)
        if use_non_overlap:
            start_s, end_s = exclusive[0]
        elif bool(cfg["allow_overlap_fallback"]):
            start_s, end_s = item.start_s, item.end_s
        else:
            return {"speaker_id": item.speaker_id, "action": "reject", "reason": "overlap"}

        hard_max_s = float(cfg["hard_max_duration_s"])
        if end_s - start_s > hard_max_s:
            center_s = (start_s + end_s) / 2.0
            start_s = center_s - hard_max_s / 2.0
            end_s = start_s + hard_max_s
        duration_s = end_s - start_s
        selected_text = _crop_text_by_time(
            item.text,
            source_start_s=item.start_s,
            source_end_s=item.end_s,
            selected_start_s=start_s,
            selected_end_s=end_s,
        )
        start_sample = max(0, round(start_s * self.SAMPLE_RATE))
        end_sample = min(len(current_audio), round(end_s * self.SAMPLE_RATE))
        candidate_audio = np.asarray(current_audio[start_sample:end_sample], dtype=np.float32)
        boundary_margin_s = min(
            start_s,
            max(0.0, len(current_audio) / self.SAMPLE_RATE - end_s),
        )
        score, quality = self._quality(
            candidate_audio,
            duration_s=duration_s,
            overlap_ratio=0.0 if use_non_overlap else overlap_ratio,
            boundary_margin_s=boundary_margin_s,
            used_non_overlap=use_non_overlap,
        )
        previous = self.items.get(item.speaker_id)
        min_duration_s = float(
            cfg["min_update_duration_s"]
            if previous is not None
            else cfg["min_register_duration_s"]
        )
        reason = "accepted"
        if duration_s < min_duration_s:
            reason = "too_short"
        elif overlap_ratio > float(cfg["max_overlap_ratio"]) and not use_non_overlap:
            reason = "too_much_overlap"
        elif quality["rms_dbfs"] < float(cfg["min_rms_dbfs"]):
            reason = "too_quiet"
        elif quality["clipping_ratio"] > float(cfg["max_clipping_ratio"]):
            reason = "clipped"

        action = "reject"
        if reason == "accepted":
            if previous is None:
                if len(self.items) < int(cfg["max_speakers"]):
                    action = "register"
                else:
                    reason = "pool_full"
            elif score >= previous.score + float(cfg["replace_score_margin"]):
                action = "replace"
            elif (
                previous.duration_s < float(cfg["preferred_min_duration_s"])
                and duration_s >= float(cfg["preferred_min_duration_s"])
                and score >= previous.score - float(cfg["replace_score_margin"])
            ):
                action = "replace"
            else:
                reason = "not_better"
        if action in {"register", "replace"}:
            self.items[item.speaker_id] = SpeakerExemplar(
                speaker_id=item.speaker_id,
                audio=candidate_audio.copy(),
                text=selected_text,
                score=score,
                quality=quality,
                source_segment_id=source_segment_id,
                source_start_s=start_s,
                source_end_s=end_s,
            )
            self.version += 1
        return {
            "speaker_id": item.speaker_id,
            "action": action,
            "reason": reason,
            "candidate_start_s": start_s,
            "candidate_end_s": end_s,
            "text": selected_text,
            "quality": quality,
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
