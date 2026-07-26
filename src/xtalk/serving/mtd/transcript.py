"""Parsing and rendering helpers for timestamped MTD output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"(?:\[(?P<speaker>S\d+)\])?"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_NESTED_SPEAKER_RE = re.compile(r"^\[*\[(S\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class DiarizationSegment:
    """One timestamped speaker segment."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str

    @property
    def duration_s(self) -> float:
        """Return the non-negative segment duration."""

        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, object]:
        """Serialize this segment into an event-safe dictionary."""

        return {
            "start_s": self.start_s,
            "end_s": self.end_s,
            "speaker_id": self.speaker_id,
            "text": self.text,
        }


def parse_segments(
    text: str,
    *,
    max_time_s: float | None = None,
) -> list[DiarizationSegment]:
    """Parse MTD timestamp output without producing an ``UNKNOWN`` label.

    Parameters
    ----------
    text : str
        Raw model output.
    max_time_s : float | None, optional
        Optional upper timestamp bound.

    Returns
    -------
    list[DiarizationSegment]
        Valid parsed segments. An implicit label inherits the previous explicit
        label; a cold-start implicit label falls back to ``S01``.
    """

    result: list[DiarizationSegment] = []
    last_speaker = "S01"
    for match in _SEGMENT_RE.finditer(text or ""):
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        if end_s <= start_s:
            continue
        if max_time_s is not None and start_s > max_time_s + 1.0:
            continue
        segment_text = match.group("text")
        nested = _LEADING_NESTED_SPEAKER_RE.match(segment_text)
        explicit_speaker = nested.group(1) if nested else match.group("speaker")
        if explicit_speaker:
            last_speaker = explicit_speaker.upper()
        if nested:
            segment_text = segment_text[nested.end() :]
        bounded_end_s = min(end_s, max_time_s) if max_time_s is not None else end_s
        if bounded_end_s <= start_s:
            continue
        result.append(
            DiarizationSegment(
                start_s=max(0.0, start_s),
                end_s=bounded_end_s,
                speaker_id=last_speaker,
                text=re.sub(r"\s+", " ", segment_text).strip(),
            )
        )
    return result


def render_segments(
    segments: Iterable[DiarizationSegment | dict[str, object]],
    *,
    precision: int = 2,
) -> str:
    """Render segments in the MTD timestamp-plus-speaker format."""

    rendered: list[str] = []
    for item in segments:
        if isinstance(item, DiarizationSegment):
            segment = item
        else:
            segment = DiarizationSegment(
                start_s=float(item["start_s"]),
                end_s=float(item["end_s"]),
                speaker_id=str(item["speaker_id"]),
                text=str(item.get("text") or ""),
            )
        rendered.append(
            f"[{segment.start_s:.{precision}f}][{segment.speaker_id}]"
            f"{segment.text}[{segment.end_s:.{precision}f}]"
        )
    return " ".join(rendered)


def offset_segments(
    segments: Iterable[dict[str, object]],
    *,
    offset_s: float,
    turn_id: int,
    segment_id: int,
) -> list[dict[str, object]]:
    """Convert current-local segments to a session-source timeline."""

    result: list[dict[str, object]] = []
    for item in segments:
        result.append(
            {
                **item,
                "start_s": float(item["start_s"]) + offset_s,
                "end_s": float(item["end_s"]) + offset_s,
                "turn_id": turn_id,
                "segment_id": segment_id,
            }
        )
    return result
