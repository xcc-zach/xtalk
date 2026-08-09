"""Speaker-diarization model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypedDict

from ..registry import model_type


class DiarizationSegment(TypedDict):
    """One timestamped speaker-diarization segment."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


@dataclass(frozen=True)
class DiarizationResult:
    """Result returned by one full-snapshot diarization request.

    Parameters
    ----------
    raw_text : str
        Full timestamped model output, including any fixed decoder prefix.
    current_segments : list[DiarizationSegment]
        Parsed segments cropped to current-audio-local coordinates.
    latency_ms : float
        End-to-end decode latency measured by the runtime or client.
    metrics : dict[str, Any]
        Optional runtime token/cache/latency metrics.
    """

    raw_text: str = ""
    current_segments: list[DiarizationSegment] = field(default_factory=list)
    latency_ms: float = 0.0
    metrics: dict[str, Any] = field(default_factory=dict)


@model_type
class SpeakerDiarization(ABC):
    """Decode timestamped speaker labels for complete PCM snapshots."""

    @abstractmethod
    async def decode_snapshot(
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
        """Decode one immutable audio snapshot.

        Parameters
        ----------
        request_id : str
            Unique request identifier used for cancellation and tracing.
        pcm16 : bytes
            Mono little-endian signed PCM16 containing exemplars and current audio.
        sample_rate : int
            PCM sampling rate in hertz.
        decoder_prefix : str
            Fixed timestamp-plus-speaker transcript for exemplar audio slots.
        context_seconds : float
            Duration occupied by exemplar audio and configured silence.
        current_audio_seconds : float
            Duration of the current VAD snapshot.
        is_final : bool
            Whether the snapshot closes its VAD segment.

        Returns
        -------
        DiarizationResult
            Parsed current-local diarization output.
        """

    async def cancel(self, request_id: str) -> None:
        """Best-effort cancel an in-flight request.

        Parameters
        ----------
        request_id : str
            Identifier previously passed to :meth:`decode_snapshot`.
        """

        del request_id

    async def close(self) -> None:
        """Release client-side resources owned by this model instance."""

        return None
