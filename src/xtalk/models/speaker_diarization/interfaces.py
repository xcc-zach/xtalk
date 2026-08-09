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
    segments : list[DiarizationSegment]
        Parsed segments relative to the supplied snapshot.
    latency_ms : float
        End-to-end decode latency measured by the runtime or client.
    metrics : dict[str, Any]
        Optional runtime token/cache/latency metrics.
    """

    raw_text: str = ""
    segments: list[DiarizationSegment] = field(default_factory=list)
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
        is_final: bool,
    ) -> DiarizationResult:
        """Decode one immutable audio snapshot.

        Parameters
        ----------
        request_id : str
            Unique request identifier used for cancellation and tracing.
        pcm16 : bytes
            Mono little-endian signed PCM16 for the current audio snapshot.
        sample_rate : int
            PCM sampling rate in hertz.
        is_final : bool
            Whether the snapshot closes its VAD segment.

        Returns
        -------
        DiarizationResult
            Parsed snapshot-local diarization output.
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
