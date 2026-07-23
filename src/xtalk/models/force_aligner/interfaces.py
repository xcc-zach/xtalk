import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from ..registry import model_type


@dataclass(frozen=True)
class ForceAlignmentUnit:
    """One text unit aligned onto a synthesized audio timeline."""

    text: str
    start_ms: float
    end_ms: float
    char_start: int = -1
    char_end: int = -1


@model_type(aliases=["force_aligner"])
class ForceAligner(ABC):
    """Abstract interface for forced alignment models."""

    @abstractmethod
    def align(
        self,
        *,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str | None = None,
    ) -> list[ForceAlignmentUnit]:
        """Align text units against PCM audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.
        text : str
            Original text that the audio speaks.
        sample_rate : int
            Audio sample rate.
        language : str | None, optional
            Optional model-specific language hint.
        """
        pass

    async def async_align(
        self,
        *,
        audio: bytes,
        text: str,
        sample_rate: int,
        language: str | None = None,
    ) -> list[ForceAlignmentUnit]:
        """Asynchronously align text units against PCM audio."""
        loop = asyncio.get_running_loop()
        func = partial(
            self.align,
            audio=audio,
            text=text,
            sample_rate=sample_rate,
            language=language,
        )
        result: list[ForceAlignmentUnit] = await loop.run_in_executor(None, func)
        return result

    @abstractmethod
    def clone(self) -> "ForceAligner":
        """Clone the aligner for a new service session."""
        pass
