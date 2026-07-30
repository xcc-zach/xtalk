import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial

from ..registry import model_type


@dataclass(frozen=True)
class ForcedAlignmentUnit:
    """One text unit aligned onto a synthesized audio timeline."""

    text: str
    start_ms: float
    end_ms: float
    char_start: int = -1
    char_end: int = -1


@model_type
class ForcedAligner(ABC):
    """Abstract interface for forced alignment models."""

    @abstractmethod
    def align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """Align text units against 48 kHz PCM audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 48 kHz.
        text : str
            Original text that the audio speaks.
        language : str | None, optional
            Optional model-specific language hint.
        """
        pass

    async def async_align(
        self,
        *,
        audio: bytes,
        text: str,
        language: str | None = None,
    ) -> list[ForcedAlignmentUnit]:
        """Asynchronously align text units against 48 kHz PCM audio."""
        loop = asyncio.get_running_loop()
        func = partial(
            self.align,
            audio=audio,
            text=text,
            language=language,
        )
        result: list[ForcedAlignmentUnit] = await loop.run_in_executor(None, func)
        return result

    @abstractmethod
    def clone(self) -> "ForcedAligner":
        """Clone the aligner for a new service session."""
        pass
