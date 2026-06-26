import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["vad"])
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """Determine whether an audio frame contains speech.

        Parameters
        ----------
        frame : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bool
            ``True`` if speech is detected, otherwise ``False``.
        """
        pass

    async def async_is_speech(self, frame: bytes) -> bool:
        """Asynchronously determine whether an audio frame contains speech.

        Parameters
        ----------
        frame : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bool
            ``True`` if speech is detected, otherwise ``False``.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.is_speech, frame)
        return bool(result)

    @abstractmethod
    def clone(self) -> "VAD":
        """Clone the VAD instance for a new session.

        Returns
        -------
        VAD
            Clone with shared weights and independent runtime state.
        """
        pass
