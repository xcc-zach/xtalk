import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["speech_enhancer"])
class SpeechEnhancer(ABC):
    """Abstract base class for speech enhancement engines.

    Notes
    -----
    Inputs and outputs use PCM 16-bit mono audio bytes at 16 kHz.
    """

    @abstractmethod
    def enhance(self, audio: bytes, far: bytes) -> bytes:
        """Enhance an audio frame.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.
        far : bytes
            Far-end reference PCM 16-bit mono audio bytes at 16 kHz. The
            upstream audio pipeline guarantees that it has the same byte length
            as ``audio``.

        Returns
        -------
        bytes
            Enhanced PCM audio bytes.
        """
        pass

    def flush(self) -> bytes:
        """Flush any internally buffered audio.

        Returns
        -------
        bytes
            Remaining enhanced PCM audio bytes.
        """
        return b""

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes:
        """Asynchronously enhance audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.
        far : bytes
            Far-end reference PCM 16-bit mono audio bytes at 16 kHz. The
            upstream audio pipeline guarantees that it has the same byte length
            as ``audio``.

        Returns
        -------
        bytes
            Enhanced PCM audio bytes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.enhance, audio, far)

    async def async_flush(self) -> bytes:
        """Asynchronously flush buffered audio.

        Returns
        -------
        bytes
            Remaining enhanced PCM audio bytes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.flush)

    @abstractmethod
    def reset(self) -> None:
        """Reset internal buffers and caches."""
        pass

    @abstractmethod
    def clone(self) -> "SpeechEnhancer":
        """Clone the speech enhancer for a new session.

        Returns
        -------
        SpeechEnhancer
            Clone with shared weights and isolated runtime state.
        """
        pass
