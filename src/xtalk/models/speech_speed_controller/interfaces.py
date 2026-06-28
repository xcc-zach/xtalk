import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["speech_speed_controller"])
class SpeechSpeedController(ABC):
    """Interface for TTS speed controllers."""

    @abstractmethod
    def process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes:
        """Apply a speed adjustment to synthesized audio.

        Parameters
        ----------
        audio_bytes : bytes
            Synthesized audio bytes.
        speed : float, optional
            Speed multiplier.

        Returns
        -------
        bytes
            Processed audio bytes.
        """
        pass

    async def async_process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes:
        """Asynchronously apply a speed adjustment to audio.

        Parameters
        ----------
        audio_bytes : bytes
            Synthesized audio bytes.
        speed : float, optional
            Speed multiplier.

        Returns
        -------
        bytes
            Processed audio bytes.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self.process(audio_bytes, speed)
        )
