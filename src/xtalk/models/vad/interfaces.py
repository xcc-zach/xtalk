import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["vad"])
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def set_threshold(self, *, threshold: float) -> None:
        """Configure the speech probability threshold.

        Parameters
        ----------
        threshold : float
            Probability threshold between zero and one.

        """

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

    def reset(self) -> None:
        """Reset session-local VAD state and release external resources."""
        return

    def detect_frames(self, frames: list[bytes]) -> list[bool]:
        """Detect consecutive frames while preserving stream state and order.

        Parameters
        ----------
        frames : list[bytes]
            Ordered PCM16 mono frames using the single-frame audio contract.

        Returns
        -------
        list[bool]
            One speech decision per input frame, in input order. Empty input
            returns an empty list without calling the model.

        Notes
        -----
        The fallback calls ``is_speech`` sequentially. Native batch clients
        may override this method without changing the single-frame interface.
        """
        return [self.is_speech(frame) for frame in frames]

    async def async_detect_frames(self, frames: list[bytes]) -> list[bool]:
        """Asynchronously detect ordered frames using sequential fallback calls.

        Parameters
        ----------
        frames : list[bytes]
            Ordered PCM16 mono frames using the single-frame audio contract.

        Returns
        -------
        list[bool]
            One speech decision per frame, in input order; empty input
            returns an empty list without calling the model.

        Notes
        -----
        Calls ``async_is_speech`` sequentially to retain existing async
        implementations and recurrent state. Native async batch clients may
        override this method. Overriding only the synchronous batch method
        does not change this asynchronous fallback.
        """
        return [await self.async_is_speech(frame) for frame in frames]

    @abstractmethod
    def clone(self) -> "VAD":
        """Clone the VAD instance for a new session.

        Returns
        -------
        VAD
            Clone with shared weights and independent runtime state.
        """
