import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterable

from ..registry import model_type


@model_type(aliases=["captioner"])
class Captioner(ABC):
    """Abstract base class for audio captioning models."""

    @abstractmethod
    def caption(self, audio: bytes) -> str:
        """Generate a caption for audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        str
            Generated caption text.
        """

    def caption_stream(self, audio: bytes) -> Iterable[str]:
        """Stream caption text for audio input.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Yields
        ------
        str
            Streamed caption text.
        """
        yield self.caption(audio)

    async def async_caption(self, audio: bytes) -> str:
        """Asynchronously caption audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        str
            Generated caption text.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.caption, audio)

    async def async_caption_stream(self, audio: bytes) -> AsyncIterator[str]:
        """Asynchronously stream caption text.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Yields
        ------
        str
            Streamed caption text.
        """
        loop = asyncio.get_running_loop()
        iterator = iter(self.caption_stream(audio))
        sentinel = object()

        try:
            while True:

                def safe_next():
                    try:
                        return next(iterator)
                    except StopIteration:
                        return sentinel

                chunk = await loop.run_in_executor(None, safe_next)
                if chunk is sentinel:
                    break
                yield chunk
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
