import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["punt_restorer_model"])
class PuntRestorer(ABC):
    """Abstract base class for punctuation restoration models."""

    @abstractmethod
    def restore(self, text: str) -> str:
        """Restore punctuation in text.

        Parameters
        ----------
        text : str
            Text without reliable punctuation.

        Returns
        -------
        str
            Text with restored punctuation.
        """
        pass

    async def async_restore(self, text: str) -> str:
        """Asynchronously restore punctuation in text.

        Parameters
        ----------
        text : str
            Text without reliable punctuation.

        Returns
        -------
        str
            Restored text.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.restore, text)
