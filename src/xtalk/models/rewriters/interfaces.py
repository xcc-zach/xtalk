import asyncio
from abc import ABC, abstractmethod

from ..registry import model_type


@model_type(aliases=["caption_rewriter"])
class Rewriter(ABC):
    """Abstract interface for text rewriting helpers."""

    @abstractmethod
    def rewrite(self, input: str) -> str:
        """Rewrite input text.

        Parameters
        ----------
        input : str
            Source text to rewrite.

        Returns
        -------
        str
            Rewritten text.
        """
        pass

    async def async_rewrite(self, input: str) -> str:
        """Asynchronously rewrite input text.

        Parameters
        ----------
        input : str
            Source text to rewrite.

        Returns
        -------
        str
            Rewritten text.
        """

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rewrite, input)
