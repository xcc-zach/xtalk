import asyncio
from abc import ABC, abstractmethod


class Rewriter(ABC):
    @abstractmethod
    def rewrite(self, input: str) -> str:
        """Rewrite input text."""
        pass

    async def async_rewrite(self, input: str) -> str:
        """Async wrapper running the sync rewrite in a thread pool."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.rewrite, input)
