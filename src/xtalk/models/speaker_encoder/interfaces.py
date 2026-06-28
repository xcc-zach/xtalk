import asyncio
from abc import ABC, abstractmethod

import numpy as np

from ..registry import model_type


@model_type(aliases=["speaker_encoder"])
class SpeakerEncoder(ABC):
    """Abstract base class for speaker embedding models."""

    @abstractmethod
    def extract(self, audio: bytes) -> np.ndarray:
        """Generate a speaker embedding vector.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        np.ndarray
            Speaker embedding vector.
        """
        pass

    async def async_extract(self, audio: bytes) -> np.ndarray:
        """Asynchronously extract a speaker embedding.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        np.ndarray
            Speaker embedding vector.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.extract, audio)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two speaker embeddings.

        Parameters
        ----------
        embedding1 : np.ndarray
            First speaker embedding.
        embedding2 : np.ndarray
            Second speaker embedding.

        Returns
        -------
        float
            Cosine similarity score.
        """
        e1 = embedding1.astype(np.float32, copy=False).ravel()
        e2 = embedding2.astype(np.float32, copy=False).ravel()
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))
