import asyncio
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, AsyncIterator, Iterable

from ..registry import model_type


@model_type(aliases=["tts"])
class TTS(ABC):
    """Abstract base class for text-to-speech engines.

    Notes
    -----
    ``synthesize`` is the required baseline API for every implementation.
    Streaming-capable engines should additionally override
    ``synthesize_stream``; non-streaming engines should inherit the default
    compatibility wrapper. The inherited streaming helpers do not by
    themselves declare native streaming capability.
    """

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize audio for a full text input.

        Parameters
        ----------
        text : str
            Text to synthesize.

        Returns
        -------
        bytes
            PCM 16-bit mono audio bytes at 48 kHz.

        Notes
        -----
        Every TTS implementation, including streaming backends, must provide
        this method.
        """
        pass

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        """Stream synthesized audio chunks for a text input.

        Parameters
        ----------
        text : str
            Text to synthesize.
        **kwargs
            Model-specific streaming options.

        Yields
        ------
        bytes
            PCM 16-bit mono audio bytes at 48 kHz.

        Notes
        -----
        Override this method only when the backend supports native streaming
        synthesis. The default implementation yields a single chunk produced
        by ``synthesize`` for compatibility and should not be treated as a
        declaration of streaming support.
        """
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Asynchronously synthesize audio for text.

        Parameters
        ----------
        text : str
            Text to synthesize.
        **kwargs
            Model-specific synthesis options.

        Returns
        -------
        bytes
            Synthesized PCM audio bytes.

        Notes
        -----
        This method is an optional async optimization. Implementations may
        inherit the default executor-based wrapper.
        """
        loop = asyncio.get_running_loop()
        func = partial(self.synthesize, text, **kwargs)
        result: bytes = await loop.run_in_executor(None, func)
        return result

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Asynchronously stream synthesized audio chunks.

        Parameters
        ----------
        text : str
            Text to synthesize.
        **kwargs
            Model-specific synthesis options.

        Yields
        ------
        bytes
            Streamed PCM audio chunks.

        Notes
        -----
        This method is an optional async optimization for streaming-capable
        backends. When not overridden, it asynchronously iterates over
        ``synthesize_stream``.
        """
        loop = asyncio.get_running_loop()
        iterable = self.synthesize_stream(text, **kwargs)
        iterator = iter(iterable)

        try:
            while True:

                def safe_next():
                    try:
                        return next(iterator)
                    except StopIteration:
                        return None

                chunk = await loop.run_in_executor(None, safe_next)
                if chunk is None:
                    break
                yield chunk
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @abstractmethod
    def clone(self) -> "TTS":
        """Clone the TTS engine for a new session.

        Returns
        -------
        TTS
            Session-safe clone.
        """
        pass

    def set_voice(self, voice_names: list[str]) -> None:
        """Update the active voice selection.

        Parameters
        ----------
        voice_names : list[str]
            One or more voice names understood by the implementation.
        """
        pass

    def set_emotion(self, emotion: str | list[float]) -> None:
        """Update the active synthesis emotion.

        Parameters
        ----------
        emotion : str | list[float]
            Emotion label or model-specific emotion vector.
        """
        pass


class StreamingTextTTS(ABC):
    """Abstract base class for live text-streaming TTS engines.

    Notes
    -----
    This interface models engines that accept text incrementally and emit audio
    concurrently from the same session. Implementations may also inherit
    ``TTS`` when they support the regular full-text synthesis API.
    """

    @abstractmethod
    async def start(self) -> None:
        """Start a live text-streaming synthesis session.

        Notes
        -----
        Implementations typically open an upstream WebSocket connection and
        send the provider-specific start event here.
        """
        pass

    @abstractmethod
    async def append_text(self, text: str) -> None:
        """Append incremental text to the active synthesis session.

        Parameters
        ----------
        text : str
            Text fragment produced by the upstream LLM or agent.
        """
        pass

    @abstractmethod
    async def flush(self) -> None:
        """Request synthesis of currently buffered upstream text.

        Notes
        -----
        ``TTSManager`` calls this when it receives ``TurnTTSFlushRequested``.
        Implementations should not rely on sentence-boundary flushes.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the active synthesis session and release connection resources."""
        pass

    @abstractmethod
    def audio_stream(self) -> AsyncIterator[bytes]:
        """Stream generated PCM audio chunks from the active session.

        Yields
        ------
        bytes
            PCM 16-bit mono audio bytes, preferably at 48 kHz.
        """
        pass

    @abstractmethod
    def clone(self) -> "StreamingTextTTS":
        """Clone the streaming TTS engine for a new session.

        Returns
        -------
        StreamingTextTTS
            Session-safe clone with independent live connection state.
        """
        pass
