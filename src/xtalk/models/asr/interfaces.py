import asyncio
from abc import ABC, abstractmethod
from functools import partial

from ..audio_utils import MockStreamRecognizer
from ..registry import model_type


@model_type(aliases=["asr"])
class ASR(ABC):
    """Abstract interface for automatic speech recognition."""

    def _get_mock_recognizer(self) -> MockStreamRecognizer:
        """Create the fallback streaming recognizer on first use.

        Returns
        -------
        MockStreamRecognizer
            Helper that adapts one-shot ASR implementations to streaming calls.
        """
        recognizer = getattr(self, "_mock_recognizer", None)
        if recognizer is None:
            recognizer = MockStreamRecognizer(
                self.async_recognize,
                window_size=10,
            )
            setattr(self, "_mock_recognizer", recognizer)
        return recognizer

    @abstractmethod
    def recognize(self, audio: bytes) -> str:
        """Recognize a full audio buffer.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        str
            Recognized text.
        """
        pass

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Recognize audio incrementally in streaming mode.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the caller is asking the ASR to treat the current point as
            a temporary boundary and optionally flush any tail audio that would
            otherwise remain buffered. This is only a decoding hint. It does
            not mean the streaming state must be reset, and previously
            recognized text for the session must be preserved so later audio
            can continue from the accumulated result.
        chat_history : str | None, optional
            Serialized chat history for the current session, excluding the
            in-progress turn when unavailable.

        Returns
        -------
        str
            Current recognition result.
        """
        del chat_history
        recognizer = self._get_mock_recognizer()

        if not audio:
            return recognizer.recognized_text

        return recognizer.recognize(audio, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        """Return the preferred streaming chunk size.

        Returns
        -------
        int | None
            Recommended byte count for each chunk passed to
            ``recognize_stream``, or ``None`` when no preference is provided.
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recognition state."""
        pass

    @abstractmethod
    def clone(self) -> "ASR":
        """Clone the ASR instance for a new session.

        Returns
        -------
        ASR
            Clone with shared weights and independent runtime state.
        """
        pass

    async def async_recognize(self, audio: bytes) -> str:
        """Asynchronously recognize a full audio buffer.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        str
            Recognized text.
        """
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(None, self.recognize, audio)
        return result

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Asynchronously recognize incremental audio input.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the caller is asking the ASR to treat the current point as
            a temporary boundary and optionally flush any tail audio that would
            otherwise remain buffered. This is only a decoding hint. It does
            not mean the streaming state must be reset, and previously
            recognized text for the session must be preserved so later audio
            can continue from the accumulated result.
        chat_history : str | None, optional
            Serialized chat history for the current session, excluding the
            in-progress turn when unavailable.

        Returns
        -------
        str
            Current recognition result.
        """
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(
            None,
            partial(
                self.recognize_stream,
                audio,
                is_final=is_final,
                chat_history=chat_history,
            ),
        )
        return result
