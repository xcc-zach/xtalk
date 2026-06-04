from abc import abstractmethod, ABC
from typing import Iterable, AsyncIterator, Any, Optional
import numpy as np
import asyncio
import threading
from functools import partial
from enum import Enum
from dataclasses import dataclass
from .utils import MockStreamRecognizer


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


class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

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
        pass

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

    @abstractmethod
    def clone(self) -> "VAD":
        """Clone the VAD instance for a new session.

        Returns
        -------
        VAD
            Clone with shared weights and independent runtime state.
        """
        pass


class SpeechEnhancer(ABC):
    """Abstract base class for speech enhancement engines.

    Notes
    -----
    Inputs and outputs use PCM 16-bit mono audio bytes at 16 kHz.
    """

    @abstractmethod
    def enhance(self, audio: bytes) -> bytes:
        """Enhance an audio frame.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

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

    async def async_enhance(self, audio: bytes) -> bytes:
        """Asynchronously enhance audio.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes at 16 kHz.

        Returns
        -------
        bytes
            Enhanced PCM audio bytes.
        """

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.enhance, audio)

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
        # Default cosine similarity implementation
        e1 = embedding1.astype(np.float32, copy=False).ravel()
        e2 = embedding2.astype(np.float32, copy=False).ravel()
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))


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


class TurnDetectionAction(Enum):
    DO_NOTHING = 1
    STOP_SPEAKING = 2
    START_GENERATION = 3


class TurnDetectionSemantic(Enum):
    IDLE = "idle"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    SHOULD_BACKCHANNEL = "should_backchannel"


class TurnVADResult(Enum):
    SPEECH = 1
    SILENCE = 2


@dataclass(frozen=True)
class TurnDetectionResult:
    """Decision emitted by a turn detector.

    Attributes
    ----------
    action : TurnDetectionAction
        Immediate action the service should take.
    semantic : TurnDetectionSemantic
        Semantic interpretation of the current conversational state.
    vad_result : TurnVADResult | None
        Optional VAD result; only used when VAD is absent
    """

    action: TurnDetectionAction
    semantic: TurnDetectionSemantic
    vad_result: TurnVADResult | None = None


class TurnDetector(ABC):
    """Abstract interface for turn-taking detectors."""

    def __init__(self) -> None:
        self._listening = True
        self._listening_lock = threading.Lock()
        self._listening_async_lock = asyncio.Lock()

    @property
    def listening(self) -> bool:
        """Return whether the detector is currently listening for user turns.

        Returns
        -------
        bool
            Current listening state.
        """
        return self._listening

    @listening.setter
    def listening(self, value: bool) -> None:
        """Update the listening state.

        Parameters
        ----------
        value : bool
            New listening state.
        """
        self._listening = value

    def listening_lock(self, is_async: bool = True):
        """Return the lock guarding listening state changes.

        Parameters
        ----------
        is_async : bool, optional
            Whether to return the async lock instead of the threading lock.

        Returns
        -------
        asyncio.Lock | threading.Lock
            Lock object matching the requested concurrency model.
        """
        return self._listening_async_lock if is_async else self._listening_lock

    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Detect conversational turn state from audio and/or text.

        Parameters
        ----------
        audio : bytes | None, optional
            Current PCM 16-bit mono audio frame at 16 kHz.
        text : str | None, optional
            ASR text for the current turn.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech. This may be
            provided without ``audio`` or ``text``.
        speech_pause : bool | None, optional
            Whether the user appears to have paused speaking. This is typically
            provided together with ``text``.

        Returns
        -------
        TurnDetectionResult
            Turn-detection decision for the current input.
        """
        pass

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Asynchronously detect conversational turn state.

        Parameters
        ----------
        audio : bytes | None, optional
            Current PCM 16-bit mono audio frame at 16 kHz.
        text : str | None, optional
            ASR text for the current turn.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech. This may be
            provided without ``audio`` or ``text``.
        speech_pause : bool | None, optional
            Whether the user appears to have paused speaking.

        Returns
        -------
        TurnDetectionResult
            Turn-detection decision for the current input.
        """
        loop = asyncio.get_running_loop()
        func = partial(
            self.detect,
            audio=audio,
            text=text,
            speech_start=speech_start,
            speech_pause=speech_pause,
        )
        result: TurnDetectionResult = await loop.run_in_executor(None, func)
        return result

    @abstractmethod
    def clone(self) -> "TurnDetector":
        """Clone the turn detector for a new session.

        Returns
        -------
        TurnDetector
            Session-safe clone.
        """
        pass
