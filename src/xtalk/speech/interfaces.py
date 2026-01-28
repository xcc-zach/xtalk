from abc import abstractmethod, ABC
from typing import Iterable, AsyncIterator, Any, Optional
import numpy as np
import asyncio
from functools import partial
from enum import Enum
from dataclasses import dataclass
from .utils import MockStreamRecognizer


class ASR(ABC):
    """Abstract interface for automatic speech recognition."""

    def _get_mock_recognizer(self) -> MockStreamRecognizer:
        """
        Lazily create a MockStreamRecognizer if not present.
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
        """Recognize audio in a single pass."""
        pass

    def recognize_stream(self, audio: bytes, *, is_final: bool = False) -> str:
        """Incremental streaming interface."""
        recognizer = self._get_mock_recognizer()

        if not audio:
            return recognizer.recognized_text

        return recognizer.recognize(audio, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        """Optional hint for how many bytes to accumulate before decoding."""
        return None

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recognition state."""
        pass

    @abstractmethod
    def clone(self) -> "ASR":
        """Clone the ASR instance with shared weights and separate state."""
        pass

    async def async_recognize(self, audio: bytes) -> str:
        """Async wrapper for one-shot recognition."""
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(None, self.recognize, audio)
        return result

    async def async_recognize_stream(
        self, audio: bytes, *, is_final: bool = False
    ) -> str:
        """Async wrapper for streaming recognition."""
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(
            None, partial(self.recognize_stream, audio, is_final=is_final)
        )
        return result


class TTS(ABC):
    """
    Abstract base class for Text-to-Speech (TTS) engines.
    """

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech.

        Args:
            text (str): The text to convert to speech.

        Returns:
            bytes: The synthesized speech as audio data. PCM 16bit mono, 48000Hz bytes.
        """
        pass

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        """
        Convert text to speech. Streaming mode.

        Args:
            text (str): The text to convert to speech.

        Returns:
            Iterable[bytes]: The synthesized speech as audio data. PCM 16bit mono, 48000Hz bytes.
        """
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        """Async wrapper around the synchronous synthesize call."""
        loop = asyncio.get_running_loop()
        func = partial(self.synthesize, text, **kwargs)
        result: bytes = await loop.run_in_executor(None, func)
        return result

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        """Async wrapper consuming the sync streaming generator."""
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
        """
        Create an original clone of the TTS engine.

        Returns:
            TTS: A clone of the TTS engine.
        """
        pass

    def set_voice(self, voice_names: list[str]) -> None:
        """
        Set the voice or merge several voices for speech synthesis.

        Args:
            voice_names (list[str]): A list of voice names to set.
        """
        pass

    def set_emotion(self, emotion: str | list[float]) -> None:
        """
        Set the emotion for speech synthesis.

        Args:
            emotion (str | list[float]): The emotion to set, either as a string label or a list of float values.
        """
        pass


class Captioner(ABC):
    """
    Abstract base class for Audio Captioning models.
    """

    @abstractmethod
    def caption(self, audio: bytes) -> str:
        """
        Generate a caption for the given audio.

        Args:
            audio (bytes): The audio data to caption. PCM 16bit mono, 16000Hz bytes.
        Returns:
            str: The generated caption.
        """

    def caption_stream(self, audio: bytes) -> Iterable[str]:
        """
        Generate a caption for the given audio. Streaming mode.

        Args:
            audio (bytes): The audio data to caption. PCM 16bit mono, 16000Hz bytes.
        Returns:
            Iterable[str]: Streamed generated caption.
        """
        yield self.caption(audio)

    async def async_caption(self, audio: bytes) -> str:
        """Async wrapper for caption()."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.caption, audio)

    async def async_caption_stream(self, audio: bytes) -> AsyncIterator[str]:
        """Async wrapper streaming caption text."""

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
    """
    Abstract base class for text punt restoration models.
    """

    @abstractmethod
    def restore(self, text: str) -> str:
        """
        Restore punt in the given text.
        """
        pass

    async def async_restore(self, text: str) -> str:
        """Async wrapper for restore()."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.restore, text)


class VAD(ABC):
    """
    Abstract base class for Voice Activity Detection (VAD) engines.
    """

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        """
        Determine if the given audio frame contains speech.

        Args:
            frame (bytes): The audio frame to analyze. PCM 16bit mono, 16000Hz bytes.

        Returns:
            bool: True if the frame contains speech, False otherwise.
        """
        pass

    async def async_is_speech(self, frame: bytes) -> bool:
        """Async wrapper for is_speech()."""

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.is_speech, frame)
        return bool(result)


class SpeechEnhancer(ABC):
    """
    Abstract base class for Speech Enhancement engines.

    Inputs/outputs are PCM 16bit mono 16000Hz raw bytes.
    """

    @abstractmethod
    def enhance(self, audio: bytes) -> bytes:
        """
        Enhance the given audio frame.

        Args:
            audio (bytes): The audio data to enhance. PCM 16bit mono, 16000Hz bytes.

        Returns:
            bytes: The enhanced audio data. PCM 16bit mono, 16000Hz bytes.
        """
        pass

    def flush(self) -> bytes:
        """
        Flush remaining buffered audio (call at end of audio stream).

        Some enhancers buffer audio internally for processing. This method
        should be called when the audio stream ends to retrieve any remaining
        enhanced audio.

        Returns:
            bytes: Remaining enhanced audio data. PCM 16bit mono, 16000Hz bytes.
        """
        return b""

    async def async_enhance(self, audio: bytes) -> bytes:
        """Async wrapper for enhance()."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.enhance, audio)

    async def async_flush(self) -> bytes:
        """Async wrapper for flush()."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.flush)

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state (buffers, caches, etc.).
        """
        pass

    @abstractmethod
    def clone(self) -> "SpeechEnhancer":
        """Clone the speech enhancer with shared weights and isolated state."""
        pass


class SpeakerEncoder(ABC):

    @abstractmethod
    def extract(self, audio: bytes) -> np.ndarray:
        """Generate a speaker embedding vector."""
        pass

    async def async_extract(self, audio: bytes) -> np.ndarray:
        """Async wrapper for extract()."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.extract, audio)

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between embeddings (default cosine)."""
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
        """Process audio and apply speed adjustments."""
        pass

    async def async_process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes:
        """Async wrapper around process()."""

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


@dataclass(frozen=True)
class TurnDetectionResult:
    action: TurnDetectionAction
    semantic: TurnDetectionSemantic


class TurnDetector(ABC):
    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        asr_final: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """
        Detect turn with audio and/or text.

        Args:
            audio (bytes): Audio data frame at this instant. PCM 16bit mono, 16000Hz bytes.
            text: Text of the turn, from ASR result.
            asr_final: Whether text is ASR final result, or still partial recognition.

            Either audio or (text, asr_final) will be present.

        Returns:
            TurnDetectionResult
        """
        pass

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        asr_final: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Async wrapper for detect()."""
        loop = asyncio.get_running_loop()
        func = partial(self.detect, audio=audio, text=text, asr_final=asr_final)
        result: TurnDetectionResult = await loop.run_in_executor(None, func)
        return result

    @abstractmethod
    def clone(self) -> "TurnDetector":
        pass
