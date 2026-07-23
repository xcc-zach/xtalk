import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Optional

from ..registry import model_type


class TurnDetectionAction(Enum):
    """Immediate action selected by a turn detector."""

    DO_NOTHING = 1
    STOP_SPEAKING = 2
    START_GENERATION = 3


class TurnDetectionSemantic(Enum):
    """Semantic state selected by a turn detector."""

    IDLE = "idle"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    SHOULD_BACKCHANNEL = "should_backchannel"


class TurnVADResult(Enum):
    """Optional VAD state produced by a turn detector."""

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


@model_type(aliases=["turn_detector"])
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
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Detect conversational turn state from audio and/or text context.

        Parameters
        ----------
        audio : bytes | None, optional
            Current PCM 16-bit mono audio frame at 16 kHz.
        text : str | None, optional
            ASR text for the current turn.
        assistant_text : str | None, optional
            Cumulative AI response text confirmed as played to the user.
            ``None`` means that this call carries no assistant response update.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech. This may be
            provided without ``audio``, ``text``, or ``assistant_text``.
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
        assistant_text: Optional[str] = None,
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
        assistant_text : str | None, optional
            Cumulative AI response text confirmed as played to the user.
            ``None`` means that this call carries no assistant response update.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech. This may be
            provided without ``audio``, ``text``, or ``assistant_text``.
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
            assistant_text=assistant_text,
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
