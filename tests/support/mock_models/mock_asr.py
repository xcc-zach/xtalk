"""ASR replacement that decodes the test speech contract."""

from __future__ import annotations

from xtalk.models.asr import ASR
from xtalk.models.registry import model

from .. import speech

RECOGNITION_FAILURE_ID = 7


@model
class MockASR(ASR):
    """Turn test speech back into its fixed sentence text.

    Partial results reveal one more character per ``ms_per_char`` of voiced
    audio accumulated in the current turn; the final result is the full text.

    Parameters
    ----------
    sample_rate : int, optional
        Sample rate of the audio handed over by the ASR manager.
    ms_per_char : float, optional
        Voiced milliseconds needed to reveal one more character.
    """

    def __init__(self, sample_rate: int = 16000, ms_per_char: float = 100.0) -> None:
        self._sample_rate = int(sample_rate)
        self._ms_per_char = float(ms_per_char)
        self._buffer = bytearray()

    def _decode(self, audio: bytes) -> tuple[int | None, str, float]:
        voiced = speech.voiced_ms(audio, self._sample_rate)
        sentence_id = speech.recognize(audio, self._sample_rate)
        text = "" if sentence_id is None else speech.text_for(sentence_id)
        return sentence_id, text, voiced

    def recognize(self, audio: bytes) -> str:
        return self._decode(audio)[1]

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        del chat_history
        self._buffer.extend(audio)
        sentence_id, text, voiced = self._decode(bytes(self._buffer))
        if is_final:
            if sentence_id == RECOGNITION_FAILURE_ID:
                raise RuntimeError("mock recognition failure")
            return text
        revealed = int(voiced // self._ms_per_char)
        return text[:revealed]

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        return self.recognize_stream(audio, is_final=is_final, chat_history=chat_history)

    async def async_recognize(self, audio: bytes) -> str:
        return self.recognize(audio)

    def reset(self) -> None:
        self._buffer.clear()

    def clone(self) -> "MockASR":
        return MockASR(sample_rate=self._sample_rate, ms_per_char=self._ms_per_char)
