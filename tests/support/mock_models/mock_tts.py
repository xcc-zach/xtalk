"""TTS replacement producing audio whose length follows the text length."""

from __future__ import annotations

import zlib
from typing import Any, AsyncIterator

import numpy as np

from xtalk.models.registry import model
from xtalk.models.tts import TTS

SYNTHESIS_FAILURE_MARKER = "合成失败"


@model
class MockTTS(TTS):
    """Synthesize a sine tone whose duration grows with the text.

    Parameters
    ----------
    min_ms : float, optional
        Minimum audio duration per call.
    ms_per_char : float, optional
        Audio milliseconds per character.
    amplitude : float, optional
        Tone amplitude relative to full scale.
    """

    sample_rate = 24000

    def __init__(
        self,
        min_ms: float = 200.0,
        ms_per_char: float = 80.0,
        amplitude: float = 0.3,
    ) -> None:
        self._min_ms = float(min_ms)
        self._ms_per_char = float(ms_per_char)
        self._amplitude = float(amplitude)

    def duration_ms(self, text: str) -> float:
        """Return the audio duration produced for ``text``."""
        return max(self._min_ms, self._ms_per_char * len(text))

    @staticmethod
    def frequency_for(text: str) -> float:
        """Return the stable tone frequency used for ``text``."""
        return 300.0 + zlib.crc32(text.encode("utf-8")) % 3000

    def synthesize(self, text: str) -> bytes:
        if SYNTHESIS_FAILURE_MARKER in text:
            raise RuntimeError("mock synthesis failure")
        samples = int(round(self.sample_rate * self.duration_ms(text) / 1000.0))
        t = np.arange(samples) / float(self.sample_rate)
        tone = np.sin(2.0 * np.pi * self.frequency_for(text) * t)
        return (tone * self._amplitude * 32767.0).astype("<i2").tobytes()

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        return self.synthesize(text)

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        yield self.synthesize(text)

    def clone(self) -> "MockTTS":
        return MockTTS(
            min_ms=self._min_ms,
            ms_per_char=self._ms_per_char,
            amplitude=self._amplitude,
        )
