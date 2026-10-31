"""Energy based VAD replacement."""

from __future__ import annotations

import numpy as np

from xtalk.models.registry import model
from xtalk.models.vad import VAD


@model
class MockVAD(VAD):
    """Classify a frame as speech when its RMS exceeds ``threshold * 0.1``.

    Parameters
    ----------
    threshold : float, optional
        Probability-like threshold mirrored from the realtime session config.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self._threshold = float(threshold)

    def set_threshold(self, *, threshold: float) -> None:
        self._threshold = float(threshold)

    def is_speech(self, frame: bytes) -> bool:
        usable = len(frame) - len(frame) % 2
        if usable == 0:
            return False
        samples = np.frombuffer(frame[:usable], dtype="<i2").astype(np.float64)
        rms = float(np.sqrt(np.mean(samples * samples))) / 32768.0
        return rms > self._threshold * 0.1

    async def async_is_speech(self, frame: bytes) -> bool:
        return self.is_speech(frame)

    def detect_frames(self, frames: list[bytes]) -> list[bool]:
        return [self.is_speech(frame) for frame in frames]

    async def async_detect_frames(self, frames: list[bytes]) -> list[bool]:
        return self.detect_frames(frames)

    def reset(self) -> None:
        return None

    def clone(self) -> "MockVAD":
        return MockVAD(threshold=self._threshold)
