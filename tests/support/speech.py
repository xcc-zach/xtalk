"""Deterministic test speech shared by the backend and frontend test suites.

A sentence is identified by an integer ``id`` in ``0..63``. Its voiced part is
a single sine tone at ``400 + 60 * id`` Hz, so audio that survived the real
resampling and buffering pipeline can be decoded back into text without any
speech model. ``frontend/tests/support/speech.ts`` implements the same
contract for generation.
"""

from __future__ import annotations

import numpy as np

MIN_ID = 0
MAX_ID = 63
BASE_FREQUENCY_HZ = 400.0
FREQUENCY_STEP_HZ = 60.0
AMPLITUDE = 0.5
VOICED_RMS_THRESHOLD = 0.05
MAX_FREQUENCY_DEVIATION_HZ = 20.0
FRAME_MS = 20

_FIXED_TEXTS = {
    0: "你好",
    1: "请讲一个很长的故事",
    2: "帮我查一下天气",
    3: "现在报错",
    4: "你刚才说了什么",
    5: "慢慢说",
    6: "让合成失败",
    7: "识别失败",
}


def text_for(sentence_id: int) -> str:
    """Return the fixed text mapped to one sentence id.

    Parameters
    ----------
    sentence_id : int
        Sentence identifier in ``0..63``.

    Returns
    -------
    str
        Sentence text.
    """
    if not MIN_ID <= sentence_id <= MAX_ID:
        raise ValueError(f"sentence id must be in {MIN_ID}..{MAX_ID}")
    return _FIXED_TEXTS.get(sentence_id, f"第{sentence_id}句话")


def frequency_for(sentence_id: int) -> float:
    """Return the tone frequency encoding one sentence id."""
    if not MIN_ID <= sentence_id <= MAX_ID:
        raise ValueError(f"sentence id must be in {MIN_ID}..{MAX_ID}")
    return BASE_FREQUENCY_HZ + FREQUENCY_STEP_HZ * sentence_id


def encode(
    sentence_id: int,
    *,
    voiced_ms: float,
    trailing_silence_ms: float = 0.0,
    sample_rate: int,
) -> bytes:
    """Generate PCM16 mono test speech for one sentence.

    Parameters
    ----------
    sentence_id : int
        Sentence identifier.
    voiced_ms : float
        Duration of the sine tone.
    trailing_silence_ms : float, optional
        Zero samples appended after the tone.
    sample_rate : int
        Output sample rate.

    Returns
    -------
    bytes
        Little-endian PCM16 audio.
    """
    frequency = frequency_for(sentence_id)
    voiced_samples = int(round(sample_rate * voiced_ms / 1000.0))
    t = np.arange(voiced_samples) / float(sample_rate)
    tone = np.sin(2.0 * np.pi * frequency * t) * AMPLITUDE * 32767.0
    pcm = tone.astype("<i2").tobytes()
    return pcm + silence(trailing_silence_ms, sample_rate=sample_rate)


def silence(ms: float, *, sample_rate: int) -> bytes:
    """Return ``ms`` milliseconds of PCM16 silence."""
    return bytes(int(round(sample_rate * ms / 1000.0)) * 2)


def _samples(pcm: bytes) -> np.ndarray:
    usable = len(pcm) - len(pcm) % 2
    return np.frombuffer(pcm[:usable], dtype="<i2").astype(np.float64) / 32768.0


def is_voiced(frame: bytes) -> bool:
    """Return whether a PCM16 frame carries speech energy."""
    samples = _samples(frame)
    if samples.size == 0:
        return False
    return float(np.sqrt(np.mean(samples * samples))) > VOICED_RMS_THRESHOLD


def _voiced_frames(pcm: bytes, sample_rate: int) -> list[np.ndarray]:
    samples = _samples(pcm)
    frame_len = max(1, int(sample_rate * FRAME_MS / 1000))
    frames = []
    for start in range(0, samples.size - frame_len + 1, frame_len):
        frame = samples[start : start + frame_len]
        if float(np.sqrt(np.mean(frame * frame))) > VOICED_RMS_THRESHOLD:
            frames.append(frame)
    return frames


def voiced_ms(pcm: bytes, sample_rate: int) -> float:
    """Return the total voiced duration of ``pcm`` in milliseconds."""
    return len(_voiced_frames(pcm, sample_rate)) * float(FRAME_MS)


def _frame_frequency(frame: np.ndarray, sample_rate: int) -> float | None:
    signs = np.sign(frame)
    signs[signs == 0] = 1
    crossings = np.nonzero(signs[1:] != signs[:-1])[0]
    if crossings.size < 2:
        return None
    # Interpolate each crossing position for sub-sample precision.
    left = frame[crossings]
    right = frame[crossings + 1]
    positions = crossings + left / (left - right)
    span = positions[-1] - positions[0]
    if span <= 0:
        return None
    return (crossings.size - 1) * sample_rate / (2.0 * span)


def estimate_frequency(pcm: bytes, sample_rate: int) -> float | None:
    """Estimate the dominant tone frequency of the voiced part of ``pcm``."""
    estimates = [
        value
        for value in (
            _frame_frequency(frame, sample_rate)
            for frame in _voiced_frames(pcm, sample_rate)
        )
        if value is not None
    ]
    if not estimates:
        return None
    return float(np.median(estimates))


def recognize(pcm: bytes, sample_rate: int) -> int | None:
    """Decode the sentence id carried by ``pcm``.

    Returns
    -------
    int | None
        Sentence id, or ``None`` when there is no voiced audio or the tone
        does not match any sentence closely enough.
    """
    frequency = estimate_frequency(pcm, sample_rate)
    if frequency is None:
        return None
    sentence_id = int(round((frequency - BASE_FREQUENCY_HZ) / FREQUENCY_STEP_HZ))
    if not MIN_ID <= sentence_id <= MAX_ID:
        return None
    if abs(frequency - frequency_for(sentence_id)) > MAX_FREQUENCY_DEVIATION_HZ:
        return None
    return sentence_id
