"""Build exemplar-plus-current PCM layouts for MTD decoding."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transcript import DiarizationSegment, render_segments


@dataclass(frozen=True)
class MtdAudioLayout:
    """Fully assembled MTD request input."""

    pcm16: bytes
    decoder_prefix: str
    context_seconds: float
    slots: list[dict[str, object]]


def pcm16_bytes_to_float32(pcm16: bytes) -> np.ndarray:
    """Convert little-endian PCM16 bytes to mono float32 samples."""

    if not pcm16:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm16, dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(audio: np.ndarray) -> bytes:
    """Convert normalized float audio to little-endian PCM16 bytes."""

    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return np.rint(clipped * 32767.0).astype("<i2").tobytes()


def build_audio_layout(
    *,
    exemplars: list[object],
    current_pcm16: bytes,
    sample_rate: int,
    inter_exemplar_silence_s: float,
    exemplar_to_current_silence_s: float,
) -> MtdAudioLayout:
    """Assemble exemplar audio, configurable silence, and current PCM."""

    chunks: list[np.ndarray] = []
    prefix_segments: list[DiarizationSegment] = []
    slots: list[dict[str, object]] = []
    cursor_s = 0.0
    for index, item in enumerate(exemplars):
        audio = np.asarray(item.audio, dtype=np.float32)
        start_s = cursor_s
        end_s = start_s + len(audio) / sample_rate
        chunks.append(audio)
        prefix_segments.append(
            DiarizationSegment(
                start_s=start_s,
                end_s=end_s,
                speaker_id=item.speaker_id,
                text=item.text,
            )
        )
        slots.append(
            {
                "speaker_id": item.speaker_id,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
            }
        )
        cursor_s = end_s
        if index + 1 < len(exemplars) and inter_exemplar_silence_s > 0:
            chunks.append(
                np.zeros(round(inter_exemplar_silence_s * sample_rate), dtype=np.float32)
            )
            cursor_s += inter_exemplar_silence_s
    if exemplars and exemplar_to_current_silence_s > 0:
        chunks.append(
            np.zeros(
                round(exemplar_to_current_silence_s * sample_rate),
                dtype=np.float32,
            )
        )
        cursor_s += exemplar_to_current_silence_s
    chunks.append(pcm16_bytes_to_float32(current_pcm16))
    request_audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return MtdAudioLayout(
        pcm16=float32_to_pcm16_bytes(request_audio),
        decoder_prefix=render_segments(prefix_segments),
        context_seconds=cursor_s,
        slots=slots,
    )
