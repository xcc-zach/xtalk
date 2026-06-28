#!/usr/bin/env python3
"""Automated backend testing and test-set generation for Xtalk.

Example commands:
    python scripts/test.py --create logs/test_templates/smoke --out logs/tests
    python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke
    python scripts/test.py --config server_configs/sample_local.json --input logs/tests/smoke --out logs/test_results/smoke --concurrency 2 --with-vad
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import multiprocessing
import re
import shutil
import socket
import subprocess
import sys
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import requests
import soundfile as sf
import uvicorn
import websockets
import yaml
from fastapi import FastAPI

from xtalk.api import Xtalk
from xtalk.model_loader import init_registered_model

try:
    import soxr
except Exception:  # pragma: no cover - optional dependency
    soxr = None


DEFAULT_TEST_CONFIG = {
    "concurrency": 1,
    "with_vad": False,
    "vad_redemption_ms": 500,
    "judge_llm": None,
}
DEFAULT_SETTLE_SECONDS = 1.5
RECORDING_STABLE_SECONDS = 1.0
RECORDING_STABLE_TIMEOUT_SECONDS = 10.0
EMBEDDED_SERVER_HOST = "127.0.0.1"
EMBEDDED_SERVER_PORT = 0
SERVICE_CONFIG_PATCH = {
    "recording": True,
    "send_full_audio_to_client": False,
    "enable_persistence": False,
}
NON_ACTIVITY_ACTIONS = {"thought_updated"}
PREFERRED_TEST_CONFIG_NAMES = (
    "test_config.json",
    "testing_config.json",
    "config.json",
)
PREFERRED_TTS_CONFIG_NAMES = (
    "tts_config.json",
    "config.json",
    "sample_local.json",
)


@dataclass(frozen=True)
class RelativeTimeSpec:
    """Represents a scheduled timestamp expression.

    Relative anchors resolve against the first matching runtime event after the
    previous scheduled input. User anchors bind to the previous user clip, while
    AI anchors bind to the response for the previous user clip.
    """

    kind: Literal["absolute", "relative"]
    value: float | None = None
    anchor: str | None = None
    offset: float = 0.0


@dataclass(frozen=True)
class ScheduledAudioInput:
    """Represents one scheduled input audio clip."""

    time_spec: RelativeTimeSpec
    audio_path: Path
    expected_text: str | None = None


@dataclass(frozen=True)
class GeneratedCaseLine:
    """Represents one scheduled text line for test-case generation."""

    time_spec: str
    text: str


@dataclass(frozen=True)
class EffectiveTestConfig:
    """Effective dataset-level runtime configuration."""

    concurrency: int
    with_vad: bool
    vad_redemption_ms: int
    judge_llm: "JudgeLLMConfig | None"


@dataclass(frozen=True)
class JudgeLLMConfig:
    """Configuration for the LLM used to judge ASR outputs."""

    model: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class CaseCriteria:
    """Optional evaluation criteria loaded from ``criteria.yaml``."""

    judge_asr: bool = False


@dataclass(frozen=True)
class ASRResultRecord:
    """One ASR event received from the backend during a test case."""

    action: str
    text: str
    received_at: float


@dataclass(frozen=True)
class CaseExecutionResult:
    """Execution artifacts and status for one test case."""

    case_name: str
    output_path: Path
    latency_samples_ms: list[float]
    criteria: CaseCriteria
    scheduled_inputs: list[ScheduledAudioInput]
    asr_results: list[ASRResultRecord]
    error: str | None = None


@dataclass(frozen=True)
class ASRJudgement:
    """Result of judging ASR outputs against expected transcripts."""

    passed: bool
    reason: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run automated Xtalk backend tests or generate test datasets."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input",
        type=Path,
        help="Path to the input test dataset folder.",
    )
    mode.add_argument(
        "--create",
        type=Path,
        help="Path to the test-case generation source folder.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the backend service configuration JSON.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Path to the output folder.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        help="Override the dataset concurrency in test mode.",
    )
    parser.add_argument(
        "--vad-redemption-ms",
        type=int,
        help="Override the client-side VAD redemption time in test mode.",
    )
    vad_group = parser.add_mutually_exclusive_group()
    vad_group.add_argument(
        "--with-vad",
        dest="with_vad_override",
        action="store_true",
        help="Force-enable client-side VAD in test mode.",
    )
    vad_group.add_argument(
        "--without-vad",
        dest="with_vad_override",
        action="store_false",
        help="Force-disable client-side VAD in test mode.",
    )
    parser.set_defaults(with_vad_override=None)
    parser.add_argument(
        "--judge-llm-model",
        type=str,
        help="Override judge_llm.model in test mode.",
    )
    parser.add_argument(
        "--judge-llm-base-url",
        type=str,
        help="Override judge_llm.base_url in test mode.",
    )
    parser.add_argument(
        "--judge-llm-api-key",
        type=str,
        help="Override judge_llm.api_key in test mode.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file into a dictionary."""
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def get_case_asr_report_path(output_root: Path, case_name: str) -> Path:
    """Return the ASR report path for one case."""
    return output_root / "logs" / f"{case_name}.asr.json"


def get_legacy_case_asr_report_path(output_root: Path, case_name: str) -> Path:
    """Return the legacy ASR report path kept for cleanup compatibility."""
    return output_root / f"{case_name}.asr.json"


def resolve_root_json(
    root: Path,
    preferred_names: tuple[str, ...],
) -> Path | None:
    """Resolve a single JSON file from a dataset root."""
    candidates = sorted(path for path in root.glob("*.json") if path.is_file())
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    by_name = {path.name: path for path in candidates}
    for name in preferred_names:
        if name in by_name:
            return by_name[name]

    names = ", ".join(path.name for path in candidates)
    raise ValueError(
        f"Found multiple JSON files under {root} and could not choose one: {names}"
    )


def discover_case_dirs(root: Path) -> list[Path]:
    """Discover direct child case directories."""
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not case_dirs:
        raise ValueError(f"No case directories found under {root}")
    return case_dirs


def merge_service_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    """Inject required testing service configuration overrides."""
    merged = dict(raw_config)
    service_config = dict(merged.get("service_config") or {})
    service_config.update(SERVICE_CONFIG_PATCH)
    merged["service_config"] = service_config
    return merged


def load_effective_test_config(
    dataset_root: Path,
    *,
    concurrency_override: int | None,
    with_vad_override: bool | None,
    vad_redemption_ms_override: int | None,
    judge_llm_model_override: str | None,
    judge_llm_base_url_override: str | None,
    judge_llm_api_key_override: str | None,
) -> EffectiveTestConfig:
    """Load and resolve the dataset test configuration."""
    config_path = resolve_root_json(
        dataset_root,
        preferred_names=PREFERRED_TEST_CONFIG_NAMES,
    )
    raw_config = dict(DEFAULT_TEST_CONFIG)
    if config_path is not None:
        raw_config.update(load_json(config_path))

    if concurrency_override is not None:
        raw_config["concurrency"] = concurrency_override
    if with_vad_override is not None:
        raw_config["with_vad"] = with_vad_override
    if vad_redemption_ms_override is not None:
        raw_config["vad_redemption_ms"] = vad_redemption_ms_override

    raw_judge_llm = raw_config.get("judge_llm")
    if raw_judge_llm is None:
        judge_llm_config: dict[str, Any] = {}
    elif isinstance(raw_judge_llm, dict):
        judge_llm_config = dict(raw_judge_llm)
    else:
        raise ValueError("judge_llm must be a JSON object when provided")

    if judge_llm_model_override is not None:
        judge_llm_config["model"] = judge_llm_model_override
    if judge_llm_base_url_override is not None:
        judge_llm_config["base_url"] = judge_llm_base_url_override
    if judge_llm_api_key_override is not None:
        judge_llm_config["api_key"] = judge_llm_api_key_override

    concurrency = int(raw_config.get("concurrency", 1))
    if concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")
    with_vad = bool(raw_config.get("with_vad", False))
    vad_redemption_ms = int(
        raw_config.get(
            "vad_redemption_ms",
            DEFAULT_TEST_CONFIG["vad_redemption_ms"],
        )
    )
    if vad_redemption_ms <= 0:
        raise ValueError("vad_redemption_ms must be a positive integer")

    judge_llm: JudgeLLMConfig | None = None
    if judge_llm_config:
        model = judge_llm_config.get("model")
        base_url = judge_llm_config.get("base_url")
        api_key = judge_llm_config.get("api_key")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("judge_llm.model must be a non-empty string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("judge_llm.base_url must be a non-empty string")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("judge_llm.api_key must be a non-empty string")
        judge_llm = JudgeLLMConfig(
            model=model.strip(),
            base_url=base_url.strip(),
            api_key=api_key.strip(),
        )

    return EffectiveTestConfig(
        concurrency=concurrency,
        with_vad=with_vad,
        vad_redemption_ms=vad_redemption_ms,
        judge_llm=judge_llm,
    )


def parse_timestamp_spec(raw_spec: str) -> RelativeTimeSpec:
    """Parse a timestamp expression from timestamp.txt."""
    spec = raw_spec.strip()
    if not spec:
        raise ValueError("Empty timestamp expression")

    try:
        return RelativeTimeSpec(kind="absolute", value=float(spec))
    except ValueError:
        pass

    if "+" in spec:
        anchor_name, offset_text = spec.split("+", 1)
        offset = float(offset_text.strip())
    else:
        anchor_name, offset = spec, 0.0

    anchor = normalize_anchor_name(anchor_name.strip())
    return RelativeTimeSpec(kind="relative", anchor=anchor, offset=offset)


def normalize_anchor_name(anchor: str) -> str:
    """Normalize timestamp anchor names across supported aliases."""
    aliases = {
        "ai_start": "last_ai_start",
        "last_ai_start": "last_ai_start",
        "ai_end": "last_ai_end",
        "last_ai_end": "last_ai_end",
        "user_start": "last_user_start",
        "last_user_start": "last_user_start",
        "user_end": "last_user_end",
        "last_user_end": "last_user_end",
    }
    if anchor not in aliases:
        raise ValueError(f"Unsupported timestamp anchor: {anchor}")
    return aliases[anchor]


def parse_case_inputs(case_dir: Path) -> list[ScheduledAudioInput]:
    """Parse timestamp.txt and resolve audio files for one case."""
    timestamp_path = case_dir / "timestamp.txt"
    if not timestamp_path.exists():
        raise ValueError(f"Missing timestamp.txt in {case_dir}")

    scheduled_inputs: list[ScheduledAudioInput] = []
    with timestamp_path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid timestamp entry in {timestamp_path}:{line_no}: {line}"
                )
            time_text, remainder = line.split(":", 1)
            audio_name_text = remainder.strip()
            expected_text: str | None = None
            if ":" in audio_name_text:
                audio_name, expected_text = audio_name_text.split(":", 1)
                audio_name = audio_name.strip()
                expected_text = expected_text.strip() or None
            else:
                audio_name = audio_name_text
            audio_path = case_dir / audio_name.strip()
            if not audio_path.exists():
                raise ValueError(f"Missing audio file {audio_path}")
            time_spec = parse_timestamp_spec(time_text)
            scheduled_inputs.append(
                ScheduledAudioInput(
                    time_spec=time_spec,
                    audio_path=audio_path,
                    expected_text=expected_text,
                )
            )

    if not scheduled_inputs:
        raise ValueError(f"No valid timestamp entries found in {timestamp_path}")
    return scheduled_inputs


def parse_generation_case(case_dir: Path) -> list[GeneratedCaseLine]:
    """Parse timestamp.txt for test-case generation."""
    timestamp_path = case_dir / "timestamp.txt"
    if not timestamp_path.exists():
        raise ValueError(f"Missing timestamp.txt in {case_dir}")

    lines: list[GeneratedCaseLine] = []
    with timestamp_path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(
                    f"Invalid timestamp entry in {timestamp_path}:{line_no}: {line}"
                )
            time_text, text = line.split(":", 1)
            lines.append(
                GeneratedCaseLine(time_spec=time_text.strip(), text=text.strip())
            )

    if not lines:
        raise ValueError(f"No valid generation entries found in {timestamp_path}")
    return lines


def load_case_criteria(case_dir: Path) -> CaseCriteria:
    """Load optional per-case evaluation criteria from ``criteria.yaml``."""
    criteria_path = case_dir / "criteria.yaml"
    if not criteria_path.exists():
        return CaseCriteria()

    with criteria_path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {criteria_path}")
    return CaseCriteria(judge_asr=bool(payload.get("judge_asr", False)))


def pick_free_port(host: str) -> int:
    """Pick a free TCP port for the embedded server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample mono audio to the target sample rate."""
    if source_sr == target_sr:
        return audio.astype(np.float32, copy=False)

    mono_audio = audio.astype(np.float32, copy=False)
    if soxr is not None:
        return soxr.resample(mono_audio, source_sr, target_sr).astype(np.float32)

    duration = mono_audio.shape[0] / float(source_sr)
    target_size = max(1, int(round(duration * target_sr)))
    if mono_audio.shape[0] == 1:
        return np.full((target_size,), float(mono_audio[0]), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=mono_audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_size, endpoint=False)
    return np.interp(x_new, x_old, mono_audio).astype(np.float32)


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert floating point audio in [-1, 1] to PCM16 bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    return pcm.tobytes()


def load_audio_as_pcm16(path: Path, *, target_sr: int = 16000) -> bytes:
    """Load an audio file as mono PCM16 bytes at the requested sample rate."""
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        resampled = resample_audio(audio, int(sample_rate), target_sr)
        return float_to_pcm16(resampled)
    except Exception:
        return decode_audio_via_ffmpeg(path, target_sr=target_sr)


def decode_audio_via_ffmpeg(path: Path, *, target_sr: int = 16000) -> bytes:
    """Use ffmpeg as a fallback decoder for formats unsupported by soundfile."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(target_sr),
        "-",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to decode {path} with ffmpeg: {stderr}")
    return bytes(result.stdout)


def decode_tts_audio_payload(
    payload: bytes,
    *,
    sample_rate_hint: int = 48000,
) -> tuple[bytes, int]:
    """Normalize a TTS payload into WAV-ready PCM16 bytes."""
    try:
        audio, sample_rate = sf.read(
            io.BytesIO(payload), dtype="float32", always_2d=False
        )
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if not isinstance(audio, np.ndarray):
            audio = np.asarray(audio, dtype=np.float32)
        return float_to_pcm16(np.asarray(audio, dtype=np.float32)), int(sample_rate)
    except Exception:
        return payload, sample_rate_hint


def write_pcm_wav(
    path: Path,
    *,
    pcm_bytes: bytes,
    sample_rate: int,
    channels: int,
) -> None:
    """Write PCM bytes into a WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)


def count_wav_frames(path: Path) -> int:
    """Return the number of frames in a WAV file."""
    with wave.open(str(path), "rb") as wav_file:
        return int(wav_file.getnframes())


def stereo_pcm_has_right_channel_signal(pcm_bytes: bytes) -> bool:
    """Return whether stereo PCM16 data contains non-zero right-channel samples."""
    if not pcm_bytes:
        return False
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size < 2:
        return False
    stereo = samples.reshape(-1, 2)
    return bool(np.any(stereo[:, 1]))


def read_audio_int16(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file as an ``int16`` NumPy array plus sample rate."""
    audio, sample_rate = sf.read(path, dtype="int16", always_2d=True)
    return np.asarray(audio, dtype=np.int16), int(sample_rate)


def downmix_stereo_to_mono(audio: np.ndarray) -> np.ndarray:
    """Downmix stereo PCM16 audio to mono while preserving headroom."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        raise ValueError("Expected stereo audio with at least two channels")
    stereo = audio[:, :2].astype(np.float32)
    mono = np.mean(stereo, axis=1)
    mono = np.clip(np.rint(mono), -32768, 32767)
    return mono.astype(np.int16)


def stereo_audio_has_right_channel_signal(audio: np.ndarray) -> bool:
    """Return whether stereo PCM16 audio contains non-zero right-channel samples."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return False
    return bool(np.any(audio[:, 1]))


def write_mono_mp3(path: Path, *, audio: np.ndarray, sample_rate: int) -> None:
    """Write mono PCM16 audio as a high-quality MP3 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-v",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "-",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "0",
        str(path),
    ]
    result = subprocess.run(
        command,
        input=np.asarray(audio, dtype=np.int16).tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Failed to write MP3 {path}: {stderr}")


def extract_active_segments(
    samples: np.ndarray,
    sample_rate: int,
    *,
    bridge_gap_seconds: float = 1.0,
    min_segment_seconds: float = 0.08,
) -> list[tuple[int, int]]:
    """Extract contiguous active audio regions from one mono PCM channel.

    Parameters
    ----------
    samples : np.ndarray
        Mono PCM16 samples.
    sample_rate : int
        Sample rate in Hz.

    Returns
    -------
    list[tuple[int, int]]
        Active segments as ``(start_sample, end_sample)`` pairs.
    """
    if samples.size == 0:
        return []
    max_abs = int(np.max(np.abs(samples)))
    if max_abs <= 0:
        return []

    threshold = max(300, int(max_abs * 0.02))
    mask = np.abs(samples) >= threshold
    bridge_gap = max(1, int(round(sample_rate * bridge_gap_seconds)))
    min_segment = max(1, int(round(sample_rate * min_segment_seconds)))

    start = 0
    while start < mask.size:
        if mask[start]:
            start += 1
            continue
        end = start
        while end < mask.size and not mask[end]:
            end += 1
        if start > 0 and end < mask.size and end - start <= bridge_gap:
            mask[start:end] = True
        start = end

    segments: list[tuple[int, int]] = []
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < mask.size and mask[idx]:
            idx += 1
        if idx - start >= min_segment:
            segments.append((start, idx))
    return segments


def compute_case_latency_samples(output_path: Path) -> list[float]:
    """Compute user-to-AI latency samples for one stereo analysis recording.

    Parameters
    ----------
    output_path : Path
        Stereo audio path whose left/right channels are user/AI audio.

    Returns
    -------
    list[float]
        One latency value in milliseconds for each matched user/AI pair.
    """
    audio, sample_rate = read_audio_int16(output_path)
    return compute_case_latency_samples_from_array(audio, sample_rate)


def compute_case_latency_samples_from_array(
    audio: np.ndarray, sample_rate: int
) -> list[float]:
    """Compute user-to-AI latency samples from stereo PCM16 audio."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return []

    user_segments = extract_active_segments(
        audio[:, 0],
        int(sample_rate),
        bridge_gap_seconds=1.0,
    )
    ai_segments = extract_active_segments(
        audio[:, 1],
        int(sample_rate),
        bridge_gap_seconds=1.0,
    )
    if not user_segments or not ai_segments:
        return []

    latencies_ms: list[float] = []
    ai_index = 0
    for user_index, (_, user_end) in enumerate(user_segments):
        next_user_start = (
            user_segments[user_index + 1][0]
            if user_index + 1 < len(user_segments)
            else None
        )
        while ai_index < len(ai_segments) and ai_segments[ai_index][0] < user_end:
            ai_index += 1
        if ai_index >= len(ai_segments):
            break

        ai_start, _ = ai_segments[ai_index]
        if next_user_start is not None and ai_start >= next_user_start:
            continue

        latencies_ms.append((ai_start - user_end) * 1000.0 / float(sample_rate))
        ai_index += 1
    return latencies_ms


def resolve_chat_completions_url(base_url: str) -> str:
    """Resolve an OpenAI-compatible chat completions endpoint from a base URL."""
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def extract_llm_text(response_payload: dict[str, Any]) -> str:
    """Extract the first textual assistant message from a chat completions response."""
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Judge LLM response did not include choices")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Judge LLM response choice has invalid shape")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Judge LLM response choice did not include a message")
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())
        if text_parts:
            return "\n".join(text_parts)
    raise RuntimeError("Judge LLM response did not include textual content")


def parse_judgement_json(raw_text: str) -> ASRJudgement:
    """Parse the judge model response into a structured ASR judgement."""
    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    candidate = match.group(0) if match else raw_text
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise RuntimeError("Judge LLM response is not a JSON object")
    passed = bool(payload.get("passed", False))
    reason = payload.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No explanation provided by judge LLM."
    return ASRJudgement(passed=passed, reason=reason.strip())


def judge_asr_with_llm(
    judge_llm: JudgeLLMConfig,
    *,
    expected_texts: list[str],
    actual_texts: list[str],
) -> ASRJudgement:
    """Judge whether ASR outputs preserve the expected utterance semantics.

    Parameters
    ----------
    judge_llm : JudgeLLMConfig
        Judge model configuration.
    expected_texts : list[str]
        Expected user utterances from ``timestamp.txt``.
    actual_texts : list[str]
        Final ASR texts reported by the backend.

    Returns
    -------
    ASRJudgement
        Structured judgement result returned by the LLM.
    """
    expected_lines = "\n".join(
        f"{index}. {text}" for index, text in enumerate(expected_texts, start=1)
    )
    actual_lines = "\n".join(
        f"{index}. {text}" for index, text in enumerate(actual_texts, start=1)
    )
    prompt = (
        "Compare the expected user utterances with the backend ASR final outputs. "
        "Allow minor punctuation, filler-word, or paraphrase differences when the meaning stays the same. "
        "Fail if meaning changes, important information is missing, order changes, or the counts differ. "
        'Return strict JSON like {"passed": true, "reason": "..."}.\n\n'
        f"Expected utterances:\n{expected_lines or '(none)'}\n\n"
        f"ASR final outputs:\n{actual_lines or '(none)'}"
    )
    payload = {
        "model": judge_llm.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict but semantics-aware ASR judge.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }
    response = requests.post(
        resolve_chat_completions_url(judge_llm.base_url),
        headers={
            "Authorization": f"Bearer {judge_llm.api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return parse_judgement_json(extract_llm_text(response.json()))


class EmbeddedServer:
    """Run an embedded uvicorn server for automated tests."""

    def __init__(self, *, config: dict[str, Any], host: str, port: int) -> None:
        self._config = config
        self._host = host
        self._port = port if port > 0 else pick_free_port(host)
        self._process: multiprocessing.Process | None = None

    @property
    def http_base_url(self) -> str:
        """Return the HTTP base URL."""
        return f"http://{self._host}:{self._port}"

    @property
    def websocket_url(self) -> str:
        """Return the WebSocket endpoint URL."""
        return f"ws://{self._host}:{self._port}/ws"

    async def __aenter__(self) -> "EmbeddedServer":
        """Start the embedded server."""
        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_run_embedded_server_process,
            args=(self._config, self._host, self._port),
            name="xtalk-test-server",
        )
        self._process.start()

        deadline = time.monotonic() + 15.0
        while True:
            if self._is_port_open():
                return self
            if self._process.exitcode is not None:
                raise RuntimeError("Embedded server exited before startup completed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for embedded server startup")
            await asyncio.sleep(0.05)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Stop the embedded server."""
        if self._process is None:
            return
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=5.0)
        self._process.close()
        self._process = None

    def _is_port_open(self) -> bool:
        """Return whether the embedded server port is accepting TCP connections."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((self._host, self._port)) == 0


def _run_embedded_server_process(
    config: dict[str, Any],
    host: str,
    port: int,
) -> None:
    """Run the embedded FastAPI/Uvicorn server inside a dedicated process."""
    app = FastAPI(title="Xtalk Automated Test Server")
    xtalk_instance = Xtalk.from_config(config)
    xtalk_instance.mount_routes(app)
    uvicorn.run(app, host=host, port=port, log_level="error", lifespan="off")


class AnchorClock:
    """Tracks absolute monotonic timestamps for schedule anchors."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._values: dict[str, float] = {}
        self._history: dict[str, list[float]] = {}

    async def set(self, name: str, value: float) -> None:
        """Set an anchor and notify waiting tasks."""
        async with self._condition:
            self._values[name] = value
            self._history.setdefault(name, []).append(value)
            self._condition.notify_all()

    async def wait_for(self, name: str) -> float:
        """Wait until an anchor becomes available."""
        async with self._condition:
            await self._condition.wait_for(lambda: name in self._values)
            return self._values[name]

    async def wait_for_occurrence(self, name: str, occurrence: int) -> float:
        """Wait until the requested anchor occurrence becomes available."""
        if occurrence <= 0:
            raise ValueError("occurrence must be a positive integer")
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self._history.get(name, [])) >= occurrence
            )
            return self._history[name][occurrence - 1]

    async def get_occurrence(self, name: str, occurrence: int) -> float | None:
        """Return one anchor occurrence if it is already available."""
        if occurrence <= 0:
            raise ValueError("occurrence must be a positive integer")
        async with self._condition:
            history = self._history.get(name, [])
            if len(history) < occurrence:
                return None
            return history[occurrence - 1]

    async def get(self, name: str) -> float | None:
        """Return an anchor value if it already exists."""
        async with self._condition:
            return self._values.get(name)

    async def get_first_at_or_after(
        self, name: str, minimum_value: float
    ) -> float | None:
        """Return the first anchor occurrence at or after ``minimum_value``."""
        async with self._condition:
            for value in self._history.get(name, []):
                if value >= minimum_value:
                    return value
            return None


class ClientVADController:
    """Client-side VAD state machine aligned with frontend/backend VAD timing."""

    SAMPLE_RATE = 16000
    FRAME_SAMPLES = 512
    FRAME_BYTES = FRAME_SAMPLES * 2
    MIN_SPEECH_MS = 250
    DEFAULT_REDEMPTION_MS = 500
    _SHARED_VAD: Any = None

    def __init__(self, *, redemption_ms: int = DEFAULT_REDEMPTION_MS) -> None:
        from xtalk.speech.vad.silero_vad import SileroVAD

        if self.__class__._SHARED_VAD is None:
            self.__class__._SHARED_VAD = SileroVAD()
        self._vad = self.__class__._SHARED_VAD
        self._redemption_ms = redemption_ms
        self._speech_run_frames = 0
        self._non_speech_run_frames = 0
        self._in_speech = False
        self._min_speech_frames = max(
            1,
            int(
                round(
                    self.MIN_SPEECH_MS
                    / ((self.FRAME_SAMPLES * 1000.0) / self.SAMPLE_RATE)
                )
            ),
        )
        self._redemption_frames = max(
            1,
            int(
                round(
                    self._redemption_ms
                    / ((self.FRAME_SAMPLES * 1000.0) / self.SAMPLE_RATE)
                )
            ),
        )

    def feed(self, frame: bytes) -> list[str]:
        """Feed one PCM frame and return zero or more VAD events."""
        if len(frame) < self.FRAME_BYTES:
            frame = frame + (b"\x00" * (self.FRAME_BYTES - len(frame)))

        is_speech = bool(self._vad.is_speech(frame))
        events: list[str] = []
        if is_speech:
            self._speech_run_frames += 1
            self._non_speech_run_frames = 0
            if (
                not self._in_speech
                and self._speech_run_frames >= self._min_speech_frames
            ):
                self._in_speech = True
                events.append("vad_speech_start")
        else:
            self._non_speech_run_frames += 1
            self._speech_run_frames = 0
            if (
                self._in_speech
                and self._non_speech_run_frames >= self._redemption_frames
            ):
                self._in_speech = False
                events.append("vad_speech_end")
        return events

    def trailing_silence_frames(self) -> int:
        """Return the number of silence frames needed to trigger speech end."""
        return self._redemption_frames + 1


class PlaybackSimulator:
    """Simulates frontend output playback behavior over a WebSocket session."""

    def __init__(
        self,
        *,
        websocket: websockets.WebSocketClientProtocol,
        anchors: AnchorClock,
        activity_callback,
    ) -> None:
        self._websocket = websocket
        self._anchors = anchors
        self._activity_callback = activity_callback
        self._queue: deque[bytes] = deque()
        self._condition = asyncio.Condition()
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._closed = False
        self._server_tts_finished = False
        self._currently_playing = False
        self._turn_started = False
        self._stop_generation = 0
        self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        try:
            while True:
                async with self._condition:
                    await self._condition.wait_for(
                        lambda: self._closed or bool(self._queue)
                    )
                    if self._closed:
                        return
                    chunk = self._queue.popleft()
                    generation = self._stop_generation
                    self._currently_playing = True

                if not self._turn_started:
                    self._turn_started = True
                    await self._anchors.set(
                        "last_ai_start", asyncio.get_running_loop().time()
                    )

                await self._activity_callback()
                duration = self._chunk_duration_seconds(chunk)
                interrupted = await self._play_chunk(duration, generation)
                async with self._condition:
                    self._currently_playing = False

                if interrupted:
                    continue

                if not await self._send_json({"action": "tts_chunk_played"}):
                    return
                await self._activity_callback()
                await self._maybe_finish_turn()
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed:
            return

    async def _play_chunk(self, duration: float, generation: int) -> bool:
        remaining = duration
        while remaining > 0.0:
            await self._resume_event.wait()
            if self._stop_generation != generation:
                return True
            start = asyncio.get_running_loop().time()
            step = min(0.05, remaining)
            await asyncio.sleep(step)
            if self._stop_generation != generation:
                return True
            if not self._resume_event.is_set():
                continue
            elapsed = asyncio.get_running_loop().time() - start
            remaining = max(0.0, remaining - elapsed)
        return False

    @staticmethod
    def _chunk_duration_seconds(chunk: bytes) -> float:
        samples = len(chunk) // 2
        return samples / 48000.0

    async def push(self, chunk: bytes) -> None:
        """Queue one TTS chunk for simulated playback."""
        async with self._condition:
            self._queue.append(chunk)
            self._condition.notify_all()
        await self._activity_callback()

    async def pause(self) -> None:
        """Pause simulated playback."""
        self._resume_event.clear()
        await self._activity_callback()

    async def resume(self) -> None:
        """Resume simulated playback."""
        self._resume_event.set()
        await self._activity_callback()

    async def stop(self) -> None:
        """Stop playback and clear pending chunks."""
        async with self._condition:
            self._queue.clear()
            self._stop_generation += 1
            self._server_tts_finished = False
            self._currently_playing = False
            self._turn_started = False
            self._condition.notify_all()
        self._resume_event.set()
        await self._activity_callback()

    async def mark_server_tts_finished(self) -> None:
        """Mark that the server finished generating TTS for the current turn."""
        async with self._condition:
            self._server_tts_finished = True
        await self._maybe_finish_turn()

    async def is_idle(self) -> bool:
        """Return whether playback is fully idle."""
        async with self._condition:
            return not self._currently_playing and not self._queue

    async def _maybe_finish_turn(self) -> None:
        async with self._condition:
            if (
                not self._server_tts_finished
                or self._currently_playing
                or self._queue
                or not self._turn_started
            ):
                return
            self._server_tts_finished = False
            self._turn_started = False
        now = asyncio.get_running_loop().time()
        await self._anchors.set("last_ai_end", now)
        if not await self._send_json({"action": "tts_playback_finished"}):
            return
        await self._activity_callback()

    async def _send_json(self, payload: dict[str, Any]) -> bool:
        """Send a JSON payload unless the WebSocket is already closed."""
        try:
            await self._websocket.send(json.dumps(payload))
        except websockets.ConnectionClosed:
            return False
        return True

    async def close(self) -> None:
        """Stop the simulator worker."""
        await self.stop()
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass


class CaseRunner:
    """Runs one automated test case against the embedded server."""

    def __init__(
        self,
        *,
        case_dir: Path,
        scheduled_inputs: list[ScheduledAudioInput],
        output_path: Path,
        temp_recording_path: Path,
        websocket_url: str,
        http_base_url: str,
        with_vad: bool,
        vad_redemption_ms: int,
        settle_seconds: float,
    ) -> None:
        self._case_dir = case_dir
        self._scheduled_inputs = scheduled_inputs
        self._output_path = output_path
        self._temp_recording_path = temp_recording_path
        self._websocket_url = websocket_url
        self._http_base_url = http_base_url
        self._with_vad = with_vad
        self._vad_redemption_ms = vad_redemption_ms
        self._settle_seconds = settle_seconds
        self._anchors = AnchorClock()
        self._connection_started: float | None = None
        self._attached_event = asyncio.Event()
        self._scheduler_done = asyncio.Event()
        self._ws_closed = asyncio.Event()
        self._activity_lock = asyncio.Lock()
        self._last_activity: float = 0.0
        self._full_audio_bytes = bytearray()
        self._receiver_task: asyncio.Task[None] | None = None
        self._playback: PlaybackSimulator | None = None
        self._error: Exception | None = None
        self._last_full_audio_at: float | None = None
        self._asr_results: list[ASRResultRecord] = []
        self._latency_samples_ms: list[float] = []

    @property
    def asr_results(self) -> list[ASRResultRecord]:
        """Return ASR events collected during the case run."""
        return list(self._asr_results)

    @property
    def latency_samples_ms(self) -> list[float]:
        """Return latency samples computed from the stereo analysis recording."""
        return list(self._latency_samples_ms)

    async def run(self) -> None:
        """Execute the case end to end and write the final compressed output."""
        token = await self._login()
        websocket = await websockets.connect(
            self._build_authenticated_ws_url(token),
            max_size=None,
        )

        try:
            self._playback = PlaybackSimulator(
                websocket=websocket,
                anchors=self._anchors,
                activity_callback=self._touch_activity,
            )
            self._receiver_task = asyncio.create_task(self._receiver_loop(websocket))

            await websocket.send(
                json.dumps({"action": "attach_session", "session_id": None})
            )
            await self._attached_event.wait()
            self._connection_started = asyncio.get_running_loop().time()
            await self._touch_activity()

            await websocket.send(
                json.dumps(
                    {
                        "action": "session_config",
                        "recording_path": str(self._temp_recording_path),
                    }
                )
            )

            await self._send_scheduled_inputs(websocket, self._scheduled_inputs)
            self._scheduler_done.set()
            await self._wait_until_idle(websocket)
            await self._wait_for_recording_stable()
        finally:
            await websocket.close()
            if self._receiver_task is not None:
                try:
                    await self._receiver_task
                except websockets.ConnectionClosed:
                    pass
            if self._playback is not None:
                await self._playback.close()

        await asyncio.sleep(0.1)
        await self._materialize_output()
        if self._error is not None:
            raise self._error

    async def _login(self) -> str:
        payload = await asyncio.to_thread(
            requests.post,
            f"{self._http_base_url}/api/auth/login",
            timeout=10,
        )
        payload.raise_for_status()
        body = payload.json()
        token = body.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Login response did not include access_token")
        return token

    def _build_authenticated_ws_url(self, token: str) -> str:
        separator = "&" if "?" in self._websocket_url else "?"
        return f"{self._websocket_url}{separator}access_token={token}"

    async def _receiver_loop(
        self, websocket: websockets.WebSocketClientProtocol
    ) -> None:
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    await self._touch_activity()
                    if self._playback is not None:
                        await self._playback.push(message)
                    continue
                await self._handle_text_message(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._ws_closed.set()

    async def _handle_text_message(self, message: str) -> None:
        payload = json.loads(message)
        action = payload.get("action")
        data = payload.get("data")
        if action not in NON_ACTIVITY_ACTIONS:
            await self._touch_activity()
        if action == "session_attached":
            self._attached_event.set()
            return
        if action == "pause_tts" and self._playback is not None:
            await self._playback.pause()
            return
        if action == "resume_tts" and self._playback is not None:
            await self._playback.resume()
            return
        if action == "stop_tts" and self._playback is not None:
            await self._playback.stop()
            return
        if action == "tts_finished" and self._playback is not None:
            await self._playback.mark_server_tts_finished()
            return
        if action in {"update_asr", "finish_asr"}:
            text = ""
            if isinstance(data, dict):
                raw_text = data.get("text")
                if isinstance(raw_text, str):
                    text = raw_text
            self._asr_results.append(
                ASRResultRecord(
                    action=action,
                    text=text,
                    received_at=asyncio.get_running_loop().time(),
                )
            )
            return
        if action == "full_audio_frame":
            self._last_full_audio_at = asyncio.get_running_loop().time()
            self._collect_full_audio_frame(data)
            return
        if action == "error":
            self._error = RuntimeError(f"{self._case_dir.name}: backend error: {data}")

    def _collect_full_audio_frame(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        audio_base64 = data.get("audio_base64")
        if not isinstance(audio_base64, str) or not audio_base64:
            return
        sample_rate = int(data.get("sample_rate", 48000))
        channels = int(data.get("channels", 2))
        audio_format = data.get("format", "pcm_s16le")
        if sample_rate != 48000 or channels != 2 or audio_format != "pcm_s16le":
            raise RuntimeError(
                f"Unsupported full_audio_frame format: sr={sample_rate}, channels={channels}, format={audio_format}"
            )
        self._full_audio_bytes.extend(base64.b64decode(audio_base64))

    async def _send_scheduled_inputs(
        self,
        websocket: websockets.WebSocketClientProtocol,
        scheduled_inputs: list[ScheduledAudioInput],
    ) -> None:
        previous_input_start: float | None = None
        previous_input_end: float | None = None
        for scheduled_input in scheduled_inputs:
            await self._send_silence_until(
                websocket,
                scheduled_input.time_spec,
                previous_input_start=previous_input_start,
                previous_input_end=previous_input_end,
            )
            (
                previous_input_start,
                previous_input_end,
            ) = await self._stream_audio_file(websocket, scheduled_input.audio_path)
            if self._error is not None:
                raise self._error

    async def _try_resolve_target_time(
        self,
        time_spec: RelativeTimeSpec,
        *,
        previous_input_start: float | None,
        previous_input_end: float | None,
    ) -> float | None:
        if self._connection_started is None:
            raise RuntimeError("Connection start time is not initialized")
        if time_spec.kind == "absolute":
            return self._connection_started + float(time_spec.value or 0.0)
        if time_spec.anchor is None:
            raise RuntimeError("Relative time spec missing anchor")
        minimum_anchor_time = self._connection_started
        if time_spec.anchor in {"last_ai_start", "last_ai_end"}:
            if previous_input_end is not None:
                minimum_anchor_time = previous_input_end
        elif previous_input_start is not None:
            minimum_anchor_time = previous_input_start
        anchor_value = await self._anchors.get_first_at_or_after(
            time_spec.anchor, minimum_anchor_time
        )
        if anchor_value is None:
            return None
        return anchor_value + time_spec.offset

    async def _stream_audio_file(
        self,
        websocket: websockets.WebSocketClientProtocol,
        audio_path: Path,
    ) -> tuple[float, float]:
        pcm_bytes = load_audio_as_pcm16(audio_path, target_sr=16000)
        frame_bytes = ClientVADController.FRAME_BYTES
        vad_controller = (
            ClientVADController(redemption_ms=self._vad_redemption_ms)
            if self._with_vad
            else None
        )
        audio_frame_count = int(math.ceil(len(pcm_bytes) / frame_bytes))
        if vad_controller is not None:
            trailing_frames = vad_controller.trailing_silence_frames()
        else:
            frame_ms = ClientVADController.FRAME_SAMPLES * 1000.0 / 16000.0
            trailing_frames = max(1, int(round(self._vad_redemption_ms / frame_ms)) + 1)
        total_frames = audio_frame_count + trailing_frames
        stream_started = asyncio.get_running_loop().time()
        await self._anchors.set("last_user_start", stream_started)
        last_user_end_set = False
        user_end_time: float | None = None

        for frame_index in range(total_frames):
            target_send_time = stream_started + (
                frame_index * (ClientVADController.FRAME_SAMPLES / 16000.0)
            )
            now = asyncio.get_running_loop().time()
            if target_send_time > now:
                await asyncio.sleep(target_send_time - now)

            start = frame_index * frame_bytes
            frame = pcm_bytes[start : start + frame_bytes]
            if len(frame) < frame_bytes:
                frame = frame + (b"\x00" * (frame_bytes - len(frame)))

            await websocket.send(frame)
            await self._touch_activity()
            if not last_user_end_set and frame_index + 1 >= audio_frame_count:
                user_end_time = asyncio.get_running_loop().time()
                await self._anchors.set("last_user_end", user_end_time)
                last_user_end_set = True
            if vad_controller is not None:
                for action in vad_controller.feed(frame):
                    await websocket.send(json.dumps({"action": action}))
                    await self._touch_activity()

        if not last_user_end_set:
            user_end_time = asyncio.get_running_loop().time()
            await self._anchors.set("last_user_end", user_end_time)

        if user_end_time is None:
            raise RuntimeError("User end time was not recorded")
        return stream_started, user_end_time

    async def _send_silence_until(
        self,
        websocket: websockets.WebSocketClientProtocol,
        time_spec: RelativeTimeSpec,
        *,
        previous_input_start: float | None,
        previous_input_end: float | None,
    ) -> None:
        """Continuously send silent PCM frames until one scheduled input is due."""
        silence_frame = b"\x00" * ClientVADController.FRAME_BYTES
        frame_duration = ClientVADController.FRAME_SAMPLES / 16000.0
        while True:
            if self._error is not None:
                raise self._error
            if self._ws_closed.is_set():
                raise RuntimeError(f"{self._case_dir.name}: websocket closed early")
            now = asyncio.get_running_loop().time()
            target_time = await self._try_resolve_target_time(
                time_spec,
                previous_input_start=previous_input_start,
                previous_input_end=previous_input_end,
            )
            if target_time is not None and target_time <= now:
                return
            await websocket.send(silence_frame)
            sleep_for = frame_duration
            if target_time is not None:
                sleep_for = min(frame_duration, max(0.0, target_time - now))
            if sleep_for > 0.0:
                await asyncio.sleep(sleep_for)

    async def _wait_until_idle(
        self, websocket: websockets.WebSocketClientProtocol
    ) -> None:
        while True:
            if self._error is not None:
                raise self._error
            playback_idle = True
            if self._playback is not None:
                playback_idle = await self._playback.is_idle()

            async with self._activity_lock:
                last_activity = self._last_activity
            quiet_for = asyncio.get_running_loop().time() - last_activity
            last_user_end = await self._anchors.get("last_user_end")
            ai_end_seen = await self._anchors.get("last_ai_end")
            if self._scheduler_done.is_set() and playback_idle:
                if last_user_end is None:
                    if quiet_for >= self._settle_seconds:
                        return
                elif ai_end_seen is not None and ai_end_seen >= last_user_end:
                    if quiet_for >= self._settle_seconds:
                        return
            if self._ws_closed.is_set():
                return
            await asyncio.sleep(0.1)

    async def _wait_for_recording_stable(self) -> None:
        """Wait briefly for server-side recording writes to settle."""
        if not self._temp_recording_path.exists():
            return

        deadline = time.monotonic() + RECORDING_STABLE_TIMEOUT_SECONDS
        stable_since = time.monotonic()
        last_size = self._temp_recording_path.stat().st_size

        while time.monotonic() < deadline:
            await asyncio.sleep(0.1)
            try:
                current_size = self._temp_recording_path.stat().st_size
            except OSError:
                return
            if current_size != last_size:
                last_size = current_size
                stable_since = time.monotonic()
                continue
            if time.monotonic() - stable_since >= RECORDING_STABLE_SECONDS:
                return

    async def _touch_activity(self) -> None:
        async with self._activity_lock:
            self._last_activity = asyncio.get_running_loop().time()

    async def _materialize_output(self) -> None:
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        audio, sample_rate = await self._load_analysis_audio()
        self._latency_samples_ms = compute_case_latency_samples_from_array(
            audio, sample_rate
        )
        mono_audio = downmix_stereo_to_mono(audio)
        write_mono_mp3(self._output_path, audio=mono_audio, sample_rate=sample_rate)

    async def _load_analysis_audio(self) -> tuple[np.ndarray, int]:
        """Load the stereo recording used for post-run analysis."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._temp_recording_path.exists():
                try:
                    size = self._temp_recording_path.stat().st_size
                except OSError:
                    size = 0
                if size > 44:
                    audio, sample_rate = await asyncio.to_thread(
                        read_audio_int16, self._temp_recording_path
                    )
                    if audio.shape[1] < 2:
                        raise RuntimeError(
                            f"{self._case_dir.name}: server recording is not stereo"
                        )
                    if not stereo_audio_has_right_channel_signal(audio):
                        raise RuntimeError(
                            f"{self._case_dir.name}: server recording AI channel is empty"
                        )
                    return audio, sample_rate
            await asyncio.sleep(0.1)

        full_audio_bytes = bytes(self._full_audio_bytes)
        if not full_audio_bytes:
            raise RuntimeError(
                f"{self._case_dir.name}: did not receive any full_audio_frame"
            )
        if not stereo_pcm_has_right_channel_signal(full_audio_bytes):
            raise RuntimeError(
                f"{self._case_dir.name}: received full_audio_frame but AI channel is empty"
            )
        audio = np.frombuffer(full_audio_bytes, dtype=np.int16)
        if audio.size % 2 != 0:
            raise RuntimeError(
                f"{self._case_dir.name}: received malformed stereo PCM frame data"
            )
        return audio.reshape(-1, 2), 48000


def validate_vad_configuration(config: dict[str, Any], *, with_vad: bool) -> None:
    """Validate frontend/backend VAD compatibility for test mode."""
    has_backend_vad = bool(config.get("vad"))
    if with_vad and has_backend_vad:
        raise ValueError(
            "with_vad=true uses client-side VAD; remove backend 'vad' from the server config to avoid duplicate turn events."
        )
    if not with_vad and not has_backend_vad:
        raise ValueError(
            "with_vad=false requires a backend 'vad' model in the server config."
        )


def build_asr_report_payload(
    *,
    scheduled_inputs: list[ScheduledAudioInput],
    asr_results: list[ASRResultRecord],
    judgement: ASRJudgement | None,
) -> dict[str, Any]:
    """Build the persisted ASR report payload for one case."""
    expected = [
        {
            "audio": item.audio_path.name,
            "text": item.expected_text,
        }
        for item in scheduled_inputs
    ]
    observed = [
        {
            "action": record.action,
            "text": record.text,
            "received_at": record.received_at,
        }
        for record in asr_results
    ]
    payload: dict[str, Any] = {
        "expected": expected,
        "observed": observed,
    }
    if judgement is not None:
        payload["judge_asr"] = {
            "passed": judgement.passed,
            "reason": judgement.reason,
        }
    return payload


def validate_case_inputs_for_criteria(
    case_name: str,
    scheduled_inputs: list[ScheduledAudioInput],
    criteria: CaseCriteria,
) -> None:
    """Validate that scheduled inputs satisfy case-level evaluation criteria.

    Parameters
    ----------
    case_name : str
        Case directory name.
    scheduled_inputs : list[ScheduledAudioInput]
        Parsed timestamp entries for the case.
    criteria : CaseCriteria
        Case-level evaluation settings.
    """
    if not criteria.judge_asr:
        return
    missing_text_audios = [
        item.audio_path.name
        for item in scheduled_inputs
        if item.expected_text is None or not item.expected_text.strip()
    ]
    if missing_text_audios:
        missing_summary = ", ".join(missing_text_audios)
        raise ValueError(
            f"{case_name}: judge_asr=true requires timestamp.txt to include transcript text as the third column for every entry; missing text for: {missing_summary}"
        )


async def evaluate_case_result(
    result: CaseExecutionResult,
    *,
    judge_llm: JudgeLLMConfig | None,
    output_root: Path,
) -> bool:
    """Evaluate one executed case and persist auxiliary analysis files."""
    audio_exists = result.output_path.exists()
    asr_report_path = get_case_asr_report_path(output_root, result.case_name)
    if not result.criteria.judge_asr:
        return result.error is None and audio_exists

    if result.error is not None or not audio_exists:
        failure_reasons: list[str] = []
        if result.error is not None:
            failure_reasons.append(result.error)
        if not audio_exists:
            failure_reasons.append("output audio is missing")
        write_json(
            asr_report_path,
            build_asr_report_payload(
                scheduled_inputs=result.scheduled_inputs,
                asr_results=result.asr_results,
                judgement=ASRJudgement(
                    passed=False,
                    reason="; ".join(failure_reasons),
                ),
            ),
        )
        return False

    expected_texts = [
        item.expected_text.strip()
        for item in result.scheduled_inputs
        if item.expected_text is not None and item.expected_text.strip()
    ]
    if len(expected_texts) != len(result.scheduled_inputs):
        report_payload = build_asr_report_payload(
            scheduled_inputs=result.scheduled_inputs,
            asr_results=result.asr_results,
            judgement=ASRJudgement(
                passed=False,
                reason="judge_asr=true requires transcript text on every timestamp.txt line.",
            ),
        )
        write_json(asr_report_path, report_payload)
        return False

    actual_texts = [
        record.text.strip()
        for record in result.asr_results
        if record.action == "finish_asr" and record.text.strip()
    ]
    if judge_llm is None:
        raise ValueError("judge_llm must be configured when any case enables judge_asr")

    try:
        judgement = await asyncio.to_thread(
            judge_asr_with_llm,
            judge_llm,
            expected_texts=expected_texts,
            actual_texts=actual_texts,
        )
    except Exception as exc:
        judgement = ASRJudgement(
            passed=False,
            reason=f"Judge LLM request failed: {exc}",
        )
    report_payload = build_asr_report_payload(
        scheduled_inputs=result.scheduled_inputs,
        asr_results=result.asr_results,
        judgement=judgement,
    )
    write_json(asr_report_path, report_payload)
    return judgement.passed


async def run_test_mode(args: argparse.Namespace) -> None:
    """Run automated backend tests."""
    if args.config is None:
        raise ValueError("--config is required in test mode")
    dataset_root = args.input.resolve()
    if not dataset_root.is_dir():
        raise ValueError(f"Input dataset directory does not exist: {dataset_root}")

    raw_service_config = load_json(args.config.resolve())
    merged_service_config = merge_service_config(raw_service_config)
    effective_test_config = load_effective_test_config(
        dataset_root,
        concurrency_override=args.concurrency,
        with_vad_override=args.with_vad_override,
        vad_redemption_ms_override=args.vad_redemption_ms,
        judge_llm_model_override=args.judge_llm_model,
        judge_llm_base_url_override=args.judge_llm_base_url,
        judge_llm_api_key_override=args.judge_llm_api_key,
    )
    validate_vad_configuration(
        merged_service_config,
        with_vad=effective_test_config.with_vad,
    )

    output_root = args.out.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs_output_root = output_root / "logs"
    logs_output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "test_config.json",
        {
            "concurrency": effective_test_config.concurrency,
            "with_vad": effective_test_config.with_vad,
            "vad_redemption_ms": effective_test_config.vad_redemption_ms,
            "judge_llm": (
                {
                    "model": effective_test_config.judge_llm.model,
                    "base_url": effective_test_config.judge_llm.base_url,
                    "api_key": effective_test_config.judge_llm.api_key,
                }
                if effective_test_config.judge_llm is not None
                else None
            ),
        },
    )
    write_json(output_root / "service_config.json", merged_service_config)

    temp_recording_root = output_root / ".server_recordings"
    temp_recording_root.mkdir(parents=True, exist_ok=True)

    case_dirs = discover_case_dirs(dataset_root)
    case_inputs = {case_dir.name: parse_case_inputs(case_dir) for case_dir in case_dirs}
    case_criteria = {case_dir.name: load_case_criteria(case_dir) for case_dir in case_dirs}
    for case_dir in case_dirs:
        validate_case_inputs_for_criteria(
            case_dir.name,
            case_inputs[case_dir.name],
            case_criteria[case_dir.name],
        )
    for case_dir in case_dirs:
        (output_root / f"{case_dir.name}.mp3").unlink(missing_ok=True)
        (output_root / f"{case_dir.name}.flac").unlink(missing_ok=True)
        (output_root / f"{case_dir.name}.wav").unlink(missing_ok=True)
        get_legacy_case_asr_report_path(output_root, case_dir.name).unlink(
            missing_ok=True
        )
        get_case_asr_report_path(output_root, case_dir.name).unlink(missing_ok=True)
    if (
        any(criteria.judge_asr for criteria in case_criteria.values())
        and effective_test_config.judge_llm is None
    ):
        raise ValueError(
            "judge_llm must be configured in the dataset config or CLI when any case enables judge_asr"
        )
    semaphore = asyncio.Semaphore(effective_test_config.concurrency)
    case_results: list[CaseExecutionResult] = []

    async with EmbeddedServer(
        config=merged_service_config,
        host=EMBEDDED_SERVER_HOST,
        port=EMBEDDED_SERVER_PORT,
    ) as server:

        async def run_one_case(case_dir: Path) -> None:
            async with semaphore:
                runner = CaseRunner(
                    case_dir=case_dir,
                    scheduled_inputs=case_inputs[case_dir.name],
                    output_path=output_root / f"{case_dir.name}.mp3",
                    temp_recording_path=temp_recording_root / f"{case_dir.name}.wav",
                    websocket_url=server.websocket_url,
                    http_base_url=server.http_base_url,
                    with_vad=effective_test_config.with_vad,
                    vad_redemption_ms=effective_test_config.vad_redemption_ms,
                    settle_seconds=DEFAULT_SETTLE_SECONDS,
                )
                error: str | None = None
                try:
                    await runner.run()
                except Exception as exc:
                    error = str(exc)
                case_results.append(
                    CaseExecutionResult(
                        case_name=case_dir.name,
                        output_path=output_root / f"{case_dir.name}.mp3",
                        latency_samples_ms=runner.latency_samples_ms,
                        criteria=case_criteria[case_dir.name],
                        scheduled_inputs=case_inputs[case_dir.name],
                        asr_results=runner.asr_results,
                        error=error,
                    )
                )

        try:
            await asyncio.gather(*(run_one_case(case_dir) for case_dir in case_dirs))
        finally:
            shutil.rmtree(temp_recording_root, ignore_errors=True)

    eval_cases: dict[str, dict[str, bool]] = {}
    latency_values_ms: list[float] = []
    for case_result in sorted(case_results, key=lambda item: item.case_name):
        passed = await evaluate_case_result(
            case_result,
            judge_llm=effective_test_config.judge_llm,
            output_root=output_root,
        )
        eval_cases[case_result.case_name] = {"passed": passed}
        if case_result.error is None and case_result.output_path.exists():
            latency_values_ms.extend(case_result.latency_samples_ms)

    latency_ms = (
        sum(latency_values_ms) / float(len(latency_values_ms))
        if latency_values_ms
        else 0.0
    )
    write_json(
        output_root / "eval.json",
        {
            "latency_ms": latency_ms,
            "cases": eval_cases,
        },
    )


def resolve_create_output_root(source_root: Path, requested_out: Path) -> Path:
    """Resolve the actual root folder for generated test cases."""
    if not requested_out.exists():
        return requested_out
    if not requested_out.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {requested_out}")
    return requested_out / source_root.name


def load_tts_from_config(config_path: Path):
    """Instantiate a TTS model from a config file using Xtalk's registry."""
    raw_config = load_json(config_path)
    tts_config = raw_config.get("tts", raw_config)
    if not isinstance(tts_config, dict):
        raise ValueError(f"Invalid TTS config in {config_path}")
    return init_registered_model(
        slot="tts",
        model_config=tts_config,
    )


def run_create_mode(args: argparse.Namespace) -> None:
    """Generate a test dataset from text lines and a TTS configuration."""
    source_root = args.create.resolve()
    if not source_root.is_dir():
        raise ValueError(f"Generation source directory does not exist: {source_root}")

    test_config_path = resolve_root_json(
        source_root,
        preferred_names=PREFERRED_TEST_CONFIG_NAMES,
    )
    tts_config_path = resolve_root_json(
        source_root,
        preferred_names=PREFERRED_TTS_CONFIG_NAMES,
    )
    if tts_config_path is None:
        raise ValueError(f"No TTS config JSON found under {source_root}.")

    tts_model = load_tts_from_config(tts_config_path)
    if tts_model is None:
        raise RuntimeError("Failed to initialize TTS model from the provided config")

    output_root = resolve_create_output_root(source_root, args.out.resolve())
    output_root.mkdir(parents=True, exist_ok=True)
    if test_config_path is not None:
        shutil.copyfile(test_config_path, output_root / "test_config.json")
    shutil.copyfile(tts_config_path, output_root / "tts_config.json")

    sample_rate_hint = int(getattr(tts_model, "sample_rate", 48000) or 48000)
    for case_dir in discover_case_dirs(source_root):
        lines = parse_generation_case(case_dir)
        case_output_dir = output_root / case_dir.name
        case_output_dir.mkdir(parents=True, exist_ok=True)
        criteria_path = case_dir / "criteria.yaml"
        if criteria_path.exists():
            shutil.copyfile(criteria_path, case_output_dir / "criteria.yaml")
        timestamp_lines: list[str] = []
        for index, line in enumerate(lines):
            audio_name = f"audio_{index:03d}.wav"
            audio_payload = tts_model.synthesize(line.text)
            pcm_bytes, sample_rate = decode_tts_audio_payload(
                audio_payload,
                sample_rate_hint=sample_rate_hint,
            )
            write_pcm_wav(
                case_output_dir / audio_name,
                pcm_bytes=pcm_bytes,
                sample_rate=sample_rate,
                channels=1,
            )
            timestamp_lines.append(f"{line.time_spec}:{audio_name}:{line.text}")

        (case_output_dir / "timestamp.txt").write_text(
            "\n".join(timestamp_lines) + "\n",
            encoding="utf-8",
        )


async def async_main(args: argparse.Namespace) -> None:
    """Dispatch to the requested mode."""
    if args.input is not None:
        await run_test_mode(args)
        return
    run_create_mode(args)


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
