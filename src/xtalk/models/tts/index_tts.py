import argparse
import io
from pathlib import Path
import socket
import sys
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse

import aiohttp
import numpy as np
import requests
import soundfile as sf
import soxr

from .interfaces import TTS
from ..registry import model


IndexTTSModelVersion = Literal["1.5", "2"]
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
_DEFAULT_BASE_URL = "http://localhost:11996"
_DEFAULT_V2_PORT = 6006


def _resolve_index_tts_base_url(base_url: str) -> str:
    """Resolve the IndexTTS HTTP base URL."""
    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must not be empty")
    parsed = urlparse(normalized)
    if not parsed.scheme:
        normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("IndexTTS base_url must use http:// or https://")
    if normalized.endswith("/tts_url"):
        normalized = normalized[: -len("/tts_url")]
    return normalized


def _resolve_model_version(model_version: str) -> IndexTTSModelVersion:
    """Validate and normalize an IndexTTS model version."""
    normalized = str(model_version).strip()
    if normalized not in {"1.5", "2"}:
        raise ValueError("IndexTTS model must be '1.5' or '2'")
    return normalized  # type: ignore[return-value]


def _base_url_from_host_port(host: Optional[str], port: Optional[int]) -> str:
    """Build a base URL from host and port compatibility parameters."""
    resolved_host = host or "localhost"
    resolved_port = port if port is not None else _DEFAULT_V2_PORT
    return f"http://{resolved_host}:{resolved_port}"


@model
class IndexTTS(TTS):
    """HTTP client adapter for IndexTTS 1.5 and IndexTTS 2 services."""

    def __init__(
        self,
        voices: Optional[List[Dict[str, str]]] = None,
        base_url: Optional[str] = None,
        sample_rate: int = 48000,
        timeout: float = 30.0,
        model: IndexTTSModelVersion = "1.5",
        voices_map: Optional[List[Dict[str, str]]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        emo_weight: float = 1.0,
        emo_random: bool = False,
        max_text_tokens_per_sentence: int = 120,
    ):
        """Initialize IndexTTS.

        Parameters
        ----------
        voices : list[dict[str, str]] | None, optional
            Voice configurations containing ``name`` and ``path``.
        base_url : str | None, optional
            Base URL of the IndexTTS HTTP service.
        sample_rate : int, optional
            Output sample rate.
        timeout : float, optional
            Request timeout in seconds.
        model : {"1.5", "2"}, optional
            IndexTTS service protocol version.
        voices_map : list[dict[str, str]] | None, optional
            Backwards-compatible alias for ``voices``.
        host : str | None, optional
            Backwards-compatible host used when ``base_url`` is omitted.
        port : int | None, optional
            Backwards-compatible port used when ``base_url`` is omitted.
        emo_weight : float, optional
            Emotion reference weight used by IndexTTS 2 audio-prompt control.
        emo_random : bool, optional
            Whether IndexTTS 2 should use random emotion sampling.
        max_text_tokens_per_sentence : int, optional
            IndexTTS 2 sentence token cap.
        """
        if base_url is None:
            base_url = (
                _base_url_from_host_port(host, port)
                if host is not None or port is not None
                else _DEFAULT_BASE_URL
            )

        self.model = _resolve_model_version(model)
        self.base_url = _resolve_index_tts_base_url(base_url)
        self.url = f"{self.base_url}/tts_url"
        self._voices = [voice.copy() for voice in (voices if voices is not None else voices_map) or []]
        self._sample_rate = int(sample_rate)
        self._timeout = float(timeout)
        self._emo_weight = float(emo_weight)
        self._emo_random = bool(emo_random)
        self._max_text_tokens_per_sentence = int(max_text_tokens_per_sentence)
        self.audio_paths = [voice.get("path", "") for voice in self._voices]
        self._base_voices = [voice.copy() for voice in self._voices]
        self._voice_path_map: Dict[str, str] = {}
        self._active_voice_names: List[str] = [
            voice["name"] for voice in self._voices if "name" in voice
        ]
        if not self._active_voice_names and self._voices:
            first_name = self._voices[0].get("name")
            if first_name:
                self._active_voice_names = [first_name]

        self._emotion_audio_map: Dict[str, Dict[str, str]] = {}
        self._current_emotion: Optional[str] = None
        self._current_emotion_vector: Optional[List[float]] = None
        self._build_emotion_map()

    def _build_emotion_map(self) -> None:
        """Build the emotion-to-audio mapping for each reference voice."""
        self._emotion_audio_map.clear()
        self._voice_path_map.clear()

        for voice in self._voices:
            voice_name = voice.get("name")
            voice_path = voice.get("path")
            if not voice_name or not voice_path:
                continue
            self._voice_path_map[voice_name] = voice_path
            path_obj = Path(voice_path)

            if path_obj.is_dir():
                emotion_files: Dict[str, str] = {}
                for audio_file in path_obj.iterdir():
                    if not audio_file.is_file():
                        continue
                    if audio_file.suffix.lower() not in _AUDIO_EXTENSIONS:
                        continue
                    file_stem = audio_file.stem
                    if file_stem == voice_name:
                        emotion_name = "neutral"
                    elif file_stem.startswith(f"{voice_name}_"):
                        emotion_name = file_stem[len(voice_name) + 1 :]
                    else:
                        emotion_name = file_stem
                    emotion_files[emotion_name] = str(audio_file)
                if emotion_files:
                    self._emotion_audio_map[voice_name] = emotion_files
            elif path_obj.is_file():
                self._emotion_audio_map[voice_name] = {"neutral": voice_path}
            else:
                self._emotion_audio_map[voice_name] = {"neutral": voice_path}

        valid_active = [
            name for name in self._active_voice_names if name in self._voice_path_map
        ]
        if valid_active:
            self._active_voice_names = valid_active
        else:
            self._active_voice_names = list(self._voice_path_map.keys())

        if self.model == "2" and len(self._active_voice_names) > 1:
            self._active_voice_names = self._active_voice_names[:1]

    def clone(self) -> "IndexTTS":
        """Create a session-safe clone of this client."""
        clone = IndexTTS(
            voices=[voice.copy() for voice in self._base_voices],
            base_url=self.base_url,
            sample_rate=self._sample_rate,
            timeout=self._timeout,
            model=self.model,
            emo_weight=self._emo_weight,
            emo_random=self._emo_random,
            max_text_tokens_per_sentence=self._max_text_tokens_per_sentence,
        )
        clone._active_voice_names = list(self._active_voice_names)
        clone._current_emotion = self._current_emotion
        clone._current_emotion_vector = (
            list(self._current_emotion_vector)
            if self._current_emotion_vector is not None
            else None
        )
        return clone

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        """Convert a float32 ndarray in [-1, 1] to PCM int16 bytes."""
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def set_voice(self, voice_names: List[str]) -> None:
        """Set the active voice or voices."""
        if not voice_names:
            raise ValueError("voice_names cannot be empty for IndexTTS")
        if self.model == "2" and len(voice_names) != 1:
            raise ValueError("IndexTTS model '2' only accepts one reference voice")
        missing = [name for name in voice_names if name not in self._voice_path_map]
        if missing:
            raise ValueError(f"Unknown voice names: {missing}")
        self._active_voice_names = list(voice_names)

    def set_emotion(self, emotion: str | list[float]) -> None:
        """Set the current emotion using a label, text description, or vector."""
        if isinstance(emotion, list):
            if self.model != "2":
                raise TypeError("IndexTTS model '1.5' only accepts string emotions")
            if len(emotion) != 8:
                raise ValueError("IndexTTS model '2' emotion vectors must have 8 values")
            self._current_emotion_vector = [float(value) for value in emotion]
            self._current_emotion = None
            return

        self._current_emotion = emotion or None
        self._current_emotion_vector = None

    def _active_voice_names_or_default(self) -> List[str]:
        """Return active voice names, falling back to all configured voices."""
        active_names = self._active_voice_names or list(self._voice_path_map.keys())
        if not active_names:
            raise ValueError("No voices configured for IndexTTS")
        return active_names

    def _resolve_audio_paths(self) -> List[str]:
        """Resolve active IndexTTS 1.5 reference audio paths."""
        audio_paths_to_use: List[str] = []
        for voice_name in self._active_voice_names_or_default():
            if voice_name in self._emotion_audio_map:
                emotion_map = self._emotion_audio_map[voice_name]
                emotion_audio: Optional[str]
                if self._current_emotion:
                    emotion_audio = emotion_map.get(self._current_emotion)
                    if emotion_audio is None:
                        emotion_audio = emotion_map.get("neutral")
                else:
                    emotion_audio = emotion_map.get("neutral")
                if emotion_audio is None and emotion_map:
                    emotion_audio = next(iter(emotion_map.values()))
                if emotion_audio is None:
                    raise ValueError(
                        f"No valid emotion audio file found for voice '{voice_name}'"
                    )
                audio_paths_to_use.append(emotion_audio)
            else:
                resolved = self._voice_path_map.get(voice_name)
                if not resolved:
                    raise ValueError(f"Voice '{voice_name}' is not configured")
                audio_paths_to_use.append(resolved)
        return audio_paths_to_use

    def _resolve_speaker_audio_path(self, voice_name: str) -> str:
        """Resolve the single IndexTTS 2 speaker reference audio path."""
        emotion_map = self._emotion_audio_map.get(voice_name, {})
        if "neutral" in emotion_map:
            return emotion_map["neutral"]
        if emotion_map:
            return next(iter(emotion_map.values()))
        resolved = self._voice_path_map.get(voice_name)
        if not resolved:
            raise ValueError(f"Voice '{voice_name}' is not configured")
        return resolved

    def _prepare_v1_payload(self, text: str) -> dict:
        """Build an IndexTTS 1.5 request payload."""
        return {"text": text, "audio_paths": self._resolve_audio_paths()}

    def _prepare_v2_payload(self, text: str) -> dict:
        """Build an IndexTTS 2 request payload with emotion controls."""
        active_names = self._active_voice_names_or_default()
        if len(active_names) != 1:
            raise ValueError("IndexTTS model '2' only accepts one reference voice")
        voice_name = active_names[0]
        payload = {
            "text": text,
            "spk_audio_path": self._resolve_speaker_audio_path(voice_name),
            "emo_control_method": 0,
            "emo_vec": [0.0] * 8,
            "emo_random": self._emo_random,
            "max_text_tokens_per_sentence": self._max_text_tokens_per_sentence,
        }

        if self._current_emotion_vector is not None:
            payload["emo_control_method"] = 2
            payload["emo_vec"] = self._current_emotion_vector
            return payload

        if self._current_emotion:
            if self._current_emotion == "neutral":
                return payload
            emotion_map = self._emotion_audio_map.get(voice_name, {})
            emotion_ref_path = emotion_map.get(self._current_emotion)
            if emotion_ref_path is not None:
                payload["emo_control_method"] = 1
                payload["emo_ref_path"] = emotion_ref_path
                payload["emo_weight"] = self._emo_weight
            else:
                payload["emo_control_method"] = 3
                payload["emo_text"] = self._current_emotion
        return payload

    def _prepare_request_payload(self, text: str) -> dict:
        """Build the request payload for the configured IndexTTS protocol."""
        if self.model == "2":
            return self._prepare_v2_payload(text)
        return self._prepare_v1_payload(text)

    def _resample_bytes(self, raw_bytes: bytes) -> bytes:
        """Decode returned WAV bytes and resample to the configured PCM rate."""
        audio, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        resampled = soxr.resample(audio, src_sr, self._sample_rate)
        return self._float32_to_pcm_bytes(resampled)

    def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize speech and return PCM audio bytes."""
        del kwargs
        response = requests.post(
            self.url,
            json=self._prepare_request_payload(text),
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._resample_bytes(response.content)

    async def async_synthesize(self, text: str, **kwargs) -> bytes:
        """Asynchronously synthesize speech and return PCM audio bytes."""
        del kwargs
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url,
                json=self._prepare_request_payload(text),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                content = await resp.read()
        return self._resample_bytes(content)



def _is_tcp_port_open(host: str, port: int, timeout: float) -> bool:
    """Return whether a TCP listener accepts connections at host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _run_synthesis_case(tts: IndexTTS, label: str, text: str) -> int:
    """Run one synthesis request and report the PCM byte count."""
    try:
        audio = tts.synthesize(text)
    except requests.RequestException as exc:
        print(f"IndexTTS {label} request failed: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        print(f"IndexTTS {label} synthesis failed: {exc}", file=sys.stderr)
        return 1

    if not audio or len(audio) % 2 != 0:
        print(
            f"IndexTTS {label} returned invalid PCM ({len(audio)} bytes).",
            file=sys.stderr,
        )
        return 1

    print(f"IndexTTS {label} succeeded: output_bytes={len(audio)}")
    return 0


def _run_remote_smoke_test() -> int:
    """Run a remote IndexTTS 1.5 or 2 client smoke test."""
    parser = argparse.ArgumentParser(description="Smoke test the IndexTTS HTTP client.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--model", choices=["1.5", "2"], default="1.5")
    parser.add_argument("--ref-audio-path", required=True)
    parser.add_argument("--text", default="这是一次语音合成测试。")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    port = args.port if args.port is not None else (11996 if args.model == "1.5" else 11997)
    if not _is_tcp_port_open(args.host, port, timeout=2.0):
        print(
            "No service is listening on "
            f"{args.host}:{port}. Please start indextts {args.model} first.",
            file=sys.stderr,
        )
        return 2

    ref_audio_path = Path(args.ref_audio_path).expanduser()
    if not ref_audio_path.exists():
        print(f"Reference audio does not exist: {ref_audio_path}", file=sys.stderr)
        return 2

    base_url = f"http://{args.host}:{port}"
    tts = IndexTTS(
        voices=[{"name": "test", "path": str(ref_audio_path)}],
        base_url=base_url,
        model=args.model,
        timeout=args.timeout,
    )

    failed = _run_synthesis_case(tts, f"model={args.model} default", args.text)
    if args.model == "2":
        tts.set_emotion([0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 0.0, 0.0])
        failed |= _run_synthesis_case(tts, "model=2 emotion_vector", args.text)
        tts.set_emotion("极度悲伤")
        failed |= _run_synthesis_case(tts, "model=2 emotion_text", args.text)

    if failed:
        return 1
    print(f"IndexTTS smoke test passed: model={args.model}, base_url={base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_remote_smoke_test())
