import requests
import aiohttp
import soxr
import io
import soundfile as sf
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass
from pathlib import Path
from ..interfaces import TTS


@dataclass
class EmotionControl:
    """Emotion control parameters."""

    method: int = 0  # 0: no emotion control, 2: emotion vector, 3: emotion text
    vec: Optional[List[float]] = (
        None  # Emotion vector ["joy", "anger", "sadness", "fear", "disgust", "low", "surprise", "calm"]
    )
    text: Optional[str] = None  # Emotion description text

    def __post_init__(self):
        # Default to an 8-dim zero vector
        if self.vec is None:
            self.vec = [0.0] * 8

    def to_dict(self) -> dict:
        """Convert to dict."""
        return {
            "emo_control_method": self.method,
            "emo_vec": self.vec,
            "emo_text": self.text,
        }


class IndexTTS2(TTS):

    def __init__(
        self,
        voices: Optional[List[Dict[str, str]]] = None,
        voices_map: Optional[List[Dict[str, str]]] = None,
        host: str = "localhost",
        port: int = 6006,
        sample_rate: int = 48000,
        timeout: float = 30.0,
    ):
        """
        Initialize IndexTTS2.

        Args:
            voices/voices_map: List of voice configs containing name/path.
            host: TTS server host.
            port: TTS server port.
            sample_rate: Output sample rate.
        """
        self._host = host
        self._port = port
        self.url = f"http://{host}:{port}/tts_url"

        self._voices_map = voices if voices is not None else voices_map
        self._voice_path_map: Dict[str, str] = {}
        self._active_voice_name: Optional[str] = next(
            (voice.get("name") for voice in self._voices_map if voice.get("name")), None
        )
        self._sample_rate = sample_rate
        self._emotion_control = EmotionControl()
        self._timeout = timeout

        self._build_voice_map()

    def _build_voice_map(self):
        """Build the voice-to-audio mapping."""
        self._voice_path_map.clear()

        for voice in self._voices_map:
            voice_name = voice.get("name")
            voice_path = voice.get("path")
            if not voice_name or not voice_path:
                continue
            resolved_path = self._resolve_audio_path(voice_path)
            self._voice_path_map[voice_name] = resolved_path

        if self._active_voice_name not in self._voice_path_map:
            self._active_voice_name = next(iter(self._voice_path_map), None)

    def clone(self):
        """Create a clone of this TTS instance."""
        return IndexTTS2(
            voices_map=[voice.copy() for voice in self._voices_map],
            host=self._host,
            port=self._port,
            sample_rate=self._sample_rate,
            timeout=self._timeout,
        )

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        """float32 ndarray [-1, 1] -> PCM int16 bytes."""
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def set_voice(self, voice_names: List[str]) -> None:
        """Set the active voice (only one is supported)."""
        if not voice_names:
            raise ValueError("voice_names cannot be empty for IndexTTS2")
        if len(voice_names) != 1:
            raise ValueError("IndexTTS2 only accepts one reference voice")
        voice_name = voice_names[0]
        if voice_name not in self._voice_path_map:
            raise ValueError(f"Unknown voice name: {voice_name}")
        self._active_voice_name = voice_name

    def set_emotion(self, emotion: str | List[float]) -> None:
        """Set the emotion via string label or vector."""
        if isinstance(emotion, list):
            self._emotion_control.method = 2
            self._emotion_control.vec = emotion
            self._emotion_control.text = None
        else:
            self._emotion_control.method = 3
            self._emotion_control.text = emotion
            self._emotion_control.vec = [0.0] * 8

    def _prepare_request_payload(self, text: str) -> dict:
        spk_path_to_use = self._get_active_voice_path()

        emo_control_to_use = self._emotion_control
        data = {
            "text": text,
            "spk_audio_path": spk_path_to_use,
            **emo_control_to_use.to_dict(),
        }
        return data

    def _resolve_audio_path(self, ref_path: str) -> str:
        """Resolve a reference path into a concrete audio file path."""
        path_obj = Path(ref_path)

        if path_obj.is_dir():
            exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
            for audio_file in path_obj.iterdir():
                if audio_file.is_file() and audio_file.suffix.lower() in exts:
                    return str(audio_file)
            raise ValueError(f"No audio file found in folder: {ref_path}")
        elif path_obj.is_file():
            return ref_path
        else:
            # Path might already be the target file
            return ref_path

    def _get_active_voice_path(self) -> str:
        if not self._voice_path_map:
            raise ValueError("No voices configured for IndexTTS2")
        voice_name = self._active_voice_name or next(iter(self._voice_path_map))
        if voice_name not in self._voice_path_map:
            raise ValueError(f"Voice '{voice_name}' is not configured")
        return self._voice_path_map[voice_name]

    def _resample_bytes(self, raw_bytes: bytes) -> bytes:
        audio, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        resampled = soxr.resample(audio, src_sr, self._sample_rate)
        return self._float32_to_pcm_bytes(resampled)

    def synthesize(self, text: str) -> bytes:
        """Synthesize speech and return audio bytes."""
        data = self._prepare_request_payload(text)
        response = requests.post(self.url, json=data)
        return self._resample_bytes(response.content)

    async def async_synthesize(self, text: str) -> bytes:
        data = self._prepare_request_payload(text)
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                content = await resp.read()
        return self._resample_bytes(content)
