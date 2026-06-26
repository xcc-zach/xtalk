import requests
import aiohttp
import soxr
import io
import soundfile as sf
import numpy as np
from typing import Optional, List, Dict
from pathlib import Path
from .interfaces import TTS
from ..registry import model


@model
class IndexTTS(TTS):
    def __init__(
        self,
        voices: Optional[List[Dict[str, str]]] = None,
        host: str = "localhost",
        port: int = 11996,
        sample_rate=48000,
        timeout: float = 30.0,
    ):
        self._host = host
        self._port = port
        self.url = f"http://{host}:{port}/tts_url"
        self._voices = [voice.copy() for voice in voices]
        self._sample_rate = sample_rate
        self._timeout = timeout
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

        # Emotion audio map: {folder_name: {emotion: audio_path}}
        self._emotion_audio_map: Dict[str, Dict[str, str]] = {}
        self._current_emotion: Optional[str] = None
        self._build_emotion_map()

    def _build_emotion_map(self):
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
                exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
                for audio_file in path_obj.iterdir():
                    if not audio_file.is_file():
                        continue
                    if audio_file.suffix.lower() not in exts:
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

    def clone(self):
        return IndexTTS(
            voices=[voice.copy() for voice in self._base_voices],
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
        """Set the active voices; multiple voices can be used simultaneously."""
        if not voice_names:
            raise ValueError("voice_names cannot be empty for IndexTTS")
        missing = [name for name in voice_names if name not in self._voice_path_map]
        if missing:
            raise ValueError(f"Unknown voice names: {missing}")
        self._active_voice_names = voice_names

    def set_emotion(self, emotion: str | list[float]) -> None:
        """Set the current emotion using a string label."""
        if isinstance(emotion, list):
            raise TypeError("IndexTTS only accepts string emotions")
        self._current_emotion = emotion

    def _resolve_audio_paths(self) -> List[str]:
        active_names = self._active_voice_names or list(self._voice_path_map.keys())
        if not active_names:
            raise ValueError("No voices configured for IndexTTS")
        audio_paths_to_use: List[str] = []
        for voice_name in active_names:
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

    def _resample_bytes(self, raw_bytes: bytes) -> bytes:
        audio, src_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        resampled = soxr.resample(audio, src_sr, self._sample_rate)
        return self._float32_to_pcm_bytes(resampled)

    def synthesize(self, text: str, **kwargs) -> bytes:
        """Synthesize speech and return audio bytes."""
        audio_paths_to_use = self._resolve_audio_paths()
        data = {"text": text, "audio_paths": audio_paths_to_use}
        response = requests.post(self.url, json=data)
        return self._resample_bytes(response.content)

    async def async_synthesize(self, text: str, **kwargs) -> bytes:
        audio_paths_to_use = self._resolve_audio_paths()
        data = {"text": text, "audio_paths": audio_paths_to_use}
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=data) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                content = await resp.read()
        return self._resample_bytes(content)
