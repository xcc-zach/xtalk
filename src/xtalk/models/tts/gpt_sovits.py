import io
from typing import List, Optional, Dict, Any
from pathlib import Path

import aiohttp
import numpy as np
import requests
import soxr
import soundfile as sf

from .interfaces import TTS
from ..registry import model


@model
class GPTSoVITS(TTS):
    """Minimal HTTP client for GPT-SoVITS api_v2 /tts endpoint."""

    _SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

    def __init__(
        self,
        ref_audio_paths: Optional[List[str]] = None,
        *,
        text_lang: str = "zh",
        prompt_lang: str = "zh",
        prompt_text: str = "",
        host: str = "127.0.0.1",
        port: int = 9880,
        sample_rate: int = 48000,
        speed_factor: float = 1.0,
        media_type: str = "wav",
        timeout: float = 30.0,
        streaming_mode: bool = False,
        top_k: int = 5,
        top_p: float = 1.0,
        temperature: float = 1.0,
        text_split_method: str = "cut5",
        batch_size: int = 1,
        batch_threshold: float = 0.75,
        split_bucket: bool = True,
        fragment_interval: float = 0.3,
        seed: int = -1,
        parallel_infer: bool = True,
        repetition_penalty: float = 1.35,
        sample_steps: int = 32,
        super_sampling: bool = False,
    ) -> None:
        if media_type != "wav":
            raise ValueError("GPTSoVITS client currently only supports wav responses")

        self._host = host
        self._port = port
        self.url = f"http://{host}:{port}/tts"

        self.audio_paths = self._expand_reference_paths(ref_audio_paths or [])
        self._base_audio_paths = self.audio_paths.copy()

        self._sample_rate = sample_rate
        self._timeout = timeout

        self._text_lang = text_lang.lower()
        self._prompt_lang = prompt_lang.lower()
        self._prompt_text = prompt_text
        self._speed_factor = speed_factor
        self._media_type = media_type
        self._streaming_mode = streaming_mode
        self._top_k = top_k
        self._top_p = top_p
        self._temperature = temperature
        self._text_split_method = text_split_method
        self._batch_size = batch_size
        self._batch_threshold = batch_threshold
        self._split_bucket = split_bucket
        self._fragment_interval = fragment_interval
        self._seed = seed
        self._parallel_infer = parallel_infer
        self._repetition_penalty = repetition_penalty
        self._sample_steps = sample_steps
        self._super_sampling = super_sampling

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    def _decode_audio(self, audio_bytes: bytes) -> bytes:
        audio, src_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if src_sr != self._sample_rate:
            audio = soxr.resample(audio, src_sr, self._sample_rate)
        return self._float32_to_pcm_bytes(audio)

    def _build_payload(
        self, text: str, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not self.audio_paths:
            raise ValueError("No reference audio paths configured for GPT-SoVITS")

        ref_audio_path = self.audio_paths[0]
        aux_paths = self.audio_paths[1:] or None

        payload: Dict[str, Any] = {
            "text": text,
            "text_lang": self._text_lang,
            "ref_audio_path": ref_audio_path,
            "prompt_text": self._prompt_text,
            "prompt_lang": self._prompt_lang,
            "media_type": self._media_type,
            "streaming_mode": self._streaming_mode,
            "speed_factor": self._speed_factor,
            "top_k": self._top_k,
            "top_p": self._top_p,
            "temperature": self._temperature,
            "text_split_method": self._text_split_method,
            "batch_size": self._batch_size,
            "batch_threshold": self._batch_threshold,
            "split_bucket": self._split_bucket,
            "fragment_interval": self._fragment_interval,
            "seed": self._seed,
            "parallel_infer": self._parallel_infer,
            "repetition_penalty": self._repetition_penalty,
            "sample_steps": self._sample_steps,
            "super_sampling": self._super_sampling,
        }

        if aux_paths:
            payload["aux_ref_audio_paths"] = aux_paths

        if overrides:
            payload.update({k: v for k, v in overrides.items() if v is not None})

        return payload

    def _expand_reference_paths(self, ref_audio_paths: List[str]) -> List[str]:
        """Accept files or directories and return resolved audio file list."""
        resolved: List[str] = []
        for ref_path in ref_audio_paths:
            path_obj = Path(ref_path)
            if path_obj.is_file():
                resolved.append(str(path_obj))
                continue
            if path_obj.is_dir():
                candidates = sorted(
                    str(p)
                    for p in path_obj.iterdir()
                    if p.is_file() and p.suffix.lower() in self._SUPPORTED_AUDIO_EXTS
                )
                if not candidates:
                    raise ValueError(
                        f"No audio files found in directory '{ref_path}' for GPT-SoVITS reference"
                    )
                resolved.extend(candidates)
                continue
            raise FileNotFoundError(
                f"Reference audio path '{ref_path}' does not exist for GPT-SoVITS"
            )
        return resolved

    def set_voice(self, voice_names: List[str]) -> None:
        self.audio_paths = self._expand_reference_paths(voice_names or [])
        self._base_audio_paths = self.audio_paths.copy()

    def clone(self) -> "GPTSoVITS":
        return GPTSoVITS(
            ref_audio_paths=self._base_audio_paths,
            text_lang=self._text_lang,
            prompt_lang=self._prompt_lang,
            prompt_text=self._prompt_text,
            host=self._host,
            port=self._port,
            sample_rate=self._sample_rate,
            speed_factor=self._speed_factor,
            media_type=self._media_type,
            timeout=self._timeout,
            streaming_mode=self._streaming_mode,
            top_k=self._top_k,
            top_p=self._top_p,
            temperature=self._temperature,
            text_split_method=self._text_split_method,
            batch_size=self._batch_size,
            batch_threshold=self._batch_threshold,
            split_bucket=self._split_bucket,
            fragment_interval=self._fragment_interval,
            seed=self._seed,
            parallel_infer=self._parallel_infer,
            repetition_penalty=self._repetition_penalty,
            sample_steps=self._sample_steps,
            super_sampling=self._super_sampling,
        )

    def synthesize(self, text: str, **overrides: Any) -> bytes:
        payload = self._build_payload(text, overrides)
        response = requests.post(self.url, json=payload, timeout=self._timeout)
        response.raise_for_status()
        return self._decode_audio(response.content)

    async def async_synthesize(self, text: str, **overrides: Any) -> bytes:
        payload = self._build_payload(text, overrides)
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self.url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                content = await resp.read()
        return self._decode_audio(content)
