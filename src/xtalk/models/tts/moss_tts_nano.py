"""HTTP client adapter for MOSS-TTS-Nano services."""

from __future__ import annotations

import base64
import binascii
from array import array
import io
import mimetypes
from pathlib import Path
import sys
from typing import Dict, List, Optional
from urllib.parse import urlparse
import wave

import aiohttp
import requests

from ..registry import model
from .interfaces import TTS


_DEFAULT_BASE_URL = "http://127.0.0.1:18083"
_SYNTHESIS_TIMEOUT_SECONDS = 300.0
_SYNTHESIS_ATTEMPT_LIMIT = 2


def _resolve_base_url(base_url: str) -> str:
    """Validate and normalize a MOSS-TTS-Nano HTTP base URL."""
    normalized = str(base_url).strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must not be empty")
    parsed = urlparse(normalized)
    if not parsed.scheme:
        normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MOSS-TTS-Nano base_url must use http:// or https://")
    for endpoint in ("/v1/audio/speech", "/api/generate"):
        if normalized.endswith(endpoint):
            normalized = normalized[: -len(endpoint)]
            break
    return normalized.rstrip("/")


def _validate_voices(
    voices: Optional[List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    """Copy and validate IndexTTS-compatible voice configuration."""
    normalized: List[Dict[str, str]] = []
    names: set[str] = set()
    for index, voice in enumerate(voices or []):
        if not isinstance(voice, dict):
            raise TypeError(f"voices[{index}] must be a dictionary")
        name = voice.get("name")
        path = voice.get("path")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"voices[{index}].name must be a non-empty string")
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"voices[{index}].path must be a non-empty string")
        normalized_name = name.strip()
        if normalized_name in names:
            raise ValueError(f"Duplicate voice name: {normalized_name}")
        names.add(normalized_name)
        normalized.append({"name": normalized_name, "path": path.strip()})
    return normalized


def _decode_wav_to_pcm48(wav_bytes: bytes) -> bytes:
    """Decode WAV bytes and return 48 kHz mono PCM16."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise RuntimeError("MOSS service must return uncompressed PCM16 WAV")
        if wav_file.getframerate() != 48_000:
            raise RuntimeError(
                "MOSS service must return audio at 48000 Hz, got "
                f"{wav_file.getframerate()} Hz"
            )
        channels = wav_file.getnchannels()
        if channels <= 0:
            raise RuntimeError("MOSS service returned WAV with no channels")
        frames = wav_file.readframes(wav_file.getnframes())
    if channels == 1:
        return frames

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    mono = array(
        "h",
        (
            round(sum(frame) / channels)
            for frame in zip(
                *(samples[channel::channels] for channel in range(channels)),
                strict=True,
            )
        ),
    )
    if sys.byteorder == "big":
        mono.byteswap()
    return mono.tobytes()


def _decode_service_payload(payload: object) -> bytes:
    """Extract WAV bytes from the shared service response payload."""
    if not isinstance(payload, dict):
        raise RuntimeError("MOSS service returned a non-object response")
    encoded = payload.get("audio_base64")
    if not isinstance(encoded, str) or not encoded:
        error = payload.get("error")
        if isinstance(error, str) and error:
            raise RuntimeError(error)
        raise RuntimeError("MOSS service response has no audio_base64")
    try:
        return base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RuntimeError("MOSS service returned invalid base64 audio") from exc


@model
class MossTTSNano(TTS):
    """Client for the shared Python and native Rust MOSS-TTS-Nano protocol.

    Parameters
    ----------
    base_url : str, optional
        Base URL of the MOSS-TTS-Nano HTTP service.
    voices : list[dict[str, str]] | None, optional
        IndexTTS-compatible voice configurations containing ``name`` and
        ``path``. The selected reference audio is uploaded to ``/api/generate``.
    """

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        voices: Optional[List[Dict[str, str]]] = None,
    ):
        """Initialize the MOSS-TTS-Nano HTTP client."""
        self.base_url = _resolve_base_url(base_url)
        self._voices = _validate_voices(voices)
        self._voice_path_map = {
            voice["name"]: voice["path"] for voice in self._voices
        }
        self._active_voice_name = (
            self._voices[0]["name"] if self._voices else None
        )

    def clone(self) -> "MossTTSNano":
        """Create a session-safe clone of this client."""
        clone = MossTTSNano(
            base_url=self.base_url,
            voices=[voice.copy() for voice in self._voices],
        )
        clone._active_voice_name = self._active_voice_name
        return clone

    def set_voice(self, voice_names: List[str]) -> None:
        """Select exactly one configured reference voice."""
        if len(voice_names) != 1:
            raise ValueError("MossTTSNano accepts exactly one voice")
        voice_name = str(voice_names[0]).strip()
        if not voice_name:
            raise ValueError("voice name must not be empty")
        if voice_name not in self._voice_path_map:
            raise ValueError(f"Unknown voice name: {voice_name}")
        self._active_voice_name = voice_name

    def synthesize(self, text: str) -> bytes:
        """Synthesize text and return 48 kHz mono PCM16 bytes."""
        normalized_text = self._validate_text(text)
        for _ in range(_SYNTHESIS_ATTEMPT_LIMIT):
            wav_bytes = self._synthesize_service(normalized_text)
            pcm = _decode_wav_to_pcm48(wav_bytes)
            if pcm:
                return pcm
        raise RuntimeError(
            "MOSS service returned no audio data after "
            f"{_SYNTHESIS_ATTEMPT_LIMIT} attempts"
        )

    async def async_synthesize(self, text: str, **kwargs: object) -> bytes:
        """Asynchronously synthesize text into 48 kHz mono PCM16 bytes."""
        del kwargs
        normalized_text = self._validate_text(text)
        timeout = aiohttp.ClientTimeout(total=_SYNTHESIS_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for _ in range(_SYNTHESIS_ATTEMPT_LIMIT):
                wav_bytes = await self._async_synthesize_service(
                    session, normalized_text
                )
                pcm = _decode_wav_to_pcm48(wav_bytes)
                if pcm:
                    return pcm
        raise RuntimeError(
            "MOSS service returned no audio data after "
            f"{_SYNTHESIS_ATTEMPT_LIMIT} attempts"
        )

    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate and normalize one synthesis text."""
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized

    def _voice_path(self) -> Path:
        """Return and validate the selected reference-audio path."""
        if self._active_voice_name is None:
            raise ValueError(
                "voices must configure a reference audio path"
            )
        voice_path = self._voice_path_map.get(self._active_voice_name)
        if voice_path is None:
            raise ValueError(
                f"Voice '{self._active_voice_name}' has no reference audio path"
            )
        resolved = Path(voice_path).expanduser()
        if not resolved.is_file():
            raise ValueError(f"Reference audio does not exist: {resolved}")
        return resolved

    def _synthesize_service(self, text: str) -> bytes:
        """Call the shared multipart generation endpoint."""
        voice_path = self._voice_path()
        media_type = mimetypes.guess_type(voice_path.name)[0] or "application/octet-stream"
        with voice_path.open("rb") as voice_file:
            response = requests.post(
                f"{self.base_url}/api/generate",
                data={"text": text},
                files={
                    "prompt_audio": (
                        voice_path.name,
                        voice_file,
                        media_type,
                    )
                },
                timeout=_SYNTHESIS_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        return _decode_service_payload(response.json())

    async def _async_synthesize_service(
        self,
        session: aiohttp.ClientSession,
        text: str,
    ) -> bytes:
        """Call the shared multipart generation endpoint asynchronously."""
        voice_path = self._voice_path()
        media_type = mimetypes.guess_type(voice_path.name)[0] or "application/octet-stream"
        form = aiohttp.FormData()
        form.add_field("text", text)
        with voice_path.open("rb") as voice_file:
            form.add_field(
                "prompt_audio",
                voice_file,
                filename=voice_path.name,
                content_type=media_type,
            )
            async with session.post(
                f"{self.base_url}/api/generate",
                data=form,
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {body}")
                payload = await response.json()
        return _decode_service_payload(payload)
