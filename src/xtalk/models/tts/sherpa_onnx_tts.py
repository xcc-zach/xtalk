"""HTTP client adapter for sherpa-onnx Matcha TTS services."""

from __future__ import annotations

from array import array
import io
import sys
from urllib.parse import urlparse
import wave

import aiohttp
import numpy as np
import requests

from ..registry import model
from .interfaces import TTS


_SPEECH_ENDPOINT = "/v1/audio/speech"
_SYNTHESIS_TIMEOUT_SECONDS = 300.0
_OUTPUT_SAMPLE_RATE = 48_000


def _resolve_base_url(base_url: str) -> str:
    """Validate and normalize a sherpa-onnx TTS HTTP base URL.

    Parameters
    ----------
    base_url : str
        Service base URL or full speech endpoint URL.

    Returns
    -------
    str
        Normalized HTTP service base URL without a trailing slash.
    """

    normalized = str(base_url).strip().rstrip("/")
    if not normalized:
        raise ValueError("base_url must not be empty")
    parsed = urlparse(normalized)
    if not parsed.scheme:
        normalized = f"http://{normalized}"
        parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("SherpaOnnxTTS base_url must use http:// or https://")
    if normalized.endswith(_SPEECH_ENDPOINT):
        normalized = normalized[: -len(_SPEECH_ENDPOINT)]
    return normalized.rstrip("/")


def _request_payload(text: str) -> dict[str, object]:
    """Build the fixed Matcha synthesis request payload.

    Parameters
    ----------
    text : str
        Non-empty synthesis text.

    Returns
    -------
    dict[str, object]
        JSON body accepted by the local sherpa-onnx HTTP service.
    """

    return {
        "model": "matcha-icefall-zh-en",
        "input": text,
        "voice": "0",
        "response_format": "wav",
        "speed": 1.0,
        "sample_rate": _OUTPUT_SAMPLE_RATE,
    }


def _decode_wav_to_pcm48(wav_bytes: bytes) -> bytes:
    """Decode a PCM16 WAV payload into 48 kHz mono PCM16 bytes.

    Parameters
    ----------
    wav_bytes : bytes
        Complete WAV response returned by the service.

    Returns
    -------
    bytes
        Raw 48 kHz mono PCM16 audio.

    Raises
    ------
    RuntimeError
        If the response is not uncompressed PCM16 audio or contains no audio.
    """

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
                raise RuntimeError(
                    "sherpa-onnx service must return uncompressed PCM16 WAV"
                )
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error) as exc:
        raise RuntimeError("sherpa-onnx service returned invalid WAV audio") from exc

    if channels <= 0 or sample_rate <= 0 or not frames:
        raise RuntimeError("sherpa-onnx service returned no audio data")

    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":
        samples.byteswap()
    audio = np.asarray(samples, dtype=np.float32).reshape(-1, channels)
    mono = audio.mean(axis=1)
    if sample_rate != _OUTPUT_SAMPLE_RATE:
        output_size = max(
            1,
            round(mono.size * _OUTPUT_SAMPLE_RATE / sample_rate),
        )
        mono = np.interp(
            np.linspace(0, max(0, mono.size - 1), output_size),
            np.arange(mono.size),
            mono,
        )
    output = np.clip(np.rint(mono), -32768, 32767).astype("<i2")
    if output.size == 0:
        raise RuntimeError("sherpa-onnx service returned no audio data")
    return output.tobytes()


@model
class SherpaOnnxTTS(TTS):
    """Client for the local sherpa-onnx Matcha Chinese-English TTS service.

    Parameters
    ----------
    base_url : str
        HTTP base URL of the local service. A full
        ``/v1/audio/speech`` endpoint is also accepted.
    """

    def __init__(self, base_url: str):
        """Initialize the sherpa-onnx HTTP client."""

        self.base_url = _resolve_base_url(base_url)
        self.url = f"{self.base_url}{_SPEECH_ENDPOINT}"

    def clone(self) -> "SherpaOnnxTTS":
        """Create a session-safe client clone.

        Returns
        -------
        SherpaOnnxTTS
            New client using the same local service URL.
        """

        return SherpaOnnxTTS(self.base_url)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text into 48 kHz mono PCM16 bytes.

        Parameters
        ----------
        text : str
            Chinese, English, or mixed-language text.

        Returns
        -------
        bytes
            Raw 48 kHz mono PCM16 audio.
        """

        normalized_text = self._validate_text(text)
        response = requests.post(
            self.url,
            json=_request_payload(normalized_text),
            timeout=_SYNTHESIS_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _decode_wav_to_pcm48(response.content)

    async def async_synthesize(self, text: str, **kwargs: object) -> bytes:
        """Asynchronously synthesize text into 48 kHz mono PCM16 bytes.

        Parameters
        ----------
        text : str
            Chinese, English, or mixed-language text.
        **kwargs : object
            Reserved for compatibility with the base TTS API.

        Returns
        -------
        bytes
            Raw 48 kHz mono PCM16 audio.
        """

        del kwargs
        normalized_text = self._validate_text(text)
        timeout = aiohttp.ClientTimeout(total=_SYNTHESIS_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url,
                json=_request_payload(normalized_text),
            ) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(f"HTTP {response.status}: {body}")
                wav_bytes = await response.read()
        return _decode_wav_to_pcm48(wav_bytes)

    @staticmethod
    def _validate_text(text: str) -> str:
        """Validate and normalize one synthesis input.

        Parameters
        ----------
        text : str
            Candidate synthesis text.

        Returns
        -------
        str
            Stripped non-empty text.
        """

        normalized = str(text).strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized
