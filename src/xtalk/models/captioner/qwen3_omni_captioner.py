from .interfaces import Captioner
from ..registry import model
from typing import Iterable, Optional, AsyncIterator
import requests
import aiohttp
import json
import base64
import mimetypes
import io
import wave


@model
class Qwen3OmniCaptioner(Captioner):
    """Qwen3-Omni audio captioner via ChatCompletions."""

    def __init__(
        self,
        base_url: str = "http://localhost:8901/v1",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        # Base service URL (e.g., http://localhost:8901/v1)
        self.base_url = base_url.rstrip("/")
        # Optional API key
        self.api_key = api_key
        # Request timeout in seconds
        self.timeout = timeout

    def _build_payload(self, audio_url: str) -> dict:
        """Build Chat Completions payload with a single audio URL message."""
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": audio_url},
                        }
                    ],
                }
            ]
        }

    def _post(self, payload: dict) -> dict:
        """Send POST request and return JSON; raise on non-200."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text}")
        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(f"Invalid JSON response: {e}; body={resp.text}")

    async def _post_async(self, payload: dict) -> dict:
        """Async POST helper."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}: {body}")
                try:
                    return json.loads(body)
                except Exception as e:
                    raise RuntimeError(f"Invalid JSON response: {e}; body={body}")

    def _extract_text(self, response_json: dict) -> str:
        """Extract text from ChatCompletions-style response."""
        try:
            choices = response_json.get("choices") or []
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
        except Exception:
            pass
        # Fallback: return raw JSON string
        return json.dumps(response_json, ensure_ascii=False)

    def _to_data_url(self, data: bytes, filename: Optional[str] = None) -> str:
        """Convert audio bytes to data URL (e.g., data:audio/wav;base64,...)."""
        mime, _ = mimetypes.guess_type(filename or "")
        if mime is None:
            mime = "audio/wav"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def _pcm_s16le_to_wav(
        self, pcm: bytes, sample_rate: int = 16000, channels: int = 1
    ) -> bytes:
        """Wrap raw PCM s16le bytes into a WAV container."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return buf.getvalue()

    def caption(self, audio: bytes) -> str:
        """Generate a caption for PCM 16-bit mono 16 kHz bytes."""
        if not isinstance(audio, (bytes, bytearray)):
            raise TypeError("audio must be bytes (PCM s16le mono 16 kHz)")

        # Wrap as WAV for backend parsing
        wav_bytes = self._pcm_s16le_to_wav(bytes(audio), sample_rate=16000, channels=1)
        data_url = self._to_data_url(wav_bytes, filename="audio.wav")
        payload = self._build_payload(data_url)
        resp_json = self._post(payload)
        return self._extract_text(resp_json)

    def caption_stream(self, audio: bytes) -> Iterable[str]:
        """Simple streaming API that yields caption() result once."""
        yield self.caption(audio)

    async def async_caption(self, audio: bytes) -> str:
        if not isinstance(audio, (bytes, bytearray)):
            raise TypeError("audio must be bytes (PCM s16le mono 16 kHz)")

        wav_bytes = self._pcm_s16le_to_wav(bytes(audio), sample_rate=16000, channels=1)
        data_url = self._to_data_url(wav_bytes, filename="audio.wav")
        payload = self._build_payload(data_url)
        resp_json = await self._post_async(payload)
        return self._extract_text(resp_json)

    async def async_caption_stream(self, audio: bytes) -> AsyncIterator[str]:
        yield await self.async_caption(audio)
