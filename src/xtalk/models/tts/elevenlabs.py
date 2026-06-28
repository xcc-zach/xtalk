import os
from typing import Any, AsyncIterator, Literal, Optional
import requests
import aiohttp
from .interfaces import TTS
from ..registry import model


@model
class ElevenLabsTTS(TTS):
    """
    ElevenLabs Text-to-Speech implementation.

    This class provides text-to-speech functionality using ElevenLabs API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: Literal[
            "eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"
        ] = "eleven_multilingual_v2",
    ):
        """
        Initialize ElevenLabsTTS.

        Args:
            api_key (Optional[str]): ElevenLabs API key. If None, will load from environment
            voice_id (str): Voice ID to use for synthesis
            model_id (str): Model ID to use. Options: "eleven_multilingual_v2", "eleven_flash_v2_5", "eleven_turbo_v2_5"
            output_format (str): Output audio format
            sample_rate (int): Sample rate of the output audio
        """

        # Use provided API key or load from environment
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable or pass api_key parameter."
            )

        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = "pcm_48000"
        self._sample_rate = 48000

    def clone(self):
        return ElevenLabsTTS(
            api_key=self.api_key,
            voice_id=self.voice_id,
            model_id=self.model_id,
        )

    def synthesize(self, text: str) -> bytes:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            text (str): The text to convert to speech

        Returns:
            bytes: The synthesized speech as audio data
        """
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {"output_format": self.output_format} if self.output_format else None
        payload = {"text": text, "model_id": self.model_id}

        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
            return response.content
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to synthesize speech with ElevenLabs: {exc}"
            ) from exc

    def synthesize_stream(self, text: str):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {"output_format": self.output_format} if self.output_format else None
        payload = {"text": text, "model_id": self.model_id}

        try:
            response = requests.post(
                url,
                headers=headers,
                params=params,
                json=payload,
                stream=True,
                timeout=30,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Failed to stream speech with ElevenLabs: {exc}"
            ) from exc

        # Read audio chunks from the HTTP stream as they arrive
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                yield chunk

    async def async_synthesize(self, text: str, **_: Any) -> bytes:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {"output_format": self.output_format} if self.output_format else None
        payload = {"text": text, "model_id": self.model_id}
        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    return await resp.read()
        except aiohttp.ClientError as exc:
            raise RuntimeError(
                f"Failed to synthesize speech with ElevenLabs: {exc}"
            ) from exc

    async def async_synthesize_stream(
        self, text: str, **_: Any
    ) -> AsyncIterator[bytes]:
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        params = {"output_format": self.output_format} if self.output_format else None
        payload = {"text": text, "model_id": self.model_id}
        timeout = aiohttp.ClientTimeout(total=30)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url,
                    headers=headers,
                    params=params,
                    json=payload,
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {body[:200]}")
                    async for chunk in resp.content.iter_chunked(4096):
                        if chunk:
                            yield chunk
        except aiohttp.ClientError as exc:
            raise RuntimeError(
                f"Failed to stream speech with ElevenLabs: {exc}"
            ) from exc
