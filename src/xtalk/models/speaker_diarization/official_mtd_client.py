"""HTTP client for the headless official-vLLM MTD runtime."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import quote

import aiohttp

from ..registry import model
from .interfaces import DiarizationResult, DiarizationSegment, SpeakerDiarization


@model
class OfficialMtdClient(SpeakerDiarization):
    """Call a thin runtime built on the official MTD vLLM implementation.

    Parameters
    ----------
    base_url : str
        Runtime root URL.
    request_timeout_s : float, optional
        Total HTTP timeout for one snapshot request.
    instruction : str, optional
        Timestamp-and-speaker transcription instruction sent to the runtime.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum completion length.
    """

    DEFAULT_INSTRUCTION = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
    )

    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
        instruction: str = DEFAULT_INSTRUCTION,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self.instruction = instruction
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._session: aiohttp.ClientSession | None = None

    def clone(self) -> "OfficialMtdClient":
        """Create a session-local HTTP client with the same configuration."""

        return OfficialMtdClient(
            base_url=self.base_url,
            request_timeout_s=self.request_timeout_s,
            instruction=self.instruction,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    async def decode_snapshot(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
        context_seconds: float,
        current_audio_seconds: float,
        is_final: bool,
    ) -> DiarizationResult:
        """Send one PCM snapshot to the MTD runtime."""

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("request_id", request_id)
        form.add_field("sample_rate", str(sample_rate))
        form.add_field("decoder_prefix", decoder_prefix)
        form.add_field("context_seconds", str(context_seconds))
        form.add_field("current_audio_seconds", str(current_audio_seconds))
        form.add_field("is_final", "true" if is_final else "false")
        form.add_field("instruction", self.instruction)
        form.add_field("temperature", str(self.temperature))
        form.add_field("max_tokens", str(self.max_tokens))
        form.add_field(
            "audio",
            pcm16,
            filename="snapshot.pcm",
            content_type="audio/pcm",
        )
        started = time.perf_counter()
        async with session.post(
            f"{self.base_url}/v1/mtd/decode", data=form
        ) as response:
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()
        latency_ms = float(payload.get("latency_ms") or 0.0)
        if latency_ms <= 0:
            latency_ms = (time.perf_counter() - started) * 1000.0
        return DiarizationResult(
            raw_text=str(payload.get("raw_text") or ""),
            current_segments=_normalize_segments(payload.get("current_segments")),
            latency_ms=latency_ms,
            metrics=dict(payload.get("metrics") or {}),
        )

    async def cancel(self, request_id: str) -> None:
        """Ask the runtime to abort a request without waiting for completion."""

        session = await self._get_session()
        url = f"{self.base_url}/v1/mtd/requests/{quote(request_id, safe='') }"
        async with session.delete(url) as response:
            if response.status not in {200, 202, 204, 404}:
                response.raise_for_status()

    async def close(self) -> None:
        """Close the lazily-created HTTP session."""

        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a live HTTP session for the current event loop."""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


def _normalize_segments(value: object) -> list[DiarizationSegment]:
    """Normalize runtime JSON into the public diarization segment contract."""

    if not isinstance(value, list):
        return []
    segments: list[DiarizationSegment] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        try:
            start_s = float(item["start_s"])
            end_s = float(item["end_s"])
            speaker_id = str(item["speaker_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if end_s <= start_s:
            continue
        segments.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "speaker_id": speaker_id,
                "text": str(item.get("text") or ""),
            }
        )
    return segments
