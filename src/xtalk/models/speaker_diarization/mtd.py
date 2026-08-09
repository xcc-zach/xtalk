"""Unified HTTP client for official-vLLM and SGLang-Omni MTD runtimes."""

from __future__ import annotations

import asyncio
import io
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import quote

import aiohttp

from ..registry import model
from .interfaces import DiarizationResult, DiarizationSegment, SpeakerDiarization


_TIMESTAMP_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*"
    r"\[*\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\s*\[*\[S\d+\]|$))",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_SPEAKER_RE = re.compile(r"^\s*\[*\[S\d+\]", re.IGNORECASE)
_TRAILING_TIMESTAMP_RE = re.compile(r"\[(\d+(?:\.\d+)?)\]\s*$")

_Backend = Literal["official", "sglang_omni"]


@dataclass(frozen=True)
class _TimedSpeakerSegment:
    """One request-global segment parsed from an MTD response."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


class MtdRequestCancelled(RuntimeError):
    """Signal that an obsolete locally-cancellable MTD request was cancelled."""


@model
class OfficialMtdClient(SpeakerDiarization):
    """Call an official-vLLM or SGLang-Omni MTD HTTP runtime.

    The client discovers the backend once per session-local clone. A successful
    OpenAI-compatible ``GET /v1/models`` response selects SGLang-Omni and
    provides the model name submitted to ``/v1/audio/transcriptions``. A 404
    response selects the headless official-vLLM ``/v1/mtd/decode`` protocol.

    Parameters
    ----------
    base_url : str
        Official-vLLM or SGLang-Omni server root URL.
    request_timeout_s : float, optional
        Total HTTP timeout for one snapshot request.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum number of generated text tokens.
    """

    _INSTRUCTION = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
    )
    _RAW_PROMPT_TEMPLATE = (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "<|audio_start|><|audio_pad|><|audio_end|>\n"
        "{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    _RESPONSE_FORMAT = "verbose_json"

    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 15.0,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout_s = float(request_timeout_s)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._session: aiohttp.ClientSession | None = None
        self._backend: _Backend | None = None
        self._model_name: str | None = None
        self._discovery_lock = asyncio.Lock()
        self._requests: dict[str, asyncio.Task[tuple[dict[str, Any], str]]] = {}

    def clone(self) -> OfficialMtdClient:
        """Create a session-local client with the same public configuration."""

        return OfficialMtdClient(
            base_url=self.base_url,
            request_timeout_s=self.request_timeout_s,
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
        """Decode one snapshot with the protocol discovered from ``base_url``."""

        backend = await self._discover_backend()
        if backend == "official":
            return await self._decode_official(
                request_id=request_id,
                pcm16=pcm16,
                sample_rate=sample_rate,
                decoder_prefix=decoder_prefix,
                context_seconds=context_seconds,
                current_audio_seconds=current_audio_seconds,
                is_final=is_final,
            )
        return await self._decode_sglang(
            request_id=request_id,
            pcm16=pcm16,
            sample_rate=sample_rate,
            decoder_prefix=decoder_prefix,
            context_seconds=context_seconds,
            current_audio_seconds=current_audio_seconds,
        )

    async def _discover_backend(self) -> _Backend:
        """Discover the protocol and SGLang model name from the server."""

        if self._backend is not None:
            return self._backend
        async with self._discovery_lock:
            if self._backend is not None:
                return self._backend
            session = await self._get_session()
            async with session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 404:
                    self._backend = "official"
                    return self._backend
                response.raise_for_status()
                payload: dict[str, Any] = await response.json()
            self._model_name = _extract_model_name(payload)
            self._backend = "sglang_omni"
            return self._backend

    async def _decode_official(
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
        """Send one PCM snapshot to the headless official-vLLM runtime."""

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("request_id", request_id)
        form.add_field("sample_rate", str(sample_rate))
        form.add_field("decoder_prefix", decoder_prefix)
        form.add_field("context_seconds", str(context_seconds))
        form.add_field("current_audio_seconds", str(current_audio_seconds))
        form.add_field("is_final", "true" if is_final else "false")
        form.add_field("instruction", self._INSTRUCTION)
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
            f"{self.base_url}/v1/mtd/decode",
            data=form,
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

    async def _decode_sglang(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
        context_seconds: float,
        current_audio_seconds: float,
    ) -> DiarizationResult:
        """Decode one snapshot through the native SGLang-Omni endpoint."""

        started = time.perf_counter()
        request_task = asyncio.create_task(
            self._post_transcription(
                pcm16=pcm16,
                sample_rate=sample_rate,
                decoder_prefix=decoder_prefix,
            )
        )
        previous = self._requests.setdefault(request_id, request_task)
        if previous is not request_task:
            request_task.cancel()
            raise ValueError(f"duplicate SGLang-Omni request_id: {request_id}")
        try:
            payload, remote_request_id = await request_task
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise MtdRequestCancelled(
                f"SGLang-Omni request was cancelled: {request_id}"
            ) from exc
        finally:
            if self._requests.get(request_id) is request_task:
                self._requests.pop(request_id, None)

        generated_suffix = str(payload.get("text") or "").strip()
        raw_text = _join_decoder_prefix_and_suffix(decoder_prefix, generated_suffix)
        raw_segments = _parse_timestamped_text(raw_text)
        current_segments = _crop_current_segments(
            raw_segments,
            context_seconds=max(0.0, float(context_seconds)),
            current_audio_seconds=max(0.0, float(current_audio_seconds)),
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return DiarizationResult(
            raw_text=raw_text,
            current_segments=current_segments,
            latency_ms=latency_ms,
            metrics={
                "backend": "sglang_omni",
                "request_id": request_id,
                "remote_request_id": remote_request_id,
                "server_duration_s": payload.get("duration"),
                "usage": payload.get("usage"),
                "registration_mode": "fixed_decoder_prefix",
                "decoder_prefix_chars": len(decoder_prefix),
                "generated_suffix": generated_suffix,
            },
        )

    async def cancel(self, request_id: str) -> None:
        """Cancel an official request remotely or an SGLang request locally."""

        backend = await self._discover_backend()
        if backend == "official":
            session = await self._get_session()
            url = f"{self.base_url}/v1/mtd/requests/{quote(request_id, safe='')}"
            async with session.delete(url) as response:
                if response.status not in {200, 202, 204, 404}:
                    response.raise_for_status()
            return
        request_task = self._requests.get(request_id)
        if request_task is not None and not request_task.done():
            request_task.cancel()

    async def close(self) -> None:
        """Cancel outstanding requests and close the HTTP session."""

        for request_task in tuple(self._requests.values()):
            request_task.cancel()
        if self._requests:
            await asyncio.gather(*self._requests.values(), return_exceptions=True)
        self._requests.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _post_transcription(
        self,
        *,
        pcm16: bytes,
        sample_rate: int,
        decoder_prefix: str,
    ) -> tuple[dict[str, Any], str]:
        """Post one WAV snapshot with the automatically discovered model name."""

        if self._model_name is None:
            raise RuntimeError("SGLang-Omni model discovery has not completed")
        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field(
            "file",
            _pcm16_to_wav(pcm16, sample_rate),
            filename="snapshot.wav",
            content_type="audio/wav",
        )
        form.add_field("model", self._model_name)
        form.add_field("prompt", self._build_forced_prefix_prompt(decoder_prefix))
        form.add_field("response_format", self._RESPONSE_FORMAT)
        form.add_field("temperature", str(self.temperature))
        form.add_field("max_new_tokens", str(self.max_tokens))
        async with session.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data=form,
        ) as response:
            response.raise_for_status()
            payload: dict[str, Any] = await response.json()
            remote_request_id = response.headers.get("X-Request-Id", "")
        return payload, remote_request_id

    def _build_forced_prefix_prompt(self, decoder_prefix: str) -> str:
        """Build the complete SGLang prompt with a fixed decoder prefix."""

        return (
            self._RAW_PROMPT_TEMPLATE.format(instruction=self._INSTRUCTION)
            + decoder_prefix
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a live HTTP session for the current event loop."""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


def _extract_model_name(payload: object) -> str:
    """Extract the first non-empty model ID from an OpenAI model-list response."""

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                model_name = str(item.get("id") or "").strip()
                if model_name:
                    return model_name
    raise RuntimeError("GET /v1/models returned no usable model ID")


def _normalize_segments(value: object) -> list[DiarizationSegment]:
    """Normalize official-runtime JSON into the public segment contract."""

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


def _pcm16_to_wav(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap mono little-endian PCM16 in a standard WAV container."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if len(pcm16) % 2:
        raise ValueError("PCM16 payload must contain complete two-byte samples")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16)
    return output.getvalue()


def _parse_timestamped_text(text: str) -> list[_TimedSpeakerSegment]:
    """Parse native MTD timestamp-plus-speaker text."""

    result: list[_TimedSpeakerSegment] = []
    for match in _TIMESTAMP_SEGMENT_RE.finditer(text):
        start_s = float(match.group("start"))
        end_s = float(match.group("end"))
        if end_s <= start_s:
            continue
        result.append(
            _TimedSpeakerSegment(
                start_s=max(0.0, start_s),
                end_s=end_s,
                speaker_id=match.group("speaker").upper(),
                text=re.sub(r"\s+", " ", match.group("text")).strip(),
            )
        )
    return result


def _join_decoder_prefix_and_suffix(decoder_prefix: str, suffix: str) -> str:
    """Reconstruct a parseable full timeline at a continuation boundary."""

    prefix = decoder_prefix.strip()
    suffix = suffix.strip()
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    if _LEADING_SPEAKER_RE.match(suffix):
        boundary = _TRAILING_TIMESTAMP_RE.search(prefix)
        if boundary is not None:
            suffix = f"[{boundary.group(1)}]{suffix}"
    return f"{prefix} {suffix}"


def _crop_current_segments(
    segments: list[_TimedSpeakerSegment],
    *,
    context_seconds: float,
    current_audio_seconds: float,
) -> list[DiarizationSegment]:
    """Remove registered-audio time and retain decoder-produced speaker IDs."""

    current_end_s = context_seconds + current_audio_seconds
    result: list[DiarizationSegment] = []
    for segment in segments:
        start_s = max(segment.start_s, context_seconds)
        end_s = min(segment.end_s, current_end_s)
        if end_s <= start_s:
            continue
        result.append(
            {
                "start_s": round(max(0.0, start_s - context_seconds), 6),
                "end_s": round(max(0.0, end_s - context_seconds), 6),
                "speaker_id": segment.speaker_id,
                "text": segment.text,
            }
        )
    return result
