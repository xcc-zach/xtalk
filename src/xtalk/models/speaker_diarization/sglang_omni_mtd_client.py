"""HTTP client for the native SGLang-Omni MTD transcription endpoint."""

from __future__ import annotations

import asyncio
import io
import re
import time
import wave
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class _TimedSpeakerSegment:
    """One request-global segment parsed from an MTD response."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


class SglangOmniRequestCancelled(RuntimeError):
    """Signal that an obsolete SGLang-Omni partial was cancelled locally."""


@model
class SglangOmniMtdClient(SpeakerDiarization):
    """Call the native SGLang-Omni MTD transcription API.

    The native endpoint forwards a prompt containing ``<|audio_pad|>`` without
    applying its own chat template. This client uses that existing behavior to
    submit the complete MTD template and append ``decoder_prefix`` immediately
    after the assistant header. Registered speaker labels are therefore fixed
    decoder context, rather than labels recovered by post-hoc overlap mapping.

    Parameters
    ----------
    base_url : str
        SGLang-Omni server root URL.
    model : str, optional
        Model name submitted to ``/v1/audio/transcriptions``. An empty value
        lets the server use its configured default model.
    request_timeout_s : float, optional
        Total HTTP timeout for one full-snapshot request.
    instruction : str, optional
        Timestamp-and-speaker transcription prompt.
    temperature : float, optional
        Sampling temperature.
    max_tokens : int, optional
        Maximum number of generated text tokens.
    response_format : str, optional
        OpenAI transcription response format. ``verbose_json`` is required for
        parsing and current-audio timestamp cropping.
    exemplar_match_min_overlap_s : float, optional
        Deprecated compatibility setting from the former time-slot matching
        implementation. It is accepted but has no effect in fixed-prefix mode.
    """

    DEFAULT_INSTRUCTION = (
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

    def __init__(
        self,
        base_url: str,
        model: str = "",
        request_timeout_s: float = 15.0,
        instruction: str = DEFAULT_INSTRUCTION,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: str = "verbose_json",
        exemplar_match_min_overlap_s: float | None = None,
    ) -> None:
        if response_format != "verbose_json":
            raise ValueError(
                "SglangOmniMtdClient requires response_format='verbose_json'"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout_s = float(request_timeout_s)
        self.instruction = instruction
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.response_format = response_format
        # Keep legacy deployment configurations loadable. Speaker registration
        # is now decoder-side, so overlap matching is intentionally unused.
        del exemplar_match_min_overlap_s
        self._session: aiohttp.ClientSession | None = None
        self._requests: dict[str, asyncio.Task[tuple[dict[str, Any], str]]] = {}

    def clone(self) -> "SglangOmniMtdClient":
        """Create a session-local HTTP client with the same configuration.

        Returns
        -------
        SglangOmniMtdClient
            Client without a shared HTTP session or in-flight request state.
        """

        return SglangOmniMtdClient(
            base_url=self.base_url,
            model=self.model,
            request_timeout_s=self.request_timeout_s,
            instruction=self.instruction,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=self.response_format,
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
        """Decode one audio snapshot using the registered global speaker IDs.

        Parameters
        ----------
        request_id : str
            X-Talk request identifier used for tracing and local cancellation.
        pcm16 : bytes
            Mono PCM16 containing registered exemplars and current audio.
        sample_rate : int
            PCM sampling rate in hertz.
        decoder_prefix : str
            Timestamped global-speaker transcript forced after the MTD assistant
            header. It describes the registered audio prefix.
        context_seconds : float
            Duration before the current VAD audio starts.
        current_audio_seconds : float
            Duration of the current VAD snapshot.
        is_final : bool
            Whether the snapshot closes its VAD segment.

        Returns
        -------
        DiarizationResult
            Current-audio-local segments carrying MTD's fixed global IDs.
        """

        del is_final
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
            raise SglangOmniRequestCancelled(
                f"SGLang-Omni request was cancelled: {request_id}"
            ) from exc
        finally:
            if self._requests.get(request_id) is request_task:
                self._requests.pop(request_id, None)

        generated_suffix = str(payload.get("text") or "").strip()
        raw_text = _join_decoder_prefix_and_suffix(decoder_prefix, generated_suffix)
        # ``verbose_json.segments`` covers only newly generated tokens. Parse
        # the reconstructed full text so prefix and output share one timeline.
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
        """Cancel the local HTTP task for an obsolete partial request.

        Parameters
        ----------
        request_id : str
            Identifier previously passed to :meth:`decode_snapshot`.

        Notes
        -----
        Closing the HTTP request is best effort. The native transcription API
        does not currently expose a public request-abort endpoint.
        """

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
        """Post one WAV snapshot and return its JSON body and server request ID."""

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field(
            "file",
            _pcm16_to_wav(pcm16, sample_rate),
            filename="snapshot.wav",
            content_type="audio/wav",
        )
        if self.model:
            form.add_field("model", self.model)
        form.add_field("prompt", self._build_forced_prefix_prompt(decoder_prefix))
        form.add_field("response_format", self.response_format)
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
        """Build a raw MTD prompt with a decoder-side fixed transcript prefix.

        SGLang-Omni treats prompts containing ``<|audio_pad|>`` as complete
        prompts and forwards them unchanged. The prefix must consequently be
        placed *after* the assistant header, never in the user instruction.
        """

        return (
            self._RAW_PROMPT_TEMPLATE.format(instruction=self.instruction)
            + decoder_prefix
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a live HTTP session for the current event loop."""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


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
    """Reconstruct a parseable full timeline at a continuation boundary.

    SGLang can generate ``[S02]text[end]`` immediately after a fixed prefix
    ending in ``[start]``. In token space that terminal prefix timestamp is
    also the new segment's start timestamp. Duplicate it in the reconstructed
    diagnostic text so the non-overlapping segment parser can retain both the
    preceding exemplar and the generated continuation.
    """

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
