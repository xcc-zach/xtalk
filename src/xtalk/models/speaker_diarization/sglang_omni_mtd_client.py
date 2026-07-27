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
from .interfaces import DiarizationResult, SpeakerDiarization


_TIMESTAMP_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\s*"
    r"\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]"
    r"(?=\s*(?:\[\d+(?:\.\d+)?\]\s*\[S\d+\]|$))",
    re.IGNORECASE | re.DOTALL,
)
_SPEAKER_PREFIX_RE = re.compile(r"^\s*\[(?P<speaker>S\d+)\]\s*", re.IGNORECASE)


@dataclass(frozen=True)
class _TimedSpeakerSegment:
    """One request-global segment parsed from an MTD response."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


@dataclass(frozen=True)
class _ExemplarSlot:
    """Time range occupied by one registered global-speaker exemplar."""

    start_s: float
    end_s: float
    speaker_id: str


class SglangOmniRequestCancelled(RuntimeError):
    """Signal that an obsolete SGLang-Omni partial was cancelled locally."""


@model
class SglangOmniMtdClient(SpeakerDiarization):
    """Call the native SGLang-Omni MTD transcription API.

    SGLang-Omni does not expose a fixed decoder-completion prefix through its
    OpenAI-compatible transcription endpoint. X-Talk still sends the existing
    exemplar-plus-current audio layout, then this client uses the exemplar time
    slots to map request-local MTD labels back to session-global speaker IDs.

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
        timestamp-based exemplar matching.
    exemplar_match_min_overlap_s : float, optional
        Minimum aggregate overlap required to bind one local label to one
        registered global speaker.
    """

    DEFAULT_INSTRUCTION = (
        "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
        "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
        "并在段末标注结束时间戳，以清晰标明该段语音范围。"
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
        exemplar_match_min_overlap_s: float = 0.05,
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
        self.exemplar_match_min_overlap_s = float(exemplar_match_min_overlap_s)
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
            exemplar_match_min_overlap_s=self.exemplar_match_min_overlap_s,
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
        """Decode and convert request-local labels to global speaker IDs.

        Parameters
        ----------
        request_id : str
            X-Talk request identifier used for tracing and local cancellation.
        pcm16 : bytes
            Mono PCM16 containing registered exemplars and current audio.
        sample_rate : int
            PCM sampling rate in hertz.
        decoder_prefix : str
            Timestamped transcript describing the global-speaker exemplar slots.
        context_seconds : float
            Duration before the current VAD audio starts.
        current_audio_seconds : float
            Duration of the current VAD snapshot.
        is_final : bool
            Whether the snapshot closes its VAD segment.

        Returns
        -------
        DiarizationResult
            Current-audio-local segments carrying session-global speaker IDs.
        """

        del is_final
        started = time.perf_counter()
        request_task = asyncio.create_task(
            self._post_transcription(
                pcm16=pcm16,
                sample_rate=sample_rate,
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

        raw_text = str(payload.get("text") or "")
        raw_segments = _parse_response_segments(payload)
        exemplar_slots = _parse_exemplar_slots(decoder_prefix)
        speaker_mapping = _match_global_speakers(
            raw_segments,
            exemplar_slots,
            min_overlap_s=self.exemplar_match_min_overlap_s,
        )
        current_segments = _crop_and_map_current_segments(
            raw_segments,
            context_seconds=max(0.0, float(context_seconds)),
            current_audio_seconds=max(0.0, float(current_audio_seconds)),
            registered_speaker_ids={slot.speaker_id for slot in exemplar_slots},
            speaker_mapping=speaker_mapping,
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
                "speaker_mapping": dict(speaker_mapping),
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
        form.add_field("prompt", self.instruction)
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


def _parse_response_segments(payload: dict[str, Any]) -> list[_TimedSpeakerSegment]:
    """Parse SGLang verbose segments and inherit missing speaker tags."""

    result: list[_TimedSpeakerSegment] = []
    last_speaker = "S01"
    payload_segments = payload.get("segments")
    if isinstance(payload_segments, list):
        for item in payload_segments:
            if not isinstance(item, dict):
                continue
            start_s = float(item.get("start") or 0.0)
            end_s = float(item.get("end") or 0.0)
            if end_s <= start_s:
                continue
            text = str(item.get("text") or "")
            speaker_match = _SPEAKER_PREFIX_RE.match(text)
            if speaker_match is not None:
                last_speaker = speaker_match.group("speaker").upper()
                text = text[speaker_match.end() :]
            result.append(
                _TimedSpeakerSegment(
                    start_s=max(0.0, start_s),
                    end_s=end_s,
                    speaker_id=last_speaker,
                    text=re.sub(r"\s+", " ", text).strip(),
                )
            )
    if result:
        return result
    return _parse_timestamped_text(str(payload.get("text") or ""))


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


def _parse_exemplar_slots(decoder_prefix: str) -> list[_ExemplarSlot]:
    """Recover registered global-speaker slots from the layout transcript."""

    return [
        _ExemplarSlot(
            start_s=segment.start_s,
            end_s=segment.end_s,
            speaker_id=segment.speaker_id,
        )
        for segment in _parse_timestamped_text(decoder_prefix)
    ]


def _overlap_duration(
    left_start_s: float,
    left_end_s: float,
    right_start_s: float,
    right_end_s: float,
) -> float:
    """Return the non-negative intersection duration of two ranges."""

    return max(0.0, min(left_end_s, right_end_s) - max(left_start_s, right_start_s))


def _match_global_speakers(
    segments: list[_TimedSpeakerSegment],
    slots: list[_ExemplarSlot],
    *,
    min_overlap_s: float,
) -> dict[str, str]:
    """Greedily match local labels to global exemplar labels one-to-one."""

    scores: dict[tuple[str, str], float] = {}
    for segment in segments:
        for slot in slots:
            overlap_s = _overlap_duration(
                segment.start_s,
                segment.end_s,
                slot.start_s,
                slot.end_s,
            )
            if overlap_s <= 0:
                continue
            key = (segment.speaker_id, slot.speaker_id)
            scores[key] = scores.get(key, 0.0) + overlap_s

    mapping: dict[str, str] = {}
    claimed_global_ids: set[str] = set()
    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    for (local_id, global_id), overlap_s in ranked:
        if overlap_s + 1e-9 < min_overlap_s:
            continue
        if local_id in mapping or global_id in claimed_global_ids:
            continue
        mapping[local_id] = global_id
        claimed_global_ids.add(global_id)
    return mapping


def _crop_and_map_current_segments(
    segments: list[_TimedSpeakerSegment],
    *,
    context_seconds: float,
    current_audio_seconds: float,
    registered_speaker_ids: set[str],
    speaker_mapping: dict[str, str],
) -> list[dict[str, object]]:
    """Crop exemplar time and map current segments without ``UNKNOWN`` labels."""

    current_end_s = context_seconds + current_audio_seconds
    mapping = dict(speaker_mapping)
    reserved_global_ids = set(registered_speaker_ids) | set(mapping.values())
    result: list[dict[str, object]] = []
    for segment in segments:
        start_s = max(segment.start_s, context_seconds)
        end_s = min(segment.end_s, current_end_s)
        if end_s <= start_s:
            continue
        global_id = mapping.get(segment.speaker_id)
        if global_id is None:
            global_id = _next_global_speaker_id(reserved_global_ids)
            mapping[segment.speaker_id] = global_id
            reserved_global_ids.add(global_id)
        result.append(
            {
                "start_s": round(max(0.0, start_s - context_seconds), 6),
                "end_s": round(max(0.0, end_s - context_seconds), 6),
                "speaker_id": global_id,
                "text": segment.text,
            }
        )
    speaker_mapping.clear()
    speaker_mapping.update(mapping)
    return result


def _next_global_speaker_id(reserved_ids: set[str]) -> str:
    """Allocate the first compact speaker label absent from the session pool."""

    index = 1
    while f"S{index:02d}" in reserved_ids:
        index += 1
    return f"S{index:02d}"
