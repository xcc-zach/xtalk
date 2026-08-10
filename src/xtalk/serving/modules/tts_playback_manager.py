# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import logging
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

from ...models import ForcedAligner, ForcedAlignmentUnit, Models
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    LLMAgentResponseFinish,
    LLMAgentResponseUpdate,
    ResponseFinish,
    ResponseUpdate,
    TTSChunkPlayed,
    TTSChunkReady,
    TTSPlaybackFinished,
    TTSPlaybackStopped,
    TTSResponseClosed,
    TTSStarted,
    TTSStopped,
    TTSStreamingTextAccepted,
    TTSTextDeliveryFinished,
    TTSTextSynthesisStarted,
    TTSTextSynthesized,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


@dataclass
class _AudioChunk:
    """One generated audio chunk waiting to be assigned to a text segment."""

    audio_chunk: bytes
    sample_rate: int
    duration_ms: float


@dataclass
class _RoughTextUnit:
    """One rough text-timing unit and its original-text end offset."""

    char_end: int
    duration_ms: float


@dataclass
class _StreamingRoughTextUnit:
    """One immutable StreamingTextTTS L1 unit on the prepared-audio timeline."""

    char_end: int
    end_ms: float


@dataclass
class _PlaybackSegment:
    """One synthesized text segment and its generated/played audio accounting."""

    text: str
    turn_id: int
    tts_mode: Literal["regular", "streaming"]
    total_audio_ms: float | None = None
    generated_audio_ms: float = 0.0
    played_audio_ms: float = 0.0
    streaming_rough_units: list[_StreamingRoughTextUnit] = field(
        default_factory=list
    )
    audio_sample_rate: int | None = None
    audio_parts: list[bytes] = field(default_factory=list)
    audio_prepared: bool = False
    alignment_audio_valid: bool = True
    alignment_state: str = "collecting"
    alignment_units: list[ForcedAlignmentUnit] = field(default_factory=list)
    alignment_task: asyncio.Task[None] | None = None


class TTSPlaybackManager(Manager):
    """Project confirmed TTS playback progress back onto response text."""

    _ROUGH_SAFETY_LAG_MS = 200.0
    _ROUGH_UNFINISHED_TAIL_MS = 300.0
    _ROUGH_CHARACTER_MS = 200.0
    _ROUGH_WORD_MS = 280.0
    _ROUGH_PUNCTUATION_MS = 80.0
    _STREAMING_L1_SAFETY_LAG_MS = 300.0
    _STREAMING_L1_CHARACTER_MS = 280.0
    _STREAMING_L1_WORD_MS = 380.0
    _STREAMING_L1_PUNCTUATION_MS = 160.0
    _ALIGNMENT_STOP_GRACE_SECONDS = 0.2

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is None and isinstance(models, dict):
            config = models
            models = None
        self.event_bus = event_bus
        self.session_id = session_id
        self.models = models
        self.config: dict[str, Any] = config or {}
        forced_alignment_config = self.config.get("forced_alignment", {})
        if not isinstance(forced_alignment_config, dict):
            forced_alignment_config = {}
        self.forced_aligner = models.get(ForcedAligner) if models else None
        self._forced_alignment_enabled = self.forced_aligner is not None
        self._forced_alignment_language = forced_alignment_config.get("language")
        self._stop_ack_timeout_ms = max(
            0.0,
            float(forced_alignment_config.get("stop_ack_timeout_ms", 500.0)),
        )
        self._current_response_text = ""
        self._pending_text = ""
        self._generated_text_by_response: dict[str, str] = {}
        self._finished_text_by_response: dict[str, str] = {}
        self._active_response_id: str | None = None
        self._segments: deque[_PlaybackSegment] = deque()
        self._started_waiting_for_preparation: deque[_PlaybackSegment] = deque()
        self._prepared_waiting_for_start: deque[_PlaybackSegment] = deque()
        self._generated_chunk_ms: deque[float] = deque()
        self._unassigned_audio_chunks: deque[_AudioChunk] = deque()
        self._collecting_segment: _PlaybackSegment | None = None
        self._streaming_segment: _PlaybackSegment | None = None
        self._alignment_tasks: set[asyncio.Task[None]] = set()
        self._stop_commit_task: asyncio.Task[None] | None = None
        self._awaiting_stop_ack = False
        self._turn_id = 0
        self._played_without_segment_ms = 0.0
        self._completed_text = ""
        self._reported_text = ""
        self._received_audio = False
        self._last_logged_alignment_segment: _PlaybackSegment | None = None
        self._last_logged_alignment_level: str | None = None

    def _reset_playback_tracking(self) -> None:
        """Reset playback-progress state while preserving response metadata."""

        self._turn_id += 1
        self._cancel_stop_commit_task()
        self._cancel_alignment_tasks()
        self._segments.clear()
        self._started_waiting_for_preparation.clear()
        self._prepared_waiting_for_start.clear()
        self._generated_chunk_ms.clear()
        self._unassigned_audio_chunks.clear()
        self._collecting_segment = None
        self._streaming_segment = None
        self._played_without_segment_ms = 0.0
        self._completed_text = ""
        self._reported_text = ""
        self._received_audio = False
        self._awaiting_stop_ack = False
        self._last_logged_alignment_segment = None
        self._last_logged_alignment_level = None

    def _reset_all_state(self) -> None:
        """Reset all cached response and playback-tracking state."""

        response_id = self._active_response_id
        self._reset_playback_tracking()
        self._current_response_text = ""
        self._pending_text = ""
        self._active_response_id = None
        if response_id is not None:
            self._generated_text_by_response.pop(response_id, None)
            self._finished_text_by_response.pop(response_id, None)

    def _matches_active_response(self, response_id: str) -> bool:
        """Return whether an event belongs to the active playback response."""

        return bool(
            self._active_response_id
            and (not response_id or response_id == self._active_response_id)
        )

    @Manager.event_handler(LLMAgentResponseUpdate, priority=20)
    async def _cache_response_update(self, event: LLMAgentResponseUpdate) -> None:
        """Track the active response text without emitting frontend-facing updates."""

        next_text = event.text or ""
        self._generated_text_by_response[event.response_id] = next_text
        if event.response_id == self._active_response_id:
            self._current_response_text = next_text

    @Manager.event_handler(TTSStarted, priority=20)
    async def _activate_response(self, event: TTSStarted) -> None:
        """Initialize playback accounting before the response reaches the client."""

        if not event.response_id:
            return
        if self._active_response_id == event.response_id:
            return
        if self._active_response_id is not None:
            logger.error(
                "TTS response started before prior playback closed - session: %s, "
                "active: %s, incoming: %s",
                self.session_id,
                self._active_response_id,
                event.response_id,
            )
            return
        self._reset_playback_tracking()
        self._active_response_id = event.response_id
        self._current_response_text = self._generated_text_by_response.get(
            event.response_id,
            "",
        )
        self._pending_text = self._finished_text_by_response.get(
            event.response_id,
            "",
        )

    @Manager.event_handler(TTSTextSynthesisStarted, priority=20)
    async def _start_text_segment(self, event: TTSTextSynthesisStarted) -> None:
        """Bind the next FIFO regular TTS segment to paced audio delivery."""

        if not self._matches_active_response(event.response_id):
            return
        text = event.text or ""
        if not text:
            return
        if self._collecting_segment is not None:
            logger.warning(
                "TTS sentence started before the prior delivery ended - session: %s",
                self.session_id,
            )
            self._reset_playback_tracking()
            return

        if self._prepared_waiting_for_start:
            segment = self._prepared_waiting_for_start.popleft()
            if not self._validate_paired_text(segment, text, "delivery start"):
                return
        else:
            if self._started_waiting_for_preparation:
                logger.warning(
                    "TTS delivery start overtook prepared audio - session: %s",
                    self.session_id,
                )
                self._reset_playback_tracking()
                return
            segment = _PlaybackSegment(
                text=text,
                turn_id=self._turn_id,
                tts_mode="regular",
            )
            self._segments.append(segment)
            self._started_waiting_for_preparation.append(segment)

        self._collecting_segment = segment
        self._attach_unassigned_audio(segment)
        await self._apply_pending_playback_time()
        await self._publish_progress_if_grew()

    @Manager.event_handler(TTSStreamingTextAccepted, priority=20)
    async def _accept_streaming_text(
        self,
        event: TTSStreamingTextAccepted,
    ) -> None:
        """Append text to an immutable StreamingTextTTS L1 timeline."""

        if not self._matches_active_response(event.response_id):
            return
        text = event.text or ""
        if not text:
            return
        if self._streaming_segment is None:
            self._streaming_segment = _PlaybackSegment(
                text="",
                turn_id=self._turn_id,
                tts_mode="streaming",
            )
            self._segments.append(self._streaming_segment)
            self._attach_unassigned_audio(self._streaming_segment)

        segment = self._streaming_segment
        text_offset = len(segment.text)
        segment.text += text
        self._append_streaming_rough_units(
            segment,
            text=text,
            text_offset=text_offset,
            prepared_audio_ms=event.prepared_audio_ms,
        )
        await self._apply_pending_playback_time()

    @Manager.event_handler(TTSChunkReady, priority=20)
    async def _track_generated_chunk(self, event: TTSChunkReady) -> None:
        """Track generated chunk durations in FIFO order for playback acks."""
        if not self._matches_active_response(event.response_id):
            return
        chunk_ms = self._chunk_duration_ms(
            audio_chunk=event.audio_chunk,
            sample_rate=event.sample_rate,
        )
        if chunk_ms <= 0.0:
            return
        self._received_audio = True
        self._generated_chunk_ms.append(chunk_ms)
        chunk = _AudioChunk(
            audio_chunk=event.audio_chunk,
            sample_rate=event.sample_rate,
            duration_ms=chunk_ms,
        )
        segment = self._collecting_segment or self._streaming_segment
        if segment is None:
            self._unassigned_audio_chunks.append(chunk)
        else:
            self._attach_audio_chunk(segment, chunk)
        await self._publish_progress_if_grew()

    @Manager.event_handler(TTSTextSynthesized, priority=20)
    async def _prepare_text_segment(self, event: TTSTextSynthesized) -> None:
        """Bind complete final PCM and start alignment without waiting for delivery."""

        if not self._matches_active_response(event.response_id):
            return
        text = event.text or ""
        if not text:
            return

        if self._streaming_segment is not None:
            segment = self._streaming_segment
            segment.text = text
        else:
            if (
                self._started_waiting_for_preparation
                and self._prepared_waiting_for_start
            ):
                logger.warning(
                    "TTS FIFO preparation state became ambiguous - session: %s",
                    self.session_id,
                )
                self._reset_playback_tracking()
                return
            if self._started_waiting_for_preparation:
                segment = self._started_waiting_for_preparation.popleft()
                if not self._validate_paired_text(segment, text, "PCM preparation"):
                    return
            else:
                segment = _PlaybackSegment(
                    text=text,
                    turn_id=self._turn_id,
                    tts_mode="regular",
                )
                self._segments.append(segment)
                self._prepared_waiting_for_start.append(segment)

        await self._apply_prepared_audio(segment, event)

    @Manager.event_handler(TTSTextDeliveryFinished, priority=20)
    async def _finish_text_delivery(
        self,
        event: TTSTextDeliveryFinished,
    ) -> None:
        """Close the current FIFO delivery without delaying audio preparation."""

        if not self._matches_active_response(event.response_id):
            return
        segment = self._collecting_segment or self._streaming_segment
        if segment is None:
            logger.warning(
                "TTS delivery finished without an active segment - session: %s",
                self.session_id,
            )
            return
        if not self._validate_paired_text(segment, event.text or "", "delivery end"):
            return

        if segment is self._collecting_segment:
            self._collecting_segment = None
        if segment is self._streaming_segment:
            self._streaming_segment = None

        if not event.succeeded:
            self._discard_segment(segment)
        await self._apply_pending_playback_time()
        await self._publish_progress_if_grew()

    def _validate_paired_text(
        self,
        segment: _PlaybackSegment,
        text: str,
        phase: str,
    ) -> bool:
        """Validate strict FIFO text pairing and reset unsafe state on mismatch."""

        if segment.text == text:
            return True
        logger.warning(
            "TTS FIFO text mismatch - session: %s, phase: %s, "
            "expected_length: %s, actual_length: %s",
            self.session_id,
            phase,
            len(segment.text),
            len(text),
        )
        self._reset_playback_tracking()
        return False

    async def _apply_prepared_audio(
        self,
        segment: _PlaybackSegment,
        event: TTSTextSynthesized,
    ) -> None:
        """Attach complete final PCM and start precise alignment immediately."""

        prepared_audio = event.audio_chunk or b""
        prepared_sample_rate = int(event.sample_rate or 0)
        if prepared_audio:
            if prepared_sample_rate <= 0:
                logger.warning(
                    "Ignoring prepared TTS audio without a sample rate - session: %s",
                    self.session_id,
                )
                self._discard_segment(segment)
                return
            if (
                segment.audio_sample_rate is not None
                and segment.audio_sample_rate != prepared_sample_rate
            ):
                segment.alignment_audio_valid = False
                logger.warning(
                    "TTS prepared audio changed sample rate - session: %s, "
                    "expected: %s, got: %s",
                    self.session_id,
                    segment.audio_sample_rate,
                    prepared_sample_rate,
                )
            segment.audio_sample_rate = prepared_sample_rate
            segment.audio_parts = [prepared_audio]
            segment.audio_prepared = True
            total_audio_ms = self._chunk_duration_ms(
                audio_chunk=prepared_audio,
                sample_rate=prepared_sample_rate,
            )
        else:
            total_audio_ms = segment.generated_audio_ms
            if total_audio_ms <= 0.0:
                total_audio_ms = max(0.0, float(event.audio_duration or 0.0))

        if total_audio_ms <= 0.0 or not segment.audio_sample_rate:
            logger.warning(
                "Ignoring synthesized TTS text without audio - session: %s, "
                "text_length: %s",
                self.session_id,
                len(segment.text),
            )
            self._discard_segment(segment)
            return

        segment.total_audio_ms = total_audio_ms
        if self._forced_alignment_enabled:
            segment.alignment_state = "pending"
        else:
            segment.alignment_state = "disabled"
            segment.audio_parts.clear()

        self._try_start_segment_alignment(segment)
        await self._apply_pending_playback_time()
        await self._publish_progress_if_grew()

    @Manager.event_handler(TTSChunkPlayed, priority=20)
    async def _publish_response_update(self, event: TTSChunkPlayed) -> None:
        """Advance played text according to frontend playback confirmations."""
        if not self._matches_active_response(event.response_id):
            return
        if not self._generated_chunk_ms:
            return
        remaining_ms = max(0.0, self._generated_chunk_ms.popleft())
        self._played_without_segment_ms += remaining_ms
        await self._apply_pending_playback_time()

    def _attach_unassigned_audio(self, segment: _PlaybackSegment) -> None:
        """Attach FIFO audio that arrived before its implicit segment opened."""

        while self._unassigned_audio_chunks:
            self._attach_audio_chunk(
                segment,
                self._unassigned_audio_chunks.popleft(),
            )

    def _attach_audio_chunk(
        self,
        segment: _PlaybackSegment,
        chunk: _AudioChunk,
    ) -> None:
        """Attach one emitted PCM chunk to the currently collecting segment."""

        if segment.audio_sample_rate is None:
            segment.audio_sample_rate = chunk.sample_rate
        elif segment.audio_sample_rate != chunk.sample_rate:
            segment.alignment_audio_valid = False
            logger.warning(
                "TTS playback alignment saw mixed sample rates - session: %s, "
                "expected: %s, got: %s",
                self.session_id,
                segment.audio_sample_rate,
                chunk.sample_rate,
            )

        segment.generated_audio_ms += chunk.duration_ms
        if (
            self._forced_alignment_enabled
            and segment.alignment_audio_valid
            and not segment.audio_prepared
        ):
            segment.audio_parts.append(chunk.audio_chunk)

    def _discard_segment(self, segment: _PlaybackSegment) -> None:
        """Discard an unusable segment without treating its text as played."""

        if segment is self._collecting_segment:
            self._collecting_segment = None
        if segment is self._streaming_segment:
            self._streaming_segment = None
        self._remove_segment_from_pairing_queues(segment)
        if segment in self._segments:
            self._segments.remove(segment)
        if segment.alignment_task is not None and not segment.alignment_task.done():
            segment.alignment_task.cancel()

    def _remove_segment_from_pairing_queues(
        self,
        segment: _PlaybackSegment,
    ) -> None:
        """Remove one segment from either side of FIFO preparation pairing."""

        if segment in self._started_waiting_for_preparation:
            self._started_waiting_for_preparation.remove(segment)
        if segment in self._prepared_waiting_for_start:
            self._prepared_waiting_for_start.remove(segment)

    def _build_reported_text(self) -> str:
        """Return the current played text prefix across completed and active segments."""

        if not self._segments:
            return self._completed_text
        segment = self._segments[0]
        if not segment.text:
            return self._completed_text

        self._log_alignment_level_if_changed(segment)
        aligned_prefix = self._build_aligned_segment_prefix(segment)
        if aligned_prefix is not None:
            return self._completed_text + aligned_prefix

        return self._completed_text + self._build_rough_segment_prefix(segment)

    @staticmethod
    def _alignment_level(segment: _PlaybackSegment) -> str:
        """Return the calibration level currently used for one segment."""

        precise_alignment_ready = (
            segment.alignment_state == "ready" and bool(segment.alignment_units)
        )
        if segment.tts_mode == "streaming":
            if precise_alignment_ready:
                return "L3-precise"
            if segment.total_audio_ms is not None:
                return "L2-total-duration-ratio"
            return "L1-online-estimate"

        if precise_alignment_ready:
            return "L2-precise"
        return "L1-rough"

    def _log_alignment_level_if_changed(
        self,
        segment: _PlaybackSegment,
    ) -> None:
        """Log the active calibration level once per segment-level transition."""

        level = self._alignment_level(segment)
        if (
            self._last_logged_alignment_segment is segment
            and self._last_logged_alignment_level == level
        ):
            return

        self._last_logged_alignment_segment = segment
        self._last_logged_alignment_level = level
        logger.debug(
            "TTS playback alignment - session: %s, mode: %s, level: %s, state: %s",
            self.session_id,
            segment.tts_mode,
            level,
            segment.alignment_state,
        )

    def _build_rough_segment_prefix(self, segment: _PlaybackSegment) -> str:
        """Estimate a monotonic text prefix before precise alignment is ready."""

        if segment.tts_mode == "streaming" and segment.total_audio_ms is None:
            return self._build_streaming_online_prefix(segment)

        units = self._rough_text_units(segment.text)
        if not units:
            return ""

        text_duration_ms = sum(unit.duration_ms for unit in units)
        if text_duration_ms <= 0.0:
            return ""

        if segment.total_audio_ms is None:
            safe_played_ms = max(
                0.0,
                segment.played_audio_ms - self._ROUGH_SAFETY_LAG_MS,
            )
            estimated_total_ms = max(
                text_duration_ms,
                segment.generated_audio_ms + self._ROUGH_UNFINISHED_TAIL_MS,
            )
            ratio = safe_played_ms / estimated_total_ms
        elif segment.total_audio_ms > 0.0:
            ratio = segment.played_audio_ms / segment.total_audio_ms
        else:
            return ""

        target_duration_ms = max(0.0, min(1.0, ratio)) * text_duration_ms
        elapsed_ms = 0.0
        prefix_len = 0
        for unit in units:
            elapsed_ms += unit.duration_ms
            if elapsed_ms > target_duration_ms + 1e-6:
                break
            prefix_len = unit.char_end

        if prefix_len <= 0:
            return ""
        return segment.text[:prefix_len]

    def _append_streaming_rough_units(
        self,
        segment: _PlaybackSegment,
        *,
        text: str,
        text_offset: int,
        prepared_audio_ms: float,
    ) -> None:
        """Append immutable L1 units after the text preparation watermark."""

        rough_units = self._rough_text_units(
            text,
            character_ms=self._STREAMING_L1_CHARACTER_MS,
            word_ms=self._STREAMING_L1_WORD_MS,
            punctuation_ms=self._STREAMING_L1_PUNCTUATION_MS,
        )
        if not rough_units:
            separator_duration_ms = sum(
                self._STREAMING_L1_PUNCTUATION_MS
                for character in text
                if unicodedata.category(character).startswith("P")
            )
            rough_units = [
                _RoughTextUnit(
                    char_end=len(text),
                    duration_ms=separator_duration_ms,
                )
            ]

        prior_end_ms = (
            segment.streaming_rough_units[-1].end_ms
            if segment.streaming_rough_units
            else 0.0
        )
        timeline_end_ms = max(
            0.0,
            prepared_audio_ms,
            segment.generated_audio_ms,
            segment.played_audio_ms,
            prior_end_ms,
        )
        for unit in rough_units:
            timeline_end_ms += max(0.0, unit.duration_ms)
            segment.streaming_rough_units.append(
                _StreamingRoughTextUnit(
                    char_end=text_offset + unit.char_end,
                    end_ms=timeline_end_ms,
                )
            )

    def _build_streaming_online_prefix(
        self,
        segment: _PlaybackSegment,
    ) -> str:
        """Build StreamingTextTTS L1 text from its immutable unit timeline."""

        safe_played_ms = max(
            0.0,
            segment.played_audio_ms - self._STREAMING_L1_SAFETY_LAG_MS,
        )
        prefix_len = 0
        for unit in segment.streaming_rough_units:
            if unit.end_ms > safe_played_ms + 1e-6:
                break
            prefix_len = unit.char_end

        if prefix_len <= 0:
            return ""
        return segment.text[:prefix_len]

    @classmethod
    def _rough_text_units(
        cls,
        text: str,
        *,
        character_ms: float | None = None,
        word_ms: float | None = None,
        punctuation_ms: float | None = None,
    ) -> list[_RoughTextUnit]:
        """Split text into weighted character or word units for rough timing."""

        resolved_character_ms = (
            cls._ROUGH_CHARACTER_MS if character_ms is None else character_ms
        )
        resolved_word_ms = cls._ROUGH_WORD_MS if word_ms is None else word_ms
        resolved_punctuation_ms = (
            cls._ROUGH_PUNCTUATION_MS
            if punctuation_ms is None
            else punctuation_ms
        )
        units: list[_RoughTextUnit] = []
        word_open = False

        def flush_word(char_end: int) -> None:
            nonlocal word_open
            if not word_open:
                return
            units.append(
                _RoughTextUnit(
                    char_end=char_end,
                    duration_ms=resolved_word_ms,
                )
            )
            word_open = False

        for index, character in enumerate(text):
            if cls._is_character_timing_unit(character):
                flush_word(index)
                units.append(
                    _RoughTextUnit(
                        char_end=index + 1,
                        duration_ms=resolved_character_ms,
                    )
                )
                continue

            if character.isalnum() or character == "'":
                word_open = True
                continue

            flush_word(index)
            if not units:
                continue
            units[-1].char_end = index + 1
            if unicodedata.category(character).startswith("P"):
                units[-1].duration_ms += resolved_punctuation_ms

        flush_word(len(text))
        return units

    @staticmethod
    def _is_character_timing_unit(character: str) -> bool:
        """Return whether one CJK or Japanese character is a timing unit."""

        codepoint = ord(character)
        return (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
            or 0x20000 <= codepoint <= 0x2CEAF
            or 0x3040 <= codepoint <= 0x30FF
            or 0xFF66 <= codepoint <= 0xFF9D
        )

    def _build_aligned_segment_prefix(
        self,
        segment: _PlaybackSegment,
    ) -> str | None:
        """Build a text prefix from forced-alignment timestamps when available."""

        if segment.alignment_state != "ready" or not segment.alignment_units:
            return None

        prefix_len = 0
        played_ms = max(0.0, segment.played_audio_ms)
        for unit in segment.alignment_units:
            if unit.end_ms <= played_ms + 1e-6:
                prefix_len = max(prefix_len, unit.char_end)
                continue
            break

        if prefix_len <= 0:
            return ""
        prefix_len = self._extend_prefix_over_attached_text(segment.text, prefix_len)
        return segment.text[:prefix_len]

    @staticmethod
    def _chunk_duration_ms(*, audio_chunk: bytes, sample_rate: int) -> float:
        """Return PCM duration in milliseconds for one mono int16 chunk."""

        if not audio_chunk or sample_rate <= 0:
            return 0.0
        sample_count = len(audio_chunk) / 2
        return sample_count * 1000.0 / sample_rate

    def _try_start_segment_alignment(self, segment: _PlaybackSegment) -> None:
        """Start forced alignment once a segment has its full audio."""

        if not self._forced_alignment_enabled or self.forced_aligner is None:
            return
        if segment.alignment_state != "pending":
            return
        if segment.total_audio_ms is None or segment.total_audio_ms <= 0.0:
            segment.alignment_state = "failed"
            segment.audio_parts.clear()
            return
        if (
            not segment.alignment_audio_valid
            or not segment.audio_parts
            or not segment.audio_sample_rate
        ):
            segment.alignment_state = "failed"
            segment.audio_parts.clear()
            return

        segment.alignment_state = "running"
        task = asyncio.create_task(self._align_segment(segment, self._turn_id))
        segment.alignment_task = task
        self._alignment_tasks.add(task)
        task.add_done_callback(self._alignment_tasks.discard)

    async def _align_segment(
        self,
        segment: _PlaybackSegment,
        turn_id: int,
    ) -> None:
        """Run forced alignment for a text segment in the background."""

        assert self.forced_aligner is not None
        try:
            audio = b"".join(segment.audio_parts)
            segment.audio_parts.clear()
            units = await self.forced_aligner.async_align(
                audio=audio,
                text=segment.text,
                language=self._forced_alignment_language,
            )
            if turn_id != self._turn_id or segment.turn_id != self._turn_id:
                return
            normalized_units = self._normalize_alignment_units(segment.text, units)
            segment.alignment_units = normalized_units
            segment.alignment_state = "ready" if normalized_units else "failed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if turn_id == self._turn_id and segment.turn_id == self._turn_id:
                segment.alignment_state = "failed"
            logger.warning(
                "TTS forced alignment failed - session: %s, error: %s",
                self.session_id,
                exc,
            )
        finally:
            if turn_id == self._turn_id and segment.turn_id == self._turn_id:
                await self._publish_progress_if_grew()

    def _normalize_alignment_units(
        self,
        text: str,
        units: list[ForcedAlignmentUnit],
    ) -> list[ForcedAlignmentUnit]:
        """Map model-returned units onto character spans in the original text."""

        normalized: list[ForcedAlignmentUnit] = []
        cursor = 0
        for unit in sorted(units, key=lambda item: (item.start_ms, item.end_ms)):
            if unit.end_ms < unit.start_ms:
                continue

            span = self._resolve_unit_span(text, unit, cursor)
            if span is None:
                continue
            char_start, char_end = span
            cursor = max(cursor, char_end)
            normalized.append(
                ForcedAlignmentUnit(
                    text=text[char_start:char_end],
                    start_ms=max(0.0, float(unit.start_ms)),
                    end_ms=max(0.0, float(unit.end_ms)),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        return normalized

    def _resolve_unit_span(
        self,
        text: str,
        unit: ForcedAlignmentUnit,
        cursor: int,
    ) -> tuple[int, int] | None:
        """Resolve one alignment unit to a non-empty span in original text."""

        if 0 <= unit.char_start < unit.char_end <= len(text):
            return unit.char_start, unit.char_end

        unit_text = (unit.text or "").strip()
        if not unit_text:
            return None

        start = text.find(unit_text, cursor)
        if start < 0:
            compact_text = "".join(unit_text.split())
            if compact_text and compact_text != unit_text:
                start = text.find(compact_text, cursor)
                unit_text = compact_text
        if start < 0:
            return None
        end = start + len(unit_text)
        return start, end

    @staticmethod
    def _extend_prefix_over_attached_text(text: str, prefix_len: int) -> int:
        """Attach punctuation and whitespace immediately after a spoken unit."""

        trailing_chars = set(" \t\r\n，。！？、；：,.!?;:)]}）】》”’\"'")
        while prefix_len < len(text) and text[prefix_len] in trailing_chars:
            prefix_len += 1
        return prefix_len

    async def _apply_pending_playback_time(self) -> None:
        """Apply queued played-audio time to known text segments."""

        remaining_ms = max(0.0, self._played_without_segment_ms)
        while self._segments:
            segment = self._segments[0]
            if segment.total_audio_ms is None:
                if remaining_ms > 0.0:
                    segment.played_audio_ms += remaining_ms
                    remaining_ms = 0.0
                break

            if segment.played_audio_ms + 1e-6 >= segment.total_audio_ms:
                self._complete_front_segment(segment)
                continue
            if remaining_ms <= 0.0:
                break

            unplayed_ms = segment.total_audio_ms - segment.played_audio_ms
            consume_ms = min(remaining_ms, unplayed_ms)
            segment.played_audio_ms += consume_ms
            remaining_ms -= consume_ms

            if segment.played_audio_ms + 1e-6 >= segment.total_audio_ms:
                self._complete_front_segment(segment)

        self._played_without_segment_ms = remaining_ms
        await self._publish_progress_if_grew()

    def _complete_front_segment(self, segment: _PlaybackSegment) -> None:
        """Move one fully played FIFO segment into completed text."""

        if not self._segments or self._segments[0] is not segment:
            return
        self._segments.popleft()
        self._completed_text += segment.text
        self._remove_segment_from_pairing_queues(segment)
        segment.audio_parts.clear()
        if segment is self._collecting_segment:
            self._collecting_segment = None
        if segment is self._streaming_segment:
            self._streaming_segment = None
        if segment.alignment_task is not None and not segment.alignment_task.done():
            segment.alignment_task.cancel()

    async def _publish_progress_if_grew(self) -> None:
        """Publish one response update when the played text prefix grows."""

        played_text = self._build_reported_text()
        if len(played_text) <= len(self._reported_text):
            return
        await self._emit_response_update(played_text)

    async def _emit_response_update(self, text: str) -> None:
        """Publish and log one frontend-facing played-text update."""

        previous_text = self._reported_text
        previous_length = len(previous_text)
        completed_length = min(len(self._completed_text), len(text))
        segment = self._segments[0] if self._segments else None
        sources: list[str] = []

        if previous_length < completed_length:
            sources.append("playback-complete")

        active_prefix_start = max(previous_length, completed_length)
        if segment is not None and len(text) > active_prefix_start:
            mode = segment.tts_mode
            level = self._alignment_level(segment)
            state = segment.alignment_state
            played_ms = f"{segment.played_audio_ms:.1f}"
            total_ms = (
                f"{segment.total_audio_ms:.1f}"
                if segment.total_audio_ms is not None
                else "n/a"
            )
            sources.append(f"{mode}:{level}")
        else:
            mode = "n/a"
            level = "n/a"
            state = "n/a"
            played_ms = "n/a"
            total_ms = "n/a"

        if not sources:
            sources.append("final-commit")

        delta = text[previous_length:] if text.startswith(previous_text) else text
        self._reported_text = text
        logger.debug(
            "TTS response update - session: %s, source: %s, mode: %s, "
            "level: %s, state: %s, played_ms: %s, total_ms: %s, "
            "delta: %r, text: %r",
            self.session_id,
            "+".join(sources),
            mode,
            level,
            state,
            played_ms,
            total_ms,
            delta,
            text,
        )
        await self.event_bus.publish(
            ResponseUpdate(
                session_id=self.session_id,
                response_id=self._active_response_id or "",
                text=text,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    @Manager.event_handler(LLMAgentResponseFinish, priority=20)
    async def _cache_response_finish(self, event: LLMAgentResponseFinish) -> None:
        """Cache the latest generated response until playback finishes."""

        next_text = event.text or ""
        self._generated_text_by_response[event.response_id] = next_text
        self._finished_text_by_response[event.response_id] = next_text
        if event.response_id == self._active_response_id:
            self._current_response_text = next_text
            self._pending_text = next_text

    @Manager.event_handler(TTSPlaybackFinished, priority=-100)
    async def _publish_response_finish(self, event: TTSPlaybackFinished) -> None:
        """Publish response-finish after frontend playback completion."""

        if not self._matches_active_response(event.response_id):
            return
        response_id = self._active_response_id
        if response_id is None:
            return
        if not self._received_audio:
            logger.warning(
                "TTS playback finished without generated audio; discarding unplayed "
                "response - session: %s",
                self.session_id,
            )
            await self._close_response(response_id)
            return
        played_text = self._build_reported_text()
        if len(self._reported_text) > len(played_text):
            played_text = self._reported_text
        if not played_text:
            logger.warning(
                "TTS playback finished without playback-confirmed text - session: %s",
                self.session_id,
            )
            await self._close_response(response_id)
            return
        if self._pending_text and played_text != self._pending_text:
            logger.warning(
                "TTS playback completed only part of the generated response - "
                "session: %s, played_length: %s, generated_length: %s",
                self.session_id,
                len(played_text),
                len(self._pending_text),
            )
        try:
            await self._commit_playback_text(played_text)
        finally:
            await self._close_response(response_id)

    @Manager.event_handler(TTSStopped, priority=20)
    async def _handle_tts_stopped(self, event: TTSStopped) -> None:
        """Wait for the frontend's exact playback position after an early stop."""

        if not self._matches_active_response(event.response_id):
            return
        self._awaiting_stop_ack = True
        self._cancel_stop_commit_task()
        turn_id = self._turn_id
        response_id = self._active_response_id
        if response_id is None:
            return
        self._stop_commit_task = asyncio.create_task(
            self._commit_stopped_playback_after_timeout(turn_id, response_id)
        )

    @Manager.event_handler(TTSPlaybackStopped, priority=-100)
    async def _handle_tts_playback_stopped(
        self,
        event: TTSPlaybackStopped,
    ) -> None:
        """Apply unacknowledged browser playback before committing interruption."""

        if (
            not self._awaiting_stop_ack
            or not self._matches_active_response(event.response_id)
        ):
            return
        self._cancel_stop_commit_task()
        self._played_without_segment_ms += max(0.0, event.played_audio_ms)
        await self._apply_pending_playback_time()
        await self._commit_stopped_playback()

    async def _commit_stopped_playback_after_timeout(
        self,
        turn_id: int,
        response_id: str,
    ) -> None:
        """Fall back to completed-chunk progress for older frontend clients."""

        try:
            await asyncio.sleep(self._stop_ack_timeout_ms / 1000.0)
            if (
                turn_id != self._turn_id
                or response_id != self._active_response_id
                or not self._awaiting_stop_ack
            ):
                return
            logger.warning(
                "Timed out waiting for frontend TTS stop position - session: %s",
                self.session_id,
            )
            await self._commit_stopped_playback()
        except asyncio.CancelledError:
            raise

    async def _commit_stopped_playback(self) -> None:
        """Commit the force-aligned prefix after playback has actually stopped."""

        response_id = self._active_response_id
        if response_id is None:
            return
        await self._wait_for_alignment_grace()
        played_text = self._build_reported_text()
        if len(self._reported_text) > len(played_text):
            played_text = self._reported_text
        try:
            if played_text:
                await self._commit_playback_text(played_text)
        finally:
            await self._close_response(response_id)

    async def _wait_for_alignment_grace(self) -> None:
        """Briefly wait for precise alignment before committing an interruption."""

        tasks = [task for task in self._alignment_tasks if not task.done()]
        if not tasks:
            return
        await asyncio.wait(
            tasks,
            timeout=self._ALIGNMENT_STOP_GRACE_SECONDS,
        )

    async def _commit_playback_text(self, text: str) -> None:
        """Publish the final playback-confirmed text for the active turn."""

        if self._reported_text != text:
            await self._emit_response_update(text)
        await self.event_bus.publish(
            ResponseFinish(
                session_id=self.session_id,
                response_id=self._active_response_id or "",
                text=text,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def _close_response(self, response_id: str) -> None:
        """Clear playback state and publish the response-delivery barrier."""

        if response_id != self._active_response_id:
            return
        self._reset_all_state()
        await self.event_bus.publish(
            TTSResponseClosed(
                session_id=self.session_id,
                response_id=response_id,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    async def shutdown(self) -> None:
        stop_commit_task = self._stop_commit_task
        self._cancel_stop_commit_task()
        if stop_commit_task is not None:
            await asyncio.gather(stop_commit_task, return_exceptions=True)
        self._cancel_alignment_tasks()
        if self._alignment_tasks:
            await asyncio.gather(*self._alignment_tasks, return_exceptions=True)

    def _cancel_alignment_tasks(self) -> None:
        """Cancel in-flight alignment tasks from an obsolete turn."""

        for task in tuple(self._alignment_tasks):
            if not task.done():
                task.cancel()

    def _cancel_stop_commit_task(self) -> None:
        """Cancel an obsolete stop-ack timeout without cancelling its caller."""

        task = self._stop_commit_task
        self._stop_commit_task = None
        if (
            task is not None
            and task is not asyncio.current_task()
            and not task.done()
        ):
            task.cancel()
