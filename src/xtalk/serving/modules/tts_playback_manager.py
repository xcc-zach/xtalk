# -*- coding: utf-8 -*-
from __future__ import annotations

# Playback tracking plan:
# 1. Extend TTSPlaybackManager state with a segment ledger, a FIFO queue of
#    generated chunk durations, the latest reported text prefix, and cached
#    final-response metadata such as the active response text.
# 2. Subscribe to TTSChunkGenerated and TTSTextSynthesized. TTSChunkGenerated
#    is only used to build the FIFO chunk-duration queue for playback acks,
#    while TTSTextSynthesized carries the sentence-level audio duration needed
#    to close each synthesized text segment.
# 3. Consume TTSChunkPlayed to advance playback through the segment ledger.
#    For each confirmed played chunk, reduce the FIFO duration queue, update the
#    active segment's played-audio counter, map the played-audio ratio back to a
#    text prefix, and publish ResponseUpdate whenever that played prefix grows.
# 4. Use a FIFO duration queue rather than a fixed chunk-size assumption. Each
#    TTSChunkGenerated contributes its exact duration, and each TTSChunkPlayed
#    consumes the next duration in order, so playback tracking matches the real
#    emitted audio timing, including short final chunks.
# 5. Tighten ResponseFinish to come from the same playback state. When playback
#    completes, flush any remaining played text via ResponseUpdate if needed,
#    publish ResponseFinish with the cached final text, and clear the playback
#    tracker state for the next turn.

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ...log_utils import logger
from ...models import ForceAligner, ForceAlignmentUnit, Models
from ..event_bus import EventBus
from ..events import (
    LLMAgentResponseFinish,
    LLMAgentResponseUpdate,
    ResponseFinish,
    ResponseUpdate,
    TTSChunkReady,
    TTSChunkPlayed,
    TTSPlaybackStopped,
    TTSPlaybackFinished,
    TTSTextSynthesized,
    TTSStopped,
)
from ..interfaces import Manager


@dataclass
class _AudioChunk:
    """One generated audio chunk waiting to be assigned to a text segment."""

    audio_chunk: bytes
    sample_rate: int
    duration_ms: float


@dataclass
class _PlaybackSegment:
    """One synthesized text segment and its generated/played audio accounting."""

    text: str
    total_audio_ms: float
    turn_id: int
    played_audio_ms: float = 0.0
    collected_audio_ms: float = 0.0
    audio_sample_rate: int | None = None
    audio_parts: list[bytes] = field(default_factory=list)
    alignment_audio_valid: bool = True
    alignment_state: str = "disabled"
    alignment_units: list[ForceAlignmentUnit] = field(default_factory=list)
    alignment_task: asyncio.Task[None] | None = None


class TTSPlaybackManager(Manager):
    """Project confirmed TTS playback progress back onto response text."""

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
        force_alignment_config = self.config.get("force_alignment", {})
        if not isinstance(force_alignment_config, dict):
            force_alignment_config = {}
        self.force_aligner = models.get(ForceAligner) if models else None
        self._force_alignment_enabled = bool(self.force_aligner) and bool(
            force_alignment_config.get("enabled", True)
        )
        self._force_alignment_language = force_alignment_config.get("language")
        self._fallback_while_alignment_pending = bool(
            force_alignment_config.get("fallback_while_pending", True)
        )
        self._stop_ack_timeout_ms = max(
            0.0,
            float(force_alignment_config.get("stop_ack_timeout_ms", 500.0)),
        )

        self._current_response_text = ""
        self._pending_text = ""
        self._segments: deque[_PlaybackSegment] = deque()
        self._generated_chunk_ms: deque[float] = deque()
        self._unassigned_audio_chunks: deque[_AudioChunk] = deque()
        self._alignment_tasks: set[asyncio.Task[None]] = set()
        self._stop_commit_task: asyncio.Task[None] | None = None
        self._awaiting_stop_ack = False
        self._turn_id = 0
        self._played_without_segment_ms = 0.0
        self._prebound_audio_ms = 0.0
        self._completed_text = ""
        self._reported_text = ""
        self._received_audio = False

    def _reset_playback_tracking(self) -> None:
        """Reset playback-progress state while preserving response metadata."""

        self._turn_id += 1
        self._cancel_stop_commit_task()
        self._cancel_alignment_tasks()
        self._segments.clear()
        self._generated_chunk_ms.clear()
        self._unassigned_audio_chunks.clear()
        self._played_without_segment_ms = 0.0
        self._prebound_audio_ms = 0.0
        self._completed_text = ""
        self._reported_text = ""
        self._received_audio = False
        self._awaiting_stop_ack = False

    def _reset_all_state(self) -> None:
        """Reset all cached response and playback-tracking state."""

        self._reset_playback_tracking()
        self._current_response_text = ""
        self._pending_text = ""

    @Manager.event_handler(LLMAgentResponseUpdate, priority=20)
    async def _cache_response_update(self, event: LLMAgentResponseUpdate) -> None:
        """Track the active response text without emitting frontend-facing updates."""

        next_text = event.text or ""
        if self._current_response_text and not next_text.startswith(
            self._current_response_text
        ):
            self._reset_all_state()
        self._current_response_text = next_text

    @Manager.event_handler(TTSChunkReady, priority=20)
    async def _track_generated_chunk(self, event: TTSChunkReady) -> None:
        """Track generated chunk durations in FIFO order for playback acks."""
        chunk_ms = self._chunk_duration_ms(
            audio_chunk=event.audio_chunk,
            sample_rate=event.sample_rate,
        )
        if chunk_ms <= 0.0:
            return
        self._received_audio = True
        self._generated_chunk_ms.append(chunk_ms)
        if self._prebound_audio_ms > 1e-6:
            self._prebound_audio_ms = max(0.0, self._prebound_audio_ms - chunk_ms)
            return

        self._unassigned_audio_chunks.append(
            _AudioChunk(
                audio_chunk=event.audio_chunk,
                sample_rate=event.sample_rate,
                duration_ms=chunk_ms,
            )
        )
        self._assign_audio_to_segments()

    @Manager.event_handler(TTSTextSynthesized, priority=20)
    async def _close_text_segment(self, event: TTSTextSynthesized) -> None:
        """Bind one synthesized text segment and align preloaded PCM before playback."""

        text = event.text or ""
        if not text:
            return

        preloaded_audio = event.audio_chunk or b""
        preloaded_sample_rate = int(event.sample_rate or 0)
        total_audio_ms = max(0.0, float(event.audio_duration or 0.0))
        if preloaded_audio and preloaded_sample_rate > 0:
            total_audio_ms = self._chunk_duration_ms(
                audio_chunk=preloaded_audio,
                sample_rate=preloaded_sample_rate,
            )
        if total_audio_ms <= 0.0:
            logger.warning(
                "Ignoring synthesized TTS text without audio - session: %s, "
                "text_length: %s",
                self.session_id,
                len(text),
            )
            return

        segment = _PlaybackSegment(
            text=text,
            total_audio_ms=total_audio_ms,
            turn_id=self._turn_id,
            alignment_state="pending"
            if self._force_alignment_enabled
            else "disabled",
        )
        if preloaded_audio and preloaded_sample_rate > 0:
            segment.collected_audio_ms = total_audio_ms
            segment.audio_sample_rate = preloaded_sample_rate
            segment.audio_parts.append(preloaded_audio)
            self._prebound_audio_ms += total_audio_ms

        self._segments.append(segment)
        if preloaded_audio and self._force_alignment_enabled:
            segment.alignment_state = "running"
            logger.info(
                "TTS forced alignment started before playback - session: %s, "
                "text_length: %s, audio_ms: %.1f",
                self.session_id,
                len(text),
                total_audio_ms,
            )
            await self._align_segment(segment, self._turn_id)
        else:
            self._assign_audio_to_segments()
        await self._apply_pending_playback_time()

    @Manager.event_handler(TTSChunkPlayed, priority=20)
    async def _publish_response_update(self, event: TTSChunkPlayed) -> None:
        """Advance played text according to frontend playback confirmations."""
        del event
        if not self._generated_chunk_ms:
            return
        remaining_ms = max(0.0, self._generated_chunk_ms.popleft())
        self._played_without_segment_ms += remaining_ms
        await self._apply_pending_playback_time()

    def _build_reported_text(self) -> str:
        """Return the current played text prefix across completed and active segments."""

        if not self._segments:
            return self._completed_text
        segment = self._segments[0]
        if not segment.text:
            return self._completed_text
        if segment.total_audio_ms <= 0.0:
            return self._completed_text + segment.text

        aligned_prefix = self._build_aligned_segment_prefix(segment)
        if aligned_prefix is not None:
            return self._completed_text + aligned_prefix

        if (
            self._force_alignment_enabled
            and segment.alignment_state in {"pending", "running"}
            and not self._fallback_while_alignment_pending
        ):
            return self._completed_text

        return self._completed_text + self._build_ratio_segment_prefix(segment)

    def _build_ratio_segment_prefix(self, segment: _PlaybackSegment) -> str:
        """Fallback mapping from played audio ratio to a text prefix."""

        ratio = max(0.0, min(1.0, segment.played_audio_ms / segment.total_audio_ms))
        prefix_len = min(len(segment.text), int(len(segment.text) * ratio))
        if ratio > 0.0 and prefix_len == 0:
            prefix_len = 1
        return segment.text[:prefix_len]

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

    def _assign_audio_to_segments(self) -> None:
        """Attach generated audio bytes to synthesized text segments in order."""

        if not self._segments or not self._unassigned_audio_chunks:
            return
        for segment in self._segments:
            self._collect_audio_for_segment(segment)
            self._try_start_segment_alignment(segment)
            if not self._unassigned_audio_chunks:
                break

    def _collect_audio_for_segment(self, segment: _PlaybackSegment) -> None:
        """Move enough FIFO audio into one text segment for alignment."""

        if segment.total_audio_ms <= 0.0:
            return
        while (
            self._unassigned_audio_chunks
            and segment.collected_audio_ms + 1e-6 < segment.total_audio_ms
        ):
            remaining_ms = segment.total_audio_ms - segment.collected_audio_ms
            chunk = self._unassigned_audio_chunks.popleft()
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

            if chunk.duration_ms <= remaining_ms + 1e-6:
                if segment.alignment_audio_valid:
                    segment.audio_parts.append(chunk.audio_chunk)
                segment.collected_audio_ms += chunk.duration_ms
                continue

            taken, remainder = self._split_audio_chunk_at_duration(
                chunk=chunk,
                duration_ms=remaining_ms,
            )
            if taken is not None:
                if segment.alignment_audio_valid:
                    segment.audio_parts.append(taken.audio_chunk)
                segment.collected_audio_ms += taken.duration_ms
            if remainder is not None:
                self._unassigned_audio_chunks.appendleft(remainder)
            break

    def _try_start_segment_alignment(self, segment: _PlaybackSegment) -> None:
        """Start forced alignment once a segment has its full audio."""

        if not self._force_alignment_enabled or self.force_aligner is None:
            return
        if segment.alignment_state != "pending":
            return
        if segment.total_audio_ms <= 0.0:
            segment.alignment_state = "failed"
            return
        if segment.collected_audio_ms + 1e-6 < segment.total_audio_ms:
            return
        if (
            not segment.alignment_audio_valid
            or not segment.audio_parts
            or not segment.audio_sample_rate
        ):
            segment.alignment_state = "failed"
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

        assert self.force_aligner is not None
        try:
            audio = b"".join(segment.audio_parts)
            units = await self.force_aligner.async_align(
                audio=audio,
                text=segment.text,
                sample_rate=segment.audio_sample_rate or 48000,
                language=self._force_alignment_language,
            )
            if turn_id != self._turn_id or segment.turn_id != self._turn_id:
                return
            normalized_units = self._normalize_alignment_units(segment.text, units)
            segment.alignment_units = normalized_units
            segment.alignment_state = "ready" if normalized_units else "failed"
            logger.info(
                "TTS forced alignment completed - session: %s, units: %s, state: %s",
                self.session_id,
                len(normalized_units),
                segment.alignment_state,
            )
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
        units: list[ForceAlignmentUnit],
    ) -> list[ForceAlignmentUnit]:
        """Map model-returned units onto character spans in the original text."""

        normalized: list[ForceAlignmentUnit] = []
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
                ForceAlignmentUnit(
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
        unit: ForceAlignmentUnit,
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

    def _split_audio_chunk_at_duration(
        self,
        *,
        chunk: _AudioChunk,
        duration_ms: float,
    ) -> tuple[_AudioChunk | None, _AudioChunk | None]:
        """Split one PCM chunk into prefix/remainder by duration."""

        if duration_ms <= 0.0:
            return None, chunk
        sample_count = max(1, round(chunk.sample_rate * duration_ms / 1000.0))
        split_bytes = min(len(chunk.audio_chunk), sample_count * 2)
        split_bytes -= split_bytes % 2
        if split_bytes <= 0:
            return None, chunk
        if split_bytes >= len(chunk.audio_chunk):
            return chunk, None

        taken_audio = chunk.audio_chunk[:split_bytes]
        remainder_audio = chunk.audio_chunk[split_bytes:]
        taken = _AudioChunk(
            audio_chunk=taken_audio,
            sample_rate=chunk.sample_rate,
            duration_ms=self._chunk_duration_ms(
                audio_chunk=taken_audio,
                sample_rate=chunk.sample_rate,
            ),
        )
        remainder = _AudioChunk(
            audio_chunk=remainder_audio,
            sample_rate=chunk.sample_rate,
            duration_ms=self._chunk_duration_ms(
                audio_chunk=remainder_audio,
                sample_rate=chunk.sample_rate,
            ),
        )
        return taken, remainder

    async def _apply_pending_playback_time(self) -> None:
        """Apply queued played-audio time to known text segments."""

        remaining_ms = max(0.0, self._played_without_segment_ms)
        if remaining_ms <= 0.0:
            return
        while remaining_ms > 0.0 and self._segments:
            segment = self._segments[0]
            if segment.total_audio_ms <= 0.0:
                self._completed_text += segment.text
                self._segments.popleft()
                continue

            unplayed_ms = max(0.0, segment.total_audio_ms - segment.played_audio_ms)
            consume_ms = min(remaining_ms, unplayed_ms)
            segment.played_audio_ms += consume_ms
            remaining_ms -= consume_ms

            if segment.played_audio_ms + 1e-6 >= segment.total_audio_ms:
                self._completed_text += segment.text
                self._segments.popleft()

        self._played_without_segment_ms = remaining_ms
        await self._publish_progress_if_grew()

    async def _publish_progress_if_grew(self) -> None:
        """Publish one response update when the played text prefix grows."""

        played_text = self._build_reported_text()
        if len(played_text) <= len(self._reported_text):
            return
        self._reported_text = played_text
        await self.event_bus.publish(
            ResponseUpdate(
                session_id=self.session_id,
                text=played_text,
            )
        )

    @Manager.event_handler(LLMAgentResponseFinish, priority=20)
    async def _cache_response_finish(self, event: LLMAgentResponseFinish) -> None:
        """Cache the latest generated response until playback finishes."""

        next_text = event.text or ""
        if self._current_response_text and not next_text.startswith(
            self._current_response_text
        ):
            self._reset_all_state()
        self._current_response_text = next_text
        self._pending_text = next_text

    @Manager.event_handler(TTSPlaybackFinished, priority=20)
    async def _publish_response_finish(self, event: TTSPlaybackFinished) -> None:
        """Publish response-finish after frontend playback completion."""

        del event
        if not self._received_audio:
            logger.warning(
                "TTS playback finished without generated audio; discarding unplayed "
                "response - session: %s",
                self.session_id,
            )
            self._reset_all_state()
            return
        played_text = self._build_reported_text()
        if len(self._reported_text) > len(played_text):
            played_text = self._reported_text
        if not played_text:
            logger.warning(
                "TTS playback finished without playback-confirmed text - session: %s",
                self.session_id,
            )
            self._reset_all_state()
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
            self._reset_all_state()

    @Manager.event_handler(TTSStopped, priority=20)
    async def _handle_tts_stopped(self, event: TTSStopped) -> None:
        """Wait for the frontend's exact playback position after an early stop."""

        del event
        self._awaiting_stop_ack = True
        self._cancel_stop_commit_task()
        turn_id = self._turn_id
        self._stop_commit_task = asyncio.create_task(
            self._commit_stopped_playback_after_timeout(turn_id)
        )

    @Manager.event_handler(TTSPlaybackStopped, priority=20)
    async def _handle_tts_playback_stopped(
        self,
        event: TTSPlaybackStopped,
    ) -> None:
        """Apply unacknowledged browser playback before committing interruption."""

        if not self._awaiting_stop_ack:
            return
        self._cancel_stop_commit_task()
        self._played_without_segment_ms += max(0.0, event.played_audio_ms)
        await self._apply_pending_playback_time()
        await self._commit_stopped_playback()

    async def _commit_stopped_playback_after_timeout(self, turn_id: int) -> None:
        """Fall back to completed-chunk progress for older frontend clients."""

        try:
            await asyncio.sleep(self._stop_ack_timeout_ms / 1000.0)
            if turn_id != self._turn_id or not self._awaiting_stop_ack:
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

        played_text = self._build_reported_text()
        if len(self._reported_text) > len(played_text):
            played_text = self._reported_text
        try:
            if played_text:
                await self._commit_playback_text(played_text)
        finally:
            self._reset_all_state()

    async def _commit_playback_text(self, text: str) -> None:
        """Publish the final playback-confirmed text for the active turn."""

        if self._reported_text != text:
            await self.event_bus.publish(
                ResponseUpdate(
                    session_id=self.session_id,
                    text=text,
                )
            )
            self._reported_text = text
        await self.event_bus.publish(
            ResponseFinish(
                session_id=self.session_id,
                text=text,
            )
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
