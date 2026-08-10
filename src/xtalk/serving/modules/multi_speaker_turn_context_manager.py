"""Orchestrate generic speaker diarization and join it with ASR hard turns."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

from ...models import Models, SpeakerDiarization
from ...models.speaker_diarization.interfaces import DiarizationResult
from ..event_bus import EventBus, EventDispatchMode, EventPropagation
from ..events import (
    ASRGateState,
    ASRResultFinal,
    ASRResultPartial,
    EnhancedAudioFrameReceived,
    MultiSpeakerTurnReady,
    SpeakerDiarizationPartial,
    SpeakerDiarizationSegmentFinal,
    SpeakerDiarizationTurnFinal,
    TurnASREndRequested,
    TurnASRPauseRequested,
    TurnASRStartRequested,
    VADSpeechEnd,
)
from ..interfaces import Manager
from ..multi_speaker_config import speaker_history_gate_enabled

logger = logging.getLogger(__name__)


def _render_timeline(
    segments: list[Mapping[str, Any]],
    *,
    precision: int = 2,
) -> str:
    """Render structured speaker segments for events and agent context."""

    return " ".join(
        f"[{float(item['start_s']):.{precision}f}]"
        f"[{str(item['speaker_id'])}]"
        f"{str(item.get('text') or '')}"
        f"[{float(item['end_s']):.{precision}f}]"
        for item in segments
    )


def _offset_segments(
    segments: list[Mapping[str, Any]],
    *,
    offset_s: float,
    turn_id: int,
    segment_id: int,
) -> list[dict[str, Any]]:
    """Convert snapshot-local segments to the session-source timeline."""

    return [
        {
            **item,
            "start_s": float(item["start_s"]) + offset_s,
            "end_s": float(item["end_s"]) + offset_s,
            "turn_id": turn_id,
            "segment_id": segment_id,
        }
        for item in segments
    ]


@dataclass
class _DiarizationSegmentState:
    """Mutable PCM and scheduling state for one VAD segment."""

    turn_id: int
    segment_id: int
    source_start_sample: int
    pcm: bytearray
    revision: int = 0
    next_partial_s: float = 0.8
    final_requested: bool = False
    last_partial_text: str = ""
    published_partial_revision: int = 0


@dataclass(frozen=True)
class _DiarizationSnapshotRequest:
    """Immutable snapshot queued for the session-local worker."""

    request_id: str
    turn_id: int
    segment_id: int
    revision: int
    source_start_sample: int
    current_pcm16: bytes
    is_final: bool


@dataclass
class _DiarizationTurnState:
    """Track diarization segment terminals for one ASR hard turn."""

    segment_ids: list[int] = field(default_factory=list)
    finals: dict[int, SpeakerDiarizationSegmentFinal] = field(default_factory=dict)
    hard_closed: bool = False
    turn_final_published: bool = False


class MultiSpeakerTurnContextManager(Manager):
    """Schedule generic diarization and join its turn results with ASR."""

    BYTES_PER_SAMPLE = 2
    SAMPLE_RATE = 16000

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config = config or {}
        multi_config = dict(self.config.get("multi_speaker") or {})
        self.model = models.get(SpeakerDiarization)
        self.enabled = self.model is not None
        self.history_gate_enabled = speaker_history_gate_enabled(models, self.config)
        self.response_policy = str(
            multi_config.get("response_policy", "focus_only")
        )
        self.focus_speaker_ids = {
            str(item) for item in multi_config.get("focus_speaker_ids", ["S01"])
        }
        self.suppress_when_speaker_missing = bool(
            multi_config.get("suppress_when_speaker_missing", False)
        )
        self.join_timeout_s = float(multi_config.get("join_timeout_s", 5.0))
        self.fallback_on_timeout = bool(multi_config.get("fallback_on_timeout", True))

        diarization_config = dict(multi_config.get("diarization") or {})
        self.pre_buffer_s = float(diarization_config.get("pre_buffer_s", 1.0))
        partial_config = dict(diarization_config.get("partial") or {})
        self.partial_interval_s = float(partial_config.get("interval_s", 1.0))
        self.first_partial_min_s = float(partial_config.get("first_partial_min_s", 0.8))

        self._sample_cursor = 0
        self._pre_buffer = bytearray()
        self._pre_buffer_max_bytes = round(
            self.pre_buffer_s * self.SAMPLE_RATE * self.BYTES_PER_SAMPLE
        )
        self._active_segment: _DiarizationSegmentState | None = None
        self._segments: dict[int, _DiarizationSegmentState] = {}
        self._diarization_turns: dict[int, _DiarizationTurnState] = {}
        self._pending_partial: _DiarizationSnapshotRequest | None = None
        self._final_queue: deque[_DiarizationSnapshotRequest] = deque()
        self._pending_event = asyncio.Event()
        self._in_flight: _DiarizationSnapshotRequest | None = None
        self._worker_task = (
            asyncio.create_task(self._worker()) if self.enabled else None
        )

        self._asr_finals: dict[int, ASRResultFinal] = {}
        self._diarization_finals: dict[int, SpeakerDiarizationTurnFinal] = {}
        self._published: set[int] = set()
        self._timeout_tasks: dict[int, asyncio.Task[None]] = {}
        self._turns_with_non_focus_speech: set[int] = set()

    @Manager.event_handler(
        EnhancedAudioFrameReceived,
        priority=-20,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Copy enhanced PCM into a bounded pre-buffer or active segment."""

        if not self.enabled or not event.audio_data:
            return
        if event.sample_rate != self.SAMPLE_RATE:
            logger.warning(
                "Diarization ignored PCM with sample rate %s; expected %s - session: %s",
                event.sample_rate,
                self.SAMPLE_RATE,
                self.session_id,
            )
            return
        pcm = event.audio_data
        self._sample_cursor += len(pcm) // self.BYTES_PER_SAMPLE
        if self._active_segment is None:
            self._pre_buffer.extend(pcm)
            if len(self._pre_buffer) > self._pre_buffer_max_bytes:
                del self._pre_buffer[
                    : len(self._pre_buffer) - self._pre_buffer_max_bytes
                ]
            return
        self._active_segment.pcm.extend(pcm)
        self._maybe_schedule_partial(self._active_segment)

    @Manager.event_handler(
        TurnASRStartRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_segment_start(self, event: TurnASRStartRequested) -> None:
        """Open a diarization segment using the same IDs and pre-buffer as ASR."""

        if not self.enabled:
            return
        if self._active_segment is not None:
            logger.warning(
                "Diarization received a new segment before the previous segment closed - session: %s",
                self.session_id,
            )
            await self._freeze_and_schedule_final(self._active_segment)
        pre_buffer = bytes(self._pre_buffer)
        source_start = max(
            0,
            self._sample_cursor - len(pre_buffer) // self.BYTES_PER_SAMPLE,
        )
        state = _DiarizationSegmentState(
            turn_id=event.turn_id,
            segment_id=event.segment_id,
            source_start_sample=source_start,
            pcm=bytearray(pre_buffer),
            next_partial_s=self.first_partial_min_s,
        )
        self._active_segment = state
        self._segments[event.segment_id] = state
        turn = self._diarization_turns.setdefault(
            event.turn_id,
            _DiarizationTurnState(),
        )
        if event.segment_id not in turn.segment_ids:
            turn.segment_ids.append(event.segment_id)
        self._pre_buffer.clear()
        self._maybe_schedule_partial(state)

    @Manager.event_handler(
        VADSpeechEnd,
        priority=100,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_vad_end(self, _event: VADSpeechEnd) -> None:
        """Freeze the current VAD segment before hard-turn control arrives."""

        if not self.enabled or self._active_segment is None:
            return
        state = self._active_segment
        self._active_segment = None
        await self._freeze_and_schedule_final(state)

    @Manager.event_handler(
        TurnASRPauseRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_segment_pause(self, event: TurnASRPauseRequested) -> None:
        """Freeze one VAD segment while leaving its ASR turn open."""

        if self.enabled:
            await self._close_matching_segment(event.turn_id, event.segment_id)

    @Manager.event_handler(
        TurnASREndRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_turn_end(self, event: TurnASREndRequested) -> None:
        """Freeze the final segment and mark the diarization turn hard-closed."""

        if not self.enabled:
            return
        await self._close_matching_segment(event.turn_id, event.segment_id)
        turn = self._diarization_turns.setdefault(
            event.turn_id,
            _DiarizationTurnState(),
        )
        turn.hard_closed = True
        await self._maybe_publish_turn_final(event.turn_id)

    async def _close_matching_segment(self, turn_id: int, segment_id: int) -> None:
        """Freeze the active matching segment and schedule a final snapshot."""

        state = self._active_segment
        if state is None or state.turn_id != turn_id or state.segment_id != segment_id:
            return
        self._active_segment = None
        await self._freeze_and_schedule_final(state)

    async def _freeze_and_schedule_final(
        self,
        state: _DiarizationSegmentState,
    ) -> None:
        """Promote one immutable segment snapshot above queued partials."""

        if state.final_requested:
            return
        state.final_requested = True
        state.revision += 1
        request = self._make_request(state, is_final=True)
        self._pending_partial = None
        self._final_queue.append(request)
        self._pending_event.set()
        if (
            self._in_flight is not None
            and not self._in_flight.is_final
            and self.model is not None
        ):
            asyncio.create_task(self._best_effort_cancel(self._in_flight.request_id))

    def _maybe_schedule_partial(self, state: _DiarizationSegmentState) -> None:
        """Queue the latest complete VAD snapshot at the configured cadence."""

        if state.final_requested:
            return
        duration_s = len(state.pcm) / (self.SAMPLE_RATE * self.BYTES_PER_SAMPLE)
        if duration_s + 1e-9 < state.next_partial_s:
            return
        state.revision += 1
        self._pending_partial = self._make_request(state, is_final=False)
        self._pending_event.set()
        while state.next_partial_s <= duration_s + 1e-9:
            state.next_partial_s += self.partial_interval_s

    def _make_request(
        self,
        state: _DiarizationSegmentState,
        *,
        is_final: bool,
    ) -> _DiarizationSnapshotRequest:
        """Copy mutable segment PCM into an immutable worker request."""

        kind = "final" if is_final else "partial"
        request_id = (
            f"{self.session_id}/{state.turn_id}/{state.segment_id}/"
            f"{state.revision}/{kind}"
        )
        return _DiarizationSnapshotRequest(
            request_id=request_id,
            turn_id=state.turn_id,
            segment_id=state.segment_id,
            revision=state.revision,
            source_start_sample=state.source_start_sample,
            current_pcm16=bytes(state.pcm),
            is_final=is_final,
        )

    async def _worker(self) -> None:
        """Process one diarization request at a time with latest-only partials."""

        try:
            while True:
                await self._pending_event.wait()
                request: _DiarizationSnapshotRequest | None
                if self._final_queue:
                    request = self._final_queue.popleft()
                else:
                    request = self._pending_partial
                    self._pending_partial = None
                if not self._final_queue and self._pending_partial is None:
                    self._pending_event.clear()
                if request is None:
                    continue
                self._in_flight = request
                try:
                    await self._decode_and_publish(request)
                finally:
                    self._in_flight = None
                    if self._final_queue or self._pending_partial is not None:
                        self._pending_event.set()
        except asyncio.CancelledError:
            raise

    async def _decode_and_publish(
        self,
        request: _DiarizationSnapshotRequest,
    ) -> None:
        """Decode one generic snapshot and publish its result."""

        if self.model is None:
            if request.is_final:
                await self._publish_degraded_final(request, "model_not_configured")
            return
        try:
            result = await self.model.decode_snapshot(
                request_id=request.request_id,
                pcm16=request.current_pcm16,
                sample_rate=self.SAMPLE_RATE,
                is_final=request.is_final,
            )
        except Exception as exc:
            logger.warning(
                "Diarization snapshot failed - session: %s, request: %s, error: %s",
                self.session_id,
                request.request_id,
                exc,
            )
            if request.is_final:
                await self._publish_degraded_final(request, type(exc).__name__)
            return
        state = self._segments.get(request.segment_id)
        if state is None or (not request.is_final and state.final_requested):
            return
        if request.is_final:
            await self._publish_segment_final(request, result)
            return
        if request.revision <= state.published_partial_revision:
            return
        segments = [dict(item) for item in result.segments]
        self._observe_diarization_segments(request.turn_id, segments)
        diarization_text = _render_timeline(segments)
        if diarization_text == state.last_partial_text:
            return
        state.last_partial_text = diarization_text
        state.published_partial_revision = request.revision
        await self.event_bus.publish(
            SpeakerDiarizationPartial(
                session_id=self.session_id,
                turn_id=request.turn_id,
                segment_id=request.segment_id,
                revision=request.revision,
                source_start_sample=request.source_start_sample,
                source_end_sample=request.source_start_sample
                + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
                sample_rate=self.SAMPLE_RATE,
                raw_text=result.raw_text,
                diarization_text=diarization_text,
                segments=segments,
                latency_ms=result.latency_ms,
            )
        )

    async def _publish_segment_final(
        self,
        request: _DiarizationSnapshotRequest,
        result: DiarizationResult,
    ) -> None:
        """Publish one successful terminal segment."""

        segments = [dict(item) for item in result.segments]
        event = SpeakerDiarizationSegmentFinal(
            session_id=self.session_id,
            turn_id=request.turn_id,
            segment_id=request.segment_id,
            source_start_sample=request.source_start_sample,
            source_end_sample=request.source_start_sample
            + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
            sample_rate=self.SAMPLE_RATE,
            raw_text=result.raw_text,
            diarization_text=_render_timeline(segments),
            segments=segments,
            metrics=result.metrics,
            latency_ms=result.latency_ms,
        )
        await self._record_segment_final(event)

    async def _publish_degraded_final(
        self,
        request: _DiarizationSnapshotRequest,
        reason: str,
    ) -> None:
        """Publish an empty terminal result so the turn join cannot deadlock."""

        await self._record_segment_final(
            SpeakerDiarizationSegmentFinal(
                session_id=self.session_id,
                turn_id=request.turn_id,
                segment_id=request.segment_id,
                source_start_sample=request.source_start_sample,
                source_end_sample=request.source_start_sample
                + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
                sample_rate=self.SAMPLE_RATE,
                degraded=True,
                degraded_reason=reason,
            )
        )

    async def _record_segment_final(
        self,
        event: SpeakerDiarizationSegmentFinal,
    ) -> None:
        """Record and publish a segment terminal before testing the turn barrier."""

        self._observe_diarization_segments(event.turn_id, event.segments)
        turn = self._diarization_turns.setdefault(
            event.turn_id,
            _DiarizationTurnState(),
        )
        turn.finals[event.segment_id] = event
        await self.event_bus.publish(event)
        await self._maybe_publish_turn_final(event.turn_id)

    async def _maybe_publish_turn_final(self, turn_id: int) -> None:
        """Publish one ordered turn timeline after its barrier is complete."""

        turn = self._diarization_turns.get(turn_id)
        if turn is None or not turn.hard_closed or turn.turn_final_published:
            return
        if any(segment_id not in turn.finals for segment_id in turn.segment_ids):
            return
        timeline: list[dict[str, Any]] = []
        degraded_reasons: list[str] = []
        for segment_id in turn.segment_ids:
            final = turn.finals[segment_id]
            timeline.extend(
                _offset_segments(
                    final.segments,
                    offset_s=final.source_start_sample / final.sample_rate,
                    turn_id=turn_id,
                    segment_id=segment_id,
                )
            )
            if final.degraded:
                degraded_reasons.append(final.degraded_reason)
        timeline.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"])))
        active_speaker_id = next(
            (
                str(item["speaker_id"])
                for item in reversed(timeline)
                if str(item.get("text") or "").strip()
            ),
            None,
        )
        turn.turn_final_published = True
        event = SpeakerDiarizationTurnFinal(
            session_id=self.session_id,
            turn_id=turn_id,
            segment_ids=list(turn.segment_ids),
            segments=timeline,
            diarization_text=_render_timeline(timeline),
            active_speaker_id=active_speaker_id,
            degraded=bool(degraded_reasons),
            degraded_reason=",".join(dict.fromkeys(degraded_reasons)),
        )
        self._observe_diarization_segments(turn_id, timeline)
        await self.event_bus.publish(event)

    def _observe_diarization_segments(
        self,
        turn_id: int,
        segments: list[Mapping[str, Any]],
    ) -> None:
        """Close the live ASR preview after any identified non-focus speaker."""

        if not self.history_gate_enabled:
            return
        for segment in segments:
            speaker = segment.get("speaker_id")
            if speaker is None:
                continue
            speaker_id = str(speaker).strip()
            if speaker_id and speaker_id not in self.focus_speaker_ids:
                self._turns_with_non_focus_speech.add(turn_id)
                return

    @Manager.event_handler(
        ASRResultPartial,
        priority=60,
        enabled_if=lambda manager: manager.history_gate_enabled,
    )
    async def _gate_asr_partial_preview(
        self,
        event: ASRResultPartial,
    ) -> EventPropagation:
        """Stop new frontend previews after non-focus speech is observed."""

        if event.origin == "text" or event.gate_state is ASRGateState.ACCEPTED:
            return EventPropagation.CONTINUE
        try:
            if event.turn_id in self._turns_with_non_focus_speech:
                return EventPropagation.STOP
            return EventPropagation.CONTINUE
        except Exception:
            logger.exception(
                "Failed to evaluate ASR partial speaker gate - session: %s, turn: %s",
                self.session_id,
                event.turn_id,
            )
            return EventPropagation.STOP

    @Manager.event_handler(
        ASRResultPartial,
        priority=30,
        enabled_if=lambda manager: manager.history_gate_enabled,
    )
    async def _block_asr_partial_from_history(
        self,
        event: ASRResultPartial,
    ) -> EventPropagation:
        """Keep audio ASR partials out of Agent and event history consumers."""

        if event.origin == "text" or event.gate_state is ASRGateState.ACCEPTED:
            return EventPropagation.CONTINUE
        return EventPropagation.STOP

    @Manager.event_handler(
        ASRResultFinal,
        priority=30,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_asr_final(
        self,
        event: ASRResultFinal,
    ) -> EventPropagation:
        """Cache ASR text and start a bounded diarization join wait."""

        if (
            not self.enabled
            or event.origin == "text"
            or event.gate_state is ASRGateState.ACCEPTED
        ):
            return EventPropagation.CONTINUE
        try:
            self._asr_finals[event.turn_id] = event
            if event.turn_id not in self._timeout_tasks:
                self._timeout_tasks[event.turn_id] = asyncio.create_task(
                    self._wait_for_timeout(event.turn_id)
                )
            await self._try_publish(event.turn_id)
        except Exception:
            logger.exception(
                "Failed to evaluate ASR final speaker gate - session: %s, turn: %s",
                self.session_id,
                event.turn_id,
            )
            if self.history_gate_enabled:
                return EventPropagation.STOP
        if self.history_gate_enabled:
            return EventPropagation.STOP
        return EventPropagation.CONTINUE

    @Manager.event_handler(
        SpeakerDiarizationTurnFinal,
        priority=30,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_diarization_final(
        self,
        event: SpeakerDiarizationTurnFinal,
    ) -> None:
        """Cache the complete speaker timeline and attempt the ASR join."""

        if not self.enabled:
            return
        self._observe_diarization_segments(event.turn_id, event.segments)
        self._diarization_finals[event.turn_id] = event
        try:
            await self._try_publish(event.turn_id)
        except Exception:
            logger.exception(
                "Failed to join diarization with ASR - session: %s, turn: %s",
                self.session_id,
                event.turn_id,
            )

    async def _try_publish(self, turn_id: int) -> None:
        """Publish a combined event when both independent results exist."""

        if turn_id in self._published:
            return
        asr_event = self._asr_finals.get(turn_id)
        diarization_event = self._diarization_finals.get(turn_id)
        if asr_event is None or diarization_event is None:
            return
        await self._publish_ready(asr_event, diarization_event)

    async def _wait_for_timeout(self, turn_id: int) -> None:
        """Apply the configured ASR-only fallback after the join timeout."""

        try:
            await asyncio.sleep(self.join_timeout_s)
            if turn_id in self._published:
                return
            asr_event = self._asr_finals.get(turn_id)
            if asr_event is None:
                return
            if not self.fallback_on_timeout:
                if self.history_gate_enabled:
                    self._finish_turn(turn_id)
                return
            await self._publish_ready(asr_event, None, timeout=True)
        except asyncio.CancelledError:
            raise

    async def _publish_ready(
        self,
        asr_event: ASRResultFinal,
        diarization_event: SpeakerDiarizationTurnFinal | None,
        *,
        timeout: bool = False,
    ) -> None:
        """Resolve response policy and emit one idempotent joined event."""

        turn_id = asr_event.turn_id
        if turn_id in self._published:
            return
        active_speaker_id = (
            diarization_event.active_speaker_id
            if diarization_event is not None
            else None
        )
        should_respond = self._should_respond(active_speaker_id)
        asr_text = asr_event.text
        diarization_segments = (
            [dict(item) for item in diarization_event.segments]
            if diarization_event is not None
            else []
        )
        diarization_text = (
            diarization_event.diarization_text if diarization_event is not None else ""
        )
        history_active_speaker_id = active_speaker_id
        degraded_reasons: list[str] = []
        if timeout:
            degraded_reasons.append("join_timeout")
        if diarization_event is not None and diarization_event.degraded:
            degraded_reasons.append(diarization_event.degraded_reason)

        if self.history_gate_enabled:
            filtered = self._filter_focus_history(asr_event, diarization_event)
            if filtered is None:
                self._finish_turn(turn_id)
                return
            (
                asr_text,
                diarization_segments,
                diarization_text,
                history_active_speaker_id,
            ) = filtered

        self._finish_turn(turn_id)
        if self.history_gate_enabled:
            accepted = replace(
                asr_event,
                text=asr_text,
                display_text=asr_text,
                gate_state=ASRGateState.ACCEPTED,
            )
            published = await self.event_bus.publish(
                accepted,
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE_OR_STOPPED,
            )
            if not published:
                logger.error(
                    "Failed to publish accepted ASR final - session: %s, turn: %s",
                    self.session_id,
                    turn_id,
                )
                return
        await self.event_bus.publish(
            MultiSpeakerTurnReady(
                session_id=self.session_id,
                turn_id=turn_id,
                asr_text=asr_text,
                diarization_text=diarization_text,
                diarization_segments=diarization_segments,
                active_speaker_id=history_active_speaker_id,
                should_respond=should_respond,
                degraded=bool(degraded_reasons),
                degraded_reason=",".join(
                    reason for reason in dict.fromkeys(degraded_reasons) if reason
                ),
            )
        )

    def _filter_focus_history(
        self,
        asr_event: ASRResultFinal,
        diarization_event: SpeakerDiarizationTurnFinal | None,
    ) -> tuple[str, list[dict[str, Any]], str, str | None] | None:
        """Select only history-safe focus content for one completed turn."""

        if (
            diarization_event is None
            and asr_event.turn_id in self._turns_with_non_focus_speech
        ):
            return None

        segments = (
            [dict(item) for item in diarization_event.segments]
            if diarization_event is not None
            else []
        )
        active_speaker_id = (
            diarization_event.active_speaker_id
            if diarization_event is not None
            else None
        )
        focus_segments: list[dict[str, Any]] = []
        unknown_segments: list[dict[str, Any]] = []
        non_focus_seen = False
        identified_speakers: set[str] = set()
        for segment in segments:
            speaker = segment.get("speaker_id")
            speaker_id = str(speaker).strip() if speaker is not None else ""
            if not speaker_id:
                unknown_segments.append(segment)
                continue
            identified_speakers.add(speaker_id)
            if speaker_id in self.focus_speaker_ids:
                focus_segments.append(segment)
            else:
                non_focus_seen = True

        if not identified_speakers:
            if active_speaker_id is not None:
                normalized_active = str(active_speaker_id).strip()
                if normalized_active:
                    if normalized_active not in self.focus_speaker_ids:
                        return None
                    return (
                        asr_event.text,
                        segments,
                        _render_timeline(segments),
                        normalized_active,
                    )
            if self.suppress_when_speaker_missing:
                return None
            return asr_event.text, segments, _render_timeline(segments), None

        must_filter_segments = non_focus_seen or (
            bool(unknown_segments) and self.suppress_when_speaker_missing
        )
        if not must_filter_segments:
            return (
                asr_event.text,
                segments,
                _render_timeline(segments),
                active_speaker_id,
            )
        if not focus_segments:
            return None

        focus_text = " ".join(
            str(segment.get("text") or "").strip()
            for segment in focus_segments
            if str(segment.get("text") or "").strip()
        )
        if not focus_text:
            return None
        filtered_active_speaker_id = next(
            (
                str(segment["speaker_id"]).strip()
                for segment in reversed(focus_segments)
                if str(segment.get("speaker_id") or "").strip()
            ),
            None,
        )
        return (
            focus_text,
            focus_segments,
            _render_timeline(focus_segments),
            filtered_active_speaker_id,
        )

    def _finish_turn(self, turn_id: int) -> None:
        """Mark one joined turn complete and release its temporary gate state."""

        self._published.add(turn_id)
        timeout_task = self._timeout_tasks.pop(turn_id, None)
        if timeout_task is not None and timeout_task is not asyncio.current_task():
            timeout_task.cancel()
        self._asr_finals.pop(turn_id, None)
        self._diarization_finals.pop(turn_id, None)
        self._turns_with_non_focus_speech.discard(turn_id)

    def _should_respond(self, active_speaker_id: str | None) -> bool:
        """Return whether response generation is allowed for the active speaker."""

        if self.response_policy != "focus_only":
            return True
        if active_speaker_id is None:
            return not self.suppress_when_speaker_missing
        return active_speaker_id in self.focus_speaker_ids

    async def _best_effort_cancel(self, request_id: str) -> None:
        """Cancel an obsolete partial without propagating model failures."""

        if self.model is None:
            return
        try:
            await self.model.cancel(request_id)
        except Exception as exc:
            logger.debug("Diarization cancel failed for %s: %s", request_id, exc)

    async def shutdown(self) -> None:
        """Cancel session tasks and release the diarization model clone."""

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self.model is not None:
            await self.model.close()
        timeout_tasks = list(self._timeout_tasks.values())
        for task in timeout_tasks:
            task.cancel()
        if timeout_tasks:
            await asyncio.gather(*timeout_tasks, return_exceptions=True)
        self._timeout_tasks.clear()
