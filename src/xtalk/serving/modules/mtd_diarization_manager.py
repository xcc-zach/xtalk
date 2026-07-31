"""Session manager for VAD-internal MTD full-snapshot decoding."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from ...log_utils import logger
from ...models import Models, SpeakerDiarization
from ...models.speaker_diarization.interfaces import DiarizationResult
from ..event_bus import EventBus
from ..events import (
    EnhancedAudioFrameReceived,
    SpeakerDiarizationPartial,
    SpeakerDiarizationSegmentFinal,
    SpeakerDiarizationTurnFinal,
    TurnASREndRequested,
    TurnASRPauseRequested,
    TurnASRStartRequested,
    VADSpeechEnd,
)
from ..interfaces import Manager
from ..mtd.audio_layout import build_audio_layout
from ..mtd.exemplar_pool import SpeakerExemplarPool
from ..mtd.transcript import offset_segments, render_segments


_DEFAULT_POOL_CONFIG: dict[str, Any] = {
    "max_speakers": 16,
    "min_register_duration_s": 0.70,
    "min_update_duration_s": 0.45,
    "preferred_min_duration_s": 1.0,
    "preferred_max_duration_s": 7.0,
    "min_rms_dbfs": -42.0,
    "max_clipping_ratio": 0.01,
    "min_boundary_margin_s": 0.08,
    "replace_score_margin": 0.08,
    "score_weights": {
        "duration": 0.35,
        "rms": 0.25,
        "non_overlap": 0.25,
        "boundary": 0.10,
        "unclipped": 0.05,
    },
}


@dataclass
class _SegmentState:
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
class _SnapshotRequest:
    """Immutable snapshot queued for the single per-session worker."""

    request_id: str
    turn_id: int
    segment_id: int
    revision: int
    source_start_sample: int
    current_pcm16: bytes
    is_final: bool


@dataclass
class _TurnState:
    """Track VAD segment terminals for one ASR hard turn."""

    segment_ids: list[int] = field(default_factory=list)
    finals: dict[int, SpeakerDiarizationSegmentFinal] = field(default_factory=dict)
    hard_closed: bool = False
    turn_final_published: bool = False


class MtdDiarizationManager(Manager):
    """Run MTD snapshots without blocking Xtalk's serialized audio frame chain."""

    BYTES_PER_SAMPLE = 2

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
        self.enabled = bool(multi_config.get("enabled", False))
        self.model = models.get(SpeakerDiarization) if self.enabled else None
        mtd_config = dict(self.config.get("mtd") or {})
        self.sample_rate = int(mtd_config.get("sample_rate", 16000))
        self.pre_buffer_s = float(mtd_config.get("pre_buffer_s", 1.0))
        partial_config = dict(mtd_config.get("partial") or {})
        self.partial_interval_s = float(partial_config.get("interval_s", 1.0))
        self.first_partial_min_s = float(
            partial_config.get("first_partial_min_s", 0.8)
        )
        self.publish_unchanged = bool(partial_config.get("publish_unchanged", False))
        self.abort_on_vad_end = bool(partial_config.get("abort_on_vad_end", True))
        layout_config = dict(mtd_config.get("audio_layout") or {})
        self.inter_exemplar_silence_s = float(
            layout_config.get("inter_exemplar_silence_s", 0.5)
        )
        self.exemplar_to_current_silence_s = float(
            layout_config.get("exemplar_to_current_silence_s", 1.0)
        )
        pool_config = _merge_nested_dict(
            _DEFAULT_POOL_CONFIG,
            dict(mtd_config.get("pool") or {}),
        )
        self.pool = SpeakerExemplarPool(pool_config)
        self._sample_cursor = 0
        self._pre_buffer = bytearray()
        self._pre_buffer_max_bytes = round(
            self.pre_buffer_s * self.sample_rate * self.BYTES_PER_SAMPLE
        )
        self._active_segment: _SegmentState | None = None
        self._segments: dict[int, _SegmentState] = {}
        self._turns: dict[int, _TurnState] = {}
        self._pending_partial: _SnapshotRequest | None = None
        self._final_queue: deque[_SnapshotRequest] = deque()
        self._pending_event = asyncio.Event()
        self._in_flight: _SnapshotRequest | None = None
        self._worker_task = asyncio.create_task(self._worker()) if self.enabled else None
        if self.enabled and self.model is None:
            logger.warning(
                "MTD multi-speaker mode enabled without a SpeakerDiarization model - session: %s",
                session_id,
            )

    @Manager.event_handler(
        EnhancedAudioFrameReceived,
        priority=-20,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Copy enhanced PCM into a bounded pre-buffer or active segment."""

        if not self.enabled or not event.audio_data:
            return
        if event.sample_rate != self.sample_rate:
            logger.warning(
                "MTD ignored PCM with sample rate %s; expected %s - session: %s",
                event.sample_rate,
                self.sample_rate,
                self.session_id,
            )
            return
        pcm = event.audio_data
        self._sample_cursor += len(pcm) // self.BYTES_PER_SAMPLE
        if self._active_segment is None:
            self._pre_buffer.extend(pcm)
            if len(self._pre_buffer) > self._pre_buffer_max_bytes:
                del self._pre_buffer[: len(self._pre_buffer) - self._pre_buffer_max_bytes]
            return
        self._active_segment.pcm.extend(pcm)
        self._maybe_schedule_partial(self._active_segment)

    @Manager.event_handler(
        TurnASRStartRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_segment_start(self, event: TurnASRStartRequested) -> None:
        """Open a VAD segment using the same ID and pre-buffer as ASR."""

        if not self.enabled:
            return
        if self._active_segment is not None:
            logger.warning(
                "MTD received a new segment before the previous segment closed - session: %s",
                self.session_id,
            )
            await self._freeze_and_schedule_final(self._active_segment)
        pre_buffer = bytes(self._pre_buffer)
        source_start = max(
            0,
            self._sample_cursor - len(pre_buffer) // self.BYTES_PER_SAMPLE,
        )
        state = _SegmentState(
            turn_id=event.turn_id,
            segment_id=event.segment_id,
            source_start_sample=source_start,
            pcm=bytearray(pre_buffer),
            next_partial_s=self.first_partial_min_s,
        )
        self._active_segment = state
        self._segments[event.segment_id] = state
        turn = self._turns.setdefault(event.turn_id, _TurnState())
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

        if not self.enabled:
            return
        await self._close_matching_segment(event.turn_id, event.segment_id)

    @Manager.event_handler(
        TurnASREndRequested,
        priority=-10,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_turn_end(self, event: TurnASREndRequested) -> None:
        """Freeze the final VAD segment and mark the ASR turn hard-closed."""

        if not self.enabled:
            return
        await self._close_matching_segment(event.turn_id, event.segment_id)
        turn = self._turns.setdefault(event.turn_id, _TurnState())
        turn.hard_closed = True
        await self._maybe_publish_turn_final(event.turn_id)

    async def _close_matching_segment(self, turn_id: int, segment_id: int) -> None:
        """Freeze the active matching segment and schedule a final snapshot."""

        state = self._active_segment
        if state is None or state.turn_id != turn_id or state.segment_id != segment_id:
            return
        self._active_segment = None
        await self._freeze_and_schedule_final(state)

    async def _freeze_and_schedule_final(self, state: _SegmentState) -> None:
        """Promote one immutable segment snapshot above all queued partials."""

        if state.final_requested:
            return
        state.final_requested = True
        state.revision += 1
        request = self._make_request(state, is_final=True)
        self._pending_partial = None
        self._final_queue.append(request)
        self._pending_event.set()
        if (
            self.abort_on_vad_end
            and self._in_flight is not None
            and not self._in_flight.is_final
            and self.model is not None
        ):
            asyncio.create_task(self._best_effort_cancel(self._in_flight.request_id))

    def _maybe_schedule_partial(self, state: _SegmentState) -> None:
        """Queue the latest complete VAD snapshot at the configured cadence."""

        if state.final_requested:
            return
        duration_s = len(state.pcm) / (self.sample_rate * self.BYTES_PER_SAMPLE)
        if duration_s + 1e-9 < state.next_partial_s:
            return
        state.revision += 1
        self._pending_partial = self._make_request(state, is_final=False)
        self._pending_event.set()
        while state.next_partial_s <= duration_s + 1e-9:
            state.next_partial_s += self.partial_interval_s

    def _make_request(
        self,
        state: _SegmentState,
        *,
        is_final: bool,
    ) -> _SnapshotRequest:
        """Copy mutable segment PCM into an immutable worker request."""

        kind = "final" if is_final else "partial"
        request_id = (
            f"{self.session_id}/{state.turn_id}/{state.segment_id}/"
            f"{state.revision}/{kind}"
        )
        return _SnapshotRequest(
            request_id=request_id,
            turn_id=state.turn_id,
            segment_id=state.segment_id,
            revision=state.revision,
            source_start_sample=state.source_start_sample,
            current_pcm16=bytes(state.pcm),
            is_final=is_final,
        )

    async def _worker(self) -> None:
        """Process at most one MTD request at a time with latest-only pending."""

        try:
            while True:
                await self._pending_event.wait()
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

    async def _decode_and_publish(self, request: _SnapshotRequest) -> None:
        """Build the registered-speaker prefix, decode, and publish one result."""

        if self.model is None:
            if request.is_final:
                await self._publish_degraded_final(request, "model_not_configured")
            return
        layout = build_audio_layout(
            exemplars=self.pool.ordered(),
            current_pcm16=request.current_pcm16,
            sample_rate=self.sample_rate,
            inter_exemplar_silence_s=self.inter_exemplar_silence_s,
            exemplar_to_current_silence_s=self.exemplar_to_current_silence_s,
        )
        current_duration_s = len(request.current_pcm16) / (
            self.sample_rate * self.BYTES_PER_SAMPLE
        )
        try:
            result = await self.model.decode_snapshot(
                request_id=request.request_id,
                pcm16=layout.pcm16,
                sample_rate=self.sample_rate,
                decoder_prefix=layout.decoder_prefix,
                context_seconds=layout.context_seconds,
                current_audio_seconds=current_duration_s,
                is_final=request.is_final,
            )
        except Exception as exc:
            logger.warning(
                "MTD snapshot failed - session: %s, request: %s, error: %s",
                self.session_id,
                request.request_id,
                exc,
            )
            if request.is_final:
                await self._publish_degraded_final(request, type(exc).__name__)
            return
        state = self._segments.get(request.segment_id)
        if state is None:
            return
        if not request.is_final and state.final_requested:
            return
        if request.is_final:
            await self._publish_segment_final(request, result)
            return
        if request.revision <= state.published_partial_revision:
            return
        diarization_text = render_segments(result.current_segments)
        if not self.publish_unchanged and diarization_text == state.last_partial_text:
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
                sample_rate=self.sample_rate,
                raw_text=result.raw_text,
                diarization_text=diarization_text,
                segments=result.current_segments,
                latency_ms=result.latency_ms,
            )
        )

    async def _publish_segment_final(
        self,
        request: _SnapshotRequest,
        result: DiarizationResult,
    ) -> None:
        """Update the pool and publish one successful terminal segment."""

        pool_actions = self.pool.update_from_final(
            request.current_pcm16,
            result.current_segments,
            source_segment_id=request.segment_id,
        )
        event = SpeakerDiarizationSegmentFinal(
            session_id=self.session_id,
            turn_id=request.turn_id,
            segment_id=request.segment_id,
            source_start_sample=request.source_start_sample,
            source_end_sample=request.source_start_sample
            + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
            sample_rate=self.sample_rate,
            raw_text=result.raw_text,
            diarization_text=render_segments(result.current_segments),
            segments=result.current_segments,
            pool_version=self.pool.version,
            pool_actions=pool_actions,
            latency_ms=result.latency_ms,
        )
        await self._record_segment_final(event)

    async def _publish_degraded_final(
        self,
        request: _SnapshotRequest,
        reason: str,
    ) -> None:
        """Publish a terminal empty result so the turn join cannot deadlock."""

        event = SpeakerDiarizationSegmentFinal(
            session_id=self.session_id,
            turn_id=request.turn_id,
            segment_id=request.segment_id,
            source_start_sample=request.source_start_sample,
            source_end_sample=request.source_start_sample
            + len(request.current_pcm16) // self.BYTES_PER_SAMPLE,
            sample_rate=self.sample_rate,
            pool_version=self.pool.version,
            degraded=True,
            degraded_reason=reason,
        )
        await self._record_segment_final(event)

    async def _record_segment_final(
        self,
        event: SpeakerDiarizationSegmentFinal,
    ) -> None:
        """Record and publish a segment terminal before testing the turn barrier."""

        turn = self._turns.setdefault(event.turn_id, _TurnState())
        turn.finals[event.segment_id] = event
        await self.event_bus.publish(event)
        await self._maybe_publish_turn_final(event.turn_id)

    async def _maybe_publish_turn_final(self, turn_id: int) -> None:
        """Publish exactly one ordered turn timeline after its barrier is complete."""

        turn = self._turns.get(turn_id)
        if turn is None or not turn.hard_closed or turn.turn_final_published:
            return
        if any(segment_id not in turn.finals for segment_id in turn.segment_ids):
            return
        timeline: list[dict[str, object]] = []
        degraded_reasons: list[str] = []
        for segment_id in turn.segment_ids:
            final = turn.finals[segment_id]
            timeline.extend(
                offset_segments(
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
        await self.event_bus.publish(
            SpeakerDiarizationTurnFinal(
                session_id=self.session_id,
                turn_id=turn_id,
                segment_ids=list(turn.segment_ids),
                segments=timeline,
                diarization_text=render_segments(timeline),
                active_speaker_id=active_speaker_id,
                degraded=bool(degraded_reasons),
                degraded_reason=",".join(dict.fromkeys(degraded_reasons)),
            )
        )

    async def _best_effort_cancel(self, request_id: str) -> None:
        """Cancel an obsolete partial without propagating runtime failures."""

        if self.model is None:
            return
        try:
            await self.model.cancel(request_id)
        except Exception as exc:
            logger.debug("MTD cancel failed for %s: %s", request_id, exc)

    async def shutdown(self) -> None:
        """Cancel session tasks and release the cloned MTD client."""

        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        if self.model is not None:
            await self.model.close()


def _merge_nested_dict(
    defaults: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Return a shallow configuration merge with nested dict preservation."""

    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
