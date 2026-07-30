# -*- coding: utf-8 -*-
import asyncio
import logging
from collections import deque
from typing import Any, NamedTuple, Optional

from ...models import (
    TTS,
    Models,
    SpeechSpeedController,
    StreamingTextTTS,
)
from ..event_bus import EventBus
from ..events import (
    ErrorOccurred,
    LLMFirstSentence,
    ToolCallOccurred,
    TTSChunkPlayed,
    TTSChunkReady,
    TTSEmotionChange,
    TTSFinished,
    TTSModelSwitchRequested,
    TTSPaused,
    TTSResumed,
    TTSSpeedChange,
    # Outbound events (unchanged for OutputGateway)
    TTSStarted,
    TTSStopped,
    TTSStreamingTextAccepted,
    TTSTextDeliveryFinished,
    TTSTextSynthesisStarted,
    TTSTextSynthesized,
    TTSVoiceChange,
    TurnTTSFlushRequested,
    TurnTTSPauseRequested,
    TurnTTSResumeRequested,
    # Inbound mediator events
    TurnTTSStartRequested,
    TurnTTSStopRequested,
    TurnTTSTextAppendRequested,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class TTSQueueItem(NamedTuple):
    """Data model for queued TTS audio chunks."""

    audio_chunk: bytes
    sample_rate: int
    speed_processed: bool = False


class _TTSSentenceStart(NamedTuple):
    """FIFO marker emitted before one regular TTS sentence."""

    text: str


class _TTSSentenceEnd(NamedTuple):
    """FIFO marker emitted after one regular TTS sentence."""

    text: str
    succeeded: bool = True


class _TTSSynthesisError(RuntimeError):
    """Signal that one TTS sentence could not produce usable audio."""


class TTSManager(Manager):
    """Event-driven TTS manager handling streaming synthesis and control."""

    # Sentence delimiters for chunking
    SENTENCE_DELIMITERS = {"。", "，", "！", "!", "？", "?", ".", ",", "：", ":"}
    # Maximum duration of each outbound PCM chunk in milliseconds.
    TTS_CHUNK_MS = 100
    # Maximum unacknowledged outbound audio budget in milliseconds.
    MAX_OUTSTANDING_MS = 300
    # Sentinel for marking end of TTS stream
    _FLUSH_SENTINEL = object()

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize TTS manager.

        Args:
            event_bus: shared event bus
            session_id: unique session identifier
            models: model container providing TTS models/controllers
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.models = models
        # Session-level config
        self.config: dict[str, Any] = config or {}

        # TTS state
        self.pending_sentence_buffer: str = ""  # Pending sentence buffer
        self._first_sentence_ready = False

        # Queue for audio chunks fed to downstream consumers
        self.tts_queue: asyncio.Queue[
            TTSQueueItem | _TTSSentenceStart | _TTSSentenceEnd
        ] = asyncio.Queue()

        self._segments_queue: Optional[asyncio.Queue] = None
        self._segments_task: Optional[asyncio.Task] = None
        self._streaming_tts: Optional[StreamingTextTTS] = None
        self._streaming_audio_task: Optional[asyncio.Task] = None
        self._streaming_text: str = ""
        self._streaming_audio_duration_ms = 0.0
        self._streaming_audio_parts: list[bytes] = []
        self._streaming_sample_rate = 0

        # Consumer status
        self._consumer_running = False
        self.consumer_task: Optional[asyncio.Task] = None

        # Optional speed controller, falls back to passthrough when absent
        self.speed_controller = self.models.get(SpeechSpeedController)
        self.current_speed: float = 1.0

        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._last_chunk_sent_for_tts = False
        self._outstanding_chunk_ms: deque[float] = deque()
        self._outstanding_total_ms = 0.0
        self._outstanding_condition = asyncio.Condition()
        self._tts_generation_failure: str | None = None

    def _ensure_segments_queue(self) -> asyncio.Queue:
        """Ensure a queue exists for sentence segments."""
        if not self._segments_queue:
            self._segments_queue = asyncio.Queue()
        return self._segments_queue

    def _is_tts_active(self) -> bool:
        """Return whether synthesis, queued delivery, or playback is active."""

        if self._streaming_tts is not None:
            return True
        if self._streaming_audio_task and not self._streaming_audio_task.done():
            return True
        if self._segments_task and not self._segments_task.done():
            return True
        return bool(
            self.consumer_task
            and not self.consumer_task.done()
            and (not self.tts_queue.empty() or self._outstanding_total_ms > 0.0)
        )

    async def _start_streaming_tts(self, tts_model: StreamingTextTTS) -> bool:
        """Start a live text-streaming TTS session and its audio reader."""
        try:
            await tts_model.start()
        except Exception as e:
            logger.error(
                "Failed to start streaming TTS - session: %s, error: %s",
                self.session_id,
                e,
            )
            await self._publish_error("streaming_tts_start_error", str(e))
            return False

        self._streaming_tts = tts_model
        self._consumer_running = True
        await self._publish_tts_started()
        await self._start_consumer()
        self._streaming_audio_task = asyncio.create_task(
            self._streaming_audio_loop(tts_model)
        )
        return True

    async def _append_streaming_text(self, text: str) -> None:
        """Forward incremental text to the active streaming TTS model."""
        if self._streaming_tts is None:
            return
        prepared_audio_ms = max(0.0, self._streaming_audio_duration_ms)
        logger.debug(
            "[realtime-tts-race] stage=append_upstream_begin "
            "session=%s chunk_chars=%d buffered_chars=%d",
            self.session_id,
            len(text),
            len(self._streaming_text),
        )
        try:
            await self._streaming_tts.append_text(text)
        except Exception as e:
            logger.error(
                "Failed to append streaming TTS text - session: %s, error: %s",
                self.session_id,
                e,
            )
            await self._publish_error("streaming_tts_append_error", str(e))
            return

        self._streaming_text += text
        logger.debug(
            "[realtime-tts-race] stage=append_upstream_complete "
            "session=%s chunk_chars=%d buffered_chars=%d",
            self.session_id,
            len(text),
            len(self._streaming_text),
        )
        await self.event_bus.publish(
            TTSStreamingTextAccepted(
                session_id=self.session_id,
                text=text,
                prepared_audio_ms=prepared_audio_ms,
            ),
            wait_for_completion=True,
        )
        if not self._first_sentence_ready:
            self._first_sentence_ready = True
            await self.event_bus.publish(LLMFirstSentence(session_id=self.session_id))

    async def _flush_and_stop_streaming_tts(self) -> None:
        """Flush residual streaming text and stop the upstream live session."""
        if self._streaming_tts is None:
            return
        will_flush = bool(self._streaming_text.strip())
        logger.debug(
            "[realtime-tts-race] stage=flush_decision "
            "session=%s buffered_chars=%d will_flush=%s",
            self.session_id,
            len(self._streaming_text),
            will_flush,
        )
        try:
            if will_flush:
                await self._streaming_tts.flush()
            logger.debug(
                "[realtime-tts-race] stage=stop_upstream_begin "
                "session=%s flushed=%s",
                self.session_id,
                will_flush,
            )
            await self._streaming_tts.stop()
            logger.debug(
                "[realtime-tts-race] stage=stop_upstream_complete "
                "session=%s flushed=%s",
                self.session_id,
                will_flush,
            )
        except Exception as e:
            logger.error(
                "Failed to flush/stop streaming TTS - session: %s, error: %s",
                self.session_id,
                e,
            )
            await self._publish_error("streaming_tts_stop_error", str(e))

    async def _stop_streaming_tts(self) -> None:
        """Abort any active streaming TTS session and cancel its audio reader."""
        streaming_tts = self._streaming_tts
        streaming_task = self._streaming_audio_task
        self._streaming_tts = None
        self._streaming_audio_task = None
        self._consumer_running = False

        async with self._outstanding_condition:
            self._outstanding_condition.notify_all()

        if streaming_tts is not None:
            try:
                await streaming_tts.stop()
            except Exception as e:
                logger.warning(
                    "Failed to stop streaming TTS during reset - session: %s, error: %s",
                    self.session_id,
                    e,
                )

        current_task = asyncio.current_task()
        if (
            streaming_task
            and streaming_task is not current_task
            and not streaming_task.done()
        ):
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass

    async def _streaming_audio_loop(self, tts_model: StreamingTextTTS) -> None:
        """Prepare live TTS audio independently from paced delivery."""

        try:
            sample_rate = int(
                getattr(
                    tts_model,
                    "output_sample_rate",
                    getattr(tts_model, "sample_rate", 48000),
                )
                or 48000
            )
            self._streaming_sample_rate = sample_rate
            async for audio in tts_model.audio_stream():
                if not audio:
                    continue
                processed_audio = audio
                if self.speed_controller is not None and self.current_speed != 1.0:
                    processed_audio = await self.speed_controller.async_process(
                        audio,
                        self.current_speed,
                    )
                if not processed_audio:
                    continue

                self._streaming_audio_parts.append(processed_audio)
                self._streaming_audio_duration_ms += self._chunk_duration_ms(
                    processed_audio,
                    sample_rate,
                )
                await self.tts_queue.put(
                    TTSQueueItem(
                        processed_audio,
                        sample_rate,
                        speed_processed=True,
                    )
                )

            synthesized_text = await self._publish_streaming_text_synthesized()
            if synthesized_text:
                await self.tts_queue.put(_TTSSentenceEnd(synthesized_text))
            self._last_chunk_sent_for_tts = True
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "Streaming TTS audio loop crashed - session: %s, error: %s",
                self.session_id,
                e,
            )
            await self._publish_error("streaming_tts_generation_error", str(e))
        finally:
            if self._streaming_tts is tts_model:
                self._streaming_tts = None
            async with self._outstanding_condition:
                self._outstanding_condition.notify_all()

    async def _publish_streaming_text_synthesized(self) -> str:
        """Publish complete processed PCM for one streaming TTS turn."""

        text = self._streaming_text.strip()
        audio = b"".join(self._streaming_audio_parts)
        sample_rate = self._streaming_sample_rate
        if not text or not audio or sample_rate <= 0:
            return ""

        await self.event_bus.publish(
            TTSTextSynthesized(
                session_id=self.session_id,
                text=text,
                audio_duration=self._chunk_duration_ms(audio, sample_rate),
                audio_chunk=audio,
                sample_rate=sample_rate,
            ),
            wait_for_completion=True,
        )
        self._streaming_text = ""
        self._streaming_audio_duration_ms = 0.0
        self._streaming_audio_parts.clear()
        self._streaming_sample_rate = 0
        return text

    @Manager.event_handler(TurnTTSStartRequested, priority=100)
    async def _handle_turn_tts_start(self, event: TurnTTSStartRequested) -> None:
        """Handle mediator request to start TTS generation."""
        del event
        if self._streaming_tts is not None:
            return
        if self._segments_task and not self._segments_task.done():
            return

        await self.reset_tts()

        tts_model = self.models.get(TTS)
        if isinstance(tts_model, StreamingTextTTS):
            if await self._start_streaming_tts(tts_model):
                return

        await self._publish_tts_started()
        await self._start_consumer()
        self._segments_task = asyncio.create_task(self._segments_producer_loop())

    @Manager.event_handler(TurnTTSTextAppendRequested, priority=98)
    async def _handle_turn_tts_append(self, event: TurnTTSTextAppendRequested) -> None:
        """Append text segments for TTS (both sim-gen and regular modes)."""
        text = event.text
        if not text:
            return
        logger.debug(
            "[realtime-tts-race] stage=append_handler_enter "
            "session=%s chunk_chars=%d buffered_chars=%d streaming_active=%s",
            self.session_id,
            len(text),
            len(self._streaming_text),
            self._streaming_tts is not None,
        )
        if self._streaming_tts is not None:
            await self._append_streaming_text(text)
            return
        await self._ensure_segments_queue().put(text)

    @Manager.event_handler(TurnTTSFlushRequested, priority=98)
    async def _handle_turn_tts_flush(self, event: TurnTTSFlushRequested) -> None:
        logger.debug(
            "[realtime-tts-race] stage=flush_handler_enter "
            "session=%s buffered_chars=%d streaming_active=%s",
            self.session_id,
            len(self._streaming_text),
            self._streaming_tts is not None,
        )
        if self._streaming_tts is not None:
            await self._flush_and_stop_streaming_tts()
            return
        # Use sentinel object to mark flush
        await self._ensure_segments_queue().put(self._FLUSH_SENTINEL)

    @Manager.event_handler(TurnTTSResumeRequested, priority=95)
    async def _handle_turn_tts_resume(self, event: TurnTTSResumeRequested) -> None:
        """Resume TTS playback when mediator requests."""
        if not self._is_tts_active():
            return
        if self._resume_event.is_set():
            logger.warning("Try to resume TTS which is not paused")
            return
        await self._resume_tts()
        tts_resumed_event = TTSResumed(
            session_id=self.session_id,
        )
        await self.event_bus.publish(tts_resumed_event)

    @Manager.event_handler(TurnTTSPauseRequested, priority=95)
    async def _handle_turn_tts_pause(self, event: TurnTTSPauseRequested) -> None:
        """Pause TTS playback when mediator requests."""
        if not self._is_tts_active():
            return
        await self._publish_tts_pause()
        await self._pause_tts()

    @Manager.event_handler(TurnTTSStopRequested, priority=95)
    async def _handle_turn_tts_stop(self, event: TurnTTSStopRequested) -> None:
        """Stop TTS playback when mediator requests."""
        await self.reset_tts()
        # Do not publish TTSStopped for playback_finished
        if event.reason == "playback_finished":
            return
        tts_stopped_event = TTSStopped(
            session_id=self.session_id,
        )
        await self.event_bus.publish(tts_stopped_event)

    async def reset_tts(self) -> None:
        """Reset all TTS state and cancel consumers."""

        # Reset state flags
        self._resume_event.set()
        self.pending_sentence_buffer = ""
        self._first_sentence_ready = False
        self._last_chunk_sent_for_tts = False
        self._streaming_text = ""
        self._streaming_audio_duration_ms = 0.0
        self._streaming_audio_parts.clear()
        self._streaming_sample_rate = 0
        self._tts_generation_failure = None

        # Stop consumer
        await self._stop_streaming_tts()
        await self._stop_consumer()
        await self._reset_outstanding_audio()

        # Cancel segments producer task
        if self._segments_task and not self._segments_task.done():
            self._segments_task.cancel()
            try:
                await self._segments_task
            except asyncio.CancelledError:
                pass
        self._segments_task = None

        # Drain queue
        while True:
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Clear segments queue
        if self._segments_queue:
            while True:
                try:
                    self._segments_queue.get_nowait()
                    self._segments_queue.task_done()
                except asyncio.QueueEmpty:
                    break

    async def _segments_producer_loop(self) -> None:
        segments_queue = self._ensure_segments_queue()
        try:
            while True:
                seg = await segments_queue.get()
                # Sentinel object signals flush
                if seg is self._FLUSH_SENTINEL:
                    await self._add_text_for_tts("", final=True)
                    continue
                await self._add_text_for_tts(seg, final=False)
        except asyncio.CancelledError:
            raise

    async def _start_consumer(self) -> None:
        """Start the TTS audio consumer."""
        if self.consumer_task and not self.consumer_task.done():
            return

        self._consumer_running = True
        self.consumer_task = asyncio.create_task(self._tts_consumer())

    async def _stop_consumer(self) -> None:
        """Stop the TTS audio consumer."""
        self._consumer_running = False

        if self.consumer_task and not self.consumer_task.done():
            self.consumer_task.cancel()
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass

        self.consumer_task = None

    async def _pause_tts(self) -> None:
        """Pause TTS by halting consumption while leaving the queue intact."""

        if not self._resume_event.is_set():
            return

        self._resume_event.clear()

    async def _resume_tts(self) -> None:
        """Resume TTS and continue consuming queued audio."""

        if self._resume_event.is_set():
            return

        self._resume_event.set()

    async def _reset_outstanding_audio(self) -> None:
        """Clear all unacknowledged outbound audio tracking state."""
        async with self._outstanding_condition:
            self._outstanding_chunk_ms.clear()
            self._outstanding_total_ms = 0.0
            self._outstanding_condition.notify_all()

    async def _wait_for_outstanding_budget(self) -> None:
        """Wait until enough outbound audio budget is available."""
        async with self._outstanding_condition:
            while (
                self._consumer_running
                and self._outstanding_total_ms >= self.MAX_OUTSTANDING_MS
            ):
                await self._outstanding_condition.wait()

    async def _track_outstanding_chunk(self, chunk_ms: float) -> None:
        """Record a just-sent outbound chunk in the outstanding budget."""
        async with self._outstanding_condition:
            self._outstanding_chunk_ms.append(chunk_ms)
            self._outstanding_total_ms += chunk_ms

    def _chunk_duration_ms(self, audio_chunk: bytes, sample_rate: int) -> float:
        """Return the PCM chunk duration in milliseconds."""
        if not audio_chunk or sample_rate <= 0:
            return 0.0
        sample_count = len(audio_chunk) / 2
        return sample_count * 1000.0 / sample_rate

    def _split_audio_chunk(self, audio_chunk: bytes, sample_rate: int) -> list[bytes]:
        """Split PCM audio into fixed-size chunks for outbound transport."""
        if not audio_chunk or sample_rate <= 0:
            return []
        samples_per_chunk = max(1, round(sample_rate * self.TTS_CHUNK_MS / 1000))
        bytes_per_chunk = samples_per_chunk * 2
        return [
            audio_chunk[offset : offset + bytes_per_chunk]
            for offset in range(0, len(audio_chunk), bytes_per_chunk)
            if audio_chunk[offset : offset + bytes_per_chunk]
        ]

    async def _tts_consumer(self) -> None:
        """Consume queued TTS output and publish audio events."""

        active_sentence_text = ""
        try:
            while self._consumer_running:
                # When paused, avoid consuming queue items or emitting events
                if not self._resume_event.is_set():
                    await self._resume_event.wait()
                    continue

                item = None  # Track if we successfully got an item
                try:
                    # Pull from the queue with a short timeout
                    item = await asyncio.wait_for(self.tts_queue.get(), timeout=0.1)

                    if isinstance(item, _TTSSentenceStart):
                        active_sentence_text = item.text
                        await self.event_bus.publish(
                            TTSTextSynthesisStarted(
                                session_id=self.session_id,
                                text=item.text,
                            ),
                            wait_for_completion=True,
                        )
                    elif isinstance(item, _TTSSentenceEnd):
                        sentence_text = active_sentence_text or item.text
                        await self.event_bus.publish(
                            TTSTextDeliveryFinished(
                                session_id=self.session_id,
                                text=sentence_text,
                                succeeded=item.succeeded,
                            ),
                            wait_for_completion=True,
                        )
                        active_sentence_text = ""
                    elif isinstance(item, TTSQueueItem) and item.audio_chunk:
                        # Apply speed control when enabled
                        processed_audio = item.audio_chunk
                        if (
                            not item.speed_processed
                            and self.speed_controller is not None
                            and self.current_speed != 1.0
                        ):
                            processed_audio = await self.speed_controller.async_process(
                                item.audio_chunk, self.current_speed
                            )

                        for chunk in self._split_audio_chunk(
                            processed_audio, item.sample_rate
                        ):
                            await self._wait_for_outstanding_budget()
                            if not self._consumer_running:
                                break

                            event = TTSChunkReady(
                                session_id=self.session_id,
                                audio_chunk=chunk,
                                sample_rate=item.sample_rate,
                            )
                            await self.event_bus.publish(
                                event, wait_for_completion=True
                            )
                            chunk_ms = self._chunk_duration_ms(
                                chunk,
                                item.sample_rate,
                            )
                            await self._track_outstanding_chunk(chunk_ms)
                    # Mark task as done after processing
                    self.tts_queue.task_done()

                    # Check if the last chunk has been sent and the queue is empty
                    if self._last_chunk_sent_for_tts and self.tts_queue.empty():
                        self._last_chunk_sent_for_tts = False
                        await self.event_bus.publish(
                            TTSFinished(session_id=self.session_id),
                        )

                except asyncio.TimeoutError:
                    # Also check when queue is empty — TTS may have produced
                    # zero chunks (e.g. synthesis service failure), so the
                    # finished condition would never be evaluated above.
                    if self._last_chunk_sent_for_tts and self.tts_queue.empty():
                        self._last_chunk_sent_for_tts = False
                        await self.event_bus.publish(
                            TTSFinished(session_id=self.session_id),
                        )
                    continue
                except Exception as e:
                    logger.error("TTS consumer error while handling audio: %s", e)
                    # If the item is not None, call task_done()
                    if item is not None:
                        try:
                            self.tts_queue.task_done()
                        except ValueError:
                            # task_done() called too many times
                            logger.warning("task_done() called without matching get()")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "TTS consumer crashed - session: %s, error: %s", self.session_id, e
            )
            await self._publish_error("tts_consumer_error", str(e))
        finally:
            self._consumer_running = False

    @Manager.event_handler(TTSChunkPlayed, priority=100)
    async def _handle_tts_chunk_played(self, event: TTSChunkPlayed) -> None:
        """Release outstanding outbound budget after frontend playback confirmation."""
        async with self._outstanding_condition:
            if self._outstanding_chunk_ms:
                self._outstanding_total_ms = max(
                    0.0,
                    self._outstanding_total_ms - self._outstanding_chunk_ms.popleft(),
                )
            self._outstanding_condition.notify_all()

    async def _publish_tts_started(self) -> None:
        """Publish TTSStarted event."""
        event = TTSStarted(
            session_id=self.session_id,
        )
        # Wait for completion to maintain ordering
        await self.event_bus.publish(event, wait_for_completion=True)

    async def _publish_tts_pause(self) -> None:
        """Publish TTSPaused event."""
        event = TTSPaused(
            session_id=self.session_id,
        )
        await self.event_bus.publish(event)

    async def _publish_error(self, error_type: str, message: str) -> None:
        """Publish a TTS error event."""
        await self.event_bus.publish(
            ErrorOccurred(
                session_id=self.session_id,
                error_type=error_type,
                error_message=message,
            )
        )

    async def shutdown(self) -> None:
        """Shut down TTS manager and reset state."""
        await self.reset_tts()

    async def _add_text_for_tts(self, text: str, *, final: bool) -> None:
        """Generate and enqueue FIFO-ordered TTS sentence audio."""

        self.pending_sentence_buffer += text
        sentences, remaining = self._split_text_by_delimiters(
            self.pending_sentence_buffer
        )
        self.pending_sentence_buffer = remaining

        if final and self.pending_sentence_buffer.strip():
            sentences.append(self.pending_sentence_buffer.strip())
            self.pending_sentence_buffer = ""

        if sentences and not self._first_sentence_ready:
            self._first_sentence_ready = True
            await self.event_bus.publish(LLMFirstSentence(session_id=self.session_id))

        for sentence in sentences:
            await self._enqueue_sentence_stream(sentence)

        if final:
            self._last_chunk_sent_for_tts = True

    async def _enqueue_sentence_stream(self, sentence: str) -> None:
        """Prepare final sentence PCM while enqueueing it for paced delivery."""

        tts_model = self.models.get(TTS)
        if not tts_model:
            await self._publish_error(
                "tts_model_missing", "TTS model is not configured"
            )
            return
        if self._tts_generation_failure is not None:
            return

        sentence_started = False
        sentence_ended = False
        try:
            sample_rate = int(getattr(tts_model, "sample_rate", 48000) or 48000)
            sentence_speed = self.current_speed
            await self.tts_queue.put(_TTSSentenceStart(sentence))
            sentence_started = True
            audio_parts: list[bytes] = []
            async for chunk in self._synthesize_stream_with_fallback(
                tts_model,
                sentence,
            ):
                processed_chunk = chunk
                if self.speed_controller is not None and sentence_speed != 1.0:
                    processed_chunk = await self.speed_controller.async_process(
                        chunk,
                        sentence_speed,
                    )
                if not processed_chunk:
                    continue

                audio_parts.append(processed_chunk)
                await self.tts_queue.put(
                    TTSQueueItem(
                        processed_chunk,
                        sample_rate,
                        speed_processed=True,
                    )
                )

            if not audio_parts:
                raise _TTSSynthesisError("TTS returned no audio data.")

            prepared_audio = b"".join(audio_parts)
            await self.event_bus.publish(
                TTSTextSynthesized(
                    session_id=self.session_id,
                    text=sentence,
                    audio_duration=self._chunk_duration_ms(
                        prepared_audio,
                        sample_rate,
                    ),
                    audio_chunk=prepared_audio,
                    sample_rate=sample_rate,
                ),
                wait_for_completion=True,
            )
            await self.tts_queue.put(_TTSSentenceEnd(sentence))
            sentence_ended = True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            if sentence_started and not sentence_ended:
                await self.tts_queue.put(
                    _TTSSentenceEnd(sentence, succeeded=False)
                )
            first_failure = self._tts_generation_failure is None
            if first_failure:
                self._tts_generation_failure = str(e)
                logger.error(
                    "TTS synthesis failed; blocking remaining sentences in this turn - "
                    "session: %s, error: %s",
                    self.session_id,
                    e,
                )
                await self._publish_error("tts_generation_error", str(e))

    async def _synthesize_stream_with_fallback(self, tts_model: Any, text: str):
        """Call streaming TTS API, falling back to sync methods on failure."""

        yielded_audio = False
        try:
            async for chunk in tts_model.async_synthesize_stream(text):
                if not chunk:
                    continue
                yielded_audio = True
                yield chunk
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            uses_default_async_wrapper = (
                type(tts_model).async_synthesize_stream
                is TTS.async_synthesize_stream
            )
            if (
                yielded_audio
                or uses_default_async_wrapper
                or self._is_non_retryable_tts_error(e)
            ):
                raise _TTSSynthesisError(str(e)) from e
            logger.warning(
                "Streaming TTS failed, trying fallback - session: %s, error: %s",
                self.session_id,
                e,
            )

        loop = asyncio.get_running_loop()

        def _sync_stream():
            try:
                return list(tts_model.synthesize_stream(text))
            except Exception as err:
                if self._is_non_retryable_tts_error(err):
                    raise _TTSSynthesisError(str(err)) from err
                logger.error(
                    "Synchronous streaming TTS failed - session: %s, error: %s",
                    self.session_id,
                    err,
                )
                try:
                    return [tts_model.synthesize(text)]
                except Exception as final_err:
                    raise _TTSSynthesisError(str(final_err)) from final_err

        chunks = await loop.run_in_executor(None, _sync_stream)
        for chunk in chunks:
            if chunk:
                yield chunk

    @staticmethod
    def _is_non_retryable_tts_error(error: Exception) -> bool:
        """Return whether retrying the same TTS request would immediately fail."""

        message = str(error).lower()
        markers = (
            "allocationquota",
            "free tier",
            "throttling",
            "rate limit",
            "ratequota",
            "invalidapikey",
            "invalid api key",
            "authentication",
            "unauthorized",
            "insufficient balance",
        )
        return any(marker in message for marker in markers)

    def _split_text_by_delimiters(self, accumulated_text: str) -> tuple[list[str], str]:
        """Split text by sentence delimiters, returning sentences + residual text."""
        if not accumulated_text:
            return [], ""

        sentences: list[str] = []
        start_idx = 0
        for idx, char in enumerate(accumulated_text):
            if char in self.SENTENCE_DELIMITERS:
                chunk = accumulated_text[start_idx : idx + 1].strip()
                if chunk:
                    sentences.append(chunk)
                start_idx = idx + 1

        remaining_text = accumulated_text[start_idx:]
        return sentences, remaining_text

    @Manager.event_handler(ToolCallOccurred, priority=100)
    async def _handle_tool_call_occurred(self, event: ToolCallOccurred):
        """Handle tool call events for TTS control (speed, voice, emotion)."""
        name = event.name
        args = event.args

        # Add parameter validation
        if name == "set_speed":
            if "speed" not in args:
                logger.warning(
                    "set_speed missing 'speed' parameter - session: %s",
                    self.session_id,
                )
                return
            try:
                speed = float(args["speed"])
                await self.event_bus.publish(
                    TTSSpeedChange(session_id=self.session_id, speed=speed)
                )
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Invalid speed value '%s': %s - session: %s",
                    args.get("speed"),
                    e,
                    self.session_id,
                )

        elif name == "set_voice":
            if "name" not in args:
                logger.warning(
                    "set_voice missing 'name' parameter - session: %s",
                    self.session_id,
                )
                return
            voice_name = str(args["name"])
            if not voice_name:
                logger.warning(
                    "set_voice received empty voice name - session: %s",
                    self.session_id,
                )
                return
            await self.event_bus.publish(
                TTSVoiceChange(session_id=self.session_id, voice_name=voice_name)
            )

        elif name == "set_emotion":
            emotion_name = args.get("name", "")
            emotion_vector = args.get("vector", None)
            # Validate at least one parameter is valid
            if not emotion_name and not emotion_vector:
                logger.warning(
                    "set_emotion received empty emotion_name and emotion_vector - session: %s",
                    self.session_id,
                )
                return
            await self.event_bus.publish(
                TTSEmotionChange(
                    session_id=self.session_id,
                    emotion_name=emotion_name,
                    emotion_vector=emotion_vector,
                )
            )

    @Manager.event_handler(TTSVoiceChange, priority=100)
    async def _handle_voice_change(self, event: TTSVoiceChange) -> None:
        """Handle requests to change the reference voice."""
        voice_name = event.voice_name

        try:
            tts_model = self.models.get(TTS)
            if tts_model is None:
                logger.warning("TTS model is not configured - session: %s", self.session_id)
                return
            tts_model.set_voice(voice_names=[voice_name])
        except Exception as e:
            logger.error(
                "Failed to change voice: %s - session: %s",
                e,
                self.session_id,
            )

    @Manager.event_handler(TTSEmotionChange, priority=100)
    async def _handle_emotion_change(self, event: TTSEmotionChange) -> None:
        """Handle requests to change TTS emotion."""
        emotion_name = event.emotion_name
        emotion_vector = event.emotion_vector

        # Validate parameters
        if not emotion_name and not emotion_vector:
            logger.warning(
                "Both emotion_name and emotion_vector are empty - session: %s",
                self.session_id,
            )
            return

        try:
            tts_model = self.models.get(TTS)
            if tts_model is None:
                logger.warning("TTS model is not configured - session: %s", self.session_id)
                return
            tts_model.set_emotion(
                emotion=emotion_name if emotion_name else emotion_vector
            )
        except Exception as e:
            # Fix error message
            logger.error(
                "Failed to change emotion: %s - session: %s",
                e,
                self.session_id,
            )

    @Manager.event_handler(TTSSpeedChange, priority=100)
    async def _handle_speed_changed(self, event: TTSSpeedChange) -> None:
        """Handle requests to adjust TTS speed."""
        speed = event.speed

        self.current_speed = speed

    @Manager.event_handler(TTSModelSwitchRequested, priority=100)
    async def _handle_tts_model_switch(self, event: TTSModelSwitchRequested) -> None:
        """Handle TTS model switch requests for IndexTTS 1.5 or 2."""
        model_type = event.model_type
        config = event.config

        try:
            self._set_tts_model(model_type, config)
        except Exception as e:
            logger.error(
                "Failed to switch TTS model: %s - session: %s",
                e,
                self.session_id,
            )

    def _set_tts_model(self, model_type: str, config: dict[str, Any]) -> None:
        """Replace the active TTS model for the current session."""
        from ...models.tts.index_tts import IndexTTS

        current_tts = self.models.get(TTS)
        current_ref_paths = []
        if current_tts and hasattr(current_tts, "audio_paths"):
            current_ref_paths = current_tts.audio_paths or []
        elif current_tts and hasattr(current_tts, "_base_audio_paths"):
            current_ref_paths = current_tts._base_audio_paths or []

        ref_audio_paths = config.get("ref_audio_paths") or current_ref_paths
        sample_rate = config.get("sample_rate", 48000)
        timeout = config.get("timeout", 30.0)
        voices = config.get("voices") or [
            {"name": str(index), "path": path}
            for index, path in enumerate(ref_audio_paths)
        ]

        if model_type != "IndexTTS":
            raise ValueError(f"Unsupported TTS model type: {model_type}")
        model_version = str(config.get("model", config.get("model_version", "1.5")))

        base_url = config.get("base_url")
        if base_url is None and ("host" in config or "port" in config):
            host = config.get("host", "localhost")
            port = config.get("port", 6006)
            base_url = f"http://{host}:{port}"

        self.models.set(
            TTS,
            IndexTTS(
                voices=voices,
                base_url=base_url,
                sample_rate=sample_rate,
                timeout=timeout,
                model=model_version,
                emo_weight=config.get("emo_weight", 1.0),
                emo_random=config.get("emo_random", False),
                max_text_tokens_per_sentence=config.get(
                    "max_text_tokens_per_sentence", 120
                ),
            ),
        )
