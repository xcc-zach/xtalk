# -*- coding: utf-8 -*-
import asyncio
from collections import deque
from typing import Optional, NamedTuple, Any

from ...log_utils import logger

from ..event_bus import EventBus
from ..events import (
    # Outbound events (unchanged for OutputGateway)
    TTSStarted,
    TTSStopped,
    TTSPaused,
    TTSResumed,
    TTSFinished,
    TTSTextSynthesized,
    TTSChunkReady,
    TTSChunkPlayed,
    ErrorOccurred,
    TTSVoiceChange,
    TTSEmotionChange,
    TTSSpeedChange,
    ToolCallOccurred,
    # Inbound mediator events
    TurnTTSStartRequested,
    TurnTTSPauseRequested,
    TurnTTSResumeRequested,
    TurnTTSStopRequested,
    TurnTTSFlushRequested,
    TTSModelSwitchRequested,
    LLMFirstSentence,
)
from ..events import TurnTTSTextAppendRequested
from ..interfaces import Manager
from ...models import Models, SpeechSpeedController, TTS


class TTSQueueItem(NamedTuple):
    """Data model for queued TTS audio chunks."""

    audio_chunk: bytes
    sample_rate: int


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
        self.tts_queue: asyncio.Queue[TTSQueueItem] = asyncio.Queue()

        self._segments_queue: Optional[asyncio.Queue] = None
        self._segments_task: Optional[asyncio.Task] = None

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

    def _ensure_segments_queue(self) -> asyncio.Queue:
        """Ensure a queue exists for sentence segments."""
        if not self._segments_queue:
            self._segments_queue = asyncio.Queue()
        return self._segments_queue

    @Manager.event_handler(TurnTTSStartRequested, priority=100)
    async def _handle_turn_tts_start(self, event: TurnTTSStartRequested) -> None:
        """Handle mediator request to start TTS generation."""
        # Always use segment queue: start only initializes, text arrives via append
        segments_queue = self._ensure_segments_queue()
        # Start consumer if not already running
        if not self._segments_task or self._segments_task.done():
            # Reset state before starting
            await self.reset_tts()
            await self._publish_tts_started()

            # Start downstream consumer and upstream producer
            await self._start_consumer()
            self._segments_task = asyncio.create_task(self._segments_producer_loop())

    @Manager.event_handler(TurnTTSTextAppendRequested, priority=98)
    async def _handle_turn_tts_append(self, event: TurnTTSTextAppendRequested) -> None:
        """Append text segments for TTS (both sim-gen and regular modes)."""
        text = event.text
        if not text:
            return
        await self._ensure_segments_queue().put(text)

    @Manager.event_handler(TurnTTSFlushRequested, priority=98)
    async def _handle_turn_tts_flush(self, event: TurnTTSFlushRequested) -> None:
        # Use sentinel object to mark flush
        await self._ensure_segments_queue().put(self._FLUSH_SENTINEL)

    @Manager.event_handler(TurnTTSResumeRequested, priority=95)
    async def _handle_turn_tts_resume(self, event: TurnTTSResumeRequested) -> None:
        """Resume TTS playback when mediator requests."""
        if not self._segments_task or self._segments_task.done():
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
        if not self._segments_task or self._segments_task.done():
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

        # Stop consumer
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

                    # Publish audio chunks (skip if paused)
                    if isinstance(item, TTSQueueItem) and item.audio_chunk:
                        # Apply speed control when enabled
                        processed_audio = item.audio_chunk
                        if (
                            self.speed_controller is not None
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
                            await self._track_outstanding_chunk(
                                self._chunk_duration_ms(chunk, item.sample_rate)
                            )
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
        """Generate TTS audio from text and enqueue sentence segments."""
        self.pending_sentence_buffer += text

        sentences, remaining = self._split_text_by_delimiters(
            self.pending_sentence_buffer
        )

        # TODO: split sentence in sentences
        self.pending_sentence_buffer = remaining

        if final and self.pending_sentence_buffer.strip():
            # TODO: split remaining
            sentences.append(self.pending_sentence_buffer.strip())
            self.pending_sentence_buffer = ""

        if len(sentences) > 0 and not self._first_sentence_ready:
            self._first_sentence_ready = True
            await self.event_bus.publish(LLMFirstSentence(session_id=self.session_id))
        for sentence in sentences:
            await self._enqueue_sentence_stream(sentence)

        # Set the flag AFTER all sentences have been enqueued to avoid a race
        # where the consumer sees the flag before the last chunks are queued.
        if final:
            self._last_chunk_sent_for_tts = True

    async def _enqueue_sentence_stream(self, sentence: str) -> None:
        """Run streaming TTS for one sentence and enqueue resulting chunks."""
        tts_model = self.models.get(TTS)
        if not tts_model:
            await self._publish_error(
                "tts_model_missing", "TTS model is not configured"
            )
            return

        try:
            sample_rate = int(getattr(tts_model, "sample_rate", 48000) or 48000)
            synthesized_duration_ms = 0.0
            speed = 1.0
            if self.speed_controller is not None and self.current_speed != 1.0:
                speed = max(0.01, float(self.current_speed or 1.0))
            async for ch in self._synthesize_stream_with_fallback(tts_model, sentence):
                synthesized_duration_ms += self._chunk_duration_ms(ch, sample_rate)
                await self.tts_queue.put(TTSQueueItem(ch, sample_rate))
            await self.event_bus.publish(
                TTSTextSynthesized(
                    session_id=self.session_id,
                    text=sentence,
                    audio_duration=synthesized_duration_ms / speed,
                ),
                wait_for_completion=True,
            )

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(
                "TTS synthesis error: %s - session: %s",
                e,
                self.session_id,
            )
            await self._publish_error("tts_generation_error", str(e))

    async def _synthesize_stream_with_fallback(self, tts_model: Any, text: str):
        """Call streaming TTS API, falling back to sync methods on failure."""
        try:
            async for chunk in tts_model.async_synthesize_stream(text):
                yield chunk
            return
        except asyncio.CancelledError:
            raise
        except Exception as e:
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
                logger.error(
                    "Synchronous streaming TTS failed - session: %s, error: %s",
                    self.session_id,
                    err,
                )
                try:
                    return [tts_model.synthesize(text)]
                except Exception as final_err:
                    logger.error(
                        "Synchronous TTS fallback failed - session: %s, error: %s",
                        self.session_id,
                        final_err,
                    )
                    return []

        chunks = await loop.run_in_executor(None, _sync_stream)
        for chunk in chunks:
            yield chunk

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
        """Handle TTS model switch requests (IndexTTS / IndexTTS2)."""
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
        from ...models.tts.index_tts2 import IndexTTS2

        current_tts = self.models.get(TTS)
        current_ref_paths = []
        if current_tts and hasattr(current_tts, "audio_paths"):
            current_ref_paths = current_tts.audio_paths or []
        elif current_tts and hasattr(current_tts, "_base_audio_paths"):
            current_ref_paths = current_tts._base_audio_paths or []

        host = config.get("host", "localhost")
        port = config.get("port")
        ref_audio_paths = config.get("ref_audio_paths") or current_ref_paths
        sample_rate = config.get("sample_rate", 48000)
        timeout = config.get("timeout", 30.0)

        if port is None:
            port = 11996 if model_type == "IndexTTS" else 6006

        if model_type == "IndexTTS":
            self.models.set(
                TTS,
                IndexTTS(
                    ref_audio_paths=ref_audio_paths,
                    host=host,
                    port=port,
                    sample_rate=sample_rate,
                    timeout=timeout,
                ),
            )
        elif model_type == "IndexTTS2":
            self.models.set(
                TTS,
                IndexTTS2(
                    ref_audio_paths=ref_audio_paths,
                    host=host,
                    port=port,
                    sample_rate=sample_rate,
                    timeout=timeout,
                ),
            )
        else:
            raise ValueError(f"Unsupported TTS model type: {model_type}")
