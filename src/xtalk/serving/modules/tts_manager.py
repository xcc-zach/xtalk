# -*- coding: utf-8 -*-
import asyncio
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
    TTSChunkGenerated,
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
from ...pipelines import Pipeline


class TTSQueueItem(NamedTuple):
    """Data model for queued TTS audio chunks."""

    audio_chunk: bytes


class TTSManager(Manager):
    """Event-driven TTS manager handling streaming synthesis and control."""

    # Sentence delimiters for chunking
    SENTENCE_DELIMITERS = {"。", "，", "！", "!", "？", "?", ".", ",", "：", ":"}

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize TTS manager.

        Args:
            event_bus: shared event bus
            session_id: unique session identifier
            pipeline: pipeline providing TTS models/controllers
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        # Session-level config
        self.config: dict[str, Any] = config or {}

        # TTS state
        self.is_paused = False
        self.pending_sentence_buffer: str = ""  # Pending sentence buffer
        self._first_sentence_ready = False

        # Queue for audio chunks fed to downstream consumers
        self.tts_queue = asyncio.Queue()

        # Settings for sim_gen mode
        self._sim_gen: bool = bool(self.config.get("sim_gen", False))
        self._segments_queue: Optional[asyncio.Queue] = None
        self._segments_task: Optional[asyncio.Task] = None

        # Consumer status
        self._consumer_running = False
        self.consumer_task: Optional[asyncio.Task] = None

        # Optional speed controller, falls back to passthrough when absent
        self.speed_controller = self.pipeline.get_speed_controller()
        self.current_speed: float = 1.0

        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._last_chunk_sent_for_tts = False

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
        reason = event.reason
        if not text:
            return
        await self._ensure_segments_queue().put(text)

    @Manager.event_handler(TurnTTSFlushRequested, priority=98)
    async def _handle_turn_tts_flush(self, event: TurnTTSFlushRequested) -> None:
        # Empty string marks flush
        await self._ensure_segments_queue().put("")

    @Manager.event_handler(
        TurnTTSResumeRequested,
        priority=95,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _handle_turn_tts_resume(self, event: TurnTTSResumeRequested) -> None:
        """Resume TTS playback when mediator requests."""
        if not self._segments_task or self._segments_task.done():
            return
        if not self.is_paused:
            logger.warning("Try to resume TTS which is not paused")
            return
        await self._resume_tts()
        tts_resumed_event = TTSResumed(
            session_id=self.session_id,
        )
        await self.event_bus.publish(tts_resumed_event)

    @Manager.event_handler(
        TurnTTSPauseRequested,
        priority=95,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
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
        # Stop sim-gen producer
        if self._segments_task and not self._segments_task.done():
            self._segments_task.cancel()
            try:
                await self._segments_task
            except asyncio.CancelledError:
                pass
        self._segments_task = None
        self._segments_queue = None
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
        self.is_paused = False
        self._resume_event.set()
        self.pending_sentence_buffer = ""
        self._first_sentence_ready = False
        self._last_chunk_sent_for_tts = False

        # Stop consumer
        await self._stop_consumer()

        # Drain queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
                self.tts_queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _segments_producer_loop(self) -> None:
        segments_queue = self._ensure_segments_queue()
        try:
            while True:
                seg = await segments_queue.get()
                # Empty segment signals flush
                if seg == "":
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

        if self.is_paused:
            return

        self.is_paused = True
        self._resume_event.clear()

    async def _resume_tts(self) -> None:
        """Resume TTS and continue consuming queued audio."""

        if not self.is_paused:
            return

        self.is_paused = False
        self._resume_event.set()

    async def _tts_consumer(self) -> None:
        """Consume queued TTS output and publish audio events."""
        last_sent_text = ""  # Track last text to avoid duplicates

        try:
            while self._consumer_running:
                # When paused, avoid consuming queue items or emitting events
                if self.is_paused:
                    await self._resume_event.wait()
                    continue
                try:
                    # Pull from the queue with a short timeout
                    item = await asyncio.wait_for(self.tts_queue.get(), timeout=0.1)

                    # Publish audio chunks (skip if paused)
                    if item and item.audio_chunk:
                        try:
                            # Apply speed control when enabled
                            processed_audio = item.audio_chunk
                            if (
                                self.speed_controller is not None
                                and self.current_speed != 1.0
                            ):
                                processed_audio = (
                                    await self.speed_controller.async_process(
                                        item.audio_chunk, self.current_speed
                                    )
                                )

                            # Publish to frontend (chunks processed in FIFO order)
                            event = TTSChunkGenerated(
                                session_id=self.session_id,
                                audio_chunk=processed_audio,
                            )
                            # Ensure ordering by waiting for completion
                            await self.event_bus.publish(
                                event, wait_for_completion=True
                            )
                            # Emit TTSFinished when the last chunk has drained
                            if self._last_chunk_sent_for_tts and self.tts_queue.empty():
                                await self.event_bus.publish(
                                    TTSFinished(session_id=self.session_id),
                                )
                        except Exception as e:
                            logger.error("Failed to publish TTS audio event: %s", e)

                    self.tts_queue.task_done()

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("TTS consumer error while handling audio: %s", e)
                    if not self.tts_queue.empty():
                        self.tts_queue.task_done()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "TTS consumer crashed - session: %s, error: %s", self.session_id, e
            )
            await self._publish_error("tts_consumer_error", str(e))
        finally:
            self._consumer_running = False

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
                component="TTSManager",
            )
        )

    async def shutdown(self) -> None:
        """Shut down TTS manager and reset state."""
        await self.reset_tts()

    async def _add_text_for_tts(self, text: str, *, final: bool) -> None:
        """Generate TTS audio from text and enqueue sentence segments."""
        text_len = len(text)
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

        if final:
            self._last_chunk_sent_for_tts = True

        if len(sentences) > 0 and not self._first_sentence_ready:
            self._first_sentence_ready = True
            await self.event_bus.publish(LLMFirstSentence(session_id=self.session_id))
        for sentence in sentences:
            await self._enqueue_sentence_stream(sentence)

    async def _enqueue_sentence_stream(self, sentence: str) -> None:
        """Run streaming TTS for one sentence and enqueue resulting chunks."""
        tts_model = self.pipeline.get_tts_model()
        if not tts_model:
            await self._publish_error(
                "tts_model_missing", "TTS model is not configured"
            )
            return

        try:
            async for ch in self._synthesize_stream_with_fallback(tts_model, sentence):
                await self.tts_queue.put(TTSQueueItem(ch))

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
        name = event.name
        args = event.args
        if name == "set_speed":
            speed = float(args["speed"])
            await self.event_bus.publish(
                TTSSpeedChange(session_id=self.session_id, speed=speed)
            )
        elif name == "set_voice":
            voice_name = str(args["name"])
            await self.event_bus.publish(
                TTSVoiceChange(session_id=self.session_id, voice_name=voice_name)
            )
        elif name == "set_emotion":
            emotion_name = args.get("name", "")
            await self.event_bus.publish(
                TTSEmotionChange(
                    session_id=self.session_id,
                    emotion_name=emotion_name,
                )
            )

    @Manager.event_handler(
        TTSVoiceChange,
        priority=100,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _handle_voice_change(self, event: TTSVoiceChange) -> None:
        """Handle requests to change the reference voice."""
        voice_name = event.voice_name

        try:
            self.pipeline.get_tts_model().set_voice(voice_names=[voice_name])
        except Exception as e:
            logger.error(
                "Failed to change voice: %s - session: %s",
                e,
                self.session_id,
            )

    @Manager.event_handler(
        TTSEmotionChange,
        priority=100,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _handle_emotion_change(self, event: TTSEmotionChange) -> None:
        """Handle requests to change TTS emotion."""
        emotion_name = event.emotion_name
        emotion_vector = event.emotion_vector

        try:
            self.pipeline.get_tts_model().set_emotion(
                emotion=emotion_name if emotion_name else emotion_vector
            )
        except Exception as e:
            logger.error(
                "Failed to change voice: %s - session: %s",
                e,
                self.session_id,
            )

    @Manager.event_handler(
        TTSSpeedChange,
        priority=100,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
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
            if hasattr(self.pipeline, "set_tts_model"):
                self.pipeline.set_tts_model(model_type, config)
            else:
                logger.warning(
                    "Pipeline does not support set_tts_model - session: %s",
                    self.session_id,
                )
        except Exception as e:
            logger.error(
                "Failed to switch TTS model: %s - session: %s",
                e,
                self.session_id,
            )
