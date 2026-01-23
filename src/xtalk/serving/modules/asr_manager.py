# -*- coding: utf-8 -*-
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Any
import math
from collections import deque
import numpy as np
import re
import unicodedata

from ...log_utils import logger

from ..event_bus import EventBus

from ..events import (
    EnhancedAudioFrameReceived,
    TurnASRStartRequested,
    TurnASREndRequested,
    TurnASRResetRequested,
    VerificationResult,
    ErrorOccurred,
    ASRResultFinal,
    ASRResultPartial,
    TranscriptionRefined,
    ASRStableSegmentReady,
    TurnASRFlushRequested,
)
from ..interfaces import Manager
from ...pipelines import Pipeline
from ...speech.interfaces import TurnType


@dataclass
class ConsumerState:
    """Track per-consumer state and keep buffers isolated per instance."""

    accumulated_audio: bytearray = field(default_factory=bytearray)
    running: bool = False
    processing: bool = False
    last_activity: float = 0.0
    processed_secs: float = 0
    errors: int = 0
    # Prevent final chunks from processing twice
    final_started_process: bool = False


# TODO: align sample rates
class ASRManager(Manager):
    """Event-driven ASR manager."""

    SAMPLE_RATE = 16000  # Sample rate constant

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize the ASR manager.

        Args:
            event_bus: event bus
            session_id: session identifier
            pipeline: pipeline instance
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        # Per-session config
        self.config: dict[str, Any] = config or {}

        # ASR state
        self.accumulated_text = ""
        self.accumulated_stable_char_count = 0
        self.chunk_count = 0

        # Whether a speech segment is active (driven by frontend VAD)
        self.speech_active = False

        # Optimized audio buffer to reduce copying
        self.audio_buffer = deque(maxlen=1000)  # Keep last 1000 audio chunks
        self.buffer_lock = asyncio.Lock()

        # Streaming ASR parameters: hold model ref only, chunking handled by model
        self.asr_model = self.pipeline.get_asr_model()
        self.turn_detector_model = self.pipeline.get_turn_detector_model()
        self._stream_chunk_bytes_hint = self._get_stream_chunk_hint()
        self.pre_buffer = deque(maxlen=self._compute_prebuffer_len())

        # Recording is handled by RecordingManager

        # Consumer state
        self.consumer_state = ConsumerState()
        self.consumer_task: Optional[asyncio.Task] = None

        # Simultaneous generation segment tracking (sim_gen only)
        self._sim_gen: bool = bool(self.config.get("sim_gen", False))
        # Pending incremental text buffer
        self._text_to_send: str = ""
        # Aggregated transcription for simultaneous generation
        self._sim_transcription: str = ""
        # Semantic timeout
        self._semantic_timeout_task: Optional[asyncio.Task] = None
        self._semantic_timeout_seconds: float = float(
            self.config.get("semantic_timeout_sec", 0.5)
        )
        # turn
        self._turn_id = 1
        # Sentence-ending punctuation list
        self._seg_punct = "，,。．.!！?？;；:：\n"
        # Segment throttle threshold (ms)
        self._seg_throttle_ms: int = 5000
        # Start timestamp for stable segment events
        self._seg_start_ts: float | None = None

        # Speaker detection moved elsewhere; SpeakerManager writes to context.

    def _get_stream_chunk_hint(self) -> int | None:
        """Cache chunk-size hint provided by the model."""
        if not self.asr_model:
            return None
        return self.asr_model.stream_chunk_bytes_hint()

    def _compute_prebuffer_len(self) -> int:
        """Estimate pre-buffer frame count based on chunk hint."""
        default_len = 20
        if not self._stream_chunk_bytes_hint:
            return default_len
        ms = (self._stream_chunk_bytes_hint / 32000) * 1000
        target_ms = max(100.0, min(400.0, ms * 2))
        est_frames = int(math.ceil(target_ms / 10.0))
        return max(20, min(50, est_frames))

    @Manager.event_handler(EnhancedAudioFrameReceived, priority=100)
    async def _handle_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Handle enhanced audio frames."""
        frame = {
            "data": event.audio_data,
            "timestamp": time.time(),
            "sample_rate": getattr(event, "sample_rate", self.SAMPLE_RATE),
            "is_final": getattr(event, "is_final", False),
        }

        # Only maintain pre_buffer when speech is inactive as padding for next turn
        if not self.speech_active:
            self.pre_buffer.append(frame)

        # Skip main buffer when speech isn't active unless final frame forces flush
        if not self.speech_active and not getattr(event, "is_final", False):
            return

        # Append audio to main buffer during speech segments
        async with self.buffer_lock:
            self.audio_buffer.append(frame)

        # When is_final=True, drain accumulated audio (default False for safety)

        if getattr(event, "is_final", False):
            try:
                # Safely drain buffer across async boundaries
                # Use the lock only to read buffer length, then consume outside it.
                while True:
                    async with self.buffer_lock:
                        remaining = len(self.audio_buffer)
                    if remaining <= 0:
                        break
                    await self._asr_consume_once(should_sleep=False)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(
                    "ASR consumer error - session: %s, error: %s", self.session_id, e
                )
                self.consumer_state.errors += 1

                # Publish error event
                error_event = ErrorOccurred(
                    session_id=self.session_id,
                    error_type="asr_consumer_error",
                    error_message=str(e),
                    component="ASRManager",
                )
                await self.event_bus.publish(error_event)

    @Manager.event_handler(TurnASRStartRequested, priority=90)
    async def _handle_turn_asr_start(self, event) -> None:
        """Handle mediator request to start an ASR cycle."""

        self.speech_active = True
        # Start ASR processing
        await self._start_consumer()
        # Ensure state is reset after unexpected interruptions
        self.consumer_state.final_started_process = False

        # Move pre-buffer frames into the main buffer as padding
        async with self.buffer_lock:
            prebuffer_frames = list(self.pre_buffer)
            for item in prebuffer_frames:
                self.audio_buffer.append(item)
            # Clear once moved to avoid reusing old frames next turn
            self.pre_buffer.clear()

    @Manager.event_handler(
        TurnASREndRequested,
        priority=90,
        enabled_if=lambda mgr: not mgr._sim_gen,
    )
    async def _handle_turn_asr_end(self, event) -> None:
        """Handle request to end ASR and finalize."""
        self.speech_active = False
        await self._stop_consumer()
        await self._finish_asr_processing()

    @Manager.event_handler(TurnASRResetRequested, priority=85)
    async def _handle_turn_asr_reset(self, event) -> None:
        """Reset ASR state upon mediator instruction."""
        await self._reset_asr()

    @Manager.event_handler(
        TurnASRFlushRequested,
        priority=88,
        enabled_if=lambda mgr: mgr._sim_gen,
    )
    async def _handle_turn_asr_flush(self, event) -> None:
        """Sim-trans request to emit current stable segment (VAD end)."""
        await self._maybe_emit_stable_segment()

    async def _validate_accumulated_text(self, text: str, semantic_tag: str) -> dict:
        """
        Validate whether ASR output should trigger interruption.

        Args:
            text: text to validate

        Returns:
            dict: validation result
        """
        validation_result = {
            "is_valid": False,
        }

        if self.turn_detector_model:
            # TODO: enhance detection logic with audio input and more logic to deal with labels
            # Remove last punctuation
            text = text[:-1] if unicodedata.category(text[-1]).startswith("P") else text
            detection_result = self.turn_detector_model.detect(audio=None, text=text)
            if detection_result == TurnType.COMPLETE:
                validation_result["is_valid"] = True
            return validation_result

        # Fallback to rule based turn detection
        try:
            # Minimum audio length gate
            MIN_INTERRUPTION_SECONDS = 1  # seconds
            audio_length = self.consumer_state.processed_secs
            if audio_length < MIN_INTERRUPTION_SECONDS:
                validation_result["is_valid"] = False
                return validation_result
            if semantic_tag:
                normalized_tag = (semantic_tag or "").strip().lower()
                if normalized_tag in {"<complete>", "<incomplete>", "<wait>"}:
                    validation_result.update(
                        {
                            "is_valid": True,
                        }
                    )
                else:
                    validation_result.update(
                        {
                            "is_valid": False,
                        }
                    )
                return validation_result

            # Basic check: reject empty text
            if not text or not text.strip():
                validation_result["is_valid"] = False
                return validation_result

            # Length validation
            text_length = len(text.strip())
            if text_length < 1:
                validation_result["is_valid"] = False
                return validation_result

            # Single-character alphanumeric filter
            if len(text.strip()) == 1 and re.match(r"^[\w]$", text.strip()):
                validation_result["is_valid"] = False
                return validation_result

            # Filler words
            filler_words = ["嗯", "啊", "呃", "哦", "呀", "吧", "呢", "嘛", "咳", "哼"]
            if text.strip() in filler_words:
                validation_result["is_valid"] = False
                return validation_result

            # Placeholder token
            if text == "<EMPTY>":
                validation_result["is_valid"] = False
                return validation_result

            # Confidence heuristic (to be refined)
            confidence = min(
                0.95, 0.5 + (text_length * 0.02) + (self.chunk_count * 0.01)
            )

            # Passed validation
            validation_result["is_valid"] = True

        except Exception as e:
            logger.error(
                "Text validation error - session: %s, error: %s", self.session_id, e
            )
            validation_result["is_valid"] = False

        return validation_result

    async def _publish_validation_result(
        self, text: str, validation_result: dict
    ) -> None:
        """Publish a validation result event."""
        try:
            # Emit verification result
            verification_result_event = VerificationResult(
                session_id=self.session_id,
                is_valid=validation_result["is_valid"],
            )
            # wait_for_completion=True ensures TTS/LLM stop flows finish before the next turn
            await self.event_bus.publish(
                verification_result_event, wait_for_completion=True
            )

        except Exception as e:
            logger.error(
                "Failed to publish ASR validation result - session: %s, error: %s",
                self.session_id,
                e,
            )

    async def _reset_asr(self) -> None:
        """Fully reset ASR state."""

        # Stop consumer
        await self._stop_consumer()

        # Reset state variables
        self._reset_consumer_state()
        self.accumulated_text = ""
        self.accumulated_stable_char_count = 0
        self.chunk_count = 0
        if self.asr_model:
            self.asr_model.reset()

        # Clear buffers
        async with self.buffer_lock:
            self.audio_buffer.clear()
            self.pre_buffer.clear()

    async def _finish_asr_processing(self) -> None:
        """Finalize ASR processing."""
        # Process remaining audio
        if not self.consumer_state.final_started_process:
            self.consumer_state.final_started_process = True
            await self._process_accumulated_audio(is_final=True)
        # Placeholder confidence (model may override later)
        final_confidence = 0
        self.accumulated_stable_char_count = len(self.accumulated_text)
        final_text = self.accumulated_text.strip()

        if not final_text:
            return

        cleaned_text, semantic_tag = self._extract_semantic_tag(final_text)

        # Display text no longer injects speaker labels (available via context)
        display_text = cleaned_text
        llm_text = cleaned_text

        # Validate and publish final ASR result
        valid_result = await self._validate_accumulated_text(llm_text, semantic_tag)
        semantic_tag = semantic_tag or "<complete>"
        await self._publish_validation_result(llm_text, valid_result)
        if valid_result["is_valid"]:
            if semantic_tag in ["<incomplete>", "<wait>"]:
                await self._publish_asr_result(
                    llm_text,
                    False,
                    final_confidence,
                    display_text=display_text,
                    semantic_tag=semantic_tag,
                    wait_for_completion=True,
                )

                await self._schedule_semantic_timeout()
            else:
                await self._publish_asr_result(
                    llm_text,
                    True,
                    final_confidence,
                    display_text=display_text,
                    semantic_tag=semantic_tag,
                    wait_for_completion=True,
                )
                # Increment turn id
                self._turn_id += 1
                await self._cancel_semantic_timeout()

    async def _start_consumer(self) -> None:
        """Start the audio consumer task."""
        if self.consumer_task and not self.consumer_task.done():
            return

        self.consumer_state.running = True
        self.consumer_task = asyncio.create_task(self._asr_consumer())

    async def _stop_consumer(self) -> None:
        """Stop the audio consumer task."""
        # Signal loop to exit and wait for graceful shutdown
        self.consumer_state.running = False
        if self.consumer_task and not self.consumer_task.done():
            try:
                await self.consumer_task
            except asyncio.CancelledError:
                pass
        self.consumer_task = None

    def _reset_consumer_state(self):
        self.consumer_state = ConsumerState()

    async def _asr_consume_once(self, should_sleep: bool) -> None:
        """Single iteration of the ASR consumer."""
        # Pull from buffer if available
        audio_chunk = None
        async with self.buffer_lock:
            if self.audio_buffer:
                audio_chunk = self.audio_buffer.popleft()

        if audio_chunk:
            # Append bytes to accumulator to avoid repeated copies
            chunk_data = audio_chunk["data"]
            self.consumer_state.accumulated_audio.extend(chunk_data)

            is_final = audio_chunk.get("is_final", False)

            # Determine if accumulated audio should be processed
            should_process = False
            if is_final:
                should_process = True
            else:
                dynamic_chunk_bytes = self._stream_chunk_bytes_hint

                if (
                    dynamic_chunk_bytes is None
                    or len(self.consumer_state.accumulated_audio) >= dynamic_chunk_bytes
                ):
                    should_process = True

            if self.consumer_state.final_started_process:
                should_process = False

            if should_process:
                if is_final:
                    self.consumer_state.final_started_process = True
                # Process accumulated audio
                await self._process_accumulated_audio(
                    is_final,
                )
        else:
            # No data; short sleep to avoid busy loop
            if should_sleep:
                await asyncio.sleep(0.005)  # 5 ms to reduce busy-waiting

    async def _asr_consumer(self) -> None:
        """Background loop that consumes audio chunks."""
        try:
            while self.consumer_state.running:
                await self._asr_consume_once(should_sleep=True)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(
                "ASR consumer error - session: %s, error: %s", self.session_id, e
            )
            self.consumer_state.errors += 1

            # Publish error event
            error_event = ErrorOccurred(
                session_id=self.session_id,
                error_type="asr_consumer_error",
                error_message=str(e),
                component="ASRManager",
            )
            await self.event_bus.publish(error_event)

    async def _process_accumulated_audio(
        self,
        is_final: bool,
    ) -> None:
        """
        Process accumulated audio (optimized path).

        Args:
            is_final: flag for final processing pass
        """
        try:
            dynamic_chunk_bytes = self._stream_chunk_bytes_hint
            self.consumer_state.processing = True

            # Ensure model exists
            if not self.asr_model:
                logger.warning(
                    "No ASR model, skipping processing - session: %s", self.session_id
                )
                return

            # Slice off bytes to process
            if dynamic_chunk_bytes and not is_final:
                audio_data = bytes(
                    self.consumer_state.accumulated_audio[:dynamic_chunk_bytes]
                )
                self.consumer_state.accumulated_audio = (
                    self.consumer_state.accumulated_audio[dynamic_chunk_bytes:]
                )
            else:
                audio_data = bytes(self.consumer_state.accumulated_audio)
                self.consumer_state.accumulated_audio.clear()

            # Double-check model presence
            if not self.asr_model:
                return

            # Skip if no audio and not a final flush
            if (not audio_data or len(audio_data) == 0) and not is_final:
                return

            if audio_data is None:
                audio_data = b""

            current_text: str = await self.asr_model.async_recognize_stream(
                audio_data, is_final=is_final
            )
            self.consumer_state.last_activity = time.time()
            # Only accumulate processed seconds when actual audio arrives
            if audio_data:
                self.consumer_state.processed_secs += (
                    len(audio_data) / 2 / self.SAMPLE_RATE
                )  # Assume 16 kHz mono 16-bit PCM
            if not current_text:
                return

            previous_text = (
                self.accumulated_text[self.accumulated_stable_char_count :]
                if len(self.accumulated_text) > self.accumulated_stable_char_count
                else ""
            )
            if current_text == previous_text:
                return

            if previous_text and current_text.startswith(previous_text):
                delta_text = current_text[len(previous_text) :]
            else:
                delta_text = current_text

            stable = min(self.accumulated_stable_char_count, len(self.accumulated_text))
            self.accumulated_text = self.accumulated_text[:stable] + current_text

            if not delta_text:
                return

            # Placeholder confidence; real value not supplied by model
            mock_confidence = 0.0
            self.chunk_count += 1

            # Allow final-triggered segments to dispatch events
            if not self.consumer_state.running and not is_final:
                return

            # TODO: check sim_gen implementation
            if self._sim_gen:
                # Simultaneous generation path
                if (not self._text_to_send) and delta_text.strip():
                    self._seg_start_ts = time.time()
                self._text_to_send += delta_text
                await self._publish_asr_result(
                    delta_text,
                    False,
                    mock_confidence,
                    wait_for_completion=False,
                )
            else:
                await self._publish_asr_result(
                    self.accumulated_text,
                    False,
                    mock_confidence,
                    wait_for_completion=False,
                )

        except Exception as e:
            logger.error(
                "ASR failed to process accumulated audio - session: %s, error: %s",
                self.session_id,
                e,
            )
            raise
        finally:
            self.consumer_state.processing = False

    def _pcm_to_float(self, pcm: bytes) -> np.ndarray:
        """Deprecated: ASR handles conversions internally (kept for compatibility)."""
        pcm_int16 = np.frombuffer(pcm, dtype=np.int16)
        return pcm_int16.astype(np.float32) / 32768.0

    def _extract_semantic_tag(self, text: str) -> tuple[str, str]:
        """Parse semantic tags and return (clean_text, tag)."""
        tag_patterns = ["<complete>", "<incomplete>", "<backchannel>", "<wait>"]
        found_tag = next((t for t in tag_patterns if t in text), "")
        cleaned_text = re.sub(r"<[^>]+>", "", text).strip()
        return cleaned_text, found_tag

    async def _schedule_semantic_timeout(self) -> None:
        """Schedule timeout to auto-complete incomplete segments."""
        await self._cancel_semantic_timeout()
        if self._semantic_timeout_seconds <= 0:
            return
        self._semantic_timeout_task = asyncio.create_task(
            self._semantic_timeout_worker()
        )

    async def _cancel_semantic_timeout(self) -> None:
        """Cancel pending semantic timeout task."""
        if self._semantic_timeout_task and not self._semantic_timeout_task.done():
            self._semantic_timeout_task.cancel()
            try:
                await self._semantic_timeout_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Failed to cancel semantic timeout task - session: %s, error: %s",
                    self.session_id,
                    exc,
                )
        self._semantic_timeout_task = None

    async def _semantic_timeout_worker(self) -> None:
        """On timeout, emit an empty complete result to unblock downstream."""
        try:
            await asyncio.sleep(self._semantic_timeout_seconds)
            await self._publish_asr_result(
                text=self.accumulated_text,
                is_final=True,
                confidence=0.0,
                display_text="",
                semantic_tag="<complete>",
                wait_for_completion=True,
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._semantic_timeout_task = None

    async def _publish_asr_result(
        self,
        text: str,
        is_final: bool,
        confidence: float,
        display_text: str = "",
        semantic_tag: str | None = None,
        wait_for_completion: bool = False,
    ) -> None:
        """Publish ASR result events."""
        cleaned_text = re.sub(r"<[^>]+>", "", text or "").strip()
        display_payload = display_text or cleaned_text
        event = (
            ASRResultFinal(
                session_id=self.session_id,
                text=cleaned_text,
                confidence=confidence,
                display_text=display_payload,
                semantic_tag=semantic_tag or "<complete>",
                turn_id=self._turn_id,
            )
            if is_final
            else ASRResultPartial(
                session_id=self.session_id,
                text=cleaned_text,
                confidence=confidence,
                display_text=display_payload,
                turn_id=self._turn_id,
            )
        )
        await self.event_bus.publish(event, wait_for_completion=wait_for_completion)

    async def _maybe_emit_stable_segment(self) -> None:
        """Emit stable segments in sim-trans mode using simple heuristics."""
        ## Skip segments shorter than 3 characters
        if len(self._text_to_send.strip()) < 3:
            return
        # Optionally restore punctuation
        punt_model = self.pipeline.get_punt_restorer_model()
        if punt_model:
            self._text_to_send = await punt_model.async_restore(self._text_to_send)

        buf = self._text_to_send
        if not buf:
            return

        should_emit = False

        # If sentence-ending punctuation exists
        if any(p in buf for p in self._seg_punct):
            should_emit = True

        if not should_emit:
            return

        segment = buf
        # Reset buffers
        self._text_to_send = ""
        # Append to transcription log
        self._sim_transcription += segment
        now = time.time()
        try:
            evt = ASRStableSegmentReady(
                session_id=self.session_id,
                text=segment,
                stability=1.0,
                start_ts=(self._seg_start_ts or 0.0),
                end_ts=now,
            )
            await self.event_bus.publish(evt)
            await self.event_bus.publish(
                TranscriptionRefined(
                    session_id=self.session_id,
                    text=self._sim_transcription,
                )
            )
            # Reset segment start timestamp
            self._seg_start_ts = None

        except Exception as e:
            logger.warning("Failed to emit ASRStableSegmentReady: %s", e)

    async def shutdown(self) -> None:
        """Shut down ASR manager."""

        # Stop ASR
        await self._reset_asr()
        # Recording logic handled in RecordingManager
