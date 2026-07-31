import asyncio
import logging
from typing import Any, Optional

from ...models import ASR, Agent, Models
from ..event_bus import EventBus
from ..events import (
    ASRResultFinal,  # emit when turn about to move to next stage like text generation
    ASRResultPartial,
    AudioFrameReceived,
    Event,
    TurnASREndRequested,
    TurnInputAbortRequested,
    TurnASRPauseRequested,
    TurnASRStartRequested,
    VADSpeechEnd,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class ByteQueue:
    def __init__(self, max_bytes: int = 0):
        # max_byte==0 indicates no cap
        if max_bytes < 0:
            raise ValueError("max_bytes must be a non-negative int")
        self._max_bytes = max_bytes
        self._buffer = bytearray()
        self._lock = asyncio.Lock()
        self._new_bytes_event = asyncio.Event()

    async def put(self, data: bytes) -> None:
        async with self._lock:
            self._buffer.extend(data)
            # Remove bytes from front if exceeding capacity
            if self._max_bytes > 0 and len(self._buffer) > self._max_bytes:
                excess = len(self._buffer) - self._max_bytes
                self._buffer = self._buffer[excess:]
            self._new_bytes_event.set()

    async def get(self, max_bytes: Optional[int] = None) -> bytes:
        # May return empty bytes
        async with self._lock:
            if max_bytes is None:
                # Return all bytes
                result = bytes(self._buffer)
                self._buffer.clear()
            else:
                # Return at most max_bytes
                bytes_to_return = min(max_bytes, len(self._buffer))
                result = bytes(self._buffer[:bytes_to_return])
                self._buffer = self._buffer[bytes_to_return:]
            return result

    async def wait_on_new_bytes(self):
        """Wait until new bytes come"""
        self._new_bytes_event.clear()
        await self._new_bytes_event.wait()

    async def interrupt_wait(self):
        """Interrupt any waiting on new bytes."""
        self._new_bytes_event.set()

    async def size(self):
        async with self._lock:
            return len(self._buffer)


class AudioConsumer:
    SAMPLE_RATE = 16000
    PRE_BUFFER_SECS = 1

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._asr_model = models.require(ASR)
        self._asr_model.reset()
        self._agent = models.get(Agent)
        self._session_id = session_id
        # Flag to avoid repeat pause or end
        self._ended = True
        self._paused = False
        # Cache for recognition used in _recognize_and_publish
        self._recognized_text = ""
        self._turn_chat_history: str | None = None
        self._recognition_generation = 0
        # Assume PCM 16bit mono
        bytes_per_second = self.SAMPLE_RATE * 1 * (16 // 8)
        # Store audio before ASR starts; make sure ASR do not leave alone the first words
        self._pre_buffer = ByteQueue(self.PRE_BUFFER_SECS * bytes_per_second)
        # For ASR consumption
        self._audio_queue = ByteQueue()
        # Indicate whether consumer is running and _audio_queue should accept audio
        self._consumer_running_event = asyncio.Event()
        # Indicate whether consumer is idle; useful for waiting consumer to process its current loop after stop
        self._consumer_idle_event = asyncio.Event()
        self._consumer_idle_event.set()
        # Serialize lifecycle transitions so pause finalization cannot race reset.
        self._lifecycle_lock = asyncio.Lock()
        # Start audio consumer task immediately
        self._audio_consumer_task = asyncio.create_task(self._audio_consumer())
    async def accept_audio_frame(self, audio_frame: bytes):
        # Add audio_frame to pre-buffer if consumer not started; add audio_frame to recognition queue if started
        if not self._consumer_running():
            await self._pre_buffer.put(audio_frame)
        else:
            await self._audio_queue.put(audio_frame)

    async def start(self):
        # Pump pre-buffer to recognition queue; start consumer
        async with self._lifecycle_lock:
            # Break start-once invariance warning
            if self._consumer_running():
                logger.warning("AudioConsumer started repeatedly; do early return")
                return
            self._ended = False
            self._paused = False
            self._turn_chat_history = self._snapshot_chat_history()
            self._start_consumer()
            await self._audio_queue.put(await self._pre_buffer.get())

    async def pause(self):
        # Stop consumer, do one recognition and publish
        async with self._lifecycle_lock:
            # Break pause-once invariance warning
            if self._paused:
                logger.warning("AudioConsumer paused repeatedly; do early return")
                return
            if self._ended:
                return
            self._paused = True
            await self._stop_consumer()
            audio = await self._audio_queue.get()
            # Sentence end but not end of recognition
            await self._recognize_and_publish(
                audio,
                is_asr_end=False,
                is_final_chunk=True,
            )

    async def end(self):
        # Stop consumer, do one recognition, publish ASRResultFinal, clean up states (including reset ASR)
        async with self._lifecycle_lock:
            # Break stop-once invariance warning
            if self._ended:
                logger.warning("AudioConsumer ended repeatedly; do early return")
                return
            self._ended = True
            if self._paused:
                # Reuse the pause-time recognition result to avoid re-entering ASR
                # during the pause->finalize handoff.
                await self._publish_final_from_cached_text()
                await self._reset_states()
                return
            await self._stop_consumer()
            audio = await self._audio_queue.get()
            await self._recognize_and_publish(
                audio,
                is_asr_end=True,
                is_final_chunk=True,
            )
            await self._reset_states()

    async def abort(self) -> None:
        """Discard the unfinished audio turn without publishing ASR results."""
        self._invalidate_recognition()
        async with self._lifecycle_lock:
            # Also invalidate work that may have started while waiting for the
            # lifecycle lock held by pause/end finalization.
            self._invalidate_recognition()
            self._ended = True
            self._paused = False
            await self._stop_consumer()
            await self._reset_states()

    async def _audio_consumer(self):
        while True:
            self._consumer_idle_event.set()
            if not self._consumer_running_event.is_set():
                await self._consumer_running_event.wait()
            self._consumer_idle_event.clear()
            # Publish audio on condition:
            # Await until new bytes come
            await self._audio_queue.wait_on_new_bytes()
            # Accumulate to proper chunk bytes, send for recognition at least 1 bytes
            least_bytes_to_send = self._asr_model.stream_chunk_bytes_hint() or 1
            if await self._audio_queue.size() < least_bytes_to_send:
                continue
            audio = await self._audio_queue.get()
            await self._recognize_and_publish(audio)

    async def shutdown(self):
        # Consumer task cancellation
        try:
            self._audio_consumer_task.cancel()
            await self._audio_consumer_task
        except asyncio.CancelledError:
            pass

    # Helpers
    def _consumer_running(self):
        return self._consumer_running_event.is_set()

    def _invalidate_recognition(self) -> None:
        """Prevent recognition calls already in flight from publishing."""
        self._recognition_generation += 1

    def _start_consumer(self):
        self._consumer_running_event.set()

    async def _stop_consumer(self):
        # Must stop after the current chunk is processed in consumer
        self._consumer_running_event.clear()
        # FIXED: make sure audio consumer does not stuck and _consumer_idle_event will be set
        await self._audio_queue.interrupt_wait()
        await self._consumer_idle_event.wait()

    async def _recognize_and_publish(
        self, audio: bytes, is_asr_end: bool = False, is_final_chunk: bool = False
    ):
        # Do ASR recognition once and publish result ASRResultPartal/Final based on is_asr_end; if not final, may use cache to determine whether to publish;
        # is_final_chunk means whether user has a pause on his speech, passed on to ASRResultPartial
        if is_asr_end and not is_final_chunk:
            raise ValueError("ASR ends but chunk is not final chunk")
        recognition_generation = self._recognition_generation
        recognized_text = self._recognized_text
        if len(audio) > 0 or is_final_chunk:
            recognized_text = await self._asr_model.async_recognize_stream(
                audio,
                is_final=is_final_chunk,
                chat_history=self._turn_chat_history,
            )
        if recognition_generation != self._recognition_generation:
            return
        if (
            is_asr_end
            and self._is_blank_text(recognized_text)
            and not self._is_blank_text(self._recognized_text)
        ):
            recognized_text = self._recognized_text
        if self._is_blank_text(recognized_text):
            return
        if is_asr_end:
            self._recognized_text = recognized_text
            await self._publish_final_text(recognized_text)
        else:
            if recognized_text == self._recognized_text and not is_final_chunk:
                return
            self._recognized_text = recognized_text
            await self._publish_event(
                ASRResultPartial(
                    session_id=self._session_id,
                    text=recognized_text,
                    display_text=recognized_text,
                    speech_pause=is_final_chunk,
                )
            )

    @staticmethod
    def _is_blank_text(text: str) -> bool:
        """Return whether recognized text is empty after trimming whitespace."""
        return not text.strip()

    async def _publish_event(self, event: Event):
        await self._event_bus.publish(event)

    async def _publish_final_text(self, text: str) -> None:
        """Publish one final ASR result."""
        await self._publish_event(
            ASRResultFinal(
                session_id=self._session_id,
                text=text,
                display_text=text,
            )
        )

    async def _publish_final_from_cached_text(self) -> None:
        """Publish the cached recognition result as the final ASR result."""
        if self._is_blank_text(self._recognized_text):
            return
        await self._publish_final_text(self._recognized_text)

    async def _reset_states(self):
        self._recognized_text = ""
        self._turn_chat_history = None
        await self._pre_buffer.get()
        await self._audio_queue.get()
        self._consumer_running_event.clear()
        self._consumer_idle_event.set()
        self._asr_model.reset()

    def _snapshot_chat_history(self) -> str | None:
        """Capture the session chat history for the current ASR turn."""
        if self._agent is None:
            return None
        try:
            return self._agent.get_chat_history()
        except Exception as exc:
            logger.warning(
                "Failed to snapshot chat history for ASR - session: %s, error: %s",
                self._session_id,
                exc,
            )
            return None


class ASRManager(Manager):
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self._audio_consumer = AudioConsumer(event_bus, session_id, models, config)
        self._text_turn_active = False
        self._input_lock = asyncio.Lock()

    @Manager.event_handler(AudioFrameReceived)
    async def _handle_audio_frame(self, event: AudioFrameReceived):
        async with self._input_lock:
            if self._text_turn_active:
                return
            audio_frame = event.audio_data
            await self._audio_consumer.accept_audio_frame(audio_frame)

    @Manager.event_handler(TurnInputAbortRequested, priority=100)
    async def _handle_input_abort(self, event: TurnInputAbortRequested) -> None:
        """Discard acoustic input before a synthetic text turn begins."""
        if event.origin != "text":
            return
        self._text_turn_active = True
        self._audio_consumer._invalidate_recognition()
        async with self._input_lock:
            await self._audio_consumer.abort()

    @Manager.event_handler(VADSpeechEnd, priority=100)
    async def _handle_text_vad_end(self, event: VADSpeechEnd) -> None:
        """Resume accepting microphone frames after a synthetic text turn."""
        if event.origin != "text":
            return
        async with self._input_lock:
            self._text_turn_active = False

    @Manager.event_handler(TurnASRStartRequested)
    async def _handle_asr_start(self, _):
        if self._text_turn_active:
            return
        await self._audio_consumer.start()

    @Manager.event_handler(TurnASREndRequested)
    async def _handle_asr_end(self, _):
        if self._text_turn_active:
            return
        await self._audio_consumer.end()

    @Manager.event_handler(TurnASRPauseRequested)
    async def _handle_asr_pause(self, _):
        if self._text_turn_active:
            return
        await self._audio_consumer.pause()

    async def shutdown(self):
        # Task cancellation
        await self._audio_consumer.shutdown()
