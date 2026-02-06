from typing import Optional, Any
import asyncio
from ..interfaces import Manager
from ..event_bus import EventBus
from ...pipelines import Pipeline
from ..events import (
    BaseEvent,
    EnhancedAudioFrameReceived,
    TurnASRStartRequested,
    TurnASREndRequested,
    TurnASRPauseRequested,
    ASRResultPartial,
    ASRResultFinal,  # emit when turn about to move to next stage like text generation
)
from ...log_utils import logger


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
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._asr_model = pipeline.get_asr_model()
        self._asr_model.reset()
        self._session_id = session_id
        # Flag to avoid repeat pause or end
        self._ended = True
        self._paused = False
        # Cache for recognition used in _recognize_and_publish
        self._recognized_text = ""
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
        # Start audio consumer task immediately
        self._audio_consumer_task = asyncio.create_task(self._audio_consumer())
        # TODO: better turn behavior
        self._turn_id = 1

    async def accept_audio_frame(self, audio_frame: bytes):
        # Add audio_frame to pre-buffer if consumer not started; add audio_frame to recognition queue if started
        if not self._consumer_running():
            await self._pre_buffer.put(audio_frame)
        else:
            await self._audio_queue.put(audio_frame)

    async def start(self):
        # Pump pre-buffer to recognition queue; start consumer
        # Break start-once invariance warning
        if self._consumer_running():
            logger.warning(f"AudioConsumer started repeatedly; do early return")
            return
        self._ended = False
        self._paused = False
        self._start_consumer()
        await self._audio_queue.put(await self._pre_buffer.get())

    async def pause(self):
        # Stop consumer, do one recognition and publish
        # Break pause-once invariance warning
        if self._paused or self._ended:
            logger.warning(
                f"AudioConsumer paused repeatedly/pause after end; do early return"
            )
            return
        self._paused = True
        await self._stop_consumer()
        
        # send all remaining audio to ASR
        audio = await self._audio_queue.get()
        
        # log remaining audio size for debugging
        # if audio:
        #     logger.warning(f"[ASRManager] Sending remaining {len(audio)} bytes to ASR before pause")
        
        # Sentence end but not end of recognition
        await self._recognize_and_publish(audio, is_asr_end=False, is_final_chunk=True)

    async def end(self):
        # Stop consumer, do one recognition, publish ASRResultFinal, clean up states (including reset ASR) and increment turn
        # Break stop-once invariance warning
        if self._ended:
            logger.warning(f"AudioConsumer ended repeatedly; do early return")
            return
        self._ended = True
        await self._stop_consumer()
        
        # send all remaining audio to ASR
        remaining_audio = await self._audio_queue.get()  # 获取所有剩余数据
        
        # log remaining audio size for debugging
        # if remaining_audio:
        #     logger.warning(f"[ASRManager] Sending remaining {len(remaining_audio)} bytes to ASR before end")
        
        await self._recognize_and_publish(remaining_audio, is_asr_end=True, is_final_chunk=True)
        await self._reset_states()
        self._turn_id += 1

    async def _audio_consumer(self):
        while True:
            self._consumer_idle_event.set()
            if not self._consumer_running_event.is_set():
                await self._consumer_running_event.wait()
            self._consumer_idle_event.clear()
            
            # wait for data in queue
            current_size = await self._audio_queue.size()
            if current_size == 0:
                # wait for new data only when queue is empty
                await self._audio_queue.wait_on_new_bytes()
            
            # check size again (may have new data during wait)
            current_size = await self._audio_queue.size()
            least_bytes_to_send = self._asr_model.stream_chunk_bytes_hint() or 1
            
            # if consumer is about to stop and queue has data, send immediately without waiting threshold
            if not self._consumer_running_event.is_set() and current_size > 0:
                audio = await self._audio_queue.get()
                if audio:
                    await self._recognize_and_publish(audio)
                continue
            
            # if data is not enough, wait for more data to accumulate
            if current_size < least_bytes_to_send:
                await asyncio.sleep(0.05)  # wait for 50ms
                continue
                
            # get specified size of data for recognition
            audio = await self._audio_queue.get(max_bytes=least_bytes_to_send)
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
        
        # log audio sending status for debugging
        # if audio:
        #     audio_duration_ms = len(audio) / (self.SAMPLE_RATE * 2) * 1000
        #     logger.warning(
        #         f"[ASR] Sending {len(audio)} bytes ({audio_duration_ms:.1f}ms) to ASR, "
        #         f"is_final={is_final_chunk}, is_end={is_asr_end}"
        #     )
        
        recognized_text = self._recognized_text
        # Do not do recognition for empty audio
        if len(audio) > 0:
            recognized_text = await self._asr_model.async_recognize_stream(
                audio, is_final=is_final_chunk
            )
        if is_asr_end:
            self._recognized_text = recognized_text
            await self._publish_event(
                ASRResultFinal(
                    session_id=self._session_id,
                    text=recognized_text,
                    display_text=recognized_text,
                    turn_id=self._turn_id,
                )
            )
        else:
            if recognized_text == self._recognized_text and not is_final_chunk:
                return
            self._recognized_text = recognized_text
            await self._publish_event(
                ASRResultPartial(
                    session_id=self._session_id,
                    text=recognized_text,
                    display_text=recognized_text,
                    turn_id=self._turn_id,
                    speech_pause=is_final_chunk,
                )
            )

    async def _publish_event(self, event: BaseEvent):
        await self._event_bus.publish(event)

    async def _reset_states(self):
        self._recognized_text = ""
        await self._pre_buffer.get()
        await self._audio_queue.get()
        self._consumer_running_event.clear()
        self._consumer_idle_event.set()
        self._asr_model.reset()


class ASRManager(Manager):
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self._audio_consumer = AudioConsumer(event_bus, session_id, pipeline, config)

    @Manager.event_handler(EnhancedAudioFrameReceived)
    async def _handle_audio_frame(self, event: EnhancedAudioFrameReceived):
        audio_frame = event.audio_data
        await self._audio_consumer.accept_audio_frame(audio_frame)

    @Manager.event_handler(TurnASRStartRequested)
    async def _handle_asr_start(self, _):
        await self._audio_consumer.start()

    @Manager.event_handler(TurnASREndRequested)
    async def _handle_asr_end(self, _):
        await self._audio_consumer.end()

    @Manager.event_handler(TurnASRPauseRequested)
    async def _handle_asr_pause(self, _):
        await self._audio_consumer.pause()

    async def shutdown(self):
        # Task cancellation
        await self._audio_consumer.shutdown()