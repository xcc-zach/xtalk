# -*- coding: utf-8 -*-
"""
EnhancerManager

Backend audio enhancer: when the frontend keeps raw audio (pure_frontend mode),
this manager runs the configured speech enhancer to denoise/enhance frames.
It also serializes downstream enhanced-audio handling so bursty frame arrival
does not become concurrent VAD/speaker/turn-detector processing.

Flow:
- Subscribe to `AudioFrameReceived`.
- If a speech enhancer exists, call `enhance()`; otherwise pass through.
- Queue raw frames and process them in order on a single worker.
- Publish `EnhancedAudioFrameReceived` for VAD, speaker, and turn detection.
- Flush buffered enhancer state on `VADSpeechEnd`.

Notes:
- Enhancer interface follows `SpeechEnhancer` in `speech/interfaces.py`.
- Input/output are PCM16 mono 16 kHz bytes.
- Enhancer maintains state internally for streaming.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ...models import Models, SpeechEnhancer
from ..event_bus import EventBus
from ..events import (
    AudioFrameReceived,
    EnhancedAudioFrameReceived,
    TTSChunkPlayed,
    TTSChunkReady,
    TTSPlaybackFinished,
    TTSStarted,
    TTSStopped,
    VADSpeechEnd,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


_ENHANCER_SAMPLE_RATE = 16000


class _FarReferenceBuffer:
    """Far-end PCM reference buffer aligned against microphone frame consumption."""

    def __init__(self, *, sample_rate: int, max_seconds: float) -> None:
        self.sample_rate = sample_rate
        self._max_bytes = max(0, int(sample_rate * max_seconds) * 2)
        self._buffer = bytearray()
        self._played_chunk_bytes: deque[int] = deque()
        self._consumed_since_played_ack = 0

    def append(self, pcm: bytes, *, sample_rate: int) -> None:
        """Append one TTS PCM chunk after converting it to enhancer format."""
        reference = self._to_reference_pcm(pcm, sample_rate)
        if not reference:
            return
        self._buffer.extend(reference)
        self._played_chunk_bytes.append(len(reference))
        self._trim_to_limit()

    def take(self, byte_count: int) -> bytes:
        """Take an exact-length far reference frame, padding silence if needed."""
        if byte_count <= 0:
            return b""
        available = min(byte_count, len(self._buffer))
        chunk = bytes(self._buffer[:available])
        del self._buffer[:available]
        self._consumed_since_played_ack += available
        if available < byte_count:
            chunk += bytes(byte_count - available)
        return chunk

    def mark_chunk_played(self) -> None:
        """Discard playback-confirmed reference that microphone frames did not use."""
        if not self._played_chunk_bytes:
            return
        played_bytes = self._played_chunk_bytes.popleft()
        if self._consumed_since_played_ack >= played_bytes:
            self._consumed_since_played_ack -= played_bytes
            return
        stale_bytes = played_bytes - self._consumed_since_played_ack
        del self._buffer[: min(stale_bytes, len(self._buffer))]
        self._consumed_since_played_ack = 0

    def clear(self) -> None:
        """Clear all buffered far-end reference audio and playback accounting."""
        self._buffer.clear()
        self._played_chunk_bytes.clear()
        self._consumed_since_played_ack = 0

    def _trim_to_limit(self) -> None:
        """Trim old reference audio when the buffer grows beyond its limit."""
        if self._max_bytes <= 0:
            self.clear()
            return
        overflow = len(self._buffer) - self._max_bytes
        if overflow <= 0:
            return
        del self._buffer[:overflow]
        while overflow > 0 and self._played_chunk_bytes:
            chunk_bytes = self._played_chunk_bytes[0]
            if overflow < chunk_bytes:
                self._played_chunk_bytes[0] = chunk_bytes - overflow
                overflow = 0
            else:
                overflow -= self._played_chunk_bytes.popleft()
        self._consumed_since_played_ack = min(
            self._consumed_since_played_ack, len(self._buffer)
        )

    def _to_reference_pcm(self, pcm: bytes, sample_rate: int) -> bytes:
        """Convert PCM16 mono bytes to 16 kHz PCM16 mono bytes."""
        even_length = len(pcm) - (len(pcm) % 2)
        if even_length <= 0 or sample_rate <= 0:
            return b""
        pcm = pcm[:even_length]
        if sample_rate == self.sample_rate:
            return pcm

        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return b""
        output_size = max(1, round(samples.size * self.sample_rate / sample_rate))
        if output_size == samples.size:
            return pcm
        x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=output_size, endpoint=False)
        resampled = np.interp(x_new, x_old, samples.astype(np.float32))
        return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


@dataclass(slots=True)
class _QueuedAudioFrame:
    """Audio frame waiting for serialized enhancement and dispatch."""

    session_id: str
    audio_data: bytes
    sample_rate: int


@dataclass(slots=True)
class _QueuedFlush:
    """Barrier request that flushes enhancer state after earlier frames."""

    session_id: str
    done: asyncio.Future[None]


class EnhancerManager(Manager):
    """Backend speech enhancement manager."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        # Only enable when enhancer model is provided
        self.enhancer = models.get(SpeechEnhancer)
        self._last_sample_rate = 16000
        far_buffer_seconds = float(
            self.config.get("far_reference_buffer_seconds", 5.0)
        )
        self._far_reference = _FarReferenceBuffer(
            sample_rate=_ENHANCER_SAMPLE_RATE,
            max_seconds=far_buffer_seconds,
        )
        # EnhancerManager is also the serialized handoff for raw audio frames so
        # network jitter does not turn into concurrent downstream audio handling.
        self._audio_queue: asyncio.Queue[_QueuedAudioFrame | _QueuedFlush | None] = (
            asyncio.Queue()
        )
        self._audio_worker_task = asyncio.create_task(self._audio_worker())

    # ----------------------------
    # Event handling
    # ----------------------------
    @Manager.event_handler(
        AudioFrameReceived,
        priority=150,  # Higher than VADManager (100) to publish enhanced audio first
    )
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        """Handle raw frames; enhance when available, otherwise passthrough."""
        try:
            # Keep the event handler lightweight: queueing here lets the manager
            # absorb bursty arrival timing and stabilize the audio stream for
            # enhancement and every downstream consumer.
            await self._audio_queue.put(
                _QueuedAudioFrame(
                    session_id=event.session_id,
                    audio_data=event.audio_data,
                    sample_rate=event.sample_rate,
                )
            )

        except Exception as e:
            logger.error("[EnhancerManager] handle frame failed: %s", e)

    @Manager.event_handler(TTSChunkReady, priority=60)
    async def _on_tts_chunk_ready(self, event: TTSChunkReady) -> None:
        """Queue outbound TTS audio as far-end echo-cancellation reference."""
        if self.enhancer is None:
            return
        try:
            self._far_reference.append(
                event.audio_chunk or b"",
                sample_rate=event.sample_rate or _ENHANCER_SAMPLE_RATE,
            )
        except Exception as e:
            logger.warning("[EnhancerManager] Far reference append failed: %s", e)

    @Manager.event_handler(TTSChunkPlayed, priority=60)
    async def _on_tts_chunk_played(self, event: TTSChunkPlayed) -> None:
        """Advance far-reference playback accounting after frontend confirmation."""
        del event
        self._far_reference.mark_chunk_played()

    @Manager.event_handler(TTSPlaybackFinished, priority=60)
    async def _on_tts_playback_finished(self, event: TTSPlaybackFinished) -> None:
        """Drop stale far reference when the frontend reports playback completion."""
        del event
        self._far_reference.clear()

    @Manager.event_handler(TTSStarted, priority=60)
    async def _on_tts_started(self, event: TTSStarted) -> None:
        """Start each TTS response with fresh far-reference state."""
        del event
        self._far_reference.clear()

    @Manager.event_handler(TTSStopped, priority=60)
    async def _on_tts_stopped(self, event: TTSStopped) -> None:
        """Clear far reference when playback is explicitly stopped."""
        del event
        self._far_reference.clear()

    @Manager.event_handler(VADSpeechEnd, priority=150)
    async def _on_vad_end(self, event: VADSpeechEnd) -> None:
        """Flush enhancer state before ASR end/pause."""
        try:
            # When VADSpeechEnd is emitted while the worker is synchronously
            # publishing a frame, flushing inline preserves order and avoids a
            # self-deadlock on the queue barrier.
            if asyncio.current_task() is self._audio_worker_task:
                await self._flush_enhancer(event.session_id)
                return

            loop = asyncio.get_running_loop()
            done = loop.create_future()
            # External VAD end events must wait until every earlier frame has
            # been enhanced and republished before downstream end-of-speech
            # handlers run.
            await self._audio_queue.put(
                _QueuedFlush(session_id=event.session_id, done=done)
            )
            await done

        except Exception as e:
            logger.warning("[EnhancerManager] Flush failed (non-critical): %s", e)

    async def _audio_worker(self) -> None:
        """Process queued audio work items in strict arrival order."""
        try:
            while True:
                item = await self._audio_queue.get()
                try:
                    if item is None:
                        return

                    if isinstance(item, _QueuedAudioFrame):
                        await self._process_audio_frame(item)
                    else:
                        await self._flush_enhancer(item.session_id)
                        if not item.done.done():
                            item.done.set_result(None)

                except Exception as e:
                    if isinstance(item, _QueuedFlush) and not item.done.done():
                        item.done.set_exception(e)
                    logger.error("[EnhancerManager] worker failed: %s", e)
                finally:
                    self._audio_queue.task_done()
        except asyncio.CancelledError:
            raise

    async def _process_audio_frame(self, item: _QueuedAudioFrame) -> None:
        """Enhance or forward one audio frame and publish it downstream."""
        self._last_sample_rate = item.sample_rate
        enhanced_data = item.audio_data

        # The enhancer keeps streaming state internally, so frames must be fed
        # to it strictly in order on a single worker.
        if self.enhancer is not None and item.audio_data:
            try:
                far_data = self._far_reference.take(len(item.audio_data))
                enhanced_data = await self.enhancer.async_enhance(
                    item.audio_data,
                    far=far_data,
                )
            except Exception as e:
                logger.error("[EnhancerManager] Enhancement failed: %s", e)

        await self._publish_enhanced_audio(
            session_id=item.session_id,
            audio_data=enhanced_data,
            sample_rate=item.sample_rate,
        )

    async def _flush_enhancer(self, session_id: str) -> None:
        """Flush enhancer state after all earlier audio frames are processed."""
        if self.enhancer is None:
            return

        try:
            flushed_data = await self.enhancer.async_flush()
        except Exception as e:
            logger.warning("[EnhancerManager] Flush failed (non-critical): %s", e)
            return

        if not flushed_data:
            return

        await self._publish_enhanced_audio(
            session_id=session_id,
            audio_data=flushed_data,
            sample_rate=self._last_sample_rate,
        )

    async def _publish_enhanced_audio(
        self,
        *,
        session_id: str,
        audio_data: bytes,
        sample_rate: int,
    ) -> None:
        """Publish one enhanced frame and wait for downstream completion."""
        # Preserve frame order through VAD/ASR/turn detection before the next
        # frame starts. EnhancerManager now stabilizes audio event handling in
        # addition to performing enhancement.
        await self.event_bus.publish(
            EnhancedAudioFrameReceived(
                session_id=session_id,
                audio_data=audio_data,
                sample_rate=sample_rate,
            ),
            wait_for_completion=True,
        )

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """Reset enhancer state on shutdown."""
        await self._audio_queue.put(None)
        try:
            await self._audio_worker_task
        except asyncio.CancelledError:
            pass
        if self.enhancer is not None:
            try:
                self.enhancer.reset()
            except Exception as e:
                logger.error("[EnhancerManager] Reset enhancer failed: %s", e)
        return None
