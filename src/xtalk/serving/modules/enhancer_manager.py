# -*- coding: utf-8 -*-
"""
EnhancerManager

Backend audio enhancer: when the frontend keeps raw audio (pure_frontend mode),
this manager runs the configured speech enhancer to denoise/enhance frames.
It also serializes downstream audio-frame handling so bursty frame arrival does
not become concurrent VAD/ASR/turn-detector processing.

Flow:
- Subscribe to `AudioFrameReceived`.
- If a speech enhancer exists, call `enhance()`; otherwise pass through.
- Queue raw frames and process them in order on a single worker.
- Publish `EnhancedAudioFrameReceived` for ASR/VAD.
- Flush buffered enhancer state on `VADSpeechEnd`.

Notes:
- Enhancer interface follows `SpeechEnhancer` in `speech/interfaces.py`.
- Input/output are PCM16 mono 16 kHz bytes.
- Enhancer maintains state internally for streaming.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, Any

from ...log_utils import logger

from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    AudioFrameReceived,
    EnhancedAudioFrameReceived,
    VADSpeechEnd,
)
from ...models import Models, SpeechEnhancer


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
                enhanced_data = await self.enhancer.async_enhance(item.audio_data)
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
