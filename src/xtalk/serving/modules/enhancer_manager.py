# -*- coding: utf-8 -*-
"""
EnhancerManager

Backend audio enhancer: when the frontend keeps raw audio (pure_frontend mode),
this manager runs `pipeline.enhancer_model` to denoise/enhance frames.

Flow:
- Subscribe to `AudioFrameReceived`.
- If `pipeline.enhancer_model` exists, call `enhance()`; otherwise pass through.
- Publish `EnhancedAudioFrameReceived` for ASR/VAD.
- Flush buffered enhancer state on `VADSpeechEnd`.

Notes:
- Enhancer interface follows `SpeechEnhancer` in `speech/interfaces.py`.
- Input/output are PCM16 mono 16 kHz bytes.
- Enhancer maintains state internally for streaming.
"""

from __future__ import annotations

from typing import Optional, Any

from ...log_utils import logger

from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    AudioFrameReceived,
    EnhancedAudioFrameReceived,
    VADSpeechEnd,
)
from ...pipelines import Pipeline


class EnhancerManager(Manager):
    """Backend speech enhancement manager."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        self.config: dict[str, Any] = config or {}

        # Only enable when enhancer model is provided
        self.enhancer = self.pipeline.get_enhancer_model()
        self._last_sample_rate = 16000

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
            self._last_sample_rate = event.sample_rate
            enhanced_data = event.audio_data

            # Run enhancer if configured and audio is present
            if self.enhancer is not None and event.audio_data:
                try:
                    enhanced_data = await self.enhancer.async_enhance(event.audio_data)
                except Exception as e:
                    logger.error("[EnhancerManager] Enhancement failed: %s", e)

            # Publish enhanced frame
            enhanced_event = EnhancedAudioFrameReceived(
                session_id=event.session_id,
                audio_data=enhanced_data,
                sample_rate=event.sample_rate,
            )
            await self.event_bus.publish(enhanced_event)

        except Exception as e:
            logger.error("[EnhancerManager] handle frame failed: %s", e)

    @Manager.event_handler(VADSpeechEnd, priority=150)
    async def _on_vad_end(self, event: VADSpeechEnd) -> None:
        """Flush enhancer state before ASR end/pause."""
        if self.enhancer is None:
            return

        try:
            flushed_data = await self.enhancer.async_flush()
        except Exception as e:
            logger.warning("[EnhancerManager] Flush failed (non-critical): %s", e)
            return

        if not flushed_data:
            return

        await self.event_bus.publish(
            EnhancedAudioFrameReceived(
                session_id=event.session_id,
                audio_data=flushed_data,
                sample_rate=self._last_sample_rate,
            )
        )

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """Reset enhancer state on shutdown."""
        if self.enhancer is not None:
            try:
                self.enhancer.reset()
            except Exception as e:
                logger.error("[EnhancerManager] Reset enhancer failed: %s", e)
        return None
