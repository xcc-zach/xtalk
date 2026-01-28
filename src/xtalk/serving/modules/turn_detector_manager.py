# -*- coding: utf-8 -*-
"""
TurnDetectorManager

Manages turn detection by processing audio and ASR results through the TurnDetector.
Subscribes to:
- EnhancedAudioFrameReceived: feeds audio to turn detector
- ASRResultPartial/ASRResultFinal: feeds text and finality to turn detector

Emits:
- TurnDetectorStopSpeaking: when action is STOP_SPEAKING
- TurnDetectorStartGeneration: when action is START_GENERATION
"""

from __future__ import annotations

from typing import Optional, Any

from ...log_utils import logger

from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    EnhancedAudioFrameReceived,
    ASRResultPartial,
    ASRResultFinal,
    TurnDetectorStopSpeaking,
    TurnDetectorStartGeneration,
)
from ...pipelines import Pipeline
from ...speech.interfaces import TurnDetectionAction, TurnDetectionResult


class TurnDetectorManager(Manager):
    """Manager for turn detection processing."""

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

        # Get turn detector from pipeline
        self.turn_detector = self.pipeline.get_turn_detector_model()

    # ----------------------------
    # Event handling
    # ----------------------------
    @Manager.event_handler(EnhancedAudioFrameReceived)
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Process audio frames through turn detector."""
        try:
            if self.turn_detector is None:
                return

            if not event.audio_data:
                return

            result = await self.turn_detector.async_detect(audio=event.audio_data)
            await self._handle_detection_result(result)

        except Exception as e:
            logger.error("[TurnDetectorManager] audio frame processing failed: %s", e)

    @Manager.event_handler(ASRResultPartial)
    async def _on_asr_partial(self, event: ASRResultPartial) -> None:
        """Process partial ASR results through turn detector."""
        try:
            if self.turn_detector is None:
                return

            if not event.text:
                return

            result = await self.turn_detector.async_detect(
                text=event.text, asr_final=False
            )
            await self._handle_detection_result(result)

        except Exception as e:
            logger.error("[TurnDetectorManager] ASR partial processing failed: %s", e)

    @Manager.event_handler(ASRResultFinal)
    async def _on_asr_final(self, event: ASRResultFinal) -> None:
        """Process final ASR results through turn detector."""
        try:
            if self.turn_detector is None:
                return

            if not event.text:
                return

            result = await self.turn_detector.async_detect(
                text=event.text, asr_final=True
            )
            await self._handle_detection_result(result)

        except Exception as e:
            logger.error("[TurnDetectorManager] ASR final processing failed: %s", e)

    async def _handle_detection_result(self, result: TurnDetectionResult) -> None:
        """Handle turn detection result and emit appropriate events."""
        if result.action == TurnDetectionAction.STOP_SPEAKING:
            evt = TurnDetectorStopSpeaking(
                session_id=self.session_id, semantic=result.semantic.value
            )
            await self.event_bus.publish(evt)
        elif result.action == TurnDetectionAction.START_GENERATION:
            evt = TurnDetectorStartGeneration(
                session_id=self.session_id, semantic=result.semantic.value
            )
            await self.event_bus.publish(evt)
        # DO_NOTHING requires no action

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """No-op shutdown hook."""
        return None
