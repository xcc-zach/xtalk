# -*- coding: utf-8 -*-
"""
TurnDetectorManager

Manages turn detection by processing audio and ASR results through the TurnDetector.
Subscribes to:
- EnhancedAudioFrameReceived: feeds audio to turn detector
- ASRResultPartial: feeds text to turn detector
- TTSChunkGenerated: sets turn detector to non-listening
- TTSPlaybackFinished: resumes turn detector listening
- TTSStopped: resumes turn detector listening

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
    TTSChunkGenerated,
    TTSPlaybackFinished,
    TTSStopped,
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
                text=event.text, speech_pause=event.speech_pause
            )
            await self._handle_detection_result(result)

        except Exception as e:
            logger.error("[TurnDetectorManager] ASR partial processing failed: %s", e)

    @Manager.event_handler(TTSChunkGenerated)
    async def _on_tts_chunk_generated(self, event: TTSChunkGenerated) -> None:
        # TODO: subscribe client audio start playing event for accurate control
        """Set turn detector to non-listening when TTS starts producing audio."""
        if self.turn_detector is None:
            return
        # No lock needed: single bool assignment is atomic (GIL), no compound read-then-write
        self.turn_detector.listening = False

    @Manager.event_handler(TTSPlaybackFinished)
    async def _on_tts_playback_finished(self, event: TTSPlaybackFinished) -> None:
        """Resume listening when TTS playback finishes."""
        if self.turn_detector is None:
            return
        # No lock needed: single bool assignment is atomic (GIL), no compound read-then-write
        self.turn_detector.listening = True

    @Manager.event_handler(TTSStopped)
    async def _on_tts_stopped(self, event: TTSStopped) -> None:
        """Resume listening when TTS is stopped (e.g. interrupted)."""
        if self.turn_detector is None:
            return
        # No lock needed: single bool assignment is atomic (GIL), no compound read-then-write
        self.turn_detector.listening = True

    async def _handle_detection_result(
        self, result: TurnDetectionResult | list[TurnDetectionResult]
    ) -> None:
        """Handle turn detection result and emit appropriate events."""
        if not isinstance(result, list):
            result = [result]
        for item in result:
            if item.action == TurnDetectionAction.STOP_SPEAKING:
                evt = TurnDetectorStopSpeaking(
                    session_id=self.session_id, semantic=item.semantic.value
                )
                await self.event_bus.publish(evt)
            elif item.action == TurnDetectionAction.START_GENERATION:
                evt = TurnDetectorStartGeneration(
                    session_id=self.session_id, semantic=item.semantic.value
                )
                await self.event_bus.publish(evt)
        # DO_NOTHING requires no action

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """No-op shutdown hook."""
        return None
