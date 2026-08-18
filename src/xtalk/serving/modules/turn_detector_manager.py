# -*- coding: utf-8 -*-
"""
TurnDetectorManager

Manages turn detection by processing audio and ASR results through the TurnDetector.
Subscribes to:
- EnhancedAudioFrameReceived: feeds audio to turn detector
- ASRResultPartial: feeds text to turn detector
- ResponseUpdate: feeds cumulative played AI response text to turn detector
- ResponseFinish: feeds final played AI response text to turn detector
- TTSChunkGenerated: sets turn detector to non-listening
- TTSPlaybackFinished: resumes turn detector listening
- TTSStopped: resumes turn detector listening

Emits:
- TurnDetectorStopSpeaking: when action is STOP_SPEAKING
- TurnDetectorStartGeneration: when action is START_GENERATION
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ...models import VAD, Models, TurnDetector
from ...models.turn_detector.interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnVADResult,
)
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    ASRResultPartial,
    EnhancedAudioFrameReceived,
    ResponseFinish,
    ResponseUpdate,
    TTSChunkReady,
    TTSPlaybackFinished,
    TTSStopped,
    TurnDetectorStartGeneration,
    TurnDetectorStopSpeaking,
    VADSpeechEnd,
    VADSpeechStart,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class TurnDetectorManager(Manager):
    """Manager for turn detection processing."""

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

        # Get turn detector from configured models.
        self.turn_detector = models.get(TurnDetector)
        self._backend_vad = models.get(VAD)
        self._proxy_vad_enabled = self._backend_vad is None
        self._proxy_vad_in_speech = False

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
            if event.origin == "text":
                return

            if not event.text:
                return

            result = await self.turn_detector.async_detect(
                text=event.text, speech_pause=event.speech_pause
            )
            await self._handle_detection_result(result)

        except Exception as e:
            logger.error("[TurnDetectorManager] ASR partial processing failed: %s", e)

    @Manager.event_handler(VADSpeechStart)
    async def _on_vad_speech_start(self, event: VADSpeechStart) -> None:
        """Notify the turn detector that VAD has just detected speech start."""
        try:
            if self.turn_detector is None:
                return
            if event.origin in {"text", "turn_detector"}:
                return

            self._disable_proxy_vad()
            result = await self.turn_detector.async_detect(speech_start=True)
            await self._handle_detection_result(result)
        except Exception as e:
            logger.error("[TurnDetectorManager] VAD start processing failed: %s", e)

    @Manager.event_handler(VADSpeechEnd)
    async def _on_vad_speech_end(self, event: VADSpeechEnd) -> None:
        """Track externally supplied VAD end events so proxy mode stays disabled."""
        if event.origin in {"text", "turn_detector"}:
            return
        self._disable_proxy_vad()

    @Manager.event_handler(ResponseUpdate, priority=100)
    @Manager.event_handler(ResponseFinish, priority=100)
    async def _on_assistant_text(
        self,
        event: ResponseUpdate | ResponseFinish,
    ) -> None:
        """Forward cumulative played AI response text to the turn detector."""
        if self.turn_detector is None:
            return
        await self.turn_detector.async_detect(assistant_text=event.text)

    @Manager.event_handler(TTSChunkReady)
    async def _on_tts_chunk_generated(self, event: TTSChunkReady) -> None:
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

    async def _handle_detection_result(self, result: TurnDetectionResult) -> None:
        """Handle one turn detection result and emit the corresponding events."""
        await self._handle_proxy_vad_result(result)
        if result.action == TurnDetectionAction.STOP_SPEAKING:
            evt = TurnDetectorStopSpeaking(session_id=self.session_id)
            await self.event_bus.publish(evt)
        elif result.action == TurnDetectionAction.START_GENERATION:
            evt = TurnDetectorStartGeneration(session_id=self.session_id)
            await self.event_bus.publish(evt)
        # DO_NOTHING requires no action

    def _disable_proxy_vad(self) -> None:
        """Disable turn-detector VAD proxy mode after observing external VAD."""
        self._proxy_vad_enabled = False

    async def _handle_proxy_vad_result(self, result: TurnDetectionResult) -> None:
        """Emit proxied VAD transitions before handling turn-detection actions."""
        if not self._proxy_vad_enabled or result.vad_result is None:
            return

        if result.vad_result == TurnVADResult.SPEECH:
            if self._proxy_vad_in_speech:
                return
            self._proxy_vad_in_speech = True
            await self.event_bus.publish(
                VADSpeechStart(
                    session_id=self.session_id,
                    origin="turn_detector",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )
            return

        if result.vad_result == TurnVADResult.SILENCE:
            if not self._proxy_vad_in_speech:
                return
            self._proxy_vad_in_speech = False
            await self.event_bus.publish(
                VADSpeechEnd(
                    session_id=self.session_id,
                    origin="turn_detector",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """No-op shutdown hook."""
        return None
