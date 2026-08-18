import asyncio
from typing import Any
from ..event_bus import EventBus, EventDispatchMode
from ..interfaces import Manager
from ..events import (
    VADSpeechStart,
    VADSpeechEnd,
    SpeakerInterruptionDecision,
    TurnLLMAgentPauseRequested,
    TurnLLMAgentResumeRequested,
    TurnLLMAgentStopRequested,
    TurnASRStartRequested,
    TurnASRPauseRequested,
    TurnASREndRequested,
    TurnDetectorStartGeneration,
    TurnDetectorStopSpeaking,
)
from ...models import Models, SpeakerDiarization, TurnDetector


class TurnTakingManager(Manager):
    """Coordinate VAD boundaries with ASR and response interruption."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self._turn_detector_model = models.get(TurnDetector)
        self._speaker_diarization_enabled = models.get(SpeakerDiarization) is not None
        self._vad_transition_lock = asyncio.Lock()
        self._next_turn_id = 0
        self._next_segment_id = 0
        self._turn_open = False
        self._current_turn_id = 0
        self._current_segment_id = 0
        self._speaker_decision_pending = False

    @Manager.event_handler(TurnDetectorStartGeneration, priority=99)
    async def _on_turn_detector_start_generation(
        self, _event: TurnDetectorStartGeneration
    ):
        await self._publish_asr_end()

    # Stop speaking must be handled before starting generation
    @Manager.event_handler(TurnDetectorStopSpeaking, priority=100)
    async def _on_turn_detector_stop_speaking(self, _event: TurnDetectorStopSpeaking):
        if self._speaker_diarization_enabled:
            return
        await self.event_bus.publish(
            TurnLLMAgentStopRequested(
                session_id=self.session_id,
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    @Manager.event_handler(VADSpeechStart)
    async def _on_vad_start(self, event: VADSpeechStart):
        """Start ASR with stable turn IDs, then interrupt the previous response."""
        async with self._vad_transition_lock:
            if event.origin == "text":
                await self.event_bus.publish(
                    TurnLLMAgentStopRequested(
                        session_id=self.session_id,
                        reason="text_input",
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
                return

            if not self._turn_open:
                self._next_turn_id += 1
                self._current_turn_id = self._next_turn_id
                self._turn_open = True
            self._next_segment_id += 1
            self._current_segment_id = self._next_segment_id
            if self._speaker_diarization_enabled:
                self._speaker_decision_pending = True
                await self.event_bus.publish(
                    TurnLLMAgentPauseRequested(session_id=self.session_id),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
            await self.event_bus.publish(
                TurnASRStartRequested(
                    session_id=self.session_id,
                    turn_id=self._current_turn_id,
                    segment_id=self._current_segment_id,
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )
            if self._speaker_diarization_enabled:
                return
            if self._turn_detector_model is None:
                await self.event_bus.publish(
                    TurnLLMAgentStopRequested(
                        session_id=self.session_id,
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )

    @Manager.event_handler(SpeakerInterruptionDecision, priority=100)
    async def _on_speaker_interruption_decision(
        self,
        event: SpeakerInterruptionDecision,
    ) -> None:
        """Commit or discard a paused interruption for the current segment."""

        if (
            not self._speaker_decision_pending
            or event.turn_id != self._current_turn_id
            or event.segment_id != self._current_segment_id
        ):
            return
        self._speaker_decision_pending = False
        event_type = (
            TurnLLMAgentStopRequested
            if event.should_interrupt
            else TurnLLMAgentResumeRequested
        )
        await self.event_bus.publish(
            event_type(session_id=self.session_id),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

    @Manager.event_handler(VADSpeechEnd)
    async def _on_vad_end(self, event: VADSpeechEnd):
        """Finalize only after any in-flight response interruption completes."""
        async with self._vad_transition_lock:
            if event.origin == "text":
                return

            if self._turn_detector_model is None:
                await self._publish_asr_end(
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE
                )
            else:
                await self.event_bus.publish(
                    TurnASRPauseRequested(
                        session_id=self.session_id,
                        turn_id=self._current_turn_id,
                        segment_id=self._current_segment_id,
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )

    async def _publish_asr_end(
        self,
        *,
        mode: EventDispatchMode = EventDispatchMode.RETURN_AFTER_DISPATCH,
    ) -> None:
        """Publish the current hard-turn boundary and close its ID scope."""

        if not self._turn_open:
            return
        await self.event_bus.publish(
            TurnASREndRequested(
                session_id=self.session_id,
                turn_id=self._current_turn_id,
                segment_id=self._current_segment_id,
            ),
            mode=mode,
        )
        self._turn_open = False

    async def shutdown(self):
        return
