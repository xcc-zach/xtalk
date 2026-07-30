import asyncio
from typing import Any
from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    VADSpeechStart,
    VADSpeechEnd,
    TurnLLMAgentStopRequested,
    TurnASRStartRequested,
    TurnASRPauseRequested,
    TurnASREndRequested,
    TurnDetectorStartGeneration,
    TurnDetectorStopSpeaking,
    TTSPlaybackFinished,
)
from ...models import Models, TurnDetector


class TurnTakingManager(Manager):
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
        self._vad_transition_lock = asyncio.Lock()

    @Manager.event_handler(TurnDetectorStartGeneration, priority=99)
    async def _on_turn_detector_start_generation(
        self, _event: TurnDetectorStartGeneration
    ):
        await self.event_bus.publish(TurnASREndRequested(session_id=self.session_id))

    # Stop speaking must be handled before starting generation
    @Manager.event_handler(TurnDetectorStopSpeaking, priority=100)
    async def _on_turn_detector_stop_speaking(self, _event: TurnDetectorStopSpeaking):
        await self.event_bus.publish(
            TurnLLMAgentStopRequested(
                session_id=self.session_id,
            ),
            wait_for_completion=True,
        )

    @Manager.event_handler(VADSpeechStart)
    async def _on_vad_start(self, _event: VADSpeechStart):
        """Start ASR immediately, then finish interrupting the previous response."""
        async with self._vad_transition_lock:
            await self.event_bus.publish(
                TurnASRStartRequested(session_id=self.session_id),
                wait_for_completion=True,
            )
            if self._turn_detector_model is None:
                await self.event_bus.publish(
                    TurnLLMAgentStopRequested(
                        session_id=self.session_id,
                    ),
                    wait_for_completion=True,
                )

    @Manager.event_handler(VADSpeechEnd)
    async def _on_vad_end(self, _event: VADSpeechEnd):
        """Finalize only after any in-flight response interruption completes."""
        async with self._vad_transition_lock:
            if self._turn_detector_model is None:
                await self.event_bus.publish(
                    TurnASREndRequested(session_id=self.session_id)
                )
            else:
                await self.event_bus.publish(
                    TurnASRPauseRequested(session_id=self.session_id)
                )

    @Manager.event_handler(TTSPlaybackFinished)
    async def _on_tts_playback_finished(self, _event: TTSPlaybackFinished) -> None:
        """Frontend playback finished - stop LLM agent to clean up."""
        # TODO: check whether this event handler is necessary
        await self.event_bus.publish(
            TurnLLMAgentStopRequested(
                session_id=self.session_id,
                reason="playback_finished",
            )
        )

    async def shutdown(self):
        return
