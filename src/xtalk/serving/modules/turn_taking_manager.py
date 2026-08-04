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
        self._next_turn_id = 0
        self._next_segment_id = 0
        self._turn_open = False
        self._current_turn_id = 0
        self._current_segment_id = 0

    @Manager.event_handler(TurnDetectorStartGeneration, priority=99)
    async def _on_turn_detector_start_generation(
        self, _event: TurnDetectorStartGeneration
    ):
        await self._publish_asr_end()

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
        """Start ASR with stable turn IDs, then interrupt the previous response."""
        async with self._vad_transition_lock:
            if not self._turn_open:
                self._next_turn_id += 1
                self._current_turn_id = self._next_turn_id
                self._turn_open = True
            self._next_segment_id += 1
            self._current_segment_id = self._next_segment_id
            await self.event_bus.publish(
                TurnASRStartRequested(
                    session_id=self.session_id,
                    turn_id=self._current_turn_id,
                    segment_id=self._current_segment_id,
                ),
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
                await self._publish_asr_end(wait_for_completion=True)
            else:
                await self.event_bus.publish(
                    TurnASRPauseRequested(
                        session_id=self.session_id,
                        turn_id=self._current_turn_id,
                        segment_id=self._current_segment_id,
                    ),
                    wait_for_completion=True,
                )

    async def _publish_asr_end(
        self,
        *,
        wait_for_completion: bool = False,
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
            wait_for_completion=wait_for_completion,
        )
        self._turn_open = False

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
