"""Regression tests for turn-taking event ordering during barge-in."""

from __future__ import annotations

import asyncio
import unittest

from xtalk.models import Models
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    TurnASREndRequested,
    TurnASRStartRequested,
    TurnLLMAgentStopRequested,
    VADSpeechEnd,
    VADSpeechStart,
)
from xtalk.serving.modules.turn_taking_manager import TurnTakingManager


class TurnTakingRaceTests(unittest.IsolatedAsyncioTestCase):
    """Verify short barge-ins survive a slow streaming-TTS interruption."""

    async def test_short_speech_end_waits_for_delayed_response_stop(self) -> None:
        """Order ASR start before stop, and ASR end after stop completes."""

        event_bus = EventBus()
        self.addAsyncCleanup(event_bus.shutdown)
        manager = TurnTakingManager(
            event_bus=event_bus,
            session_id="session",
            models=Models(),
        )
        transitions: list[str] = []
        stop_started = asyncio.Event()
        allow_stop_to_finish = asyncio.Event()
        asr_end_seen = asyncio.Event()

        async def handle_asr_start(event: TurnASRStartRequested) -> None:
            """Record that ASR began accepting the interruption audio."""

            del event
            transitions.append("asr_start")

        async def handle_response_stop(event: TurnLLMAgentStopRequested) -> None:
            """Simulate a StreamingTextTTS stop waiting on upstream flush."""

            del event
            transitions.append("response_stop")
            stop_started.set()
            await allow_stop_to_finish.wait()

        async def handle_asr_end(event: TurnASREndRequested) -> None:
            """Record finalization of the short interruption utterance."""

            del event
            transitions.append("asr_end")
            asr_end_seen.set()

        event_bus.subscribe(TurnASRStartRequested, handle_asr_start)
        event_bus.subscribe(TurnLLMAgentStopRequested, handle_response_stop)
        event_bus.subscribe(TurnASREndRequested, handle_asr_end)

        start_task = asyncio.create_task(
            manager._on_vad_start(
                VADSpeechStart(session_id="session"),
            )
        )
        await asyncio.wait_for(stop_started.wait(), timeout=1.0)
        self.assertEqual(transitions, ["asr_start", "response_stop"])

        end_task = asyncio.create_task(
            manager._on_vad_end(
                VADSpeechEnd(session_id="session"),
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(asr_end_seen.is_set())

        allow_stop_to_finish.set()
        await asyncio.gather(start_task, end_task)
        await asyncio.wait_for(asr_end_seen.wait(), timeout=1.0)

        self.assertEqual(
            transitions,
            ["asr_start", "response_stop", "asr_end"],
        )


if __name__ == "__main__":
    unittest.main()
