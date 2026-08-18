"""Tests for websocket text input as a synthetic speech turn."""

from __future__ import annotations

import asyncio
import json
import unittest
from typing import Any, cast
from unittest.mock import AsyncMock

from fastapi import WebSocket

from xtalk.models import Models
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    ASRResultFinal,
    ASRResultPartial,
    EnhancedAudioFrameReceived,
    TurnASREndRequested,
    TurnASRStartRequested,
    TurnInputAbortRequested,
    TurnLLMAgentStopRequested,
    VADSpeechEnd,
    VADSpeechStart,
    WebSocketMessageReceived,
)
from xtalk.serving.modules.input_gateway import (
    MAX_TEXT_INPUT_BYTES,
    TextMsgHandler,
)
from xtalk.serving.modules.latency_manager import LatencyManager
from xtalk.serving.modules.output_gateway import OutputGateway
from xtalk.serving.modules.turn_detector_manager import TurnDetectorManager
from xtalk.serving.modules.turn_taking_manager import TurnTakingManager
from xtalk.serving.modules.vad_manager import VADManager


def _unused_websocket() -> WebSocket:
    """Return a type-compatible websocket unused by text submission tests."""

    return cast(WebSocket, cast(Any, object()))


class _RecordingWebSocket:
    """Record JSON text frames sent by an output gateway."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send_text(self, message: str) -> None:
        """Record one outbound text frame."""

        self.messages.append(message)


class TextInputEventTests(unittest.IsolatedAsyncioTestCase):
    """Verify text submissions emit one ordered synthetic speech turn."""

    async def asyncSetUp(self) -> None:
        """Create one event bus and text handler for each test."""

        self.event_bus = EventBus(enable_history=True)
        self.addAsyncCleanup(self.event_bus.shutdown)
        self.handler = TextMsgHandler(
            event_bus=self.event_bus,
            session_id="session",
            websocket=_unused_websocket(),
        )

    async def test_submit_text_emits_complete_ordered_turn(self) -> None:
        """Emit abort, VAD boundaries, and ASR text in the expected order."""

        await self.handler._handle_websocket_message_received(
            WebSocketMessageReceived(
                session_id="session",
                message=json.dumps(
                    {
                        "action": "submit_text",
                        "text": "  设置一个两秒计时器。  ",
                        "origin": "client-spoof",
                    }
                ),
            )
        )

        events = self.event_bus.get_history()
        self.assertEqual(
            [type(event) for event in events],
            [
                TurnInputAbortRequested,
                VADSpeechStart,
                ASRResultPartial,
                VADSpeechEnd,
                ASRResultFinal,
            ],
        )
        self.assertTrue(all(event.origin == "text" for event in events))
        partial = cast(ASRResultPartial, events[2])
        final = cast(ASRResultFinal, events[4])
        self.assertEqual(partial.text, "设置一个两秒计时器。")
        self.assertEqual(partial.display_text, partial.text)
        self.assertTrue(partial.speech_pause)
        self.assertEqual(final.text, partial.text)
        self.assertEqual(final.display_text, partial.text)

    async def test_submit_text_rejects_invalid_payloads(self) -> None:
        """Do not begin synthetic turns for invalid or oversized text."""

        for payload in (
            {"action": "submit_text"},
            {"action": "submit_text", "text": 42},
            {"action": "submit_text", "text": " \n\t "},
            {
                "action": "submit_text",
                "text": "x" * (MAX_TEXT_INPUT_BYTES + 1),
            },
        ):
            await self.handler._handle_websocket_message_received(
                WebSocketMessageReceived(
                    session_id="session",
                    message=json.dumps(payload),
                )
            )

        self.assertEqual(self.event_bus.get_history(), [])

    async def test_text_turns_are_serialized_per_session(self) -> None:
        """Prevent two text submissions from interleaving their event sequences."""

        first_start_seen = asyncio.Event()
        allow_first_start = asyncio.Event()
        start_count = 0

        async def block_first_start(event: VADSpeechStart) -> None:
            """Hold the first turn after VAD start while the second is submitted."""

            nonlocal start_count
            if event.origin != "text":
                return
            start_count += 1
            if start_count == 1:
                first_start_seen.set()
                await allow_first_start.wait()

        self.event_bus.subscribe(VADSpeechStart, block_first_start, priority=200)

        first_task = asyncio.create_task(
            self.handler._handle_submit_text({"text": "first"})
        )
        await asyncio.wait_for(first_start_seen.wait(), timeout=1.0)
        second_task = asyncio.create_task(
            self.handler._handle_submit_text({"text": "second"})
        )
        await asyncio.sleep(0)

        self.assertEqual(
            [type(event) for event in self.event_bus.get_history()],
            [TurnInputAbortRequested, VADSpeechStart],
        )

        allow_first_start.set()
        await asyncio.gather(first_task, second_task)
        self.assertEqual(
            [type(event) for event in self.event_bus.get_history()],
            [
                TurnInputAbortRequested,
                VADSpeechStart,
                ASRResultPartial,
                VADSpeechEnd,
                ASRResultFinal,
                TurnInputAbortRequested,
                VADSpeechStart,
                ASRResultPartial,
                VADSpeechEnd,
                ASRResultFinal,
            ],
        )

    async def test_finish_asr_echoes_text_origin(self) -> None:
        """Expose origin on the existing receipt so audio cannot confirm text."""

        websocket = _RecordingWebSocket()
        gateway = OutputGateway(
            event_bus=self.event_bus,
            session_id="session",
            websocket=cast(WebSocket, cast(Any, websocket)),
        )

        await gateway._send_finish_asr_signal(
            ASRResultFinal(
                session_id="session",
                text="typed",
                display_text="typed",
                origin="text",
            )
        )

        self.assertEqual(
            json.loads(websocket.messages[0]),
            {
                "action": "finish_asr",
                "data": {
                    "text": "typed",
                    "origin": "text",
                },
            },
        )


class TextInputManagerTests(unittest.IsolatedAsyncioTestCase):
    """Verify managers treat text as an explicit interrupting turn."""

    async def asyncSetUp(self) -> None:
        """Create a fresh event bus for each manager test."""

        self.event_bus = EventBus()
        self.addAsyncCleanup(self.event_bus.shutdown)

    async def test_turn_taking_interrupts_without_starting_audio_asr(self) -> None:
        """Stop the current response while bypassing audio ASR lifecycle events."""

        manager = TurnTakingManager(
            event_bus=self.event_bus,
            session_id="session",
            models=Models(),
        )
        transitions: list[str] = []

        async def record_asr_start(event: TurnASRStartRequested) -> None:
            """Record an unexpected audio-ASR start."""

            del event
            transitions.append("asr_start")

        async def record_asr_end(event: TurnASREndRequested) -> None:
            """Record an unexpected audio-ASR end."""

            del event
            transitions.append("asr_end")

        async def record_response_stop(event: TurnLLMAgentStopRequested) -> None:
            """Record interruption of the active assistant response."""

            transitions.append(f"response_stop:{event.reason}")

        self.event_bus.subscribe(TurnASRStartRequested, record_asr_start)
        self.event_bus.subscribe(TurnASREndRequested, record_asr_end)
        self.event_bus.subscribe(TurnLLMAgentStopRequested, record_response_stop)

        await manager._on_vad_start(
            VADSpeechStart(session_id="session", origin="text")
        )
        await manager._on_vad_end(
            VADSpeechEnd(session_id="session", origin="text")
        )

        self.assertEqual(transitions, ["response_stop:text_input"])

    async def test_vad_state_is_suppressed_until_text_vad_end(self) -> None:
        """Clear stale VAD state and drop microphone frames during text input."""

        manager = VADManager(
            event_bus=self.event_bus,
            session_id="session",
            models=Models(),
        )
        manager._buf.extend(b"stale")
        manager._st.in_speech = True

        await manager._on_input_abort(
            TurnInputAbortRequested(session_id="session", origin="text")
        )
        self.assertTrue(manager._text_turn_active)
        self.assertEqual(manager._buf, bytearray())
        self.assertFalse(manager._st.in_speech)

        await manager._on_audio_frame(
            EnhancedAudioFrameReceived(
                session_id="session",
                audio_data=b"new-audio",
            )
        )
        self.assertEqual(manager._buf, bytearray())

        await manager._on_text_vad_end(
            VADSpeechEnd(session_id="session", origin="text")
        )
        self.assertFalse(manager._text_turn_active)

    async def test_turn_detector_ignores_text_origin_events(self) -> None:
        """Do not ask a Turn Detector to approve an explicit Send action."""

        manager = TurnDetectorManager(
            event_bus=self.event_bus,
            session_id="session",
            models=Models(),
        )
        async_detect = AsyncMock()
        manager.turn_detector = cast(Any, type("Detector", (), {})())
        manager.turn_detector.async_detect = async_detect

        await manager._on_vad_speech_start(
            VADSpeechStart(session_id="session", origin="text")
        )
        await manager._on_asr_partial(
            ASRResultPartial(
                session_id="session",
                text="typed",
                display_text="typed",
                speech_pause=True,
                origin="text",
            )
        )
        await manager._on_vad_speech_end(
            VADSpeechEnd(session_id="session", origin="text")
        )

        async_detect.assert_not_awaited()

    async def test_text_turn_clears_stale_frontend_latency_markers(self) -> None:
        """Do not attribute a previous audio VAD timestamp to text input."""

        manager = LatencyManager(
            event_bus=self.event_bus,
            session_id="session",
        )
        manager._frontend_vad_start_ts = 1.0
        manager._backend_vad_start_ts = 2.0

        await manager._on_vad_start(
            VADSpeechStart(session_id="session", origin="text")
        )

        self.assertIsNone(manager._frontend_vad_start_ts)
        self.assertIsNone(manager._backend_vad_start_ts)


if __name__ == "__main__":
    unittest.main()
