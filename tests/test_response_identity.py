"""Regression tests for response-scoped playback and message updates."""

from __future__ import annotations

import asyncio
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from xtalk.models import Models
from xtalk.models.agents.interfaces import ChatHistory
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import (
    TTSResponseClosed,
    TTSChunkReady,
    TTSFinished,
    TTSStarted,
    ToolCallOccurred,
    TurnTTSDeliveryStartRequested,
    TurnTTSStartRequested,
    TurnTTSStopRequested,
)
from xtalk.serving.modules.tts_response_coordinator import TTSResponseCoordinator
from xtalk.serving.modules.tts_manager import TTSManager
from xtalk.serving.modules.tts_playback_manager import TTSPlaybackManager


class ResponseIdentityTests(unittest.IsolatedAsyncioTestCase):
    """Verify independent response text and serialized client delivery."""

    def test_chat_history_updates_the_identified_response(self) -> None:
        """Update an older response without replacing a newer response."""

        history = ChatHistory("system")
        history.append_or_update_ai_message(
            "A 开始",
            final=False,
            response_id="response-a",
        )
        history.append_message(HumanMessage(content="打断 A"))
        history.append_or_update_ai_message(
            "B 工具结果",
            final=False,
            response_id="response-b",
        )
        history.append_or_update_ai_message(
            "A 实际播放部分",
            final=True,
            response_id="response-a",
        )
        history.append_or_update_ai_message(
            "B 工具结果完整",
            final=True,
            response_id="response-b",
        )

        assistant_messages = [
            message
            for message in history.messages
            if isinstance(message, AIMessage)
        ]
        self.assertEqual(
            [message.content for message in assistant_messages],
            ["A 实际播放部分", "B 工具结果完整"],
        )

    async def test_next_response_waits_for_prior_response_close(self) -> None:
        """Prepare B immediately but release it only after A settles."""

        event_bus = EventBus(enable_history=True)
        TTSResponseCoordinator(event_bus, "session")

        await event_bus.publish(
            TurnTTSStartRequested(
                session_id="session",
                response_id="response-a",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        await event_bus.publish(
            TurnTTSStartRequested(
                session_id="session",
                response_id="response-b",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )

        delivery_ids = [
            event.response_id
            for event in event_bus.get_history(
                event_type=TurnTTSDeliveryStartRequested.TYPE
            )
        ]
        self.assertEqual(delivery_ids, ["response-a"])
        stop_ids = [
            event.response_id
            for event in event_bus.get_history(
                event_type=TurnTTSStopRequested.TYPE
            )
        ]
        self.assertEqual(stop_ids, ["response-a"])

        await event_bus.publish(
            TTSResponseClosed(
                session_id="session",
                response_id="response-a",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        delivery_ids = [
            event.response_id
            for event in event_bus.get_history(
                event_type=TurnTTSDeliveryStartRequested.TYPE
            )
        ]
        self.assertEqual(delivery_ids, ["response-a", "response-b"])

    async def test_latest_prepared_response_wins(self) -> None:
        """Discard B when C arrives before A has closed."""

        event_bus = EventBus(enable_history=True)
        TTSResponseCoordinator(event_bus, "session")
        for response_id in ("response-a", "response-b", "response-c"):
            await event_bus.publish(
                TurnTTSStartRequested(
                    session_id="session",
                    response_id=response_id,
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

        await event_bus.publish(
            TTSResponseClosed(
                session_id="session",
                response_id="response-a",
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        delivery_ids = [
            event.response_id
            for event in event_bus.get_history(
                event_type=TurnTTSDeliveryStartRequested.TYPE
            )
        ]
        self.assertEqual(delivery_ids, ["response-a", "response-c"])

    async def test_direct_audio_uses_the_identified_delivery_stream(self) -> None:
        """Send direct audio only after a matching start event."""

        event_bus = EventBus(enable_history=True)
        tts_manager = TTSManager(event_bus, "session", Models())
        playback_manager = TTSPlaybackManager(event_bus, "session")
        TTSResponseCoordinator(event_bus, "session")
        self.addAsyncCleanup(tts_manager.shutdown)
        self.addAsyncCleanup(playback_manager.shutdown)

        await event_bus.publish(
            ToolCallOccurred(
                session_id="session",
                name="direct_audio",
                args={
                    "audio": b"\x00\x00" * 4_800,
                    "sample_rate": 48_000,
                },
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        for _ in range(100):
            if event_bus.get_history(event_type=TTSFinished.TYPE):
                break
            await asyncio.sleep(0)

        started = event_bus.get_history(event_type=TTSStarted.TYPE)
        chunks = event_bus.get_history(event_type=TTSChunkReady.TYPE)
        finished = event_bus.get_history(event_type=TTSFinished.TYPE)
        self.assertEqual(len(started), 1)
        self.assertTrue(chunks)
        self.assertEqual(len(finished), 1)
        response_id = started[0].response_id
        self.assertTrue(response_id)
        self.assertTrue(all(event.response_id == response_id for event in chunks))
        self.assertEqual(finished[0].response_id, response_id)


if __name__ == "__main__":
    unittest.main()
