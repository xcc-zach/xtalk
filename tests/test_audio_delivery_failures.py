"""Regression coverage for checked playback delivery and transport feedback."""

from __future__ import annotations

import asyncio
import unittest

from starlette.websockets import WebSocketState

from xtalk.models import EndToEnd, Models
from xtalk.models.end_to_end.gpt_live import GPTLive
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import (
    AudioStreamControlRequested,
    EndToEndAudioChunkReady,
    EndToEndSignal,
    EndToEndTranscriptDelta,
)
from xtalk.serving.modules.end_to_end_manager import (
    EndToEndManager,
    create_stream_profile,
)
from xtalk.serving.modules.output_gateway import OutputGateway


class _BrokenSocket:
    """Fail both wire formats without opening a network connection."""

    client_state = WebSocketState.CONNECTED

    async def send_json(self, payload: dict) -> None:
        """Reject JSON delivery."""
        raise OSError("fixture JSON send failed")

    async def send_bytes(self, payload: bytes) -> None:
        """Reject PCM delivery."""
        raise OSError("fixture PCM send failed")


class AudioDeliveryFailureTests(unittest.IsolatedAsyncioTestCase):
    """Exercise actual gateways, managers and EventBus failure contracts."""

    def make_manager(self):
        """Create an unopened Live model and session-local orchestration."""
        bus = EventBus()
        model = GPTLive(
            api_key="fixture",
            delegation={
                "type": "responses",
                "responses": {"model": "fixture", "tools": [], "tool_choice": "none"},
            },
        )
        profile = create_stream_profile(model, {})
        config = {"_end_to_end_profile": profile}
        manager = EndToEndManager(bus, "call", Models({EndToEnd: model}), config)
        self.addAsyncCleanup(bus.shutdown)
        self.addAsyncCleanup(manager.shutdown)
        return bus, manager, config

    async def test_failed_custom_output_rolls_back_before_shutdown(self):
        """A caught subscriber exception immediately releases its reservation."""
        bus, manager, _ = self.make_manager()

        async def failing_output(event):
            raise OSError("fixture output failed")

        bus.subscribe(EndToEndAudioChunkReady, failing_output)
        await manager._budget.reset(manager.stream_id)
        manager._pending_samples = 2400
        manager._pending.put_nowait(bytes(4800))
        with self.assertRaisesRegex(RuntimeError, "audio delivery failed"):
            await asyncio.wait_for(manager._deliver(), 0.5)
        self.assertEqual(manager._budget.outstanding_ms, 0)

    async def test_live_json_and_pcm_failures_notify_once_without_recursion(self):
        """Failed error feedback also uses the bounded transport failure path."""
        bus, manager, config = self.make_manager()
        gateway = OutputGateway(bus, "call", _BrokenSocket(), config)
        failures = []
        bus.subscribe(AudioStreamControlRequested, failures.append)
        await manager._budget.reset(manager.stream_id)
        await manager._budget.reserve(100)
        await asyncio.wait_for(manager._signal("audio_stream_start"), 0.5)
        with self.assertRaises(OSError):
            await gateway._stream_audio(
                EndToEndAudioChunkReady(
                    session_id="call",
                    stream_id=manager.stream_id,
                    audio_data=bytes(4800),
                )
            )
        with self.assertRaises(OSError):
            await gateway._stream_transcript(
                EndToEndTranscriptDelta(
                    session_id="call",
                    stream_id=manager.stream_id,
                    delta="hello",
                )
            )
        self.assertTrue(manager._failed)
        self.assertTrue(manager._stop.is_set())
        self.assertEqual(manager._budget.outstanding_ms, 0)
        self.assertEqual([event.command for event in failures], ["_transport_failure"])

    async def test_failed_attachment_notifies_call_owner(self):
        """Attachment failure stops startup through the same Live feedback path."""
        bus, manager, config = self.make_manager()
        gateway = OutputGateway(bus, "call", _BrokenSocket(), config)
        await asyncio.wait_for(gateway._send_session_created(), 0.5)
        self.assertTrue(manager._failed)
        self.assertTrue(manager._stop.is_set())

    async def test_failed_custom_control_delivery_stops_without_recursion(self):
        """A replacement signal subscriber may fail without stranding startup."""
        bus, manager, _ = self.make_manager()

        async def failing_signal(event):
            raise OSError("fixture control failed")

        bus.subscribe(EndToEndSignal, failing_signal)
        await asyncio.wait_for(manager._signal("audio_stream_start"), 0.5)
        self.assertTrue(manager._failed)
        self.assertTrue(manager._stop.is_set())

    async def test_fast_playback_ack_remains_valid(self):
        """A successful subscriber can acknowledge before publication returns."""
        bus, manager, _ = self.make_manager()
        await manager._budget.reset(manager.stream_id)
        await manager._budget.reserve(100)

        async def immediate_ack(event):
            await manager._budget.acknowledge_samples(manager.stream_id, 2400, 24000)

        bus.subscribe(EndToEndAudioChunkReady, immediate_ack)
        self.assertTrue(
            await bus.publish(
                EndToEndAudioChunkReady(
                    session_id="call",
                    stream_id=manager.stream_id,
                    audio_data=bytes(4800),
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                require_success=True,
            )
        )
        self.assertEqual(manager._budget.outstanding_ms, 0)
