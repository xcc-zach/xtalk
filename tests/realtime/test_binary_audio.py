"""Verify binary PCM and Base64 share input state and validation."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock

import numpy as np
import pytest

from xtalk.serving.event_bus import EventBus
from xtalk.serving.modules.input_gateway import InputGateway


async def _send(gateway, pcm, binary):
    """Deliver PCM through the selected ASGI message representation."""
    if binary:
        await gateway._process_message(
            {"type": "websocket.receive", "text": None, "bytes": pcm}
        )
    else:
        await gateway._process_message(
            {
                "type": "websocket.receive",
                "bytes": None,
                "text": json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode(),
                    }
                ),
            }
        )


def _gateway(bus, manual=False):
    """Create an input gateway with captured output and audio dispatch."""
    transport = AsyncMock()
    gateway = InputGateway(
        bus,
        "session",
        transport,
        {"realtime_turn_detection": None if manual else True},
    )
    gateway._handlers = {
        "input_audio_buffer.append": InputGateway._handle_audio_append,
        "input_audio_buffer.commit": InputGateway._handle_audio_commit,
        "input_audio_buffer.clear": InputGateway._handle_audio_clear,
    }
    gateway._handle_audio_message = AsyncMock()
    return gateway


@pytest.mark.parametrize("manual", [False, True])
def test_mixed_input_matches_base64_and_retains_turn_semantics(manual):
    """Both encodings produce identical resampled audio and commit/clear state."""

    async def run():
        bus = EventBus()
        mixed, reference = _gateway(bus, manual), _gateway(bus, manual)
        pcm = np.arange(4800, dtype="<i2").tobytes()
        for binary in (True, False, True):
            await _send(mixed, pcm, binary)
            await _send(reference, pcm, False)
        assert mixed._buffered_bytes == reference._buffered_bytes == len(pcm) * 3
        assert mixed._turn_input_bytes == reference._turn_input_bytes
        assert mixed._handle_audio_message.call_args_list == (
            reference._handle_audio_message.call_args_list
        )
        assert bool(mixed._handle_audio_message.call_count) is not manual
        mixed.websocket.send_json.assert_not_called()
        if manual:
            assert mixed._turn_input_bytes
            received = []

            async def capture(event):
                received.append(event)

            from xtalk.serving.events import AudioFrameReceived

            bus.subscribe(AudioFrameReceived, capture)
            await mixed._handle_text_message('{"type":"input_audio_buffer.commit"}')
            assert received and received[0].audio_data == bytes(
                reference._turn_input_bytes
            )
            assert mixed._buffered_bytes == 0
            assert not mixed._turn_input_bytes
            assert not mixed._manual_speech_started
        await mixed._handle_text_message('{"type":"input_audio_buffer.clear"}')
        assert mixed._buffered_bytes == 0
        assert not mixed._turn_input_bytes
        assert (
            mixed.websocket.send_json.call_args.args[0]["type"]
            == "input_audio_buffer.cleared"
        )
        await bus.shutdown()

    asyncio.run(run())


def test_invalid_binary_preserves_state_and_connection_recovers():
    """Rejected PCM never reaches resampling and a later message still works."""

    async def run():
        bus = EventBus()
        gateway = _gateway(bus, True)
        resampler = gateway._resampler
        for pcm in (b"\x00", bytes(1024 * 1024 + 2)):
            await _send(gateway, pcm, True)
            error = gateway.websocket.send_json.call_args.args[0]["error"]
            assert error["code"] == "invalid_value"
            assert error["event_id"] is None
            assert gateway._buffered_bytes == 0
            assert not gateway._turn_input_bytes
            assert gateway._resampler is resampler
        await _send(gateway, bytes(9600), True)
        assert gateway._buffered_bytes == 9600
        assert gateway._turn_input_bytes
        await bus.shutdown()

    asyncio.run(run())


def test_message_and_shared_buffer_boundaries():
    """Allow exactly one MiB and sixty seconds; reject overflow atomically."""

    async def run():
        bus = EventBus()
        gateway = _gateway(bus, True)
        await _send(gateway, bytes(1024 * 1024), True)
        gateway.websocket.send_json.assert_not_called()
        assert gateway._buffered_bytes == 1024 * 1024
        limit = 24000 * 2 * 60
        # Fill the remaining shared budget with both input encodings.
        await _send(gateway, bytes(1024 * 1024), True)
        await _send(gateway, bytes(limit - gateway._buffered_bytes), False)
        assert gateway._buffered_bytes == limit
        buffered = bytes(gateway._turn_input_bytes)
        for binary in (True, False):
            await _send(gateway, b"\x00\x00", binary)
            assert (
                gateway.websocket.send_json.call_args.args[0]["error"]["code"]
                == "invalid_value"
            )
            assert gateway._buffered_bytes == limit
            assert bytes(gateway._turn_input_bytes) == buffered
        await bus.shutdown()

    asyncio.run(run())


def test_receive_loop_handles_nullable_asgi_fields_and_recovers():
    """Keep receiving after invalid binary audio and accept both representations."""

    async def run():
        bus = EventBus()
        gateway = _gateway(bus, True)
        pcm = bytes(9600)
        gateway.websocket.receive.side_effect = [
            {"type": "websocket.receive", "text": None, "bytes": b"\x00"},
            {"type": "websocket.receive", "text": None, "bytes": pcm},
            {
                "type": "websocket.receive",
                "text": json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm).decode(),
                    }
                ),
                "bytes": None,
            },
            {"type": "websocket.disconnect"},
        ]
        await gateway.handle_message_loop()
        assert gateway._buffered_bytes == len(pcm) * 2
        assert gateway._turn_input_bytes
        gateway.websocket.send_json.assert_awaited_once()
        assert (
            gateway.websocket.send_json.call_args.args[0]["error"]["code"]
            == "invalid_value"
        )
        await bus.shutdown()

    asyncio.run(run())
