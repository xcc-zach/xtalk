"""Exercise negotiated binary output using the actual input and output gateways."""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import FullAudioFrameReady, TTSChunkReady
from xtalk.serving.modules.input_gateway import InputGateway
from xtalk.serving.modules.output_gateway import OutputGateway


class Transport:
    """Capture mixed JSON and binary output in delivery order."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        """Capture a JSON message."""
        self.messages.append(message)

    async def send_bytes(self, data):
        """Capture a complete binary message."""
        self.messages.append(data)


def make_gateways():
    """Create gateways with isolated session configuration and one transport."""
    bus = EventBus()
    transport = Transport()
    output = OutputGateway(bus, "session", transport)
    gateway = InputGateway(bus, "session", transport)
    gateway._service = SimpleNamespace(output_gateway=output)
    return bus, transport, gateway, output


def decode(data):
    """Read an envelope independently of the production encoder."""
    size = int.from_bytes(data[:4], "big")
    assert 1 <= size <= 65536
    return json.loads(data[4:4 + size]), data[4 + size:]


def test_binary_and_base64_deliver_identical_response_and_mix_audio():
    """Negotiation affects encoding only, including multi-block mixed audio."""
    async def run():
        captures = []
        for encoding in ("base64", "binary_pcm"):
            bus, transport, gateway, output = make_gateways()
            try:
                await gateway._update_session({"type": "session.update", "session": {"xtalk": {"output_audio_encoding": encoding}}})
                assert transport.messages[0]["type"] == "session.updated"
                await output._send_tts_chunk_signal(TTSChunkReady(
                    session_id="session", response_id="r", sample_rate=24000,
                    audio_chunk=b"\x01\x00" * 240,
                ))
                await output._send_full_audio_frame_signal(FullAudioFrameReady(
                    session_id="session", audio_chunk=b"\x02\x00" * 100000,
                    channels=2,
                ))
                if encoding == "binary_pcm":
                    blocks = [decode(message) for message in transport.messages if isinstance(message, bytes)]
                    assert not any(isinstance(message, dict) and message["type"] in (
                        "response.output_audio.delta", "session.audio.mix.delta"
                    ) for message in transport.messages)
                    assert blocks[0][0]["response_id"] == "r"
                    assert blocks[0][0]["item_id"] == "item_r"
                    assert all(header["version"] == 1 and "delta" not in header for header, _ in blocks)
                else:
                    blocks = [(message, base64.b64decode(message["delta"]))
                              for message in transport.messages if "delta" in message]
                captures.append([pcm for _, pcm in blocks])
            finally:
                await bus.shutdown()
        assert captures[0] == captures[1]
    asyncio.run(run())


def test_encoding_changes_wait_for_generation_and_playback_settlement():
    """Reject changes until a completed response's last chunk is acknowledged."""
    async def run():
        bus, transport, gateway, output = make_gateways()
        try:
            await output._send_tts_chunk_signal(TTSChunkReady(
                session_id="session", response_id="r", audio_chunk=b"00", sample_rate=24000,
            ))
            patch = {"type": "session.update", "session": {"xtalk": {"output_audio_encoding": "binary_pcm"}}}
            with pytest.raises(ValueError):
                await gateway._update_session(patch)
            await output._finish_response("r")
            with pytest.raises(ValueError):
                await gateway._update_session(patch)
            assert output.session_snapshot()["xtalk"]["output_audio_encoding"] == "base64"
            await gateway._update_session({"type": "session.update", "session": {"xtalk": {"output_audio_encoding": "base64"}}})
            await gateway._handle_playback_ack({
                "type": "response.output_audio.playback.chunk_played", "response_id": "r",
            })
            await gateway._update_session(patch)
            assert transport.messages[-1]["session"]["xtalk"]["output_audio_encoding"] == "binary_pcm"
            before = output.session_snapshot()
            with pytest.raises(ValueError):
                await gateway._update_session({"type": "session.update", "session": {"xtalk": {"output_audio_encoding": "invalid"}}})
            assert before == output.session_snapshot()
            await gateway._update_session({"type": "session.update", "session": {"instructions": ""}})
            assert output.session_snapshot()["xtalk"]["output_audio_encoding"] == "binary_pcm"
            other_bus, _, _, other = make_gateways()
            assert other.session_snapshot()["xtalk"]["output_audio_encoding"] == "base64"
            await other_bus.shutdown()
        finally:
            await bus.shutdown()
    asyncio.run(run())


def test_binary_audio_waits_until_session_confirmation_is_queued():
    """A concurrent mixed block cannot overtake the encoding acknowledgement."""
    async def run():
        bus, transport, gateway, output = make_gateways()
        confirming = asyncio.Event()
        release = asyncio.Event()
        original_send = transport.send_json

        async def hold_confirmation(message):
            if message["type"] == "session.updated":
                confirming.set()
                await release.wait()
            await original_send(message)

        transport.send_json = hold_confirmation
        update = asyncio.create_task(gateway._update_session({
            "type": "session.update", "session": {"xtalk": {"output_audio_encoding": "binary_pcm"}},
        }))
        await confirming.wait()
        audio = asyncio.create_task(output._send_full_audio_frame_signal(
            FullAudioFrameReady(session_id="session", audio_chunk=b"0000")
        ))
        await asyncio.sleep(0)
        assert transport.messages == []
        release.set()
        await asyncio.gather(update, audio)
        assert transport.messages[0]["type"] == "session.updated"
        assert isinstance(transport.messages[1], bytes)
        await bus.shutdown()
    asyncio.run(run())
