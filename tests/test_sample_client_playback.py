"""Regress device churn, cross-block gaps, early ACKs and stopped audio."""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from examples.sample_client import _AudioPlayback


@pytest.fixture
def device(monkeypatch):
    """Provide a controllable DAC clock without using a physical soundcard."""

    devices = []

    class Device:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]
            self.time = 10.0
            self.starts = self.aborts = self.closes = 0
            devices.append(self)

        def start(self):
            self.starts += 1

        def abort(self):
            self.aborts += 1

        def close(self):
            self.closes += 1

        def render(self, at, frames=480):
            output = bytearray(frames * 2)
            self.callback(output, frames, SimpleNamespace(outputBufferDacTime=at), None)
            return bytes(output)

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(RawOutputStream=Device)
    )
    return devices


def test_persistent_stream_fills_across_blocks_and_acks_after_dac(device):
    """Original event boundaries must not cause silence or reopen the device."""

    async def run():
        player = _AudioPlayback(no_speaker=False)
        stream = device[0]
        first = player.enqueue(response_id="r", pcm=b"a" * 400)
        second = player.enqueue(response_id="r", pcm=b"b" * 560)
        third = player.enqueue(response_id="r", pcm=b"c" * 960)
        assert stream.render(10.1) == b"a" * 400 + b"b" * 560
        assert stream.render(10.12) == b"c" * 960
        await asyncio.sleep(0)
        assert not first.done.done()
        assert not second.done.done()
        assert not third.done.done()
        stream.time = 10.3
        # Allow timers to observe the controlled clock without sleeping for audio.
        for block in [first, second, third]:
            player._schedule_completion(block, 10.2, block.generation)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        for block in [first, second, third]:
            assert block.done.done()
            player.acknowledge(block)
        assert not player.has_pending("r")
        assert len(device) == 1
        assert stream.starts == 1
        assert stream.aborts == stream.closes == 0
        await player.close()
        assert stream.closes == 1

    asyncio.run(run())


def test_stop_reports_only_unconfirmed_dac_audio_and_retains_next_response(device):
    """Flush old audio, preserve the next response and reject stale completions."""

    async def run():
        player = _AudioPlayback(no_speaker=False)
        stream = device[0]
        old = player.enqueue(response_id="old", pcm=b"a" * 1920)
        next_block = player.enqueue(response_id="next", pcm=b"b" * 960)
        stream.render(10)
        stream.render(10.02)
        stream.render(10.04)
        stream.time = 10.015
        elapsed = await player.stop_response("old")
        assert elapsed == pytest.approx(15)
        assert old.done.cancelled()
        assert player.enqueue(response_id="old", pcm=b"late") is None
        assert next_block.offset == 0
        assert stream.render(10.02) == b"b" * 960
        assert not next_block.done.done()
        assert stream.aborts == 1
        assert stream.starts == 2
        # Repeated old stop cannot flush current playback.
        await player.stop_response("old")
        assert stream.aborts == 1
        await player.close()

    asyncio.run(run())


def test_pause_keeps_audio_queued_and_excludes_silent_time_from_stop(device):
    """Pause must not consume PCM or count paused time as heard audio."""

    async def run():
        player = _AudioPlayback(no_speaker=False)
        block = player.enqueue(response_id="r", pcm=b"a" * 1920)
        stream = device[0]
        assert stream.render(10) == b"a" * 960
        player.pause(True)
        assert stream.render(10.02) == bytes(960)
        assert block.offset == 960
        player.pause(False)
        assert stream.render(11) == b"a" * 960
        stream.time = 11.01
        assert await player.stop_response("r") == pytest.approx(30)
        await player.close()

    asyncio.run(run())


def test_soundcard_free_playback_uses_the_same_block_completion_accounting():
    """The synthetic sink must finish original blocks and release pending state."""

    async def run():
        player = _AudioPlayback(no_speaker=True)
        first = player.enqueue(response_id="r", pcm=bytes(960))
        second = player.enqueue(response_id="r", pcm=bytes(960))
        await asyncio.wait_for(asyncio.gather(first.done, second.done), 1)
        player.acknowledge(first)
        player.acknowledge(second)
        assert not player.has_pending("r")
        await player.close()

    asyncio.run(run())
