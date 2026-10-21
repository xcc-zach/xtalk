"""Verify batch scheduling, speech boundaries, and session lifecycle."""

import asyncio
import threading
from unittest.mock import AsyncMock

import pytest

from xtalk.models import Models
from xtalk.models.vad.interfaces import VAD
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import (
    EnhancedAudioFrameReceived,
    TurnInputAbortRequested,
    VADSpeechEnd,
    VADSpeechStart,
)
from xtalk.serving.modules.vad_manager import VADManager


class RecordingVAD(VAD):
    """Small stateful model supporting arbitrary frame lengths for regression tests."""

    def __init__(self):
        """Record sequential single-frame calls."""
        self.frames = []

    def set_threshold(self, *, threshold: float) -> None:
        """Reject threshold updates for this deterministic boolean test model."""
        raise NotImplementedError("RecordingVAD does not support threshold updates")

    def is_speech(self, frame):
        """Record a frame and classify its first byte."""
        self.frames.append(frame)
        return bool(frame[0])

    def clone(self):
        """Return independent stream state."""
        return RecordingVAD()


def make_manager(config=None):
    """Create a real manager with deterministic mocked batch inference."""
    vad = RecordingVAD()
    vad.async_detect_frames = AsyncMock(
        side_effect=lambda frames: [frame[0] != 0 for frame in frames]
    )
    vad.reset = lambda: None
    bus = EventBus(enable_history=True)
    manager = VADManager(
        bus, "test", Models({VAD: vad}), {"vad_batch_frames": 4, **(config or {})}
    )
    return manager, vad, bus


def test_batches_preserve_fragmented_audio_and_emit_boundaries():
    """Register each event once and preserve frames across input fragments."""

    async def run():
        manager, vad, bus = make_manager(
            {"vad_min_speech_ms": 64, "vad_redemption_ms": 64}
        )
        ended = asyncio.Event()

        async def on_end(event):
            ended.set()

        bus.subscribe(VADSpeechEnd, on_end)
        audio = bytes([1]) * 8192 + bytes(8192)
        try:
            for fragment in [audio[:101], audio[101:5000], audio[5000:]]:
                await bus.publish(
                    EnhancedAudioFrameReceived(session_id="test", audio_data=fragment),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
            await asyncio.wait_for(ended.wait(), 1)
            while len(vad.async_detect_frames.call_args_list) < 4:
                await asyncio.sleep(0)
            assert [
                len(call.args[0]) for call in vad.async_detect_frames.call_args_list
            ] == [4] * 4
            assert (
                b"".join(
                    b"".join(call.args[0])
                    for call in vad.async_detect_frames.call_args_list
                )
                == audio
            )
            history = bus.get_history()
            assert sum(isinstance(event, VADSpeechStart) for event in history) == 1
            assert sum(isinstance(event, VADSpeechEnd) for event in history) == 1
        finally:
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


def test_partial_batch_flushes_after_deadline():
    """Detect final full frames even when no fourth frame arrives."""

    async def run():
        manager, vad, bus = make_manager({"vad_batch_wait_ms": 10})
        try:
            await manager._on_audio_frame(
                EnhancedAudioFrameReceived(session_id="test", audio_data=bytes(2048))
            )

            async def completed():
                while not vad.async_detect_frames.await_count:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(completed(), 1)
            vad.async_detect_frames.assert_awaited_once_with([bytes(1024)] * 2)
        finally:
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


def test_ingestion_does_not_wait_for_remote_and_abort_drops_old_results():
    """Keep ingesting while inference is pending and reject stale text-turn results."""

    async def run():
        manager, vad, bus = make_manager({"vad_min_speech_ms": 32})
        started, release = asyncio.Event(), asyncio.Event()

        async def blocked(audio):
            started.set()
            await release.wait()
            return [True] * 4

        vad.async_detect_frames = AsyncMock(side_effect=blocked)
        try:
            event = EnhancedAudioFrameReceived(
                session_id="test", audio_data=bytes(4096)
            )
            await manager._on_audio_frame(event)
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.wait_for(manager._on_audio_frame(event), 0.1)
            abort = asyncio.create_task(
                manager._on_input_abort(
                    TurnInputAbortRequested(session_id="test", origin="text")
                )
            )
            await asyncio.sleep(0)
            release.set()
            await asyncio.wait_for(abort, 1)
            assert not any(
                isinstance(event, VADSpeechStart) for event in bus.get_history()
            )
            assert not manager._buf
            assert manager._text_turn_active
        finally:
            release.set()
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


def test_queue_overflow_is_explicit_and_shutdown_waits_for_inference():
    """Bound pending audio and finish the active request before resetting the client."""

    async def run():
        manager, vad, bus = make_manager({"vad_max_queue_frames": 4})
        started, release = asyncio.Event(), asyncio.Event()
        reset_calls = []
        vad.reset = lambda: reset_calls.append(True)

        async def blocked(audio):
            started.set()
            await release.wait()
            return [False] * 4

        vad.async_detect_frames = AsyncMock(side_effect=blocked)
        try:
            event = EnhancedAudioFrameReceived(
                session_id="test", audio_data=bytes(4096)
            )
            await manager._on_audio_frame(event)
            await asyncio.wait_for(started.wait(), 1)
            await manager._on_audio_frame(event)
            try:
                await manager._on_audio_frame(event)
                raise AssertionError("Expected queue overflow")
            except BufferError:
                pass
            shutdown = asyncio.create_task(manager.shutdown())
            await asyncio.sleep(0)
            assert not shutdown.done() and not reset_calls
            release.set()
            await asyncio.wait_for(shutdown, 1)
            assert reset_calls == [True]
        finally:
            release.set()
            await bus.shutdown()

    asyncio.run(run())


def test_sequential_sync_and_async_fallbacks():
    """Retain ordering and dispatch through existing async single-frame overrides."""
    frames = [b"\1\0", b"\0\0", b"\2\0"]
    model = RecordingVAD()
    assert model.detect_frames([]) == [] and not model.frames
    assert model.detect_frames(frames) == [True, False, True]
    assert model.frames == frames
    model.frames.clear()
    assert asyncio.run(model.async_detect_frames([])) == []
    assert asyncio.run(model.async_detect_frames(frames)) == [True, False, True]
    assert model.frames == frames
    model.async_is_speech = AsyncMock(side_effect=[False, True, False])
    assert asyncio.run(model.async_detect_frames(frames)) == [False, True, False]
    assert [call.args[0] for call in model.async_is_speech.call_args_list] == frames


@pytest.mark.parametrize(
    "config",
    [
        {"vad_batch_frames": True},
        {"vad_batch_frames": 0},
        {"vad_batch_frames": 33},
        {"vad_batch_frames": 4.0},
        {"vad_batch_wait_ms": float("nan")},
        {"vad_batch_wait_ms": float("inf")},
        {"vad_batch_wait_ms": False},
        {"vad_batch_wait_ms": 0},
        {"vad_max_queue_frames": True},
        {"vad_max_queue_frames": 3},
        {"vad_max_queue_frames": 4.0},
    ],
)
def test_invalid_batch_configuration(config):
    """Reject invalid values without starting tasks."""
    with pytest.raises(ValueError):
        make_manager(config)


def test_default_mode_preserves_handler_completion_and_error_smoothing():
    """Return only after every frame has passed through the original state machine."""

    async def run():
        model = RecordingVAD()
        model.async_is_speech = AsyncMock(side_effect=[True, OSError("test"), False])
        bus = EventBus(enable_history=True)
        manager = VADManager(
            bus,
            "default",
            Models({VAD: model}),
            {"vad_frame_samples": 16, "vad_min_speech_ms": 1, "vad_redemption_ms": 1},
        )
        try:
            await manager._on_audio_frame(
                EnhancedAudioFrameReceived(session_id="default", audio_data=bytes(96))
            )
            assert model.async_is_speech.call_count == 3
            assert manager._worker is None
            assert [type(e) for e in bus.get_history()] == [
                VADSpeechStart,
                VADSpeechEnd,
            ]
        finally:
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


@pytest.mark.parametrize(
    "result", [[True], [True, 1], [True, False, True], None, OSError("batch")]
)
def test_invalid_batch_or_model_failure_is_silence(result):
    """Reject the whole invalid batch before applying any decisions."""

    async def run():
        manager, model, bus = make_manager({"vad_frame_samples": 2})
        if isinstance(result, Exception):
            model.async_detect_frames = AsyncMock(side_effect=result)
        else:
            model.async_detect_frames = AsyncMock(return_value=result)
        try:
            assert await manager._detect_batch([bytes(4)] * 2) == [False, False]
        finally:
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


def test_expired_tail_does_not_wait_again_after_inflight_batch():
    """Flush a queued tail immediately when its original deadline has passed."""

    async def run():
        manager, model, bus = make_manager(
            {"vad_batch_wait_ms": 500, "vad_frame_samples": 2}
        )
        started, release, tail = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def infer(frames):
            if len(frames) == 4:
                started.set()
                await release.wait()
            else:
                tail.set()
            return [False] * len(frames)

        model.async_detect_frames = AsyncMock(side_effect=infer)
        try:
            await manager._on_audio_frame(
                EnhancedAudioFrameReceived(session_id="test", audio_data=bytes(16))
            )
            await asyncio.wait_for(started.wait(), 1)
            await manager._on_audio_frame(
                EnhancedAudioFrameReceived(session_id="test", audio_data=bytes(4))
            )
            manager._frame_times[0] = asyncio.get_running_loop().time() - 1
            release.set()
            await asyncio.wait_for(tail.wait(), 0.2)
            assert [
                len(c.args[0]) for c in model.async_detect_frames.call_args_list
            ] == [4, 1]
        finally:
            release.set()
            await manager.shutdown()
            await bus.shutdown()

    asyncio.run(run())


def test_cancelled_worker_finishes_sync_inference_before_reset():
    """Protect executor work so reset never runs concurrently with inference."""

    async def run():
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        model = RecordingVAD()

        def infer(frame):
            started.set()
            assert release.wait(2)
            finished.set()
            return True

        model.is_speech = infer
        reset_calls = []

        def reset():
            assert finished.is_set()
            reset_calls.append(True)

        model.reset = reset
        bus = EventBus(enable_history=True)
        manager = VADManager(
            bus,
            "cancel",
            Models({VAD: model}),
            {"vad_batch_frames": 2, "vad_frame_samples": 2},
        )
        try:
            await manager._on_audio_frame(
                EnhancedAudioFrameReceived(session_id="cancel", audio_data=bytes(8))
            )
            while not started.is_set():
                await asyncio.sleep(0.001)
            manager._worker.cancel()
            shutdown = asyncio.create_task(manager.shutdown())
            await asyncio.sleep(0.01)
            assert not shutdown.done() and not reset_calls
            release.set()
            await asyncio.wait_for(shutdown, 1)
            assert reset_calls == [True] and manager._worker.done()
            assert not bus.get_history()
        finally:
            release.set()
            await bus.shutdown()

    asyncio.run(run())


def test_reset_failure_releases_manager_resources():
    """Log reset errors and still release locks and stop the worker."""

    async def run():
        manager, model, bus = make_manager()

        def reset():
            raise OSError("reset")

        model.reset = reset
        await manager.shutdown()
        assert not manager._request_lock.locked() and not manager._state_lock.locked()
        await bus.shutdown()

    asyncio.run(run())
