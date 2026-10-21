"""Retain speech while remote VAD is deciding its boundary."""

import asyncio
from unittest.mock import Mock

from xtalk.models import ASR, VAD, Models
from xtalk.serving.event_bus import EventBus, EventDispatchMode
from xtalk.serving.events import VADSpeechStart, TurnASRStartRequested
from xtalk.serving.modules.asr_manager import AudioConsumer
from xtalk.serving.modules.turn_taking_manager import TurnTakingManager
from xtalk.serving.modules.vad_manager import VADManager


def test_delayed_vad_retains_speech_before_the_notification_padding():
    """A late speech-start notification must replay speech, not recent silence."""

    async def run():
        bus = EventBus()
        model = Mock(spec=ASR)
        consumer = AudioConsumer(
            bus,
            "session",
            Models({ASR: model, VAD: Mock(spec=VAD)}),
            {
                "realtime_turn_detection": {
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "vad_batch_frames": 32,
            },
        )
        speech = bytes([1, 2]) * 16000
        silence = bytes(32000)
        await consumer.accept_audio_frame(speech)
        await consumer.accept_audio_frame(silence)
        await consumer.start(turn_id=1, audio_lookback_ms=2000)
        assert await consumer._audio_queue.get() == speech + silence
        await consumer.shutdown()
        await bus.shutdown()

    asyncio.run(run())


def test_vad_onset_age_reaches_asr_through_the_existing_turn_manager():
    """Include inference delay and requested padding in the internal start event."""

    async def run():
        bus = EventBus()
        model = Mock(spec=VAD)
        models = Models({VAD: model})
        manager = VADManager(
            bus,
            "session",
            models,
            {
                "vad_min_speech_ms": 32,
                "realtime_turn_detection": {
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
            },
        )
        TurnTakingManager(bus, "session", models)
        started = asyncio.Future()

        async def capture(event):
            started.set_result(event)

        bus.subscribe(TurnASRStartRequested, capture)
        await manager._advance_state(
            True, frame_time=asyncio.get_running_loop().time() - 2
        )
        event = await asyncio.wait_for(started, 1)
        assert event.audio_lookback_ms >= 2300
        assert event.turn_id == 1
        await manager.shutdown()
        await bus.shutdown()

    asyncio.run(run())
