"""Contract tests for generation and playback separation."""

import asyncio
from xtalk import Models, Service
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    LLMAgentResponseUpdate,
    LLMAgentResponseFinish,
    ResponseUpdate,
    TTSFinished,
)
from xtalk.serving.modules.output_gateway import OutputGateway


class Transport:
    """Capture JSON events without a framework or audio device."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        self.messages.append(message)


def test_generated_and_played_text_use_distinct_events():
    async def run():
        transport = Transport()
        bus = EventBus()
        gateway = OutputGateway(
            bus,
            "session",
            transport,
            _event_overrides=Service(models=Models())._event_overrides[OutputGateway],
        )
        await gateway._on_generated_text(
            LLMAgentResponseUpdate(
                session_id="session", response_id="r1", text="hello", delta="hello"
            )
        )
        await gateway._send_update_resp_signal(
            ResponseUpdate(session_id="session", response_id="r1", text="he")
        )
        assert transport.messages[0]["type"] == "response.created"
        assert (
            transport.messages[-1]["type"] == "response.output_audio.playback.updated"
        )
        assert transport.messages[-1]["text"] == "he"
        await gateway._send_tts_finished_signal(
            TTSFinished(session_id="session", response_id="r1")
        )
        await gateway._send_tts_finished_signal(
            TTSFinished(session_id="session", response_id="r1")
        )
        assert not any(e["type"] == "response.done" for e in transport.messages)
        await gateway._on_generated_done(
            LLMAgentResponseFinish(session_id="session", response_id="r1", text="hello")
        )
        assert sum(e["type"] == "response.done" for e in transport.messages) == 1
        await bus.shutdown()

    asyncio.run(run())


def test_builtin_output_mapping_is_replaced_in_prototype_and_live_session():
    """A replacement emits once and does not retain an earlier subscription."""
    from xtalk import Models, Service
    from xtalk.serving.event_bus import EventDispatchMode
    from xtalk.serving.events import CaptionUpdated

    async def run():
        prototype = Service(models=Models())
        prototype.register_output_handler(
            source_event=CaptionUpdated,
            type="conversation.caption.updated",
            handler=lambda event: {"text": "replacement"},
        )
        transport = Transport()
        service = prototype.clone(transport)
        await service.event_bus.publish(
            CaptionUpdated(session_id=service.session_id, text="original"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        assert len(transport.messages) == 1
        assert transport.messages[0]["text"] == "replacement"
        service.register_output_handler(
            source_event=CaptionUpdated,
            type="conversation.caption.updated",
            handler=lambda event: {"text": "live"},
        )
        await service.event_bus.publish(
            CaptionUpdated(session_id=service.session_id, text="original"),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
        )
        assert len(transport.messages) == 2
        assert transport.messages[-1]["text"] == "live"
        await service.stop()
        await prototype.stop()

    asyncio.run(run())


def test_transcript_emits_once_for_successful_synthesis_delivery():
    """Both TTS modes produce synthesis markers and one final delivery marker."""
    from xtalk.serving.event_bus import EventDispatchMode
    from xtalk.serving.events import TTSTextSynthesized, TTSTextDeliveryFinished

    async def run():
        transport = Transport()
        bus = EventBus()
        OutputGateway(
            bus,
            "session",
            transport,
            _event_overrides=Service(models=Models())._event_overrides[OutputGateway],
        )
        for event in [
            TTSTextSynthesized(
                session_id="session",
                response_id="r",
                text="hello",
                audio_chunk=b"00",
                sample_rate=24000,
            ),
            TTSTextDeliveryFinished(
                session_id="session", response_id="r", text="hello", succeeded=True
            ),
            TTSFinished(session_id="session", response_id="r"),
        ]:
            await bus.publish(event, mode=EventDispatchMode.WAIT_UNTIL_COMPLETE)
        deltas = [
            event["delta"]
            for event in transport.messages
            if event["type"] == "response.output_audio_transcript.delta"
        ]
        assert deltas == ["hello"]
        done = next(
            event
            for event in transport.messages
            if event["type"] == "response.output_audio_transcript.done"
        )
        assert done["transcript"] == "hello"
        await bus.shutdown()

    asyncio.run(run())


def test_provider_asr_failure_is_correlated_and_does_not_kill_the_consumer():
    """Provider errors produce one failed transcription for the original input."""
    from unittest.mock import AsyncMock, Mock
    from xtalk.models import ASR, Models
    from xtalk.serving.events import ASRFailed
    from xtalk.serving.modules.asr_manager import AudioConsumer

    async def run():
        model = Mock(spec=ASR)
        model.async_recognize_stream = AsyncMock(
            side_effect=RuntimeError("ASR unavailable")
        )
        bus = EventBus(enable_history=True)
        consumer = AudioConsumer(bus, "session", Models({ASR: model}))
        consumer._turn_id = 3
        consumer._item_id = "custom_audio_item"
        transport = Transport()
        OutputGateway(
            bus,
            "session",
            transport,
            _event_overrides=Service(models=Models())._event_overrides[OutputGateway],
        )
        await consumer._recognize_and_publish(b"00", is_final_chunk=True)
        await consumer._recognize_and_publish(b"00", is_final_chunk=True)
        failures = [
            event for event in bus.get_history() if isinstance(event, ASRFailed)
        ]
        assert len(failures) == 1
        assert failures[0].item_id == "custom_audio_item"

        async def delivered():
            while not transport.messages:
                await asyncio.sleep(0)

        await asyncio.wait_for(delivered(), 1)
        assert (
            transport.messages[-1]["type"]
            == "conversation.item.input_audio_transcription.failed"
        )
        assert not consumer._audio_consumer_task.done()
        await consumer.shutdown()
        await bus.shutdown()

    asyncio.run(run())


def test_input_commit_is_sent_before_the_asr_provider_finishes():
    """Slow ASR must not delay the commit receipt or duplicate it on final text."""
    from xtalk.serving.event_bus import EventDispatchMode
    from xtalk.serving.events import ASRResultFinal, TurnASREndRequested

    async def run():
        transport = Transport()
        service = Service(models=Models()).clone(transport, session_id="session")
        provider_started = asyncio.Event()
        provider_release = asyncio.Event()

        async def provider(event):
            provider_started.set()
            await provider_release.wait()
            await service.event_bus.publish(
                ASRResultFinal(
                    session_id="session",
                    turn_id=1,
                    item_id="item_input_session_1",
                    text="hello",
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )

        service.event_bus.subscribe(TurnASREndRequested, provider)
        task = asyncio.create_task(
            service.event_bus.publish(
                TurnASREndRequested(
                    session_id="session", turn_id=1, item_id="item_input_session_1"
                ),
                mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            )
        )
        await asyncio.wait_for(provider_started.wait(), 1)
        assert [message["type"] for message in transport.messages] == [
            "input_audio_buffer.committed",
            "conversation.item.added",
        ]
        provider_release.set()
        await task
        assert (
            sum(
                message["type"] == "input_audio_buffer.committed"
                for message in transport.messages
            )
            == 1
        )
        assert (
            transport.messages[-1]["type"]
            == "conversation.item.input_audio_transcription.completed"
        )
        await service.stop()

    asyncio.run(run())
