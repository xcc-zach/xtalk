"""Verify the JSON boundary and manager request outcome contract."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    AgentInstructionsUpdateRequested,
    LLMAgentResponseFinish,
    ResponseCancelRequested,
    TTSConfigurationUpdateRequested,
    TurnLLMAgentStopRequested,
    VADThresholdUpdateRequested,
)
from xtalk.serving.modules.input_gateway import InputGateway
from xtalk.serving.modules.output_gateway import OutputGateway


class Transport:
    """Capture protocol events in send order."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        """Record a server event."""
        self.messages.append(message)


def make_gateway():
    """Create real gateways sharing a request bus and session configuration."""
    bus = EventBus()
    transport = Transport()
    gateway = InputGateway(bus, "session", transport)
    output = OutputGateway(bus, "session", transport)
    gateway._service = SimpleNamespace(output_gateway=output)
    gateway._handlers = {
        "session.update": InputGateway._update_session,
        "response.cancel": InputGateway._handle_response_cancel,
    }
    return bus, transport, gateway, output


@pytest.mark.parametrize(
    "threshold", [-0.1, 1.1, True, "0.5", float("nan"), float("inf")]
)
def test_event_threshold_constraints_are_enforced_on_construction(threshold):
    """Internal requests share the boundary's authoritative probability type."""
    with pytest.raises(ValidationError):
        VADThresholdUpdateRequested(session_id="session", threshold=threshold)


@pytest.mark.parametrize("threshold", [0, 1])
def test_threshold_boundaries_are_valid(threshold):
    assert (
        VADThresholdUpdateRequested(session_id="session", threshold=threshold).threshold
        == threshold
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"audio": []},
        {"audio": {"output": {"emotion": {"name": "", "vector": []}}}},
        {"audio": {"input": None}},
        {"audio": {"input": {"turn_detection": {"threshold": None}}}},
        {"audio": {"output": {"speed": True}}},
        {"audio": {"output": {"speed": "1"}}},
        {"audio": {"output": {"speed": 1.6}}},
        {"audio": {"input": {"turn_detection": {"silence_duration_ms": 1.5}}}},
        {"audio": {"output": {"unknown": 1}}},
        {"model": "other"},
    ],
)
def test_invalid_shapes_have_no_side_effects_and_connection_remains_usable(patch):
    async def run():
        bus, transport, gateway, output = make_gateway()
        before = output.session_snapshot()
        calls = []
        bus.subscribe(TTSConfigurationUpdateRequested, calls.append)
        bus.subscribe(VADThresholdUpdateRequested, calls.append)
        await gateway._handle_text_message(
            json.dumps(
                {"type": "session.update", "event_id": "client1", "session": patch}
            )
        )
        error = transport.messages[-1]
        assert error["type"] == "error"
        assert error["error"]["type"] == "invalid_request_error"
        assert error["error"]["event_id"] == "client1"
        assert error["error"]["param"].startswith("session.")
        assert output.session_snapshot() == before
        assert calls == []
        await gateway._handle_text_message('{"type":"session.update","session":{}}')
        assert transport.messages[-1]["type"] == "session.updated"
        await bus.shutdown()

    asyncio.run(run())


def test_combined_update_prechecks_all_owners_and_records_partial_application():
    async def run():
        bus, transport, gateway, output = make_gateway()
        calls = []

        def vad(event):
            calls.append(("vad", event.check_only))

        def tts(event):
            calls.append(("tts", event.check_only))
            if not event.check_only:
                raise RuntimeError("private backend detail")

        bus.subscribe(VADThresholdUpdateRequested, vad)
        bus.subscribe(TTSConfigurationUpdateRequested, tts)
        await gateway._handle_text_message(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "audio": {
                            "input": {"turn_detection": {"threshold": 0.7}},
                            "output": {"speed": 1.2},
                        }
                    },
                }
            )
        )
        assert calls == [("vad", True), ("tts", True), ("vad", False), ("tts", False)]
        assert (
            output.session_snapshot()["audio"]["input"]["turn_detection"]["threshold"]
            == 0.7
        )
        assert output.session_snapshot()["audio"]["output"]["speed"] == 1
        assert transport.messages[-1]["error"]["type"] == "server_error"
        assert (
            "private backend detail" not in transport.messages[-1]["error"]["message"]
        )
        await bus.shutdown()

    asyncio.run(run())


def test_precheck_rejection_prevents_all_application():
    async def run():
        bus, transport, gateway, output = make_gateway()
        calls = []
        bus.subscribe(
            VADThresholdUpdateRequested, lambda event: calls.append(event.check_only)
        )

        def reject(event):
            raise ValueError("Configured agent does not support instruction updates")

        bus.subscribe(AgentInstructionsUpdateRequested, reject)
        before = output.session_snapshot()
        await gateway._handle_text_message(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "new",
                        "audio": {"input": {"turn_detection": {"threshold": 0.8}}},
                    },
                }
            )
        )
        assert calls == [True]
        assert output.session_snapshot() == before
        assert transport.messages[-1]["type"] == "error"
        await bus.shutdown()

    asyncio.run(run())


def test_unchanged_settings_do_not_dispatch_and_null_disables_detection():
    async def run():
        bus, transport, gateway, output = make_gateway()
        calls = []
        bus.subscribe(TTSConfigurationUpdateRequested, calls.append)
        bus.subscribe(VADThresholdUpdateRequested, calls.append)
        await gateway._handle_text_message(
            '{"type":"session.update","session":{"audio":{"output":{"speed":1},"input":{"turn_detection":{"threshold":0.5}}}}}'
        )
        assert calls == []
        await gateway._handle_text_message(
            '{"type":"session.update","session":{"audio":{"input":{"turn_detection":null}}}}'
        )
        assert output.session_snapshot()["audio"]["input"]["turn_detection"] is None
        assert transport.messages[-1]["type"] == "session.updated"
        await bus.shutdown()

    asyncio.run(run())


def test_cancellation_wins_racing_completion_and_emits_one_terminal_event():
    async def run():
        bus, transport, gateway, output = make_gateway()
        output._session["output_modalities"] = ["text"]
        await output._ensure_response("r1")

        async def stop(event):
            await output._on_generated_done(
                LLMAgentResponseFinish(
                    session_id="session", response_id="r1", text="done"
                )
            )

        bus.subscribe(TurnLLMAgentStopRequested, stop)
        await bus.request(
            ResponseCancelRequested(session_id="session", response_id="r1")
        )
        terminal = [
            message
            for message in transport.messages
            if message["type"] == "response.done"
        ]
        assert len(terminal) == 1
        assert terminal[0]["response"]["status"] == "cancelled"
        await gateway._handle_text_message(
            '{"type":"response.cancel","event_id":"cancel2","response_id":"r1"}'
        )
        assert transport.messages[-1]["type"] == "error"
        assert transport.messages[-1]["error"]["event_id"] == "cancel2"
        await bus.shutdown()

    asyncio.run(run())


def test_request_requires_exactly_one_owner_and_propagates_failure():
    async def run():
        bus = EventBus()
        request = VADThresholdUpdateRequested(session_id="session", threshold=0.5)
        with pytest.raises(ValueError, match="one handler"):
            await bus.request(request)

        def reject(event):
            raise NotImplementedError("unsupported threshold")

        bus.subscribe(VADThresholdUpdateRequested, reject)
        with pytest.raises(NotImplementedError, match="unsupported threshold"):
            await bus.request(request)
        bus.subscribe(VADThresholdUpdateRequested, lambda event: None)
        with pytest.raises(ValueError, match="one handler"):
            await bus.request(request)
        await bus.shutdown()

    asyncio.run(run())


def test_real_managers_apply_configuration_and_context():
    """The gateway reaches real manager handlers without operating models itself."""
    from xtalk.models import TTS, VAD, Agent, Models
    from xtalk.serving.modules.llm_agent_context_manager import LLMAgentContextManager
    from xtalk.serving.modules.tts_manager import TTSManager
    from xtalk.serving.modules.vad_manager import VADManager

    class RecordingModels:
        """Record calls made by the corresponding manager."""

        def __init__(self):
            self.calls = []

        def set_threshold(self, *, threshold):
            """Record a VAD update."""
            self.calls.append(("threshold", threshold))

        def set_voice(self, *, voice_names):
            """Record a voice update."""
            self.calls.append(("voice", voice_names))

        def set_emotion(self, *, emotion):
            """Record an emotion update."""
            self.calls.append(("emotion", emotion))

        def update_instructions(self, *, instructions):
            """Record an instructions update."""
            self.calls.append(("instructions", instructions))

        def append_context(self, *, role, content):
            """Record a context update."""
            self.calls.append(("context", role, content))

    async def run():
        bus, transport, gateway, output = make_gateway()
        model = RecordingModels()
        models = Models({VAD: model, TTS: model, Agent: model})
        vad = VADManager(bus, "session", models)
        tts = TTSManager(bus, "session", models)
        LLMAgentContextManager(bus, "session", models)
        await gateway._handle_text_message(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "instructions": "new",
                        "audio": {
                            "input": {"turn_detection": {"threshold": 0.7}},
                            "output": {
                                "voice": "voice2",
                                "emotion": {"name": "happy"},
                                "speed": 0.25,
                            },
                        },
                    },
                }
            )
        )
        assert transport.messages[-1]["type"] == "session.updated"
        assert model.calls == [
            ("threshold", 0.7),
            ("instructions", "new"),
            ("voice", ["voice2"]),
            ("emotion", "happy"),
        ]
        assert tts.current_speed == 0.25
        gateway._handlers["conversation.item.create"] = InputGateway._handle_item_create
        await gateway._handle_text_message(
            '{"type":"conversation.item.create","item":{"type":"message","role":"system","content":[{"type":"input_text","text":"history"}]}}'
        )
        assert model.calls[-1] == ("context", "system", "history")
        assert transport.messages[-1]["type"] == "conversation.item.done"
        await vad.shutdown()
        await tts.shutdown()
        await bus.shutdown()

    asyncio.run(run())


def test_cancellation_failure_does_not_emit_a_successful_terminal_event():
    """A failed generation stop reaches the input error boundary."""

    async def run():
        bus, transport, gateway, output = make_gateway()
        await output._ensure_response("r1")

        async def fail(event):
            raise RuntimeError("private stop failure")

        bus.subscribe(TurnLLMAgentStopRequested, fail)
        await gateway._handle_text_message(
            '{"type":"response.cancel","event_id":"cancel1","response_id":"r1"}'
        )
        assert transport.messages[-1]["error"]["type"] == "server_error"
        assert not any(
            message["type"] == "response.done" for message in transport.messages
        )
        assert "r1" not in output._cancelling_responses
        await bus.shutdown()

    asyncio.run(run())
