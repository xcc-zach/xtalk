"""Verify explicit text generation and incremental context injection."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import Mock

from xtalk.models import Agent, Models
from xtalk.serving.event_bus import EventBus
from xtalk.serving.modules.input_gateway import InputGateway
from xtalk.serving.modules.llm_agent_context_manager import LLMAgentContextManager


class Transport:
    """Capture outbound protocol events."""

    def __init__(self):
        self.messages = []

    async def send_json(self, message):
        """Capture one JSON event."""
        self.messages.append(message)


def test_context_appends_without_generation_and_invalid_items_are_atomic():
    async def run():
        agent = Mock(spec=Agent)
        transport = Transport()
        bus = EventBus()
        gateway = InputGateway(bus, "session", transport)
        gateway._handlers = {
            "conversation.item.create": InputGateway._handle_item_create
        }
        gateway._service = SimpleNamespace(
            models=Models({Agent: agent}), service_config={}
        )
        LLMAgentContextManager(bus, "session", gateway._service.models)
        gateway._schedule_text_submission = Mock()

        async def submit(role, text, extension=None, identifier="item1"):
            event = {
                "type": "conversation.item.create",
                "item": {
                    "id": identifier,
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": text}],
                },
            }
            if extension is not None:
                event["xtalk"] = extension
            await gateway._handle_text_message(json.dumps(event))

        await submit("user", "history")
        agent.append_context.assert_called_once_with(role="user", content="history")
        gateway._schedule_text_submission.assert_not_called()
        await submit("assistant", "invalid", {"generate_response": True}, "bad")
        assert transport.messages[-1]["type"] == "error"
        assert len(gateway._items) == 1
        await submit("user", "live", {"generate_response": True}, "live")
        gateway._schedule_text_submission.assert_called_once_with("live", item_id="live")
        assert len(gateway._items) == 2
        assert agent.append_context.call_count == 1
        await submit("user", "duplicate", identifier="item1")
        assert transport.messages[-1]["type"] == "error"
        assert agent.append_context.call_count == 1
        await bus.shutdown()

    asyncio.run(run())


def test_empty_service_settings_still_share_live_protocol_configuration():
    """Default configuration must reach gateways and managers by reference."""
    from xtalk import Service
    from xtalk.serving.modules.llm_agent_generation_manager import (
        LLMAgentConsumptionManager,
    )
    from xtalk.serving.modules.vad_manager import VADManager

    async def run():
        prototype = Service(
            models=Models(), manager_classes=[LLMAgentConsumptionManager, VADManager]
        )
        service = prototype.clone(Transport())
        assert service.input_gateway.config is service.service_config
        assert service.output_gateway.config is service.service_config
        assert all(
            manager.config is service.service_config for manager in service._managers
        )
        service.service_config["output_modalities"] = ["text"]
        assert service._managers[0].config["output_modalities"] == ["text"]
        assert prototype.service_config["output_modalities"] == ["audio"]
        await service.stop()
        await prototype.stop()

    asyncio.run(run())
