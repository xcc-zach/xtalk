"""Tests that the desktop Agent preserves the default conversation flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage

from backend.desktop_agent import DesktopDefaultAgent
from xtalk.models.agents.default import DefaultAgent
from xtalk.models.agents.interfaces import AgentOutput
from xtalk.models.agents.tools.utils import build_tool_call_result


@dataclass
class _History:
    """Minimal mutable chat history used by the desktop Agent tests."""

    messages: list[BaseMessage] = field(default_factory=list)

    def append_message(self, message: BaseMessage) -> None:
        """Append one LangChain message."""

        self.messages.append(message)


def _bare_agent() -> DesktopDefaultAgent:
    """Create a desktop Agent without constructing a real model provider."""

    agent = DesktopDefaultAgent.__new__(DesktopDefaultAgent)
    agent._chat_history = _History()
    agent._pending_final_reports = []
    agent._async_tool_update_queue = asyncio.Queue()
    agent._human_input_finished = True
    return agent


def test_desktop_agent_inherits_default_conversation_flow() -> None:
    """Prevent desktop code from diverging from the default Agent loop."""

    flow_methods = (
        "async_accept",
        "clone",
        "_loop_runner",
        "_handle_asr_final",
        "_stream_messages",
        "_stream_messages_unlocked",
        "_on_async_tool_update",
        "_record_async_tool_update",
    )
    for method_name in flow_methods:
        assert method_name not in DesktopDefaultAgent.__dict__
        assert getattr(DesktopDefaultAgent, method_name) is getattr(
            DefaultAgent,
            method_name,
        )


def _tool_update(
    *, running: bool, label: str
) -> tuple[ToolCall, ToolMessage, AgentOutput]:
    """Build one representative asynchronous tool update."""

    tool_call = ToolCall(id=label, name="async_tool_updated", args={})
    tool_message = ToolMessage(
        content=(
            f'{{"running": {str(running).lower()}, '
            f'"tool_output": "{label}"}}'
        ),
        tool_call_id=label,
        name="async_tool_updated",
    )
    output = build_tool_call_result(
        tool_call=tool_call,
        result_content=str(tool_message.content),
    )
    return tool_call, tool_message, output


def test_running_and_final_updates_wake_agent_after_user_finishes() -> None:
    """Let every subscribed update trigger the default Agent report loop."""

    agent = _bare_agent()
    running_call, running_message, running_output = _tool_update(
        running=True,
        label="working",
    )

    agent._record_async_tool_update(
        running_call,
        running_message,
        running_output,
    )

    assert agent._async_tool_update_queue.get_nowait() == running_output

    final_call, final_message, final_output = _tool_update(
        running=False,
        label="done",
    )
    agent._record_async_tool_update(final_call, final_message, final_output)

    assert agent._async_tool_update_queue.get_nowait() == final_output


def test_updates_while_user_speaks_follow_default_queue_policy() -> None:
    """Ignore progress while listening and defer only a final report."""

    agent = _bare_agent()
    agent._human_input_finished = False
    running_call, running_message, running_output = _tool_update(
        running=True,
        label="working",
    )
    agent._record_async_tool_update(
        running_call,
        running_message,
        running_output,
    )

    assert agent._async_tool_update_queue.empty()
    assert agent._pending_final_reports == []

    final_call, final_message, final_output = _tool_update(
        running=False,
        label="done",
    )
    agent._record_async_tool_update(final_call, final_message, final_output)

    assert agent._async_tool_update_queue.empty()
    assert agent._pending_final_reports == [final_output]
