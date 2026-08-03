"""Tests for the desktop-only asynchronous Agent policy."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage, ToolCall, ToolMessage

from backend.desktop_agent import DesktopDefaultAgent
from xtalk.models.agents.tools import AsyncTool, ToolEngine
from xtalk.models.agents.tools.utils import build_tool_call_result


@dataclass
class _History:
    """Minimal mutable chat history used by the desktop Agent tests."""

    messages: list[BaseMessage] = field(default_factory=list)

    def append_message(self, message: BaseMessage) -> None:
        """Append one LangChain message."""

        self.messages.append(message)


class _ToolCallChunk:
    """One complete fake model chunk containing an async tool call."""

    content = ""

    def __init__(self, tool_call: ToolCall) -> None:
        """Store the structured tool call exposed to the Agent."""

        self.tool_calls = [tool_call]


class _OneCallModel:
    """Fake streaming model that requests one tool exactly once."""

    def __init__(self, tool_call: ToolCall) -> None:
        """Initialize the one-shot model stream."""

        self.tool_call = tool_call
        self.calls = 0

    async def astream(self, messages: list[BaseMessage]):
        """Yield one tool call and fail if the Agent asks for narration."""

        del messages
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("async tool receipt must not trigger narration")
        yield _ToolCallChunk(self.tool_call)


class _AsyncToolEngine:
    """Minimal tool engine returning an async operation receipt."""

    def __init__(self) -> None:
        """Expose one abstract AsyncTool marker under the fake name."""

        self._name_to_tool = {"codex_session_create": AsyncTool}

    async def ainvoke_and_append(
        self,
        tool_call: ToolCall,
        messages: list[BaseMessage],
    ) -> ToolMessage:
        """Append and return one initial running receipt."""

        result = ToolMessage(
            content=f"Codex started. Tool call ID: {tool_call['id']}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        ToolEngine.append_tool_message(tool_call, result, messages)
        return result


def _bare_agent() -> DesktopDefaultAgent:
    """Create a desktop Agent without constructing a real model provider."""

    agent = DesktopDefaultAgent.__new__(DesktopDefaultAgent)
    agent._chat_history = _History()
    agent._pending_final_reports = []
    agent._async_tool_update_queue = asyncio.Queue()
    agent._human_input_finished = True
    return agent


def test_async_tool_receipt_stops_until_the_final_update() -> None:
    """Avoid turning an async receipt into a separate assistant message."""

    async def scenario() -> tuple[list[Any], int]:
        tool_call = ToolCall(
            id="call-1",
            name="codex_session_create",
            args={"task": "list files"},
        )
        model = _OneCallModel(tool_call)
        agent = _bare_agent()
        agent.model_with_tools = model
        agent.model_for_async_updates = model
        agent.tool_engine = _AsyncToolEngine()  # type: ignore[assignment]
        output = [
            item
            async for item in agent._stream_messages_unlocked(allow_tools=True)
        ]
        return output, model.calls

    output, model_calls = asyncio.run(scenario())

    assert model_calls == 1
    assert len(output) == 2
    assert output[0]["id"] == "call-1"


def test_running_updates_stay_in_tool_ui_and_final_update_wakes_agent() -> None:
    """Queue one LLM report only after an async desktop tool finishes."""

    agent = _bare_agent()
    running_call = ToolCall(id="running", name="async_tool_updated", args={})
    running_message = ToolMessage(
        content='{"running": true, "tool_output": "working"}',
        tool_call_id="running",
        name="async_tool_updated",
    )
    running_output = build_tool_call_result(
        tool_call=running_call,
        result_content=str(running_message.content),
    )

    agent._record_async_tool_update(
        running_call,
        running_message,
        running_output,
    )

    assert agent._async_tool_update_queue.empty()

    final_call = ToolCall(id="final", name="async_tool_updated", args={})
    final_message = ToolMessage(
        content='{"running": false, "tool_output": "done"}',
        tool_call_id="final",
        name="async_tool_updated",
    )
    final_output = build_tool_call_result(
        tool_call=final_call,
        result_content=str(final_message.content),
    )
    agent._record_async_tool_update(final_call, final_message, final_output)

    assert agent._async_tool_update_queue.get_nowait() == final_output
