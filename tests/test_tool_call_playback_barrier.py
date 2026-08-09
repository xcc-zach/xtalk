"""Regression tests for playback-gated tool invocation."""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, ToolCall, ToolMessage

from xtalk.models import Models
from xtalk.models.agents import AgentOutput, AgentTurnBoundary
from xtalk.models.agents.default import DefaultAgent
from xtalk.models.agents.interfaces import ChatHistory
from xtalk.models.agents.tools import ToolEngine
from xtalk.serving.event_bus import EventBus
from xtalk.serving.events import (
    ConsumeLLMAgentGenerationRequested,
    TTSFinished,
    TTSResponseClosed,
    ToolCallOccurred,
    TurnLLMAgentStopRequested,
    TurnTTSFlushRequested,
    TurnTTSStartRequested,
)
from xtalk.serving.modules.llm_agent_generation_manager import (
    LLMAgentConsumptionManager,
)


async def _wait_for_history(
    event_bus: EventBus,
    event_type: str,
    count: int = 1,
) -> list[object]:
    """Wait until the requested event count is recorded."""

    for _ in range(200):
        events = event_bus.get_history(event_type=event_type)
        if len(events) >= count:
            return events
        await asyncio.sleep(0)
    raise AssertionError(f"Timed out waiting for event type {event_type}")


def _tool_call(name: str = "write_board") -> ToolCall:
    """Build one deterministic tool call for a test stream."""

    return ToolCall(name=name, args={"text": "内容"}, id=f"call-{name}")


async def _gated_tool_stream(
    resumed_after_call: asyncio.Event,
    closed: asyncio.Event | None = None,
) -> AsyncIterator[AgentOutput]:
    """Yield a spoken preamble and mark when the consumer resumes execution."""

    try:
        yield "我先把这句话念完。"
        yield _tool_call()
        resumed_after_call.set()
    finally:
        if closed is not None:
            closed.set()


@dataclass
class _ModelChunk:
    """Represent one additive model chunk for agent-history tests."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    def __add__(self, other: object) -> "_ModelChunk":
        """Combine streamed text and tool calls like a LangChain chunk."""

        if not isinstance(other, _ModelChunk):
            return NotImplemented
        return _ModelChunk(
            content=self.content + other.content,
            tool_calls=[*self.tool_calls, *other.tool_calls],
        )


class _ToolCallingModel:
    """Emit a spoken preamble followed by one tool-call batch."""

    def __init__(self, preamble: str, tool_calls: list[ToolCall]) -> None:
        """Initialize the deterministic stream contents."""

        self.preamble = preamble
        self.tool_calls = tool_calls

    async def astream(self, messages: list[object]) -> AsyncIterator[_ModelChunk]:
        """Yield the configured response without inspecting its prompt."""

        del messages
        yield _ModelChunk(content=self.preamble)
        yield _ModelChunk(tool_calls=self.tool_calls)


class _AppendingToolEngine:
    """Append deterministic tool results to the supplied message history."""

    async def ainvoke_and_append(
        self,
        tool_call: ToolCall,
        messages: list[Any],
    ) -> ToolMessage:
        """Return and append one result matching the requested call ID."""

        tool_message = ToolMessage(
            content=f"result:{tool_call['name']}",
            tool_call_id=tool_call["id"],
            name=tool_call["name"],
        )
        ToolEngine.append_tool_message(tool_call, tool_message, messages)
        return tool_message


class ToolCallPlaybackBarrierTests(unittest.IsolatedAsyncioTestCase):
    """Verify that executable tools wait for preceding speech playback."""

    async def test_tool_waits_for_response_close_not_tts_generation(self) -> None:
        """Release a tool only after playback settlement, not TTS generation."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        resumed_after_call = asyncio.Event()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=_gated_tool_stream(resumed_after_call),
            ),
            wait_for_completion=True,
        )

        await _wait_for_history(event_bus, TurnTTSFlushRequested.TYPE)
        started = event_bus.get_history(event_type=TurnTTSStartRequested.TYPE)
        self.assertEqual(len(started), 1)
        response_id = started[0].response_id
        self.assertFalse(resumed_after_call.is_set())
        self.assertFalse(event_bus.get_history(event_type=ToolCallOccurred.TYPE))

        await event_bus.publish(
            TTSFinished(session_id="session", response_id=response_id),
            wait_for_completion=True,
        )
        await asyncio.sleep(0)
        self.assertFalse(resumed_after_call.is_set())
        self.assertFalse(event_bus.get_history(event_type=ToolCallOccurred.TYPE))

        await event_bus.publish(
            TTSResponseClosed(session_id="session", response_id=response_id),
            wait_for_completion=True,
        )
        await asyncio.wait_for(resumed_after_call.wait(), timeout=1.0)

        tool_events = event_bus.get_history(event_type=ToolCallOccurred.TYPE)
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].name, "write_board")
        history = event_bus.get_history()
        close_index = next(
            index
            for index, event in enumerate(history)
            if isinstance(event, TTSResponseClosed)
        )
        tool_index = next(
            index
            for index, event in enumerate(history)
            if isinstance(event, ToolCallOccurred)
        )
        self.assertLess(close_index, tool_index)

    async def test_turn_stop_cancels_tool_waiting_for_playback(self) -> None:
        """Do not resume a pending tool when the user interrupts its preamble."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        resumed_after_call = asyncio.Event()
        stream_closed = asyncio.Event()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=_gated_tool_stream(resumed_after_call, stream_closed),
            ),
            wait_for_completion=True,
        )
        await _wait_for_history(event_bus, TurnTTSFlushRequested.TYPE)

        await manager._handle_generation_stop(
            TurnLLMAgentStopRequested(session_id="session", reason="user_interrupt")
        )
        await asyncio.wait_for(stream_closed.wait(), timeout=1.0)

        self.assertFalse(resumed_after_call.is_set())
        self.assertFalse(event_bus.get_history(event_type=ToolCallOccurred.TYPE))
        self.assertFalse(manager._response_close_waiters)

    async def test_tool_without_spoken_preamble_runs_immediately(self) -> None:
        """Keep tool-only model outputs free from an unnecessary playback wait."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        resumed_after_call = asyncio.Event()

        async def stream() -> AsyncIterator[AgentOutput]:
            """Yield one tool call without preceding text."""

            yield _tool_call()
            resumed_after_call.set()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=stream(),
            ),
            wait_for_completion=True,
        )
        await asyncio.wait_for(resumed_after_call.wait(), timeout=1.0)

        self.assertEqual(
            len(event_bus.get_history(event_type=ToolCallOccurred.TYPE)),
            1,
        )
        self.assertFalse(
            event_bus.get_history(event_type=TurnTTSFlushRequested.TYPE)
        )

    async def test_each_spoken_preamble_gates_its_following_tool(self) -> None:
        """Apply a fresh playback barrier when speech separates two tools."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        first_resumed = asyncio.Event()
        second_resumed = asyncio.Event()

        async def stream() -> AsyncIterator[AgentOutput]:
            """Yield two tool calls separated by independently spoken text."""

            yield "第一句话。"
            yield _tool_call("first_tool")
            first_resumed.set()
            yield "第二句话。"
            yield _tool_call("second_tool")
            second_resumed.set()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=stream(),
            ),
            wait_for_completion=True,
        )

        started = await _wait_for_history(
            event_bus,
            TurnTTSStartRequested.TYPE,
        )
        first_response_id = started[0].response_id
        await _wait_for_history(event_bus, TurnTTSFlushRequested.TYPE)
        self.assertFalse(first_resumed.is_set())

        await event_bus.publish(
            TTSResponseClosed(
                session_id="session",
                response_id=first_response_id,
            ),
            wait_for_completion=True,
        )
        await asyncio.wait_for(first_resumed.wait(), timeout=1.0)

        started = await _wait_for_history(
            event_bus,
            TurnTTSStartRequested.TYPE,
            count=2,
        )
        await _wait_for_history(
            event_bus,
            TurnTTSFlushRequested.TYPE,
            count=2,
        )
        self.assertFalse(second_resumed.is_set())

        await event_bus.publish(
            TTSResponseClosed(
                session_id="session",
                response_id=started[1].response_id,
            ),
            wait_for_completion=True,
        )
        await asyncio.wait_for(second_resumed.wait(), timeout=1.0)

        tool_events = event_bus.get_history(event_type=ToolCallOccurred.TYPE)
        self.assertEqual(
            [event.name for event in tool_events],
            ["first_tool", "second_tool"],
        )

    async def test_tool_result_does_not_reenter_playback_barrier(self) -> None:
        """Forward post-execution result events without treating them as tools."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        result_forwarded = asyncio.Event()

        async def stream() -> AsyncIterator[AgentOutput]:
            """Yield text followed by the internal tool-result pseudo call."""

            yield "工具结果播报。"
            yield ToolCall(
                name="tool_call_result",
                args={
                    "name": "write_board",
                    "args": {},
                    "content": "完成",
                },
                id="tool-result",
            )
            result_forwarded.set()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=stream(),
            ),
            wait_for_completion=True,
        )
        await asyncio.wait_for(result_forwarded.wait(), timeout=1.0)

        tool_events = event_bus.get_history(event_type=ToolCallOccurred.TYPE)
        self.assertEqual(len(tool_events), 1)
        self.assertEqual(tool_events[0].name, "tool_call_result")

    async def test_non_tool_boundary_does_not_wait_for_playback_close(self) -> None:
        """Preserve the existing non-tool response-boundary behavior."""

        event_bus = EventBus(enable_history=True)
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        stream_finished = asyncio.Event()

        async def stream() -> AsyncIterator[AgentOutput]:
            """Yield a normal response and its existing turn boundary."""

            yield "普通回复。"
            yield AgentTurnBoundary()
            stream_finished.set()

        await event_bus.publish(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=stream(),
            ),
            wait_for_completion=True,
        )
        await asyncio.wait_for(stream_finished.wait(), timeout=1.0)

        self.assertEqual(
            len(event_bus.get_history(event_type=TurnTTSFlushRequested.TYPE)),
            1,
        )


class DefaultAgentToolHistoryTests(unittest.IsolatedAsyncioTestCase):
    """Verify playback text precedes the structured tool-call batch."""

    async def test_playback_message_is_committed_before_tool_call_history(self) -> None:
        """Keep every tool result adjacent to its declaring assistant message."""

        preamble = "好的，我先把这句话说完。"
        tool_calls = [
            ToolCall(name="fetch_text", args={}, id="call-fetch"),
            ToolCall(
                name="set_text",
                args={"text": "雅可比定理"},
                id="call-set",
            ),
        ]
        agent = DefaultAgent.__new__(DefaultAgent)
        agent.model_with_tools = _ToolCallingModel(preamble, tool_calls)
        agent.model_for_async_updates = agent.model_with_tools
        agent._chat_history = ChatHistory(system_prompt="system")
        agent.tool_engine = _AppendingToolEngine()

        stream = agent._stream_messages_unlocked(allow_tools=True)
        try:
            self.assertEqual(await anext(stream), preamble)
            first_call = await anext(stream)
            self.assertEqual(first_call["id"], "call-fetch")
            self.assertEqual(len(agent.messages), 1)

            agent._handle_response_finish(
                preamble,
                response_id="response-preamble",
            )

            first_result = await anext(stream)
            self.assertEqual(first_result["name"], "tool_call_result")
            second_call = await anext(stream)
            self.assertEqual(second_call["id"], "call-set")
            second_result = await anext(stream)
            self.assertEqual(second_result["name"], "tool_call_result")

            history = agent.messages[1:]
            self.assertEqual(len(history), 4)
            self.assertIsInstance(history[0], AIMessage)
            self.assertEqual(history[0].content, preamble)
            self.assertIsInstance(history[1], AIMessage)
            self.assertEqual(
                [call["id"] for call in history[1].tool_calls],
                ["call-fetch", "call-set"],
            )
            self.assertIsInstance(history[2], ToolMessage)
            self.assertEqual(history[2].tool_call_id, "call-fetch")
            self.assertIsInstance(history[3], ToolMessage)
            self.assertEqual(history[3].tool_call_id, "call-set")
        finally:
            await stream.aclose()


if __name__ == "__main__":
    unittest.main()
