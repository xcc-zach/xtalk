"""Regression tests for persistent LLM-agent loop consumption."""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import AsyncIterator
from typing import Any

from xtalk.models import Agent, Models
from xtalk.models.agents import AgentContext, AgentOutput
from xtalk.serving.events import (
    ConsumeLLMAgentGenerationRequested,
    Event,
    LLMAgentLoop,
    LLMAgentResponseUpdate,
    TurnLLMAgentStopRequested,
)
from xtalk.serving.modules.llm_agent_context_manager import LLMAgentContextManager
from xtalk.serving.modules.llm_agent_generation_manager import (
    LLMAgentConsumptionManager,
)


class _RecordingEventBus:
    """Record published events without dispatching them."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def subscribe(
        self,
        event_class: type[Event] | str,
        handler: Any,
        priority: int = 0,
    ) -> None:
        """Accept manager handler registrations.

        Parameters
        ----------
        event_class : type[Event] | str
            Event type registered by the manager.
        handler : Any
            Handler registered by the manager.
        priority : int, optional
            Handler execution priority.
        """

        del event_class
        del handler
        del priority

    async def publish(
        self,
        event: Event,
        *,
        wait_for_completion: bool = False,
    ) -> None:
        """Record one event.

        Parameters
        ----------
        event : Event
            Event being published.
        wait_for_completion : bool, optional
            Ignored completion-waiting preference.
        """

        del wait_for_completion
        self.events.append(event)


class _StubAgent:
    """Return inert streams and record accepted contexts."""

    def __init__(self) -> None:
        self.contexts: list[AgentContext] = []

    def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
        """Record a context and return an inert output stream.

        Parameters
        ----------
        context : AgentContext
            Context forwarded by the serving layer.

        Returns
        -------
        AsyncIterator[AgentOutput]
            Inert output stream.
        """

        self.contexts.append(context)
        return self._empty_stream()

    async def _empty_stream(self) -> AsyncIterator[AgentOutput]:
        """Yield no output."""

        if False:
            yield ""


async def _queue_stream(
    queue: asyncio.Queue[AgentOutput],
    started: asyncio.Event,
) -> AsyncIterator[AgentOutput]:
    """Yield queue items until the consumer cancels the stream."""

    started.set()
    while True:
        yield await queue.get()


class PersistentAgentStreamTest(unittest.IsolatedAsyncioTestCase):
    """Verify loop streams survive turn-level interruption."""

    async def test_context_manager_marks_agent_loop_persistent(self) -> None:
        """Mark only the long-lived agent-loop stream as persistent."""

        event_bus = _RecordingEventBus()
        agent = _StubAgent()
        models = Models({Agent: agent})
        manager = LLMAgentContextManager(event_bus, "session", models)

        await manager._handle_llm_agent_loop(LLMAgentLoop(session_id="session"))

        request = event_bus.events[-1]
        self.assertIsInstance(request, ConsumeLLMAgentGenerationRequested)
        assert isinstance(request, ConsumeLLMAgentGenerationRequested)
        self.assertTrue(request.persistent)

    async def test_turn_stop_preserves_loop_and_forwards_later_update(self) -> None:
        """Forward async-tool output after interrupting a turn stream."""

        event_bus = _RecordingEventBus()
        manager = LLMAgentConsumptionManager(event_bus, "session", Models())
        self.addAsyncCleanup(manager.shutdown)
        loop_queue: asyncio.Queue[AgentOutput] = asyncio.Queue()
        turn_queue: asyncio.Queue[AgentOutput] = asyncio.Queue()
        loop_started = asyncio.Event()
        turn_started = asyncio.Event()

        await manager._handle_generation_request(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=_queue_stream(loop_queue, loop_started),
                persistent=True,
            )
        )
        await manager._handle_generation_request(
            ConsumeLLMAgentGenerationRequested(
                session_id="session",
                stream=_queue_stream(turn_queue, turn_started),
            )
        )
        await asyncio.gather(loop_started.wait(), turn_started.wait())

        await manager._handle_generation_stop(
            TurnLLMAgentStopRequested(session_id="session", reason="test")
        )
        self.assertEqual(len(manager._active_streams), 1)
        self.assertEqual(len(manager._persistent_streams), 1)

        await loop_queue.put("timer progress")
        updates: list[LLMAgentResponseUpdate] = []
        for _ in range(20):
            updates = [
                event
                for event in event_bus.events
                if isinstance(event, LLMAgentResponseUpdate)
            ]
            if updates:
                break
            await asyncio.sleep(0)

        self.assertTrue(updates)
        self.assertEqual(updates[-1].text, "timer progress")


if __name__ == "__main__":
    unittest.main()
