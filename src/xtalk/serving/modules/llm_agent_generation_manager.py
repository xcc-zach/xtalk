# -*- coding: utf-8 -*-
"""Consume LLM-agent output streams and coordinate shared TTS emission."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from ...models import Agent, Models
from ...models.agents import AgentOutput, AgentTurnBoundary
from ..event_bus import EventBus
from ..events import (
    ConsumeLLMAgentGenerationRequested,
    ErrorOccurred,
    LLMAgentResponseFinish,
    LLMAgentResponseUpdate,
    LLMFirstChunk,
    LLMModelSwitchRequested,
    ToolCallOccurred,
    TurnLLMAgentPauseRequested,
    TurnLLMAgentResumeRequested,
    TurnLLMAgentStopRequested,
    TurnTTSFlushRequested,
    TurnTTSPauseRequested,
    TurnTTSResumeRequested,
    TurnTTSStartRequested,
    TurnTTSStopRequested,
    TurnTTSTextAppendRequested,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


@dataclass
class _ResponseOutputState:
    """Track one agent stream's current response output."""

    response_id: str | None = None
    accumulated_text: str = ""
    first_chunk_generated: bool = False
    tts_started: bool = False

    def reset(self) -> None:
        """Clear the completed response while preserving the stream."""

        self.response_id = None
        self.accumulated_text = ""
        self.first_chunk_generated = False
        self.tts_started = False


class LLMAgentConsumptionManager(Manager):
    """Consume one or more agent streams and forward their output downstream.

    Notes
    -----
    Multiple agent streams may be active concurrently. Each stream owns an
    independent response so asynchronous tool reports cannot reset or append to
    a response that is already playing.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.models = models
        self.config: dict[str, Any] = config or {}

        self._active_streams: dict[asyncio.Task[None], AsyncIterator[AgentOutput]] = {}
        self._response_states: dict[asyncio.Task[None], _ResponseOutputState] = {}
        self._persistent_streams: set[asyncio.Task[None]] = set()
        self._streams_lock = asyncio.Lock()
        self._output_lock = asyncio.Lock()
        self._resume_event = asyncio.Event()
        self._resume_event.set()

    @Manager.event_handler(LLMModelSwitchRequested, priority=100)
    async def _handle_llm_model_switch(self, event: LLMModelSwitchRequested) -> None:
        """Handle runtime LLM model switch requests."""

        try:
            self._set_llm_model(
                event.model_name,
                event.base_url,
                event.api_key,
                event.extra_body,
            )
        except Exception as exc:
            logger.error(
                "Failed to switch LLM model: %s - session: %s",
                exc,
                self.session_id,
            )

    def _set_llm_model(
        self,
        model: str,
        base_url: str = "",
        api_key: str = "",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        """Replace the current agent LLM with a ``ChatOpenAI`` instance."""
        import os

        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model}

        if api_key:
            kwargs["api_key"] = api_key
        elif os.environ.get("OPENAI_API_KEY"):
            kwargs["api_key"] = os.environ["OPENAI_API_KEY"]

        if base_url:
            kwargs["base_url"] = base_url
        elif os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        if extra_body:
            kwargs["extra_body"] = extra_body

        new_model = ChatOpenAI(**kwargs)
        agent = self.models.get(Agent)
        if agent is None or not hasattr(agent, "model"):
            return

        agent.model = new_model
        if hasattr(agent, "tool_engine"):
            agent.model_with_tools = agent.tool_engine.bind(new_model)
            return
        if not hasattr(agent, "_model_with_tools"):
            return

        tools = list(getattr(agent, "tools_map", {}).values())
        if not tools:
            agent._model_with_tools = new_model
            return

        try:
            agent._model_with_tools = new_model.bind_tools(tools)
        except Exception:
            agent._model_with_tools = new_model

    @Manager.event_handler(ConsumeLLMAgentGenerationRequested, priority=98)
    async def _handle_generation_request(
        self,
        event: ConsumeLLMAgentGenerationRequested,
    ) -> None:
        """Start consuming one newly requested agent stream."""

        iterator = event.stream.__aiter__()
        async with self._streams_lock:
            task = asyncio.create_task(
                self._consume_stream(
                    iterator=iterator,
                )
            )
            self._active_streams[task] = iterator
            self._response_states[task] = _ResponseOutputState()
            if event.persistent:
                self._persistent_streams.add(task)

    @Manager.event_handler(TurnLLMAgentResumeRequested, priority=95)
    async def _handle_generation_resume(
        self,
        event: TurnLLMAgentResumeRequested,
    ) -> None:
        """Resume all paused stream-consumption tasks."""

        del event
        if not self._active_streams:
            return
        if self._resume_event.is_set():
            logger.warning(
                "Try to resume active LLM consumption - session: %s",
                self.session_id,
            )
            return

        self._resume_event.set()
        await self.event_bus.publish(TurnTTSResumeRequested(session_id=self.session_id))

    @Manager.event_handler(TurnLLMAgentPauseRequested, priority=96)
    async def _handle_generation_pause(
        self,
        event: TurnLLMAgentPauseRequested,
    ) -> None:
        """Pause all active stream-consumption tasks."""

        del event
        if not self._active_streams:
            return
        if not self._resume_event.is_set():
            logger.warning(
                "Try to pause paused LLM consumption - session: %s",
                self.session_id,
            )
            return

        self._resume_event.clear()
        await self.event_bus.publish(TurnTTSPauseRequested(session_id=self.session_id))

    @Manager.event_handler(TurnLLMAgentStopRequested, priority=95)
    async def _handle_generation_stop(self, event: TurnLLMAgentStopRequested) -> None:
        """Stop interruptible generation streams and downstream TTS."""

        tasks: list[asyncio.Task[None]]
        iterators: list[AsyncIterator[AgentOutput]]
        async with self._streams_lock:
            tasks, iterators = self._detach_interruptible_streams_locked()

        await self._cancel_tasks(tasks)
        await self._close_iterators(iterators)
        await self.event_bus.publish(
            TurnTTSStopRequested(
                session_id=self.session_id,
                response_id=None,
                reason=event.reason,
            ),
            wait_for_completion=True,
        )

    async def _consume_stream(
        self,
        *,
        iterator: AsyncIterator[AgentOutput],
    ) -> None:
        """Consume one agent stream and forward emitted items downstream."""

        task = asyncio.current_task()
        if task is None:
            logger.warning(
                "LLM stream consumption task missing current task - session: %s",
                self.session_id,
            )
            return
        try:
            while True:
                if not self._resume_event.is_set():
                    await self._resume_event.wait()
                    continue
                try:
                    item = await iterator.__anext__()
                except StopAsyncIteration:
                    break
                await self._consume_stream_item(
                    task=task,
                    item=item,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "LLM stream consumption failed: %s - session: %s",
                exc,
                self.session_id,
            )
            await self._publish_error("llm_stream_error", str(exc))
            await self._finish_response(task)
        finally:
            await self._finalize_stream(
                task=task,
                iterator=iterator,
            )

    async def _consume_stream_item(
        self,
        *,
        task: asyncio.Task[None],
        item: AgentOutput,
    ) -> None:
        """Emit one consumed stream item into the shared turn output channel."""

        async with self._output_lock:
            if task not in self._active_streams:
                return

            if isinstance(item, AgentTurnBoundary):
                await self._finish_response_locked(task)
                return

            state = self._response_states.get(task)
            if state is None:
                return

            if not state.first_chunk_generated:
                state.first_chunk_generated = True
                await self.event_bus.publish(LLMFirstChunk(session_id=self.session_id))

            if isinstance(item, dict):
                if "name" not in item:
                    logger.warning(
                        "Received dict stream item without 'name', skipping - session: %s",
                        self.session_id,
                    )
                    return
                await self._publish_tool_call(item)
                return

            chunk_text = str(item)
            if not chunk_text:
                return

            if state.response_id is None:
                state.response_id = uuid4().hex
            response_id = state.response_id
            state.accumulated_text += chunk_text
            accumulated_text = state.accumulated_text
            should_start_tts = not state.tts_started
            if should_start_tts:
                state.tts_started = True

            await self._publish_llm_response_update(response_id, accumulated_text)
            if should_start_tts:
                await self.event_bus.publish(
                    TurnTTSStartRequested(
                        session_id=self.session_id,
                        response_id=response_id,
                    ),
                    wait_for_completion=True,
                )
            logger.debug(
                "[realtime-tts-race] stage=append_schedule "
                "session=%s chunk_chars=%d accumulated_chars=%d",
                self.session_id,
                len(chunk_text),
                len(accumulated_text),
            )
            await self.event_bus.publish(
                TurnTTSTextAppendRequested(
                    session_id=self.session_id,
                    response_id=response_id,
                    text=chunk_text,
                ),
                wait_for_completion=True,
            )

    async def _finalize_stream(
        self,
        *,
        task: asyncio.Task[None],
        iterator: AsyncIterator[AgentOutput],
    ) -> None:
        """Finalize one stream and finish only its current response."""

        try:
            async with self._streams_lock:
                popped_iterator = self._active_streams.pop(task, None)
                self._persistent_streams.discard(task)
                if popped_iterator is None:
                    return
            await self._finish_response(task)
            self._response_states.pop(task, None)
        finally:
            await self._close_iterator(iterator)

    async def _finish_response(self, task: asyncio.Task[None]) -> None:
        """Finish one stream-owned response under the output lock."""

        async with self._output_lock:
            await self._finish_response_locked(task)

    async def _finish_response_locked(self, task: asyncio.Task[None]) -> None:
        """Flush one response while holding the output lock."""

        state = self._response_states.get(task)
        if state is None:
            return
        response_id = state.response_id
        final_text = state.accumulated_text
        tts_started = state.tts_started
        has_text_output = bool(final_text) or tts_started
        state.reset()
        if not has_text_output:
            return
        if response_id is None:
            return
        if tts_started:
            logger.debug(
                "[realtime-tts-race] stage=flush_schedule "
                "session=%s final_chars=%d",
                self.session_id,
                len(final_text),
            )
            await self.event_bus.publish(
                TurnTTSFlushRequested(
                    session_id=self.session_id,
                    response_id=response_id,
                ),
                wait_for_completion=True,
            )
        await self._publish_llm_response_finish(response_id, final_text)

    def _detach_interruptible_streams_locked(
        self,
    ) -> tuple[list[asyncio.Task[None]], list[AsyncIterator[AgentOutput]]]:
        """Detach turn-scoped streams while preserving persistent streams."""

        tasks = [
            task
            for task in self._active_streams
            if task not in self._persistent_streams
        ]
        iterators = [self._active_streams.pop(task) for task in tasks]
        for task in tasks:
            self._response_states.pop(task, None)
        self._resume_event.set()
        return tasks, iterators

    def _detach_all_streams_locked(
        self,
    ) -> tuple[list[asyncio.Task[None]], list[AsyncIterator[AgentOutput]]]:
        """Detach all active streams while holding ``_streams_lock``."""

        tasks = list(self._active_streams.keys())
        iterators = list(self._active_streams.values())
        self._active_streams.clear()
        self._response_states.clear()
        self._persistent_streams.clear()
        self._resume_event.set()
        return tasks, iterators

    async def _cancel_tasks(self, tasks: list[asyncio.Task[None]]) -> None:
        """Cancel the provided tasks and await their completion."""

        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _close_iterators(
        self,
        iterators: list[AsyncIterator[AgentOutput]],
    ) -> None:
        """Close a list of iterators, suppressing closure errors."""

        for iterator in iterators:
            await self._close_iterator(iterator)

    async def _close_iterator(self, iterator: AsyncIterator[AgentOutput]) -> None:
        """Close one iterator when it exposes ``aclose()``."""

        aclose = getattr(iterator, "aclose", None)
        if not callable(aclose):
            return
        try:
            await aclose()
        except (asyncio.CancelledError, Exception):
            pass

    async def _publish_llm_response_update(
        self,
        response_id: str,
        text: str,
    ) -> None:
        """Publish an incremental update for one response."""

        await self.event_bus.publish(
            LLMAgentResponseUpdate(
                session_id=self.session_id,
                response_id=response_id,
                text=text,
            ),
            wait_for_completion=True,
        )

    async def _publish_llm_response_finish(
        self,
        response_id: str,
        text: str,
    ) -> None:
        """Publish the final generated text for one response."""

        await self.event_bus.publish(
            LLMAgentResponseFinish(
                session_id=self.session_id,
                response_id=response_id,
                text=text,
            ),
            wait_for_completion=True,
        )

    async def _publish_tool_call(self, tool_call: dict[str, Any]) -> None:
        """Forward one tool-call-like payload to downstream consumers."""

        name = tool_call["name"]
        args = tool_call.get("args", {})
        if not name:
            logger.warning(
                "Tool call has empty 'name' field - session: %s",
                self.session_id,
            )
            return

        try:
            await self.event_bus.publish(
                ToolCallOccurred(
                    session_id=self.session_id,
                    name=str(name),
                    args=dict(args) if isinstance(args, dict) else {},
                ),
                wait_for_completion=True,
            )
        except Exception as exc:
            logger.error(
                "Failed to publish tool call '%s': %s - session: %s",
                name,
                exc,
                self.session_id,
            )

    async def _publish_error(self, error_type: str, message: str) -> None:
        """Publish one consumption error event."""

        await self.event_bus.publish(
            ErrorOccurred(
                session_id=self.session_id,
                error_type=error_type,
                error_message=message,
            )
        )

    async def shutdown(self) -> None:
        """Cancel all active streams during service shutdown."""

        async with self._streams_lock:
            tasks, iterators = self._detach_all_streams_locked()
        await self._cancel_tasks(tasks)
        await self._close_iterators(iterators)

        agent = self.models.get(Agent)
        if agent is not None and hasattr(agent, "tool_engine"):
            await agent.tool_engine.shutdown()
