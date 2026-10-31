"""Stateless LangChain chat model driven by a fixed script.

The model inspects the message list it receives and answers from a rule
table, so the real ``DefaultAgent`` and ``ToolEngine`` run exactly as they
would with a hosted LLM while every output stays predictable.
"""

from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator, Iterator
from uuid import uuid4

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from .mock_agent import GREETING_REPLY, LOOKUP_PREAMBLE, LOOKUP_RESULT_REPLY, echo_reply

ASYNC_TOOL_UPDATED = "async_tool_updated"
TIMER_STARTED_REPLY = "计时器已开始。"
TIMER_SUBSCRIBED_REPLY = "计时器已开始，会提醒你。"
TIMER_STOPPED_REPLY = "计时器已停止。"
TIMER_STATUS_PREFIX = "计时器进度："
TIMER_PROGRESS_PREFIX = "计时进行中："
TIMER_FINISHED_PREFIX = "计时结束："
TIMER_DURATIONS = {
    "定一个计时器": {"duration_seconds": 1.0},
    "定一个长计时器": {"duration_seconds": 3.0},
    "定一个带提醒的计时器": {"duration_seconds": 1.2, "reminder_interval_seconds": 0.3},
}
_NUMBERED_SENTENCE = re.compile(r"^第\d+句话$")


class ScriptedPlan:
    """One scripted model output: text followed by optional tool calls."""

    def __init__(self, text: str = "", tool_calls: list[dict[str, Any]] | None = None) -> None:
        self.text = text
        self.tool_calls = tool_calls or []


def _tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid4().hex}"}


def _latest_timer_call_id(messages: list[BaseMessage]) -> str | None:
    for message in reversed(messages):
        if isinstance(message, ToolMessage) and message.name == "mock_timer":
            return str(message.tool_call_id)
    return None


def _call_args(messages: list[BaseMessage], call_id: str) -> dict[str, Any]:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        for call in message.tool_calls:
            if call.get("id") == call_id:
                return dict(call.get("args") or {})
    return {}


def plan_for(messages: list[BaseMessage]) -> ScriptedPlan:
    """Decide the scripted output for one prompt.

    Parameters
    ----------
    messages : list[BaseMessage]
        Prompt messages, possibly ending with a transient system instruction.
    """
    if len(messages) > 1 and isinstance(messages[-1], SystemMessage):
        # A transient instruction asks for a proactive report of the newest
        # asynchronous tool update.
        for message in reversed(messages):
            if isinstance(message, ToolMessage) and message.name == ASYNC_TOOL_UPDATED:
                payload = json.loads(str(message.content))
                output = str(payload.get("tool_output", ""))
                prefix = TIMER_PROGRESS_PREFIX if payload.get("running") else TIMER_FINISHED_PREFIX
                return ScriptedPlan(f"{prefix}{output}。")
        return ScriptedPlan()

    last = messages[-1] if messages else None
    if isinstance(last, ToolMessage) and last.name != ASYNC_TOOL_UPDATED:
        content = str(last.content)
        if last.name == "mock_lookup":
            return ScriptedPlan(LOOKUP_RESULT_REPLY)
        if last.name == "mock_timer":
            call_id = str(last.tool_call_id)
            if _call_args(messages, call_id).get("reminder_interval_seconds") is not None:
                return ScriptedPlan(tool_calls=[_tool_call("subscribe_async_tool", {"source_call_id": call_id})])
            return ScriptedPlan(TIMER_STARTED_REPLY)
        if last.name == "subscribe_async_tool":
            return ScriptedPlan(TIMER_SUBSCRIBED_REPLY)
        if last.name == "stop_async_tool":
            return ScriptedPlan(TIMER_STOPPED_REPLY)
        if last.name == "id_to_async_tool_status":
            return ScriptedPlan(f"{TIMER_STATUS_PREFIX}{content}。")
        return ScriptedPlan(echo_reply(content))

    text = ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = str(message.content).strip()
            break
    if text == "你好":
        return ScriptedPlan(GREETING_REPLY)
    if _NUMBERED_SENTENCE.match(text):
        return ScriptedPlan(echo_reply(text))
    if text == "帮我查一下天气":
        return ScriptedPlan(LOOKUP_PREAMBLE, [_tool_call("mock_lookup", {"query": "天气"})])
    if text in TIMER_DURATIONS:
        return ScriptedPlan(tool_calls=[_tool_call("mock_timer", dict(TIMER_DURATIONS[text]))])
    if text == "停止计时器":
        call_id = _latest_timer_call_id(messages)
        if call_id is not None:
            return ScriptedPlan(tool_calls=[_tool_call("stop_async_tool", {"source_call_id": call_id})])
    if text == "计时器进度如何":
        call_id = _latest_timer_call_id(messages)
        if call_id is not None:
            return ScriptedPlan(tool_calls=[_tool_call("id_to_async_tool_status", {"source_call_id": call_id})])
    return ScriptedPlan(echo_reply(text))


class ScriptedChatModel(BaseChatModel):
    """Chat model that answers from :func:`plan_for` and streams per character."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        del tools, kwargs
        return self

    def _chunks(self, messages: list[BaseMessage]) -> Iterator[ChatGenerationChunk]:
        plan = plan_for(messages)
        for character in plan.text:
            yield ChatGenerationChunk(message=AIMessageChunk(content=character))
        for index, call in enumerate(plan.tool_calls):
            chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[
                    ToolCallChunk(
                        name=call["name"],
                        args=json.dumps(call["args"], ensure_ascii=False),
                        id=call["id"],
                        index=index,
                    )
                ],
            )
            yield ChatGenerationChunk(message=chunk)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        del stop, run_manager, kwargs
        plan = plan_for(messages)
        message = AIMessage(content=plan.text, tool_calls=list(plan.tool_calls))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        yield from self._chunks(messages)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        del stop, run_manager, kwargs
        for chunk in self._chunks(messages):
            yield chunk
