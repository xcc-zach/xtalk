"""Asynchronous tool for delegating reasoning to an upstream LLM."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .core import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolState,
)


class ThinkInput(ToolInput):
    """Input for the asynchronous thinking tool."""

    question: str


class ThinkOutput(ToolOutput):
    """Output returned by the asynchronous thinking tool."""

    answer: str = ""
    error: str | None = None


def _content_to_text(content: Any) -> str:
    """Normalize upstream model content into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)
    return str(content)


def build_think_tool(
    model: BaseChatModel | dict[str, Any],
    delay_seconds: float = 0.0,
) -> type[AsyncTool]:
    """Build a tool that delegates reasoning to an upstream LLM.

    Parameters
    ----------
    model : BaseChatModel | dict[str, Any]
        Upstream model or model configuration used for reasoning.
    delay_seconds : float, optional
        Artificial delay before invoking the upstream model.

    Returns
    -------
    type[AsyncTool]
        Configured asynchronous thinking tool class.
    """

    resolved_model = model if isinstance(model, BaseChatModel) else ChatOpenAI(**model)

    class ThinkTool(AsyncTool):
        """Delegate difficult questions to a stronger upstream model.

        Use this tool when a question requires additional reasoning. The task
        runs in the background so the conversation can continue while the
        upstream model prepares its answer.
        """

        name = "think"

        @classmethod
        def emit_initial(
            cls,
            tool_call_id: str,
            tool_input: ThinkInput,
            tool_state: ToolState,
            global_state: ToolEngineState,
        ) -> Running:
            """Immediately report that the thinking task has started."""

            del cls, tool_input, tool_state, global_state
            return Running(
                content=(
                    f"思考任务 {tool_call_id} 已经开始。"
                    "请先简短告诉用户诸如“稍等让我想一下”、“好的我先想一想”的话，"
                    "然后继续和用户聊天，不要在结果返回前猜测答案。"
                )
            )

        @classmethod
        def emit_updates(
            cls,
            tool_input: ThinkInput,
            tool_state: ToolState,
            global_state: ToolEngineState,
        ) -> Iterator[Finished[ThinkOutput]]:
            """Run the upstream model and yield its final answer."""

            del cls, tool_state, global_state
            if delay_seconds > 0:
                time.sleep(delay_seconds)

            try:
                response = resolved_model.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a reasoning assistant. Analyze the "
                                "question carefully and return a clear, "
                                "self-contained final answer."
                            )
                        ),
                        HumanMessage(content=tool_input.question),
                    ]
                )
                answer = _content_to_text(response.content)
            except Exception as exc:
                yield Finished(
                    content=ThinkOutput(
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                return

            yield Finished(content=ThinkOutput(answer=answer))

    return ThinkTool


__all__ = [
    "ThinkInput",
    "ThinkOutput",
    "build_think_tool",
]
