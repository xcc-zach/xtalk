"""Asynchronous tool for LLM-guided web research."""

from __future__ import annotations

import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from pydantic import Field

from .core import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolState,
)
from .retrievers import build_web_search_tool


class DeepResearchInput(ToolInput):
    """Input accepted by the deep research tool."""

    topic: str = Field(min_length=1)
    description: str = ""


class DeepResearchOutput(ToolOutput):
    """Final result returned by the deep research tool."""

    report: str = ""
    sources: list[str] = Field(default_factory=list)
    error: str | None = None


@dataclass
class DeepResearchState(ToolState):
    """Mutable progress state for one deep research call."""

    phase: str = "starting"
    round_number: int = 0
    queries: list[str] = field(default_factory=list)
    completed_queries: int = 0
    current_query: str = ""


def _content_to_text(content: Any) -> str:
    """Normalize model response content into plain text."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _extract_sources(search_result: str) -> list[str]:
    """Extract source URLs from a formatted search result."""

    return [
        url.rstrip(".,;:!?)]}") for url in re.findall(r"https?://\S+", search_result)
    ]


def build_deep_research_tool(
    model: BaseChatModel | dict[str, Any],
    max_rounds: int = 5,
    max_total_searches: int = 15,
) -> type[AsyncTool]:
    """Build an asynchronous LLM-guided deep research tool.

    Parameters
    ----------
    model : BaseChatModel | dict[str, Any]
        Non-networked LLM used to plan searches and synthesize the report.
    max_rounds : int, optional
        Maximum number of times the LLM may request another search round.
    max_total_searches : int, optional
        Maximum number of web searches allowed across all rounds.

    Returns
    -------
    type[AsyncTool]
        Deep research tool class.
    """

    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    if max_total_searches < 1:
        raise ValueError("max_total_searches must be at least 1")

    resolved_model = model if isinstance(model, BaseChatModel) else ChatOpenAI(**model)
    search_tool = build_web_search_tool()
    research_model = resolved_model.bind_tools([search_tool])

    class DeepResearchTool(AsyncTool):
        """Research a topic through planned web searches and LLM synthesis.

        Use this tool when the user requests an in-depth investigation rather
        than a single factual lookup. Research runs in the background and
        reports progress while the conversation continues.
        """

        name = "deep_research"
        subscribe_by_default = True

        @classmethod
        def emit_initial(
            cls,
            tool_call_id: str,
            tool_input: DeepResearchInput,
            tool_state: DeepResearchState,
            global_state: ToolEngineState,
        ) -> Running:
            """Immediately report that deep research has started."""

            del cls, tool_input, global_state
            tool_state.phase = "planning"
            return Running(
                content=(
                    f"后台研究 {tool_call_id} 已经开始。"
                    "不要向用户提及工具、任务、研究计划或进度数字。"
                    "请像自然聊天一样说你需要仔细想想，并结合用户的主题追问一个"
                    "真正有帮助的侧重点。"
                )
            )

        @classmethod
        def emit_updates(
            cls,
            tool_input: DeepResearchInput,
            tool_state: DeepResearchState,
            global_state: ToolEngineState,
        ) -> Iterator[Running | Finished[DeepResearchOutput]]:
            """Let the model search iteratively and run each round concurrently."""

            del cls, global_state
            sources: list[str] = []

            try:
                messages: list[BaseMessage] = [
                    SystemMessage(
                        content=(
                            "You are an autonomous web researcher. Decide what "
                            "information is still needed and call web_search for "
                            "it. You must search before writing the final report. "
                            "Request independent searches together in the same "
                            "response so they can run concurrently. After "
                            "receiving results, reassess whether more evidence is "
                            "needed. When the evidence is sufficient, stop calling "
                            "tools and write a clear, self-contained final report. "
                            f"Use at most {max_total_searches} searches across "
                            f"{max_rounds} rounds. Use the same language as the "
                            "user, preserve source URLs, distinguish facts from "
                            "uncertainty, and never invent unsupported claims."
                        )
                    ),
                    HumanMessage(
                        content=(
                            f"Research topic: {tool_input.topic}\n"
                            f"Description: {tool_input.description or 'None'}"
                        )
                    ),
                ]

                for round_number in range(1, max_rounds + 1):
                    tool_state.phase = "reasoning"
                    tool_state.round_number = round_number
                    response = research_model.invoke(messages)
                    if not isinstance(response, AIMessage):
                        raise TypeError("Research model must return an AIMessage")
                    messages.append(response)

                    if not response.tool_calls:
                        tool_state.phase = "finished"
                        yield Finished(
                            content=DeepResearchOutput(
                                report=_content_to_text(response.content),
                                sources=sources,
                            )
                        )
                        return

                    remaining_searches = (
                        max_total_searches - tool_state.completed_queries
                    )
                    selected_calls = response.tool_calls[:remaining_searches]
                    skipped_calls = response.tool_calls[remaining_searches:]
                    tool_state.phase = "searching"

                    with ThreadPoolExecutor() as executor:
                        future_to_call = {
                            executor.submit(
                                search_tool.invoke,
                                call["args"],
                            ): call
                            for call in selected_calls
                        }
                        for future in as_completed(future_to_call):
                            tool_call = future_to_call[future]
                            result = str(future.result())
                            query = str(tool_call["args"].get("query", ""))
                            messages.append(
                                ToolMessage(
                                    content=result,
                                    tool_call_id=tool_call["id"],
                                    name=tool_call["name"],
                                )
                            )
                            tool_state.queries.append(query)
                            tool_state.completed_queries += 1
                            tool_state.current_query = query
                            for source in _extract_sources(result):
                                if source not in sources:
                                    sources.append(source)

                            yield Running(
                                content=(
                                    "一部分研究资料已经返回：\n\n"
                                    f"{result}\n\n"
                                    "不要播报检索过程、查询词或完成数量。"
                                    "如果资料中已经出现有价值的新发现，就像聊天一样"
                                    "简短地和用户分享一点；否则自然地继续之前的话题。"
                                )
                            )

                    for tool_call in skipped_calls:
                        messages.append(
                            ToolMessage(
                                content=(
                                    "Search skipped because the research search "
                                    "budget has been reached."
                                ),
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            )
                        )

                    tool_state.current_query = ""
                    if tool_state.completed_queries >= max_total_searches:
                        break

                tool_state.phase = "synthesizing"
                final_response = resolved_model.invoke(
                    messages
                    + [
                        HumanMessage(
                            content=(
                                "Do not perform more searches. Write the final "
                                "report now using the evidence already collected."
                            )
                        )
                    ]
                )
                tool_state.phase = "finished"
                yield Finished(
                    content=DeepResearchOutput(
                        report=_content_to_text(final_response.content),
                        sources=sources,
                    )
                )
            except Exception as exc:
                tool_state.phase = "failed"
                yield Finished(
                    content=DeepResearchOutput(
                        sources=sources,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        @classmethod
        def status(
            cls,
            tool_input: DeepResearchInput,
            tool_state: DeepResearchState,
            global_state: ToolEngineState,
        ) -> str:
            """Return the latest human-readable research status."""

            del cls, tool_input, global_state
            if tool_state.phase == "searching":
                return (
                    f"正在检索，已完成 {tool_state.completed_queries}/"
                    f"{len(tool_state.queries)} 项。"
                )
            if tool_state.phase == "synthesizing":
                return "检索已完成，正在生成最终研究报告。"
            if tool_state.phase == "finished":
                return "深度研究已完成。"
            if tool_state.phase == "failed":
                return "深度研究执行失败。"
            return "正在制定研究计划。"

    return DeepResearchTool


__all__ = [
    "DeepResearchInput",
    "DeepResearchOutput",
    "DeepResearchState",
    "build_deep_research_tool",
]
