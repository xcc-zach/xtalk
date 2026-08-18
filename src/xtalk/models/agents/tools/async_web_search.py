"""Asynchronous wrapper for the existing web search tool."""

from collections.abc import Iterator

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


class AsyncWebSearchInput(ToolInput):
    """Input accepted by the asynchronous web search tool."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)
    region: str | None = None
    lang: str | None = None


class AsyncWebSearchOutput(ToolOutput):
    """Final result returned by the asynchronous web search tool."""

    results: str


def build_async_web_search_tool() -> type[AsyncTool]:
    """Build an asynchronous wrapper around the existing web search tool.

    Returns
    -------
    type[AsyncTool]
        Asynchronous web search tool class.
    """

    search_tool = build_web_search_tool()

    class AsyncWebSearchTool(AsyncTool):
        """Search the web in the background for current factual information.

        Use this tool for news, weather, current events, specific facts, or any
        question whose answer should be verified with up-to-date sources. The
        user can continue the conversation while the search is running.
        """

        name = "web_search"

        @classmethod
        def emit_initial(
            cls,
            tool_call_id: str,
            tool_input: AsyncWebSearchInput,
            tool_state: ToolState,
            global_state: ToolEngineState,
        ) -> Running:
            """Immediately report that the web search has started."""

            del cls, tool_input, tool_state, global_state
            return Running(
                content=(
                    f"网络搜索任务 {tool_call_id} 已经开始。"
                    "请简短告诉用户正在查询，结果完成后会主动汇报。"
                )
            )

        @classmethod
        def emit_updates(
            cls,
            tool_input: AsyncWebSearchInput,
            tool_state: ToolState,
            global_state: ToolEngineState,
        ) -> Iterator[Finished[AsyncWebSearchOutput]]:
            """Run the existing web search and yield its final result."""

            del cls, tool_state, global_state
            result = search_tool.invoke(tool_input.model_dump(exclude_none=True))
            yield Finished(content=AsyncWebSearchOutput(results=str(result)))

    return AsyncWebSearchTool


__all__ = [
    "AsyncWebSearchInput",
    "AsyncWebSearchOutput",
    "build_async_web_search_tool",
]
