import asyncio
from abc import ABC, abstractmethod
from typing import Iterable, Union, TypedDict, List, AsyncIterator, Callable
from langchain.chat_models.base import BaseChatModel
from ..pipelines.context import PipelineContext
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool


class AgentInput(TypedDict):
    """Structured payload for LLM generation input.

    - content: raw user text
    - context: pipeline context (PipelineContext)
    """

    content: str
    context: PipelineContext


class Agent(ABC):
    @abstractmethod
    def generate(
        self, input: Union[str, AgentInput]
    ) -> Union[str, tuple[str, List[ToolCall]]]:
        """
        Generate a full reply for the input.

        Input formats:
        - str: just raw text (no context)
        - dict: {"content": raw text, "context": PipelineContext}

        Return format (backward compatible):
        - plain `str` reply, or
        - `(text, tool_calls)` tuple when tool invocations exist, where
          text is the reply and tool_calls is a List[ToolCall]
        """
        pass

    def generate_stream(
        self, input: Union[str, AgentInput]
    ) -> Iterable[Union[str, ToolCall]]:
        """Stream responses for the input (same format as `generate`).

        Default implementation runs `generate` first, then yields text/tool_calls.
        """
        result = self.generate(input)
        if isinstance(result, tuple):
            text, tool_calls = result
            # Yield tool calls first so upstream can react early
            for tc in tool_calls:
                yield tc
            yield text
        else:
            yield result

    async def async_generate(
        self, input: Union[str, AgentInput]
    ) -> Union[str, tuple[str, List[ToolCall]]]:
        """Async wrapper around `generate`, running sync logic in an executor."""

        loop = asyncio.get_running_loop()

        def _invoke():
            return self.generate(input)

        return await loop.run_in_executor(None, _invoke)

    async def async_generate_stream(
        self, input: Union[str, AgentInput]
    ) -> AsyncIterator[Union[str, ToolCall]]:
        """Async streaming wrapper pulling from the sync generator in executor."""

        loop = asyncio.get_running_loop()
        iterator = iter(self.generate_stream(input))
        sentinel = object()

        try:
            while True:

                def _next_item():
                    try:
                        return next(iterator)
                    except StopIteration:
                        return sentinel

                item = await loop.run_in_executor(None, _next_item)
                if item is sentinel:
                    break
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    @abstractmethod
    def clone(self) -> "Agent":
        pass

    def get_llm(self) -> BaseChatModel | None:
        """Return the underlying LLM instance if available."""
        return None

    def get_chat_history(self) -> str | None:
        """Return textual conversation history if available."""
        return None

    def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]):
        pass
