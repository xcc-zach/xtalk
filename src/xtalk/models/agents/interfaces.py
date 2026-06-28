import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, TypedDict, AsyncIterator, Callable, Any, Union, TypeVar

from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, ToolCall
from langchain_core.tools import BaseTool

from .tools.utils import ToolCallResult
from ..registry import model_type


class AgentContext(TypedDict):
    """Incremental context update accepted by an agent.

    Notes
    -----
    ``type`` identifies the logical context stream, while ``data`` carries the
    event-derived payload for that stream.
    """

    type: str
    data: dict[str, Any]


AgentOutput = Union[str, ToolCall, ToolCallResult]
T = TypeVar("T")


@dataclass
class PlaybackAIMessageMeta:
    """Track merge state for one playback-managed assistant message."""

    final: bool = False
    prefix: str | None = None


class ChatHistory:
    """Manage chat history plus playback-aware assistant-message merging."""

    def __init__(self, system_prompt: str) -> None:
        """Initialize the history with one system message.

        Parameters
        ----------
        system_prompt:
            The system prompt to place at the start of the message list.
        """

        self._messages: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self._playback_ai_meta: dict[int, PlaybackAIMessageMeta] = {}

    @property
    def messages(self) -> list[BaseMessage]:
        """Return the current chat-history message list."""

        return self._messages

    def append_message(self, message: BaseMessage) -> None:
        """Append one message to the history unchanged.

        Parameters
        ----------
        message:
            The message to append.
        """

        self._messages.append(message)

    def append_or_update_ai_message(self, full_text: str, *, final: bool) -> None:
        """Append or merge one playback-managed assistant message.

        Parameters
        ----------
        full_text:
            The cumulative assistant text confirmed by playback.
        final:
            Whether this update closes the playback-managed assistant message.
        """

        if not full_text:
            return

        previous_playback = self._find_previous_playback_ai_message()
        if previous_playback is not None:
            index, message, meta = previous_playback
            if index == len(self._messages) - 1 and not meta.final:
                if meta.prefix and full_text.startswith(meta.prefix):
                    message.content = full_text[len(meta.prefix) :]
                else:
                    message.content = full_text
                    meta.prefix = None
                meta.final = final
                return

            previous_full_text = self._resolve_playback_full_text(message, meta)
            if not meta.final and full_text.startswith(previous_full_text):
                suffix = full_text[len(previous_full_text) :]
                meta.final = final
                if not suffix:
                    return
                self._messages.append(AIMessage(content=suffix))
                self._playback_ai_meta[len(self._messages) - 1] = PlaybackAIMessageMeta(
                    final=final,
                    prefix=previous_full_text,
                )
                return

        self._append_playback_ai_message(full_text, final=final)

    def _append_playback_ai_message(self, content: str, *, final: bool) -> int:
        """Append one playback-managed assistant message."""

        message_index = len(self._messages)
        self.append_message(AIMessage(content=content))
        self._playback_ai_meta[message_index] = PlaybackAIMessageMeta(final=final)
        return message_index

    def _find_previous_playback_ai_message(
        self,
    ) -> tuple[int, AIMessage, PlaybackAIMessageMeta] | None:
        """Return the latest playback-managed assistant message if present."""

        for index in range(len(self._messages) - 1, -1, -1):
            if index not in self._playback_ai_meta:
                continue
            message = self._messages[index]
            if isinstance(message, AIMessage):
                return index, message, self._playback_ai_meta[index]
        return None

    @staticmethod
    def _resolve_playback_full_text(
        message: AIMessage,
        meta: PlaybackAIMessageMeta,
    ) -> str:
        """Resolve the full playback text represented by one assistant message."""

        return f"{meta.prefix or ''}{message.content}"


@model_type(aliases=["llm_agent"])
class Agent(ABC):
    """Abstract interface for conversational agents used by Xtalk."""

    @staticmethod
    def content_to_text(content: Any) -> str:
        """Normalize model content blocks into plain text.

        Parameters
        ----------
        content:
            Content emitted by a LangChain model chunk or message.

        Returns
        -------
        str
            Plain-text content extracted from the input.
        """

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

    @abstractmethod
    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Accept an incremental context update.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving-layer events.

        Yields
        ------
        AgentStreamItem
            Zero or more streamed response items triggered by the context
            update.
        """
        pass

    async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
        """Asynchronously accept an incremental context update.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving-layer events.

        Yields
        ------
        AgentStreamItem
            Streamed response items triggered by the context update.
        """

        loop = asyncio.get_running_loop()
        iterator = iter(self.accept(context))
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

    def sync_iter_from_async(self, async_iter: AsyncIterator[T]) -> Iterable[T]:
        """Convert an async iterator into a synchronous generator.

        Parameters
        ----------
        async_iter : AsyncIterator[T]
            Async iterator to bridge into synchronous iteration.

        Yields
        ------
        T
            Items produced by ``async_iter``.
        """

        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    item = loop.run_until_complete(async_iter.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        finally:
            aclose = getattr(async_iter, "aclose", None)
            if callable(aclose):
                try:
                    loop.run_until_complete(aclose())
                except Exception:
                    pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()

    @abstractmethod
    def clone(self) -> "Agent":
        """Clone the agent for a new session.

        Returns
        -------
        Agent
            Session-safe agent instance.
        """
        pass

    @abstractmethod
    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore persisted conversation messages into the agent state.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Persisted chat messages ordered by session history.
        """
        pass

    def get_chat_history(self, with_system: bool = False) -> str | None:
        """Return the serialized conversation history when available.

        Parameters
        ----------
        with_system : bool, optional
            Whether to include the system prompt message when supported by the
            concrete implementation.

        Returns
        -------
        str | None
            Conversation history or ``None``.
        """
        return None

    def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None:
        """Attach tools to the agent.

        Parameters
        ----------
        tools : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or factories that produce tool instances.
        """
        pass
