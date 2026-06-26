"""Long-term-state agent scaffold.

Implementation overview
-----------------------
- Start one LLM inference task for every ``ASRPartial`` event.
- Each partial-path inference uses the slow model and produces structured
  output with two fields: reasoning content and a partial-draft reply.
- Partial-path inference treats the current partial ASR text as if it were
  complete and takes the dialogue history, the latest completed partial ASR
  text, and the latest completed partial-draft reply as input.
- Partial-path inference never emits user-visible output directly in this
  version. It only prepares draft state for the final path.
- When ``ASRResultFinal`` arrives, aggregate partial-path results to generate
  the final reply. The current strategy is to use the latest completed partial
  result for the same turn.
- Final-path generation uses the fast model and takes the dialogue history, the
  final ASR text, and the selected partial reply draft as input. When no
  partial draft is available, the fast model falls back to direct chat-history
  messages plus the final ASR text.
- Every committed final reply must be injected into ``messages`` so later turns
  can see the updated conversation history.
- All generation paths use streaming generation internally.
- This version does not support tools.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator, Callable, Iterable

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from ...log_utils import logger
from .default import AgentSession
from .interfaces import Agent, AgentContext, AgentOutput
from ..registry import model

@dataclass
class PartialInferenceResult:
    """Structured output returned by the slow-model partial inference."""

    reasoning_content: str
    reply_content: str


@dataclass
class PartialInferenceState:
    """Track one slow-model inference task launched from an ASR partial."""

    partial_id: int
    text: str
    task: asyncio.Task[PartialInferenceResult] | None = None
    result: PartialInferenceResult | None = None


@dataclass
class LTSRuntimeState:
    """Mutable runtime state for concurrent partial/final generation."""

    next_partial_id: int = 1
    partials: dict[int, PartialInferenceState] = field(default_factory=dict)


def _format_chat_history(
    messages: list[BaseMessage],
    *,
    with_system: bool = False,
) -> str | None:
    """Render plain-text chat history from LangChain messages.

    Parameters
    ----------
    messages : list[BaseMessage]
        Chat messages to serialize.
    with_system : bool, optional
        Whether to include system messages in the serialized output.

    Returns
    -------
    str | None
        Rendered history string, or ``None`` when no messages are available.
    """

    if not messages:
        return None
    lines: list[str] = []
    for message in messages:
        role = "System"
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        if role == "System" and not with_system:
            continue
        content = message.content
        lines.append(f"{role}: {content if isinstance(content, str) else str(content)}")
    return "\n".join(lines)


@model
class LTSAgent(Agent):
    """Scaffold for an agent with long-term state support.

    Parameters
    ----------
    slow_model : BaseChatModel | dict[str, Any]
        Backing slow chat model instance or deferred model configuration.
    fast_model : BaseChatModel | dict[str, Any]
        Backing fast chat model instance or deferred model configuration.
    system_prompt : str, optional
        Base system prompt for future LTS interactions.
    """

    BASE_PROMPT: str = (
        "Your reply should be conversational and concise. DO NOT include any special symbols that cannot be read aloud, but can include puntuations."
    )
    PARTIAL_INFERENCE_PROMPT_TEMPLATE: str = (
        "You are assisting a streaming speech agent.\n"
        "Given the dialogue history, the newest partial ASR text, "
        "and the latest completed partial-draft reply, create a draft reply to "
        "the dialogue history and partial ASR text.\n"
        "Reply should be generated as if the current partial ASR text is complete."
        "Return exactly one JSON object and nothing else.\n"
        'The JSON object must contain keys "reasoning_content" and '
        '"reply_content".\n'
        '"reasoning_content" must be a string.\n'
        '"reply_content" must be a string.\n\n'
        "Dialogue history:\n{history}\n\n"
        "Current partial ASR:\n{partial_text}\n\n"
        'Latest completed partial-draft reply to "{latest_partial_asr}": '
        "{latest_partial_reply}\n"
    )
    FINAL_RESPONSE_PROMPT_TEMPLATE: str = (
        "You are assisting a streaming speech agent.\n"
        "Generate the final assistant reply for the current user turn.\n"
        "Use the dialogue history, the final ASR text, and the partial-reply "
        "draft when it is helpful.\n"
        "Return plain text only.\n\n"
        "Dialogue history:\n{history}\n\n"
        'Latest partial reply draft for "{latest_partial_asr}":\n'
        "{latest_partial_reply}\n\n"
        "Final ASR:\n{final_text}\n"
    )

    def __init__(
        self,
        slow_model: BaseChatModel | dict[str, Any],
        fast_model: BaseChatModel | dict[str, Any],
        system_prompt: str = "",
    ) -> None:
        """Initialize the LTS agent scaffold."""

        self.slow_model = self._coerce_model(slow_model)
        self.fast_model = self._coerce_model(fast_model)
        self.system_prompt = self.BASE_PROMPT + system_prompt
        self._session = AgentSession()
        self._runtime = LTSRuntimeState()
        self._state_lock = asyncio.Lock()

    @staticmethod
    def _coerce_model(model: BaseChatModel | dict[str, Any]) -> BaseChatModel:
        """Coerce model configuration into a concrete chat model.

        Parameters
        ----------
        model : BaseChatModel | dict[str, Any]
            Chat model instance or ``ChatOpenAI`` configuration dict.

        Returns
        -------
        BaseChatModel
            Concrete chat model instance.
        """

        if isinstance(model, dict):
            return ChatOpenAI(**model)
        return model

    def _set_system_prompt(self) -> None:
        """Ensure the first session message contains the current system prompt."""

        if self._session.messages and isinstance(
            self._session.messages[0], SystemMessage
        ):
            self._session.messages[0].content = self.system_prompt
            return
        self._session.messages.insert(0, SystemMessage(content=self.system_prompt))

    def _append_final_reply(self, user_text: str, reply_text: str) -> None:
        """Append one committed user/assistant turn into session history."""

        self._set_system_prompt()
        self._session.messages.append(HumanMessage(content=user_text))
        self._session.messages.append(AIMessage(content=reply_text))

    def _get_chat_messages_snapshot(self) -> list[BaseMessage]:
        """Return one detached snapshot of the current chat messages.

        Returns
        -------
        list[BaseMessage]
            Chat history snapshot with the current system prompt normalized into
            the first message.
        """

        messages = deepcopy(self._session.messages)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0].content = self.system_prompt
            return messages
        return [SystemMessage(content=self.system_prompt), *messages]

    def _get_latest_completed_partial_context(self) -> tuple[str, str] | None:
        """Return the latest completed partial input text and reply content."""

        completed = [
            state
            for state in self._runtime.partials.values()
            if state.result is not None and state.result.reply_content.strip()
        ]
        if not completed:
            return None
        latest_state = max(completed, key=lambda state: state.partial_id)
        result = latest_state.result
        if result is None:
            return None
        return latest_state.text, result.reply_content

    def _build_partial_prompt(
        self,
        *,
        history_text: str | None,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
        partial_text: str,
    ) -> str:
        """Build the slow-model prompt for one partial ASR update."""

        history = history_text or "<empty>"
        latest_partial_asr_text = latest_partial_asr or "<none>"
        latest_partial_reply_text = latest_partial_reply or "<none>"
        return self.PARTIAL_INFERENCE_PROMPT_TEMPLATE.format(
            history=history,
            latest_partial_asr=latest_partial_asr_text,
            latest_partial_reply=latest_partial_reply_text,
            partial_text=partial_text,
        )

    def _build_partial_messages(
        self,
        *,
        history_text: str | None,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
        partial_text: str,
    ) -> list[BaseMessage]:
        """Build the slow-model messages for one partial ASR update.

        Parameters
        ----------
        history_text : str | None
            Serialized chat history without the current partial text.
        latest_partial_asr : str | None
            Input ASR text corresponding to the latest completed partial draft.
        latest_partial_reply : str | None
            Latest completed partial-draft reply, if any.
        partial_text : str
            Current partial ASR text.

        Returns
        -------
        list[BaseMessage]
            Messages sent to the slow model.
        """

        prompt = self._build_partial_prompt(
            history_text=history_text,
            latest_partial_asr=latest_partial_asr,
            latest_partial_reply=latest_partial_reply,
            partial_text=partial_text,
        )
        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]

    def _build_final_prompt(
        self,
        *,
        history_text: str | None,
        final_text: str,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
    ) -> str:
        """Build the fast-model prompt for final response generation."""

        history = history_text or "<empty>"
        latest_partial_asr_text = latest_partial_asr or "<none>"
        latest_partial_reply_text = latest_partial_reply or "<none>"
        return self.FINAL_RESPONSE_PROMPT_TEMPLATE.format(
            history=history,
            latest_partial_asr=latest_partial_asr_text,
            latest_partial_reply=latest_partial_reply_text,
            final_text=final_text,
        )

    def _build_final_messages(
        self,
        *,
        history_text: str | None,
        final_text: str,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
    ) -> list[BaseMessage]:
        """Build the fast-model messages for one final ASR update.

        Parameters
        ----------
        history_text : str | None
            Serialized chat history without the current final text.
        final_text : str
            Final ASR text for the current turn.
        latest_partial_asr : str | None
            Input ASR text corresponding to the latest completed partial draft
            for the current turn.
        latest_partial_reply : str | None
            Latest completed partial-draft reply for the current turn.

        Returns
        -------
        list[BaseMessage]
            Messages sent to the fast model.
        """

        if latest_partial_reply and latest_partial_reply.strip():
            prompt = self._build_final_prompt(
                history_text=history_text,
                final_text=final_text,
                latest_partial_asr=latest_partial_asr,
                latest_partial_reply=latest_partial_reply,
            )
            return [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt),
            ]

        messages = self._get_chat_messages_snapshot()
        messages.append(HumanMessage(content=final_text))
        return messages

    def _extract_json_text(self, text: str) -> str:
        """Extract one JSON object from model text output."""

        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _parse_partial_result(self, text: str) -> PartialInferenceResult:
        """Parse the slow-model JSON output into a typed partial result."""

        payload = json.loads(self._extract_json_text(text))
        if not isinstance(payload, dict):
            raise ValueError("Partial inference output must be a JSON object.")
        reasoning_content = payload.get("reasoning_content", "")
        reply_content = payload.get("reply_content", "")
        return PartialInferenceResult(
            reasoning_content=str(reasoning_content),
            reply_content=str(reply_content),
        )

    def _select_latest_completed_partial(self) -> PartialInferenceState | None:
        """Return the latest completed partial state."""

        completed = [
            state
            for state in self._runtime.partials.values()
            if state.result is not None
        ]
        if not completed:
            return None
        return max(completed, key=lambda state: state.partial_id)

    def _drop_partials(self) -> None:
        """Drop all cached partial states."""

        self._runtime.partials.clear()

    async def _run_partial_inference(
        self,
        *,
        history_text: str | None,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
        partial_text: str,
    ) -> PartialInferenceResult:
        """Run one slow-model partial inference to completion."""

        messages = self._build_partial_messages(
            history_text=history_text,
            latest_partial_asr=latest_partial_asr,
            latest_partial_reply=latest_partial_reply,
            partial_text=partial_text,
        )
        chunks: list[str] = []
        async for chunk in self.slow_model.astream(messages):
            text = str(chunk.content or "")
            if text:
                chunks.append(text)
        return self._parse_partial_result("".join(chunks))

    async def _stream_final_response(
        self,
        *,
        history_text: str | None,
        final_text: str,
        latest_partial_asr: str | None,
        latest_partial_reply: str | None,
    ) -> AsyncIterator[str]:
        """Stream the fast-model final reply text for one final ASR."""

        messages = self._build_final_messages(
            history_text=history_text,
            final_text=final_text,
            latest_partial_asr=latest_partial_asr,
            latest_partial_reply=latest_partial_reply,
        )
        async for chunk in self.fast_model.astream(messages):
            text = str(chunk.content or "")
            if text:
                yield text

    async def _handle_asr_partial(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[AgentOutput]:
        """Handle one partial ASR update."""

        if False:
            yield ""

        partial_text = str(payload.get("text", "") or "").strip()
        if not partial_text:
            return
        history_text = self.get_chat_history(with_system=True)

        async with self._state_lock:
            latest_partial_context = self._get_latest_completed_partial_context()
            latest_partial_asr = None
            latest_partial_reply = None
            if latest_partial_context is not None:
                latest_partial_asr, latest_partial_reply = latest_partial_context
            partial_id = self._runtime.next_partial_id
            self._runtime.next_partial_id += 1
            state = PartialInferenceState(
                partial_id=partial_id,
                text=partial_text,
            )
            state.task = asyncio.create_task(
                self._run_partial_inference(
                    history_text=history_text,
                    latest_partial_asr=latest_partial_asr,
                    latest_partial_reply=latest_partial_reply,
                    partial_text=partial_text,
                )
            )
            self._runtime.partials[partial_id] = state

        task = state.task
        if task is None:
            return
        result = await task

        async with self._state_lock:
            current_state = self._runtime.partials.get(partial_id)
            if current_state is None:
                return
            current_state.result = result

    async def _handle_asr_final(
        self,
        payload: dict[str, Any],
    ) -> AsyncIterator[AgentOutput]:
        """Handle one final ASR update."""

        final_text = str(payload.get("text", "") or "").strip()
        if not final_text:
            return
        async with self._state_lock:
            latest_partial = self._select_latest_completed_partial()

        history_text = self.get_chat_history(with_system=True)
        latest_partial_asr = latest_partial.text if latest_partial else None
        latest_partial_reply = (
            latest_partial.result.reply_content if latest_partial else None
        )
        chunks: list[str] = []
        async for text in self._stream_final_response(
            history_text=history_text,
            final_text=final_text,
            latest_partial_asr=latest_partial_asr,
            latest_partial_reply=latest_partial_reply,
        ):
            chunks.append(text)
            yield text

        reply_text = "".join(chunks)
        async with self._state_lock:
            self._append_final_reply(final_text, reply_text)
            self._drop_partials()

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Accept an incremental context update.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving-layer events.

        Returns
        -------
        Iterable[AgentOutput]
            Streamed response items produced for the accepted context.
        """

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Asynchronously accept an incremental context update.

        Parameters
        ----------
        context : AgentContext
            Context payload forwarded from serving-layer events.

        Yields
        ------
        AgentOutput
            Streamed response items produced for the accepted context.
        """

        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return
        context_type = str(context.get("type", "") or "")
        if context_type == "asr_partial":
            async for item in self._handle_asr_partial(payload):
                yield item
            return
        if context_type == "asr_final":
            async for item in self._handle_asr_final(payload):
                yield item
            return
        return

    def clone(self) -> "Agent":
        """Clone the agent for a new session.

        Returns
        -------
        Agent
            Session-safe LTS agent instance.
        """

        return type(self)(
            slow_model=self.slow_model,
            fast_model=self.fast_model,
            system_prompt=self.system_prompt,
        )

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore persisted conversation messages into agent state.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Persisted chat messages ordered by session history.
        """

        restored: list[BaseMessage] = []
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                restored.append(HumanMessage(content=content))
            elif role == "assistant":
                restored.append(AIMessage(content=content))

        self._session.messages = restored
        self._set_system_prompt()

    def get_chat_history(self, with_system: bool = False) -> str | None:
        """Return the serialized conversation history when available.

        Parameters
        ----------
        with_system : bool, optional
            Whether the serialized history should include system messages.

        Returns
        -------
        str | None
            Serialized history when implemented.
        """

        try:
            return _format_chat_history(self._session.messages, with_system=with_system)
        except Exception as exc:
            logger.warning("Failed to build chat history: %s", exc)
            return None

    def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None:
        """Attach tools to the agent.

        Parameters
        ----------
        tools : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or factories to attach.
        """

        del tools
        return None
