"""Turn detector that confirms XTurnix keep decisions with timed LLM judgements."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..registry import model
from .interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
)
from .xturnix import _IDLE_RESULT, _PAUSE_TOKEN, XTurnix, _StreamState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are the turn-taking judge of a voice assistant in a live spoken conversation.
The user has gone quiet. Decide whether the assistant should start replying now.

Output "respond" only if BOTH conditions hold:
1. Finished: given the silence duration, the user has most likely finished and is waiting for the assistant.
   - A complete question, request, or statement usually needs only a short silence (about 1-2 seconds).
   - An utterance that sounds unfinished needs a longer silence, about 3 seconds or more. Examples: trailing conjunctions, fillers or hesitations such as "and", "so", "嗯…", "然后", "就是"; a cut-off clause; a list in progress.
   - If the user asked the assistant to wait or hold on (e.g. "wait", "等一下", "我想想"), the user has not finished.
2. Addressed to the assistant: the latest user speech is directed at the assistant, not at another person nearby or on the phone.
   - Signs of speech aimed at someone else: calling another person's name; talking about the assistant in the third person; side talk unrelated to the conversation; instructions or replies clearly meant for people in the room.
   - If the latest speech is unrelated to the conversation and has no sign of being aimed at the assistant, treat it as not addressed to the assistant.

Otherwise output "wait".

Additional rules:
- User text comes from streaming speech recognition and may contain recognition errors or missing punctuation. Judge by meaning, not by punctuation.
- Isolated fillers, noise-like fragments, or bare acknowledgements (e.g. "嗯", "哦", "uh") with no request are "wait".
- Assistant turns contain only the part of the reply the user actually heard; a reply may have been cut off when the user interrupted.

{output_format}"""
_OUTPUT_FORMAT_WITH_REASON = """Return exactly one JSON object and nothing else:
{"reason": "<one short sentence>", "judgement": "respond" | "wait"}"""
_OUTPUT_FORMAT_WITHOUT_REASON = """Return exactly one JSON object and nothing else:
{"judgement": "respond" | "wait"}"""
_ROLE_LABELS = {"user": "User", "assistant": "Assistant"}
_JUDGEMENT_PATTERN = re.compile(r'"judgement"\s*:\s*"(respond|wait)"')
_START_RESULT = TurnDetectionResult(
    action=TurnDetectionAction.START_GENERATION,
    semantic=TurnDetectionSemantic.COMPLETE,
)


def _message_text(message: BaseMessage) -> str:
    """Return the text parts of one chat model reply."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else str(part.get("text", ""))
        for part in content
    )


@model
class XTurnixLLM(XTurnix):
    """Extend XTurnix with timed LLM judgements while it keeps listening.

    XTurnix decides turns exactly as in :class:`XTurnix`. Whenever an accepted
    listening decision does not start a reply, a waiting window opens. While
    the window stays open, every ``llm_interval`` seconds an LLM is asked
    whether the user has finished and is addressing the assistant; the first
    ``respond`` judgement starts the reply. Audio-frame calls drive the
    schedule, so the window does not depend on VAD pause events.
    """

    def __init__(
        self,
        llm: dict[str, Any] | BaseChatModel,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 2.0,
        max_model_len: int = 2048,
        llm_interval: float = 1.0,
        llm_timeout: float = 10.0,
        llm_output_reason: bool = True,
    ) -> None:
        """Initialize the XTurnix client and the LLM judge.

        Parameters
        ----------
        llm : dict[str, Any] | BaseChatModel
            Chat model instance or ``ChatOpenAI`` configuration dict. Reasoning
            switches and request timeouts belong to this configuration.
        base_url : str, optional
            Root URL of the vLLM server hosting XTurnix.
        timeout : float, optional
            Total timeout in seconds for each XTurnix HTTP request.
        max_model_len : int, optional
            Maximum combined XTurnix prompt and generated-token length.
        llm_interval : float, optional
            Seconds between LLM requests within one waiting window, also the
            delay before the first request.
        llm_timeout : float, optional
            Seconds after the window start beyond which no new LLM request is
            sent. Requests already sent are not limited by it.
        llm_output_reason : bool, optional
            Whether the prompt asks the LLM to output a ``reason`` field.

        Raises
        ------
        ValueError
            If an LLM scheduling parameter is invalid.
        """
        super().__init__(
            base_url=base_url,
            timeout=timeout,
            max_model_len=max_model_len,
        )
        if llm_interval <= 0:
            raise ValueError("llm_interval must be positive")
        if llm_timeout <= 0:
            raise ValueError("llm_timeout must be positive")
        if not isinstance(llm_output_reason, bool):
            raise ValueError("llm_output_reason must be a bool")

        self._llm = ChatOpenAI(**llm) if isinstance(llm, dict) else llm
        self._llm_interval = float(llm_interval)
        self._llm_timeout = float(llm_timeout)
        self._llm_output_reason = llm_output_reason
        self._llm_system_prompt = _SYSTEM_PROMPT.format(
            output_format=(
                _OUTPUT_FORMAT_WITH_REASON
                if llm_output_reason
                else _OUTPUT_FORMAT_WITHOUT_REASON
            )
        )

        self._window_generation = 0
        self._window_open = False
        self._window_started_at = 0.0
        self._next_request_index = 1
        self._window_tasks: set[asyncio.Task[None]] = set()
        self._start_pending = False

    @property
    def listening(self) -> bool:
        """Return whether the detector is deciding when to start speaking.

        Returns
        -------
        bool
            ``True`` while waiting for a user turn to finish.
        """
        return self._listening

    @listening.setter
    def listening(self, value: bool) -> None:
        """Update the speaking state and close the window when speaking starts.

        Parameters
        ----------
        value : bool
            ``True`` when the assistant is listening and ``False`` while its
            response is being played.
        """
        XTurnix.listening.fset(self, value)
        if not self._listening:
            self._close_window()

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult:
        """Run XTurnix detection and advance the LLM waiting window.

        Parameters
        ----------
        audio : bytes | None, optional
            Current audio frame. Its arrival drives the LLM schedule; the
            audio content is not used.
        text : str | None, optional
            Cumulative ASR text for the current user stream.
        assistant_text : str | None, optional
            Cumulative assistant text confirmed as played to the user.
        speech_start : bool, optional
            Whether VAD has detected the start of a user speech segment.
        speech_pause : bool | None, optional
            Whether a user pause should be inserted at the current text offset.
        assistant_interrupted : bool, optional
            Whether assistant speech has just been interrupted before normal
            playback completion.

        Returns
        -------
        TurnDetectionResult
            XTurnix decision, or ``START_GENERATION`` with ``COMPLETE`` on the
            audio call that delivers an LLM ``respond`` judgement.
        """
        result = await super().async_detect(
            audio=audio,
            text=text,
            assistant_text=assistant_text,
            speech_start=speech_start,
            speech_pause=speech_pause,
            assistant_interrupted=assistant_interrupted,
        )
        if audio is None or result.action != TurnDetectionAction.DO_NOTHING:
            return result
        return self._advance_window()

    def clone(self) -> "XTurnixLLM":
        """Create a session-isolated detector sharing the same LLM client.

        Returns
        -------
        XTurnixLLM
            New detector without dialogue history, window, or pending requests.
        """
        return XTurnixLLM(
            llm=self._llm,
            base_url=self._base_url,
            timeout=self._timeout,
            max_model_len=self._max_model_len,
            llm_interval=self._llm_interval,
            llm_timeout=self._llm_timeout,
            llm_output_reason=self._llm_output_reason,
        )

    def _map_action_locked(
        self,
        action: str,
        listening: bool,
        speech_pause_seen: bool,
    ) -> TurnDetectionResult:
        """Map an accepted XTurnix action and restart the waiting window."""
        result = super()._map_action_locked(action, listening, speech_pause_seen)
        self._close_window()
        if (
            listening
            and result.action != TurnDetectionAction.START_GENERATION
            and not self._start_latched
        ):
            self._open_window()
        return result

    def _update_stream_text_locked(
        self,
        role: str,
        stream: _StreamState,
        text: str,
    ) -> bool:
        """Apply a transcript update and close the window on new user text."""
        changed = super()._update_stream_text_locked(role, stream, text)
        if changed and role == "user":
            self._close_window()
        return changed

    def _open_window(self) -> None:
        """Start a new waiting window at the current time."""
        self._window_generation += 1
        self._window_open = True
        self._window_started_at = time.monotonic()
        self._next_request_index = 1

    def _close_window(self) -> None:
        """Close the waiting window and cancel its unfinished LLM requests."""
        self._window_open = False
        self._start_pending = False
        for task in self._window_tasks:
            task.cancel()
        self._window_tasks = set()

    def _advance_window(self) -> TurnDetectionResult:
        """Deliver a pending start or send the LLM request that is due."""
        if self._start_pending:
            self._close_window()
            self._start_latched = True
            return _START_RESULT
        if not self._window_open:
            return _IDLE_RESULT

        elapsed = time.monotonic() - self._window_started_at
        due_index = int(elapsed // self._llm_interval)
        if (
            due_index >= self._next_request_index
            and due_index * self._llm_interval <= self._llm_timeout
        ):
            self._next_request_index = due_index + 1
            self._send_request(elapsed)
        return _IDLE_RESULT

    def _send_request(self, wait_seconds: float) -> None:
        """Send one background LLM request for the current window."""
        messages = [
            SystemMessage(content=self._llm_system_prompt),
            HumanMessage(content=self._render_llm_input(wait_seconds)),
        ]
        task = asyncio.create_task(
            self._request_judgement(self._window_generation, wait_seconds, messages)
        )
        self._window_tasks.add(task)
        task.add_done_callback(self._on_request_done)

    def _render_llm_input(self, wait_seconds: float) -> str:
        """Render the full dialogue and silence duration for the LLM judge."""
        lines = ["Conversation so far (oldest first):"]
        for message in self._render_dialogue_locked():
            content = message["content"].replace(_PAUSE_TOKEN, "")
            if content:
                lines.append(f"{_ROLE_LABELS[message['role']]}: {content}")
        lines.append("")
        lines.append(f"Seconds since the user's last words: {wait_seconds:.1f}")
        return "\n".join(lines)

    async def _request_judgement(
        self,
        generation: int,
        wait_seconds: float,
        messages: list[BaseMessage],
    ) -> None:
        """Ask the LLM once and mark a start when it answers ``respond``."""
        reply = _message_text(await self._llm.ainvoke(messages))
        match = _JUDGEMENT_PATTERN.search(reply)
        respond = match is not None and match.group(1) == "respond"
        current = self._window_open and generation == self._window_generation
        logger.debug(
            "XTurnixLLM judgement completed; wait seconds: %.1f; input: %r; "
            "reply: %r; respond: %s; window current: %s",
            wait_seconds,
            messages[-1].content,
            reply,
            respond,
            current,
        )
        if respond and current:
            self._start_pending = True

    def _on_request_done(self, task: asyncio.Task[None]) -> None:
        """Forget a finished request and log its failure, if any."""
        self._window_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "XTurnixLLM judgement request failed: %s",
                error,
                exc_info=error,
            )


__all__ = ["XTurnixLLM"]
