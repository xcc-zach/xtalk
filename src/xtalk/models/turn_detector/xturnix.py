"""Turn detector backed by an XTurnix model served through vLLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

import aiohttp

from ..registry import model
from .interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
    TurnDetector,
)


_MODEL_NAME = "xturnix"
_LISTENING_STATE = "<|listening|>"
_SPEAKING_STATE = "<|speaking|>"
_START_ACTION = "<|start|>"
_KEEP_ACTION = "<|keep|>"
_STOP_ACTION = "<|stop|>"
_PAUSE_TOKEN = "<|pause|>"
_PARTIAL_TERMINAL_PUNCTUATION = "。！？.!?；;…．"
_ALLOWED_ACTIONS = {
    _LISTENING_STATE: (_START_ACTION, _KEEP_ACTION),
    _SPEAKING_STATE: (_STOP_ACTION, _KEEP_ACTION),
}
_ACTION_TOKENS = (_START_ACTION, _KEEP_ACTION, _STOP_ACTION)
_SYSTEM_PROMPT = (
    "You decide when to start or stop speaking. When your state is "
    f"{_LISTENING_STATE}, you can only output {_START_ACTION} to start speaking, "
    f"or {_KEEP_ACTION} to keep listening; when your state is {_SPEAKING_STATE}, "
    f"you can output {_STOP_ACTION} to stop speaking, or {_KEEP_ACTION} to "
    "keep speaking. Output one token only. Your current state is {state}."
)
_IDLE_RESULT = TurnDetectionResult(
    action=TurnDetectionAction.DO_NOTHING,
    semantic=TurnDetectionSemantic.IDLE,
)


@dataclass
class _StreamState:
    """Track one cumulative transcript source within the current session."""

    epoch: int = 0
    text: str = ""
    pause_offsets: set[int] = field(default_factory=set)
    pending_pause_offset: int | None = None
    active: bool = False


@dataclass(frozen=True)
class _DialogueSegment:
    """Represent one chronological delta from a cumulative transcript."""

    role: Literal["user", "assistant"]
    epoch: int
    source_start: int
    source_end: int
    content: str
    is_pause: bool = False


def _longest_common_prefix(left: str, right: str) -> int:
    """Return the character length of the common prefix of two strings."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _normalize_base_url(base_url: str) -> str:
    """Normalize a vLLM URL to the server root."""
    server_url = base_url.strip().rstrip("/")
    for suffix in ("/v1/chat/completions", "/v1/models", "/v1"):
        if server_url.endswith(suffix):
            server_url = server_url[: -len(suffix)].rstrip("/")
            break

    parsed = urlsplit(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute HTTP or HTTPS URL")
    return server_url


@model(aliases=["XTurnDetector"])
class XTurnix(TurnDetector):
    """Use a vLLM-hosted XTurnix model for session-aware turn detection.

    The deployed model name is always ``xturnix``. Played assistant response
    updates are stored as cumulative session context but do not trigger remote
    inference by themselves. Each new cumulative user ASR boundary triggers
    one constrained XTurnix action prediction.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 2.0,
        max_model_len: int = 2048,
    ) -> None:
        """Initialize the XTurnix vLLM client.

        Parameters
        ----------
        base_url : str, optional
            Root URL of the vLLM OpenAI-compatible server.
        timeout : float, optional
            Total timeout in seconds for each vLLM HTTP request.
        max_model_len : int, optional
            Maximum combined prompt and generated-token length configured on
            the vLLM server.
        """
        super().__init__()
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_model_len <= 1:
            raise ValueError("max_model_len must be greater than one")

        self._base_url = _normalize_base_url(base_url)
        self._timeout = float(timeout)
        self._max_model_len = int(max_model_len)
        self._tokenize_url = f"{self._base_url}/tokenize"
        self._chat_completions_url = f"{self._base_url}/v1/chat/completions"

        self._state_lock = asyncio.Lock()
        self._inference_lock = asyncio.Lock()
        self._token_id_lock = asyncio.Lock()
        self._action_token_ids: dict[str, int] = {}

        self._user_stream = _StreamState()
        self._assistant_stream = _StreamState()
        self._segments: list[_DialogueSegment] = []
        self._user_revision = 0
        self._last_inferred_key: tuple[int, int] | None = None

        self._listening_revision = 0
        self._processed_listening_revision = 0
        self._start_latched = False
        self._stop_latched = False

    @property
    def listening(self) -> bool:
        """Return whether XTurnix is deciding when to start speaking.

        Returns
        -------
        bool
            ``True`` while waiting for a user turn to finish.
        """
        return self._listening

    @listening.setter
    def listening(self, value: bool) -> None:
        """Update the externally controlled assistant speaking state.

        Parameters
        ----------
        value : bool
            ``True`` when the assistant is listening and ``False`` while its
            response is being played.
        """
        normalized = bool(value)
        if normalized != self._listening:
            self._listening_revision += 1
        self._listening = normalized

    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Synchronously run XTurnix detection.

        Parameters
        ----------
        audio : bytes | None, optional
            Current audio frame. XTurnix is text-only and ignores this input.
        text : str | None, optional
            Cumulative ASR text for the current user stream.
        assistant_text : str | None, optional
            Cumulative assistant text confirmed as played to the user.
        speech_start : bool, optional
            Whether VAD has detected the start of a user speech segment.
        speech_pause : bool | None, optional
            Whether a user pause should be inserted at the current text offset.

        Returns
        -------
        TurnDetectionResult
            Turn action selected by XTurnix.
        """
        return asyncio.run(
            self.async_detect(
                audio=audio,
                text=text,
                assistant_text=assistant_text,
                speech_start=speech_start,
                speech_pause=speech_pause,
            )
        )

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Update dialogue context and asynchronously predict a turn action.

        Parameters
        ----------
        audio : bytes | None, optional
            Current audio frame. XTurnix is text-only and ignores this input.
        text : str | None, optional
            Cumulative ASR text for the current user stream.
        assistant_text : str | None, optional
            Cumulative assistant text confirmed as played to the user.
        speech_start : bool, optional
            Whether VAD has detected the start of a user speech segment.
        speech_pause : bool | None, optional
            Whether a user pause should be inserted at the current text offset.

        Returns
        -------
        TurnDetectionResult
            Turn action selected for the newest user-text revision. Context-only
            calls return ``DO_NOTHING`` with ``IDLE``.
        """
        del audio
        async with self._state_lock:
            self._sync_listening_state_locked()

            if speech_start:
                self._ensure_user_stream_locked()

            if assistant_text is not None:
                self._update_stream_text_locked(
                    "assistant",
                    self._assistant_stream,
                    assistant_text,
                )

            has_user_boundary = text is not None
            if text is not None:
                changed = self._update_stream_text_locked(
                    "user",
                    self._user_stream,
                    text,
                )
                if speech_pause:
                    changed = self._mark_user_pause_locked() or changed
                if changed:
                    self._user_revision += 1

        if not has_user_boundary:
            return _IDLE_RESULT
        return await self._infer_latest_user_revision()

    def clone(self) -> "XTurnix":
        """Create a session-isolated detector with the same server settings.

        Returns
        -------
        XTurnix
            New detector without dialogue history or pending action state.
        """
        return XTurnix(
            base_url=self._base_url,
            timeout=self._timeout,
            max_model_len=self._max_model_len,
        )

    def _sync_listening_state_locked(self) -> None:
        """Apply external listening transitions to cumulative stream epochs."""
        if self._processed_listening_revision == self._listening_revision:
            return

        if self._listening:
            self._stop_latched = False
        else:
            self._assistant_stream.active = False
            self._user_stream.active = False
            self._user_stream.pending_pause_offset = None
            self._start_latched = False
        self._processed_listening_revision = self._listening_revision

    @staticmethod
    def _begin_stream_locked(stream: _StreamState) -> None:
        """Reset cumulative source state while retaining rendered history."""
        stream.epoch += 1
        stream.text = ""
        stream.pause_offsets = set()
        stream.pending_pause_offset = None
        stream.active = True

    def _ensure_user_stream_locked(self) -> None:
        """Start a user stream unless the current one remains active."""
        if not self._user_stream.active:
            self._begin_stream_locked(self._user_stream)

    def _ensure_assistant_stream_locked(self) -> None:
        """Start an assistant stream unless the current one remains active."""
        if not self._assistant_stream.active:
            self._begin_stream_locked(self._assistant_stream)

    def _update_stream_text_locked(
        self,
        role: Literal["user", "assistant"],
        stream: _StreamState,
        text: str,
    ) -> bool:
        """Apply one cumulative transcript update to the chronological history."""
        if role == "user":
            self._ensure_user_stream_locked()
        else:
            self._ensure_assistant_stream_locked()

        previous = stream.text
        if text == previous:
            return False

        common_length = _longest_common_prefix(previous, text)
        self._rollback_stream_locked(role, stream.epoch, common_length)
        stream.pause_offsets = {
            offset for offset in stream.pause_offsets if offset <= common_length
        }

        if (
            role == "user"
            and stream.pending_pause_offset is not None
            and common_length < stream.pending_pause_offset
        ):
            stream.pending_pause_offset = None
        if (
            role == "user"
            and stream.pending_pause_offset is not None
            and len(text) > stream.pending_pause_offset
        ):
            pause_offset = stream.pending_pause_offset
            if pause_offset not in stream.pause_offsets:
                stream.pause_offsets.add(pause_offset)
                self._segments.append(
                    _DialogueSegment(
                        role="user",
                        epoch=stream.epoch,
                        source_start=pause_offset,
                        source_end=pause_offset,
                        content=_PAUSE_TOKEN,
                        is_pause=True,
                    )
                )
            stream.pending_pause_offset = None

        suffix = text[common_length:]
        if suffix:
            self._segments.append(
                _DialogueSegment(
                    role=role,
                    epoch=stream.epoch,
                    source_start=common_length,
                    source_end=len(text),
                    content=suffix,
                )
            )
        stream.text = text
        return True

    def _rollback_stream_locked(
        self,
        role: Literal["user", "assistant"],
        epoch: int,
        common_length: int,
    ) -> None:
        """Remove superseded deltas from one cumulative source stream."""
        retained: list[_DialogueSegment] = []
        for segment in self._segments:
            if segment.role != role or segment.epoch != epoch:
                retained.append(segment)
                continue

            if segment.is_pause:
                if segment.source_start <= common_length:
                    retained.append(segment)
                continue
            if segment.source_end <= common_length:
                retained.append(segment)
                continue
            if segment.source_start < common_length:
                retained.append(
                    replace(
                        segment,
                        source_end=common_length,
                        content=segment.content[: common_length - segment.source_start],
                    )
                )
        self._segments = retained

    def _mark_user_pause_locked(self) -> bool:
        """Remember a pause until later user text confirms continuation."""
        self._ensure_user_stream_locked()
        offset = len(self._user_stream.text)
        if offset == 0:
            return False
        if self._user_stream.pending_pause_offset == offset:
            return False
        self._user_stream.pending_pause_offset = offset
        return True

    def _render_dialogue_locked(self) -> list[dict[str, str]]:
        """Render chronological deltas as alternating chat messages."""
        messages: list[dict[str, str]] = []
        for segment in self._segments:
            if not segment.content:
                continue
            if messages and messages[-1]["role"] == segment.role:
                messages[-1]["content"] += segment.content
            else:
                messages.append({"role": segment.role, "content": segment.content})
        return messages

    def _render_model_dialogue_locked(self) -> list[dict[str, str]]:
        """Render dialogue while hiding provisional ASR sentence endings."""
        messages = self._render_dialogue_locked()
        if not self._user_stream.active:
            return messages

        for message in reversed(messages):
            if message["role"] != "user":
                continue
            normalized = message["content"].rstrip()
            normalized = normalized.rstrip(_PARTIAL_TERMINAL_PUNCTUATION).rstrip()
            if normalized:
                message["content"] = normalized
            break
        return messages

    async def _infer_latest_user_revision(self) -> TurnDetectionResult:
        """Coalesce queued partials and return only the latest valid decision."""
        async with self._inference_lock:
            while True:
                async with self._state_lock:
                    self._sync_listening_state_locked()
                    inference_key = (
                        self._user_revision,
                        self._listening_revision,
                    )
                    if inference_key == self._last_inferred_key:
                        return _IDLE_RESULT
                    dialogue = self._render_model_dialogue_locked()
                    listening = self._listening

                if not any(message["role"] == "user" for message in dialogue):
                    return _IDLE_RESULT

                state = _LISTENING_STATE if listening else _SPEAKING_STATE
                action = await self._infer_action(dialogue, state)

                async with self._state_lock:
                    self._sync_listening_state_locked()
                    latest_key = (
                        self._user_revision,
                        self._listening_revision,
                    )
                    if latest_key != inference_key:
                        continue
                    self._last_inferred_key = inference_key
                    return self._map_action_locked(action, listening)

    async def _infer_action(
        self,
        dialogue: list[dict[str, str]],
        state: str,
    ) -> str:
        """Request one constrained action from the vLLM server."""
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            action_token_ids = await self._ensure_action_token_ids(session)
            messages = [
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT.format(state=state),
                },
                *dialogue,
            ]
            messages = await self._truncate_messages(session, messages)
            allowed_actions = _ALLOWED_ACTIONS[state]
            payload = await self._post_json(
                session,
                self._chat_completions_url,
                {
                    "model": _MODEL_NAME,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 1,
                    "logprobs": True,
                    "top_logprobs": 2,
                    "allowed_token_ids": [
                        action_token_ids[action] for action in allowed_actions
                    ],
                    "skip_special_tokens": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

        action = self._parse_action(payload)
        if action not in allowed_actions:
            raise RuntimeError(
                f"vLLM returned {action!r}; expected one of {allowed_actions}"
            )
        return action

    async def _ensure_action_token_ids(
        self,
        session: aiohttp.ClientSession,
    ) -> dict[str, int]:
        """Resolve and validate the three atomic XTurnix action IDs."""
        if len(self._action_token_ids) == len(_ACTION_TOKENS):
            return self._action_token_ids

        async with self._token_id_lock:
            if len(self._action_token_ids) == len(_ACTION_TOKENS):
                return self._action_token_ids
            resolved: dict[str, int] = {}
            for action in _ACTION_TOKENS:
                payload = await self._post_json(
                    session,
                    self._tokenize_url,
                    {
                        "model": _MODEL_NAME,
                        "prompt": action,
                        "add_special_tokens": False,
                    },
                )
                tokens = self._token_ids(payload)
                if len(tokens) != 1:
                    raise RuntimeError(
                        f"vLLM must tokenize {action} as one token, got {tokens!r}"
                    )
                resolved[action] = tokens[0]
            self._action_token_ids = resolved
            return self._action_token_ids

    async def _truncate_messages(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Drop oldest dialogue messages until prompt plus one action fits."""
        if await self._messages_fit(session, messages):
            return messages

        system_message = messages[0]
        history = messages[1:]
        low = 1
        high = len(history)
        while low < high:
            dropped = (low + high) // 2
            candidate = [system_message, *history[dropped:]]
            if await self._messages_fit(session, candidate):
                high = dropped
            else:
                low = dropped + 1

        truncated = [system_message, *history[low:]]
        if not any(
            message["role"] == "user" for message in truncated[1:]
        ) or not await self._messages_fit(session, truncated):
            raise ValueError(
                "the XTurnix system prompt and newest user message exceed max_model_len"
            )
        return truncated

    async def _messages_fit(
        self,
        session: aiohttp.ClientSession,
        messages: list[dict[str, str]],
    ) -> bool:
        """Return whether chat-template tokens plus one action fit the limit."""
        payload = await self._post_json(
            session,
            self._tokenize_url,
            {
                "model": _MODEL_NAME,
                "messages": messages,
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return len(self._token_ids(payload)) + 1 <= self._max_model_len

    @staticmethod
    async def _post_json(
        session: aiohttp.ClientSession,
        url: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Post JSON and validate that the response body is an object."""
        async with session.post(url, json=body) as response:
            response.raise_for_status()
            payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"vLLM returned a non-object JSON response from {url}")
        return payload

    @staticmethod
    def _token_ids(payload: dict[str, Any]) -> list[int]:
        """Extract and validate token IDs from a vLLM tokenize response."""
        tokens = payload.get("tokens", payload.get("token_ids"))
        if not isinstance(tokens, list) or any(
            not isinstance(token, int) or isinstance(token, bool) for token in tokens
        ):
            raise RuntimeError(f"vLLM returned invalid tokenizer output: {tokens!r}")
        return tokens

    @staticmethod
    def _parse_action(payload: dict[str, Any]) -> str:
        """Extract the generated action token from a chat completion."""
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM chat completion contains no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("vLLM chat completion choice is not an object")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("vLLM chat completion contains no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("vLLM chat completion content is not text")
        return content.strip()

    def _map_action_locked(
        self,
        action: str,
        listening: bool,
    ) -> TurnDetectionResult:
        """Map one model action to XTalk action and semantic enums."""
        if listening:
            if action == _START_ACTION:
                if self._start_latched:
                    return _IDLE_RESULT
                self._start_latched = True
                return TurnDetectionResult(
                    action=TurnDetectionAction.START_GENERATION,
                    semantic=TurnDetectionSemantic.COMPLETE,
                )
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.INCOMPLETE,
            )

        if action == _STOP_ACTION:
            if self._stop_latched:
                return _IDLE_RESULT
            self._stop_latched = True
            return TurnDetectionResult(
                action=TurnDetectionAction.STOP_SPEAKING,
                semantic=TurnDetectionSemantic.INCOMPLETE,
            )
        return _IDLE_RESULT


__all__ = ["XTurnix"]
