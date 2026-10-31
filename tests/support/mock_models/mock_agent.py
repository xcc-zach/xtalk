"""Rule based agent whose replies expose its own memory."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Iterable

from xtalk.models.agents import Agent, AgentContext, AgentOutput, AgentTurnBoundary
from xtalk.models.registry import model

GREETING_REPLY = "你好，我是测试助手。"
STORY_SENTENCES = tuple(
    f"故事第{word}句。" for word in ("一", "二", "三", "四", "五", "六", "七", "八")
)
LOOKUP_PREAMBLE = "我来查一下。"
LOOKUP_TOOL_CALL = {"name": "mock_lookup", "args": {"query": "天气"}}
LOOKUP_RESULT_REPLY = "今天晴。"
FAILURE_PREAMBLE = "马上出错。"
SLOW_SENTENCES = tuple(f"慢慢说第{word}句。" for word in ("一", "二", "三", "四", "五"))
SYNTHESIS_FAILURE_REPLY = "合成失败。"
NO_ASSISTANT_HISTORY = "无"


def summary_reply(last_assistant_text: str | None, history_count: int) -> str:
    """Build the reply to ``你刚才说了什么``.

    Parameters
    ----------
    last_assistant_text : str | None
        Most recent assistant text in memory, ``None`` when absent.
    history_count : int
        Number of memory entries before the current user message.
    """
    quoted = (
        NO_ASSISTANT_HISTORY
        if not last_assistant_text
        else last_assistant_text.rstrip("。")
    )
    return f"上一句是：{quoted}。历史共{history_count}条。"


def echo_reply(text: str) -> str:
    """Build the fallback reply for unscripted text."""
    return f"收到：{text}。"


@model
class MockAgent(Agent):
    """Reply from a fixed rule table and keep a playback-confirmed history.

    Parameters
    ----------
    sentence_interval_ms : float, optional
        Delay between the sentences of the deliberately slow reply.
    """

    def __init__(self, sentence_interval_ms: float = 200.0) -> None:
        self._sentence_interval_ms = float(sentence_interval_ms)
        self._history: list[dict[str, Any]] = []

    @property
    def history(self) -> list[dict[str, Any]]:
        """Return memory entries as ``{"role", "text", "response_id"}`` dicts."""
        return list(self._history)

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
        context_type = context["type"]
        data = context["data"]
        if context_type == "asr_final":
            text = str(data.get("text") or "").strip()
            history_count = len(self._history)
            self._history.append({"role": "user", "text": text, "response_id": None})
            async for item in self._reply(text, history_count):
                yield item
            return
        if context_type in {"response_update", "response_finish"}:
            self._upsert_assistant(
                str(data.get("response_id") or ""), str(data.get("text") or "")
            )

    def _upsert_assistant(self, response_id: str, text: str) -> None:
        if not response_id:
            return
        for entry in self._history:
            if entry["role"] == "assistant" and entry["response_id"] == response_id:
                entry["text"] = text
                return
        self._history.append(
            {"role": "assistant", "text": text, "response_id": response_id}
        )

    def _last_assistant_text(self, limit: int) -> str | None:
        for entry in reversed(self._history[:limit]):
            if entry["role"] == "assistant" and entry["text"]:
                return entry["text"]
        return None

    async def _reply(self, text: str, history_count: int) -> AsyncIterator[AgentOutput]:
        if text == "你好":
            yield GREETING_REPLY
        elif text == "请讲一个很长的故事":
            for sentence in STORY_SENTENCES:
                yield sentence
        elif text == "帮我查一下天气":
            yield LOOKUP_PREAMBLE
            yield dict(LOOKUP_TOOL_CALL)
            yield AgentTurnBoundary()
            yield LOOKUP_RESULT_REPLY
        elif text == "现在报错":
            yield FAILURE_PREAMBLE
            raise RuntimeError("mock generation failure")
        elif text == "你刚才说了什么":
            yield summary_reply(self._last_assistant_text(history_count), history_count)
        elif text == "慢慢说":
            for index, sentence in enumerate(SLOW_SENTENCES):
                if index:
                    await asyncio.sleep(self._sentence_interval_ms / 1000.0)
                yield sentence
        elif text == "让合成失败":
            yield SYNTHESIS_FAILURE_REPLY
        else:
            yield echo_reply(text)

    def append_context(self, *, role: str, content: str) -> None:
        self._history.append({"role": role, "text": content, "response_id": None})

    def get_chat_history(self, with_system: bool = False) -> str | None:
        del with_system
        if not self._history:
            return None
        return "\n".join(f"{entry['role']}: {entry['text']}" for entry in self._history)

    def clone(self) -> "MockAgent":
        return MockAgent(sentence_interval_ms=self._sentence_interval_ms)
