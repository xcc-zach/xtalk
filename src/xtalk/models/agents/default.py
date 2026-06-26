from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any, AsyncIterator, Callable, Iterable, Optional

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from ...log_utils import logger
from .tools import (
    build_set_emotion_tool,
    build_set_speed_tool,
    build_set_voice_tool,
    build_silence_tool,
    build_time_tool,
    build_web_search_tool,
)
from .tools.retrievers import LOCAL_SEARCH_TOOL, build_local_search_tool
from .tools.utils import build_tool_call_result
from .interfaces import Agent, AgentContext, AgentOutput
from ..registry import model

_EMBEDDING_PROCESSING_PROMPT = """
You are a helpful assistant whose only task is to generate a short, natural-sounding transitional sentence to indicate that you are aware of a Doc uploaded.
You should mention that you are aware that the User uploaded a Doc, and mention that you are processing it.
Your response should sound like friendly spoken language.
Your response should be catered to the given Chat history, e.g. respond in the same language as the User.
"""

_EMBEDDING_FINISHED_PROMPT = """
You are a helpful assistant whose only task is to generate a short, natural-sounding transitional sentence to indicate that you have processed a Doc user just uploaded, and user can ask about it.
Your response should sound like friendly spoken language.
You should mention the Doc summary.
Your response should be catered to the given Chat history, e.g. respond in the same language as the User.
"""

_AGENT_CONTEXTS_KEY = "_agent_runtime_contexts"
@dataclass
class AgentSession:
    """Mutable per-session state for the default LLM agent.

    Parameters
    ----------
    messages : list[BaseMessage]
        Conversation history, including system, user, assistant, and tool
        messages.
    metadata : dict[str, Any]
        Session-scoped mutable state used by context updates and dynamic tools.
    """

    messages: list[BaseMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TurnState:
    """Internal turn-scoped state derived from accepted contexts."""

    speaker_id: str | None = None
    caption: str | None = None


def get_context_data(
    session: AgentSession,
    context_type: str,
) -> dict[str, Any]:
    """Return stored agent-context data for one logical context type.

    Parameters
    ----------
    session : AgentSession
        Mutable runtime session state.
    context_type : str
        Logical context stream name such as ``"caption"``.

    Returns
    -------
    dict[str, Any]
        Stored payload for the context type, or an empty dict.
    """

    raw_contexts = session.metadata.get(_AGENT_CONTEXTS_KEY)
    if not isinstance(raw_contexts, dict):
        return {}
    data = raw_contexts.get(context_type)
    if not isinstance(data, dict):
        return {}
    return dict(data)


def _format_chat_history(
    messages: list[BaseMessage],
    *,
    with_system: bool = False,
) -> str | None:
    """Render plain-text chat history from LangChain messages."""

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


class MutableToolProvider:
    """Provide a clone-safe list of tools that can be extended at runtime.

    Parameters
    ----------
    tools : list[BaseTool | Callable[[], BaseTool]] | None, optional
        Initial tool instances or factories.
    """

    def __init__(
        self,
        tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None,
    ) -> None:
        self._tool_factories = self.normalize_tool_specs(tools or [])

    def add_tools(
        self,
        tools: list[BaseTool | Callable[[], BaseTool]],
    ) -> None:
        """Append tools or tool factories.

        Parameters
        ----------
        tools : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or factories to append.
        """

        self._tool_factories.extend(self.normalize_tool_specs(tools))

    def get_tool_specs(self) -> list[Callable[[], BaseTool]]:
        """Return clone-safe tool factories.

        Returns
        -------
        list[Callable[[], BaseTool]]
            Normalized factories used by this provider.
        """

        return list(self._tool_factories)

    def get_tools(self, session: AgentSession) -> list[BaseTool]:
        """Return session-scoped tool instances for the current session.

        Parameters
        ----------
        session : AgentSession
            Runtime session state used to cache tool instances.

        Returns
        -------
        list[BaseTool]
            Tool instances reused within the current session.
        """

        cache_key = f"_mutable_tool_provider_cache_{id(self)}"
        cached_tools = session.metadata.get(cache_key)
        if isinstance(cached_tools, list) and all(
            isinstance(tool, BaseTool) for tool in cached_tools
        ):
            return cached_tools

        tools: list[BaseTool] = []
        for factory in self._tool_factories:
            try:
                tool = factory()
            except Exception as exc:
                logger.warning("Tool factory %r failed: %s", factory, exc)
                continue
            if isinstance(tool, BaseTool):
                tools.append(tool)
            else:
                logger.warning(
                    "Tool factory %r returned non-BaseTool: %s",
                    factory,
                    type(tool),
                )
        session.metadata[cache_key] = tools
        return tools

    @staticmethod
    def normalize_tool_specs(
        tools: list[BaseTool | Callable[[], BaseTool]],
    ) -> list[Callable[[], BaseTool]]:
        """Normalize tools into zero-argument factories.

        Parameters
        ----------
        tools : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or factories.

        Returns
        -------
        list[Callable[[], BaseTool]]
            Factory list.
        """

        factories: list[Callable[[], BaseTool]] = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                factories.append(lambda tool=tool: tool)
            elif callable(tool):
                factories.append(tool)
            else:
                logger.warning("Unsupported tool spec: %r", tool)
        return factories


class DefaultToolProvider(MutableToolProvider):
    """Build the default tool set for each session."""

    def __init__(
        self,
        *,
        voice_names: Optional[list[str]] = None,
        emotions: Optional[list[str]] = None,
        tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None,
    ) -> None:
        """Initialize the default tool provider.

        Parameters
        ----------
        voice_names : list[str] | None, optional
            Available voice names for the voice-switch tool.
        emotions : list[str] | None, optional
            Available emotions for the emotion tool.
        tools : list[BaseTool | Callable[[], BaseTool]] | None, optional
            Explicit tool instances or factories. When omitted, the default
            tool set is used.
        """

        self.voice_names = list(voice_names or [])
        self.emotions = list(emotions or [])
        super().__init__(
            self._build_default_tool_factories() if tools is None else tools
        )

    def get_tools(self, session: AgentSession) -> list[BaseTool]:
        """Build the tool set for the current session.

        Parameters
        ----------
        session : AgentSession
            Runtime session state.

        Returns
        -------
        list[BaseTool]
            Enabled tools for this session.
        """

        tools_by_name: dict[str, BaseTool] = {}
        for factory in self._tool_factories:
            try:
                tool = factory()
            except Exception as exc:
                logger.warning("Tool factory %r failed: %s", factory, exc)
                continue
            if not isinstance(tool, BaseTool):
                logger.warning(
                    "Tool factory %r returned non-BaseTool: %s",
                    factory,
                    type(tool),
                )
                continue
            tools_by_name[tool.name] = tool

        for name, dynamic_tool in (session.metadata.get("dynamic_tools", {}) or {}).items():
            if isinstance(dynamic_tool, BaseTool):
                tools_by_name[name] = dynamic_tool

        return list(tools_by_name.values())

    def _build_default_tool_factories(self) -> list[Callable[[], BaseTool]]:
        """Build factory callbacks for the default tool set.

        Returns
        -------
        list[Callable[[], BaseTool]]
            Tool factories.
        """

        factories: list[Callable[[], BaseTool]] = []
        try:
            voice_names = [name for name in self.voice_names if name]
            if voice_names:
                factories.append(
                    lambda voice_names=voice_names: build_set_voice_tool(voice_names)
                )
            if self.emotions:
                factories.append(
                    lambda emotions=self.emotions: build_set_emotion_tool(emotions)
                )
            factories.append(build_silence_tool)
            factories.append(build_web_search_tool)
            factories.append(build_time_tool)
            factories.append(build_set_speed_tool)
        except Exception as exc:
            logger.warning("Failed to build tools: %s", exc)
        return factories


def summarize_embedding_doc(doc: str, max_sentences: int | None = None) -> str:
    """Build a lightweight extractive summary for uploaded text.

    Parameters
    ----------
    doc : str
        Document text.
    max_sentences : int | None, optional
        Maximum number of summary sentences.

    Returns
    -------
    str
        Extractive summary.
    """

    def split_sentences(text: str) -> list[str]:
        text = re.sub(r"[\r\n]+", "\n", text)
        text = re.sub(r"([.!?。！？；;:])\s*", r"\1\n", text)
        parts = [part.strip() for part in text.split("\n")]
        return [part for part in parts if part]

    def tokenize(sentence: str) -> list[str]:
        return re.findall(r"\w+", sentence.lower())

    if not doc:
        return ""

    text = re.sub(r"\s+", " ", doc).strip()
    if len(text) <= 80:
        return text

    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return text

    tokenized = [tokenize(sentence) for sentence in sentences]
    all_tokens = [token for sentence in tokenized for token in sentence]
    if not all_tokens:
        sentence_limit = max_sentences or min(3, len(sentences))
        return " ".join(sentences[:sentence_limit])

    freq = Counter(all_tokens)
    max_freq = max(freq.values())
    scores: list[float] = []
    total_sentences = len(sentences)
    for idx, tokens in enumerate(tokenized):
        if not tokens:
            scores.append(0.0)
            continue
        tf_score = sum(freq[token] / max_freq for token in tokens) / len(tokens)
        pos_norm = 1.0 - idx / (total_sentences - 1) if total_sentences > 1 else 1.0
        pos_score = 0.2 * pos_norm + (0.05 if idx == total_sentences - 1 else 0.0)
        scores.append(tf_score + pos_score)

    sentence_limit = max_sentences or min(6, max(1, total_sentences // 3))
    selected = sorted(
        range(total_sentences),
        key=lambda idx: scores[idx],
        reverse=True,
    )[:sentence_limit]
    selected.sort()
    return " ".join(sentences[idx] for idx in selected)


@model
class DefaultAgent(Agent):
    """Default speech-first conversational agent implementation."""

    BASE_PROMPT: str = """
You are a friendly conversational partner whose response will be converted to speech using TTS. Please follow rules below:
1. Respond with the same language as user.
Examples:
- user: 你好。
- assistant: 你好呀，今天感觉怎么样？
- user: Hello.
- assistant: Hello, how are you today?

2. Your response should not contain content that cannot be synthesize by the TTS model, such as parentheses, ordered lists (starting by - ), etc. Numbers should be written in English words rather than Arabic numerals.

3. Your response should be informative and adequately detailed, but avoid unnecessary repetition or filler. Keep it suitable for spoken delivery.

4. If you find user input (ASR result) unclear, incomplete, or likely incorrect — for example:
- contains obvious ASR hallucinations,
- contains broken words or meaningless fragments,
- does not form a valid sentence,
- semantic intention cannot be determined,
then DO NOT guess the user's meaning.
Instead, politely ask the user to repeat their last utterance.

5. Each distinct speaker ID corresponds to a separate dialogue user.
The system should distinguish users based on their speaker IDs, with one user mapped to one speaker ID.

6. You have access to tools. You MUST use them proactively:
- get_time: call when user asks about current time, date, or day of week.
- web_search: you MUST default to searching for ANY question about specific facts, including but not limited to:
  * Weather, news, current events, real-time data (stock prices, sports scores, exchange rates)
  * Specific places, buildings, campuses, addresses, floor numbers, room numbers, opening hours
  * Restaurants, shops, cafes, businesses and their details (location, menu, price, how many)
  * Specific people, organizations, companies, products, events
  * Questions involving numbers, statistics, rankings, or comparisons that require accuracy
  * Any question where giving an INCORRECT answer is worse than taking a moment to search
- set_voice: call when user asks to change voice or sound like someone.
- set_speed: call when user asks to speak faster or slower.
- GOLDEN RULE: If you are not 100% certain your answer is accurate AND up-to-date, call web_search. When in doubt, ALWAYS search.
- NEVER say "I cannot access real-time information" or "I don't have internet access". You have search tools — USE THEM.
- NEVER answer specific factual questions from memory alone — search first, then answer based on search results.

7. When citing times, numbers, names, or other specific facts from search results, you SHOULD reproduce them faithfully. Do NOT reinterpret or convert values based on your assumptions. For example, if search results say "10:30", treat it as 10:30 AM unless the source explicitly says PM or evening.

8. SEARCH QUERY RULE: When constructing a web_search query, ALWAYS replace relative time references ("今天", "昨天", "明天", "上个月", "去年", "today", "yesterday", etc.) with the actual date from <current_date>. For example, if today is 2026-02-28 and the user asks "今天NBA有哪些比赛", your query should be "2026年2月28日 NBA比赛赛程", NOT "今天NBA有哪些比赛".

你是一位友好的对话伙伴，你的回复会通过 TTS 转成语音。请遵守以下规则：

1. 用和用户相同的语言回复。
示例：
- user: 你好。
- assistant: 你好呀，今天感觉怎么样？
- user: Hello.
- assistant: Hello, how are you today?

2. 你的回复中不能出现 TTS 无法合成的内容，例如括号、编号列表（以- 开始）等。数字要用英文单词书写，不要使用阿拉伯数字。

3. 你的回复应当信息充分、适当详细，但避免不必要的重复或废话。回复长度要适合语音播报。

4. 如果你发现用户输入（ASR 结果）不清晰、不完整或可能有误，例如：
- 包含明显的 ASR 幻觉内容；
- 包含残缺的词语或无意义的片段；
- 无法构成有效句子；
- 无法判断其语义意图；
那么不要猜测用户的意思。
请礼貌地请求用户重复上一句内容。
5. 有几个不同说话人id就有几个不同的对话用户，每个说话人id对应一个用户，你要根据说话人id来区分用户。

6. 你可以使用工具，必须主动调用：
- get_time：用户问当前时间、日期、星期几时调用。
- web_search：遇到任何关于具体事实的问题时，必须优先搜索，包括但不限于：
  * 天气、新闻、时事、实时数据（股价、比分、汇率等）
  * 具体地点、建筑、校园、地址、楼层、房间号、营业时间
  * 餐厅、商店、咖啡厅、商家及其详细信息（位置、菜单、价格、数量）
  * 具体人物、机构、公司、产品、事件
  * 涉及数字、统计、排名或需要准确性的比较类问题
  * 任何回答错误比多花一点时间搜索更糟糕的问题
- set_voice：用户要求换声音或模仿某人声音时调用。
- set_speed：用户要求说快一点或慢一点时调用。
- 黄金原则：如果你不能百分之百确定答案准确且是最新的，就调用 web_search。有疑问时，永远先搜索。
- 绝对不要说"我无法获取实时信息"或"我没有联网能力"。你拥有搜索工具，请使用它们。
- 绝对不要仅凭记忆回答具体的事实性问题——先搜索，再根据搜索结果回答。

7. 引用搜索结果中的时间、数字、名称等具体事实时，应该忠实于原文，不要根据自己的推测重新解读。例如搜索结果写"10:30"，应说"上午十点三十分"，除非原文明确标注是下午或晚上。

8. 搜索用语规则：构造 web_search 的 query 时，必须将"今天"、"昨天"、"明天"、"上个月"、"去年"等相对时间词替换为 <current_date> 中的具体日期。例如今天是2026-02-28，用户问"今天NBA有哪些比赛"，你的 query 应为"2026年2月28日 NBA比赛赛程"，而不是"今天NBA有哪些比赛"。
"""

    CONTEXT_AWARE_PROMPT: str = (
        """
You are a multimodal conversational assistant with access to:
1) Non-verbal environmental context extracted from recent audio, wrapped in <caption>...</caption>.

About <caption>:
- It describes the user's environment, emotional cues, ambient sounds, and relevant non-verbal context.
- It may contain incomplete or approximate descriptions; treat it as helpful hints, not absolute truth.
- Use it only to enrich understanding and respond more naturally, not to hallucinate details that are not implied.
- DO NOT reveal <caption> content directly in your replies.

When generating your final response:
- Use <caption> as a private hint to better understand the user's situation.
- Never output the tags themselves, nor refer to them explicitly.
- Do NOT invent nonexistent sensations, emotions, or events.
- Focus on giving a helpful, grounded, natural reply to the user's last message.
- If caption and user text conflict, ALWAYS prioritize the user's explicit message.

Caption:
""".strip()
    )

    def __init__(
        self,
        model: BaseChatModel | dict[str, Any],
        system_prompt: str = BASE_PROMPT,
        voice_names: Optional[list[str]] = None,
        emotions: Optional[list[str]] = None,
        tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None,
    ) -> None:
        """Initialize the default agent.

        Parameters
        ----------
        model : BaseChatModel | dict[str, Any]
            Chat model or ``ChatOpenAI`` configuration dict.
        system_prompt : str, optional
            Base system prompt.
        voice_names : list[str] | None, optional
            Available voice names.
        emotions : list[str] | None, optional
            Available emotions.
        tools : list[BaseTool | Callable[[], BaseTool]] | None, optional
            Explicit tool set or factories.
        """

        self._model = self._coerce_model(model)
        self.system_prompt = system_prompt
        self.voice_names = list(voice_names or [])
        self.emotions = list(emotions or [])
        self._tool_provider = DefaultToolProvider(
            voice_names=self.voice_names,
            emotions=self.emotions,
            tools=tools,
        )
        self._session = AgentSession()

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

    @property
    def model(self) -> BaseChatModel:
        """Return the backing model."""

        return self._model

    @model.setter
    def model(self, model: BaseChatModel) -> None:
        """Update the backing model.

        Parameters
        ----------
        model : BaseChatModel
            New backing model.
        """

        self._model = model

    @property
    def session_history(self) -> list[BaseMessage]:
        """Expose session history for compatibility."""

        return self._session.messages

    @session_history.setter
    def session_history(self, messages: list[BaseMessage]) -> None:
        """Replace session history for compatibility.

        Parameters
        ----------
        messages : list[BaseMessage]
            New session message list.
        """

        self._session.messages = messages

    def _filter_text(self, text: str) -> str:
        """Normalize one response fragment for TTS output."""

        filtered = text.replace("#", "").replace("**", "").replace("`", "").replace(
            "-",
            "",
        )
        return re.sub(r"(\d+)\.", r"\1", filtered)

    def _build_turn_state(self) -> _TurnState:
        """Build internal turn state from accepted context streams."""

        speaker_context = get_context_data(self._session, "speaker")
        caption_context = get_context_data(self._session, "caption")
        return _TurnState(
            speaker_id=speaker_context.get("speaker_id") or None,
            caption=caption_context.get("text") or None,
        )

    def _build_system_prompt(self, turn_state: _TurnState) -> str:
        """Build the system prompt for the current turn."""

        parts = [f"<current_date>{datetime.now().strftime('%Y-%m-%d %A')}</current_date>"]
        if turn_state.caption:
            parts.append(f"<caption>{turn_state.caption.strip()}</caption>")
        session_system_prompt = f"{self.system_prompt}\n\n{self.CONTEXT_AWARE_PROMPT}"
        return f"{session_system_prompt}\n" + "\n".join(parts)

    def _build_user_message(
        self,
        request: dict[str, Any],
        turn_state: _TurnState,
    ) -> str:
        """Build the user message content passed to the model."""

        content = str(request.get("content", ""))
        if turn_state.speaker_id:
            return f"The current speaker is {turn_state.speaker_id}, saying: {content}"
        return content

    def _set_system_prompt(self, turn_state: _TurnState) -> None:
        """Ensure the first session message contains the current system prompt."""

        prompt = self._build_system_prompt(turn_state)
        if self._session.messages and isinstance(self._session.messages[0], SystemMessage):
            self._session.messages[0].content = prompt
            return
        self._session.messages.insert(0, SystemMessage(content=prompt))

    def _accept_context_update(self, context: AgentContext) -> None:
        """Merge an incremental context update into the session state."""

        context_type = str(context.get("type", "") or "").strip()
        if not context_type:
            return
        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return

        raw_contexts = self._session.metadata.setdefault(_AGENT_CONTEXTS_KEY, {})
        if not isinstance(raw_contexts, dict):
            raw_contexts = {}
            self._session.metadata[_AGENT_CONTEXTS_KEY] = raw_contexts

        existing = raw_contexts.get(context_type)
        if not isinstance(existing, dict):
            existing = {}

        merged = dict(existing)
        merged.update(payload)
        raw_contexts[context_type] = merged

    def _build_runtime_request(self, context: AgentContext) -> dict[str, Any] | None:
        """Build an internal runtime request from one accepted context update."""

        context_type = str(context.get("type", "") or "")
        if context_type != "asr_final":
            return None

        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return None
        return {"content": str(payload.get("text", ""))}

    @staticmethod
    def _normalize_tool_call(tool_call: ToolCall | dict[str, Any]) -> ToolCall:
        """Convert LangChain tool-call payloads into a stable dict form."""

        if isinstance(tool_call, dict):
            name = str(tool_call.get("name") or tool_call.get("tool") or "")
            args = tool_call.get("args") or tool_call.get("arguments") or {}
            call_id = tool_call.get("id") or tool_call.get("tool_call_id")
            if isinstance(args, dict):
                return ToolCall(name=name, args=args, id=call_id)
        return tool_call

    async def _invoke_tool(
        self,
        *,
        tools_map: dict[str, BaseTool],
        tool_call: ToolCall,
    ) -> ToolMessage:
        """Invoke one tool call and normalize the result as ``ToolMessage``."""

        name = tool_call["name"]
        tool = tools_map.get(name)
        if tool is None:
            return ToolMessage(
                content=f"Tool {name} not found",
                tool_call_id=tool_call["id"],
            )
        tool_input = tool_call.get("args", {})
        try:
            if hasattr(tool, "ainvoke"):
                result = await tool.ainvoke(tool_input)
            else:
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(None, tool.invoke, tool_input)
            if isinstance(result, ToolMessage):
                return result
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        except Exception as exc:
            logger.warning("Tool invocation failed for %s: %s", name, exc)
            return ToolMessage(
                content=f"Tool {name} invocation failed: {exc}",
                tool_call_id=tool_call["id"],
            )

    async def _stream_request(
        self,
        request: dict[str, Any],
    ) -> AsyncIterator[AgentOutput]:
        """Stream response items for one runtime request."""

        turn_state = self._build_turn_state()
        self._set_system_prompt(turn_state)

        tools = self._tool_provider.get_tools(self._session)
        model_with_tools = self.model.bind_tools(tools) if tools else self.model
        tools_map = {tool.name: tool for tool in tools}

        user_content = self._build_user_message(request, turn_state)
        if turn_state.speaker_id:
            user_message = HumanMessage(content=user_content, name=turn_state.speaker_id)
        else:
            user_message = HumanMessage(content=user_content)
        self._session.messages.append(user_message)

        while True:
            response_message = AIMessage(content="")
            self._session.messages.append(response_message)

            gathered = None
            async for chunk in model_with_tools.astream(self._session.messages):
                text = self._filter_text(str(chunk.content or ""))
                if text:
                    response_message.content += text
                    yield text
                gathered = chunk if gathered is None else gathered + chunk

            tool_calls = (
                list(getattr(gathered, "tool_calls", []) or []) if gathered else []
            )
            if tool_calls:
                response_message.tool_calls = tool_calls
            if not tool_calls:
                return

            for tool_call in tool_calls:
                normalized_tool_call = self._normalize_tool_call(tool_call)
                yield ToolCall(
                    name=normalized_tool_call["name"],
                    args=dict(normalized_tool_call["args"]),
                    id=normalized_tool_call["id"],
                )
                tool_result = await self._invoke_tool(
                    tools_map=tools_map,
                    tool_call=normalized_tool_call,
                )
                self._session.messages.append(tool_result)
                result_content = str(getattr(tool_result, "content", ""))
                yield build_tool_call_result(
                    tool_call=normalized_tool_call,
                    result_content=result_content,
                )

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Accept a context update and return any triggered stream items.

        Parameters
        ----------
        context : AgentContext
            Incremental session context update.

        Returns
        -------
        Iterable[AgentOutput]
            Streamed response items triggered by the update.
        """

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Asynchronously accept a context update.

        Parameters
        ----------
        context : AgentContext
            Incremental session context update.

        Yields
        ------
        AgentOutput
            Streamed response items triggered by the update.
        """

        self._accept_context_update(context)
        context_type = str(context.get("type", "") or "")
        if context_type == "embedding":
            payload = context.get("data") or {}
            if not isinstance(payload, dict):
                return

            status = str(payload.get("status") or "")
            if status not in {"processing", "finished"}:
                return

            if status == "finished":
                self._register_embedding_search_tool(payload.get("vector_store_instance"))

            text = await self._build_embedding_direct_output(
                status=status,
                doc_text=str(payload.get("text") or ""),
            )
            if text:
                yield text
            return

        request = self._build_runtime_request(context)
        if request is None:
            return
        async for item in self._stream_request(request):
            yield item

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore persisted conversation history into the session state.

        Parameters
        ----------
        messages : list[dict[str, Any]]
            Persisted chat messages.
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
        self._set_system_prompt(self._build_turn_state())

    def get_chat_history(self, with_system: bool = False) -> str | None:
        """Render plain-text chat history."""

        try:
            return _format_chat_history(self._session.messages, with_system=with_system)
        except Exception as exc:
            logger.warning("Failed to build chat history: %s", exc)
            return None

    def clone(self) -> Agent:
        """Clone the agent with a fresh session.

        Returns
        -------
        Agent
            Session-safe cloned agent.
        """

        return type(self)(
            model=self.model,
            system_prompt=self.system_prompt,
            voice_names=list(self.voice_names),
            emotions=list(self.emotions),
            tools=self._tool_provider.get_tool_specs(),
        )

    def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None:
        """Attach additional tools to the agent.

        Parameters
        ----------
        tools : list[BaseTool | Callable[[], BaseTool]]
            Tool instances or factories.
        """

        if not tools:
            return
        self._tool_provider.add_tools(tools)

    def _register_embedding_search_tool(self, vector_store_instance: Any) -> None:
        """Register the local-search tool for uploaded documents when ready."""

        if vector_store_instance is None:
            return
        dynamic_tools = self._session.metadata.setdefault("dynamic_tools", {})
        if not isinstance(dynamic_tools, dict):
            dynamic_tools = {}
            self._session.metadata["dynamic_tools"] = dynamic_tools
        if LOCAL_SEARCH_TOOL in dynamic_tools:
            return
        try:
            dynamic_tools[LOCAL_SEARCH_TOOL] = build_local_search_tool(
                db=vector_store_instance
            )
        except Exception as exc:
            logger.warning("Failed to register local search tool: %s", exc)

    async def _build_embedding_direct_output(
        self,
        *,
        status: str,
        doc_text: str,
    ) -> str | None:
        """Build one direct-output utterance for embedding lifecycle updates."""

        history = _format_chat_history(self._session.messages) or ""
        if status == "processing":
            prompt_history = [
                SystemMessage(content=_EMBEDDING_PROCESSING_PROMPT),
                HumanMessage(content=f"Chat history:\n{history}"),
            ]
        else:
            doc_summary = summarize_embedding_doc(doc_text)
            prompt_history = [
                SystemMessage(content=_EMBEDDING_FINISHED_PROMPT),
                HumanMessage(
                    content=f"Doc summary:\n{doc_summary}\n\nChat history:\n{history}"
                ),
            ]

        try:
            response = await self.model.ainvoke(prompt_history)
            content = getattr(response, "content", "")
            text = self._filter_text(str(content or ""))
            if text:
                return text
        except Exception as exc:
            logger.warning("Failed to build embedding direct output: %s", exc)

        return self._fallback_embedding_direct_output(status=status, doc_text=doc_text)

    def _fallback_embedding_direct_output(
        self,
        *,
        status: str,
        doc_text: str,
    ) -> str:
        """Return a deterministic fallback embedding utterance."""

        history = _format_chat_history(self._session.messages) or ""
        prefers_chinese = bool(re.search(r"[\u4e00-\u9fff]", history + doc_text))
        if status == "processing":
            return (
                "我知道你上传了一份文档，我正在处理它。"
                if prefers_chinese
                else "I know you uploaded a document. I am processing it now."
            )
        return (
            "文档已经处理好了，你可以继续问我相关内容。"
            if prefers_chinese
            else "The document is ready. You can ask me about it now."
        )
