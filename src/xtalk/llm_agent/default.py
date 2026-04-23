import asyncio
from typing import (
    Iterable,
    Optional,
    List,
    Dict,
    Union,
    Tuple,
    AsyncIterator,
    Any,
    Callable,
    Coroutine,
    TypeVar,
)
from contextlib import contextmanager
from .interfaces import Agent, AgentInput, PipelineContext
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolCall,
    ToolMessage,
    BaseMessage,
)
from langchain_core.tools import BaseTool
import re
from ..log_utils import logger
from .tools import (
    build_set_voice_tool,
    build_set_emotion_tool,
    build_silence_tool,
    build_set_speed_tool,
    build_web_search_tool,
    build_time_tool,
)
from .tools.retrievers import build_local_search_tool, LOCAL_SEARCH_TOOL
from .tools.utils import build_tool_call_result_payload
from .tools.pipeline_control import (
    SET_VOICE_TOOL,
    SET_EMOTION_TOOL,
    SILENCE_TOOL,
    SET_SPEED_TOOL,
)
from collections import Counter
from langchain_openai import ChatOpenAI

T = TypeVar("T")


class DefaultAgent(Agent):
    """
    Lightweight conversational agent: it does not inject tool docs or recover
    fallbacks and only forwards LangChain-provided tool calls.
    """

    _BASE_PROMPT: str = """
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
    _CONTEXT_AWARE_PROMPT: str = ("""
You are a multimodal conversational assistant with access to:
1) Non-verbal environmental context extracted from recent audio, wrapped in <caption>...</caption>.
2) Your internal reasoning summary for the latest turn, wrapped in <thought>...</thought>.

About <caption>:
- It describes the user's environment, emotional cues, ambient sounds, and relevant non-verbal context.
- It may contain incomplete or approximate descriptions; treat it as helpful hints, not absolute truth.
- Use it only to enrich understanding and respond more naturally, not to hallucinate details that are not implied.
- DO NOT reveal <caption> content directly in your replies.

About <thought>:
- It summarizes your internal intention and reasoning for this turn.
- It represents your state before generating the answer.
- DO NOT continue thinking while composing the answer.
- DO NOT reveal the content of <thought> or mention that you have internal thoughts.
- Treat it as internal context to adjust tone, structure, and direction of your reply.

When generating your final response:
- Use both <caption> and <thought> as private hints to better understand the user's situation.
- Never output the tags themselves, nor refer to them explicitly.
- Do NOT invent nonexistent sensations, emotions, or events.
- Focus on giving a helpful, grounded, natural reply to the user's last message.
- If caption and user text conflict, ALWAYS prioritize the user's explicit message.

Caption and thought:
""").strip()

    _PRE_TOOL_FILLER_PROMPT: str = (
        """You are a helpful assistant whose only task is to generate a short, natural-sounding transitional sentence before a tool call is executed. 
Your response should sound like friendly spoken language.
Your response should be catered to the given Chat history, e.g. respond in the same language as the User.
"""
    )

    _EMBEDDING_PROCESSING_PROMPT: str = (
        """You are a helpful assistant whose only task is to generate a short, natural-sounding transitional sentence to indicate that you are aware of a Doc uploaded. 
You should mention that you are aware that the User uploaded a Doc, and mention that you are processing it.
Your response should sound like friendly spoken language.
Your response should be catered to the given Chat history, e.g. respond in the same language as the User.
"""
    )

    _EMBEDDING_FINISHED_PROMPT: str = (
        """You are a helpful assistant whose only task is to generate a short, natural-sounding transitional sentence to indicate that you have processed a Doc user just uploaded, and user can ask about it. 
Your response should sound like friendly spoken language.
You should mention the Doc summary.
Your response should be catered to the given Chat history, e.g. respond in the same language as the User.
"""
    )

    def __init__(
        self,
        model: BaseChatModel | dict,
        system_prompt: str = _BASE_PROMPT,
        voice_names: Optional[List[str]] = None,
        emotions: Optional[List[str]] = None,
        tools: Optional[List[Union[BaseTool, Callable[[], BaseTool]]]] = None,
    ):
        """Initialize the agent (with caption-aware prompt)."""
        if isinstance(model, dict):
            model = ChatOpenAI(**model)
        self.model = model
        self.voice_names = list(voice_names or [])
        self.emotions = list(emotions or [])

        self._bind_tools(tools)

        # Compose system prompt (tool descriptions are auto-injected by LangChain)
        self.sys_prompt_for_session = f"{system_prompt}\n\n{self._CONTEXT_AWARE_PROMPT}"
        self._base_system_prompt = system_prompt

        system_prompt_msg = SystemMessage(content=self.sys_prompt_for_session)
        self.session_history = [system_prompt_msg]

    @contextmanager
    def _temporary_event_loop(self):
        """Create a temporary event loop and clean it up on exit."""
        loop = asyncio.new_event_loop()
        try:
            yield loop
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            except Exception:
                pass
            loop.close()

    def _run_async_task(self, coro: Coroutine[Any, Any, T]) -> T:
        """Execute a coroutine in sync context and return the result."""
        with self._temporary_event_loop() as loop:
            return loop.run_until_complete(coro)

    def _sync_iter_from_async(self, async_iter: AsyncIterator[T]) -> Iterable[T]:
        """Convert an async iterator into a sync generator."""
        with self._temporary_event_loop() as loop:
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

    def _build_default_tool_factories(self) -> List[Callable[[], BaseTool]]:
        """Build default tool factory callbacks."""
        factories: List[Callable[[], BaseTool]] = []
        try:
            voice_names = [name for name in self.voice_names if name]
            if voice_names:
                factories.append(
                    lambda voice_names=voice_names: build_set_voice_tool(voice_names)
                )
            emotion_values = self.emotions
            if emotion_values:
                factories.append(
                    lambda emotion_values=emotion_values: build_set_emotion_tool(
                        emotion_values
                    )
                )
            factories.append(build_silence_tool)
            factories.append(build_web_search_tool)
            factories.append(build_time_tool)
            factories.append(build_set_speed_tool)
        except Exception as e:
            logger.warning(f"Failed to build tools: {e}")
        return factories

    def _normalize_tool_specs(
        self, tools: List[Union[BaseTool, Callable[[], BaseTool]]]
    ) -> List[Callable[[], BaseTool]]:
        """Normalize BaseTool or factory specs into factories."""
        factories: List[Callable[[], BaseTool]] = []
        for item in tools:
            if isinstance(item, BaseTool):
                factories.append(lambda t=item: t)
            elif callable(item):
                factories.append(item)
            else:
                logger.warning(f"Unsupported tool spec: {item!r}")
        return factories

    def _apply_tool_factories(self, factories: List[Callable[[], BaseTool]]) -> None:
        """Instantiate tools from factories and bind them to the model."""
        # Keep factories to rebuild tools on clone() and avoid sharing stateful ones
        self._tool_factories = list(factories)

        instantiated: List[BaseTool] = []
        for factory in self._tool_factories:
            try:
                tool_obj = factory()
            except Exception as e:
                logger.warning(f"Tool factory {factory!r} failed: {e}")
                continue
            if not isinstance(tool_obj, BaseTool):
                logger.warning(
                    f"Tool factory {factory!r} returned non-BaseTool: {type(tool_obj)}"
                )
                continue
            instantiated.append(tool_obj)

        self.tools = instantiated
        self._model_with_tools = (
            self.model.bind_tools(instantiated) if instantiated else self.model
        )

        # Register tools by name for quick lookups
        self._tools_map = {tool.name: tool for tool in instantiated}

    def _bind_tools(
        self, tools: Optional[List[Union[BaseTool, Callable[[], BaseTool]]]]
    ):
        # Register all tools and bind them to the model (LangChain injects docs)
        if tools is None:
            factories = self._build_default_tool_factories()
        else:
            factories = self._normalize_tool_specs(tools)
        self._apply_tool_factories(factories)

    def get_llm(self):
        return self.model

    def get_chat_history(self, with_system: bool = False):
        """Return plain-text conversation history."""
        try:
            history = getattr(self, "session_history", None)
            if not history:
                return None
            lines: List[str] = []
            for msg in history:
                role = "System"
                if isinstance(msg, HumanMessage):
                    role = "User"
                elif isinstance(msg, AIMessage):
                    role = "Assistant"
                elif isinstance(msg, SystemMessage):
                    role = "System"
                content = getattr(msg, "content", "")
                if not isinstance(content, str):
                    content = str(content)
                if role == "System" and not with_system:
                    continue
                lines.append(f"{role}: {content}")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to build chat history: {e}")
            return None

    def _extract_from_input(self, input: Union[str, AgentInput]):
        """Extract content and optional context info from user input."""
        if isinstance(input, dict):
            content = str(input.get("content", ""))
            ctx = input.get("context")
            if not isinstance(ctx, dict):
                ctx = None
            return content, ctx
        return str(input), None

    def _restore_message_content(self, marker) -> None:
        """Restore message content based on a stored marker."""
        if not marker:
            return
        idx, prev = marker
        try:
            if 0 <= idx < len(self.session_history):
                self.session_history[idx].content = prev
        except Exception:
            pass

    def _filter_response(self, response: str) -> str:
        """Remove unwanted Markdown characters from responses."""
        filtered_response = (
            response.replace("#", "")
            .replace("**", "")
            .replace("`", "")
            .replace("-", "")
        )
        filtered_response = re.sub(r"(\d+)\.", r"\1", filtered_response)
        return filtered_response

    def _compose_system_prompt(
        self, caption: Optional[str], thought: Optional[str]
    ) -> str:
        """Build system prompt augmented with caption/thought for this turn."""
        from datetime import datetime

        now = datetime.now()
        date_info = f"<current_date>{now.strftime('%Y-%m-%d %A')}</current_date>\n"
        parts: List[str] = [date_info]
        if caption and str(caption).strip():
            parts.append(f"<caption>{str(caption).strip()}</caption>")
        if thought and str(thought).strip():
            parts.append(f"<thought>{str(thought).strip()}</thought>")
        return self.sys_prompt_for_session + "\n".join(parts)

    def _update_first_system(
        self, caption: Optional[str], thought: Optional[str]
    ) -> None:
        """Overwrite the first SystemMessage with current caption/thought."""
        if not self.session_history or not isinstance(
            self.session_history[0], SystemMessage
        ):
            # Reinsert the first system message if it was lost
            self.session_history.insert(0, SystemMessage(content=""))
        self.session_history[0].content = self._compose_system_prompt(caption, thought)

    def generate(
        self, input: Union[str, AgentInput]
    ) -> Union[str, Tuple[str, List[ToolCall]]]:
        """Generate a one-shot response and forward tool calls."""
        return self._run_async_task(self.async_generate(input))

    async def _async_generate_pre_tool_call_response(
        self, tool_call: ToolCall
    ) -> AsyncIterator[str]:
        if tool_call["name"] in [
            SILENCE_TOOL,
            SET_EMOTION_TOOL,
            SET_VOICE_TOOL,
            SET_SPEED_TOOL,
        ]:
            return
        prompt_history = [
            SystemMessage(content=self._PRE_TOOL_FILLER_PROMPT),
            HumanMessage(
                content=f"Tool call name: {tool_call['name']}\nTool call arguments: {tool_call['args']}\nChat history:\n{self.get_chat_history()}"
            ),
        ]
        response = self.model.astream(prompt_history)
        async for chunk in response:
            yield chunk.content

    async def _async_invoke_tool(self, tool_call: ToolCall) -> ToolMessage:
        name = tool_call["name"]
        tool = self._tools_map.get(name)
        if not tool:
            return ToolMessage(
                content=f"Tool {name} not found", tool_call_id=tool_call["id"]
            )
        if hasattr(tool, "ainvoke"):
            return await tool.ainvoke(tool_call)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, tool.invoke, tool_call)

    def generate_stream(
        self, input: Union[str, AgentInput]
    ) -> Iterable[Union[str, ToolCall]]:
        """Stream the response while forwarding tool calls."""
        yield from self._sync_iter_from_async(self.async_generate_stream(input))

    def _prepare_context(
        self, ctx: Optional[PipelineContext]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract caption, thought, and speaker_id from context; update system prompt."""
        caption = (ctx.get("caption") if isinstance(ctx, dict) else None) or None
        thought = (ctx.get("thought") if isinstance(ctx, dict) else None) or None
        speaker_id = (ctx.get("speaker_id") if isinstance(ctx, dict) else None) or None
        self._update_first_system(caption, thought)
        return caption, thought, speaker_id

    @staticmethod
    def _summarize_doc(doc: str, max_sentences: int | None = None) -> str:
        """
        Language-agnostic rule-based extractive summary.
        """

        # === Helper: sentence split ===
        def split_sentences(text: str) -> List[str]:
            # Normalize newlines
            text = re.sub(r"[\r\n]+", "\n", text)
            # Split by punctuation (language-agnostic via symbols)
            text = re.sub(r"([.!?。！？；;:])\s*", r"\1\n", text)
            parts = [s.strip() for s in text.split("\n")]
            return [s for s in parts if s]

        # === Helper: tokenization (language-agnostic) ===
        def tokenize(sentence: str) -> List[str]:
            # Use unicode \w+ for tokenization without language assumptions
            return re.findall(r"\w+", sentence.lower())

        # =============================================
        # Main logic
        # =============================================

        if not doc:
            return ""

        text = re.sub(r"\s+", " ", doc).strip()
        if len(text) <= 80:
            return text

        sentences = split_sentences(text)
        if len(sentences) <= 2:
            return text

        tokenized = [tokenize(s) for s in sentences]
        all_tokens = [t for sent in tokenized for t in sent]

        if not all_tokens:
            k = max_sentences or min(3, len(sentences))
            return " ".join(sentences[:k])

        # Global token frequency
        freq = Counter(all_tokens)
        max_freq = max(freq.values())

        # Sentence scoring: high-frequency tokens + position
        n = len(sentences)
        scores = []
        for idx, tokens in enumerate(tokenized):
            if not tokens:
                scores.append(0.0)
                continue

            tf_score = sum(freq[t] / max_freq for t in tokens) / len(tokens)

            # Positional weight: boost start and slightly boost ending sentence
            if n > 1:
                pos_norm = 1.0 - idx / (n - 1)
            else:
                pos_norm = 1.0
            pos_score = 0.2 * pos_norm
            if idx == n - 1:
                pos_score += 0.05

            scores.append(tf_score + pos_score)

        # Default number of summary sentences
        if max_sentences is None:
            max_sentences = min(6, max(1, n // 3))

        # Choose sentences with highest scores
        top_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)[
            :max_sentences
        ]

        top_indices.sort()
        return " ".join(sentences[i] for i in top_indices)

    async def _response_to_history(self, history: List[BaseMessage], stream: bool):
        """Stream or invoke model and append the AI response to session history."""
        response = AIMessage(content="")
        if stream:
            appended_to_history = False
            async for chunk in self.model.astream(history):
                response.content += chunk.content
                yield chunk.content
                if not appended_to_history:
                    # Append the "processing" message, merging with latest AI if possible.
                    last_ai: Optional[AIMessage] = None
                    if isinstance(self.session_history[-1], AIMessage):
                        last_ai = self.session_history[-1]
                    if last_ai is not None:
                        prev = last_ai.content
                        self.session_history[-1] = response
                        response.content = prev + response.content
                    else:
                        self.session_history.append(response)
                    appended_to_history = True
        else:
            processing_msg = await self.model.ainvoke(history)
            response.content = processing_msg.content
            yield processing_msg.content
            # Append the "processing" message, merging with latest AI if possible.
            last_ai: Optional[AIMessage] = None
            if isinstance(self.session_history[-1], AIMessage):
                last_ai = self.session_history[-1]
            if last_ai is not None:
                prev = last_ai.content
                last_ai.content = f"{prev}{response.content}"
            else:
                self.session_history.append(response)

    async def _try_process_embeddings(
        self, ctx: Optional[PipelineContext], stream: bool
    ):
        """
        Return whether this method fully handled the request (True means skip the
        regular dialogue generation upstream).
        """
        status = ctx.get("embedding_status") if ctx is not None else None
        if not status or status == "idle":
            yield False
            return

        text_to_embed = ctx.get("text_to_embed") or ""

        if status == "processing":
            # Generate transitional speech acknowledging document reception
            processing_prompt_history = [
                SystemMessage(content=self._EMBEDDING_PROCESSING_PROMPT),
                HumanMessage(content=f"Chat history:\n{self.get_chat_history()}"),
            ]
            async for content in self._response_to_history(
                processing_prompt_history, stream
            ):
                yield content

        if status == "finished":
            # Register local_search tool if it is not registered
            if LOCAL_SEARCH_TOOL not in self._tools_map:
                db = ctx.get("vector_store_instance")
                local_search_tool = build_local_search_tool(db=db)
                self._tools_map[LOCAL_SEARCH_TOOL] = local_search_tool
                self.tools += [local_search_tool]
                self._model_with_tools = (
                    self.model.bind_tools(self.tools) if self.tools else self.model
                )
            # Generate completion acknowledgement so the user can ask about doc
            doc_summary = self._summarize_doc(text_to_embed)
            finished_prompt_history = [
                SystemMessage(content=self._EMBEDDING_FINISHED_PROMPT),
                HumanMessage(
                    content=f"Doc summary:\n{doc_summary}\n\nChat history:\n{self.get_chat_history()}"
                ),
            ]
            async for content in self._response_to_history(
                finished_prompt_history, stream
            ):
                yield content

        yield True

    # TODO: check implementation of tool calling and process embeddings
    async def async_generate(
        self, input: Union[str, AgentInput]
    ) -> Union[str, Tuple[str, List[ToolCall]]]:
        """Asynchronously generate a full response."""
        content, ctx = self._extract_from_input(input)
        # async for content_or_status in self._try_process_embeddings(
        #     ctx=ctx, stream=True
        # ):
        #     if isinstance(content_or_status, bool):
        #         if content_or_status:
        #             return
        #     else:
        #         yield content
        self._prepare_context(ctx)

        user_prompt = HumanMessage(content)
        self.session_history.append(user_prompt)

        response = await self._model_with_tools.ainvoke(self.session_history)

        tool_calls: List[ToolCall] = []
        try:
            for tc in getattr(response, "tool_calls", []) or []:
                if isinstance(tc, ToolCall):
                    tool_calls.append(tc)
                elif isinstance(tc, dict):
                    name = tc.get("name") or tc.get("tool")
                    args = tc.get("args") or tc.get("arguments")
                    call_id = tc.get("id") or tc.get("tool_call_id")
                    if isinstance(name, str) and isinstance(args, dict):
                        tool_calls.append(ToolCall(name=name, args=args, id=call_id))
        except Exception:
            pass

        text = self._filter_response(getattr(response, "content", ""))
        self.session_history.append(AIMessage(content=text))

        if tool_calls:
            return text, tool_calls
        return text

    async def _async_generate_stream_once(
        self,
        previous_pre_tool_call_response: Optional[str] = None,
    ) -> AsyncIterator[Union[str, ToolCall, Dict[str, Any]]]:
        new_ai_message = AIMessage(content="")
        self.session_history.append(new_ai_message)
        # Add previous filler text before new generation if needed
        if previous_pre_tool_call_response:
            new_ai_message.content += previous_pre_tool_call_response
            # TODO: enhance this logic
            self.session_history.append(HumanMessage(content="continue"))
            new_ai_message = AIMessage(content="")
            self.session_history.append(new_ai_message)
            ##
        response = self._model_with_tools.astream(self.session_history)
        gathered = None
        async for chunk in response:
            piece = self._filter_response(chunk.content)
            if not isinstance(piece, str):
                piece = str(piece)
            new_ai_message.content += piece
            yield piece
            gathered = chunk if gathered is None else gathered + chunk
            if len(chunk.tool_call_chunks) == 0:
                new_ai_message.tool_calls = gathered.tool_calls
                for tool_call in gathered.tool_calls:
                    raw_args = getattr(tool_call, "arguments", None) or getattr(
                        tool_call, "raw_arguments", None
                    )
                    yield tool_call
        tool_calls_seq = getattr(gathered, "tool_calls", []) if gathered else []
        if tool_calls_seq:
            pre_tool_call_response = ""
            for tool_call in tool_calls_seq:
                chunks: List[str] = []
                async for filler in self._async_generate_pre_tool_call_response(
                    tool_call
                ):
                    chunks.append(filler)
                    yield filler
                pre_tool_call_response += "".join(chunks)
                name = tool_call["name"]
                args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
                try:
                    tool_result = await self._async_invoke_tool(tool_call)
                except Exception as e:
                    tool_result = ToolMessage(
                        content=f"Tool {name} invocation failed: {e}",
                        tool_call_id=tool_call["id"],
                    )
                self.session_history.append(tool_result)

                payload = build_tool_call_result_payload(
                    name=str(name),
                    args=args,
                    content=str(getattr(tool_result, "content", "")),
                )
                yield payload

            async for item in self._async_generate_stream_once(pre_tool_call_response):
                yield item

    async def async_generate_stream(
        self, input: Union[str, AgentInput]
    ) -> AsyncIterator[Union[str, ToolCall]]:
        """Asynchronously stream responses while forwarding tool calls."""
        content, ctx = self._extract_from_input(input)
        async for content_or_status in self._try_process_embeddings(
            ctx=ctx, stream=True
        ):
            if isinstance(content_or_status, bool):
                if content_or_status:
                    return
            else:
                yield content_or_status
        _, _, speaker_id = self._prepare_context(ctx)
        if speaker_id:
            user_prompt = HumanMessage(
                f"The current speaker is {ctx['speaker_id']}, saying: {content}",
                name=ctx["speaker_id"],
            )
        else:
            user_prompt = HumanMessage(content)
        self.session_history.append(user_prompt)
        async for item in self._async_generate_stream_once():
            yield item

    def clone(self) -> "DefaultAgent":
        # Re-create tools from factories to avoid sharing stateful instances
        tool_factories = self._tool_factories
        return DefaultAgent(
            model=self.model,
            system_prompt=self._base_system_prompt,
            voice_names=self.voice_names,
            emotions=self.emotions,
            tools=tool_factories,
        )

    def add_tools(
        self, tools_or_factories: List[Union[BaseTool, Callable[[], BaseTool]]]
    ):
        """Add more tools/factories and rebind to the model."""
        if not tools_or_factories:
            return

        new_factories = self._normalize_tool_specs(tools_or_factories)
        base_factories: List[Callable[[], BaseTool]] = getattr(
            self, "_tool_factories", []
        )
        self._apply_tool_factories(base_factories + new_factories)
