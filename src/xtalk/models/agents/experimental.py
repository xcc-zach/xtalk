from .interfaces import (
    Agent,
    AgentContext,
    AgentOutput,
    AgentTurnBoundary,
    ChatHistory,
)
from .tools.utils import build_tool_call_result
from .tools import (
    AsyncTool,
    SyncTool,
    Tool,
    ToolEngine,
)
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolCall,
    ToolMessage
)
from langchain_core.tools import BaseTool
from typing import Any, Iterable, AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import json
from uuid import uuid4
from ..registry import model


@dataclass
class TextCollector:
    parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(self.parts)


@model
class ExperimentalAgent(Agent):
    BASE_SYSTEM_PROMPT = "你的回复应贴近日常对话，保持简要但信息丰富。你的回复不能出现TTS无法合成的内容，例如* - （） ()。也不要有序号列表，例如1. **；采用口语化的方式表述分点的内容。"
    GREETING_GEN_PROMPT = "根据以下角色设定/角色设定，生成一句该角色可能会发出的问候语。角色设定/系统提示："
    BACKCHANNEL_JUDGE_PROMPT = """
    附和规则：
    1. 如果用户当前语义片段还不完整，不进行附和
    2. 如果用户整体内容还未完整表达，但当前小句/意群已经语义完整，可以进行轻量附和
    3. 用户当前语义片段完整后，如果符合以下情况之一，则进行附和，写入附和类型并从该类型的适用附和词中选择合适的附和词；否则不进行附和：

    附和类型：维持对话流畅  触发条件：对方在讲故事、叙述经历、描述背景、解释过程；整体内容尚未结束，但当前小句/意群已经完整；需要用轻量反馈表示“我在听”  适用的附和词：嗯、嗯嗯、哦
    附和类型：表达共鸣  触发条件：对方在表达情绪、感受、态度、偏好等主观内容；当前语义片段完整；需要表达情绪上的理解或共鸣  适用的附和词：嗯嗯、对、对对、确实
    附和类型：理解确认  触发条件：对方在解释概念、说明流程、描述机制或传递信息；当前语义片段完整；需要表示“我理解了/我跟上了”  适用的附和词：嗯、嗯嗯、哦、是的
    附和类型：轻度认同  触发条件：对方表达观点、判断、结论或评价；当前语义片段完整；需要表示轻微认可，但不展开新内容  适用的附和词：对、对对、是的、没错、确实
    附和类型：鼓励继续  触发条件：对方正在讲述较长内容，当前语义片段完整但明显还有后续；需要鼓励对方继续表达  适用的附和词：嗯、嗯嗯、哦
    附和类型：惊讶兴趣  触发条件：对方表达意外、新奇、反常、夸张或值得关注的信息；当前语义片段完整；需要表达惊讶、兴趣或继续关注  适用的附和词：哦、嗯嗯
    附和类型：安抚支持  触发条件：对方表达压力、焦虑、困惑、委屈、挫败等负面情绪；当前语义片段完整；需要表达理解和支持  适用的附和词：嗯嗯、确实
    附和类型：接收确认  触发条件：对方提出请求、指令、安排、修改意见或约束条件；当前语义完整；需要表示已经接收该要求  适用的附和词：好的、嗯嗯、是的

    可选附和词：__BACKCHANNEL_OPTIONS__

    根据以下对话历史、用户输入与以上规则，判断是否要进行附和，附和的类型是什么，以及附和的内容是什么；仅返回JSON，格式如下：
    {"reasoning_content": str, "should_backchannel": bool, "backchannel_type": Optional[附和类型], "backchannel_content": Optional[str]}。

    对话历史：__CHAT_HISTORY__
    本轮已经附和过的内容：__ALREADY_BACKCHANNELED_TEXT__
    待判断附和的用户输入：__USER_INPUT__"""

    def __init__(
        self,
        model: BaseChatModel | dict[str, Any],
        backchannel_model: BaseChatModel | dict[str, Any] | None = None,
        backchannel_source_dir: str | Path | None = None,
        tools: list[Tool | Callable[[], Tool]] | None = None,
        system_prompt: str = "",
        proactive: bool = True,
    ) -> None:
        """Initialize the experimental agent.

        Parameters
        ----------
        model : BaseChatModel | dict[str, Any]
            Primary chat model or configuration.
        backchannel_model : BaseChatModel | dict[str, Any] | None, optional
            Optional model used to judge backchannel insertion.
        backchannel_source_dir : str | Path | None, optional
            Directory containing backchannel assets.
        tools : list[Tool | Callable[[], Tool]] | None, optional
            Optional tool instances or factories.
        system_prompt : str, optional
            Additional system prompt appended after the base prompt.
        proactive : bool, optional
            Whether the agent should proactively emit the startup greeting on
            the session loop.
        """

        self.model = model if isinstance(model, BaseChatModel) else ChatOpenAI(**model)
        self.backchannel_model = (
            backchannel_model
            if isinstance(backchannel_model, BaseChatModel)
            else ChatOpenAI(**backchannel_model)
            if backchannel_model is not None
            else None
        )
        self._additional_system_prompt = system_prompt
        self.proactive = proactive
        self.system_prompt = self.BASE_SYSTEM_PROMPT + system_prompt
        self._chat_history = ChatHistory(system_prompt=self.system_prompt)
        self.backchannel_source_dir = (
            backchannel_source_dir
            if isinstance(backchannel_source_dir, Path) or backchannel_source_dir is None
            else Path(backchannel_source_dir)
        )

        self._base_tools = tools
        self.tools = self._init_tools(tools or [])
        self.tool_engine = ToolEngine(
            tools=self.tools,
            state={},
        )
        self.model_with_tools = self.tool_engine.bind(self.model)
        self._async_tool_update_queue: asyncio.Queue[AgentOutput] = asyncio.Queue()
        self.tool_engine.on_async_tool_update(self._on_async_tool_update)

        # every time remove this prefix before judging backchannel
        self._already_backchanneled_text = ""
        # record backchannel prefix+backchannel response each turn
        self._turn_already_to_backchannel_response = {}
        # mark ASRFinal generating
        self._asr_final_response_generating = False

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        yield from self.sync_iter_from_async(self.async_accept(context))

    @property
    def messages(self) -> list[BaseMessage]:
        """Return the current chat history for prompting and inspection."""

        return self._chat_history.messages

    async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
        context_type = context["type"]
        context_data = context["data"]
        # dispatch context
        if context_type == "loop":
            async for item in self._loop_runner():
                yield item
        if context_type == "asr_final":
            self._asr_final_response_generating = True
            async for item in self._handle_asr_final(context_data["text"]):
                yield item
            self._asr_final_response_generating = False
        if context_type == "asr_partial":
            async for item in self._handle_asr_partial(context_data["text"]):
                yield item
        if context_type == "response_update":
            self._handle_response_update(context_data["text"])
        if context_type == "response_finish":
            self._handle_response_finish(context_data["text"])

    def clone(self) -> "ExperimentalAgent":
        return ExperimentalAgent(
            model=self.model,
            backchannel_model=self.backchannel_model,
            backchannel_source_dir=self.backchannel_source_dir,
            system_prompt=self._additional_system_prompt,
            tools=self._base_tools,
            proactive=self.proactive,
        )

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        self._chat_history = ChatHistory(system_prompt=self.system_prompt)
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                self._chat_history.append_message(HumanMessage(content=content))
            elif role == "assistant":
                self._chat_history.append_or_update_ai_message(content, final=True)

    def get_chat_history(self, with_system: bool = False) -> str | None:
        if not self.messages:
            return None
        start_idx = 0 if with_system else 1
        return "\n".join(
            f"{msg.type}: {msg.content}" for msg in self.messages[start_idx:]
        )

    async def _stream_and_collect_text(
        self, stream: AsyncIterator[AgentOutput], collector: TextCollector
    ) -> AsyncIterator[str]:
        async for chunk in stream:
            if isinstance(chunk, str):
                collector.parts.append(chunk)
            yield chunk

    ###

    async def _loop_runner(self) -> AsyncIterator[AgentOutput]:
        # greeting: AI take initiative to call user on session start
        if self.proactive and len(self.messages) == 1:
            self._chat_history.append_message(HumanMessage(content="你好。"))
            collector = TextCollector()
            async for item in self._stream_and_collect_text(
                self._stream_greeting(), collector
            ):
                yield item
            yield AgentTurnBoundary()

        while True:
            item = await self._async_tool_update_queue.get()
            yield item

            while self._asr_final_response_generating:
                await asyncio.sleep(0.05)

            async for generated_item in self._stream_messages():
                yield generated_item
            yield AgentTurnBoundary()

    async def _handle_asr_final(self, asr_text: str) -> AsyncIterator[AgentOutput]:
        self._chat_history.append_message(HumanMessage(content=asr_text))
        async for item in self._stream_messages():
            yield item
        yield AgentTurnBoundary()
        # reset backchannel state
        self._already_backchanneled_text = ""
        self._turn_already_to_backchannel_response = {}

    def _handle_response_update(self, text: str) -> None:
        """Record one incremental playback-confirmed assistant update."""

        self._chat_history.append_or_update_ai_message(text, final=False)

    def _handle_response_finish(self, text: str) -> None:
        """Record one final playback-confirmed assistant update."""

        self._chat_history.append_or_update_ai_message(text, final=True)

    # backchannel
    async def _handle_asr_partial(self, asr_text: str) -> AsyncIterator[AgentOutput]:
        if self.backchannel_model is None or self.backchannel_source_dir is None:
            return
        # remove prefix from asr text to avoid repeated backchannel for the same content
        text_to_judge = asr_text[len(self._already_backchanneled_text) :]
        messages = [
            HumanMessage(
                content=self.BACKCHANNEL_JUDGE_PROMPT.replace(
                    "__BACKCHANNEL_OPTIONS__", "、".join(self._load_backchannel_texts())
                )
                .replace("__CHAT_HISTORY__", self.get_chat_history())
                .replace("__ALREADY_BACKCHANNELED_TEXT__", self._already_backchanneled_text)
                .replace("__USER_INPUT__", text_to_judge)
            )
        ]
        
        response_content = self.content_to_text(
            (await self.backchannel_model.ainvoke(messages)).content
        )
        structured_content = None
        try:
            structured_content = json.loads(response_content)
            if (
                not isinstance(structured_content, dict)
                or "should_backchannel" not in structured_content
            ):
                return
        except json.JSONDecodeError:
            return
        should_backchannel = bool(structured_content["should_backchannel"])
        if (
            not should_backchannel
            or "backchannel_content" not in structured_content
            or not structured_content["backchannel_content"]
        ):
            return
        audio_bytes = self._load_backchannel_audio(
            structured_content["backchannel_content"]
        )
        if not audio_bytes:
            return
        # to avoid conflict, do not yield if ASRFinal generation already starts
        if self._asr_final_response_generating:
            return
        # to avoid concurrent generation all produce positive result for the same prefix
        if self._turn_already_to_backchannel_response.get(self._already_backchanneled_text):
            return
        # update already backchanneled text for later to avoid repeated backchannel, and turn_already_to_backchannel_response to prevent concurrent response
        if not self._asr_final_response_generating:
            self._already_backchanneled_text = asr_text
            self._turn_already_to_backchannel_response[self._already_backchanneled_text] = structured_content["backchannel_content"]
        yield ToolCall(
            name="direct_audio",
            args={
                "audio": audio_bytes,
                "sample_rate": 48000,
            },
            id=f"direct-audio-{uuid4().hex}",
        )

    async def _stream_greeting(self) -> AsyncIterator[str]:
        prompt = self.GREETING_GEN_PROMPT + self.system_prompt
        messages = [SystemMessage(content=prompt)]
        async for chunk in self.model.astream(messages):
            text = self.content_to_text(chunk.content)
            if text:
                yield text

    async def _stream_messages(self) -> AsyncIterator[AgentOutput]:
        while True:
            response_message = AIMessage(content="")
            gathered = None
            async for chunk in self.model_with_tools.astream(self.messages):
                text = self.content_to_text(chunk.content)
                if text:
                    response_message.content += text
                    yield text
                gathered = chunk if gathered is None else gathered + chunk
            tool_calls: list[ToolCall] = (
                list(getattr(gathered, "tool_calls", []) or []) if gathered else []
            )
            if not tool_calls:
                return
            # Spoken text is tracked by playback-managed AI messages. Keep the
            # tool-call message focused on the structured tool invocation to
            # avoid duplicating the same text in chat history.
            response_message.content = ""
            response_message.tool_calls = tool_calls
            self._chat_history.append_message(response_message)
            for tool_call in tool_calls:
                yield tool_call
                try:
                    tool_result = await self.tool_engine.ainvoke_and_append(
                        tool_call,
                        self._chat_history.messages,
                    )
                except Exception as exc:
                    tool_result = ToolMessage(
                        content=f"Error invoking tool: {exc}",
                        tool_call_id=tool_call["id"],
                        name=tool_call["name"],
                    )
                    ToolEngine.append_tool_message(
                        tool_call,
                        tool_result,
                        self._chat_history.messages,
                    )
                yield build_tool_call_result(
                    tool_call=tool_call,
                    result_content=str(tool_result.content),
                )

    def _on_async_tool_update(
        self,
        tool_call: ToolCall,
        tool_message: ToolMessage,
    ) -> None:
        """Save an asynchronous tool update and notify the agent loop."""
        ToolEngine.append_tool_message(
            tool_call=tool_call,
            tool_message=tool_message,
            messages=self._chat_history.messages,
        )
        self._async_tool_update_queue.put_nowait(
            build_tool_call_result(
                tool_call=tool_call,
                result_content=str(tool_message.content),
            )
        )

    def _load_backchannel_audio(self, backchannel_content: str) -> bytes | None:
        map_path = self.backchannel_source_dir / "map.json"
        if not map_path.is_file():
            return None
        with open(map_path, "r", encoding="utf-8") as f:
            content_map: dict[str, str] = json.load(f)
        filename = content_map.get(backchannel_content)
        if not filename:
            return None
        audio_path = self.backchannel_source_dir / filename
        if not audio_path.is_file():
            return None
        try:
            return audio_path.read_bytes()
        except Exception:
            return None

    def _load_backchannel_texts(self) -> list[str]:
        map_path = self.backchannel_source_dir / "map.json"
        if not map_path.is_file():
            return []
        with open(map_path, "r", encoding="utf-8") as f:
            content_map: dict[str, str] = json.load(f)
        return list(content_map.keys())

    # Tools
    @staticmethod
    def _init_tools(
        tools: list[Tool | Callable[[], Tool]],
    ) -> list[Tool]:
        """Initialize tool instances, native tool classes, and factories."""

        initialized_tools: list[Tool] = []
        for tool_item in tools:
            if isinstance(tool_item, BaseTool):
                initialized_tools.append(tool_item)
                continue
            if (
                isinstance(tool_item, type)
                and issubclass(tool_item, (AsyncTool, SyncTool))
            ):
                initialized_tools.append(tool_item)
                continue
            if callable(tool_item):
                initialized_tool = tool_item()
                if isinstance(initialized_tool, BaseTool) or (
                    isinstance(initialized_tool, type)
                    and issubclass(initialized_tool, (SyncTool, AsyncTool))
                ):
                    initialized_tools.append(initialized_tool)
                    continue
            raise TypeError(f"Unsupported tool: {tool_item!r}")
        return initialized_tools
