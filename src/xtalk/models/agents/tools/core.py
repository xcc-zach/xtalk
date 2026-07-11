"""X-Talk 原生同步、异步工具的基础类型。"""

from __future__ import annotations

import asyncio
import inspect
import json
import types
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel


class ToolInput(BaseModel):
    """X-Talk 原生工具输入模型的基类。"""


class ToolOutput(BaseModel):
    """X-Talk 原生工具输出模型的基类。"""

    def to_content(self) -> str:
        """将结构化结果序列化为 ToolMessage 可保存的文本。"""

        return self.model_dump_json()


class _TextToolOutput(ToolOutput):
    """将外部工具结果包装为原生工具输出。"""

    content: str

    def to_content(self) -> str:
        """返回外部工具的文本结果。"""

        return self.content


@dataclass
class ToolState:
    """一次异步工具调用独享的可变状态。
    call_id 本次工具调用的唯一标识。
    metadata 供具体工具保存进度等自定义状态的容器。
    """

    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


TO = TypeVar("TO", bound=ToolOutput)


@dataclass(frozen=True)
class Running:
    """异步工具仍在运行时产生的文本更新。
    content 可写入 ToolMessage 的阶段性状态。
    """

    content: str


@dataclass(frozen=True)
class Finished(Generic[TO]):
    """异步工具完成时产生的结构化最终结果。
    content 具体工具定义的 ``ToolOutput`` 子类实例。
    """

    content: TO


ToolResult: TypeAlias = Running | Finished[TO]
ToolEngineState: TypeAlias = Any

# StopIteration 不能直接通过 asyncio Future 传播，因此用哨兵表示迭代结束。
_SENTINEL = object()
_ASYNC_TOOL_UPDATED_NAME = "async_tool_updated"
_ASYNC_TOOL_STATUS_NAME = "id_to_async_tool_status"
_SUBSCRIBE_ASYNC_TOOL_NAME = "subscribe_async_tool"
_UNSUBSCRIBE_ASYNC_TOOL_NAME = "unsubscribe_async_tool"
_STOP_ASYNC_TOOL_NAME = "stop_async_tool"


def _next_or_sentinel(iterator: Iterator[ToolResult[TO]]) -> ToolResult[TO] | object:
    """取出同步迭代器的下一项，耗尽时返回哨兵对象。

    iterator 同步工具更新迭代器。

    ToolResult[TO] 下一项工具结果，或表示迭代结束的 ``_SENTINEL``。
    """

    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


class _NativeTool(ABC):
    """同步和异步原生工具共用的内部基类。"""

    name: ClassVar[str | None] = None

    @staticmethod
    def _validate_input_type(value: Any) -> type[ToolInput]:
        """Validate and return a native tool input type."""

        if isinstance(value, type) and issubclass(value, ToolInput):
            return value
        raise TypeError("tool_input must be annotated as a ToolInput subclass")

    @staticmethod
    def _validate_output_type(value: Any) -> type[ToolOutput]:
        """Validate and return a native tool output type."""

        if isinstance(value, type) and issubclass(value, ToolOutput):
            return value
        raise TypeError("return must be annotated as a ToolOutput subclass")


class AsyncTool(_NativeTool):
    """可产生阶段性更新的长耗时工具基类。

    子类至少实现 ``emit_initial`` 和 ``emit_updates``。默认的异步方法会
    在线程中执行对应同步方法；如果底层 SDK 本身提供 async API，子类可
    覆写相应的 ``a*`` 方法，避免不必要的线程切换。
    """

    subscribe_by_default: ClassVar[bool] = False
    input_type: ClassVar[type[ToolInput]]
    state_type: ClassVar[type[ToolState]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """在具体子类创建时，从方法注解推断其三种数据类型。

        工具作者只需写方法注解，无须手动声明 ``input_type``、
        ``state_type`` 和 ``output_type``。抽象中间类会跳过推断。
        """

        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        cls.input_type, cls.state_type, cls.output_type = (
            cls._infer_types_from_method_annotations()
        )

    @classmethod
    def _infer_types_from_method_annotations(
        cls,
    ) -> tuple[type[ToolInput], type[ToolState], type[ToolOutput]]:
        """Infer native asynchronous tool types from lifecycle methods."""

        input_type = getattr(cls, "input_type", None)
        state_type = getattr(cls, "state_type", None)
        output_type = getattr(cls, "output_type", None)

        for method_name in (
            "emit_initial",
            "aemit_initial",
            "emit_updates",
            "aemit_updates",
            "status",
            "astatus",
            "stop",
            "astop",
            "subscribe",
            "asubscribe",
            "unsubscribe",
            "aunsubscribe",
        ):
            raw_method = cls.__dict__.get(method_name)
            if raw_method is None:
                continue
            # 类字典中的 classmethod 是描述符，需要取出底层函数的注解。
            method = (
                raw_method.__func__
                if isinstance(raw_method, classmethod)
                else raw_method
            )
            hints = get_type_hints(method)

            if "tool_input" in hints:
                input_type = cls._validate_input_type(hints["tool_input"])
            if "tool_state" in hints:
                state_type = cls._validate_state_type(hints["tool_state"])
            if "return" in hints:
                inferred_output_type = cls._infer_output_type(hints["return"])
                if inferred_output_type is not None:
                    output_type = inferred_output_type

        if input_type is None:
            raise TypeError(f"{cls.__name__} must annotate tool_input")
        if state_type is None:
            raise TypeError(f"{cls.__name__} must annotate tool_state")
        if output_type is None:
            raise TypeError(f"{cls.__name__} must annotate a ToolOutput return type")
        return input_type, state_type, output_type

    @staticmethod
    def _validate_state_type(value: Any) -> type[ToolState]:
        """Validate and return an asynchronous tool state type."""

        if isinstance(value, type) and issubclass(value, ToolState):
            return value
        raise TypeError("tool_state must be annotated as a ToolState subclass")

    @classmethod
    def _infer_output_type(cls, annotation: Any) -> type[ToolOutput] | None:
        """递归寻找返回注解中包含的具体 ToolOutput 类型。"""

        if isinstance(annotation, type) and issubclass(annotation, ToolOutput):
            return annotation

        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin is Finished and args:
            return cls._infer_output_type(args[0])
        if origin in {Iterator, AsyncIterator} and args:
            return cls._infer_output_type(args[0])
        if origin in {Union, types.UnionType}:
            for arg in args:
                output_type = cls._infer_output_type(arg)
                if output_type is not None:
                    return output_type
        return None

    @classmethod
    @abstractmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> Running:
        """立即生成满足 LLM 工具调用协议的首条结果。

        部分模型要求每个 tool call 后立刻出现对应的 ToolMessage，
        因此这里先返回 ``Running``，真正工作由 ``emit_updates`` 继续执行。
        """

        raise NotImplementedError

    @classmethod
    async def aemit_initial(
        cls,
        tool_call_id: str,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> Running:
        """在线程中生成首条结果，避免阻塞事件循环。"""

        return await asyncio.to_thread(
            cls.emit_initial,
            tool_call_id,
            tool_input,
            tool_state,
            global_state,
        )

    @classmethod
    @abstractmethod
    def emit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[ToolOutput]]:
        """依次产生阶段性更新，并以一个最终结果结束。"""

        raise NotImplementedError

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[ToolOutput]]:
        """把同步更新迭代器桥接成非阻塞的异步迭代器。"""

        iterator = cls.emit_updates(tool_input, tool_state, global_state)
        while True:
            item = await asyncio.to_thread(_next_or_sentinel, iterator)
            if item is _SENTINEL:
                return
            yield item

    @classmethod
    def status(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> str:
        """返回一次异步调用的最新可读状态。

        具体工具可覆写该钩子，为 LLM 的主动状态查询提供信息。默认返回
        空字符串，表示工具没有额外状态可报告。
        """

        return ""

    @classmethod
    async def astatus(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> str:
        """Query synchronous tool status without blocking the event loop."""

        return await asyncio.to_thread(cls.status, tool_input, tool_state, global_state)

    @classmethod
    def stop(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """执行一次调用被终止时所需的工具侧清理。

        默认不做任何事。持有外部任务或连接的工具应覆写它；ToolEngine
        自身取消后台 Task 的职责不属于这个钩子。
        """

    @classmethod
    async def astop(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """Run the synchronous stop hook without blocking the event loop."""

        await asyncio.to_thread(cls.stop, tool_input, tool_state, global_state)

    @classmethod
    def subscribe(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """当一次调用订阅阶段性更新时执行工具侧逻辑。

        默认不做任何事。订阅只影响过程更新，最终结果仍应由 ToolEngine
        写回消息历史。
        """

    @classmethod
    async def asubscribe(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """Run the synchronous subscribe hook without blocking the event loop."""

        await asyncio.to_thread(cls.subscribe, tool_input, tool_state, global_state)

    @classmethod
    def unsubscribe(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """当一次调用取消订阅阶段性更新时执行工具侧逻辑。

        默认不做任何事。取消订阅只抑制过程更新，最终结果仍应由
        ToolEngine 写回消息历史。
        """

    @classmethod
    async def aunsubscribe(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> None:
        """Run the synchronous unsubscribe hook without blocking the event loop."""

        await asyncio.to_thread(cls.unsubscribe, tool_input, tool_state, global_state)


class SyncTool(_NativeTool):
    """只返回一次结构化最终结果的原生工具基类。

    子类实现 ``invoke``；框架提供的 ``ainvoke`` 会自动把同步调用
    放入线程，供异步 Agent 安全调用。
    """

    input_type: ClassVar[type[ToolInput]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """在具体子类创建时，从 invoke 注解推断输入和输出类型。"""

        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        cls.input_type, cls.output_type = cls._infer_types_from_method_annotations()

    @classmethod
    def _infer_types_from_method_annotations(
        cls,
    ) -> tuple[type[ToolInput], type[ToolOutput]]:
        """Infer native synchronous tool types from ``invoke`` annotations."""

        input_type = getattr(cls, "input_type", None)
        output_type = getattr(cls, "output_type", None)
        raw_method = cls.__dict__.get("invoke")
        if raw_method is not None:
            method = (
                raw_method.__func__
                if isinstance(raw_method, classmethod)
                else raw_method
            )
            hints = get_type_hints(method)
            if "tool_input" in hints:
                input_type = cls._validate_input_type(hints["tool_input"])
            if "return" in hints:
                output_type = cls._validate_output_type(hints["return"])

        if input_type is None:
            raise TypeError(f"{cls.__name__} must annotate tool_input")
        if output_type is None:
            raise TypeError(f"{cls.__name__} must annotate a ToolOutput return type")
        return input_type, output_type

    @classmethod
    @abstractmethod
    def invoke(
        cls,
        tool_input: ToolInput,
        global_state: ToolEngineState,
    ) -> ToolOutput:
        """Execute the tool and return its final structured output."""

        raise NotImplementedError

    @classmethod
    async def ainvoke(
        cls,
        tool_input: ToolInput,
        global_state: ToolEngineState,
    ) -> ToolOutput:
        """Run the synchronous implementation without blocking the event loop."""

        return await asyncio.to_thread(cls.invoke, tool_input, global_state)


Tool: TypeAlias = BaseTool | type[SyncTool] | type[AsyncTool]


@dataclass
class ToolRun:
    """一次工具调用的结果封装。

    ``result`` 保存工具目前最新的运行结果。
    """

    result: ToolResult


@dataclass
class AsyncToolRun(ToolRun):
    """一次异步工具调用的结果封装。"""

    tool_class: type[AsyncTool]
    tool_input: ToolInput
    tool_state: ToolState
    subscribed: bool
    running: bool
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    error: BaseException | None = None


class ToolEngine:
    """管理 Agent 工具的调用和运行状态。"""

    def __init__(
        self,
        tools: list[Tool],
        state: ToolEngineState,
    ) -> None:
        """初始化工具引擎。"""
        self.tools = list(tools)
        has_async_tool = any(
            isinstance(tool_item, type) and issubclass(tool_item, AsyncTool)
            for tool_item in self.tools
        )
        if has_async_tool:
            self.tools.extend(self._create_assist_tools())
        self.state = state
        self._state_lock = asyncio.Lock()
        self._name_to_tool: dict[str, Tool] = {}
        self._id_to_tool_runs: dict[str, ToolRun] = {}
        self._closed = False
        self._runs_lock = asyncio.Lock()
        self._reserved_call_ids: set[str] = set()
        self._async_tool_update_callback: (
            Callable[[ToolCall, ToolMessage], None] | None
        ) = None

        for tool_item in self.tools:
            name = self._tool_name(tool_item)
            if name in self._name_to_tool:
                raise ValueError(f"Duplicate tool name: {name}")
            self._name_to_tool[name] = tool_item

    @staticmethod
    def _tool_name(tool: Tool) -> str:
        """获取工具的名称。"""
        if isinstance(tool, BaseTool):
            name = tool.name
        else:
            name = tool.name or tool.__name__
        if not name:
            raise ValueError(f"Tool {tool} must have a name")
        return name

    @staticmethod
    def _build_async_tool_update(
        source_call_id: str,
        result: ToolResult,
    ) -> tuple[ToolCall, ToolMessage]:
        """构建一个异步工具更新的 ToolCall 和 ToolMessage。"""

        update_call_id = (
            f"{source_call_id}:{_ASYNC_TOOL_UPDATED_NAME}:{uuid4().hex}"
        )
        tool_call = ToolCall(
            id=update_call_id,
            name=_ASYNC_TOOL_UPDATED_NAME,
            args={"source_call_id": source_call_id},
        )

        tool_message = ToolMessage(
            content=json.dumps(
                ToolEngine._tool_result_payload(result),
                ensure_ascii=False,
            ),
            tool_call_id=update_call_id,
            name=_ASYNC_TOOL_UPDATED_NAME,
        )
        return tool_call, tool_message

    @staticmethod
    def _tool_result_payload(result: ToolResult) -> dict[str, Any]:
        """将工具运行结果转换为可序列化的状态负载。"""

        if isinstance(result, Running):
            return {
                "running": True,
                "tool_output": result.content,
            }
        return {
            "running": False,
            "tool_output": result.content.to_content(),
        }

    @staticmethod
    def _append_tool_call(
        tool_call: ToolCall,
        messages: list[BaseMessage],
    ) -> None:
        """确保消息历史中存在对应的 AI 工具调用声明。"""

        call_id = str(tool_call.get("id", "") or "")
        if not call_id:
            raise ValueError("ToolCall must have an id")

        for message in reversed(messages):
            if not isinstance(message, AIMessage):
                continue
            for existing_call in message.tool_calls:
                if str(existing_call.get("id", "") or "") != call_id:
                    continue
                if (
                    existing_call.get("name") != tool_call.get("name")
                    or existing_call.get("args", {}) != tool_call.get("args", {})
                ):
                    raise ValueError(f"Conflicting ToolCall id: {call_id}")
                return

        pending_batch = ToolEngine._find_pending_tool_call_message(messages)
        if pending_batch is not None:
            pending_message, insertion_index = pending_batch
            pending_message.tool_calls.insert(insertion_index, tool_call)
            return

        messages.append(
            AIMessage(
                content="",
                tool_calls=[tool_call],
            )
        )

    @staticmethod
    def _find_pending_tool_call_message(
        messages: list[BaseMessage],
    ) -> tuple[AIMessage, int] | None:
        """查找最近的未完成 AI 工具调用批次及合成调用插入位置。"""

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage):
                continue
            if not message.tool_calls:
                return None

            trailing_messages = messages[index + 1 :]
            if not all(
                isinstance(trailing_message, ToolMessage)
                for trailing_message in trailing_messages
            ):
                return None

            answered_ids = {
                str(trailing_message.tool_call_id or "")
                for trailing_message in trailing_messages
            }
            if any(
                str(existing_call.get("id", "") or "") not in answered_ids
                for existing_call in message.tool_calls
            ):
                return message, len(trailing_messages)
            return None
        return None

    @staticmethod
    def append_tool_message(
        tool_call: ToolCall,
        tool_message: ToolMessage,
        messages: list[BaseMessage],
    ) -> None:
        """将工具调用和工具消息追加到消息历史。"""

        call_id = str(tool_call.get("id", "") or "")
        if not call_id:
            raise ValueError("ToolCall must have an id")
        if str(tool_message.tool_call_id or "") != call_id:
            raise ValueError("ToolMessage tool_call_id must match ToolCall id")
        if any(
            isinstance(message, ToolMessage)
            and str(message.tool_call_id or "") == call_id
            for message in messages
        ):
            raise ValueError(
                f"ToolMessage for this call id {call_id} already exists"
            )
        ToolEngine._append_tool_call(tool_call, messages)
        messages.append(tool_message)

    async def ainvoke_and_append(
        self,
        tool_call: ToolCall,
        messages: list[BaseMessage],
    ) -> ToolMessage:
        """异步调用工具并将结果追加到消息历史。"""

        tool_message = await self.ainvoke(tool_call)
        self.append_tool_message(tool_call, tool_message, messages)
        return tool_message

    def invoke_and_append(
        self,
        tool_call: ToolCall,
        messages: list[BaseMessage],
    ) -> ToolMessage:
        """同步调用工具并将结果追加到消息历史。"""

        tool_message = self.invoke(tool_call)
        self.append_tool_message(tool_call, tool_message, messages)
        return tool_message

    @classmethod
    def _to_bindable_tool(cls, tool: Tool) -> BaseTool:
        """将工具转换为聊天模型可绑定的 LangChain 工具。"""

        if isinstance(tool, BaseTool):
            return tool

        name = cls._tool_name(tool)

        def invoke_through_engine(**kwargs: Any) -> str:
            """阻止绕过 ToolEngine 直接执行原生工具。"""

            del kwargs
            raise RuntimeError(
                f"Native tool {name} must be invoked through ToolEngine"
            )

        return StructuredTool.from_function(
            func=invoke_through_engine,
            name=name,
            description=inspect.getdoc(tool) or f"Call the {name} tool.",
            args_schema=tool.input_type,
        )

    def bind(
        self,
        model: BaseChatModel,
    ) -> BaseChatModel | Runnable[Any, Any]:
        """将引擎管理的工具绑定到聊天模型。"""

        if not self.tools:
            return model
        bindable_tools = [self._to_bindable_tool(tool) for tool in self.tools]
        return model.bind_tools(bindable_tools)

    def on_async_tool_update(
        self,
        callback: Callable[[ToolCall, ToolMessage], None],
    ) -> None:
        """注册异步工具主动更新回调。"""

        self._async_tool_update_callback = callback

    def _create_async_tool_updated_tool(self) -> BaseTool:
        """创建用于接收异步工具更新的系统工具。"""

        @tool(_ASYNC_TOOL_UPDATED_NAME)
        def async_tool_updated(source_call_id: str) -> str:
            """Receive an update from an asynchronous tool.

            This tool is called only by the system. Never call it directly.
            Report any unreported update in the next response. If a new user
            message is also present, briefly report the update before replying
            to the user.

            Parameters
            ----------
            source_call_id : str
                The ID returned by the original asynchronous tool call.
            """

            del source_call_id
            return "This tool can only be invoked by ToolEngine."

        return async_tool_updated

    def _create_async_tool_status_tool(self) -> BaseTool:
        """创建用于查询异步工具状态的工具。"""

        @tool(_ASYNC_TOOL_STATUS_NAME)
        async def id_to_async_tool_status(source_call_id: str) -> str:
            """Return the latest status of an asynchronous tool call.

            Parameters
            ----------
            source_call_id : str
                The ID returned by the original asynchronous tool call.
            """

            async with self._runs_lock:
                run = self._id_to_tool_runs.get(source_call_id)

            if not isinstance(run, AsyncToolRun):
                return self._async_run_not_found_content(source_call_id)
            async with self._state_lock:
                status = await run.tool_class.astatus(
                    run.tool_input,
                    run.tool_state,
                    self.state,
                )
            return json.dumps(
                {
                    "running": run.running,
                    "status": status,
                    "error": str(run.error) if run.error is not None else None,
                },
                ensure_ascii=False,
            )
        return id_to_async_tool_status

    def _create_subscribe_tool(self) -> BaseTool:
        """创建用于订阅异步工具更新的工具。"""

        @tool(_SUBSCRIBE_ASYNC_TOOL_NAME)
        async def subscribe_async_tool(source_call_id: str) -> str:
            """Subscribe to progress updates from an asynchronous tool call.

            Parameters
            ----------
            source_call_id : str
                The ID returned by the original asynchronous tool call.
            """

            run = await self._get_async_run(source_call_id)
            if run is None:
                return self._async_run_not_found_content(source_call_id)

            async with run.lifecycle_lock:
                if run.subscribed:
                    return json.dumps(
                        {
                            "subscribed": True,
                            "running": run.running,
                            "latest": self._tool_result_payload(run.result),
                        },
                        ensure_ascii=False,
                    )
                run.subscribed = True
                try:
                    async with self._state_lock:
                        await run.tool_class.asubscribe(
                            run.tool_input,
                            run.tool_state,
                            self.state,
                        )
                except BaseException:
                    run.subscribed = False
                    raise
                return json.dumps(
                    {
                        "subscribed": True,
                        "running": run.running,
                        "latest": self._tool_result_payload(run.result),
                    },
                    ensure_ascii=False,
                )
        return subscribe_async_tool

    def _create_unsubscribe_tool(self) -> BaseTool:
        """创建用于取消订阅异步工具更新的工具。"""

        @tool(_UNSUBSCRIBE_ASYNC_TOOL_NAME)
        async def unsubscribe_async_tool(source_call_id: str) -> str:
            """Stop receiving progress updates from an asynchronous tool call.

            Parameters
            ----------
            source_call_id : str
                The ID returned by the original asynchronous tool call.
            """

            run = await self._get_async_run(source_call_id)
            if run is None:
                return self._async_run_not_found_content(source_call_id)

            async with run.lifecycle_lock:
                if not run.subscribed:
                    return json.dumps(
                        {
                            "subscribed": False,
                            "running": run.running,
                        },
                        ensure_ascii=False,
                    )
                run.subscribed = False
                try:
                    async with self._state_lock:
                        await run.tool_class.aunsubscribe(
                            run.tool_input,
                            run.tool_state,
                            self.state,
                        )
                except BaseException:
                    run.subscribed = True
                    raise
                return json.dumps(
                    {
                        "subscribed": False,
                        "running": run.running,
                    },
                    ensure_ascii=False,
                )
        return unsubscribe_async_tool

    def _create_stop_tool(self) -> BaseTool:
        """创建异步工具停止工具。"""

        @tool(_STOP_ASYNC_TOOL_NAME)
        async def stop_async_tool(source_call_id: str) -> str:
            """Stop an asynchronous tool call and release its resources.

            Parameters
            ----------
            source_call_id : str
                The ID returned by the original asynchronous tool call.
            """

            run = await self._get_async_run(source_call_id)
            if run is None:
                return self._async_run_not_found_content(source_call_id)

            async with run.lifecycle_lock:
                if not run.running:
                    return json.dumps(
                        {
                            "running": False,
                            "stopped": True,
                        },
                        ensure_ascii=False,
                    )
                run.running = False
                try:
                    async with self._state_lock:
                        await run.tool_class.astop(
                            run.tool_input,
                            run.tool_state,
                            self.state,
                        )
                except BaseException as exc:
                    run.error = exc
                    raise
                finally:
                    if run.task is not None and not run.task.done():
                        run.task.cancel()
                    run.running = False

                return json.dumps(
                    {
                        "running": False,
                        "stopped": True,
                    },
                    ensure_ascii=False,
                )
        return stop_async_tool

    async def _get_async_run(self, source_call_id: str) -> AsyncToolRun | None:
        """按调用 ID 查找异步工具运行记录。"""

        async with self._runs_lock:
            run = self._id_to_tool_runs.get(source_call_id)
        return run if isinstance(run, AsyncToolRun) else None

    @staticmethod
    def _async_run_not_found_content(source_call_id: str) -> str:
        """构造异步工具调用不存在时的 JSON 结果。"""

        return json.dumps(
            {"error": f"Async tool run not found: {source_call_id}"},
            ensure_ascii=False,
        )

    def _create_assist_tools(self) -> list[BaseTool]:
        """创建异步工具协议所需的辅助工具。"""

        return [
            self._create_async_tool_updated_tool(),
            self._create_async_tool_status_tool(),
            self._create_subscribe_tool(),
            self._create_unsubscribe_tool(),
            self._create_stop_tool(),
        ]

    async def _reserve_call_id(self, call_id: str) -> None:
        """为一次工具调用保留唯一的 call_id。"""

        async with self._runs_lock:
            if self._closed:
                raise RuntimeError("ToolEngine is closed")
            if (
                call_id in self._reserved_call_ids
                or call_id in self._id_to_tool_runs
            ):
                raise ValueError(f"Duplicate tool call id: {call_id}")
            self._reserved_call_ids.add(call_id)

    async def _store_run(self, call_id: str, run: ToolRun) -> None:
        """保存工具运行记录并结束调用 ID 的预留状态。"""

        async with self._runs_lock:
            if self._closed:
                raise RuntimeError("ToolEngine is closed")
            self._id_to_tool_runs[call_id] = run
            self._reserved_call_ids.discard(call_id)

    async def _release_call_id(self, call_id: str) -> None:
        """释放一次工具调用的 call_id 预留状态。"""

        async with self._runs_lock:
            self._reserved_call_ids.discard(call_id)

    async def _start_async_run(
        self,
        call_id: str,
        run: AsyncToolRun,
    ) -> None:
        """在引擎仍开放时订阅并启动异步工具的后台更新任务。"""

        async with run.lifecycle_lock:
            async with self._runs_lock:
                if self._closed:
                    raise RuntimeError("ToolEngine is closed")
            if run.subscribed:
                async with self._state_lock:
                    await run.tool_class.asubscribe(
                        run.tool_input,
                        run.tool_state,
                        self.state,
                    )
            async with self._runs_lock:
                if self._closed:
                    raise RuntimeError("ToolEngine is closed")
                task = asyncio.create_task(self._consume_async_updates(call_id))
                task.add_done_callback(self._consume_task_exception)
                run.task = task

    async def ainvoke(self, tool_call: ToolCall) -> ToolMessage:
        """异步调用工具。"""

        if self._closed:
            raise RuntimeError("ToolEngine is closed")
        call_id = str(tool_call.get("id", "") or "")
        name = str(tool_call.get("name", "") or "")
        args = tool_call.get("args", {})

        if not call_id:
            raise ValueError("ToolCall must have an id")
        if not name:
            raise ValueError("ToolCall must have a name")
        if not isinstance(args, dict):
            raise TypeError("ToolCall args must be a dict")

        selected_tool = self._name_to_tool.get(name)
        if selected_tool is None:
            raise ValueError(f"Unknown tool: {name}")

        if isinstance(selected_tool, BaseTool):
            await self._reserve_call_id(call_id)

            try:
                result = await selected_tool.ainvoke(args)
                content = result if isinstance(result, str) else str(result)
                output = _TextToolOutput(content=content)
                await self._store_run(
                    call_id,
                    ToolRun(result=Finished(content=output)),
                )
            except BaseException:
                await self._release_call_id(call_id)
                raise

            return ToolMessage(
                content=output.to_content(),
                tool_call_id=call_id,
                name=name,
            )

        if isinstance(selected_tool, type) and issubclass(selected_tool, SyncTool):
            tool_input = selected_tool.input_type.model_validate(args)
            await self._reserve_call_id(call_id)

            try:
                async with self._state_lock:
                    output = await selected_tool.ainvoke(
                        tool_input,
                        self.state,
                    )

                finished = Finished(content=output)
                await self._store_run(
                    call_id,
                    ToolRun(result=finished),
                )
            except BaseException:
                await self._release_call_id(call_id)
                raise

            return ToolMessage(
                content=output.to_content(),
                tool_call_id=call_id,
                name=name,
            )

        # AsyncTool 的调用逻辑
        if isinstance(selected_tool, type) and issubclass(selected_tool, AsyncTool):
            tool_input = selected_tool.input_type.model_validate(args)
            tool_state = selected_tool.state_type(call_id=call_id)

            await self._reserve_call_id(call_id)

            try:
                async with self._state_lock:
                    initial = await selected_tool.aemit_initial(
                        call_id,
                        tool_input,
                        tool_state,
                        self.state,
                    )

                if not isinstance(initial, Running):
                    raise TypeError("AsyncTool.aemit_initial() must return Running")
                if (
                    not isinstance(initial.content, str)
                    or call_id not in initial.content
                ):
                    raise ValueError(
                        "AsyncTool initial content must contain its tool call id"
                    )

                run = AsyncToolRun(
                    result=initial,
                    tool_class=selected_tool,
                    tool_input=tool_input,
                    tool_state=tool_state,
                    subscribed=selected_tool.subscribe_by_default,
                    running=True,
                )
                await self._store_run(call_id, run)
            except BaseException:
                await self._release_call_id(call_id)
                raise

            try:
                await self._start_async_run(call_id, run)
            except BaseException as exc:
                run.error = exc
                run.running = False
                raise

            return ToolMessage(
                content=initial.content,
                tool_call_id=call_id,
                name=name,
            )

        raise NotImplementedError("Native tool invocation is not implemented yet")

    def invoke(self, tool_call: ToolCall) -> ToolMessage:
        """从同步上下文调用一个同步工具。

        异步工具需要持续运行的事件循环来消费后续更新，因此必须通过
        ``ainvoke`` 调用。已有事件循环的调用方也应直接等待 ``ainvoke``。
        """

        name = str(tool_call.get("name", "") or "")
        if not name:
            raise ValueError("ToolCall must have a name")

        selected_tool = self._name_to_tool.get(name)
        if selected_tool is None:
            raise ValueError(f"Unknown tool: {name}")
        if isinstance(selected_tool, type) and issubclass(
            selected_tool,
            AsyncTool,
        ):
            raise RuntimeError(
                "AsyncTool must be invoked with await ToolEngine.ainvoke()"
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(tool_call))
        raise RuntimeError(
            "ToolEngine.invoke() cannot run inside an active event loop; "
            "use await ToolEngine.ainvoke()"
        )

    async def _consume_async_updates(
        self,
        call_id: str,
    ) -> None:
        """持续消费一次异步工具调用的更新。"""

        run = self._id_to_tool_runs.get(call_id)
        if not isinstance(run, AsyncToolRun):
            raise RuntimeError(f"Async tool run not found: {call_id}")

        updates = run.tool_class.aemit_updates(
            run.tool_input,
            run.tool_state,
            self.state,
        )
        try:
            async for result in updates:
                if not isinstance(result, (Running, Finished)):
                    raise TypeError(
                        "AsyncTool.aemit_updates() must yield Running or Finished"
                    )
                if (
                    isinstance(result, Finished)
                    and not isinstance(result.content, run.tool_class.output_type)
                ):
                    raise TypeError(
                        "AsyncTool Finished content must match its output_type"
                    )

                run.result = result

                should_notify = (
                    run.subscribed
                    or isinstance(result, Finished)
                )
                callback = self._async_tool_update_callback
                if should_notify and callback is not None:
                    tool_call, tool_message = self._build_async_tool_update(
                        call_id,
                        result,
                    )
                    callback(tool_call, tool_message)

                if isinstance(result, Finished):
                    return

            raise RuntimeError(
                f"Async tool {call_id} ended without a Finished result"
            )
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            run.error = exc
            raise
        finally:
            run.running = False

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[None]) -> None:
        """读取后台任务异常，避免产生未处理任务警告。"""

        if not task.cancelled():
            task.exception()

    async def shutdown(self) -> None:
        """关闭工具引擎并取消所有仍在运行的异步工具调用。"""

        async with self._runs_lock:
            if self._closed:
                return
            self._closed = True
            runs = [
                run
                for run in self._id_to_tool_runs.values()
                if isinstance(run, AsyncToolRun)
            ]

        async def stop_run(run: AsyncToolRun) -> None:
            async with run.lifecycle_lock:
                if not run.running:
                    return
                try:
                    async with self._state_lock:
                        await run.tool_class.astop(
                            run.tool_input,
                            run.tool_state,
                            self.state,
                        )
                except BaseException as exc:
                    run.error = exc
                finally:
                    if run.task is not None and not run.task.done():
                        run.task.cancel()
                    run.running = False

        await asyncio.gather(*(stop_run(run) for run in runs), return_exceptions=True)
        tasks = [run.task for run in runs if run.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "AsyncTool",
    "Finished",
    "Running",
    "SyncTool",
    "ToolEngineState",
    "ToolInput",
    "ToolOutput",
    "ToolResult",
    "ToolState",
    "Tool",
    "ToolRun",
    "AsyncToolRun",
    "ToolEngine",
]
