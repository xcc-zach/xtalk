# 异步工具调用

## 原则

- 是同步工具的超集；异步工具仅包含工具执行函数时变为同步工具
- 异步调用后立刻返回一条结果塞入LLM历史以兼容部分LLM tool call后紧跟ToolMessage的要求
    - 协议层不通过的情况： AIMessage(tool_calls不为空)后不接ToolMessage；ToolMessage前文所有AIMessage的tool_calls中没有对应id
- LLM可向工具查询最新结果
- LLM可关闭工具调用
- LLM可订阅/取消订阅工具调用；订阅时工具会将最新结果反馈给LLM
- 工具分为执行中和结束的结果返回

## 接口设计

```python
import asyncio
import types
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Optional, TypeAlias, TypeVar, Union, get_args, get_origin, get_type_hints

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel


class ToolInput(BaseModel):
    pass


class ToolOutput(BaseModel):
    def to_content(self) -> str:
        return self.model_dump_json()


@dataclass
class ToolState:
    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


TI = TypeVar("TI", bound=ToolInput)
TS = TypeVar("TS", bound=ToolState)
TO = TypeVar("TO", bound=ToolOutput)


@dataclass(frozen=True)
class Running:
    content: str


@dataclass(frozen=True)
class Finished(Generic[TO]):
    content: TO


ToolResult: TypeAlias = Running | Finished[TO]
ToolEngineState: TypeAlias = Any


_SENTINEL = object()


def _next_or_sentinel(iterator: Iterator[str]) -> str | object:
    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


class AsyncTool(ABC):
    name: ClassVar[Optional[str]] # 不设置时使用工具类名
    subscribe_by_default: ClassVar[bool] = False
    input_type: ClassVar[type[ToolInput]]
    state_type: ClassVar[type[ToolState]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.input_type, cls.state_type, cls.output_type = (
            cls._infer_types_from_method_annotations()
        )

    @classmethod
    def _infer_types_from_method_annotations(
        cls,
    ) -> tuple[type[ToolInput], type[ToolState], type[ToolOutput]]:
        input_type: type[ToolInput] | None = None
        state_type: type[ToolState] | None = None
        output_type: type[ToolOutput] | None = None

        for method_name in (
            "emit_initial",
            "aemit_initial",
            "emit_updates",
            "aemit_updates",
            "status",
            "stop",
            "subscribe",
            "unsubscribe",
        ):
            raw_method = cls.__dict__.get(method_name)
            if raw_method is None:
                continue
            method = raw_method.__func__ if isinstance(raw_method, classmethod) else raw_method
            hints = get_type_hints(method)

            if input_type is None and "tool_input" in hints:
                input_type = cls._validate_input_type(hints["tool_input"])
            if state_type is None and "tool_state" in hints:
                state_type = cls._validate_state_type(hints["tool_state"])
            if output_type is None and "return" in hints:
                output_type = cls._infer_output_type(hints["return"])

        if input_type is None:
            raise TypeError(f"{cls.__name__} must annotate tool_input")
        if state_type is None:
            raise TypeError(f"{cls.__name__} must annotate tool_state")
        if output_type is None:
            raise TypeError(f"{cls.__name__} must annotate a ToolOutput return type")
        return input_type, state_type, output_type

    @staticmethod
    def _validate_input_type(value: Any) -> type[ToolInput]:
        if isinstance(value, type) and issubclass(value, ToolInput):
            return value
        raise TypeError("tool_input must be annotated as a ToolInput subclass")

    @staticmethod
    def _validate_state_type(value: Any) -> type[ToolState]:
        if isinstance(value, type) and issubclass(value, ToolState):
            return value
        raise TypeError("tool_state must be annotated as a ToolState subclass")

    @classmethod
    def _infer_output_type(cls, annotation: Any) -> type[ToolOutput] | None:
        origin = get_origin(annotation)
        args = get_args(annotation)

        if isinstance(annotation, type) and issubclass(annotation, ToolOutput):
            return annotation
        if origin in {Running, Finished} and args:
            output_type = args[0]
            if isinstance(output_type, type) and issubclass(output_type, ToolOutput):
                return output_type
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
    def emit_initial(cls, tool_call_id: str, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> Running:
        # 异步工具被调用时必须立刻返回一条信息以满足”异步调用后立刻返回一条结果塞入LLM历史以兼容部分LLM tool call后紧跟ToolMessage的要求“
        # 运行时强制检查返回的Running str中带有tool_call_id，以供async_tool_updated工具区分
        pass

    @classmethod
    async def aemit_initial(
        cls, tool_call_id: str, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> Running:
        # 默认将同步emit_initial放入线程池执行；异步工具可覆写此方法
        return await asyncio.to_thread(cls.emit_initial, tool_input, tool_state, global_state)

    @classmethod
    @abstractmethod
    def emit_updates(
        cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> Iterator[ToolResult[ToolOutput]]:
        # 主动yield中间结果
        pass

    @classmethod
    async def aemit_updates(
        cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> AsyncIterator[ToolResult[ToolOutput]]:
        # 默认将同步emit_updates的next放入线程池，避免阻塞event loop
        iterator = cls.emit_updates(tool_input, tool_state, global_state)
        while True:
            item = await asyncio.to_thread(_next_or_sentinel, iterator)
            if item is _SENTINEL:
                break
            yield item

    @classmethod
    def status(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str:
        # LLM查询时返回
        return ""

    @classmethod
    async def astatus(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str:
        return await asyncio.to_thread(cls.status, tool_input, tool_state, global_state)

    @classmethod
    def stop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # 终止工具调用
        pass

    @classmethod
    async def astop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.stop, tool_input, tool_state, global_state)

    @classmethod
    def subscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # 启用emit监听时的额外逻辑
        pass

    @classmethod
    async def asubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.subscribe, tool_input, tool_state, global_state)

    @classmethod
    def unsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # 关闭emit监听时的额外逻辑
        pass

    @classmethod
    async def aunsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.unsubscribe, tool_input, tool_state, global_state)


class SyncTool(ABC):
    name: ClassVar[Optional[str]] # 不设置时使用工具类名
    input_type: ClassVar[type[ToolInput]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.input_type, cls.output_type = (
            cls._infer_types_from_method_annotations()
        )

    @classmethod
    def _infer_types_from_method_annotations(
        cls,
    ) -> tuple[type[ToolInput], type[ToolOutput]]:
        raw_method = cls.__dict__.get("invoke")
        if raw_method is None:
            raise TypeError(f"{cls.__name__} must define invoke")

        method = raw_method.__func__ if isinstance(raw_method, classmethod) else raw_method
        hints = get_type_hints(method)

        if "tool_input" not in hints:
            raise TypeError(f"{cls.__name__} must annotate tool_input")
        if "return" not in hints:
            raise TypeError(f"{cls.__name__} must annotate a ToolOutput return type")

        return (
            cls._validate_input_type(hints["tool_input"]),
            cls._validate_output_type(hints["return"]),
        )

    @staticmethod
    def _validate_input_type(value: Any) -> type[ToolInput]:
        if isinstance(value, type) and issubclass(value, ToolInput):
            return value
        raise TypeError("tool_input must be annotated as a ToolInput subclass")

    @staticmethod
    def _validate_output_type(value: Any) -> type[ToolOutput]:
        if isinstance(value, type) and issubclass(value, ToolOutput):
            return value
        raise TypeError("return must be annotated as a ToolOutput subclass")

    @classmethod
    @abstractmethod
    def invoke(
        cls, tool_input: ToolInput, global_state: ToolEngineState
    ) -> ToolOutput:
        pass

    @classmethod
    async def ainvoke(
        cls, tool_input: ToolInput, global_state: ToolEngineState
    ) -> ToolOutput:
        return await asyncio.to_thread(cls.invoke, tool_input, global_state)
```

子类通过方法参数与返回值注解自动推断`input_type`、`state_type`和`output_type`：

```python
class SearchInput(ToolInput):
    query: str
    limit: int = 5


@dataclass
class SearchState(ToolState):
    pages_done: int = 0


class SearchOutput(ToolOutput):
    content: str


class SearchTool(AsyncTool):
    name = "search"

    @classmethod
    def emit_initial(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Running:
        return Running(f"开始搜索：{tool_input.query}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[SearchOutput]]:
        yield Running("正在检索")
        yield Finished(SearchOutput(content="搜索完成"))


class SyncSearchTool(SyncTool):
    name = "sync_search"

    @classmethod
    def invoke(
        cls,
        tool_input: SearchInput,
        global_state: ToolEngineState,
    ) -> SearchOutput:
        return SearchOutput(content=f"搜索完成：{tool_input.query}")
```

## 实现

### 同步异步兼容

```python
Tool = BaseTool | type[SyncTool] | type[AsyncTool]
```

- BaseTool：外部 LangChain 工具，保持兼容，不承诺能访问 ToolEngineState
- SyncTool：XTalk 原生同步工具，可访问 ToolEngineState
- AsyncTool：XTalk 原生异步工具，可访问 ToolEngineState

### ToolEngine

- 保证Tools跨Agent clone
- 将含有异步工具时额外绑定的工具绑定到model
- 负责管理工具订阅状态
- 反馈Tool主动触发的ToolMessage
    - 控制异步工具的调用时机：触发异步工具时先调用`aemit_initial`生成必须立即回填的`ToolMessage`，随后启动工具执行并按订阅状态消费`aemit_updates`

```python
@dataclass
class ToolRun:
    result: ToolResult

@dataclass
class AsyncToolRun(ToolRun):
    task: asyncio.Task[Any]
    tool_class: type[AsyncTool]
    tool_input: ToolInput
    tool_state: ToolState
    subscribed: bool
    running: bool

class ToolEngine:
    def __init__(self, tools: list[Tool], state: ToolEngineState):
        # 复制一份tools挂载到self.tools；复制一份state挂载到self.state
        # 初始化id到工具运行的字典self._id_to_tool_runs: dict[str, ToolRun]
        # 若工具中有异步工具，self.tools额外添加工具self._create_assist_tools
        pass

    def bind(self, model: ChatOpenAI) -> ChatOpenAI:
        # 把tools绑定到model并返回绑定后的model
        pass
    def on_async_tool_update(self, cb: Callable[[ToolCall, ToolMessage], None]):
        # 异步工具主动发ToolMessage且被订阅时触发cb
        # self._async_tool_update_callback = cb
        # 客户端推荐挂载的cb:调用append_tool_message后触发生成/如果append前最后一条是未完的HumanMessage则先不生成
        pass
    async def ainvoke(self, tool_call: ToolCall) -> ToolMessage:
        # 触发工具并产生ToolMessage(同步工具直接产出，异步工具emit_initial产出)
        # 同步工具调用把Result填到self._id_to_tool_runs；异步工具用AsyncToolRun额外所需参数：
        # task为不断从aemit_updates中yield并调用self._async_tool_update_callback（被订阅的工具调用会把Running和Finished的yield都调用callback，未被订阅的工具调用只会把Finished调用callback)，并注意锁住global_state
        pass
    def invoke(self, tool_call: ToolCall) -> ToolMessage:
        pass
    @staticmethod
    def extract_tool_calls(gathered: AIMessageChunk) -> list[ToolCall]:
        pass
    # 耦合messages的处理逻辑的方法：------
    async def ainvoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]):
        # ainvoke并调用append_tool_message
        pass
    def invoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]):
        # invoke并调用append_tool_message
        pass
    @staticmethod
    def append_tool_message(tool_call: ToolCall, tool_message: ToolMessage, list[BaseMessage]):
        # 先调用_append_tool_call，然后根据其返回ToolCall决定下一步
        # 为透传的同步工具调用：tool_message append到messages
        # 为async_tool_updated调用：用tool_message的content创建符合async_tool_updated产出的ToolMessage的消息然后append到messages
        pass
    @staticmethod
    def _append_tool_call(tool_call: ToolCall, messages: list[BaseMessage]) -> ToolCall:
        # 将tool_call无重复追加到最后一条AIMessage的tool_calls或者新建一条空的AIMessage(tool_calls=tool_calls)
        # 对于对应同步工具的tool_call,直接追加tool_call本身；对于对应异步工具的tool_call, 追加的tool_call的name应为async_tool_updated，id为原始tool_call id加一段辨识字符串，args应为{"source_call_id":原始tool_call id}
        # 返回实际被追加的工具调用
        pass
    # ------
    def _create_assist_tools(self):
        return [
            self._create_async_tool_updated_tool(),
            self._create_id_to_async_tool_status_tool(),
            self._create_subscribe_tool(),
            self._create_unsubscribe_tool(),
            self._create_stop_tool(),
        ]

    def _create_async_tool_updated_tool(self) -> BaseTool:
        @tool("async_tool_updated")
        def async_tool_updated(
            source_call_id: str,
        ) -> str:
            """部分工具被调用时会返回tool_call id，这些工具为异步工具。异步工具会不断通过该工具产生新的输出：该工具输入异步工具被调用时返回的tool_call id，输出对应异步工具新的输出。该工具由系统调用，你不能主动调用该工具。如果对话历史中包含一条工具消息，并且该工具消息的结果尚未汇报给用户，那么你的下一次回复必须提到这个工具结果。如果同时还有一条更新的用户消息，那么需要同时回应两者：先简要汇报工具更新，再对用户消息作出回复。

            Args:
                source_call_id: 异步工具调用的tool_call id。
            """
            # 系统中创建ToolMessage时包含 {"running": bool, "tool_output": str}
            return "这个工具不能被主动调用。"

        return async_tool_updated

    def _create_id_to_async_tool_status_tool(self) -> BaseTool:
        @tool("id_to_async_tool_status")
        async def id_to_async_tool_status(source_call_id: str) -> str:
            """查询异步工具调用的最新运行状态。

            Args:
                source_call_id: 异步工具调用的tool_call id。
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"异步工具调用{source_call_id不存在}"
            tool_status = await run.tool_class.astatus(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            return {
                "running": run.running
                "status": tool_status
            }

        return id_to_async_tool_status

    def _create_subscribe_tool(self) -> BaseTool:
        @tool("subscribe_async_tool")
        async def subscribe_async_tool(source_call_id: str) -> str:
            """订阅异步工具调用的后续主动更新。被订阅的异步工具会使用async_tool_updated返回过程性输出。

            Args:
                source_call_id: 原始异步工具调用的tool_call id。
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"异步工具调用{source_call_id不存在}"
            run.subscribed = True
            await run.tool_class.asubscribe(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            return f"订阅了异步工具{source_call_id}"

        return subscribe_async_tool

    def _create_unsubscribe_tool(self) -> BaseTool:
        @tool("unsubscribe_async_tool")
        async def unsubscribe_async_tool(source_call_id: str) -> str:
            """取消订阅异步工具调用的后续主动更新。取消订阅的异步工具只会使用async_tool_updated返回其最终结果。

            Args:
                source_call_id: 原始异步工具调用的tool_call id。
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"异步工具调用{source_call_id不存在}"
            run.subscribed = False
            await run.tool_class.aunsubscribe(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            return f"取消订阅了异步工具{source_call_id}"

        return unsubscribe_async_tool

    def _create_stop_tool(self) -> BaseTool:
        @tool("stop_async_tool")
        async def stop_async_tool(source_call_id: str) -> str:
            """终止仍在运行的异步工具调用。

            Args:
                source_call_id: 原始异步工具调用的tool_call id。
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"异步工具调用{source_call_id不存在}"
            await run.tool_class.astop(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            run.task.cancel()
            return f"终止了异步工具{source_call_id}"

        return stop_async_tool
```

### LLM Agent调用接口

#### 工具绑定到模型

- 含有异步工具时额外绑定的工具：按照id获取异步工具调用运行状态的工具、异步工具上报结果时自动注入到上一条AIMessage的工具（该工具描述中说明不要主动调用，有新的工具主动反馈时会自动触发）、按照id订阅/取消订阅/关闭的工具

#### 从模型回复中提取tool_calls

#### 触发工具，工具调用结果作为ToolMessage回填对话历史

- 同步工具强制工具调用结果出来后塞入ToolMessage
- 异步工具强制emit_inital塞入ToolMessage

#### 工具主动产生的ToolMessage

- 在上一条AIMessage回填工具调用，然后插入ToolMessage

## 未来拓展

- LLM可更新tool的运行状态（“二次输入”）
