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
TR = TypeVar("TR")


@dataclass(frozen=True)
class Running(Generic[TR]):
    content: TR


@dataclass(frozen=True)
class Finished(Generic[TO]):
    content: TO


ToolResult: TypeAlias = Running[str] | Finished[TO]
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
    def emit_initial(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> Running[str]:
        # 异步工具被调用时必须立刻返回一条信息以满足”异步调用后立刻返回一条结果塞入LLM历史以兼容部分LLM tool call后紧跟ToolMessage的要求“
        pass

    @classmethod
    async def aemit_initial(
        cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> Running[str]:
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
    ) -> Running[str]:
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

### LLM Agent调用接口

#### 工具绑定到模型

- 含有异步工具时额外绑定的工具：按照id获取工具调用运行状态的工具、异步工具上报结果时自动注入到上一条AIMessage的工具（该工具描述中说明不要主动调用，有新的工具主动反馈时会自动触发）、按照id订阅/取消订阅/关闭的工具

```python
def bind_tools(model: ChatOpenAI, tools: list[Tool]) -> None:
    pass
```

#### 从模型回复中提取tool_calls

```python
def extract_tool_calls(gathered: AIMessageChunk) -> list[ToolCall]:
    pass
def append_tool_calls(tool_calls: list[ToolCall]) -> None:
    # 将tool_calls放到最后一条AIMessage的tool_calls或者新建一条空的AIMessage(tool_calls=tool_calls)
    pass
```

#### 触发工具，工具调用结果作为ToolMessage回填对话历史

- 同步工具强制工具调用结果出来后塞入ToolMessage后插入一条空AIMeessage对话历史才能接下一条用户消息
- 异步工具强制emit_inital塞入ToolMessage后插入一条空AIMeessage对话历史才能接下一条用户消息

```python
async def atrigger_tool_call(tool_call: ToolCall, tool: Tool, on_async_tool_update: Optional[Callable[[ToolCall, ToolResult], Awaitable[None]]]=None):
    pass
# 同时写一个同步版本trigger_tool_call
```

#### 工具主动产生的ToolMessage

- 在上一条AIMessage回填工具调用，然后插入ToolMessage

## 未来拓展

- LLM可更新tool的运行状态（“二次输入”）
- 带状态的同步/异步工具
