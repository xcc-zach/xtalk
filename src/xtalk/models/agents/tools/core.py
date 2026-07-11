"""X-Talk 原生同步、异步工具的基础类型。"""

from __future__ import annotations

import asyncio
import inspect
import types
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Generic,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.tools import BaseTool
from pydantic import BaseModel


class ToolInput(BaseModel):
    """X-Talk 原生工具输入模型的基类。"""


class ToolOutput(BaseModel):
    """X-Talk 原生工具输出模型的基类。"""

    def to_content(self) -> str:
        """将结构化结果序列化为 ToolMessage 可保存的文本。"""

        return self.model_dump_json()


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
]
