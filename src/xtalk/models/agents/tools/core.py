"""Core types for X-Talk native synchronous and asynchronous tools."""

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
    """Base input model for X-Talk native tools."""


class ToolOutput(BaseModel):
    """Base output model for X-Talk native tools."""

    def to_content(self) -> str:
        """Serialize the structured result for storage in a ToolMessage."""

        return self.model_dump_json()


class _TextToolOutput(ToolOutput):
    """Wrap an external tool result as a native tool output."""

    content: str

    def to_content(self) -> str:
        """Return the external tool result as text."""

        return self.content


@dataclass
class ToolState:
    """Mutable state owned by one asynchronous tool call.

    ``call_id`` uniquely identifies the call. ``metadata`` stores custom
    progress and state owned by the concrete tool.
    """

    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


TO = TypeVar("TO", bound=ToolOutput)


@dataclass(frozen=True)
class Running:
    """Text update emitted while an asynchronous tool is still running.

    ``content`` contains the intermediate state stored in a ToolMessage.
    """

    content: str


@dataclass(frozen=True)
class Finished(Generic[TO]):
    """Structured final result emitted by an asynchronous tool.

    ``content`` is the concrete ``ToolOutput`` instance declared by the tool.
    """

    content: TO


ToolResult: TypeAlias = Running | Finished[TO]
ToolEngineState: TypeAlias = Any

# StopIteration cannot cross an asyncio Future, so use a sentinel instead.
_SENTINEL = object()
_ASYNC_TOOL_UPDATED_NAME = "async_tool_updated"
_ASYNC_TOOL_STATUS_NAME = "id_to_async_tool_status"
_SUBSCRIBE_ASYNC_TOOL_NAME = "subscribe_async_tool"
_UNSUBSCRIBE_ASYNC_TOOL_NAME = "unsubscribe_async_tool"
_STOP_ASYNC_TOOL_NAME = "stop_async_tool"


def _next_or_sentinel(iterator: Iterator[ToolResult[TO]]) -> ToolResult[TO] | object:
    """Return the next iterator item or the exhaustion sentinel.

    Parameters
    ----------
    iterator : Iterator[ToolResult[TO]]
        Synchronous tool update iterator.

    Returns
    -------
    ToolResult[TO] | object
        The next result or ``_SENTINEL`` when the iterator is exhausted.
    """

    try:
        return next(iterator)
    except StopIteration:
        return _SENTINEL


class _NativeTool(ABC):
    """Internal base class shared by native synchronous and async tools."""

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
    """Base class for long-running tools that emit incremental updates.

    Subclasses implement ``emit_initial`` and ``emit_updates``. Default async
    methods run their synchronous counterparts in worker threads. Tools backed
    by native async SDKs may override the corresponding ``a*`` methods.
    """

    subscribe_by_default: ClassVar[bool] = False
    input_type: ClassVar[type[ToolInput]]
    state_type: ClassVar[type[ToolState]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Infer input, state, and output types when a subclass is created.

        Tool authors only need method annotations. Abstract intermediate
        classes skip inference.
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
            # classmethod values are descriptors; inspect the wrapped function.
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
        """Find the concrete ToolOutput type inside a return annotation."""

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
        """Immediately produce the first protocol-compliant tool result.

        Some models require every tool call to be followed immediately by a
        ToolMessage. This method returns ``Running`` before the actual work
        continues through ``emit_updates``.
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
        """Produce the initial result in a worker thread."""

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
        """Yield incremental updates and end with one final result."""

        raise NotImplementedError

    @classmethod
    async def aemit_updates(
        cls,
        tool_input: ToolInput,
        tool_state: ToolState,
        global_state: ToolEngineState,
    ) -> AsyncIterator[ToolResult[ToolOutput]]:
        """Bridge the synchronous update iterator into an async iterator."""

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
        """Return the latest human-readable status for one call.

        Concrete tools may override this hook for model-initiated status
        queries. The default empty string means no extra status is available.
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
        """Perform tool-specific cleanup when a call is stopped.

        Tools that own external tasks or connections should override this hook.
        Cancelling the ToolEngine background task is handled by ToolEngine.
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
        """Perform tool-specific work when progress updates are subscribed.

        Subscription affects intermediate updates only. ToolEngine always
        writes the final result back to message history.
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
        """Perform tool-specific work when progress updates are unsubscribed.

        Unsubscription suppresses intermediate updates only. ToolEngine still
        writes the final result back to message history.
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
    """Base class for native tools that return one structured final result.

    Subclasses implement ``invoke``. The framework-provided ``ainvoke`` runs
    the synchronous implementation in a worker thread.
    """

    input_type: ClassVar[type[ToolInput]]
    output_type: ClassVar[type[ToolOutput]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Infer input and output types from the invoke annotations."""

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
    """Result record for one tool call.

    ``result`` stores the latest available tool result.
    """

    result: ToolResult


@dataclass
class AsyncToolRun(ToolRun):
    """Lifecycle record for one asynchronous tool call."""

    tool_class: type[AsyncTool]
    tool_input: ToolInput
    tool_state: ToolState
    subscribed: bool
    running: bool
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    task: asyncio.Task[None] | None = None
    error: BaseException | None = None


class ToolEngine:
    """Manage agent tool binding, invocation, and lifecycle state."""

    def __init__(
        self,
        tools: list[Tool],
        state: ToolEngineState,
    ) -> None:
        """Initialize the tool engine."""
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
        """Return the registered name of a tool."""
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
        """Build the ToolCall and ToolMessage for one async update."""

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
        """Convert a tool result into a serializable status payload."""

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
        """Ensure message history declares the corresponding AI tool call."""

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
        """Find the latest pending AI tool-call batch and insertion index."""

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
        """Append a tool call and its result to message history."""

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
        """Invoke a tool asynchronously and append its result to history."""

        tool_message = await self.ainvoke(tool_call)
        self.append_tool_message(tool_call, tool_message, messages)
        return tool_message

    def invoke_and_append(
        self,
        tool_call: ToolCall,
        messages: list[BaseMessage],
    ) -> ToolMessage:
        """Invoke a tool synchronously and append its result to history."""

        tool_message = self.invoke(tool_call)
        self.append_tool_message(tool_call, tool_message, messages)
        return tool_message

    @classmethod
    def _to_bindable_tool(cls, tool: Tool) -> BaseTool:
        """Convert a native tool into a LangChain-bindable tool schema."""

        if isinstance(tool, BaseTool):
            return tool

        name = cls._tool_name(tool)

        def invoke_through_engine(**kwargs: Any) -> str:
            """Prevent native tool execution from bypassing ToolEngine."""

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
        """Bind all engine-managed tools to a chat model."""

        if not self.tools:
            return model
        bindable_tools = [
            self._to_bindable_tool(tool)
            for tool in self.tools
            if self._tool_name(tool) != _ASYNC_TOOL_UPDATED_NAME
        ]
        return model.bind_tools(bindable_tools)

    def bind_without_tool_calls(
        self,
        model: BaseChatModel,
    ) -> BaseChatModel | Runnable[Any, Any]:
        """Bind tool schemas while preventing model-initiated calls.

        Parameters
        ----------
        model : BaseChatModel
            Chat model that should receive the registered tool schemas.

        Returns
        -------
        BaseChatModel or Runnable[Any, Any]
            Model binding that exposes the schemas with tool choice disabled.
        """

        if not self.tools:
            return model
        bindable_tools = [
            self._to_bindable_tool(tool)
            for tool in self.tools
        ]
        return model.bind_tools(
            bindable_tools,
            tool_choice="none",
        )

    def on_async_tool_update(
        self,
        callback: Callable[[ToolCall, ToolMessage], None],
    ) -> None:
        """Register the callback for proactive asynchronous tool updates."""

        self._async_tool_update_callback = callback

    def _create_async_tool_updated_tool(self) -> BaseTool:
        """Create the system-only asynchronous update tool."""

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
        """Create the asynchronous tool status query tool."""

        @tool(_ASYNC_TOOL_STATUS_NAME)
        async def id_to_async_tool_status(source_call_id: str) -> str:
            """Return the latest status of an asynchronous tool call.

            Use this tool when the latest progress is needed without changing
            the subscription state. The result reports whether the call is
            still running, its tool-defined status, and any recorded error.
            Call it only when the user explicitly asks for the current status.
            Never poll it repeatedly; subscribe and wait for system updates
            when future progress is needed.

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
        """Create the asynchronous tool subscription tool."""

        @tool(_SUBSCRIBE_ASYNC_TOOL_NAME)
        async def subscribe_async_tool(source_call_id: str) -> str:
            """Subscribe to progress updates from an asynchronous tool call.

            After subscription, the system may deliver intermediate results
            through ``async_tool_updated``. The subscription response also
            includes the latest known result. Final results are always sent.
            After subscribing, stop calling tools and wait for the system to
            deliver updates. Do not poll the status tool.

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
        """Create the asynchronous tool unsubscription tool."""

        @tool(_UNSUBSCRIBE_ASYNC_TOOL_NAME)
        async def unsubscribe_async_tool(source_call_id: str) -> str:
            """Stop receiving progress updates from an asynchronous tool call.

            Unsubscribing suppresses intermediate updates only. The system
            still delivers the final result through ``async_tool_updated``.

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
        """Create the asynchronous tool stop command."""

        @tool(_STOP_ASYNC_TOOL_NAME)
        async def stop_async_tool(source_call_id: str) -> str:
            """Stop an asynchronous tool call and release its resources.

            Use this tool only for a call that is still running. Repeated stop
            requests are safe and report the call as already stopped.

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
        """Find an asynchronous tool run by its source call ID."""

        async with self._runs_lock:
            run = self._id_to_tool_runs.get(source_call_id)
        return run if isinstance(run, AsyncToolRun) else None

    @staticmethod
    def _async_run_not_found_content(source_call_id: str) -> str:
        """Build the JSON response for an unknown asynchronous tool call."""

        return json.dumps(
            {"error": f"Async tool run not found: {source_call_id}"},
            ensure_ascii=False,
        )

    def _create_assist_tools(self) -> list[BaseTool]:
        """Create the assistant tools required by the async tool protocol."""

        return [
            self._create_async_tool_updated_tool(),
            self._create_async_tool_status_tool(),
            self._create_subscribe_tool(),
            self._create_unsubscribe_tool(),
            self._create_stop_tool(),
        ]

    async def _reserve_call_id(self, call_id: str) -> None:
        """Reserve a unique ID before starting a tool call."""

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
        """Store a tool run and complete its call ID reservation."""

        async with self._runs_lock:
            if self._closed:
                raise RuntimeError("ToolEngine is closed")
            self._id_to_tool_runs[call_id] = run
            self._reserved_call_ids.discard(call_id)

    async def _release_call_id(self, call_id: str) -> None:
        """Release a call ID reserved by a failed tool invocation."""

        async with self._runs_lock:
            self._reserved_call_ids.discard(call_id)

    async def _start_async_run(
        self,
        call_id: str,
        run: AsyncToolRun,
    ) -> None:
        """Subscribe and start an async update task while the engine is open."""

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
        """Invoke a tool asynchronously."""

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

        # Start the asynchronous tool and return its immediate protocol result.
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
        """Invoke a synchronous tool from a synchronous context.

        AsyncTool requires a persistent event loop for subsequent updates and
        must use ``ainvoke``. Callers already inside an event loop should also
        await ``ainvoke`` directly.
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
        """Consume updates from one asynchronous tool call."""

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
        """Retrieve a background exception to avoid unhandled-task warnings."""

        if not task.cancelled():
            task.exception()

    async def shutdown(self) -> None:
        """Close the engine and cancel all active asynchronous tool calls."""

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
