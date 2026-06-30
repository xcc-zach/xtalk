# Asynchronous Tool Calls

## Principles

- Asynchronous tools are a superset of synchronous tools; an asynchronous tool becomes a synchronous tool when it only contains the tool execution function.
- Immediately after an asynchronous call starts, return one result and insert it into the LLM history. This keeps compatibility with LLMs that require a `ToolMessage` immediately after a tool call.
    - Protocol-level invalid cases: an `AIMessage` with non-empty `tool_calls` is not followed by a `ToolMessage`; or no previous `AIMessage.tool_calls` contains the id referenced by a `ToolMessage`.
- The LLM can query the latest tool result.
- The LLM can stop a tool call.
- The LLM can subscribe to or unsubscribe from a tool call. When subscribed, the tool reports the latest result back to the LLM.
- Tool results are divided into running results and finished results.

## Interface Design

### Tool

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
    name: ClassVar[Optional[str]] # Use the tool class name when omitted.
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
        # An asynchronous tool must immediately return one message when called.
        # This satisfies LLMs that require a ToolMessage immediately after a tool call.
        # Runtime checks should require the returned Running string to include
        # tool_call_id, so async_tool_updated can distinguish tool calls.
        pass

    @classmethod
    async def aemit_initial(
        cls, tool_call_id: str, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> Running:
        # By default, run the synchronous emit_initial in a thread pool.
        # Asynchronous tools may override this method.
        return await asyncio.to_thread(cls.emit_initial, tool_input, tool_state, global_state)

    @classmethod
    @abstractmethod
    def emit_updates(
        cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> Iterator[ToolResult[ToolOutput]]:
        # Actively yield intermediate results.
        pass

    @classmethod
    async def aemit_updates(
        cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState
    ) -> AsyncIterator[ToolResult[ToolOutput]]:
        # By default, run next() for the synchronous emit_updates iterator in a
        # thread pool to avoid blocking the event loop.
        iterator = cls.emit_updates(tool_input, tool_state, global_state)
        while True:
            item = await asyncio.to_thread(_next_or_sentinel, iterator)
            if item is _SENTINEL:
                break
            yield item

    @classmethod
    def status(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str:
        # Returned when the LLM queries the tool.
        return ""

    @classmethod
    async def astatus(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str:
        return await asyncio.to_thread(cls.status, tool_input, tool_state, global_state)

    @classmethod
    def stop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # Stop the tool call.
        pass

    @classmethod
    async def astop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.stop, tool_input, tool_state, global_state)

    @classmethod
    def subscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # Additional logic when update listening is enabled.
        pass

    @classmethod
    async def asubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.subscribe, tool_input, tool_state, global_state)

    @classmethod
    def unsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        # Additional logic when update listening is disabled.
        pass

    @classmethod
    async def aunsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None:
        await asyncio.to_thread(cls.unsubscribe, tool_input, tool_state, global_state)


class SyncTool(ABC):
    name: ClassVar[Optional[str]] # Use the tool class name when omitted.
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

Subclasses automatically infer `input_type`, `state_type`, and `output_type` from method parameter and return annotations:

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
        return Running(f"Starting search: {tool_input.query}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[SearchOutput]]:
        yield Running("Searching")
        yield Finished(SearchOutput(content="Search complete"))


class SyncSearchTool(SyncTool):
    name = "sync_search"

    @classmethod
    def invoke(
        cls,
        tool_input: SearchInput,
        global_state: ToolEngineState,
    ) -> SearchOutput:
        return SearchOutput(content=f"Search complete: {tool_input.query}")
```

### Synchronous and Asynchronous Compatibility

```python
Tool = BaseTool | type[SyncTool] | type[AsyncTool]
```

- `BaseTool`: external LangChain tools. Kept compatible, but not guaranteed to access `ToolEngineState`.
- `SyncTool`: native XTalk synchronous tools. Can access `ToolEngineState`.
- `AsyncTool`: native XTalk asynchronous tools. Can access `ToolEngineState`.

### ToolEngine

- Keep tools available across `Agent.clone`.
- Bind additional tools to the model when asynchronous tools exist.
- Manage tool subscription state.
- Feed back `ToolMessage` values actively triggered by tools.
    - Control the timing of asynchronous tool calls: when an asynchronous tool is triggered, first call `aemit_initial` to generate the `ToolMessage` that must be immediately written back, then start tool execution and consume `aemit_updates` according to subscription state.

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
        # Copy tools into self.tools; copy state into self.state.
        # Initialize self._id_to_tool_runs: dict[str, ToolRun].
        # If asynchronous tools exist, append self._create_assist_tools to self.tools.
        # TODO
        pass

    def bind(self, model: ChatOpenAI) -> ChatOpenAI:
        # Bind tools to the model and return the bound model.
        # TODO
        pass
    def on_async_tool_update(self, cb: Callable[[ToolCall, ToolMessage], None]):
        # Trigger cb when an asynchronous tool actively emits a ToolMessage and
        # the tool call is subscribed.
        # self._async_tool_update_callback = cb
        # Recommended client callback: call append_tool_message and trigger
        # generation afterward. If the last message before append is an
        # unfinished HumanMessage, do not generate yet.
        # Partial HumanMessage values should also stay in history. Use a bool
        # variable to mark whether the user has finished speaking. If a partial
        # message is interrupted by inserted AI/tool messages, and a later ASR
        # partial starts with the previous partial, append a new HumanMessage
        # containing only the suffix not covered by the prefix; otherwise append
        # the full new partial.
        # TODO

        pass
    async def ainvoke(self, tool_call: ToolCall) -> ToolMessage:
        # Trigger a tool and produce a ToolMessage. Synchronous tools directly
        # produce the result; asynchronous tools produce emit_initial.
        # Synchronous tool calls store Result in self._id_to_tool_runs.
        # Asynchronous tool calls use AsyncToolRun with extra required fields:
        # task continuously yields from aemit_updates and calls
        # self._async_tool_update_callback. Subscribed tool calls call the
        # callback for both Running and Finished yields; unsubscribed tool calls
        # call the callback only for Finished. Remember to lock global_state.
        # TODO

        pass
    def invoke(self, tool_call: ToolCall) -> ToolMessage:
        # TODO

        pass
    @staticmethod
    def extract_tool_calls(gathered: AIMessageChunk) -> list[ToolCall]:
        # TODO

        pass
    # Methods coupled to message-list handling: ------
    async def ainvoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]):
        # Call ainvoke and then append_tool_message.
        # TODO

        pass
    def invoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]):
        # Call invoke and then append_tool_message.
        # TODO

        pass
    @staticmethod
    def append_tool_message(tool_call: ToolCall, tool_message: ToolMessage, list[BaseMessage]):
        # First call _append_tool_call, then decide the next step based on the
        # ToolCall it returns.
        # For forwarded synchronous tool calls: append tool_message to messages.
        # For async_tool_updated calls: use tool_message.content to create and
        # append a ToolMessage matching async_tool_updated output.
        # TODO

        pass
    @staticmethod
    def _append_tool_call(tool_call: ToolCall, messages: list[BaseMessage]) -> ToolCall:
        # Append tool_call without duplication to the last AIMessage.tool_calls,
        # or create an empty AIMessage(tool_calls=tool_calls).
        # For a synchronous tool_call, append the tool_call itself.
        # For an asynchronous tool_call, append a tool_call whose name is
        # async_tool_updated, whose id is the original tool_call id plus a
        # distinguishing suffix, and whose args are
        # {"source_call_id": original tool_call id}.
        # Return the actual appended tool call.
        # TODO

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
            """Some tool calls return a tool_call id. These tools are asynchronous tools. An asynchronous tool continuously produces new output through this tool: this tool takes the tool_call id returned when the asynchronous tool was called, and outputs the new result for that asynchronous tool. This tool is called by the system; you must not call it proactively. If the conversation history contains a tool message whose result has not yet been reported to the user, your next response must mention this tool result. If there is also a newer user message, respond to both: first briefly report the tool update, then respond to the user message.

            Args:
                source_call_id: The tool_call id of the asynchronous tool call.
            """
            # When ToolMessage is created in the system, it contains
            # {"running": bool, "tool_output": str}.
            return "This tool cannot be called proactively."

        return async_tool_updated

    def _create_id_to_async_tool_status_tool(self) -> BaseTool:
        @tool("id_to_async_tool_status")
        async def id_to_async_tool_status(source_call_id: str) -> str:
            """Query the latest running status of an asynchronous tool call.

            Args:
                source_call_id: The tool_call id of the asynchronous tool call.
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"Async tool call {source_call_id} does not exist"
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
            """Subscribe to subsequent active updates from an asynchronous tool call. Subscribed asynchronous tools return intermediate outputs through async_tool_updated.

            Args:
                source_call_id: The original tool_call id of the asynchronous tool call.
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"Async tool call {source_call_id} does not exist"
            run.subscribed = True
            await run.tool_class.asubscribe(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            return f"Subscribed to asynchronous tool {source_call_id}"

        return subscribe_async_tool

    def _create_unsubscribe_tool(self) -> BaseTool:
        @tool("unsubscribe_async_tool")
        async def unsubscribe_async_tool(source_call_id: str) -> str:
            """Unsubscribe from subsequent active updates from an asynchronous tool call. Unsubscribed asynchronous tools only return their final result through async_tool_updated.

            Args:
                source_call_id: The original tool_call id of the asynchronous tool call.
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"Async tool call {source_call_id} does not exist"
            run.subscribed = False
            await run.tool_class.aunsubscribe(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            return f"Unsubscribed from asynchronous tool {source_call_id}"

        return unsubscribe_async_tool

    def _create_stop_tool(self) -> BaseTool:
        @tool("stop_async_tool")
        async def stop_async_tool(source_call_id: str) -> str:
            """Stop an asynchronous tool call that is still running.

            Args:
                source_call_id: The original tool_call id of the asynchronous tool call.
            """
            run = self._id_to_tool_runs.get(source_call_id)
            if run is None or not isinstance(run, AsyncToolRun):
                return f"Async tool call {source_call_id} does not exist"
            await run.tool_class.astop(
                run.tool_input,
                run.tool_state,
                self.state,
            )
            run.task.cancel()
            return f"Stopped asynchronous tool {source_call_id}"

        return stop_async_tool
```

# Implementation

### Implementation Location

Place `Tool`, `ToolEngine`, and related types in `src/xtalk/models/agents/tools/core.py`, and export the required types from `src/xtalk/models/agents/tools/__init__.py` (types used for creating new tools and types used by agents). Also migrate required content from `src/xtalk/models/agents/tools/utils.py` into `core.py`. Then update `src/xtalk/models/agents/experimental.py` according to `### How an LLM Agent Uses ToolEngine`.

### How an LLM Agent Uses ToolEngine

#### # __init__

```python
self._base_tools = tools
self.tool_engine = ToolEngine(
    tools=tools or [],
    state={},
)
self.model_with_tools = self.tool_engine.bind(self.model)

self._human_input_finished = True
self._last_partial_human_text = ""
self._active_partial_human_index: int | None = None
self._active_partial_human_prefix = ""

self._async_tool_update_queue: asyncio.Queue[AgentOutput] = asyncio.Queue()
self.tool_engine.on_async_tool_update(self._on_async_tool_update)
```

#### # _stream_messages

```python
async def _stream_messages(self) -> AsyncIterator[AgentOutput]:
    while True:
        gathered = None

        async for chunk in self.model_with_tools.astream(self.messages):
            text = self.content_to_text(chunk.content)
            if text:
                yield text
            gathered = chunk if gathered is None else gathered + chunk

        tool_calls = ToolEngine.extract_tool_calls(gathered)
        if not tool_calls:
            return

        for tool_call in tool_calls:
            yield tool_call

            tool_message = await self.tool_engine.ainvoke_and_append(
                tool_call,
                self._chat_history.messages,
            )

            yield build_tool_call_result(
                tool_call=tool_call,
                result_content=str(tool_message.content),
            )
```

#### # async tool update callback

```python
def _on_async_tool_update(
    self,
    tool_call: ToolCall,
    tool_message: ToolMessage,
) -> None:
    ToolEngine.append_tool_message(
        tool_call,
        tool_message,
        self._chat_history.messages,
    )

    if not self._human_input_finished:
        return

    self._async_tool_update_queue.put_nowait(
        build_tool_call_result(
            tool_call=tool_call,
            result_content=str(tool_message.content),
        )
    )
```

#### # _loop_runner

```python
async def _loop_runner(self) -> AsyncIterator[AgentOutput]:
    while True:
        if self.proactive and len(self.messages) == 1:
            self._chat_history.append_message(HumanMessage(content="Hello."))
            async for item in self._stream_greeting():
                yield item
            break

        try:
            item = await asyncio.wait_for(
                self._async_tool_update_queue.get(),
                timeout=0.2,
            )
            yield item
        except asyncio.TimeoutError:
            pass
```

#### # deal with human message

```python
def _append_or_update_partial_human_message(
    self,
    text: str,
    *,
    final: bool,
) -> None:
    if not text:
        return

    messages = self._chat_history.messages
    active_message_is_last = (
        self._active_partial_human_index is not None
        and self._active_partial_human_index == len(messages) - 1
        and isinstance(messages[-1], HumanMessage)
    )

    if active_message_is_last:
        if text.startswith(self._active_partial_human_prefix):
            messages[-1].content = text[len(self._active_partial_human_prefix):]
        else:
            messages[-1].content = text
            self._active_partial_human_prefix = ""
    else:
        if self._last_partial_human_text and text.startswith(
            self._last_partial_human_text
        ):
            content = text[len(self._last_partial_human_text):]
            self._active_partial_human_prefix = self._last_partial_human_text
        else:
            content = text
            self._active_partial_human_prefix = ""

        self._chat_history.append_message(HumanMessage(content=content))
        self._active_partial_human_index = len(messages) - 1

    self._last_partial_human_text = text
    self._human_input_finished = final

    if final:
        self._active_partial_human_index = None
        self._active_partial_human_prefix = ""
        self._last_partial_human_text = ""


async def _handle_asr_partial(self, asr_text: str) -> AsyncIterator[AgentOutput]:
    self._append_or_update_partial_human_message(asr_text, final=False)

    if self.backchannel_model is None or self.backchannel_source_dir is None:
        return

    ...


async def _handle_asr_final(self, asr_text: str) -> AsyncIterator[AgentOutput]:
    self._append_or_update_partial_human_message(asr_text, final=True)

    async for item in self._stream_messages():
        yield item

    self._already_backchanneled_text = ""
    self._turn_already_to_backchannel_response = {}
```


## Future Extensions

- The LLM can update a tool's running state ("secondary input").
