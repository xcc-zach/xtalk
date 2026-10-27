# Tools

Tools are callables the Agent exposes to its chat model. They are attached with `XtalkBuilder.add_agent_tools()` and reach the Agent constructor through `llm_agent.params.tools`. X-Talk accepts three tool types and executes them through `ToolEngine`.

## Built-in Tools

> **Note**
> See [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) for all built-in tools.

Built-in tools include `web_search`, `get_time`, and pipeline controls for silence, speech speed, voice, and emotion. `DefaultAgent` registers `web_search`, `get_time`, `set_speed`, and `silence` by default. `set_voice` and `set_emotion` are registered only when their corresponding configuration is available.

To enable `web_search`, set `SERPER_API_KEY` or `GOOGLE_SERPER_API_KEY`. See [SerperDev](https://serper.dev/).

## Tool Types

| Type | Description |
| --- | --- |
| LangChain `BaseTool` | Existing LangChain integrations or simple function tools. The concrete tool supplies its input schema and sync or async invocation behavior. |
| `SyncTool` | Native synchronous X-Talk tool. Validates `ToolInput`, returns structured `ToolOutput`, and receives a non-blocking async bridge from the framework. |
| `AsyncTool` | Native long-running X-Talk tool. Immediately returns `Running`, emits progress in the background, and ends with `Finished`. Supports status, stop, subscribe, and unsubscribe hooks. A timer is the typical case: it reports that it has started, keeps reporting progress, returns a final result when time expires, and can be queried, subscribed to, or stopped while running. |

### LangChain Tools

```python
@tool
def name(...) -> ...: ...
```

Define a LangChain tool with `@tool`. Its complete function signature and docstring become the name, description, and input schema shown to the model.

```python
from langchain_core.tools import tool


@tool
def convert_temperature(value: float, to_unit: str) -> str:
    """Convert a temperature between Celsius and Fahrenheit.

    Args:
        value: Temperature to convert.
        to_unit: Target unit, either celsius or fahrenheit.
    """

    if to_unit == "celsius":
        return str((value - 32) * 5 / 9)
    if to_unit == "fahrenheit":
        return str(value * 9 / 5 + 32)
    raise ValueError("to_unit must be celsius or fahrenheit")


sync_result = convert_temperature.invoke(
    {"value": 32, "to_unit": "celsius"}
)
async_result = await convert_temperature.ainvoke(
    {"value": 0, "to_unit": "fahrenheit"}
)
```

### SyncTool

```python
class SyncTool:
    name: str

    @classmethod
    def invoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput: ...

    @classmethod
    async def ainvoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput: ...
```

`SyncTool` infers its input and output types from the annotations on `invoke()`.

**Parameters**

- `tool_input`: instance of the `ToolInput` subclass annotated on `invoke()`; validated before the call.
- `global_state`: session-level object shared by tools in one `ToolEngine`.

**Returns**

- `ToolOutput` subclass. `ToolOutput.to_content()` serializes results as JSON by default and may be overridden.

**Notes**

- The class name is used when `name` is omitted.
- Tool authors must implement `invoke()`; inherited `ainvoke()` calls it in a worker thread to avoid blocking the event loop.

```python
from xtalk.models.agents.tools import (
    SyncTool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
)


class AddInput(ToolInput):
    left: int
    right: int


class AddOutput(ToolOutput):
    value: int


class AddTool(SyncTool):
    """Add two integers."""

    name = "add"

    @classmethod
    def invoke(
        cls,
        tool_input: AddInput,
        global_state: ToolEngineState,
    ) -> AddOutput:
        del global_state
        return AddOutput(value=tool_input.left + tool_input.right)


output = AddTool.invoke(AddInput(left=2, right=3), {})
async_output = await AddTool.ainvoke(AddInput(left=2, right=3), {})
content = output.to_content()
```

### AsyncTool

```python
class AsyncTool:
    name: str
    subscribe_by_default: bool = False

    @classmethod
    def emit_initial(cls, tool_call_id, tool_input, tool_state, global_state) -> Running: ...
    @classmethod
    def emit_updates(cls, tool_input, tool_state, global_state) -> Iterator[ToolResult]: ...
    @classmethod
    def status(cls, tool_input, tool_state, global_state) -> str: ...
    @classmethod
    def stop(cls, tool_input, tool_state, global_state) -> None: ...
    @classmethod
    def subscribe(cls, tool_input, tool_state, global_state) -> None: ...
    @classmethod
    def unsubscribe(cls, tool_input, tool_state, global_state) -> None: ...
```

`AsyncTool` stores per-call state in `ToolState` and manages background work through lifecycle hooks.

**Required hooks**

- `emit_initial()`: returns the first `Running`; its `content` must include `tool_call_id`.
- `emit_updates()`: update stream; may emit multiple `Running` values and must emit one `Finished` on normal completion.

**Optional hooks**

- `status()`, `stop()`, `subscribe()`, `unsubscribe()`.

**Async variants**

- `aemit_initial()`, `aemit_updates()`, `astatus()`, `astop()`, `asubscribe()`, `aunsubscribe()` bridge the synchronous implementations through a worker thread by default. Override these `a*` methods when the underlying SDK is natively asynchronous.
- `subscribe_by_default=True` subscribes before the background update task starts.

```python
from collections.abc import Iterator
from dataclasses import dataclass
import time

from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)


class TimerInput(ToolInput):
    seconds: int


class TimerOutput(ToolOutput):
    elapsed_seconds: int


@dataclass
class TimerState(ToolState):
    elapsed_seconds: int = 0
    subscribed: bool = False
    stopped: bool = False


class TimerTool(AsyncTool):
    """Count time in the background and report progress."""

    name = "timer"
    subscribe_by_default = False

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Running:
        del tool_input, global_state
        tool_state.call_id = tool_call_id
        return Running(content=f"Timer started, call ID: {tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[TimerOutput]]:
        del global_state
        for elapsed in range(1, tool_input.seconds + 1):
            if tool_state.stopped:
                return
            time.sleep(1)
            tool_state.elapsed_seconds = elapsed
            if elapsed < tool_input.seconds:
                yield Running(content=f"Elapsed: {elapsed} seconds")
        yield Finished(
            content=TimerOutput(elapsed_seconds=tool_state.elapsed_seconds)
        )

    @classmethod
    def status(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> str:
        del tool_input, global_state
        return f"Elapsed: {tool_state.elapsed_seconds} seconds"

    @classmethod
    def stop(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.stopped = True

    @classmethod
    def subscribe(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = True

    @classmethod
    def unsubscribe(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = False
```

## ToolEngine

```python
ToolEngine(tools: list[Tool], state: ToolEngineState)
```

`ToolEngine` binds all three tool types to a model, executes model-produced `ToolCall` values, preserves valid `AIMessage`/`ToolMessage` history, and manages the `AsyncTool` lifecycle.

**Methods**

- `ToolEngine(tools, state)`: create an engine; tool names must be unique within it.
- `bind(model)`: expose schemas and allow model-initiated calls.
- `bind_without_tool_calls(model)`: keep schemas available with `tool_choice="none"`.
- `on_async_tool_update(callback)`: receive subscribed `Running` updates and every `Finished` update. See [AsyncTool](#asynctool).
- `ainvoke(tool_call)` / `invoke(tool_call)`: execute a tool and return `ToolMessage` without changing history.
- `ainvoke_and_append(tool_call, messages)` / `invoke_and_append(...)`: execute a tool and append a matching call/result pair to history.
- `append_tool_message(tool_call, tool_message, messages)`: append an existing call/result pair without executing the tool.
- `shutdown()`: reject new calls, run async-tool stop hooks, and cancel background tasks; repeated calls are safe.

**Constraints**

- Use `await ainvoke()` or `await ainvoke_and_append()` in asynchronous contexts. `invoke()` and `invoke_and_append()` are only for synchronous contexts without a running event loop and cannot invoke `AsyncTool`.
- When an engine contains an `AsyncTool`, it also binds tools for status, subscribe, unsubscribe, and stop operations. Asynchronous updates return proactively through the callback registered by `on_async_tool_update()`. Unsubscribed calls proactively emit only the final `Finished`; subscribed calls emit intermediate `Running` values and the final `Finished`.

**Example**

This Agent uses every public `ToolEngine` interface:

```python
import asyncio
from typing import Any, AsyncIterator, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolCall, ToolMessage

from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
from xtalk.models.agents.tools import Tool, ToolEngine


class ToolAgent(Agent):
    def __init__(self, model: BaseChatModel, tools: list[Tool]) -> None:
        self.model = model
        self.tools = tools
        self.messages: list[BaseMessage] = []
        self.engine = ToolEngine(tools=tools, state={})
        self.model_with_tools = self.engine.bind(model)
        self.model_without_calls = self.engine.bind_without_tool_calls(model)
        self._async_update_queue: asyncio.Queue[None] = asyncio.Queue()
        self._model_lock = asyncio.Lock()
        self.engine.on_async_tool_update(self._on_async_tool_update)

    def _on_async_tool_update(
        self,
        tool_call: ToolCall,
        tool_message: ToolMessage,
    ) -> None:
        ToolEngine.append_tool_message(tool_call, tool_message, self.messages)
        self._async_update_queue.put_nowait(None)

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        yield from self.sync_iter_from_async(self.async_accept(context))

    async def _invoke_model(self, *, allow_tools: bool):
        model = (
            self.model_with_tools
            if allow_tools
            else self.model_without_calls
        )
        async with self._model_lock:
            response = await model.ainvoke(list(self.messages))
            self.messages.append(response)
            return response

    async def _report_async_tool_updates(
        self,
    ) -> AsyncIterator[AgentOutput]:
        while True:
            await self._async_update_queue.get()
            response = await self._invoke_model(allow_tools=False)
            yield str(response.content)

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        if context["type"] == "loop":
            async for output in self._report_async_tool_updates():
                yield output
            return

        if context["type"] != "asr_final":
            return

        self.messages.append(HumanMessage(content=context["data"]["text"]))
        response = await self._invoke_model(allow_tools=True)

        for tool_call in response.tool_calls:
            yield tool_call
            tool_message = await self.engine.ainvoke_and_append(
                tool_call,
                self.messages,
            )
            yield str(tool_message.content)

    def clone(self) -> "ToolAgent":
        return ToolAgent(self.model, self.tools)

    async def shutdown(self) -> None:
        await self.engine.shutdown()
```

The asynchronous-update callback first appends a valid call/result pair to history, then puts a notification in `_async_update_queue`. The long-lived `loop` context waits on that queue and calls `_report_async_tool_updates()` for each notification. The generated text is yielded as `AgentOutput`, which passes the proactive report back to the service output pipeline.

`async_accept()` passes `allow_tools=True` for user requests, selecting the model returned by `bind()` so it may produce new tool calls. `_report_async_tool_updates()` passes `allow_tools=False` and selects the model returned by `bind_without_tool_calls()`; the model can still interpret tool schemas in history but cannot call another tool during that response. `_model_lock` serializes generation and history insertion for normal replies and proactive reports, while `list(self.messages)` gives each invocation a stable history snapshot.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Attach Tools →](../tutorial/attach_tools.md)** · Tutorial

-   **[Introduce an LLM Agent →](../tutorial/introduce_an_llm_agent.md)** · Tutorial

-   **[Generated tools module API →](../api/server/xtalk/models/agents/tools/index.md)** · Reference

</div>
