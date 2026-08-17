# Introduce Tools

## Attach Custom Tools

X-Talk attaches Python-only tools during staged configuration, before the configured Agent is instantiated:

```python
from xtalk import Xtalk

from my_tools import TimerTool


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .add_agent_tools([TimerTool])
    .build()
)
```

`add_agent_tools` is an `XtalkBuilder` method. It accepts LangChain tool instances, native X-Talk `SyncTool` or `AsyncTool` classes, and zero-argument tool factories. Repeated calls append tools in order without modifying the source configuration dictionary.

Use a factory when every session needs an independent tool instance. Native asynchronous tools keep their mutable state in each tool call. See [`examples/sample_app/custom_async_tool.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_async_tool.py) for a complete timer example.

## Built-in Tools

> **Note**
> See source code under [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) for all built-in tools.

Built-in tools include agent-scope ones like `web_search` and `get_time`, and pipeline-control ones such as silence, speech speed, and, when configured, voice and emotion switching. `DefaultAgent` registers `web_search`, `get_time`, `set_speed`, and `silence` by default. `set_voice` and `set_emotion` are only registered when the corresponding configuration is available.

> **Note**
> To enable the `web_search` tool, set `SERPER_API_KEY` or `GOOGLE_SERPER_API_KEY`. See [SerperDev](https://serper.dev/).

## Tool Types and Invocation Flow

X-Talk supports three kinds of agent tools: LangChain `BaseTool` instances,
native synchronous `SyncTool` classes, and `AsyncTool` classes that can keep
producing updates in the background. `ToolEngine` binds and invokes all three
kinds while maintaining asynchronous lifecycles and valid message history.

## Goals

- A synchronous tool returns one final result.
- An asynchronous tool immediately returns `Running`, so the agent is not
  blocked and models that require a matching `ToolMessage` directly after
  `AIMessage(tool_calls=...)` remain protocol compliant.
- An asynchronous tool may emit progress updates and must end with `Finished`.
- The LLM can query, subscribe to, unsubscribe from, or stop an asynchronous
  call.
- Every call ID is unique within one `ToolEngine`.
- Tool results are inserted into history as valid ToolCall/ToolMessage pairs.

The implementation lives in
[`src/xtalk/models/agents/tools/core.py`](../../src/xtalk/models/agents/tools/core.py).
Public types are exported from `xtalk.models.agents.tools`.

## Core data types

```python
from dataclasses import dataclass, field
from typing import Any

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


@dataclass(frozen=True)
class Running:
    content: str


@dataclass(frozen=True)
class Finished:
    content: ToolOutput
```

Their responsibilities are:

- `ToolInput`: structured LLM-generated input validated by Pydantic.
- `ToolOutput`: structured final output, serialized to JSON by default.
- `ToolState`: mutable state owned by one asynchronous call.
- `Running`: textual state emitted before the asynchronous call finishes.
- `Finished`: the final structured result.

`ToolEngineState` is the session-level shared object passed to native tools. It
currently has type `Any`. `ToolEngine` retains the supplied object by reference;
it does not make a shallow or deep copy.

## Creating a synchronous tool

Inherit from `SyncTool` and implement a fully annotated `invoke()` method:

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
    name = "add"

    @classmethod
    def invoke(
        cls,
        tool_input: AddInput,
        global_state: ToolEngineState,
    ) -> AddOutput:
        del global_state
        return AddOutput(value=tool_input.left + tool_input.right)
```

The framework infers `input_type` and `output_type` from the annotations.
`ainvoke()` uses `asyncio.to_thread()` by default so the synchronous
implementation does not block the event loop.

When `name` is omitted, the class name is exposed to the LLM.

## Creating an asynchronous tool

Inherit from `AsyncTool` and implement at least `emit_initial()` and
`emit_updates()`. Their annotations determine the input, state, and output
types.

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


class SearchInput(ToolInput):
    query: str


@dataclass
class SearchState(ToolState):
    pages_done: int = 0
    subscribed: bool = False
    stopped: bool = False


class SearchOutput(ToolOutput):
    content: str


class SearchTool(AsyncTool):
    name = "search"
    subscribe_by_default = False

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Running:
        del tool_input, global_state
        tool_state.call_id = tool_call_id
        return Running(content=f"Search started; call ID: {tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[SearchOutput]]:
        del global_state
        time.sleep(2)  # Simulate slow retrieval; the async bridge runs this in a thread.
        tool_state.pages_done = 1
        yield Running(content="Searching the first page")
        time.sleep(2)
        yield Finished(
            content=SearchOutput(content=f"Search complete: {tool_input.query}")
        )

    # Optional hooks for status, stopping, and subscription state.
    @classmethod
    def status(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> str:
        del tool_input, global_state
        return f"Retrieved {tool_state.pages_done} pages"

    @classmethod
    def stop(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.stopped = True

    @classmethod
    def subscribe(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = True

    @classmethod
    def unsubscribe(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = False
```

`emit_initial()` must return `Running` quickly. The current protocol requires
its `content` to contain `tool_call_id`. `emit_updates()` may emit multiple
`Running` values, but it must end with one `Finished`; ending without
`Finished` is a runtime error.

Default `aemit_initial()` and `aemit_updates()` methods bridge synchronous
implementations through worker threads. A tool backed by an async SDK may
override the corresponding `a*` methods.

### Optional lifecycle hooks

`AsyncTool` also provides synchronous and asynchronous hook pairs:

- `status()` / `astatus()`: return the current human-readable status.
- `stop()` / `astop()`: stop external work and release resources.
- `subscribe()` / `asubscribe()`: run extra work when progress is subscribed.
- `unsubscribe()` / `aunsubscribe()`: run extra work when it is unsubscribed.

## State and concurrency

`ToolState` belongs to one call. Store progress, external task handles, and
call-level locks there whenever possible. Different calls do not share a
`ToolState` instance.

`ToolEngineState` is shared at session scope. ToolEngine coordinates
synchronous calls and short lifecycle hooks, but it does **not** hold a global
lock for the entire `aemit_updates()` wait. An asynchronous update may wait on
network I/O, a queue, or an external event; holding the lock would prevent
`status` and `stop` from running.

If `aemit_updates()` mutates shared state, the tool must use short critical
sections, its own `asyncio.Lock`, or a thread-safe data structure. Do not keep a
shared-state lock across a network request or a complete update stream.

## ToolEngine

The supported union is:

```python
Tool = BaseTool | type[SyncTool] | type[AsyncTool]
```

Create an engine and bind its schemas:

```python
from langchain_core.messages import ToolCall
from xtalk.models.agents.tools import ToolEngine

engine = ToolEngine(
    tools=[AddTool, SearchTool],
    state={},
)
model_with_tools = engine.bind(model)

message = await engine.ainvoke(
    ToolCall(
        id="call-add-1",
        name="add",
        args={"left": 2, "right": 3},
    )
)
```

`bind()` exposes schemas to the model. A native tool must still execute through
ToolEngine and cannot invoke the schema wrapper directly.

### Invocation semantics

- `BaseTool`: call LangChain `ainvoke()` and store a textual result.
- `SyncTool`: validate input, await the final output, and return ToolMessage.
- `AsyncTool`: validate input, call `aemit_initial()`, store `AsyncToolRun`,
  start a background update task, and immediately return the initial
  ToolMessage.

`invoke()` is limited to `BaseTool` and `SyncTool` in synchronous contexts. Use
`await ainvoke()` when an event loop is already running. `AsyncTool` always
requires `await ainvoke()` on a persistent event loop so its updates can
continue after the initial result.

Each call ID is reserved before execution. Concurrent reuse or reuse after a
call has completed raises `ValueError`.

### Asynchronous run records

`AsyncToolRun` stores:

- the latest `Running` or `Finished`;
- the tool class, validated input, and call state;
- `subscribed` and `running` flags;
- the background `task`;
- a lifecycle lock and any background exception.

Background exceptions are stored on the run and retrieved to avoid unhandled
Task warnings.

## Assistant tools

When at least one `AsyncTool` exists, ToolEngine registers five assistant
tools:

- `async_tool_updated`: system-only update; its callable schema is hidden from
  the LLM.
- `id_to_async_tool_status`: query running state, tool status, and errors.
- `subscribe_async_tool`: subscribe and return the latest known result.
- `unsubscribe_async_tool`: suppress progress; the final result is still sent.
- `stop_async_tool`: call `astop()` and cancel the background task.

The latter four control tools are bound to the model. `async_tool_updated`
remains a ToolCall/ToolMessage protocol name only, preventing the model from
mistaking an internal notification for a callable tool.
Status queries are only for explicit user requests and must not be polled.
After subscribing, the model should stop calling tools and wait for proactive
system updates.

With `subscribe_by_default=True`, the subscription hook runs before the update
task starts. Unsubscribed calls proactively report only `Finished`; subscribed
calls report both `Running` and `Finished`.

## Message history protocol

Every ToolMessage requires a preceding AI ToolCall with the same ID.
`append_tool_message()` and `ainvoke_and_append()` preserve this constraint and
reject:

- empty call IDs;
- mismatched ToolCall and ToolMessage IDs;
- duplicate ToolMessages for one ID;
- one ID reused with a different name or arguments.

An asynchronous update uses a new `async_tool_updated` ToolCall whose arguments
contain the original `source_call_id`. Its message content has this form:

```json
{
  "running": true,
  "tool_output": "Searching the first page"
}
```

The final update sets `running` to `false` and stores the result of
`ToolOutput.to_content()` in `tool_output`.

## DefaultAgent integration

`DefaultAgent` creates one ToolEngine per session, binds the model, and
registers an asynchronous update callback. When no model generation is active,
the callback appends a valid ToolCall/ToolMessage pair and wakes the session
loop. Updates arriving during generation are deferred until it finishes, so
the current tool-call chain and a later loop cannot consume the same result.

When several updates arrive before generation starts, every downstream update
event is preserved while the model generations are coalesced into one. An
update arriving during generation triggers another generation afterward.
User requests use a model that may call tools. Proactive reports triggered by
background updates still bind every tool schema so the model can interpret
system ToolCalls in history, but set `tool_choice="none"` to prevent tool calls
during that generation.

### Concurrent ASR partials

ASR partials are stored as HumanMessages. If a tool update interrupts an
unfinished user message, later partials behave as follows:

- if the previous partial is a prefix, append only the new suffix;
- otherwise append the complete replacement text;
- if final ASR equals the latest partial, do not append an empty HumanMessage;
- always clear the final-generation flag after completion, failure, or
  cancellation.

While the user is still speaking, an asynchronous update is written to history
without immediately starting model generation. After final ASR, the model sees
both the tool update and the complete user input.

## Shutdown

Session shutdown must call:

```python
await engine.shutdown()
```

`shutdown()` rejects new calls, invokes `astop()` for active asynchronous
tools, cancels their background tasks, and waits for task completion. Repeated
shutdown calls are safe.

## Current limitations

- The LLM cannot yet provide a second input to a running tool.
- Applications and tools must agree on the concurrency policy of
  `ToolEngineState`.
- Real models may interpret tool descriptions and proactive updates
  differently, so optional end-to-end integration tests should complement unit
  tests.
