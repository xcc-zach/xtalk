<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.agents.tools

Agent tools for controlling TTS voice/emotion parameters.

Includes tool definitions/factories for LLM tool-calling usage or prompt docs
that help the model produce structured tool-call outputs.

## AsyncTool

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
class AsyncTool(_NativeTool)
```

Base class for long-running tools that emit incremental updates.

Subclasses implement ``emit_initial`` and ``emit_updates``. Default async
methods run their synchronous counterparts in worker threads. Tools backed
by native async SDKs may override the corresponding ``a*`` methods.

### 类字段

- `subscribe_by_default: ClassVar[bool]` = `False`
- `input_type: ClassVar[type[ToolInput]]`
- `state_type: ClassVar[type[ToolState]]`
- `output_type: ClassVar[type[ToolOutput]]`

### 方法

#### emit_initial

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def emit_initial(cls, tool_call_id: str, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> Running
```

Immediately produce the first protocol-compliant tool result.

Some models require every tool call to be followed immediately by a
ToolMessage. This method returns ``Running`` before the actual work
continues through ``emit_updates``.

#### aemit_initial

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def aemit_initial(cls, tool_call_id: str, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> Running
```

Produce the initial result in a worker thread.

#### emit_updates

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def emit_updates(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> Iterator[ToolResult[ToolOutput]]
```

Yield incremental updates and end with one final result.

#### aemit_updates

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def aemit_updates(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> AsyncIterator[ToolResult[ToolOutput]]
```

Bridge the synchronous update iterator into an async iterator.

#### status

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def status(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str
```

Return the latest human-readable status for one call.

Concrete tools may override this hook for model-initiated status
queries. The default empty string means no extra status is available.

#### astatus

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def astatus(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> str
```

Query synchronous tool status without blocking the event loop.

#### stop

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def stop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Perform tool-specific cleanup when a call is stopped.

Tools that own external tasks or connections should override this hook.
Cancelling the ToolEngine background task is handled by ToolEngine.

#### astop

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def astop(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Run the synchronous stop hook without blocking the event loop.

#### subscribe

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def subscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Perform tool-specific work when progress updates are subscribed.

Subscription affects intermediate updates only. ToolEngine always
writes the final result back to message history.

#### asubscribe

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def asubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Run the synchronous subscribe hook without blocking the event loop.

#### unsubscribe

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def unsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Perform tool-specific work when progress updates are unsubscribed.

Unsubscription suppresses intermediate updates only. ToolEngine still
writes the final result back to message history.

#### aunsubscribe

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def aunsubscribe(cls, tool_input: ToolInput, tool_state: ToolState, global_state: ToolEngineState) -> None
```

Run the synchronous unsubscribe hook without blocking the event loop.

## Finished

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
@dataclass(frozen=True)
class Finished(Generic[TO])
```

Structured final result emitted by an asynchronous tool.

``content`` is the concrete ``ToolOutput`` instance declared by the tool.

### 类字段

- `content: TO`

## Running

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
@dataclass(frozen=True)
class Running
```

Text update emitted while an asynchronous tool is still running.

``content`` contains the intermediate state stored in a ToolMessage.

### 类字段

- `content: str`

## SyncTool

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
class SyncTool(_NativeTool)
```

Base class for native tools that return one structured final result.

Subclasses implement ``invoke``. The framework-provided ``ainvoke`` runs
the synchronous implementation in a worker thread.

### 类字段

- `input_type: ClassVar[type[ToolInput]]`
- `output_type: ClassVar[type[ToolOutput]]`

### 方法

#### invoke

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def invoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput
```

Execute the tool and return its final structured output.

#### ainvoke

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def ainvoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput
```

Run the synchronous implementation without blocking the event loop.

## Tool

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
Tool: TypeAlias
```

**值:** `BaseTool | type[SyncTool] | type[AsyncTool]`

## ToolEngineState

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
ToolEngineState: TypeAlias
```

**值:** `Any`

## ToolInput

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
class ToolInput(BaseModel)
```

Base input model for X-Talk native tools.

## ToolOutput

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
class ToolOutput(BaseModel)
```

Base output model for X-Talk native tools.

### 方法

#### to_content

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def to_content(self) -> str
```

Serialize the structured result for storage in a ToolMessage.

## ToolResult

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
ToolResult: TypeAlias
```

**值:** `Running | Finished[TO]`

## ToolState

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
@dataclass
class ToolState
```

Mutable state owned by one asynchronous tool call.

``call_id`` uniquely identifies the call. ``metadata`` stores custom
progress and state owned by the concrete tool.

### 类字段

- `call_id: str` = `''`
- `metadata: dict[str, Any]` = `field(default_factory=dict)`

## ToolRun

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
@dataclass
class ToolRun
```

Result record for one tool call.

``result`` stores the latest available tool result.

### 类字段

- `result: ToolResult`

## AsyncToolRun

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
@dataclass
class AsyncToolRun(ToolRun)
```

Lifecycle record for one asynchronous tool call.

### 类字段

- `tool_class: type[AsyncTool]`
- `tool_input: ToolInput`
- `tool_state: ToolState`
- `subscribed: bool`
- `running: bool`
- `lifecycle_lock: asyncio.Lock` = `field(default_factory=asyncio.Lock)`
- `task: asyncio.Task[None] | None` = `None`
- `error: BaseException | None` = `None`

## ToolEngine

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
class ToolEngine
```

Manage agent tool binding, invocation, and lifecycle state.

### 方法

#### __init__

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def __init__(self, tools: list[Tool], state: ToolEngineState) -> None
```

Initialize the tool engine.

#### append_tool_message

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def append_tool_message(tool_call: ToolCall, tool_message: ToolMessage, messages: list[BaseMessage]) -> None
```

Append a tool call and its result to message history.

#### ainvoke_and_append

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def ainvoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]) -> ToolMessage
```

Invoke a tool asynchronously and append its result to history.

#### invoke_and_append

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def invoke_and_append(self, tool_call: ToolCall, messages: list[BaseMessage]) -> ToolMessage
```

Invoke a tool synchronously and append its result to history.

#### bind

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def bind(self, model: BaseChatModel) -> BaseChatModel | Runnable[Any, Any]
```

Bind all engine-managed tools to a chat model.

#### bind_without_tool_calls

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def bind_without_tool_calls(self, model: BaseChatModel) -> BaseChatModel | Runnable[Any, Any]
```

Bind tool schemas while preventing model-initiated calls.

##### 参数

- `model` (`BaseChatModel`)
  Chat model that should receive the registered tool schemas.

##### 返回

- `BaseChatModel or Runnable[Any, Any]`
  Model binding that exposes the schemas with tool choice disabled.

#### on_async_tool_update

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def on_async_tool_update(self, callback: Callable[[ToolCall, ToolMessage], None]) -> None
```

Register the callback for proactive asynchronous tool updates.

#### ainvoke

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def ainvoke(self, tool_call: ToolCall) -> ToolMessage
```

Invoke a tool asynchronously.

#### invoke

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
def invoke(self, tool_call: ToolCall) -> ToolMessage
```

Invoke a synchronous tool from a synchronous context.

AsyncTool requires a persistent event loop for subsequent updates and
must use ``ainvoke``. Callers already inside an event loop should also
await ``ainvoke`` directly.

#### shutdown

_定义于 [`xtalk.models.agents.tools.core`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/core.py)。_

```python
async def shutdown(self) -> None
```

Close the engine and cancel all active asynchronous tool calls.

## build_set_voice_tool

_定义于 [`xtalk.models.agents.tools.speech_control`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/speech_control.py)。_

```python
def build_set_voice_tool(available_voice_names: Optional[List[str]] = None)
```

Create a compact tool for switching TTS voice.

## build_set_emotion_tool

_定义于 [`xtalk.models.agents.tools.speech_control`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/speech_control.py)。_

```python
def build_set_emotion_tool(available_emotions: Optional[List[str]] = None)
```

Create a compact tool for switching speech emotion.

## build_silence_tool

_定义于 [`xtalk.models.agents.tools.speech_control`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/speech_control.py)。_

```python
def build_silence_tool()
```

Create a tool that only displays text without audio.

## build_set_speed_tool

_定义于 [`xtalk.models.agents.tools.speech_control`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/speech_control.py)。_

```python
def build_set_speed_tool(*, min_speed: float = 0.5, max_speed: float = 2.0)
```

Create a compact tool for adjusting speaking speed.

## AVAILABLE_EMOTIONS

_定义于 [`xtalk.models.agents.tools.speech_control`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/speech_control.py)。_

```python
AVAILABLE_EMOTIONS: List[str]
```

**值:** `['happy', 'angry', 'sad', 'fear', 'disgust', 'depressed', 'surprised', 'calm', 'normal']`

## build_web_search_tool

_定义于 [`xtalk.models.agents.tools.retrievers`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/retrievers.py)。_

```python
def build_web_search_tool() -> BaseTool
```

Build a Serper-based web search tool with graceful degradation.

## build_time_tool

_定义于 [`xtalk.models.agents.tools.retrievers`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/retrievers.py)。_

```python
def build_time_tool() -> BaseTool
```

Build a current-time tool with optional timezone, format, and date offset.

## ThinkInput

_定义于 [`xtalk.models.agents.tools.thinking`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/thinking.py)。_

```python
class ThinkInput(ToolInput)
```

Input for the asynchronous thinking tool.

### 类字段

- `question: str`

## ThinkOutput

_定义于 [`xtalk.models.agents.tools.thinking`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/thinking.py)。_

```python
class ThinkOutput(ToolOutput)
```

Output returned by the asynchronous thinking tool.

### 类字段

- `answer: str` = `''`
- `error: str | None` = `None`

## build_think_tool

_定义于 [`xtalk.models.agents.tools.thinking`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/thinking.py)。_

```python
def build_think_tool(model: BaseChatModel | dict[str, Any], delay_seconds: float = 0.0) -> type[AsyncTool]
```

Build a tool that delegates reasoning to an upstream LLM.

### 参数

- `model` (`BaseChatModel | dict[str, Any]`)
  Upstream model or model configuration used for reasoning.
- `delay_seconds` (`float, optional`)
  Artificial delay before invoking the upstream model.

### 返回

- `type[AsyncTool]`
  Configured asynchronous thinking tool class.

## AsyncWebSearchInput

_定义于 [`xtalk.models.agents.tools.async_web_search`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/async_web_search.py)。_

```python
class AsyncWebSearchInput(ToolInput)
```

Input accepted by the asynchronous web search tool.

### 类字段

- `query: str` = `Field(min_length=1)`
- `max_results: int` = `Field(default=5, ge=1, le=10)`
- `region: str | None` = `None`
- `lang: str | None` = `None`

## AsyncWebSearchOutput

_定义于 [`xtalk.models.agents.tools.async_web_search`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/async_web_search.py)。_

```python
class AsyncWebSearchOutput(ToolOutput)
```

Final result returned by the asynchronous web search tool.

### 类字段

- `results: str`

## build_async_web_search_tool

_定义于 [`xtalk.models.agents.tools.async_web_search`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/async_web_search.py)。_

```python
def build_async_web_search_tool() -> type[AsyncTool]
```

Build an asynchronous wrapper around the existing web search tool.

### 返回

- `type[AsyncTool]`
  Asynchronous web search tool class.

## DeepResearchInput

_定义于 [`xtalk.models.agents.tools.deepresearch`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/deepresearch.py)。_

```python
class DeepResearchInput(ToolInput)
```

Input accepted by the deep research tool.

### 类字段

- `topic: str` = `Field(min_length=1)`
- `description: str` = `''`

## DeepResearchOutput

_定义于 [`xtalk.models.agents.tools.deepresearch`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/deepresearch.py)。_

```python
class DeepResearchOutput(ToolOutput)
```

Final result returned by the deep research tool.

### 类字段

- `report: str` = `''`
- `sources: list[str]` = `Field(default_factory=list)`
- `error: str | None` = `None`

## DeepResearchState

_定义于 [`xtalk.models.agents.tools.deepresearch`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/deepresearch.py)。_

```python
@dataclass
class DeepResearchState(ToolState)
```

Mutable progress state for one deep research call.

### 类字段

- `phase: str` = `'starting'`
- `round_number: int` = `0`
- `queries: list[str]` = `field(default_factory=list)`
- `completed_queries: int` = `0`
- `current_query: str` = `''`

## build_deep_research_tool

_定义于 [`xtalk.models.agents.tools.deepresearch`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/tools/deepresearch.py)。_

```python
def build_deep_research_tool(model: BaseChatModel | dict[str, Any], max_rounds: int = 5, max_total_searches: int = 15) -> type[AsyncTool]
```

Build an asynchronous LLM-guided deep research tool.

### 参数

- `model` (`BaseChatModel | dict[str, Any]`)
  Non-networked LLM used to plan searches and synthesize the report.
- `max_rounds` (`int, optional`)
  Maximum number of times the LLM may request another search round.
- `max_total_searches` (`int, optional`)
  Maximum number of web searches allowed across all rounds.

### 返回

- `type[AsyncTool]`
  Deep research tool class.
