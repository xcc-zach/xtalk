<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.agents.interfaces

## AgentContext

```python
class AgentContext(TypedDict)
```

Incremental context update accepted by an agent.

### 说明

``type`` identifies the logical context stream, while ``data`` carries the
event-derived payload for that stream.

### 类字段

- `type: str`
- `data: dict[str, Any]`

## AgentOutput

```python
AgentOutput
```

**值:** `Union[str, ToolCall, ToolCallResult]`

## T

```python
T
```

**值:** `TypeVar('T')`

## PlaybackAIMessageMeta

```python
@dataclass
class PlaybackAIMessageMeta
```

Track merge state for one playback-managed assistant message.

### 类字段

- `final: bool` = `False`
- `prefix: str | None` = `None`

## ChatHistory

```python
class ChatHistory
```

Manage chat history plus playback-aware assistant-message merging.

### 方法

#### __init__

```python
def __init__(self, system_prompt: str) -> None
```

Initialize the history with one system message.

##### 参数

- `system_prompt:`
  The system prompt to place at the start of the message list.

#### messages

```python
def messages(self) -> list[BaseMessage]
```

Return the current chat-history message list.

#### append_message

```python
def append_message(self, message: BaseMessage) -> None
```

Append one message to the history unchanged.

##### 参数

- `message:`
  The message to append.

#### append_or_update_ai_message

```python
def append_or_update_ai_message(self, full_text: str, *, final: bool) -> None
```

Append or merge one playback-managed assistant message.

##### 参数

- `full_text:`
  The cumulative assistant text confirmed by playback.
- `final:`
  Whether this update closes the playback-managed assistant message.

## Agent

```python
@model_type(aliases=['llm_agent'])
class Agent(ABC)
```

Abstract interface for conversational agents used by Xtalk.

### 方法

#### content_to_text

```python
def content_to_text(content: Any) -> str
```

Normalize model content blocks into plain text.

##### 参数

- `content:`
  Content emitted by a LangChain model chunk or message.

##### 返回

- `str`
  Plain-text content extracted from the input.

#### accept

```python
def accept(self, context: AgentContext) -> Iterable[AgentOutput]
```

Accept an incremental context update.

##### 参数

- `context` (`AgentContext`)
  Context payload forwarded from serving-layer events.

##### 生成

- `AgentStreamItem`
  Zero or more streamed response items triggered by the context
  update.

#### async_accept

```python
async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]
```

Asynchronously accept an incremental context update.

##### 参数

- `context` (`AgentContext`)
  Context payload forwarded from serving-layer events.

##### 生成

- `AgentStreamItem`
  Streamed response items triggered by the context update.

#### sync_iter_from_async

```python
def sync_iter_from_async(self, async_iter: AsyncIterator[T]) -> Iterable[T]
```

Convert an async iterator into a synchronous generator.

##### 参数

- `async_iter` (`AsyncIterator[T]`)
  Async iterator to bridge into synchronous iteration.

##### 生成

- `T`
  Items produced by ``async_iter``.

#### clone

```python
def clone(self) -> 'Agent'
```

Clone the agent for a new session.

##### 返回

- `Agent`
  Session-safe agent instance.

#### restore_history

```python
def restore_history(self, messages: list[dict[str, Any]]) -> None
```

Restore persisted conversation messages into the agent state.

##### 参数

- `messages` (`list[dict[str, Any]]`)
  Persisted chat messages ordered by session history.

#### get_chat_history

```python
def get_chat_history(self, with_system: bool = False) -> str | None
```

Return the serialized conversation history when available.

##### 参数

- `with_system` (`bool, optional`)
  Whether to include the system prompt message when supported by the
  concrete implementation.

##### 返回

- `str | None`
  Conversation history or ``None``.

#### add_tools

```python
def add_tools(self, tools: list[BaseTool | Callable[[], BaseTool]]) -> None
```

Attach tools to the agent.

##### 参数

- `tools` (`list[BaseTool | Callable[[], BaseTool]]`)
  Tool instances or factories that produce tool instances.
