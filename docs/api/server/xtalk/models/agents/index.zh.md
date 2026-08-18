<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.agents

## Agent

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
@model_type(aliases=['llm_agent'])
class Agent(ABC)
```

Abstract interface for conversational agents used by Xtalk.

### 方法

#### content_to_text

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def clone(self) -> 'Agent'
```

Clone the agent for a new session.

##### 返回

- `Agent`
  Session-safe agent instance.

#### restore_history

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def restore_history(self, messages: list[dict[str, Any]]) -> None
```

Restore persisted conversation messages into the agent state.

##### 参数

- `messages` (`list[dict[str, Any]]`)
  Persisted chat messages ordered by session history.

#### get_chat_history

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

## AgentContext

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

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

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
AgentOutput
```

**值:** `Union[str, ToolCall, ToolCallResult, AgentTurnBoundary]`

## AgentTurnBoundary

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
@dataclass(frozen=True)
class AgentTurnBoundary
```

Mark the end of one agent response segment.
