# Introduce an LLM Agent

*Experimental feature*

An LLM Agent receives contexts such as speech recognition, response playback, and tool state, then streams text or tool calls. This guide starts with an Agent that only echoes finalized ASR text and progressively adds session state, tool calls, and persistent output.

## 1. Emit the First Response

```python
async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] == "asr_final":
        yield context["data"]["text"]
```

This example handles only the finalized recognition result `asr_final` and echoes its text as the response.

The runtime primarily calls `async_accept(context)`. Here, `context` is an `AgentContext` containing:

- `context["type"]`: the context type;
- `context["data"]`: data from the corresponding service event, excluding base event fields such as `session_id`.

This guide covers the following `context["type"]` values:

| Type | Main fields in `data` | Purpose |
| --- | --- | --- |
| `asr_partial` | `text`, `display_text`, `speech_pause`, `origin`, `turn_id`, `segment_id`, `gate_state` | Incremental speech-recognition text |
| `asr_final` | `text`, `display_text`, `origin`, `turn_id`, `segment_id`, `gate_state` | Final speech recognition or text input |
| `response_update` | `response_id`, `text` | Playback-confirmed partial assistant response |
| `response_finish` | `response_id`, `text` | Playback-confirmed completed or interrupted assistant response |
| `loop` | None | Persistent stream for proactive responses and asynchronous updates |

A custom Agent may ignore types it does not need. `loop` is the only persistent stream; a response from any other context finishes automatically when its output stream ends.

For example, this `loop` keeps waiting for proactive responses from a queue. `AgentTurnBoundary()` finishes each response while the stream continues waiting for the next one:

```python
import asyncio


def __init__(self) -> None:
    self.pending_updates: asyncio.Queue[str] = asyncio.Queue()

async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] != "loop":
        return

    while True:
        text = await self.pending_updates.get()
        yield text
        yield AgentTurnBoundary()
```

`pending_updates` is not provided by the `Agent` base class. It is a session-level asynchronous queue created by this custom Agent in `__init__()`. Other asynchronous tasks can put text into the queue for `loop` to emit in order.

`async_accept()` returns an asynchronous stream of `AgentOutput` items. Each item may be:

- `str`: response text sent to TTS and the frontend;
- `ToolCall`: a tool request containing its name, arguments, and call ID. The serving layer does not automatically execute arbitrary tools merely because an Agent emits a normal `ToolCall`. Tool invocation should happen inside the Agent. Currently, `ToolCall` mainly tells the serving layer that a tool call occurred;
- `ToolCallResult`: a completion notification containing the original tool name, arguments, and result text, telling the serving layer that the tool returned a result;
- `AgentTurnBoundary`: the end of the current response in a persistent stream, triggering TTS flush and response finish without ending the stream itself. Put simply, emit `yield AgentTurnBoundary()` after each complete response in `loop` has finished producing its output.

For example, these two text chunks belong to the same response, so the boundary is emitted only once after the complete response:

```python
yield "The weather today is "
yield "sunny and suitable for going out."
yield AgentTurnBoundary()
```

## 2. Complete the Agent Interface

A custom Agent inherits from `Agent` and implements `accept()`, `clone()`, and `restore_history()`. When the main logic lives in `async_accept()`, `accept()` can use the inherited `sync_iter_from_async()` helper. `@model` registers the implementation so it can be selected by configuration:

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput


@model
class EchoAgent(Agent):
    """Echo finalized ASR text."""

    def __init__(self) -> None:
        """Initialize conversation history."""

        self.messages: list[dict[str, Any]] = []

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Process context synchronously."""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Process context asynchronously."""

        if context["type"] == "asr_final":
            yield context["data"]["text"]

    def clone(self) -> "EchoAgent":
        """Create an Agent instance for a new session."""

        return EchoAgent()

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore persisted history."""

        self.messages = list(messages)
```

`clone()` must return an instance that a new session can use independently, preventing mutable state from being shared across sessions. The `restore_history()` example copies persisted messages into the current session's `messages` list.

## 3. Register and Enable the Agent

Change `llm_agent.type` to the registered class name:

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

Then create the service:

```python
from xtalk import Xtalk


xtalk_instance = Xtalk.from_config("config.json")
```

You can also select the class during staged configuration:

```python
xtalk_instance = (
    Xtalk.configure("config.json")
    .set_model(EchoAgent)
    .build()
)
```

## 4. Handle More Contexts

With these contexts defined, `async_accept()` can branch by type. Final input was handled earlier; add this branch if the Agent also needs incremental input:

```python
if context["type"] == "asr_partial":
    self.partial_text = context["data"]["text"]
```

## 5. Manage Conversation History

The Agent owns its conversation state. The earlier `EchoAgent` already stores persisted messages in `restore_history()`. Implement `get_chat_history()` when ASR or another component needs a textual representation of that history:

```python
def get_chat_history(self, with_system: bool = False) -> str | None:
    messages = self.messages if with_system else [
        message for message in self.messages if message["role"] != "system"
    ]
    return "\n".join(
        f'{message["role"]}: {message["content"]}' for message in messages
    )
```

`get_chat_history()` is optional. At the start of each recognition turn, ASR calls it to obtain the current conversation history and passes the result to the ASR model as `chat_history`; ASR models that do not use history ignore the value. `DefaultAgent` also handles `response_update` and `response_finish` to record what was actually played to the user. A custom Agent only needs these contexts when it requires the same semantics.

## 6. Tool Calls

See [Introduce Tools](tool_design.md).

## 7. Use Optional Helpers

`Agent` provides `content_to_text()` to convert strings or content blocks returned by LangChain into plain text.
