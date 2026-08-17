# Introduce an LLM Agent

An LLM Agent receives contexts such as ASR results, playback state, and tool state, then streams response text or tool calls. This guide implements an `EchoAgent` that returns finalized ASR text.

## Implement the Agent

Inherit from `Agent`, register the implementation with `@model`, and implement `accept()`, `async_accept()`, `restore_history()`, and `clone()`:

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput, AgentTurnBoundary


@model
class EchoAgent(Agent):
    """Echo finalized ASR text."""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Process context synchronously."""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Process context asynchronously."""

        if context["type"] != "asr_final":
            return
        text = context["data"]["text"]
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Restore history; a stateless agent may ignore it."""

        del messages

    def clone(self) -> "EchoAgent":
        """Create an Agent instance for a new session."""

        return EchoAgent()
```

The runtime primarily calls `async_accept()`. `context["type"]` identifies the context and `context["data"]` contains its payload. Common types include `asr_final`, `asr_partial`, `multi_speaker_final`, `response_update`, `response_finish`, and the long-lived `loop` context used for asynchronous updates.

`AgentOutput` may be response text, a tool call, a tool-call result, or `AgentTurnBoundary`. A finite output stream finishes its current response when the iterator ends. A stream started by `loop` remains active, so it should emit `AgentTurnBoundary()` after each response to trigger TTS flush and response finish without ending the stream.

## Handle Persistent Output

An Agent that responds proactively to asynchronous tool updates can wait for updates in the `loop` context:

```python
async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] == "asr_final":
        text = context["data"]["text"]
        if text:
            yield text
        return

    if context["type"] == "loop":
        while True:
            text = await self.pending_updates.get()
            yield text
            yield AgentTurnBoundary()
```

`AgentTurnBoundary()` ends only the current response, not the `loop` stream. Session state such as the queue must not be shared across sessions, so `clone()` should return an instance with independent state.

## Enable the Agent

Set `llm_agent.type` to the registered class name in the JSON configuration:

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

You can also select the class during staged configuration:

```python
xtalk_instance = (
    Xtalk.configure("config.json")
    .set_model(EchoAgent)
    .build()
)
```

`set_model()` preserves the existing `params`, so they must be compatible with the custom Agent constructor. `clone()` should create an independently usable instance for each session. If the Agent keeps conversation history, restore the supplied messages in `restore_history()`.

See the [Agent API](../api/server/xtalk/models/agents/interfaces.md) for the complete interface.
