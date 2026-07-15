> **Note**
> See [`examples/sample_app/custom_model.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_model.py) for the complete example. The example defines an `EchoAgent` in the server file, registers it with `@model`, and replaces the default `llm_agent` with this custom agent through config.

You may want to introduce a new model for an existing model type, such as a new text-to-speech model. This page uses `custom_model.py` as an example and walks through adding a new `EchoAgent`. This agent reads the final ASR text and returns that text directly as the assistant response.

## 1. Import the Model Interface and Registration Decorator

`EchoAgent` belongs to the existing `Agent` model type, so it should inherit from `xtalk.model_types.Agent`. Use `@model` to register the implementation class in the model registry. For model interface details, see [ASR Design](../docs/asr_design.md) and related documents.

```python
import asyncio
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
```

## 2. Define and Register the Model Implementation

The key step is adding `@model` to the class. By default, the config `type` can use the class name directly: `EchoAgent`.

```python
@model
class EchoAgent(Agent):
    """A simple agent that echoes finalized ASR text."""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Synchronously bridge ``async_accept()`` for compatibility."""

        yield from self._sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        if str(context.get("type", "") or "") != "asr_final":
            return
        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", ""))
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        del messages

    def clone(self) -> "EchoAgent":
        return EchoAgent()

    def _sync_iter_from_async(
        self,
        async_iter: AsyncIterator[AgentOutput],
    ) -> Iterable[AgentOutput]:
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    item = loop.run_until_complete(async_iter.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        finally:
            loop.close()
```

Notes:

- `@model` must run before `Xtalk.from_config(...)`, which means the module defining the class must be imported first.
- `async_accept` is the main async runtime entrypoint.
- `accept` bridges to `async_accept` for synchronous compatibility.
- `clone()` should return a model instance suitable for a new session, avoiding shared mutable state across sessions.
- Finite output streams finish the current response when their iterator ends.
  A long-lived stream started by `loop` should also yield
  `AgentTurnBoundary()` after each response. This triggers TTS flush and
  response finish without ending the stream itself.

## 3. Use the New Model in Config

The model type is still `llm_agent`; only change that type's `type` value to `EchoAgent`.

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

## 4. Register Before Creating Xtalk

Because `EchoAgent` is defined in the same server file, Python executes the class definition and `@model` registration before reaching `Xtalk.from_config(...)`.

```python
with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

xtalk_instance = Xtalk.from_config("path/to/config.json")
xtalk_instance.mount_routes(app)
```

If the model is defined in a separate file, import that file first:

```python
import my_app.echo_agent

xtalk_instance = Xtalk.from_config("path/to/config.json")
```

## 5. Complete Config Example

The following config keeps the original ASR and TTS while replacing the agent with `EchoAgent`:

```json
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    },
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```
