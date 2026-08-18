> **Note**
> See [`examples/sample_app/custom_model.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_model.py) for the complete example. The example defines an `EchoAgent` in the server file, registers it with `@model`, and replaces the configured `llm_agent` during staged configuration.

You may want to introduce a new model for an existing model type, such as a new LLM Agent. This page uses `custom_model.py` as an example and walks through adding a new `EchoAgent`. This agent reads the final ASR text and returns that text directly as the assistant response.

## 1. Import the Model Interface and Registration Decorator

`EchoAgent` belongs to the existing `Agent` model type, so it should inherit from `xtalk.model_types.Agent`. Use `@model` to register the implementation class in the model registry. For interface details, see the [Agent API](../api/server/xtalk/models/agents/interfaces.md).

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
```

## 2. Define and Register the Model Implementation

```python
@model
class EchoAgent(Agent):
    """A simple agent that echoes finalized ASR text."""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Synchronously bridge ``async_accept()`` for compatibility."""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Emit finalized ASR text."""

        if context["type"] != "asr_final":
            return
        text = context["data"]["text"]
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Ignore persisted history."""

        del messages

    def clone(self) -> "EchoAgent":
        """Create a fresh stateless agent."""

        return EchoAgent()

```

Notes:

- `async_accept` is the main async runtime entrypoint.
- `clone()` should return a model instance suitable for a new session, avoiding shared mutable state across sessions.
- For a complete development guide, see [Introduce an LLM Agent](introduce_an_llm_agent.md).

## 3. Select the New Model

Change the `llm_agent` `type` to `EchoAgent` and continue using
`Xtalk.from_config(...)`:

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

When the base config should remain reusable, select the registered Python class
during staged configuration instead:

```python
def clear_agent_params(config: dict[str, Any]) -> dict[str, Any]:
    agent_config = config.get("llm_agent")
    if isinstance(agent_config, dict):
        agent_config["params"] = {}
    return config


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
```

## 4. Register Before Creating Xtalk

Because `EchoAgent` is defined in the same server file, Python executes the
class definition and `@model` registration before reaching `set_model(...)`.

```python
xtalk_instance = (
    Xtalk.configure(args.config)
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
xtalk_instance.mount_routes(app)
```

If the model is defined in a separate file, import that file first:

```python
from my_app.echo_agent import EchoAgent

xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
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
