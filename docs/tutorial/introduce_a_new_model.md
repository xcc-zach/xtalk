# Introduce a New Model

This tutorial ends with the service answering through a model you registered yourself. You will add an `EchoAgent` for the existing `Agent` model type: it reads the final ASR text and returns that text directly as the assistant response, so every reply repeats what you said.

Before starting, complete [Configure the Service](config_the_service.md) so that you know how `type` and `params` select a model.

> **Note**
> See [`examples/sample_app/custom_model.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_model.py) for the complete example. The example defines `EchoAgent` in the server file, registers it with `@model`, and replaces the configured `llm_agent` during staged configuration.

## 1. Import the Model Interface and Registration Decorator

`EchoAgent` belongs to the existing `Agent` model type, so it inherits from `xtalk.model_types.Agent`. `@model` registers the implementation class in the model registry, which is what lets a config file select it by class name. For interface details, see the [Agent API](../api/server/xtalk/models/agents/interfaces.md).

Add these imports to `server.py`:

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

    def clone(self) -> "EchoAgent":
        """Create a fresh stateless agent."""

        return EchoAgent()

```

`async_accept` is the main async runtime entrypoint; `accept` bridges it for callers that need a synchronous iterator. `clone()` returns a model instance for a new session, which keeps mutable state from being shared across sessions.

Defining the class has no visible effect yet: the config still selects `DefaultAgent`. The next step switches to `EchoAgent`.

## 3. Select the New Model

Change the `llm_agent` `type` to `EchoAgent` and continue using `Xtalk.from_config(...)`:

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

Restart the server and speak a sentence. The reply is your own sentence read back by TTS.

When the base config should remain reusable, keep `DefaultAgent` in the file and select the registered Python class during staged configuration instead:

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

`clear_agent_params` drops the `DefaultAgent` constructor arguments that `EchoAgent` does not accept. Without it `build()` fails while constructing `EchoAgent` with unexpected keyword arguments.

## 4. Register Before Creating Xtalk

Because `EchoAgent` is defined in the same server file, Python executes the class definition and `@model` registration before reaching `set_model(...)`.

```python
xtalk_instance = (
    Xtalk.configure(args.config)
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
app = xtalk_instance.create_asgi_app()
```

For running the application, see [Connect a Client](connect_a_client.md).

If the model is defined in a separate file, import that file first so the registration runs before `build()`:

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

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Introduce an LLM Agent →](introduce_an_llm_agent.md)** · Tutorial

-   **[Supported Models and where each is adapted →](../reference/supported_models.md)** · Reference

-   **[Model `clone()` and `reset()` →](../reference/model_clone_reset.md)** · Reference

</div>
