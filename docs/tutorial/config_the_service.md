# Configure the Service

This tutorial ends with a service that runs on a configuration you wrote yourself: you will read the config file structure, swap one model for another, add a Python tool during staged configuration, and turn on session recording.

Before starting, complete [Start the Service](start_the_service.md) so that `server.py` runs with the Quickstart `config.json`.

## Configuration file

`Xtalk.from_config(path/to/config.json)` reads a config file to instantiate X-Talk. A minimal config file looks like this:

```json
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": {
        "type": "DefaultAgent",
        "params": {
            "model": {
                "api_key": "<API_KEY>",
                "model": "qwen-plus-2025-12-01",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }
    },
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```

Keys such as `asr` represent model types, `type` selects the model for that model type, and `params` contains the model initialization arguments.

`params` maps directly onto the constructor of the selected class. For example, the `DefaultAgent` for the `llm_agent` model type is defined in [`src/xtalk/models/agents/default.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/default.py):

```python
class DefaultAgent(Agent):
    def __init__(
        self,
        model: BaseChatModel | dict[str, Any],
        backchannel_model: BaseChatModel | dict[str, Any] | None = None,
        backchannel_source_dir: str | Path | None = None,
        tools: list[Tool | Callable[[], Tool]] | None = None,
        system_prompt: str = "",
        proactive: bool = False,
    ) -> None:
        ...
```

To point this agent at a locally served OpenAI-compatible LLM, replace the `llm_agent` block with:

```json
{
  "llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      }
    }
  }
}
```

Other optional keys can be ignored; only `model` is required. Restart `uvicorn` after editing `config.json`; the next conversation is answered by the local LLM.

See [Supported Models](../reference/supported_models.md) for the full list of model types, their optional dependencies, and where they are adapted in the source code.

> **Note**
> Most model implementations are client-side adapters. You may also need to start the model instance itself according to its corresponding instructions.

## Staged configuration

Some settings are Python objects rather than JSON values. `Xtalk.configure(...)` opens a builder that lets you change the loaded config before models are instantiated. Here it adds a Python tool to the Agent:

```python
from langchain_core.tools import tool
from xtalk import Xtalk


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def add_multiply_tool(config: dict) -> dict:
    agent_config = config["llm_agent"]
    params = agent_config.setdefault("params", {})
    params.setdefault("tools", []).append(multiply)
    return config


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(add_multiply_tool)
    .build()
)
```

Replace the `Xtalk.from_config(...)` line in `server.py` with this builder and restart. Asking "what is twelve times seven" now returns the product computed by `multiply`.

Agent tools can also be added with the dedicated `.add_agent_tools([multiply])` method.

## Customize service behavior

Keys outside the model types configure the service itself. For example, save each session's audio under `logs/`:

```json
    "service_config": {
        "recording": true
    }
```

After the next conversation, `logs/` contains the recorded audio of that session.

See [Service Configuration](../reference/service_config.md) for the full list of service configuration options.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Connect a Client →](connect_a_client.md)** · Tutorial

-   **[Service Configuration →](../reference/service_config.md)** · Reference

-   **[Supported Models →](../reference/supported_models.md)** · Reference

-   **[Deploy with fully local models →](../how_to/deploy_fully_local.md)** · How-to

</div>
