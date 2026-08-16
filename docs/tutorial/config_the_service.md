# Configure the service

## Configuration file

As mentioned in [Start the Service](start_the_service.md), `Xtalk.from_config(path/to/config.json)` reads a config file to instantiate X-Talk. A minimal config file looks like this:

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

For example, the `DefaultAgent` for the `llm_agent` model type is defined in [`src/xtalk/models/agents/default.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/default.py):
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

To match those initialization arguments, the config item should look like:
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

Other optional keys can be ignored; only `model` is required.

See [Supported Models](../technical_reference/supported_models.md) for the full list of model types, their optional dependencies, and where they are adapted in the source code.
> **Note**
> Most model implementations are client-side adapters. You may also need to start the model instance itself according to its corresponding instructions.

## Staged configuration

Use `Xtalk.configure(...)` to add a Python tool to the Agent before models are
instantiated:

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

Agent tools can also be added with the dedicated `.add_agent_tools([multiply])` method.

## Customize service behavior

You can also customize service behavior, such as whether to save session audio under `logs/`:
```json
    "service_config": {
        "recording": true
    }
```

See [all service configuration](service_config.md) for the full list of service configuration options.


## Frontend configuration

The frontend accepts a `SessionConfig` through `createSession(wsUrl, config)`. It currently supports three groups of options:

- `inputConfig`
- `outputConfig`
- `serviceURLs`

For example:

```ts
const session = createSession(wsUrl, {
    inputConfig: {
        sampleRate: 16000,
        enableVAD: true,
        enableEnhancer: true,
        vadRedemptionMs: 500,
        frontendUtilitiesBaseUrl: "/xtalk/frontend-utilities",
    },
    outputConfig: {
        sampleRate: 48000,
    },
    serviceURLs: {
        login: "/api/auth/login",
        sessions: "/api/sessions",
        sessionDetail: (sessionId) => `/api/sessions/${sessionId}`,
        upload: "/api/upload",
    },
});
```

### inputConfig

`inputConfig` controls the frontend input audio session. In normal browser microphone mode, the most commonly used fields are:

- `sampleRate`
  Input audio sample rate. The default is `16000`.
- `enableVAD`
  Whether to enable frontend VAD. The default is `true`.
- `enableEnhancer`
  Whether to enable frontend speech enhancement. The default is `true`.
- `vadRedemptionMs`
  VAD redemption window in milliseconds. The default is `500`.
- `frontendUtilitiesBaseUrl`
  Base URL for browser-side ONNX Runtime, VAD, and FastEnhancer assets. The default is `/xtalk/frontend-utilities`. If ONNX Runtime or VAD assets are unavailable at the corresponding server path, the frontend loads them from public CDNs instead.

### outputConfig

`outputConfig` currently mainly supports:

- `sampleRate`
  Output playback sample rate. The default is `48000`.

### serviceURLs

`serviceURLs` overrides server HTTP endpoints. It currently supports:

- `login`
- `sessions`
- `sessionDetail`
  This can be either a fixed URL or a function of the form `(sessionId) => URL`.
- `upload`

If omitted, the frontend derives default URLs automatically from `wsUrl`:

- `POST /api/auth/login`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/upload`
