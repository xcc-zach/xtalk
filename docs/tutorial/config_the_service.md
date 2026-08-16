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
            model: BaseChatModel | dict,
            system_prompt: str = _BASE_PROMPT,
            voice_names: Optional[List[str]] = None,
            emotions: Optional[List[str]] = None,
            tools: Optional[List[Union[BaseTool, Callable[[], BaseTool]]]] = None,
        ):
    ...
```

To match those initialization arguments, the config item should look like:
```
"llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      },
      "voice_names": [
        "Man",
        "Woman",
        "Child"
      ],
      "emotions": [
        "happy",
        "angry",
        "sad",
        "fear",
        "disgust",
        "depressed",
        "surprised",
        "calm",
        "normal"
      ]
    }
  },
```

Optional keys such as `voice_names`, `emotions`, and `tools` can be omitted. `tools` is not supported in config yet.

See [Supported Models](../technical_reference/supported_models.md) for the full list of model types, their optional dependencies, and where they are adapted in the source code.
> **Note**
> Most model implementations are client-side adapters. You may also need to start the model instance itself according to its corresponding instructions.

## Staged configuration

Use `Xtalk.from_config(...)` when the JSON file or configuration dictionary is
already complete. Use `Xtalk.configure(...)` when Python code needs to modify
the configuration before models are instantiated:

```python
def enable_recording(config: dict) -> dict:
    updated_config = dict(config)
    service_config = dict(updated_config.get("service_config", {}))
    service_config["recording"] = True
    updated_config["service_config"] = service_config
    return updated_config


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(enable_recording)
    .build()
)
```

`transform_config()` accepts a `dict -> dict` function. Transformations run in
registration order and receive a structural copy of the source config.
Prefer focused Builder methods such as `set_model()` and
`add_agent_tools()` when they express the required change directly.


## Customize service behavior

You can also customize service behavior, such as whether to save session audio under `logs/` and whether to send full session audio back to the client:
```json
    "service_config": {
        "recording": true,
        "send_full_audio_to_client": true
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
