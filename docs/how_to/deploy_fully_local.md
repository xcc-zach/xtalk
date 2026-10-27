# Deploy Fully Local

This guide replaces the cloud models from the [Quickstart](../quickstart.md) with a configuration in which every model runs on your own machines.

The configuration below uses SherpaOnnxASR, an OpenAI-compatible local LLM, IndexTTS, TurnSense, Qwen3ForcedAligner, and FastEnhancer. Start each local service first, then set its `base_url` to the address it listens on. Installation and dependencies for each model are listed in [Supported Models](../reference/supported_models.md).

```json
{
  "asr": {
    "type": "SherpaOnnxASR",
    "params": {
      "base_url": "ws://127.0.0.1:6006",
      "mode": "offline"
    }
  },
  "llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      }
    }
  },
  "tts": {
    "type": "IndexTTS",
    "params": {
      "base_url": "http://127.0.0.1:11996",
      "voices": [
        {
          "name": "Man",
          "path": "ReferenceVoice/Man"
        },
        {
          "name": "Woman",
          "path": "ReferenceVoice/Woman"
        },
        {
          "name": "Child",
          "path": "ReferenceVoice/Child"
        }
      ]
    }
  },
  "turn_detector": {
    "type": "TurnSense",
    "params": {
      "base_url": "http://127.0.0.1:8003"
    }
  },
  "forced_aligner": {
    "type": "Qwen3ForcedAligner",
    "params": {
      "base_url": "http://127.0.0.1:8001",
      "model": "Qwen/Qwen3-ForcedAligner-0.6B"
    }
  },
  "speech_enhancer": {
    "type": "FastEnhancer",
    "params": {
      "base_url": "http://127.0.0.1:8005"
    }
  }
}
```

Start the server with this file as `--config`, as in the Quickstart.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Supported Models →](../reference/supported_models.md)** · Reference

-   **[Configure the Service →](../tutorial/config_the_service.md)** · Tutorial

-   **[Mount in FastAPI →](mount_in_fastapi.md)** · How-to

</div>
