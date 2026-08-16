The following X-Talk example configuration runs every model locally. It uses SherpaOnnxASR, an OpenAI-compatible local LLM, IndexTTS, TurnSense, Qwen3ForcedAligner, and FastEnhancer. Start each local service first and adjust its `base_url` to match the actual listening address. See [Supported Models](../technical_reference/supported_models.md) for model dependencies and deployment instructions.

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
