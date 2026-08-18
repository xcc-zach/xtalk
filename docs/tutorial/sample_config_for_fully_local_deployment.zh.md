以下是所有模型均在本地运行时的 X-Talk 示例配置。配置使用 SherpaOnnxASR、OpenAI 兼容的本地 LLM、IndexTTS、TurnSense、Qwen3ForcedAligner 和 FastEnhancer。请先分别启动对应的本地服务，并根据实际监听地址修改 `base_url`。各模型的依赖和部署方式请参阅[支持的模型](../technical_reference/supported_models.zh.md)。

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
