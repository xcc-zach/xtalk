# 完全本地部署

本指南把[快速开始](../quickstart.zh.md)中的云端模型换成全部在自有机器上运行的配置。

下面的配置使用 SherpaOnnxASR、OpenAI 兼容的本地 LLM、IndexTTS、TurnSense、Qwen3ForcedAligner 和 FastEnhancer。先分别启动对应的本地服务，再把各自的 `base_url` 改成实际监听地址。各模型的安装方式和依赖见[支持的模型](../reference/supported_models.zh.md)。

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

把这个文件作为 `--config` 启动服务，命令与快速开始相同。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[支持的模型 →](../reference/supported_models.zh.md)** · 参考

-   **[配置服务 →](../tutorial/config_the_service.zh.md)** · 教程

-   **[挂载到 FastAPI →](mount_in_fastapi.zh.md)** · 操作指南

</div>
