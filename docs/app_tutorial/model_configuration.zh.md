# 模型配置

XTalk 桌面应用使用与[配置服务](../tutorial/config_the_service.zh.md)相同的模型配置格式。顶层键用于选择模型类型，`type` 用于选择具体实现，`params` 包含该实现的初始化参数。

## 配置文件

最简云端模型配置如下：

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

配置必须与所选模型类的初始化参数匹配。可用实现和可选依赖请参阅[支持的模型](../technical_reference/supported_models.zh.md)。

首次启动时，请根据提示选择 JSON 文件。如需在之后替换配置，请打开**设置与诊断**，选择其他文件并应用更改。XTalk 将重启本地服务并加载新配置。

选中的文件必须包含一个 JSON 对象，大小不超过 1 MiB，在后续启动时仍保留在所选路径，并且仅使用当前 XTalk 构建已支持的服务提供商。

## 受管模型

大多数 XTalk 模型实现会连接由用户单独启动的服务。如果 URL 使用 `managed://` 方案，XTalk 桌面应用则可以管理指定的本地模型：

```json
{
  "asr": {
    "type": "SherpaOnnxASR",
    "params": {
      "base_url": "managed://sensevoice-small",
      "asr_mode": "offline"
    }
  },
  "tts": {
    "type": "MossTTSNano",
    "params": {
      "base_url": "managed://moss-tts-nano",
      "voices": [
        {
          "name": "default",
          "path": "managed://moss-tts-nano/voices/zh_1.wav"
        }
      ]
    }
  }
}
```

对于受管 URL，桌面应用会：

1. 在固定版本的清单中解析对应服务；
2. 仅下载所需模型文件；
3. 校验文件大小和 SHA-256 哈希；
4. 将模型保存到应用数据目录；
5. 启动打包在应用中的本地运行时；
6. 在启动 XTalk 前，将受管 URL 替换为运行时的本地回环地址。

所选 JSON 文件不会被修改。后续启动会在使用前重新校验已安装的模型快照。如果应用新配置失败，XTalk 会停止新启动的运行时并恢复之前的配置。

## 选择推理后端

在 URL 后追加 `?backend=<name>` 可强制使用支持的推理后端：

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

如果省略 `?backend=`，URL 会使用自动选择。可接受值和默认规则取决于受管服务：

| 受管服务 | 可接受值 | 自动默认值 |
| --- | --- | --- |
| SenseVoice、AgenticASR Refiner、MOSS-TTS-Nano | `cpu`、`cuda`、`mlx` | 已打包且可用时优先 CUDA；否则在受支持的 Apple Silicon 构建上使用 MLX；其他情况使用 CPU |
| Matcha TTS | `cpu`、`cuda` | 已打包且可用时使用 CUDA；否则使用 CPU |
| Qwen3-ASR 0.6B INT8 | `cpu`、`cuda`、`coreml` | CUDA 可用时优先 CUDA；否则在 Core ML 可用时使用 Core ML；其他情况使用 CPU |
| CAM++ | `cpu`、`cuda`、`coreml` | CUDA 可用时优先 CUDA；否则在 Core ML 可用时使用 Core ML；其他情况使用 CPU |
| MOSS Turn Detector (MTD) | `cpu`、`metal` | Metal 可用时使用 Metal；否则使用 CPU |

如果显式指定的后端不可用或不受支持，XTalk 会报错，不会静默改用其他后端。自动 Qwen3-ASR 启动有一项例外：如果自动选中的加速后端启动失败，XTalk 会改用 CPU 重试。

CUDA 和 CPU 使用受管 ONNX 模型快照，MLX 使用单独固定版本的 safetensor 快照。Matcha 不支持 MLX，Qwen3-ASR 也不接受 `backend=mlx`。

模型服务提供商凭据仍属于模型配置的一部分。工具服务凭据需在**设置与诊断**中单独管理；不要将 `SERPER_API_KEY` 等工具凭据写入模型 JSON。
