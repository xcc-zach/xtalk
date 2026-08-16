# 模型配置

## 应用配置

模型配置保存在 JSON 文件中。首次启动 X-Talk 时，请根据提示选择配置文件，X-Talk 会应用该配置并启动本地服务。

如需在之后应用新配置：

1. 打开右上角的**设置与诊断**；
2. 选择新的模型配置 JSON 文件；
3. 应用更改。

应用更改时，X-Talk 会重启本地服务并加载新配置。配置文件必须保留在所选路径；X-Talk 后续启动时会继续读取该文件。若新配置应用失败，X-Talk 会恢复之前的配置。

X-Talk 桌面应用使用与[配置服务](../tutorial/config_the_service.zh.md)相同的模型配置格式。

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

选中的文件必须包含一个 JSON 对象，大小不超过 1 MiB，并且仅使用当前 X-Talk 构建已支持的服务提供商。

## 受管模型

在配置中使用 `managed://` URL 即可启用受管模型。X-Talk 会自动将所需模型下载到本地并启用。

例如：

```json
{
  "asr": {
    "type": "SherpaOnnxASR",
    "params": {
      "base_url": "managed://sensevoice-small",
      "asr_mode": "offline"
    }
  }
}
```

### 推理后端

在 URL 后追加 `?backend=<name>` 可强制使用支持的推理后端：

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

如果省略 `?backend=`，URL 会使用自动选择。可接受值和默认规则请参考[支持的模型](../technical_reference/supported_models.zh.md)。
