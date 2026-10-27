# 模型配置

本教程为 X-Talk 桌面应用编写一份模型配置文件并应用它，最终得到一个用你选定的模型进行语音对话的本地服务。本页没有前置页面，只需要安装好桌面应用。

## 1. 编写配置文件

桌面应用从一个 JSON 文件读取模型，格式与服务端相同，见[配置服务](../config_the_service.zh.md)。把下面的最简云端配置保存为 `config.json`，并填入 `<API_KEY>`：

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

每个 `params` 对象必须与所选模型类的初始化参数匹配。可用的类及其可选依赖见[支持的模型](../../reference/supported_models.zh.md)。

文件必须包含一个 JSON 对象，大小不超过 1 MiB，并且仅使用当前 X-Talk 构建已支持的服务提供商。

## 2. 应用配置

首次启动时，应用会提示选择配置文件。选择 `config.json`，应用会应用该配置并启动本地服务。

之后要换用另一个文件：

1. 打开右上角的**设置与诊断**；
2. 选择新的模型配置 JSON 文件；
3. 应用更改。

应用会重启本地服务并加载新文件。文件必须保留在所选路径，应用每次启动都会重新读取它。配置无法应用时，应用会恢复之前的配置，因此 `config.json` 里的笔误只会让上一次可用的配置继续生效。

## 3. 进行一次对话

在主窗口开始语音对话并说一句话。你说的话会作为用户消息出现在聊天视图中，随后播放由所配置 TTS 模型合成的回复。此后应用每次启动都会使用这份配置。

## 4. 换用受管模型

`managed://` URL 让应用把模型下载到本地并启用，从而摆脱对云端 ASR 服务商的依赖。把 `asr` 段替换为：

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

按第 2 步再次应用文件。应用会先下载模型再启用它，因此这次应用比上一次耗时更长。

在 URL 后追加 `?backend=<name>` 可强制使用支持的推理后端：

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

省略 `?backend=` 时由应用自动选择。可接受的值和默认规则见[支持的模型](../../reference/supported_models.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[添加工具 →](add_tools.zh.md)** · 教程

-   **[支持的模型 →](../../reference/supported_models.zh.md)** · 参考

-   **[配置服务 →](../config_the_service.zh.md)** · 教程

</div>
