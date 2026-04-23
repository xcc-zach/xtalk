# 🚀 快速开始

我们将使用阿里云的 API 来演示 **X-Talk** 的基本功能。

首先，安装阿里云和服务器脚本的依赖：
```bash
pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"
pip install jinja2 python-multipart 'uvicorn[standard]'
```

然后，从[阿里云百炼平台](https://bailian.console.aliyun.com/?tab=model#/api-key)获取 API 密钥。我们将使用阿里云的免费服务。

> 在线服务可能不稳定且延迟较高。我们建议使用本地部署的模型以获得更好的用户体验。详情请参阅[服务器配置教程](tutorial/config_the_server.zh.md)和[支持的模型](docs/supported_models.zh.md)。

之后，创建一个 JSON 配置文件来指定要使用的模型，并**用您获取的密钥填写 <API_KEY>**：

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

> 如果您发现 *Qwen3ASRFlashRealtime* 无法正常工作，可以改用 `"asr": "SenseVoiceSmallLocal",`，这是一个约 1GB 的本地模型。此外，您还可以尝试使用本地语音生成模型 *IndexTTS*（[配置教程](https://github.com/Ksuriuri/index-tts-vllm)）：
> ```json
> "tts": {
>     "type": "IndexTTS",
>     "params": {
>         "port": 6006
>     }
> },
> ```
> 如果您想要所有模型都在本地部署，请参阅[这里](tutorial/sample_config_for_fully_local_deployment.zh.md)。

下一步是编写启动脚本。由于我们还需要链接前端网页和脚本来使演示正常工作，启动脚本已准备好在 `examples/sample_app/configurable_server.py`。我们只需使用配置文件（**用刚才创建的配置文件路径填写 <PATH_TO_CONFIG>.json**）和自定义端口启动服务器：
```bash
git clone https://github.com/xcc-zach/xtalk.git
cd xtalk
python examples/sample_app/configurable_server.py  --port 7635 --config <PATH_TO_CONFIG>.json
```

最后，我们的演示已准备好在 `http://localhost:7635`。在浏览器中查看吧！
