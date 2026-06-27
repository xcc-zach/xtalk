# 🚀 快速开始

我们将使用阿里云的 API 来演示 **X-Talk** 的基本功能。

## 第 1 步：安装依赖

安装阿里云和服务器脚本的依赖：
```bash
pip install "xtalk[ali,example] @ git+https://github.com/xcc-zach/xtalk.git@main"
```

> 对于开发者，请克隆仓库、创建分支，并在仓库目录下使用 `pip install -e .`。

## 第 2 步：获取 API 密钥

从[阿里云百炼平台](https://bailian.console.aliyun.com/?tab=model#/api-key)获取 API 密钥。我们将使用阿里云的（目前）免费服务。

> 在线服务可能不稳定且延迟较高。我们建议使用本地部署的模型以获得更好的用户体验。我们推荐使用 *SherpaOnnx* 的 ASR 模型（[配置教程](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)）和 *IndexTTS*（[配置教程](https://github.com/Ksuriuri/index-tts-vllm)）。详情请参阅[服务器配置教程](tutorial/config_the_service.zh.md)和[本地完整部署示例配置](tutorial/sample_config_for_fully_local_deployment.zh.md)。

## 第 3 步：创建配置文件

创建一个 JSON 配置文件来指定要使用的模型，并**用您获取的密钥填写 <API_KEY>**：

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

## 第 4 步：启动服务器

示例服务器脚本已准备好在 [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)。我们只需使用配置文件（**用刚才创建的配置文件路径填写 <PATH_TO_CONFIG>.json**）和自定义端口启动服务器：
```bash
git clone https://github.com/xcc-zach/xtalk.git
cd xtalk
python examples/sample_app/configurable_server.py  --port 7635 --config <PATH_TO_CONFIG>.json
```

最后，我们的演示已准备好在 `http://localhost:7635`。在浏览器中查看吧！
