# 🚀 Quickstart

We will use APIs from AliCloud to demonstrate the basic capability of **X-Talk**.

First, install dependencies for AliCloud and server script:
```bash
pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"
pip install jinja2 python-multipart 'uvicorn[standard]'
```

Then, obtain an API key from [AliCloud Bailian Platform](https://bailian.console.aliyun.com/?tab=model#/api-key). We will be using free-tier service from AliCloud.

> Online service may be unstable and of high latency. We recommend using locally deployed models for better user experience. See [server config tutorial](tutorial/config_the_server.md) and [supported models](docs/supported_models.md) for details.

After that, create a JSON config specifying the models to use, and **fill in <API_KEY>** with the key you obtained:

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

> If you find *Qwen3ASRFlashRealtime* not working properly, you can use `"asr": "SenseVoiceSmallLocal",` instead which is a ~1GB local model. Also, you can try to use local speech generation model *IndexTTS* ([setup tutorial](https://github.com/Ksuriuri/index-tts-vllm)):
> ```json
> "tts": {
>     "type": "IndexTTS",
>     "params": {
>         "port": 6006
>     }
> },
> ```
> If you want all models deployed locally, see [here](tutorial/sample_config_for_fully_local_deployment.md).

The next step is to compose the startup script. Since we also need to link frontend webpage and scripts to get the demo working, the startup script is ready at `examples/sample_app/configurable_server.py`. We simply need to start the server with the config file (**fill in <PATH_TO_CONFIG>.json** with the path to the config file we just created) and a custom port:
```bash
git clone https://github.com/xcc-zach/xtalk.git
cd xtalk
python examples/sample_app/configurable_server.py  --port 7635 --config <PATH_TO_CONFIG>.json
```

Finally, our demo is ready at `http://localhost:7635`. View it in the browser!
