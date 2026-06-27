# 🚀 Quickstart

We will use APIs from AliCloud to demonstrate the basic capability of **X-Talk**.

## Step 1. Install dependencies

Install dependencies for AliCloud and server script:
```bash
pip install "xtalk[ali,example] @ git+https://github.com/xcc-zach/xtalk.git@main"
```

> For developers, clone the repository, create a branch and use `pip install -e .` under the repository.

## Step 2. Obtain an API key

Obtain an API key from [AliCloud Bailian Platform](https://bailian.console.aliyun.com/?tab=model#/api-key). We will be using free-tier (currently) service from AliCloud.

> Online service may be unstable and of high latency. We recommend using locally deployed models for better user experience. We recommend ASR models from *SherpaOnnx* ([setup tutorial](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)) and *IndexTTS* ([setup tutorial](https://github.com/Ksuriuri/index-tts-vllm)).See [server config tutorial](tutorial/config_the_service.md) and [sample config for fully local deployment](tutorial/sample_config_for_fully_local_deployment.md) for details. 

## Step 3. Create the config file

Create a JSON config specifying the models to use, and **fill in <API_KEY>** with the key you obtained:

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
## Step 4. Start the server

A sample server script is ready at [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py). We simply need to start the server with the config file (**fill in <PATH_TO_CONFIG>.json** with the path to the config file we just created) and a custom port:
```bash
git clone https://github.com/xcc-zach/xtalk.git
cd xtalk
python examples/sample_app/configurable_server.py  --port 7635 --config <PATH_TO_CONFIG>.json
```

Finally, our demo is ready at `http://localhost:7635`. View it in the browser!
