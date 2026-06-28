以下是当您希望所有模型都在本地托管时 X-Talk 的示例配置文件。`SherpaOnnxASR` 用于语音识别，您可以查看[这里](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)来设置服务器。对于 LLM 代理和嵌入模型，任何遵循 OpenAI 协议的模型都可以。您需要提供 `api_key`、`base_url` 和 `model`。`IndexTTS` 用于语音生成，请参阅[这里](https://github.com/Ksuriuri/index-tts-vllm)进行服务器设置。参考语音可以在[这里](https://github.com/xcc-zach/xtalk/releases/tag/audio-v0.1)下载。字幕生成器的设置比较复杂，但您可以参考[这里](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)的教程。**最后，请记得查看[支持的模型](../docs/supported_models.zh.md)中的每个模型类型，了解如何为该模型安装 X-Talk 的可选依赖项**。

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
            },
            "voice_names": [
                "Man",
                "Woman",
                "Child"
            ],
            "emotions": [
                "happy",
                "angry",
                "sad",
                "fear",
                "disgust",
                "depressed",
                "surprised",
                "calm",
                "normal"
            ]
        }
    },
    "embeddings": {
        "type": "OpenAIEmbeddings",
        "params": {
            "api_key": "none",
            "base_url": "http://127.0.0.1:8002/v1",
            "model": "Qwen/Qwen3-Embedding-0.6B"
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
    "speaker_encoder": "PyannoteSpeakerEncoder",
    "captioner": {
        "type": "Qwen3OmniCaptioner",
        "params": {
            "base_url": "http://localhost:8901/v1",
            "api_key": "none"
        }
    },
    "caption_rewriter": {
        "type": "DefaultCaptionRewriter",
        "params": {
            "model": {
                "api_key": "none",
                "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
                "base_url": "http://127.0.0.1:8000/v1"
            }
        }
    },
    "thought_rewriter": {
        "type": "DefaultThoughtRewriter",
        "params": {
            "model": {
                "api_key": "none",
                "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
                "base_url": "http://127.0.0.1:8000/v1"
            }
        }
    },
    "speech_speed_controller": "RubberbandSpeedController",
    "turn_detector": {
        "type": "LLMTurnDetector",
        "params": {
            "model": {
                "api_key": "none",
                "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
                "base_url": "http://127.0.0.1:8000/v1"
            }
        }
    },
}
```
