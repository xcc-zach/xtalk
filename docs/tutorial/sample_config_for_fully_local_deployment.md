Below is an example config file for X-Talk when you want to have all models hosted locally. `SherpaOnnxASR` is used for speech recognition, and you can see [here](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example) to set up the server. For LLM agent and embeddings, any model adhering to OpenAI protocol is fine. You should provide `api_key`, `base_url` and `model`. `IndexTTS` is used for speech generation, and see [here](https://github.com/Ksuriuri/index-tts-vllm) for server setup. Reference voices can be downloaded [here](https://github.com/xcc-zach/xtalk/releases/tag/audio-v0.1). The captioner is hard to set up, but you can refer to the tutorial [here](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner). **Finally, remember to look into each model type in [Supported Models](../docs/supported_models.md) for how to install the optional dependencies of X-Talk for that model**.

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
