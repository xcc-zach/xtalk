# Model Configuration

XTalk Desktop uses the same model configuration format described in [Configure the service](../tutorial/config_the_service.md). The top-level keys select model types, `type` selects an implementation, and `params` contains that implementation's initialization arguments.

## Configuration file

A minimal cloud-model configuration looks like this:

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

The configuration must match the selected model classes' initialization arguments. See [Supported Models](../technical_reference/supported_models.md) for available implementations and optional dependencies.

On first launch, select the JSON file when prompted. To replace it later, open **Settings and diagnostics**, choose another file, and apply the change. XTalk restarts its local service and loads the new configuration.

The selected file must contain a JSON object, be no larger than 1 MiB, remain available at the selected path, and use providers supported by the installed XTalk build.

## Managed models

Most XTalk model implementations connect to a service that you start separately. XTalk Desktop can instead manage selected local models when their URL uses the `managed://` scheme:

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

For a managed URL, the desktop application:

1. resolves the service in its pinned manifest;
2. downloads only the required model files;
3. verifies their sizes and SHA-256 hashes;
4. stores them under the application data directory;
5. starts the packaged local runtime; and
6. replaces the managed URL with the runtime's loopback address before starting XTalk.

The selected JSON file is not modified. Later launches revalidate the installed snapshot before use. If applying a new configuration fails, XTalk stops newly started runtimes and restores the previous configuration.

## Selecting a backend

Append `?backend=<name>` to force a supported inference backend:

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

If `?backend=` is omitted, the URL uses automatic selection. Defaults and accepted values depend on the managed service:

| Managed service | Accepted values | Automatic default |
| --- | --- | --- |
| SenseVoice, AgenticASR Refiner, MOSS-TTS-Nano | `cpu`, `cuda`, `mlx` | CUDA when packaged and available; otherwise MLX on supported Apple Silicon builds; otherwise CPU |
| Matcha TTS | `cpu`, `cuda` | CUDA when packaged and available; otherwise CPU |
| Qwen3-ASR 0.6B INT8 | `cpu`, `cuda`, `coreml` | CUDA when available; otherwise Core ML when available; otherwise CPU |
| CAM++ | `cpu`, `cuda`, `coreml` | CUDA when available; otherwise Core ML when available; otherwise CPU |
| MOSS Turn Detector (MTD) | `cpu`, `metal` | Metal when available; otherwise CPU |

An explicitly requested unavailable or unsupported backend is an error; XTalk does not silently select another backend. One exception applies to automatic Qwen3-ASR startup: if the automatically selected accelerated backend fails to start, XTalk retries with CPU.

CUDA and CPU use the managed ONNX snapshots. MLX uses separately pinned safetensor snapshots. Matcha does not support MLX, and Qwen3-ASR does not accept `backend=mlx`.

Model-provider credentials remain part of the model configuration. Tool-service credentials are managed separately in **Settings and diagnostics**; do not add tool credentials such as `SERPER_API_KEY` to the model JSON.
