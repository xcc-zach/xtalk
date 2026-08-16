# Model Configuration

## Applying a configuration

Model configuration is stored in a JSON file. On first launch, select the configuration file when prompted. X-Talk applies it and starts the local service.

To apply a new configuration later:

1. Open **Settings and diagnostics** in the upper-right corner.
2. Select the new model configuration JSON file.
3. Apply the change.

When applying the change, X-Talk restarts its local service and loads the new configuration. Keep the file available at the selected path; X-Talk reads it again on later launches. If the new configuration cannot be applied, X-Talk restores the previous configuration.

X-Talk Desktop uses the same model configuration format described in [Configure the service](../tutorial/config_the_service.md).

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

The selected file must contain a JSON object, be no larger than 1 MiB, and use providers supported by the installed X-Talk build.

## Managed models

Use a `managed://` URL in the configuration to enable a managed model. X-Talk automatically downloads the required model locally and enables it.

For example:

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

### Inference backend

Append `?backend=<name>` to force a supported inference backend:

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

If `?backend=` is omitted, the URL uses automatic selection. See [Supported Models](../technical_reference/supported_models.md) for accepted values and default rules.
