# Model Configuration

In this tutorial you write a model configuration file for the X-Talk desktop app, apply it, and finish with a running local service that answers a voice conversation using the models you chose. No earlier page is required; the app itself is the only prerequisite.

## 1. Write the configuration file

The desktop app reads its models from a JSON file in the same format as the server, described in [Configure the Service](../config_the_service.md). Save the following minimal cloud configuration as `config.json` and fill in `<API_KEY>`:

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

Each `params` object must match the initialization arguments of the selected model class. [Supported Models](../../reference/supported_models.md) lists the available classes and their optional dependencies.

The file must contain a JSON object, be no larger than 1 MiB, and use providers supported by the installed X-Talk build.

## 2. Apply the configuration

On first launch, the app prompts for a configuration file. Select `config.json`; the app applies it and starts the local service.

To apply a different file later:

1. Open **Settings and diagnostics** in the upper-right corner.
2. Select the new model configuration JSON file.
3. Apply the change.

The app restarts its local service and loads the new file. Keep the file at the selected path; the app reads it again on every later launch. When a configuration cannot be applied, the app restores the previous one, so a typo in `config.json` leaves the last working setup in place.

## 3. Hold a conversation

Start a voice conversation from the main window and say a sentence. Your sentence appears as a user message in the chat view, followed by a spoken reply from the configured TTS model. Your configuration is now the one the app uses on every launch.

## 4. Switch to a managed model

A `managed://` URL asks the app to download a model locally and enable it, which removes the dependency on a cloud ASR provider. Replace the `asr` block with:

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

Apply the file again as in step 2. The app downloads the model before enabling it, so this apply takes longer than the previous one.

Append `?backend=<name>` to force a supported inference backend:

```text
managed://sensevoice-small?backend=cpu
managed://moss-tts-nano?backend=mlx
managed://qwen3-asr-0.6b-int8?backend=coreml
managed://moss-transcribe-diarize?backend=metal
```

Without `?backend=`, the app selects a backend automatically. Accepted values and default rules are listed in [Supported Models](../../reference/supported_models.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Add Tools →](add_tools.md)** · Tutorial

-   **[Supported Models →](../../reference/supported_models.md)** · Reference

-   **[Configure the Service →](../config_the_service.md)** · Tutorial

</div>
