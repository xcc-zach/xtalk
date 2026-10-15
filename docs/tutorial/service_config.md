# Service Configuration

`service_config` is the optional top-level config object passed into `DefaultService`.
It is shared with all session-scoped managers and gateways.

Example:

```json
{
  "service_config": {
    "realtime_api": {
      "model": "xtalk",
      "api_keys": []
    },
    "enable_persistence": true,
    "recording": true,
    "send_full_audio_to_client": false,
    "data_dir": "data",
    "forced_alignment": {
      "language": "en",
      "stop_ack_timeout_ms": 500
    },
    "multi_speaker": {
      "response_policy": "focus_only",
      "focus_speaker_ids": ["S01"]
    }
  }
}
```

## Reference

| Key | Type | Default | Used by | Effect |
| --- | --- | --- | --- | --- |
| `realtime_api` | `object` | `{}` | `Xtalk` Realtime ASGI app | Configures the public Realtime API model alias and optional API key authentication. See below for details. |
| `enable_persistence` | `bool` | `true` | `Xtalk`, `ServiceManager`, `PersistenceManager` | Enables session history persistence in `<data_dir>/chat_history.sqlite3` together with session listing and restoration. When disabled, the built-in auth and websocket attach flow still work, but chat history is kept in memory only for the current live connection. |
| `data_dir` | `str` | `"data"` | `Service`, `EmbeddingsManager` | Root directory for session-scoped embedding data. Embeddings are persisted under `<data_dir>/sessions/<session_id>/embeddings` and the session directory is removed on shutdown. |
| `recording` | `bool` | `false` | `RecordingManager` | Enables session recording to a stereo WAV file. Left channel is raw user audio, right channel is played TTS audio. Default output path is `logs/session_audio/<timestamp>.wav`. |
| `send_full_audio_to_client` | `bool` | `false` | `RecordingManager`, `OutputGateway`, frontend | Sends assembled full-conversation stereo PCM chunks to the client as `full_audio_frame` messages. The payload is 48 kHz, 16-bit, 2-channel PCM encoded as base64. |
| `forced_alignment` | `object` | `{}` | `TTSPlaybackManager` | Configures forced alignment during TTS playback. It takes effect only when a `forced_aligner` model is configured. |
| `multi_speaker` | `object` | `{}` | `MultiSpeakerTurnContextManager` | Configures joining diarization with ASR, focus-speaker response policy, and history filtering. It takes effect only when a `speaker_diarization` model is configured. |

### `realtime_api`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `model` | `str` | `"xtalk"` | Public Realtime API model alias used by `GET /v1/models` and `WS /v1/realtime?model=xtalk`. Must be a nonblank string. |
| `api_keys` | `list[str]` | `[]` | Allowed API keys; an empty list disables API key authentication. Each key must be a nonempty string. |

When `api_keys` is configured, clients use `Authorization: Bearer <key>`. Browsers can use the `openai-insecure-api-key.<key>` WebSocket subprotocol. API keys are managed separately from application login credentials.

`realtime_api` supports only `model` and `api_keys`. See [Realtime API](realtime_api.md) for application startup, mounting, and event exchange.

### `forced_alignment`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `language` | `str \| null` | `null` | Language passed to the forced-alignment model. When omitted, the model handles language selection. |
| `stop_ack_timeout_ms` | `float` | `500` | Maximum time to wait for client acknowledgement after stopping playback, in milliseconds. Negative values are treated as `0`. |

### `multi_speaker`

| Key | Type | Default | Effect |
| --- | --- | --- | --- |
| `response_policy` | `str` | `"focus_only"` | `focus_only` responds only to focus speakers; other values do not enable this restriction. |
| `focus_speaker_ids` | `list[str]` | `["S01"]` | Speaker IDs allowed to trigger a response. |
| `exclude_non_focus_from_history` | `bool` | `true` | Whether to exclude non-focus speech from Agent conversation history. |
| `suppress_when_speaker_missing` | `bool` | `false` | Whether to suppress responses when the speaker result is missing. |
| `join_timeout_s` | `float` | `5.0` | Maximum time to wait for ASR and diarization results to join, in seconds. |
| `fallback_on_timeout` | `bool` | `true` | Whether to continue with the available result after a join timeout. |
| `diarization.pre_buffer_s` | `float` | `1.0` | Audio retained before each speech segment, in seconds. |
| `diarization.partial.interval_s` | `float` | `1.0` | Interval between partial diarization requests, in seconds. |
| `diarization.partial.first_partial_min_s` | `float` | `0.8` | Minimum audio length required for the first partial request, in seconds. |
