# Service Configuration

`service_config` is the optional top-level config object passed into `DefaultService`.
It is shared with all session-scoped managers and gateways.

Example:

```json
{
  "service_config": {
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
| `enable_persistence` | `bool` | `true` | `Xtalk`, `ServiceManager`, `PersistenceManager` | Enables session history persistence in `<data_dir>/chat_history.sqlite3` together with session listing and restoration. When disabled, the built-in auth and websocket attach flow still work, but chat history is kept in memory only for the current live connection. |
| `data_dir` | `str` | `"data"` | `Service`, `EmbeddingsManager` | Root directory for session-scoped embedding data. Embeddings are persisted under `<data_dir>/sessions/<session_id>/embeddings` and the session directory is removed on shutdown. |
| `recording` | `bool` | `false` | `RecordingManager` | Enables session recording to a stereo WAV file. Left channel is raw user audio, right channel is played TTS audio. Default output path is `logs/session_audio/<timestamp>.wav`. |
| `send_full_audio_to_client` | `bool` | `false` | `RecordingManager`, `OutputGateway`, frontend | Sends assembled full-conversation stereo PCM chunks to the client as `full_audio_frame` messages. The payload is 48 kHz, 16-bit, 2-channel PCM encoded as base64. |
| `forced_alignment` | `object` | `{}` | `TTSPlaybackManager` | Configures forced alignment during TTS playback. It takes effect only when a `forced_aligner` model is configured. |
| `multi_speaker` | `object` | `{}` | `MultiSpeakerTurnContextManager` | Configures joining diarization with ASR, focus-speaker response policy, and history filtering. It takes effect only when a `speaker_diarization` model is configured. |

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
