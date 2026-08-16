# 服务配置项

`service_config` 是顶层配置中的一个可选对象，会传给 `DefaultService`，并在每个会话内共享给各个 manager 和 gateway。

示例：

```json
{
  "service_config": {
    "enable_persistence": true,
    "recording": true,
    "send_full_audio_to_client": false,
    "data_dir": "data",
    "forced_alignment": {
      "language": "zh",
      "stop_ack_timeout_ms": 500
    },
    "multi_speaker": {
      "response_policy": "focus_only",
      "focus_speaker_ids": ["S01"]
    }
  }
}
```

## 配置项列表

| 键名 | 类型 | 默认值 | 使用位置 | 作用 |
| --- | --- | --- | --- | --- |
| `enable_persistence` | `bool` | `true` | `Xtalk`、`ServiceManager`、`PersistenceManager` | 控制是否启用会话历史持久化，并将数据写入 `<data_dir>/chat_history.sqlite3`。关闭后，内置登录和 WebSocket `attach_session` 握手仍可使用，但聊天记录只在当前活动连接的内存中保留，不支持历史会话恢复。 |
| `data_dir` | `str` | `"data"` | `Service`、`EmbeddingsManager` | 会话级 embedding 数据的根目录。向量数据会持久化到 `<data_dir>/sessions/<session_id>/embeddings`，并在会话结束时删除对应会话目录。 |
| `recording` | `bool` | `false` | `RecordingManager` | 开启会话录音并输出为双声道 WAV 文件。左声道是原始用户音频，右声道是实际播放的 TTS 音频。默认输出路径为 `logs/session_audio/<timestamp>.wav`。 |
| `send_full_audio_to_client` | `bool` | `false` | `RecordingManager`、`OutputGateway`、前端 | 将拼装好的完整对话双声道 PCM 音频块通过 `full_audio_frame` 消息发给前端。数据格式为 48 kHz、16-bit、双声道 PCM，并以 base64 编码传输。 |
| `forced_alignment` | `object` | `{}` | `TTSPlaybackManager` | 配置 TTS 播放过程中的强制对齐。仅在已配置 `forced_aligner` 模型时生效。 |
| `multi_speaker` | `object` | `{}` | `MultiSpeakerTurnContextManager` | 配置说话人分离结果与 ASR 的合并、目标说话人响应策略和历史过滤。仅在已配置 `speaker_diarization` 模型时生效。 |

### `forced_alignment`

| 键名 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `language` | `str \| null` | `null` | 传给强制对齐模型的语言。未指定时由模型自行处理。 |
| `stop_ack_timeout_ms` | `float` | `500` | 停止播放后等待客户端确认的最长时间，单位为毫秒；负数会按 `0` 处理。 |

### `multi_speaker`

| 键名 | 类型 | 默认值 | 作用 |
| --- | --- | --- | --- |
| `response_policy` | `str` | `"focus_only"` | `focus_only` 表示仅响应目标说话人；其他值不启用该限制。 |
| `focus_speaker_ids` | `list[str]` | `["S01"]` | 允许触发响应的目标说话人 ID。 |
| `exclude_non_focus_from_history` | `bool` | `true` | 是否从 Agent 对话历史中排除非目标说话人的内容。 |
| `suppress_when_speaker_missing` | `bool` | `false` | 说话人结果缺失时是否禁止响应。 |
| `join_timeout_s` | `float` | `5.0` | 等待 ASR 与说话人分离结果合并的最长时间，单位为秒。 |
| `fallback_on_timeout` | `bool` | `true` | 合并超时后是否使用已有结果继续处理。 |
| `diarization.pre_buffer_s` | `float` | `1.0` | 为每个语音片段保留的前置音频长度，单位为秒。 |
| `diarization.partial.interval_s` | `float` | `1.0` | 请求 partial 说话人分离结果的间隔，单位为秒。 |
| `diarization.partial.first_partial_min_s` | `float` | `0.8` | 首次请求 partial 结果所需的最短音频长度，单位为秒。 |
