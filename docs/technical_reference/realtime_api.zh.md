# Realtime API 协议参考

Xtalk 提供 OpenAI Realtime 风格的 API，包含标准子集和 Xtalk 扩展。客户端需要适配播放确认和 ASR 替换语义。

## 传输约定

连接 `WS /v1/realtime?model=xtalk`；模型别名可由服务端配置。控制事件和所有服务端输出使用 JSON，包含 `type`；服务端事件包含 `event_id`，客户端可选提供 `event_id`。表中 C→S 表示客户端到服务端，S→C 表示服务端到客户端。一次连接对应一个临时 Session（会话）及 Conversation（对话上下文）；Item 是其中的一条消息，Response 是一次回复生成。“内容标识”指 `response_id`、`item_id`、`output_index`、`content_index`，后两个索引从零开始。

普通输入/回复音频为 **24 kHz、单声道、signed PCM16 little-endian**，不含 WAV 文件头。输入既支持 JSON Base64，也支持裸 PCM 二进制，默认同时开启；输出仍为 Base64 JSON。`session.audio.mix.delta` 的格式单独以事件元数据为准。

单条输入消息最多 1 MiB（JSON 按 UTF-8 字节计，二进制按 PCM 字节计）；两种输入共享最多 60 秒的音频缓冲，提交或完成音频轮次后会释放对应输入预算，可继续发送音频。奇数字节、非法 Base64 和超限输入被拒绝，不改变有效缓冲；建议每包发送 20–100 ms 音频。

## 会话配置

`session.update.session` 支持 `type`（仅 `realtime`）、`instructions`、`output_modalities`（`["audio"]` 或 `["text"]`），以及 `audio.input.format/transcription/turn_detection` 和 `audio.output.format/voice/speed/emotion`。两侧 format 固定为 `{"type":"audio/pcm","rate":24000}`；transcription 默认是已配置模型别名的 `{"model":"xtalk"}`，设为 `null` 仅关闭对外转写事件，内部 ASR 仍运行。

`turn_detection` 为 `null` 或 `server_vad` 对象，后者包含 `type`、`threshold`、`prefix_padding_ms`、`silence_duration_ms`、`create_response`、`interrupt_response`。默认配置为 `{"type":"server_vad","threshold":0.5,"prefix_padding_ms":300,"silence_duration_ms":500,"create_response":true,"interrupt_response":true}`；模型必须支持所请求的 VAD 阈值设置。手动 commit 后仍沿用 Xtalk 自动轮次回复。

`voice` 默认 `default`，`speed` 默认 `1.0`，范围 `0.25–1.5`；活动回复期间更新速度是 Xtalk 扩展，已处理音频不重复变速。`emotion` 是 Xtalk 扩展，形如 `{"name":"happy","vector":[]}`，名称非空时优先于向量；更新时二者不能同时为空。修改 instructions、voice、emotion 需要相应模型支持，失败返回错误。配置项省略时保持原值。

## 事件表

| 阶段 | 事件或消息 | 方向 | 关键字段（不含公共信封） | 行为及客户端处理 |
| --- | --- | --- | --- | --- |
| 会话 | `session.queue.updated` / `session.queue.granted` | S→C | updated：`position`、`limit`、`active`；granted：— | Xtalk 扩展：准入队列状态和获准通知，可早于 session.created；获准不等于会话已就绪。 |
| 会话 | `session.created` | S→C | `session` | 会话就绪后主动发送完整有效配置，无需 attach；每条连接为独立临时会话。 |
| 会话 | `session.update` | C→S | `session` | 部分更新会话；缺省字段不变，不支持的字段或模型能力返回错误。允许字段见上方“会话配置”。 |
| 会话 | `session.updated` | S→C | `session` | 更新成功后返回完整有效配置；配置回显不应当作一次性控制命令。 |
| 音频输入 | `input_audio_buffer.append` | C→S | `audio` | Base64 编码 PCM 音频；按序追加，无逐包确认，不隐式提交。 |
| 音频输入 | 裸 PCM 二进制消息 | C→S | WebSocket 二进制 | Xtalk 扩展，无需开关；等价于 append，可与 Base64 混用，共享重采样、缓冲和轮次处理。 |
| 音频输入 | `input_audio_buffer.speech_start` | C→S | — | Xtalk 扩展：客户端讲话开始通知，不提交；手动模式启动 ASR 并重放已缓冲音频。 |
| 音频输入 | `input_audio_buffer.commit` | C→S | — | 提交当前音频，空缓冲报错；沿用自动轮次回复机制，不需要 response.create。 |
| 音频输入 | `input_audio_buffer.clear` | C→S | — | 清空未提交输入并重置重采样及手动讲话状态。 |
| 音频输入 | `input_audio_buffer.committed` | S→C | `item_id` | 确认音频轮次提交，关联输入 Item；不等于转写或回复已完成。 |
| 音频输入 | `input_audio_buffer.cleared` | S→C | — | 确认缓冲清空。 |
| 音频输入 | `input_audio_buffer.speech_started` / `input_audio_buffer.speech_stopped` | S→C | `origin` | 讲话边界通知，保留 Xtalk 来源语义；收到后仅更新状态，不回发 speech_start 或 commit。 |
| 输入消息与转写 | `conversation.item.create` | C→S | `item`、`xtalk.generate_response` | 接受 message，role 为 user/system/assistant，content 为 text 或 input_text；id 可省略但不能重复。默认仅注入上下文；user 携带 xtalk.generate_response=true 才启动实时文本轮次，文本须非空白且最多 8192 UTF-8 字节。 |
| 输入消息与转写 | `conversation.item.added` / `conversation.item.done` | S→C | `item` | 输入 Item 的添加及完成通知；音频输入的最终 Item 可包含 transcript。 |
| 输入消息与转写 | `conversation.item.input_audio_transcription.delta` | S→C | `item_id`、`content_index`、`delta`、`xtalk.mode`、`xtalk.revision` | ASR 预览扩展：append 追加，replace 用完整 delta 替换；revision 单调递增，忽略过期修订及 completed 后的预览。 |
| 输入消息与转写 | `conversation.item.input_audio_transcription.completed` | S→C | `item_id`、`content_index`、`transcript` | 最终完整转写；与回复推进异步，不能假定先于 response 输出。 |
| 输入消息与转写 | `conversation.item.input_audio_transcription.failed` | S→C | `item_id`、`content_index`、`error` | 输入转写失败，保留连接。 |
| 回复生成 | `response.created` | S→C | `response` | 生成开始，response 含 id/status/output。 |
| 回复生成 | `response.output_item.added` / `response.output_item.done` | S→C | `response_id`、`output_index`、`item` | 输出 Item 的创建及完成通知。 |
| 回复生成 | `response.content_part.added` / `response.content_part.done` | S→C | `response_id`、`item_id`、`output_index`、`content_index`、`part` | 输出内容部分的创建及完成通知。 |
| 回复生成 | `response.output_text.delta` / `response.output_text.done` | S→C | 内容标识；`delta` / `text` | 仅文本输出模式：delta 为新增文本，done 的 text 为完整生成文本。 |
| 回复生成 | `response.output_audio.delta` / `response.output_audio.done` | S→C | 内容标识；delta 事件含 `delta` | 音频模式：delta 为 Base64 PCM 音频；done 表示音频生成结束，不表示播放结束。 |
| 回复生成 | `response.output_audio_transcript.delta` / `response.output_audio_transcript.done` | S→C | 内容标识；`delta` / `transcript` | 实际合成音频的文本，不能用尚未合成的 LLM 文本替代。分句成功后追加；流式整轮成功后发送一次完整 delta，再发送完整 transcript。 |
| 回复生成 | `response.done` | S→C | `response.id`、`response.status`、`response.output` | 生成终态，status 为 completed/cancelled/failed/incomplete；不等待播放确认，不代表用户已听完。 |
| 回复生成 | `response.cancel` | C→S | `response_id`（可选） | 取消指定或当前活动生成；没有可取消回复时返回错误。客户端播放队列由播放控制和回执另行处理。 |
| 播放控制与回执 | `response.output_audio.playback.chunk_played` | C→S | `response_id` | Xtalk 扩展：每个原始音频事件块实际播放完成后确认一次，释放输出预算；不能按重新切分的播放块确认。 |
| 播放控制与回执 | `response.output_audio.playback.finished` | C→S | `response_id` | Xtalk 扩展：所有音频实际播放完成后发送，不替代逐块确认。 |
| 播放控制与回执 | `response.output_audio.playback.stopped` | C→S | `response_id`、`played_audio_ms` | Xtalk 扩展：停止后报告尚未确认音频中已播放的增量毫秒数，不是累计播放位置。 |
| 播放控制与回执 | `response.output_audio.playback.pause` / `response.output_audio.playback.resume` | S→C | `response_id` | Xtalk 扩展：暂停或恢复指定回复的播放队列。 |
| 播放控制与回执 | `response.output_audio.playback.stop` | S→C | `response_id` | Xtalk 扩展：立即停止并清空待播队列，再发送 playback.stopped 增量回执。 |
| 播放控制与回执 | `response.output_audio.playback.updated` / `response.output_audio.playback.done` | S→C | `response_id`、`text` | Xtalk 扩展：完整已播放文本快照及最终结算，用于展示和后续历史；不同于生成文本 delta/done。 |
| 辅助通知 | `conversation.caption.updated` | S→C | `text`、`is_final`、`reason` | Xtalk 扩展：完整字幕快照，替换展示，不当作 ASR 增量。 |
| 辅助通知 | `conversation.retrieval.updated` | S→C | `text`、`is_final` | Xtalk 扩展：完整检索结果快照，替换展示；知识入库由宿主负责。 |
| 辅助通知 | `conversation.speaker.updated` | S→C | `speaker_id`、`reason` | Xtalk 扩展：说话人识别通知。 |
| 辅助通知 | `response.tool.called` | S→C | `name`、`args` | Xtalk 扩展：后端工具调用通知；工具仍由服务端执行，不要求客户端提交 function_call_output。 |
| 辅助通知 | `session.latency.updated` | S→C | `network_latency_ms`、`asr_latency_ms`、`llm_first_token_ms`、`llm_sentence_ms`、`tts_first_chunk_ms` | Xtalk 扩展：延迟指标，单位毫秒；不提供时钟同步，network_latency_ms 不代表实测 RTT。 |
| 辅助通知 | `session.audio.mix.delta` | S→C | `delta`、`sample_rate`、`channels`、`format` | Xtalk 扩展：启用 send_full_audio_to_client 时输出完整对话混合音频块；delta 为 Base64，格式以事件元数据为准，不等同单次回复的音频输出；文件保存由宿主负责。 |
| 错误 | `error` | S→C | `error.type`、`error.code`、`error.message`；可选 `error.param`、`error.event_id` | 协议或运行错误；客户端请求错误可关联 event_id，裸二进制错误的 error.event_id 为 null。无效输入通常保持连接；持续输出队列超限可断开。 |

## 生成与播放时序

同一输出内容依次经历 `response.created` → `response.output_item.added` → `response.content_part.added` → 对应 delta/done → `response.content_part.done` → `response.output_item.done` → `response.done`。生成进度与播放进度分开：每个原始音频块实际播放后发送 chunk_played，全部播放后发送 finished；收到 stop 时清空队列，再以 stopped 结算未确认的已播放时长。缺少逐块确认可能阻塞后续音频投递。

## 支持范围

未识别的输入事件返回 `unsupported_event`。内置处理器不支持 `response.create`、`conversation.item.truncate`、`change_tts_model`、`change_llm_model`、`ping`、`pong` 和 `clock_sync`。回复由音频轮次或带 `xtalk.generate_response=true` 的文本输入触发；模型由服务端配置，打断后的播放结算使用 `response.output_audio.playback.stopped`。

## 发送示例

以下片段使用已连接且收到 `session.created` 的 `websocket`。`pcm_bytes` 为符合上述格式的原始音频。每个音频块选择一种编码发送一次：

```python
import base64
import json

# 方式一：JSON Base64。
await websocket.send(json.dumps({
    "type": "input_audio_buffer.append",
    "audio": base64.b64encode(pcm_bytes).decode("ascii"),
}))
```

或者直接发送二进制：

```python
# 方式二：裸 PCM；替代上面的 Base64 发送。
await websocket.send(pcm_bytes)
```

默认由服务端 VAD 判断轮次。需要手动控制时，先关闭自动检测，等待 `session.updated`，再发送音频；本轮发送完毕后提交：

```python
await websocket.send(json.dumps({
    "type": "session.update",
    "session": {"audio": {"input": {"turn_detection": None}}},
}))
# 接收循环确认 session.updated 后，发送本轮全部音频块。
await websocket.send(pcm_bytes)
await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
```

实时文本输入单独使用 `conversation.item.create`：

```python
await websocket.send(json.dumps({
    "type": "conversation.item.create",
    "item": {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "你好"}],
    },
    "xtalk": {"generate_response": True},
}))
```

省略 `xtalk.generate_response` 时，该消息仅加入上下文，可用于按顺序注入历史。
