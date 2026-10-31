# GPT-Live 单次通话规格（协议 2）

## 范围

每次新建语音通话，支持开始、连续收音与静音帧、双向 PCM、Live 原生打断、麦克风静音、增量转写、停止和可见错误。原级联处理图保留历史、工具、录音、说话人和响应对齐功能。Live 的工具、增强、说话人、检索、录音、逐字对齐、持久化、历史恢复和自动重连留待后续阶段。

## 框架边界

```text
应用鉴权 → /v1/realtime → Service/EventBus
采集 → InputGateway → EndToEndManager → GPTLive 发送者 → OpenAI
OpenAI → GPTLive 接收者 → 标准化事件
  ├─ 转写/错误 → OutputGateway → ActionHandler → Conversation/UI
  └─ 待播 PCM → 投递任务 → _AudioPlaybackBudget → OutputGateway
       → 共享 WebOutputAudioSession 排程 → 累计播放回执
```

- `EndToEnd`：`clone()`、不可变 `audio_profile`、不含密钥的元信息、`open()`、`append_audio()`、`events()`、`set_input_muted()`、`close()`。
- `GPTLive`：一个上游接收者、一个按音频节奏发送的发送者、一份有界音频与控制发送队列。启动、命令确认和关闭确认独立于浏览器播放。
- `EndToEndManager`：客户端就绪、当前通话、模型事件消费、待播 PCM、投递和播放停滞监督。
- `InputGateway` / `OutputGateway`：输入检查与 WebSocket 写入，复用 ASGI `RealtimeConnection` 原有的有界、顺序发送队列；播放预算由 Manager 管理。
- `_AudioPlaybackBudget`：位于 `end_to_end_manager.py` 内部，沿用原版 TTS 的分块与 Condition 预算控制方法，负责 Live 时长记账、登记撤销、流身份、累计样本回执和取消。`TTSManager` 保留原有内部记账实现。
- `WebOutputAudioSession`：两个入口共用一个 PCM 排程核心，保留原有音频时钟、恢复与缓存、淡入、实际开始通知、幂等完成、WebKit 音频时钟补偿、停止时部分播放结算和清理。Live 队列暂时为空没有轮次结束含义。
- 现有平台子类维持原有抽象方法要求；连续播放属于可选能力。

新版 `dev` 的分层保持不变：登录、历史和浏览器快照由示例应用 `host/` 与 `conversation-store.js` 管理；SDK 保持纯 Realtime 客户端。Live 在鉴权后跳过业务会话创建、历史写入与恢复。协议处理复用 Service 的输入处理器注册及输出订阅覆盖 API；克隆只继承一次订阅。

Live 链路已移除 `AudioStreamTransport`、独立连续播放器、输出屏蔽开关、Live SQL/历史投影、Manager 重复输入队列、`DuplexState`、`restore_history()`、`append_context()` 及第二套 action 分发。现有数据库表与数据保留。

## 配置与限制

示例使用 `gpt-live-1`、`marin` 音色、显式 Responses 后端 `gpt-5.6-luna`、`tools: []` 和 `tool_choice: "none"`。后端模型需要相应访问权限。模型的 `sample_rate` 同时决定采集与播放格式：单声道有符号 PCM16LE，支持 16000/24000 Hz，默认 24000。冲突的 `input_sample_rate` 会被拒绝。

| 配置 | 默认 | 管理者 |
| --- | --- | --- |
| 投递分块 | 100 ms | Live Manager 内部预算 |
| 在途音频 | 300 ms | Live Manager 内部预算 |
| 待播输出 | 5000 ms | Manager 的 `service_config.end_to_end.pending_audio_ms` |
| 无播放进展期限 | 15 s | Manager 的 `playback_stall_timeout_s` |
| 客户端媒体就绪 | 30 s | Manager 的 `client_ready_timeout_s` |
| 未发送输入/适配器待消费音频 | 5000 ms | GPTLive 的 `max_pending_audio_ms` |
| 适配器事件/发送队列 | 各 256 项 | GPTLive 的 `max_pending_events` |
| 命令确认/优雅关闭 | 各 15 s | GPTLive 的超时配置 |
| 浏览器 PCM 帧/JSON/上游消息 | 200 ms / 64 KiB / 1 MiB | Gateway / 适配器 |

300 ms 窗口用满后等待播放进展。长期待播积压或无播放进展才结束通话并提示原因。关闭唤醒等待中的投递任务并丢弃未发送输入。发布前登记预算。Live 音频投递通过 `WAIT_UNTIL_COMPLETE` 和 `require_success=True` 检查投递：缺少处理器或处理器抛错时返回 `False`，立即撤销预算。EventBus 默认的异常隔离行为保留。Live JSON 或 PCM 写入失败时仅通知通话管理者一次，覆盖启动消息和错误通知的发送失败。

Live 默认 `enable_persistence: false`。显式配置持久化、录音、完整音频导出、多说话人策略或强制对齐时，启动前报错。默认模型组装拒绝混入级联模型；显式 Manager 列表与 EventBus 订阅覆盖机制继续可用。

## 协议与启动

1. 完成示例应用鉴权，连接 `/v1/realtime?model=xtalk`。纯 SDK 使用宿主传入的 URL/API key。
2. 标准 `session.created` 的 `session.xtalk.end_to_end` 返回协议 2、全新 `stream_id`、媒体、能力、策略与元信息。
3. 客户端发送 `session.update`，通过 `session.xtalk` 选择 `end_to_end_protocol: 2` 和 `output_audio_encoding`；服务端回复 `session.updated`。
4. 只创建一套输入/输出对象，初始化媒体，携带 `stream_id` 发送 `xtalk.audio_stream.ready`。
5. 服务端打开上游并等待 `session.started`，随后发送 `xtalk.audio_stream.started`。当前有效且未取消的 Runtime 才标记 `connected`、开放上传、显示可以讲话。

协议 1 客户端收到明确的重新构建/刷新提示。就绪前的音频在本地丢弃；用户应在 UI 提示就绪后讲话。拒绝权限、关闭与取消都会撤销等待器。迟到的授权会停止媒体轨道，旧连接不能恢复新 Runtime 的状态。

客户端控制消息：

```json
{"type":"session.update","session":{"xtalk":{"end_to_end_protocol":2,"output_audio_encoding":"base64"}}}
{"type":"xtalk.audio_stream.ready","stream_id":"current-call"}
{"type":"xtalk.audio_stream.progress","stream_id":"current-call","played_samples":2400}
{"type":"xtalk.audio_stream.input_muted","stream_id":"current-call","muted":true}
{"type":"xtalk.audio_stream.close","stream_id":"current-call"}
```

输入使用二进制 PCM 或标准 `input_audio_buffer.append`。输出使用 `xtalk.audio_stream.delta`：默认 JSON 中的 `delta` 为 Base64；协商 `binary_pcm` 后，使用新版框架的 4 字节大端 JSON 头长度、UTF-8 JSON 头和 PCM 载荷。头包含 `version: 1`、`type: "xtalk.audio_stream.delta"`、`event_id` 和 `stream_id`。整个信封是一个 WebSocket 消息。

播放回执仅包含流身份与累计已播样本。旧流、递减、非整数或超过已发送量的回执被拒绝。实际音频块完成和停止结算产生回执；前端不增加定时进度轮询或第二个播放窗口。

## 输入、转写与 UI

浏览器采集保留 AEC，Live 关闭额外 VAD 与增强。持续发送 PCM；麦克风静音仅在输入层生成一次等时长零音频。上游静音命令的确认在内部管理。原 `Session.muted` 偏好在重开后保留。

`xtalk.transcript.delta` 保留说话方、精确增量文本、近似开始/结束时间、上游事件身份、流身份与本地到达序号。重复事件去重。这些独立片段不具有 ASR 收尾、ResponseFinish 或已听文本含义，时间戳仅用于展示排序。

示例复用开始/停止/静音按钮与聊天气泡。连接中可以取消。错误通过 Toast 和状态文本展示，默认语音视图也可见。转写在停止后保留，到下次开始或刷新时清空。Live 隐藏历史、上传、动态音色和调试入口，并停止相应请求。宿主可通过 `SessionConfig.mode: "end_to_end"` 在连接前跳过浏览器历史快照；示例从服务端渲染的模式传入此值。

## 停止与错误

立即停止本地采集、排程、观察任务及待播 PCM。适配器丢弃未发送输入，发送 `session.close`，保持接收者，最多等待 15 秒的 `session.closed`。`close()` 幂等。

`xtalk.audio_stream.ended` 包含 `outcome`、`reason` 和 `finalization`（`complete`、`incomplete`、`not_started`），区分用户停止、上游正常结束、传输/协议失败和未确认关闭。上游安全终止、到期和连接丢失有可见原因。命令拒绝、音频或后端错误可以在通话仍活跃时发生。传输故障结束当前通话，提示手动重开，新通话使用空上下文。

## 验证

运行后端行为测试与[浏览器验收流程](../tutorial/gpt_live.zh.md)，覆盖六块突发、播放延迟、暂停、`onended` 缺失、停滞、启动取消、权限拒绝、就绪后的首段音频、静音偏好、十次重开、错误、16/24 kHz、版本拒绝及旧数据保留。级联回归覆盖响应身份、TTS 对齐、轮次竞态、ASR 收尾、工具前播放屏障和历史。

真实 API 连接与合成语音验证协议及浏览器链路。对话听感、附和、真实麦克风声学条件与原生插话效果需要真人验证，已验证边界见教程。

官方资料：[WebSockets](https://developers.openai.com/api/docs/guides/voice-websockets?api=live)、[Managing sessions](https://developers.openai.com/api/docs/guides/live-conversations)。
