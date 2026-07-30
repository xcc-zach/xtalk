# 文本输入

## 实现概览

当前 `xtalk-client` 已公开 `Session.sendText(text)`。只要实时连接有效，就可以在
`idle`、`listening`、`processing` 和 `speaking` 任意流状态发送文本；
`sendText` 本身不限制流状态。

打开 Session 时仍按原有逻辑申请麦克风权限并启动音频链路。发送文本不会启停、
静音或重新创建麦克风采集。

## 客户端行为

```ts
const session = createSession(websocketURL, config);

await session.open();
await session.sendText("设置一个两秒计时器。");
```

`sendText` 依次执行：

1. 删除首尾空白；
2. 拒绝空文本，以及规范化后 JavaScript `length` 超过 2,048 的字符串；
3. 要求 `connectionState` 为 `connected`；
4. 同一时间只允许一条文本等待确认；
5. 先注册 `finish_asr` waiter，再发送请求。

WebSocket 请求为：

```json
{
  "action": "submit_text",
  "text": "设置一个两秒计时器。"
}
```

客户端不会乐观写入用户消息。现有 `finish_asr` handler 负责追加最终用户消息，
因此 Conversation 中只保留一份经过服务端确认的内容。

确认超时为 10 秒。断线、关闭、重新打开或切换 Session 都会取消尚未完成的调用。
客户端不会排队，也不会自动重试文本。

## 服务端处理

`TextMsgHandler` 要求 `text` 为字符串，删除首尾空白，拒绝空文本，并限制 UTF-8
编码后不超过 8 KiB。服务端忽略客户端传入的 `origin`，按 Session 串行发布一个
合成语音回合：

```text
submit_text
  -> TurnInputAbortRequested(origin="text")
  -> VADSpeechStart(origin="text")
  -> ASRResultPartial(origin="text")
  -> VADSpeechEnd(origin="text")
  -> ASRResultFinal(origin="text")
  -> 现有 Agent、Tool、TTS 和持久化流程
```

ASR partial 位于两个 VAD 边界之间；ASR final 在 VAD end 完成后发布，使前端最终
进入 `processing`，不会被迟到的 VAD end 改回 `idle`。

各 Manager 对文本来源事件的处理如下：

- `ASRManager` 立即使正在执行的识别失效，清空语音 ASR 状态，并在合成 VAD end
  之前丢弃麦克风帧。识别 generation token 会阻止被替代音频回合的迟到结果发布。
- `VADManager` 清空缓存和检测状态，并在同一时间段暂停处理语音 VAD。
- `TurnTakingManager` 不为文本启动或结束语音 ASR。文本 VAD start 会取消可中断的
  LLM 生成，并以 `text_input` 为原因请求停止 TTS。
- `TurnDetectorManager` 忽略文本来源的 VAD 和 ASR partial，因为点击发送已经确定
  用户回合结束。
- `LatencyManager` 在统计文本回合前清除旧语音回合遗留的前端 VAD 时间戳。

输入元数据统一使用 `origin`：

- `client`：客户端产生的音频 VAD；
- `turn_detector`：Turn Detector 产生的边界；
- `asr`：普通音频识别；
- `text`：用户提交的文本回合。

## 状态语义

| 当前流状态 | 发送文本后的行为 |
| --- | --- |
| `idle` | 执行合成文本回合，没有活动输入需要替换。 |
| `listening` | 先使未完成的语音输入失效并清空，再处理文本。 |
| `processing` | 取消可中断的生成，再开始文本回合。 |
| `speaking` | 请求停止生成和 TTS 播放，再开始文本回合。 |

## 隐式确认

`OutputGateway` 已在现有 ASR 消息中包含 `origin`。实现没有增加独立的成功确认
消息或 request ID。成功处理文本后会收到：

```json
{
  "action": "finish_asr",
  "data": {
    "text": "设置一个两秒计时器。",
    "origin": "text"
  }
}
```

普通 `finish_asr` handler 会先更新 Conversation。只有消息满足
`origin="text"`，且 `text` 与规范化后的提交内容一致时，`sendText` Promise 才会
resolve。

## 当前限制

- Session 必须处于已连接状态；断线和重连期间直接拒绝发送。
- 每个客户端 Session 同一时间只能有一条文本等待确认。
- 客户端限制为规范化后 JavaScript 字符串 `length` 不超过 2,048；服务端协议限制
  为 UTF-8 编码后不超过 8 KiB。
- 确认通过 `origin` 和规范化文本匹配，没有 request ID。
- 客户端不会自动重试，因为重复回合可能再次执行具有副作用的工具。
- 麦克风继续正常采集；处理合成文本边界期间，VAD 和 ASR 会暂时丢弃对应音频帧。

## 自动化测试

`tests/test_text_input.py` 覆盖请求校验、事件顺序、Session 内串行化、`origin`
回显、绕过语音 ASR、VAD 抑制、跳过 Turn Detector 和延迟状态重置。
`tests/test_asr_finalization.py` 覆盖正在执行的最终 ASR 被文本回合取消后不再发布。

前端生产构建会验证发布类型声明和浏览器 bundle 中包含 `sendText`。
