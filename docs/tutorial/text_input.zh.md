# 文本输入

连接 Session 后，使用 `sendText()` 发送文本：

```ts
const session = createSession(websocketURL, config);

await session.open();
await session.sendText("设置一个两秒计时器。");
```

文本会作为一个完整的用户回合发送，并继续经过现有的 Agent、工具调用和 TTS 流程。Session 处于 `idle`、`listening`、`processing` 或 `speaking` 状态时均可发送；如果当前正在识别、生成或播放回复，新文本会替代当前回合。

发送文本不会关闭或重新创建麦克风采集。

## 使用限制

- Session 必须处于已连接状态。
- 文本不能为空，且最多包含 2,048 个 JavaScript 字符。
- 同一 Session 同时只能提交一条等待确认的文本。
- 确认超时时间为 10 秒。
- 断线、关闭或切换 Session 会取消尚未完成的提交。
- 客户端不会自动排队或重试。
