# 音频桥接

```ts
createAudioBridge(): WebAudioBridge
```

创建 bot2bot 会话使用的浏览器端共享音频总线。`createAudioBridge()` 是包根提供的统一入口；它会自动检测当前前端平台，现阶段落到 Web bridge 实现。生成的签名见 [createAudioBridge](../api/client/functions/createAudioBridge.zh.md)。

下面的具体类型名描述的是当前 Web 实现。包根只导出 `createAudioBridge()`，不导出这些带 `Web` 前缀的类型。

```ts
type WebBridgeParticipantId = string;

interface WebAudioBridgeUserInputConfig {
    sourceId?: WebBridgeParticipantId;
    sampleRate?: number;
    enableVAD?: boolean;
    enableEnhancer?: boolean;
    vadRedemptionMs?: number;
}

interface WebAudioBridgePublishOptions {
    sourceId: WebBridgeParticipantId;
    sampleRate: number;
}

interface WebAudioBridge {
    openUserInput(config?: WebAudioBridgeUserInputConfig): Promise<void>;
    closeUserInput(): Promise<void>;
    publishAudio(
        pcmChunkInt16: ArrayBuffer,
        options: WebAudioBridgePublishOptions,
    ): void;
    publishSpeechStart(sourceId: WebBridgeParticipantId): void;
    publishSpeechEnd(sourceId: WebBridgeParticipantId): void;
    close(): Promise<void>;
}
```

## 方法

| 方法 | 说明 |
| --- | --- |
| `openUserInput(config?)` | 把浏览器麦克风接入共享流。`enableVAD` 为用户音频广播 `speechStart`/`speechEnd`；`enableEnhancer` 在发布前先做前端增强；`vadRedemptionMs` 是判定用户语音结束所需的静默时长。 |
| `closeUserInput()` | 停止麦克风采集。 |
| `publishAudio(pcmChunkInt16, options)` | 以 `options.sourceId` 的身份把一段采样率为 `options.sampleRate` 的 PCM 混入共享流。 |
| `publishSpeechStart(sourceId)` | 为 `sourceId` 广播语音开始信号。 |
| `publishSpeechEnd(sourceId)` | 为 `sourceId` 广播语音结束信号。 |
| `close()` | 释放 bridge。应在用户输入和 bot session 关闭之后调用。 |

## 会话输入字段

会话通过 `inputConfig` 读取 bridge：

| 字段 | 说明 |
| --- | --- |
| `mode` | 设为 `"web_bridge"` 时，会话读取共享 bridge 流。 |
| `participantId` | 前端 bridge 使用的参与者标识，用于按 source 控制 VAD 行为以及调试。 |
| `bridge` | 要订阅的 `WebAudioBridge` 实例。 |
| `autoEmitVad` | 该 bot 的输出重新写回共享流时，是否同时广播前端 VAD 信号。 |
| `vadRedemptionMs` | 启用 `autoEmitVad` 时，语音结束的静默收尾时间。 |

## 限制

- bridge 只支持 Web 平台，实现位于 [`frontend/src/platforms`](https://github.com/xcc-zach/xtalk/tree/main/frontend/src/platforms)。
- 共享音频总线完全在前端实现，不会修改服务端 agent 配置。
- `participantId` 只是前端 bridge 的参与者标识，服务端没有对应的 speaker id。
- bot 不会听到自己刚刚发布回 bridge 的音频。
- 当多个 bot 的输出持续写回同一条共享流时，它们可能会一直互相回应，直到您显式停止它们。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[Bot2Bot →](../tutorial/bot2bot.zh.md)** · Tutorial

-   **[音频桥接模型 →](../explanation/audio_bridge.zh.md)** · Explanation

-   **[生成的 createAudioBridge 签名 →](../api/client/functions/createAudioBridge.zh.md)** · Reference

</div>
