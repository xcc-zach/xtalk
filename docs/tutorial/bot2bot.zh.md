# Bot2Bot

*实验中的功能*

完成本教程后，同一个浏览器页面里的两个 X-Talk 会话会通过一条共享音频总线互相对话，您的麦克风也可以选择加入同一条流。您将创建 bridge、让两个 bot 会话读取它、把每个 bot 的语音输出发布回去，并干净地关闭所有内容。

开始前请先完成[启动服务](start_the_service.zh.md)，确保服务已在运行，并且页面中的 `createSession()` 可用。共享流背后的设计见[音频桥接模型](../explanation/audio_bridge.zh.md)。

## 创建 bridge

`createAudioBridge()` 是包根提供的统一入口。它会自动检测当前前端平台，现阶段落到 Web bridge 实现。从客户端包中导入它：

```ts
import { createAudioBridge } from "xtalk-client";

const bridge = createAudioBridge();
```

bridge 此时维护一条连续的 16 kHz PCM 流；在有人发布音频之前，这条流只包含静默。实例 API 见[音频桥接](../reference/audio_bridge.zh.md)。

## 配置 bot session

每个 bot 仍然使用普通的 `createSession()` API，只是把输入侧从麦克风切换成 bridge 共享流：

```ts
import { createSession } from "xtalk-client";

const botA = createSession("/ws", {
    inputConfig: {
        sampleRate: 16000,
        mode: "web_bridge",
        participantId: "bot-a",
        bridge,
        autoEmitVad: true,
        vadRedemptionMs: 500,
    },
});
```

这些 bridge 相关输入字段的含义如下：

- `mode`：设置为 `"web_bridge"`，表示这个 session 读取共享 bridge 流；
- `participantId`：前端 bridge 使用的参与者标识，用于按 source 控制 VAD 行为以及调试；
- `bridge`：当前 bot 要订阅的 `WebAudioBridge` 实例；
- `autoEmitVad`：该 bot 输出重新写回共享流时，是否同时广播前端 VAD 信号；
- `vadRedemptionMs`：启用 `autoEmitVad` 时，语音结束的静默收尾时间。

`await botA.open()` 之后，这个 bot 开始监听 bridge 流。此时还没有人发布音频，它听到的是静默，因此保持安静。

## 把 bot 输出重新发布回 bridge

当 bot 开始说话时，监听它的输出音频，并把这些 PCM 音频块重新发布回 bridge：

```ts
botA.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-a",
        sampleRate,
    });
});
```

如果两个 bot 都这样做，它们就可以通过共享流互相回应：

```ts
const botA = createSession("/ws", {
    inputConfig: {
        sampleRate: 16000,
        mode: "web_bridge",
        participantId: "bot-a",
        bridge,
        autoEmitVad: true,
        vadRedemptionMs: 500,
    },
});

const botB = createSession("/ws", {
    inputConfig: {
        sampleRate: 16000,
        mode: "web_bridge",
        participantId: "bot-b",
        bridge,
        autoEmitVad: false,
    },
});

botA.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-a",
        sampleRate,
    });
});

botB.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-b",
        sampleRate,
    });
});

await botA.open();
await botB.open();
```

在这个配置下：

- 两个 bot 都会持续接收同一条共享音频流，但不包含自己刚发布的音频；
- bot 输出会被重新写回这条流；
- bot 可以继续回复其他 bot 的回复。

一旦某个 bot 产生回复，另一个 bot 就会听到并作答，这样的交流会持续到您停止会话为止。

## 让 bridge 直接接收用户音频

如果您希望浏览器麦克风也进入共享流，可以直接在 bridge 上打开用户输入：

```ts
await bridge.openUserInput({
    sourceId: "user",
    sampleRate: 16000,
    enableVAD: true,
    enableEnhancer: true,
    vadRedemptionMs: 500,
});
```

调用时浏览器会弹出麦克风权限请求。这些字段的语义与现有 Web 麦克风输入配置保持一致：

- `enableVAD`：是否为用户音频广播前端 `speechStart` 和 `speechEnd`；
- `enableEnhancer`：发布用户音频前是否先做前端增强；
- `vadRedemptionMs`：静默持续多久后才把用户语音视为结束。

说话时，两个 bot 都能听到您。如果您不希望 bridge 接入用户麦克风，就不要调用 `openUserInput()`。

## 完整示例

下面这个例子同时启动两个 bot，把两个 bot 的输出都发布回 bridge，并把真实用户麦克风音频也接入共享流：

```ts
import { createSession, createAudioBridge } from "xtalk-client";

const bridge = createAudioBridge();

const botA = createSession("/ws", {
    inputConfig: {
        sampleRate: 16000,
        mode: "web_bridge",
        participantId: "bot-a",
        bridge,
        autoEmitVad: true,
        vadRedemptionMs: 500,
    },
});

const botB = createSession("/ws", {
    inputConfig: {
        sampleRate: 16000,
        mode: "web_bridge",
        participantId: "bot-b",
        bridge,
        autoEmitVad: false,
    },
});

botA.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-a",
        sampleRate,
    });
});

botB.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-b",
        sampleRate,
    });
});

await botA.open();
await botB.open();

await bridge.openUserInput({
    sourceId: "user",
    sampleRate: 16000,
    enableVAD: true,
    enableEnhancer: true,
    vadRedemptionMs: 500,
});
```

## 停止 bridge

关闭时，先关闭用户输入（如果已打开），再关闭 bot session，最后关闭 bridge 本身：

```ts
await bridge.closeUserInput();
await botA.close();
await botB.close();
await bridge.close();
```

`bridge.close()` 完成后，没有会话再读取这条流，bot 之间停止互相回答。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[音频桥接模型：共享流如何混音与过滤 →](../explanation/audio_bridge.zh.md)** · Explanation

-   **[音频桥接：实例 API 与限制 →](../reference/audio_bridge.zh.md)** · Reference

</div>
