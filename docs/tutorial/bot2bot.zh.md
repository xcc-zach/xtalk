*测试中的功能*

X-Talk 现在支持在浏览器端把多个前端会话接到同一条共享音频总线上，让 bot 听到的是一条连续音频流，而不只是麦克风输入。

本文档介绍当前这套仅前端实现的 bot2bot bridge API：

- 共享音频总线通过 `createAudioBridge()` 创建；
- 每个 bot session 使用 `inputConfig.mode = "web_bridge"`；
- 用户麦克风可以选择由 bridge 直接接入，也可以完全不接入；
- bot 的输出音频会重新发布回 bridge，从而让其他 bot 继续听到并回复。

`createAudioBridge()` 是包根提供的统一入口。它会自动检测当前前端平台；在现阶段，检测到 Web 时会落到 Web bridge 实现。

## 适用场景

当您希望一个浏览器页面同时做到下面几件事时，可以使用 web audio bridge：

- 同时连接多个 X-Talk session；
- 让 bot 回应其他 bot 的语音输出；
- 可选地把真实用户麦克风音频也注入同一条共享流；
- 保持现有服务端接口不变，把音频编排放在前端完成。

这套 API 目前只支持 Web 平台，实现位于 `frontend/src/platforms`。

## 共享流模型

bridge 内部维护一条 `16000` Hz 的连续 PCM 音频流。

- 没有人说话时，这条流会持续输出静默；
- 用户或 bot 发布音频时，这些音频会被混入这条共享流；
- 所有配置为 `mode: "web_bridge"` 的 bot session 都会持续收到来自同一条共享流的音频帧，但不会收到自己刚发布回去的那一路音频。

这意味着 bridge 并不是点对点路由 bot 音频，而是让所有发布者都往一条总线上写，所有 bridge bot 都从这条总线上读，同时每个 bot 会自动过滤掉自己刚发布的音频。

## 创建 bridge

从客户端包中导入 bridge 构造函数：

```ts
import { createAudioBridge } from "xtalk-client";

const bridge = createAudioBridge();
```

bridge 实例当前公开的 API 如下：

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

这里展示的具体类型名描述的是当前 Web 实现；包根现在只导出 `createAudioBridge()`，不再直接导出这些带 `Web` 前缀的类型。

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

- `mode`：设置为 `"web_bridge"`，表示这个 session 不再读取麦克风，而是读取共享 bridge 流；
- `participantId`：前端 bridge 使用的参与者标识，用于按 source 控制 VAD 行为以及调试；
- `bridge`：当前 bot 要订阅的 `WebAudioBridge` 实例；
- `autoEmitVad`：该 bot 输出重新写回共享流时，是否同时广播前端 VAD 信号；
- `vadRedemptionMs`：启用 `autoEmitVad` 时，语音结束的静默收尾时间。

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

如果两个 bot 都这样做，它们就可以通过共享流继续互相回应：

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

这里这些字段的语义故意与现有 Web 麦克风输入配置保持一致：

- `enableVAD`：是否为用户音频广播前端 `speechStart` 和 `speechEnd`；
- `enableEnhancer`：发布用户音频前是否先做前端增强；
- `vadRedemptionMs`：静默持续多久后才把用户语音视为结束。

如果您不希望 bridge 接入用户麦克风，就不要调用 `openUserInput()`。

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

关闭时，建议先关闭用户输入，再关闭 bot session，最后关闭 bridge 本身：

```ts
await bridge.closeUserInput();
await botA.close();
await botB.close();
await bridge.close();
```

## 当前限制

- 这套 bridge 目前只支持 Web 平台；
- 共享音频总线完全在前端实现，不会修改服务端 agent 配置；
- `participantId` 不是服务端的 speaker id，它只是前端 bridge 的参与者标识；
- bot 不会听到自己刚刚发布回 bridge 的音频；
- 当多个 bot 的输出持续写回同一条共享流时，它们可能会一直互相回应，直到您显式停止它们。
