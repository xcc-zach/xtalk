*Experimental functionality*

X-Talk can bridge multiple frontend sessions into one shared browser-side audio bus so that bots hear a continuous audio stream instead of only microphone input.

This tutorial documents the current frontend-only bot-to-bot bridge API:

- the shared bus is created with `createAudioBridge()`,
- each bot session uses `inputConfig.mode = "web_bridge"`,
- user audio may optionally be captured directly by the bridge,
- bot output audio is published back into the bridge so other bots can respond.

`createAudioBridge()` is the package-level entrypoint. It detects the current frontend platform and currently resolves to the Web bridge implementation.

## When to use it

Use the web audio bridge when you want one browser page to:

- connect multiple X-Talk sessions at the same time,
- let bots respond to other bots' spoken output,
- optionally inject real user microphone audio into the same shared stream,
- keep the server contract unchanged and do the routing on the frontend.

This API is currently Web-only. It lives under `frontend/src/platforms`.

## Shared stream model

The bridge maintains one continuous PCM stream at `16000` Hz.

- When nobody is speaking, the stream contains silence.
- When the user or a bot publishes audio, that audio is mixed into the stream.
- Every bot configured with `mode: "web_bridge"` keeps receiving frames from the same shared stream, except for audio chunks that bot published itself.

This means the bridge does not route bot audio point-to-point. Instead, all publishers write into one shared stream, and all bridge-backed bot sessions read from it, while each bot's own published audio is filtered out from its local input stream.

## Create the bridge

Import the bridge constructor from the client bundle:

```ts
import { createAudioBridge } from "xtalk-client";

const bridge = createAudioBridge();
```

The public bridge instance API is:

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

The concrete type names shown here describe the current Web implementation. The package root only exports `createAudioBridge()`, not these Web-specific types.

## Configure a bot session

Each bot still uses the normal `createSession()` API, but the input side is switched from microphone capture to the shared bridge stream:

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

The bridge-related input fields are:

- `mode`: set to `"web_bridge"` so the session reads from the shared bridge stream.
- `participantId`: a frontend-side identifier used by the bridge for source-specific VAD behavior and diagnostics.
- `bridge`: the `WebAudioBridge` instance to subscribe to.
- `autoEmitVad`: whether audio published by this bot should also trigger frontend VAD events when re-injected into the shared stream.
- `vadRedemptionMs`: the speech-end redemption window used when `autoEmitVad` is enabled.

## Publish bot output back into the bridge

When a bot speaks, listen to its output audio and publish those PCM chunks back into the bridge:

```ts
botA.onOutputAudioChunk((pcm, sampleRate) => {
    bridge.publishAudio(pcm, {
        sourceId: "bot-a",
        sampleRate,
    });
});
```

If two bots both do this, they can respond to each other through the shared stream:

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

With this setup:

- both bots keep receiving the same shared stream except for their own published audio,
- bot output is written back into that stream,
- bots can continue replying to other bots' replies.

## Let the bridge capture real user audio

If you want the browser microphone to also feed the shared stream, open user input directly on the bridge:

```ts
await bridge.openUserInput({
    sourceId: "user",
    sampleRate: 16000,
    enableVAD: true,
    enableEnhancer: true,
    vadRedemptionMs: 500,
});
```

The meaning of these fields is intentionally aligned with the existing web microphone input config:

- `enableVAD`: whether frontend VAD should emit `speechStart` and `speechEnd` for user audio.
- `enableEnhancer`: whether frontend enhancement runs before user audio is published.
- `vadRedemptionMs`: how long silence must last before user speech is considered finished.

If you do not want the bridge to capture the user microphone, do not call `openUserInput()`.

## Full example

The example below starts two bots, publishes both bots' output back into the bridge, and also injects real user microphone audio:

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

## Stop the bridge

When shutting down, close user input first if it is active, then close bot sessions, then close the bridge itself:

```ts
await bridge.closeUserInput();
await botA.close();
await botB.close();
await bridge.close();
```

## Current limitations

- The bridge currently exists only for the Web platform.
- The shared bus is implemented on the frontend and does not change server-side agent configuration.
- `participantId` is not a server-side speaker id. It is only a frontend bridge identifier.
- A bot does not hear the audio it just published back into the bridge itself.
- When several bots publish output back into the same stream, they may continue responding to each other indefinitely until you stop them.
