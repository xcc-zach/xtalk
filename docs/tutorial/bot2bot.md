# Bot2Bot

*Experimental feature*

This tutorial ends with two X-Talk sessions in one browser page talking to each other through a shared audio bus, with the option of your own microphone joining the same stream. You will create the bridge, point two bot sessions at it, feed each bot's spoken output back in, and shut everything down cleanly.

Before starting, complete [Start the Service](start_the_service.md) so that a server is running and `createSession()` works from your page. The design behind the shared stream is explained in [Audio Bridge Model](../explanation/audio_bridge.md).

## Create the bridge

`createAudioBridge()` is the package-level entrypoint. It detects the current frontend platform and currently resolves to the Web bridge implementation. Import it from the client bundle:

```ts
import { createAudioBridge } from "xtalk-client";

const bridge = createAudioBridge();
```

The bridge now maintains one continuous 16 kHz PCM stream that contains silence until someone publishes audio. Its instance API is listed in [Audio Bridge](../reference/audio_bridge.md).

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

After `await botA.open()`, this bot hears the bridge stream. Nobody has published audio yet, so it hears silence and stays quiet.

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

Once one bot produces a reply, the other hears it and answers, and the exchange keeps going until you stop the sessions.

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

The browser asks for microphone permission at this call. The meaning of these fields is intentionally aligned with the existing web microphone input config:

- `enableVAD`: whether frontend VAD should emit `speechStart` and `speechEnd` for user audio.
- `enableEnhancer`: whether frontend enhancement runs before user audio is published.
- `vadRedemptionMs`: how long silence must last before user speech is considered finished.

Speak, and both bots hear you. If you do not want the bridge to capture the user microphone, do not call `openUserInput()`.

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

After `bridge.close()` resolves, no session reads from the stream and the bots stop answering each other.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Audio Bridge Model: how the shared stream mixes and filters audio →](../explanation/audio_bridge.md)** · Explanation

-   **[Audio Bridge: instance API and limitations →](../reference/audio_bridge.md)** · Reference

</div>
