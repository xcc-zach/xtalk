# Audio Bridge

```ts
createAudioBridge(): WebAudioBridge
```

Creates the shared browser-side audio bus used by bot-to-bot sessions. `createAudioBridge()` is the package-level entrypoint; it detects the current frontend platform and currently resolves to the Web bridge implementation. The generated signature is in [createAudioBridge](../api/client/functions/createAudioBridge.md).

The concrete type names below describe the current Web implementation. The package root only exports `createAudioBridge()`, not these Web-specific types.

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

## Methods

| Method | Description |
| --- | --- |
| `openUserInput(config?)` | Captures the browser microphone into the shared stream. `enableVAD` emits `speechStart`/`speechEnd` for user audio; `enableEnhancer` runs frontend enhancement before publishing; `vadRedemptionMs` is the silence length that ends user speech. |
| `closeUserInput()` | Stops microphone capture. |
| `publishAudio(pcmChunkInt16, options)` | Mixes a PCM chunk recorded at `options.sampleRate` into the shared stream under `options.sourceId`. |
| `publishSpeechStart(sourceId)` | Broadcasts a speech-start signal for `sourceId`. |
| `publishSpeechEnd(sourceId)` | Broadcasts a speech-end signal for `sourceId`. |
| `close()` | Releases the bridge. Call after user input and bot sessions are closed. |

## Session input fields

A session reads from the bridge through `inputConfig`:

| Field | Description |
| --- | --- |
| `mode` | `"web_bridge"` makes the session read from the shared bridge stream. |
| `participantId` | Frontend-side identifier used by the bridge for source-specific VAD behavior and diagnostics. |
| `bridge` | The `WebAudioBridge` instance to subscribe to. |
| `autoEmitVad` | Whether audio published by this bot also triggers frontend VAD events when re-injected into the shared stream. |
| `vadRedemptionMs` | Speech-end redemption window used when `autoEmitVad` is enabled. |

## Limitations

- The bridge exists only for the Web platform. It lives under [`frontend/src/platforms`](https://github.com/xcc-zach/xtalk/tree/main/frontend/src/platforms).
- The shared bus is implemented on the frontend and does not change server-side agent configuration.
- `participantId` is a frontend bridge identifier; the server has no matching speaker id.
- A bot does not hear the audio it just published back into the bridge itself.
- When several bots publish output back into the same stream, they may continue responding to each other indefinitely until you stop them.

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Bot2Bot →](../tutorial/bot2bot.md)** · Tutorial

-   **[Audio Bridge Model →](../explanation/audio_bridge.md)** · Explanation

-   **[Generated createAudioBridge signature →](../api/client/functions/createAudioBridge.md)** · Reference

</div>
