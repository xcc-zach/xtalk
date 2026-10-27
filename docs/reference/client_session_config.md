# Client Session Config

```ts
createSession(wsUrl: string, config?: SessionConfig): Session
```

`SessionConfig` carries the overrides applied when a browser session is created. It has three groups of options: `inputConfig`, `outputConfig`, and `serviceURLs`. Every group is optional; omitted fields keep their defaults. The generated type listing is in [SessionConfig](../api/client/interfaces/SessionConfig.md).

```ts
const session = createSession(wsUrl, {
    inputConfig: {
        sampleRate: 16000,
        enableVAD: true,
        enableEnhancer: true,
        vadRedemptionMs: 500,
        frontendUtilitiesBaseUrl: "/xtalk/frontend-utilities",
    },
    outputConfig: {
        sampleRate: 48000,
    },
    serviceURLs: {
        login: "/api/auth/login",
        sessions: "/api/sessions",
        sessionDetail: (sessionId) => `/api/sessions/${sessionId}`,
        upload: "/api/upload",
    },
});
```

## inputConfig

Controls the frontend input audio session. In browser microphone mode the commonly used fields are:

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | `number` | `16000` | Input audio sample rate. |
| `enableVAD` | `boolean` | `true` | Run frontend VAD on the input audio. |
| `enableEnhancer` | `boolean` | `true` | Run frontend speech enhancement on the input audio. |
| `vadRedemptionMs` | `number` | `500` | VAD redemption window in milliseconds. |
| `frontendUtilitiesBaseUrl` | `string` | `/xtalk/frontend-utilities` | Base URL for browser-side ONNX Runtime, VAD, and FastEnhancer assets. When ONNX Runtime or VAD assets are missing at that server path, the frontend loads them from public CDNs. |

## outputConfig

Controls playback.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `sampleRate` | `number` | `48000` | Output playback sample rate. |

## serviceURLs

Overrides the server HTTP endpoints the client calls.

| Field | Type | Description |
| --- | --- | --- |
| `login` | `string` | Login endpoint. |
| `sessions` | `string` | Session list endpoint. |
| `sessionDetail` | `string \| (sessionId: string) => string` | Session detail endpoint, as a fixed URL or a function of the session ID. |
| `upload` | `string` | Upload endpoint. |

When omitted, the frontend derives the defaults from `wsUrl`:

- `POST /api/auth/login`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/upload`

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Generated SessionConfig type →](../api/client/interfaces/SessionConfig.md)** · Reference

-   **[Start the Service →](../tutorial/start_the_service.md)** · Tutorial

-   **[How frontend VAD decides speech boundaries →](../explanation/vad.md)** · Explanation

</div>
