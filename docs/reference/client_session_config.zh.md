# 客户端会话配置

```ts
createSession(wsUrl: string, config?: SessionConfig): Session
```

`SessionConfig` 携带创建浏览器会话时应用的覆盖项，包含三组配置：`inputConfig`、`outputConfig` 和 `serviceURLs`。每组都可省略；省略的字段保持默认值。生成的类型定义见 [SessionConfig](../api/client/interfaces/SessionConfig.zh.md)。

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

配置前端输入音频会话。浏览器麦克风模式下常用字段如下：

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sampleRate` | `number` | `16000` | 输入音频采样率。 |
| `enableVAD` | `boolean` | `true` | 对输入音频运行前端 VAD。 |
| `enableEnhancer` | `boolean` | `true` | 对输入音频运行前端语音增强。 |
| `vadRedemptionMs` | `number` | `500` | VAD 的 redemption 窗口，单位为毫秒。 |
| `frontendUtilitiesBaseUrl` | `string` | `/xtalk/frontend-utilities` | 浏览器端 ONNX Runtime、VAD 和 FastEnhancer 资源的基础地址。服务端对应路径缺少 ONNX Runtime 或 VAD 资源时，前端改从公共 CDN 加载。 |

## outputConfig

配置播放。

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `sampleRate` | `number` | `48000` | 输出播放采样率。 |

## serviceURLs

覆盖客户端调用的服务端 HTTP 接口地址。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `login` | `string` | 登录接口。 |
| `sessions` | `string` | 会话列表接口。 |
| `sessionDetail` | `string \| (sessionId: string) => string` | 会话详情接口，可以是固定 URL，也可以是以会话 ID 为参数的函数。 |
| `upload` | `string` | 上传接口。 |

省略时，前端根据 `wsUrl` 推导默认地址：

- `POST /api/auth/login`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/upload`

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[生成的 SessionConfig 类型 →](../api/client/interfaces/SessionConfig.zh.md)** · Reference

-   **[启动服务 →](../tutorial/start_the_service.zh.md)** · Tutorial

-   **[前端 VAD 如何判定语音边界 →](../explanation/vad.zh.md)** · Explanation

</div>
