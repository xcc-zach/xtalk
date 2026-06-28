X-Talk 的大多数模型与执行逻辑都运行在服务端。客户端主要负责访问麦克风、传输音频、收发 WebSocket 消息，以及处理轻量级的会话逻辑。

客户端 API 已经发布为独立包 [xtalk-client](https://www.npmjs.com/package/xtalk-client)，其公开接口由 [`frontend/src/index.ts`](https://github.com/xcc-zach/xtalk/blob/main/frontend/src/index.ts) 导出。

## 客户端最简接入

`createSession()` 接收一个 WebSocket 地址。当您调用 `session.open()` 时，客户端会：

1. 调用登录接口；
2. 获取访问令牌；
3. 携带该令牌连接 WebSocket；
4. 保持当前会话状态与服务端同步。

如果您使用打包工具，请先安装：

```bash
npm install xtalk-client
```

> 对开发者来说，客户端代码位于 [`frontend`](https://github.com/xcc-zach/xtalk/tree/main/frontend) 目录下。在该目录中，您可以运行 `npm run watch` 进行开发构建。

然后在前端代码中导入并使用：

```ts
import { createSession } from "xtalk-client";

const wsUrl =
    location.protocol === "https:"
        ? `wss://${location.host}/ws`
        : `ws://${location.host}/ws`;

const session = createSession(wsUrl);

session.onStateChange((state) => {
    console.log("connection:", state.connectionState);
    console.log("session:", state.sessionId);
    console.log("messages:", state.messages);
});

await session.open();
```

如果您不想自行打包客户端代码，也可以直接通过 CDN 加载：

```html
<script type="module">
    const { createSession } = await import("https://unpkg.com/xtalk-client@0.2.6/dist/index.js");

    const wsUrl =
        location.protocol === "https:"
            ? `wss://${location.host}/ws`
            : `ws://${location.host}/ws`;

    const session = createSession(wsUrl);
    await session.open();
</script>
```

## 服务端最简接入

对于公开发布的 `xtalk-client`，推荐的服务端接法是：从配置文件创建一个 `Xtalk` 实例，并把它的内置路由挂到 FastAPI 应用上：

```python
from fastapi import FastAPI
from xtalk import Xtalk

app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config("path/to/config.json")
xtalk_instance.mount_routes(app)
```

关于 `config.json`，参考[配置服务](config_the_service.zh.md)。

这会挂载前端默认使用的这些接口：

- `POST /api/auth/login`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/upload`
- `WS /ws`

在这些默认路由存在时，前端只需要调用 `createSession(wsUrl)`，然后执行 `session.open()` 即可。

## 自定义路由

如果您的后端没有使用默认的内置路由路径，那么在创建 session 时需要显式传入这些服务地址：

```ts
const session = createSession(wsUrl, {
    serviceURLs: {
        login: "/custom/login",
        sessions: "/custom/sessions",
        sessionDetail: (sessionId) => `/custom/sessions/${sessionId}`,
        upload: "/custom/upload",
    },
});
```

## 完整示例

如果您需要包含静态资源、模板、上传界面、会话切换和音色切换的完整示例，可继续参考：

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)
