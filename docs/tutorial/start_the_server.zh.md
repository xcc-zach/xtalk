> [!NOTE]
> 详情请参阅 `examples/sample_app/configurable_server.py`、`frontend/src`、`examples/sample_app/templates` 和 `examples/sample_app/static`。

X-Talk 的大多数模型与执行逻辑都运行在服务端，客户端负责与麦克风交互、传输音频和 WebSocket 消息，以及处理语音活动检测这类轻量操作。

对于客户端部分，您可以从 `examples\sample_app\static\js\index.js` 中的代码片段开始，并跟踪 `convo` 的使用位置来了解如何使用客户端 API：
```javascript
async function loadXtalk() {
    try {
        return await import("../../xtalk/index.js"); // 优先尝试本地导入，仅用于开发环境
    } catch (e) {
        return await import("https://unpkg.com/xtalk-client@latest/dist/index.js"); // 生产环境使用 unpkg CDN
    }
}

const { createConversation } = await loadXtalk();


function getWebSocketURL() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsPath = new URL("./ws", window.location.href);
    wsPath.protocol = proto;
    wsPath.host = window.location.host;
    return wsPath
}

const convo = createConversation(getWebSocketURL());
```
我们最近将客户端 API 独立发布成了一个单独的包：[xtalk-client](https://www.npmjs.com/package/xtalk-client)。因此，您可以像上面演示的那样，直接从 `https://unpkg.com/xtalk-client@latest/dist/index.js` 导入，而不必自行托管客户端代码。后续我们也会持续改进客户端 API。

对于服务端，核心逻辑是将一个 X-Talk 实例连接到 FastAPI 实例的 WebSocket：
```python
from fastapi import FastAPI, WebSocket
from xtalk import Xtalk
app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config("path/to/config.json")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)
```

之后，您可以查看 `examples/sample_app/configurable_server.py`，了解如何挂载客户端脚本和页面。
