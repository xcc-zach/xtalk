X-Talk 的大多数模型与执行逻辑都运行在服务端。客户端主要负责访问麦克风、传输音频、收发 WebSocket 消息，以及处理轻量级的会话逻辑。

## 客户端接入

### npm

使用打包工具时，通过 npm 安装客户端：

```bash
npm install xtalk-client
```

创建 `index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <body>
    <button id="start">开始对话</button>

    <script type="module">
      import { createSession } from "xtalk-client";

      const wsUrl =
        location.protocol === "https:"
          ? `wss://${location.host}/ws`
          : `ws://${location.host}/ws`;

      const session = createSession(wsUrl);

      document.querySelector("#start").addEventListener("click", async () => {
        await session.open();
      });
    </script>
  </body>
</html>
```

### CDN

使用 CDN，创建 `index.html`：

```html
<!doctype html>
<html lang="zh-CN">
  <body>
    <button id="start">开始对话</button>

    <script type="module">
      const { createSession } = await import(
        "https://unpkg.com/xtalk-client@latest/dist/index.js"
      );

      const wsUrl =
        location.protocol === "https:"
          ? `wss://${location.host}/ws`
          : `ws://${location.host}/ws`;

      const session = createSession(wsUrl);

      document.querySelector("#start").addEventListener("click", async () => {
        await session.open();
      });
    </script>
  </body>
</html>
```

## 服务端接入

参考[快速开始](../quickstart.zh.md)安装 X-Talk 并创建 JSON 配置文件。

创建 `server.py`：

```python
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from xtalk import Xtalk

BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config(str(BASE_DIR / "config.json"))
xtalk_instance.mount_routes(app)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """返回前端页面。"""
    return FileResponse(BASE_DIR / "index.html")
```

将 `server.py`、`config.json` 和 `index.html` 放在同一目录。

启动服务：

```bash
uvicorn server:app --host 0.0.0.0 --port 11995
```

启动后访问 `http://localhost:11995`。

首次点击“开始对话”后，浏览器需要从公共 CDN 加载 ONNX Runtime 和 VAD。请等待浏览器弹出麦克风权限请求；弹出并完成授权后，方为对话启动成功。

## 完整示例

如果您需要包含静态资源、模板、上传界面、会话切换和音色切换的完整示例，可继续参考：

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/css/index.css`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/css/index.css)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)
