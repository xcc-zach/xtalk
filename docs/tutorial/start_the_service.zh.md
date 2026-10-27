# 启动服务

完成本教程后，您会得到一个可以与 X-Talk 进行语音对话的浏览器页面。您将编写一个只有一个按钮的客户端页面，以及一个同时提供该页面和 X-Talk Realtime API 的 `server.py`。

开始前请先完成[快速开始](../quickstart.zh.md)，确保已安装 X-Talk，并且已有一个配置了可用 ASR、LLM 和 TTS 模型的 `config.json`。

X-Talk 的大多数模型与执行逻辑都运行在服务端。客户端负责访问麦克风、传输音频、收发 WebSocket 消息，以及处理轻量级的会话逻辑。

## 客户端接入

从下面两种加载客户端库的方式中任选一种。两者生成的 `index.html` 相同：页面上有一个“开始对话”按钮，点击后向当前主机的 `/v1/realtime?model=xtalk` 建立会话。

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
          ? `wss://${location.host}/v1/realtime?model=xtalk`
          : `ws://${location.host}/v1/realtime?model=xtalk`;

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
          ? `wss://${location.host}/v1/realtime?model=xtalk`
          : `ws://${location.host}/v1/realtime?model=xtalk`;

      const session = createSession(wsUrl);

      document.querySelector("#start").addEventListener("click", async () => {
        await session.open();
      });
    </script>
  </body>
</html>
```

`createSession(wsUrl)` 使用默认的输入和输出设置。完整配置项见[客户端会话配置](../reference/client_session_config.zh.md)。

## 服务端接入

服务端做两件事：在 `/` 返回 `index.html`，并提供 X-Talk Realtime API。`Xtalk.from_config()` 根据 `config.json` 加载模型，`create_asgi_app()` 把实例变成一个可由 FastAPI 挂载的 ASGI 应用。`LifespanManager` 把 FastAPI 的启动和关闭事件转发给这个被挂载的应用，使模型随宿主一起加载和释放。

安装宿主依赖：

```bash
pip install fastapi uvicorn asgi-lifespan
```

创建 `server.py`：

```python
from pathlib import Path

from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from xtalk import Xtalk

BASE_DIR = Path(__file__).resolve().parent
xtalk_instance = Xtalk.from_config(str(BASE_DIR / "config.json"))
realtime_app = xtalk_instance.create_asgi_app()

@asynccontextmanager
async def lifespan(app):
    async with LifespanManager(realtime_app):
        yield

app = FastAPI(title="Xtalk Server", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """返回前端页面。"""
    return FileResponse(BASE_DIR / "index.html")

app.mount("/", realtime_app)
```

将 `server.py`、`config.json` 和 `index.html` 放在同一目录。

启动服务：

```bash
uvicorn server:app --host 0.0.0.0 --port 11995
```

模型加载完成后，uvicorn 会打印 `Uvicorn running on http://0.0.0.0:11995`。在浏览器中打开 `http://localhost:11995`，页面显示“开始对话”按钮。

点击“开始对话”。首次点击时，浏览器需要从公共 CDN 加载 ONNX Runtime 和 VAD 模型，因此会有几秒钟没有反应。随后浏览器弹出麦克风权限请求；授权后提示消失，对话即已开始。说一句话，回复会通过扬声器播放出来。

## 完整示例

如果您需要包含静态资源、模板、上传界面、会话切换和音色切换的完整示例，可继续参考：

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/css/index.css`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/css/index.css)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[配置服务 →](config_the_service.zh.md)** · Tutorial

-   **[以路径前缀挂载到 FastAPI →](../how_to/mount_in_fastapi.zh.md)** · How-to

-   **[客户端会话配置 →](../reference/client_session_config.zh.md)** · Reference

</div>
