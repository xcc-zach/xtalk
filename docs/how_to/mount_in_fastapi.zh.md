# 挂载到 FastAPI

把 X-Talk Realtime API 以路径前缀挂载到已有的 FastAPI 应用中；本文假定您已按[配置服务](../tutorial/config_the_service.zh.md)准备好可用的 `config.json`。

宿主负责安装框架并组合生命周期。挂载的应用需要转发生命周期启动、关闭事件；此例需要安装 `fastapi`、`uvicorn` 和 `asgi-lifespan`。

```python
from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from xtalk import Xtalk

realtime_app = Xtalk.from_config("config.json").create_asgi_app()

@asynccontextmanager
async def lifespan(app):
    async with LifespanManager(realtime_app):
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/speech", realtime_app)
```

连接 `/speech/v1/realtime?model=xtalk`。每条 WebSocket 独立拥有临时对话，断开后释放资源。登录、历史存储、上传及录音文件由宿主应用负责。[`examples/sample_app/host`](https://github.com/xcc-zach/xtalk/tree/main/examples/sample_app/host) 展示了组合方式。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[Realtime API 协议 →](../reference/realtime_api.zh.md)** · Reference

-   **[服务配置项：模型别名与 API key →](../reference/service_config.zh.md)** · Reference

-   **[启动服务 →](../tutorial/start_the_service.zh.md)** · Tutorial

</div>
