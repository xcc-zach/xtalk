# Realtime API

Xtalk 通过 OpenAI Realtime 风格的 API提供服务，包含标准子集和 Xtalk 扩展。
## 运行 ASGI 应用

自行选择并安装 ASGI 服务，例如 `pip install uvicorn`：

```python
import uvicorn
from xtalk import Xtalk

runtime = Xtalk.from_config("path/to/config.json")
uvicorn.run(runtime.create_asgi_app(), host="127.0.0.1", port=8000)
```

配置文件中可设置API模型与API keys：
```json
{"service_config": {"realtime_api": {"model": "xtalk", "api_keys": []}}}
```
默认模型为 `xtalk`，默认关闭 API Key 鉴权。配置 `api_keys` 后使用 `Authorization: Bearer <key>`。浏览器可使用 `openai-insecure-api-key.<key>` WebSocket 子协议。

服务公开端点为 `GET /v1/models` 和 `WS /v1/realtime?model=xtalk`。

## 挂载到 FastAPI

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

## 最小音频客户端

```sh
pip install websockets numpy soxr soundfile sounddevice
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk
```

无声卡验证可附加 `--input-wav input.wav --no-speaker --rounds 2`。启用鉴权时附加 `--api-key <key>`。示例仅处理 WebSocket 通信和音频输入输出，VAD 与增强由服务端执行。

示例保持单个音频输出流打开，连续填充协议音频块，并在每块最后一个采样实际输出后发送播放确认。打断时清除被停止回复的音频，仅结算尚未确认的已播放时长。

## 交换事件

事件方向、字段、音频格式、播放确认和兼容范围见 [Realtime API 协议参考](../technical_reference/realtime_api.zh.md)。
