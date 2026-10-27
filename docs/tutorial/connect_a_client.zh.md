# 连接客户端

完成本教程后，您会用一个终端客户端与 X-Talk 完成一轮语音对话：您将把 Realtime API 作为独立 ASGI 应用运行，并用只依赖 WebSocket 和 PCM 音频的 `examples/sample_client.py` 驱动它。

开始前请先完成[配置服务](config_the_service.zh.md)，确保已有可用的 `config.json`。

X-Talk 通过 OpenAI Realtime 风格的 API 提供服务，包含标准子集和 X-Talk 扩展。[启动服务](start_the_service.zh.md)中的浏览器客户端隐藏了这套协议；本教程直接与它对话。

## 运行 ASGI 应用

`create_asgi_app()` 返回一个普通的 ASGI 应用，因此任何 ASGI 服务器都可以直接运行它，中间无需 Web 框架。安装一个 ASGI 服务器，例如 `pip install uvicorn`，然后创建 `run.py`：

```python
import uvicorn
from xtalk import Xtalk

runtime = Xtalk.from_config("path/to/config.json")
uvicorn.run(runtime.create_asgi_app(), host="127.0.0.1", port=8000)
```

运行 `python run.py`。模型加载完成后，uvicorn 打印 `Uvicorn running on http://127.0.0.1:8000`。

应用提供两个端点：`GET /v1/models` 和 `WS /v1/realtime?model=xtalk`。先验证第一个：

```bash
curl http://127.0.0.1:8000/v1/models
```

响应中列出一个名为 `xtalk` 的模型。

## 设置模型别名与 API key

两者都来自配置文件：

```json
{"service_config": {"realtime_api": {"model": "xtalk", "api_keys": []}}}
```

默认模型为 `xtalk`，默认关闭 API key 鉴权，因此上一篇教程的配置文件无需修改。配置 `api_keys` 后需要携带 `Authorization: Bearer <key>`；浏览器可改用 `openai-insecure-api-key.<key>` WebSocket 子协议。

## 运行最小音频客户端

`examples/sample_client.py` 只处理 WebSocket 通信和音频输入输出；VAD 与增强由服务端执行。

```sh
pip install websockets numpy soxr soundfile sounddevice
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk
```

对着麦克风说话。客户端打印 `conversation.item.input_audio_transcription.completed` 及识别文本，播放回复，并在播放结束后打印 `response.output_audio.playback.done` 及回复文本。

没有声卡的机器上麦克风流无法打开。改为输入 WAV 文件并跳过播放：

```sh
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk \
    --input-wav input.wav --no-speaker --rounds 2
```

客户端在两轮中的每一轮结束后打印 `Completed audio round`，然后退出。配置了 `api_keys` 时附加 `--api-key <key>`；缺少该参数时服务端拒绝连接。

示例保持单个音频输出流打开，连续填充协议音频块，并在每块最后一个采样实际输出后发送播放确认。打断时清除被停止回复的音频，仅结算尚未确认的已播放时长。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[引入新模型 →](introduce_a_new_model.zh.md)** · Tutorial

-   **[Realtime API 协议：事件、字段与音频格式 →](../reference/realtime_api.zh.md)** · Reference

-   **[挂载到 FastAPI →](../how_to/mount_in_fastapi.zh.md)** · How-to

</div>
