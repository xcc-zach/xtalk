> **Note**
> 示例请参阅 `examples/sample_app/custom_model.py`。该示例在服务端文件中定义了一个 `EchoAgent`，用 `@model` 注册后，通过配置把默认 `llm_agent` 替换为这个自定义 Agent。

您可能希望为已有模型类型引入一个新模型，例如新的文本转语音模型。下面以 `custom_model.py` 为例，说明如何一步步引入一个新的 `EchoAgent`。这个 Agent 会读取最终 ASR 文本，然后把这段文本原样作为助手回复输出。

## 1. 导入模型接口和注册装饰器

`EchoAgent` 属于已有模型类型 `Agent`，因此需要继承 `xtalk.model_types.Agent`。同时用 `@model` 将实现类注册到模型 registry 中。模型接口细节参考[ASR设计](../docs/asr_design.zh.md)等文档。

```python
import asyncio
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
```

## 2. 定义并注册模型实现

核心是给类加上 `@model`。默认情况下，配置文件中的 `type` 可以直接使用类名 `EchoAgent`。

```python
@model
class EchoAgent(Agent):
    """回显最终 ASR 文本的简单 Agent。"""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """兼容同步接口，内部桥接到 async_accept。"""

        yield from self._sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        if str(context.get("type", "") or "") != "asr_final":
            return
        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", ""))
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        del messages

    def clone(self) -> "EchoAgent":
        return EchoAgent()

    def _sync_iter_from_async(
        self,
        async_iter: AsyncIterator[AgentOutput],
    ) -> Iterable[AgentOutput]:
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    item = loop.run_until_complete(async_iter.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        finally:
            loop.close()
```

这里有几点需要注意：

- `@model` 必须在调用 `Xtalk.from_config(...)` 前执行，也就是定义该类的模块必须先被 import。
- `async_accept` 是运行时主要使用的异步入口。
- `accept` 桥接到 `async_accept`，用于兼容同步调用路径。
- `clone()` 要返回新会话可用的模型实例，避免会话之间共享可变状态。

## 3. 让配置使用新模型

配置里把 `llm_agent`的 `type` 改成 `EchoAgent`。

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

## 4. 在创建 Xtalk 前完成注册

因为 `EchoAgent` 定义在同一个服务端文件中，所以 Python 执行到 `Xtalk.from_config(...)` 前，`@model` 已经完成注册。

```python
with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)

xtalk_instance = Xtalk.from_config("path/to/config.json")
xtalk_instance.mount_routes(app)
```

如果模型定义在单独文件中，需要先 import 该文件：

```python
import my_app.echo_agent

xtalk_instance = Xtalk.from_config("path/to/config.json")
```

## 5. 完整配置示例

下面的配置会继续使用原来的 ASR 和 TTS，只把 Agent 替换为 `EchoAgent`：

```json
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    },
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```