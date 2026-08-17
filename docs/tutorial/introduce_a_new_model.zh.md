> **Note**
> 示例请参阅 [`examples/sample_app/custom_model.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_model.py)。该示例在服务端文件中定义了一个 `EchoAgent`，用 `@model` 注册后，在分阶段配置过程中把已有 `llm_agent` 替换为这个自定义 Agent。

您可能希望为已有模型类型引入一个新模型，例如新的 LLM Agent。下面以 `custom_model.py` 为例，说明如何一步步引入一个新的 `EchoAgent`。这个 Agent 会读取最终 ASR 文本，然后把这段文本原样作为助手回复输出。

## 1. 导入模型接口和注册装饰器

`EchoAgent` 属于已有模型类型 `Agent`，因此需要继承 `xtalk.model_types.Agent`。同时用 `@model` 将实现类注册到模型 registry 中。接口细节参考 [Agent API](../api/server/xtalk/models/agents/interfaces.zh.md)。

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
```

## 2. 定义并注册模型实现

```python
@model
class EchoAgent(Agent):
    """回显最终 ASR 文本的简单 Agent。"""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """兼容同步接口，内部桥接到 async_accept。"""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """输出最终 ASR 文本。"""

        if context["type"] != "asr_final":
            return
        text = context["data"]["text"]
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """忽略持久化的历史消息。"""

        del messages

    def clone(self) -> "EchoAgent":
        """创建无状态 Agent 的新实例。"""

        return EchoAgent()

```

这里有几点需要注意：

- `async_accept` 是运行时主要使用的异步入口。
- `clone()` 要返回新会话可用的模型实例，避免会话之间共享可变状态。
- LLM Agent 完整开发教程见[引入LLM Agent](introduce_an_llm_agent.zh.md)。

## 3. 选择新模型

把 `llm_agent` 的 `type` 改为 `EchoAgent`，并继续使用 `Xtalk.from_config(...)`：

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

如果希望基础配置可被不同服务复用，则在分阶段配置期间选择已经注册的 Python 类：

```python
def clear_agent_params(config: dict[str, Any]) -> dict[str, Any]:
    agent_config = config.get("llm_agent")
    if isinstance(agent_config, dict):
        agent_config["params"] = {}
    return config


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
```

## 4. 在创建 Xtalk 前完成注册

因为 `EchoAgent` 定义在同一个服务端文件中，所以 Python 执行到 `set_model(...)` 前，
`@model` 已经完成注册。

```python
xtalk_instance = (
    Xtalk.configure(args.config)
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
xtalk_instance.mount_routes(app)
```

如果模型定义在单独文件中，需要先 import 该文件：

```python
from my_app.echo_agent import EchoAgent

xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
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
