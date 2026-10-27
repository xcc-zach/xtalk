# 引入新模型

完成本教程后，服务会通过您自己注册的模型回答问题。您将为已有的 `Agent` 模型类型添加一个 `EchoAgent`：它读取最终 ASR 文本，并把这段文本原样作为助手回复输出，因此每条回复都会复述您说的话。

开始前请先完成[配置服务](config_the_service.zh.md)，了解 `type` 与 `params` 如何选择模型。

> **Note**
> 完整示例请参阅 [`examples/sample_app/custom_model.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_model.py)。该示例在服务端文件中定义 `EchoAgent`，用 `@model` 注册后，在分阶段配置过程中把已有 `llm_agent` 替换为这个自定义 Agent。

## 1. 导入模型接口和注册装饰器

`EchoAgent` 属于已有模型类型 `Agent`，因此继承 `xtalk.model_types.Agent`。`@model` 把实现类注册到模型 registry 中，配置文件正是据此按类名选择模型。接口细节参考 [Agent API](../api/server/xtalk/models/agents/interfaces.zh.md)。

在 `server.py` 中添加以下 import：

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

    def clone(self) -> "EchoAgent":
        """创建无状态 Agent 的新实例。"""

        return EchoAgent()

```

`async_accept` 是运行时主要使用的异步入口；`accept` 为需要同步迭代器的调用方桥接它。`clone()` 返回供新会话使用的模型实例，避免会话之间共享可变状态。

此时定义类还没有可见效果：配置仍然选择的是 `DefaultAgent`。下一步切换到 `EchoAgent`。

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

重启服务并说一句话。回复是 TTS 读出的您刚才那句话。

如果希望基础配置可被不同服务复用，则保留文件中的 `DefaultAgent`，改为在分阶段配置期间选择已经注册的 Python 类：

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

`clear_agent_params` 去掉了 `EchoAgent` 不接受的 `DefaultAgent` 构造参数。缺少它时，`build()` 会在用多余的关键字参数构造 `EchoAgent` 时失败。

## 4. 在创建 Xtalk 前完成注册

因为 `EchoAgent` 定义在同一个服务端文件中，所以 Python 执行到 `set_model(...)` 前，`@model` 已经完成注册。

```python
xtalk_instance = (
    Xtalk.configure(args.config)
    .transform_config(clear_agent_params)
    .set_model(EchoAgent)
    .build()
)
app = xtalk_instance.create_asgi_app()
```

应用的运行方式见[连接客户端](connect_a_client.zh.md)。

如果模型定义在单独文件中，需要先 import 该文件，使注册发生在 `build()` 之前：

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

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[引入 LLM Agent →](introduce_an_llm_agent.zh.md)** · Tutorial

-   **[支持的模型及其适配位置 →](../reference/supported_models.zh.md)** · Reference

-   **[模型对象的 `clone()` 与 `reset()` →](../reference/model_clone_reset.zh.md)** · Reference

</div>
