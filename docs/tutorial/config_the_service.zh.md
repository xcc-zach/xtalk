# 配置服务

完成本教程后，服务会运行在您自己编写的配置之上：您将读懂配置文件结构、替换一个模型、在分阶段配置中添加一个 Python 工具，并开启会话录音。

开始前请先完成[启动服务](start_the_service.zh.md)，确保 `server.py` 能用快速开始中的 `config.json` 运行。

## 配置文件

`Xtalk.from_config(path/to/config.json)` 读取配置文件来实例化 X-Talk。最简配置文件内容如下：

```json
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": {
        "type": "DefaultAgent",
        "params": {
            "model": {
                "api_key": "<API_KEY>",
                "model": "qwen-plus-2025-12-01",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }
    },
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```

键值如 `asr` 代表模型类型，`type` 代表该模型类型选择的模型，`params` 为模型初始化参数。

`params` 直接对应所选类的构造函数。例如，模型类型为 `llm_agent` 的 `DefaultAgent` 定义位于 [`src/xtalk/models/agents/default.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/default.py)：

```python
class DefaultAgent(Agent):
    def __init__(
        self,
        model: BaseChatModel | dict[str, Any],
        backchannel_model: BaseChatModel | dict[str, Any] | None = None,
        backchannel_source_dir: str | Path | None = None,
        tools: list[Tool | Callable[[], Tool]] | None = None,
        system_prompt: str = "",
        proactive: bool = False,
    ) -> None:
        ...
```

要让这个 Agent 使用本地部署的 OpenAI 兼容 LLM，把 `llm_agent` 块替换为：

```json
{
  "llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      }
    }
  }
}
```

除必需的 `model` 外，其他可选键可以忽略。修改 `config.json` 后重启 `uvicorn`，下一次对话就由本地 LLM 回答。

完整的模型类型、对应的可选依赖以及其在源码中的适配位置，请参阅[支持的模型](../reference/supported_models.zh.md)。

> **Note**
> 大多数模型实现都是客户端适配器。您可能还需要按照相应说明启动模型实例本身。

## 分阶段配置

有些设置是 Python 对象，无法写进 JSON。`Xtalk.configure(...)` 返回一个 builder，允许在模型实例化之前修改已加载的配置。下面用它为 Agent 添加一个 Python 工具：

```python
from langchain_core.tools import tool
from xtalk import Xtalk


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数的乘积。"""
    return a * b


def add_multiply_tool(config: dict) -> dict:
    agent_config = config["llm_agent"]
    params = agent_config.setdefault("params", {})
    params.setdefault("tools", []).append(multiply)
    return config


xtalk_instance = (
    Xtalk.configure("path/to/config.json")
    .transform_config(add_multiply_tool)
    .build()
)
```

把 `server.py` 中的 `Xtalk.from_config(...)` 一行替换为这个 builder 并重启。此时问“十二乘七是多少”，回复中的结果由 `multiply` 计算得出。

添加 Agent 工具也可以使用专用方法 `.add_agent_tools([multiply])`。

## 自定义服务行为

模型类型之外的键用于配置服务本身。例如，把每个会话的音频保存到 `logs/` 下：

```json
    "service_config": {
        "recording": true
    }
```

下一次对话结束后，`logs/` 中会出现该会话的录音。

完整的服务配置项列表，请参阅[服务配置项](../reference/service_config.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[连接客户端 →](connect_a_client.zh.md)** · Tutorial

-   **[服务配置项 →](../reference/service_config.zh.md)** · Reference

-   **[支持的模型 →](../reference/supported_models.zh.md)** · Reference

-   **[完全本地部署 →](../how_to/deploy_fully_local.zh.md)** · How-to

</div>
