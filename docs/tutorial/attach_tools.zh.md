# 挂载工具

完成本教程后，Agent 会调用您编写的工具：一个统计字符数的 LangChain 工具，通过 builder 挂载，并在对话中由 Agent 执行。随后您会用工厂让工具按会话隔离，并了解自定义 Agent 如何接收这些工具。

开始前请先完成[引入 LLM Agent](introduce_an_llm_agent.zh.md)。内置的 `DefaultAgent` 已经能执行工具，因此前两步可以直接使用快速开始中的配置；自定义 Agent 一步基于下面的 `ToolAgent` 骨架。

## 1. 定义并挂载工具

定义一个 LangChain 工具，再通过 `XtalkBuilder.add_agent_tools()` 挂载到配置中的 Agent：

```python
from langchain_core.tools import tool

from xtalk import Xtalk


@tool
def text_length(text: str) -> int:
    """返回文本包含的字符数。"""

    return len(text)


xtalk_instance = (
    Xtalk.configure("config.json")
    .add_agent_tools([text_length])
    .build()
)
```

重启服务并问“hello 这个词有几个字符”。回复中包含由 `text_length` 计算出的 `5`。

`add_agent_tools()` 支持 LangChain 工具实例、X-Talk 原生 `SyncTool` 或 `AsyncTool` 类，以及返回这些工具的无参数工厂。多次调用会按顺序追加工具，不会修改传入的原始配置字典。其他工具类型见[工具](../reference/tools.zh.md)。

## 2. 在自定义 Agent 中接收工具

工具最终通过 `llm_agent.params.tools` 传给 Agent 的构造函数，因此自定义 Agent 需要自行接收并使用 `tools` 参数。在此之前，挂载的工具会被该 Agent 静默忽略。

自定义 Agent 可以在构造函数中创建 `ToolEngine`，再把工具绑定给聊天模型：

```python
from langchain_core.language_models.chat_models import BaseChatModel

from xtalk.model_types import Agent
from xtalk.models.agents.tools import Tool, ToolEngine


class ToolAgent(Agent):
    def __init__(
        self,
        model: BaseChatModel,
        tools: list[Tool] | None = None,
    ) -> None:
        self.model = model
        self.tools = list(tools or [])
        self.tool_engine = ToolEngine(tools=self.tools, state={})
        self.model_with_tools = self.tool_engine.bind(model)
```

后续调用 `self.model_with_tools` 时，模型可以生成工具调用；Agent 再通过 `self.tool_engine` 执行它们。执行这些调用并把结果写入历史的完整 Agent 见 [ToolEngine](../reference/tools.zh.md#toolengine)。

## 3. 让每个会话拥有独立的工具实例

如果每个会话都需要独立的工具实例，请传入工具工厂：

```python
from langchain_core.tools import tool


def create_counter_tool():
    count = 0

    @tool
    def increment_counter() -> int:
        """增加并返回当前会话的计数。"""

        nonlocal count
        count += 1
        return count

    return increment_counter


xtalk_instance = (
    Xtalk.configure("config.json")
    .add_agent_tools([create_counter_tool])
    .build()
)
```

每次创建会话 Agent 时都会调用 `create_counter_tool()`，因此不同会话不会共享 `count`。打开两个浏览器标签页，分别让它们增加计数：两者都回答 `1`。工具工厂必须是不接收参数的可调用对象。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[定制服务 →](customize_the_service.zh.md)** · Tutorial

-   **[工具：内置工具、`SyncTool`、`AsyncTool` 与 `ToolEngine` →](../reference/tools.zh.md)** · Reference

</div>
