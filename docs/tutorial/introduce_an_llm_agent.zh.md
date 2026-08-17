# 引入LLM Agent

LLM Agent 接收 ASR、播放状态和工具状态等上下文，并以流式方式输出回复文本或工具调用。本教程以一个回显最终 ASR 文本的 `EchoAgent` 为例，说明如何实现并启用自定义 Agent。

## 实现 Agent

继承 `Agent`，使用 `@model` 注册实现，并实现 `accept()`、`async_accept()`、`restore_history()` 和 `clone()`：

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput, AgentTurnBoundary


@model
class EchoAgent(Agent):
    """回显最终 ASR 文本的 Agent。"""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """同步处理上下文。"""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """异步处理上下文。"""

        if context["type"] != "asr_final":
            return
        text = context["data"]["text"]
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """恢复会话历史；无状态 Agent 可以忽略历史。"""

        del messages

    def clone(self) -> "EchoAgent":
        """为新会话创建 Agent 实例。"""

        return EchoAgent()
```

运行时主要调用 `async_accept()`。`context["type"]` 表示上下文类型，`context["data"]` 包含对应数据。常见类型包括最终识别结果 `asr_final`、增量识别结果 `asr_partial`、多说话人结果 `multi_speaker_final`、回复状态 `response_update` 和 `response_finish`，以及用于持续处理异步更新的 `loop`。

`AgentOutput` 可以是回复文本、工具调用、工具调用结果或 `AgentTurnBoundary`。普通输出流结束时会自动结束本轮回复。由 `loop` 启动的输出流会持续运行，因此每次回复后应输出 `AgentTurnBoundary()`，以触发本轮 TTS flush 和 response finish，同时保持输出流继续运行。

## 处理持续输出

如果 Agent 需要在异步工具状态变化后主动回复，可以在 `loop` 上下文中持续等待更新：

```python
async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] == "asr_final":
        text = context["data"]["text"]
        if text:
            yield text
        return

    if context["type"] == "loop":
        while True:
            text = await self.pending_updates.get()
            yield text
            yield AgentTurnBoundary()
```

`AgentTurnBoundary()` 只结束当前回复，不会结束 `loop` 输出流。队列等会话状态不能由不同会话共享，因此 `clone()` 应返回带有独立状态的新实例。

## 启用 Agent

在 JSON 配置中将 `llm_agent.type` 设置为注册后的类名：

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

也可以在分阶段配置中选择该类：

```python
xtalk_instance = (
    Xtalk.configure("config.json")
    .set_model(EchoAgent)
    .build()
)
```

`set_model()` 会保留原配置中的 `params`，因此这些参数必须与自定义 Agent 的初始化参数兼容。`clone()` 应为每个会话创建可独立使用的实例；需要保留对话历史时，应在 `restore_history()` 中把传入消息恢复到该实例。

接口的完整定义见 [Agent API](../api/server/xtalk/models/agents/interfaces.zh.md)。
