# 引入LLM Agent

*实验中的功能*

LLM Agent 接收语音识别、回复播放和工具状态等上下文，并以流式方式输出文本或工具调用。本教程从一个只回显最终识别文本的 Agent 开始，再逐步增加会话状态、工具调用和持续输出能力。

## 1. 输出第一条回复

```python
async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] == "asr_final":
        yield context["data"]["text"]
```

这个例子只处理最终识别结果 `asr_final`，并将识别文本原样作为回复输出。

运行时主要调用 `async_accept(context)`。其中 `context` 是一个 `AgentContext`，包含：

- `context["type"]`：上下文类型；
- `context["data"]`：由对应服务事件携带的数据，不包含 `session_id` 等事件基础字段。

本教程介绍以下 `context["type"]`：

| 类型 | `data` 中的主要字段 | 用途 |
| --- | --- | --- |
| `asr_partial` | `text`、`display_text`、`speech_pause`、`origin`、`turn_id`、`segment_id`、`gate_state` | 增量语音识别文本 |
| `asr_final` | `text`、`display_text`、`origin`、`turn_id`、`segment_id`、`gate_state` | 最终语音识别或文本输入 |
| `response_update` | `response_id`、`text` | 已确认播放的部分助手回复 |
| `response_finish` | `response_id`、`text` | 已确认播放完成或中断的助手回复 |
| `loop` | 无 | 用于主动回复和异步更新的持续输出流 |

自定义 Agent 可以忽略不需要的类型。`loop` 是唯一的持久流；其他上下文产生的输出流结束后，本轮回复会自动结束。

例如，下面的 `loop` 会持续等待队列中的主动回复；每次输出后用 `AgentTurnBoundary()` 结束本轮，但继续等待下一条回复：

```python
import asyncio


def __init__(self) -> None:
    self.pending_updates: asyncio.Queue[str] = asyncio.Queue()

async def async_accept(
    self,
    context: AgentContext,
) -> AsyncIterator[AgentOutput]:
    if context["type"] != "loop":
        return

    while True:
        text = await self.pending_updates.get()
        yield text
        yield AgentTurnBoundary()
```

`pending_updates` 不是 `Agent` 基类提供的字段，而是这个自定义 Agent 在 `__init__()` 中创建的会话级异步队列。其他异步任务可以向该队列写入文本，由 `loop` 依次输出。

`async_accept()` 返回异步的 `AgentOutput` 流。每一项可以是：

- `str`：交给 TTS 和前端的回复文本；
- `ToolCall`：包含工具名称、参数和调用 ID 的工具调用请求。服务层不会因为 Agent 输出了普通 `ToolCall` 就自动执行任意工具。工具调用应在 Agent 内执行。目前 `ToolCall` 主要用于告诉服务层“发生了一个工具调用”；
- `ToolCallResult`：包含原工具名称、参数和结果文本的完成通知，用于告诉服务层“工具已经返回结果”；
- `AgentTurnBoundary`：结束持久流中的当前回复，触发 TTS flush 和 response finish，但不结束输出流本身。简单理解就是在 `loop` 中，每次完整回复输出结束后都需要 `yield AgentTurnBoundary()`。

例如，下面两个文本分片属于同一条回复，只在完整回复输出结束后添加一次边界：

```python
yield "今天天气"
yield "晴朗，适合出行。"
yield AgentTurnBoundary()
```

## 2. 补全 Agent 接口

自定义 Agent 需要继承 `Agent`，并实现 `accept()` 和 `clone()`。如果主要逻辑写在 `async_accept()` 中，`accept()` 可以使用继承的 `sync_iter_from_async()`。`@model` 用于将该实现注册为配置文件可以选择的模型：

```python
from typing import Any, AsyncIterator, Iterable

from xtalk import model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput


@model
class EchoAgent(Agent):
    """回显最终 ASR 文本的 Agent。"""

    def __init__(self) -> None:
        """初始化对话历史。"""

        self.messages: list[dict[str, Any]] = []

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """同步处理上下文。"""

        yield from self.sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """异步处理上下文。"""

        if context["type"] == "asr_final":
            yield context["data"]["text"]

    def clone(self) -> "EchoAgent":
        """为新会话创建 Agent 实例。"""

        return EchoAgent()

```

`clone()` 必须返回可供新会话独立使用的实例，避免不同会话共享可变状态。如果应用需要恢复对话，可以在会话中将之前的消息作为上下文发送。

## 3. 注册并启用 Agent

把配置中的 `llm_agent.type` 改为注册类名：

```json
{
    "llm_agent": {
        "type": "EchoAgent",
        "params": {}
    }
}
```

随后可以创建服务：

```python
from xtalk import Xtalk


xtalk_instance = Xtalk.from_config("config.json")
```

也可以在分阶段配置中选择该类：

```python
xtalk_instance = (
    Xtalk.configure("config.json")
    .set_model(EchoAgent)
    .build()
)
```

## 4. 处理更多上下文

了解这些上下文后，可以在 `async_accept()` 中按类型分支。最终输入的处理已经在前文展示；如果还需要使用增量输入，可以增加以下分支：

```python
if context["type"] == "asr_partial":
    self.partial_text = context["data"]["text"]
```

## 5. 管理对话历史

Agent 自己负责维护对话状态。如果还需要向 ASR 或其他组件提供文本形式的历史，可以实现 `get_chat_history()`：

```python
def get_chat_history(self, with_system: bool = False) -> str | None:
    messages = self.messages if with_system else [
        message for message in self.messages if message["role"] != "system"
    ]
    return "\n".join(
        f'{message["role"]}: {message["content"]}' for message in messages
    )
```

`get_chat_history()` 是可选接口。每轮识别开始时，ASR 会调用它获取当前对话历史，并将返回值作为 `chat_history` 传给 ASR 模型；不使用历史的 ASR 模型会忽略该值。`DefaultAgent` 还会处理 `response_update` 和 `response_finish`，以记录实际播放给用户的内容；自定义 Agent 只有需要这种语义时才需要处理它们。

## 6. 更新对话历史与系统提示

自定义 Agent 可以实现 `append_context()` 和 `update_instructions()`，以支持客户端通过 WebSocket 更新当前会话。

客户端发送 `conversation.item.create` 追加上下文消息时，若未要求生成回复，会调用 `append_context()`。该方法需要将消息加入 Agent 的对话历史，供后续回复使用，也可用于逐条恢复已有对话。消息角色支持 `user`、`assistant` 和 `system`：

```python
def append_context(self, *, role: str, content: str) -> None:
    """追加一条上下文消息。"""
    self.messages.append({"role": role, "content": content})
```

客户端发送包含 `instructions` 的 `session.update` 时，会调用 `update_instructions()`。该方法需要替换当前系统提示，并保留已有对话消息：

```python
def update_instructions(self, *, instructions: str) -> None:
    """更新后续回复使用的系统提示。"""
    self.instructions = instructions
```

上例假设 Agent 在初始化时创建了 `self.messages` 和 `self.instructions`，并在生成回复时使用它们。请根据自己的消息存储方式调整实现。通过 `append_context()` 追加 `system` 消息会在历史末尾增加一条消息；`update_instructions()` 则替换当前系统提示。未实现对应接口时，客户端的上下文追加或系统提示更新请求会被拒绝。

## 7. 工具调用

详见[引入工具](tool_design.zh.md)。

## 8. 使用辅助能力

`Agent` 提供 `content_to_text()`，用于把 LangChain 返回的字符串或内容块转换为纯文本。
