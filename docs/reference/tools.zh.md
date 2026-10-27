# 工具

工具是 Agent 暴露给聊天模型的可调用对象。它们通过 `XtalkBuilder.add_agent_tools()` 挂载，并经由 `llm_agent.params.tools` 传入 Agent 构造函数。X-Talk 支持三类工具，并通过 `ToolEngine` 执行它们。

## 内置工具

> **Note**
> 所有内置工具请参阅 [`src/xtalk/models/agents/tools`](https://github.com/xcc-zach/xtalk/tree/main/src/xtalk/models/agents/tools) 下的源码。

内置工具包括 `web_search`、`get_time`，以及静音、语速、音色和情绪等流水线控制工具。`DefaultAgent` 默认注册 `web_search`、`get_time`、`set_speed` 和 `silence`；`set_voice` 与 `set_emotion` 只在对应配置可用时注册。

要启用 `web_search`，需要设置 `SERPER_API_KEY` 或 `GOOGLE_SERPER_API_KEY`。详见 [SerperDev](https://serper.dev/)。

## 工具类型

| 类型 | 说明 |
| --- | --- |
| LangChain `BaseTool` | 已有 LangChain 生态工具或简单函数工具，输入 schema、同步和异步调用能力由具体工具提供。 |
| `SyncTool` | X-Talk 原生同步工具，使用 `ToolInput` 校验输入并返回结构化 `ToolOutput`；框架提供非阻塞的异步桥接。 |
| `AsyncTool` | X-Talk 原生长任务工具，立即返回 `Running`，随后在后台输出进度更新，并以 `Finished` 结束；支持状态查询、停止、订阅和取消订阅。计时器是典型例子：启动后立即返回任务已开始，后台持续报告进度，时间到达后返回最终结果，期间可以查询、订阅或停止。 |

### LangChain 工具

```python
@tool
def name(...) -> ...: ...
```

LangChain 工具使用 `@tool` 定义。完整的函数签名和文档字符串会形成提供给模型的名称、描述和输入 schema。

```python
from langchain_core.tools import tool


@tool
def convert_temperature(value: float, to_unit: str) -> str:
    """在摄氏度和华氏度之间转换温度。

    Args:
        value: 要转换的温度。
        to_unit: 目标单位，只能是 celsius 或 fahrenheit。
    """

    if to_unit == "celsius":
        return str((value - 32) * 5 / 9)
    if to_unit == "fahrenheit":
        return str(value * 9 / 5 + 32)
    raise ValueError("to_unit must be celsius or fahrenheit")


sync_result = convert_temperature.invoke(
    {"value": 32, "to_unit": "celsius"}
)
async_result = await convert_temperature.ainvoke(
    {"value": 0, "to_unit": "fahrenheit"}
)
```

### SyncTool

```python
class SyncTool:
    name: str

    @classmethod
    def invoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput: ...

    @classmethod
    async def ainvoke(cls, tool_input: ToolInput, global_state: ToolEngineState) -> ToolOutput: ...
```

`SyncTool` 通过 `invoke()` 的类型注解推断输入和输出类型。

**参数**

- `tool_input`：`invoke()` 注解所指 `ToolInput` 子类的实例；调用前已完成校验。
- `global_state`：同一个 `ToolEngine` 内多个工具共享的会话级对象。

**返回值**

- `ToolOutput` 子类。`ToolOutput.to_content()` 默认把结果序列化为 JSON，也可以按需覆写。

**说明**

- `name` 省略时使用类名。
- 工具作者必须实现 `invoke()`；继承的 `ainvoke()` 会通过工作线程调用它，避免阻塞事件循环。

```python
from xtalk.models.agents.tools import (
    SyncTool,
    ToolEngineState,
    ToolInput,
    ToolOutput,
)


class AddInput(ToolInput):
    left: int
    right: int


class AddOutput(ToolOutput):
    value: int


class AddTool(SyncTool):
    """计算两个整数之和。"""

    name = "add"

    @classmethod
    def invoke(
        cls,
        tool_input: AddInput,
        global_state: ToolEngineState,
    ) -> AddOutput:
        del global_state
        return AddOutput(value=tool_input.left + tool_input.right)


output = AddTool.invoke(AddInput(left=2, right=3), {})
async_output = await AddTool.ainvoke(AddInput(left=2, right=3), {})
content = output.to_content()
```

### AsyncTool

```python
class AsyncTool:
    name: str
    subscribe_by_default: bool = False

    @classmethod
    def emit_initial(cls, tool_call_id, tool_input, tool_state, global_state) -> Running: ...
    @classmethod
    def emit_updates(cls, tool_input, tool_state, global_state) -> Iterator[ToolResult]: ...
    @classmethod
    def status(cls, tool_input, tool_state, global_state) -> str: ...
    @classmethod
    def stop(cls, tool_input, tool_state, global_state) -> None: ...
    @classmethod
    def subscribe(cls, tool_input, tool_state, global_state) -> None: ...
    @classmethod
    def unsubscribe(cls, tool_input, tool_state, global_state) -> None: ...
```

`AsyncTool` 使用 `ToolState` 保存单次调用状态，并通过生命周期 Hook 管理后台任务。

**必须实现的 Hook**

- `emit_initial()`：返回首个 `Running`，其 `content` 必须包含 `tool_call_id`。
- `emit_updates()`：更新流；可以产生多个 `Running`，正常完成时必须产生一个 `Finished`。

**可选 Hook**

- `status()`、`stop()`、`subscribe()`、`unsubscribe()`。

**异步变体**

- `aemit_initial()`、`aemit_updates()`、`astatus()`、`astop()`、`asubscribe()` 和 `aunsubscribe()` 默认通过工作线程桥接上述同步实现。底层 SDK 原生支持异步时，可以覆写这些 `a*` 方法。
- `subscribe_by_default=True` 会在后台更新任务启动前自动订阅。

```python
from collections.abc import Iterator
from dataclasses import dataclass
import time

from xtalk.models.agents.tools import (
    AsyncTool,
    Finished,
    Running,
    ToolEngineState,
    ToolInput,
    ToolOutput,
    ToolResult,
    ToolState,
)


class TimerInput(ToolInput):
    seconds: int


class TimerOutput(ToolOutput):
    elapsed_seconds: int


@dataclass
class TimerState(ToolState):
    elapsed_seconds: int = 0
    subscribed: bool = False
    stopped: bool = False


class TimerTool(AsyncTool):
    """在后台计时并报告进度。"""

    name = "timer"
    subscribe_by_default = False

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Running:
        del tool_input, global_state
        tool_state.call_id = tool_call_id
        return Running(content=f"计时器已启动，调用编号：{tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[TimerOutput]]:
        del global_state
        for elapsed in range(1, tool_input.seconds + 1):
            if tool_state.stopped:
                return
            time.sleep(1)
            tool_state.elapsed_seconds = elapsed
            if elapsed < tool_input.seconds:
                yield Running(content=f"已计时 {elapsed} 秒")
        yield Finished(
            content=TimerOutput(elapsed_seconds=tool_state.elapsed_seconds)
        )

    @classmethod
    def status(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> str:
        del tool_input, global_state
        return f"已计时 {tool_state.elapsed_seconds} 秒"

    @classmethod
    def stop(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.stopped = True

    @classmethod
    def subscribe(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = True

    @classmethod
    def unsubscribe(
        cls,
        tool_input: TimerInput,
        tool_state: TimerState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = False
```

## ToolEngine

```python
ToolEngine(tools: list[Tool], state: ToolEngineState)
```

`ToolEngine` 负责把三类工具绑定给模型、执行模型返回的 `ToolCall`、维护合法的 `AIMessage`/`ToolMessage` 历史，并管理 `AsyncTool` 的后台生命周期。

**方法**

- `ToolEngine(tools, state)`：创建引擎；工具名在同一引擎中必须唯一。
- `bind(model)`：向模型暴露工具 schema，并允许模型调用。
- `bind_without_tool_calls(model)`：保留 schema，但设置 `tool_choice="none"`。
- `on_async_tool_update(callback)`：接收已订阅的 `Running` 和所有 `Finished` 更新。参见 [AsyncTool](#asynctool)。
- `ainvoke(tool_call)` / `invoke(tool_call)`：执行工具并返回 `ToolMessage`，但不修改历史。
- `ainvoke_and_append(tool_call, messages)` / `invoke_and_append(...)`：执行工具，并用匹配的调用 ID 写入历史。
- `append_tool_message(tool_call, tool_message, messages)`：只维护已有调用及结果的历史配对，不执行工具。
- `shutdown()`：停止新调用，执行异步工具的停止 Hook，并取消后台任务；可重复调用。

**约束**

- 异步上下文应使用 `await ainvoke()` 或 `await ainvoke_and_append()`。`invoke()` 和 `invoke_and_append()` 只用于没有运行中事件循环的同步上下文，且不能用于 `AsyncTool`。
- 只要引擎包含 `AsyncTool`，它还会向模型注册状态查询、订阅、取消订阅和停止工具。异步更新通过 `on_async_tool_update()` 注册的回调主动送回 Agent；未订阅的调用只主动发送最终 `Finished`，已订阅的调用会发送过程中的 `Running` 和最终 `Finished`。

**示例**

下面的 Agent 用到了 `ToolEngine` 的全部公开接口：

```python
import asyncio
from typing import Any, AsyncIterator, Iterable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, ToolCall, ToolMessage

from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput
from xtalk.models.agents.tools import Tool, ToolEngine


class ToolAgent(Agent):
    def __init__(self, model: BaseChatModel, tools: list[Tool]) -> None:
        self.model = model
        self.tools = tools
        self.messages: list[BaseMessage] = []
        self.engine = ToolEngine(tools=tools, state={})
        self.model_with_tools = self.engine.bind(model)
        self.model_without_calls = self.engine.bind_without_tool_calls(model)
        self._async_update_queue: asyncio.Queue[None] = asyncio.Queue()
        self._model_lock = asyncio.Lock()
        self.engine.on_async_tool_update(self._on_async_tool_update)

    def _on_async_tool_update(
        self,
        tool_call: ToolCall,
        tool_message: ToolMessage,
    ) -> None:
        ToolEngine.append_tool_message(tool_call, tool_message, self.messages)
        self._async_update_queue.put_nowait(None)

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        yield from self.sync_iter_from_async(self.async_accept(context))

    async def _invoke_model(self, *, allow_tools: bool):
        model = (
            self.model_with_tools
            if allow_tools
            else self.model_without_calls
        )
        async with self._model_lock:
            response = await model.ainvoke(list(self.messages))
            self.messages.append(response)
            return response

    async def _report_async_tool_updates(
        self,
    ) -> AsyncIterator[AgentOutput]:
        while True:
            await self._async_update_queue.get()
            response = await self._invoke_model(allow_tools=False)
            yield str(response.content)

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        if context["type"] == "loop":
            async for output in self._report_async_tool_updates():
                yield output
            return

        if context["type"] != "asr_final":
            return

        self.messages.append(HumanMessage(content=context["data"]["text"]))
        response = await self._invoke_model(allow_tools=True)

        for tool_call in response.tool_calls:
            yield tool_call
            tool_message = await self.engine.ainvoke_and_append(
                tool_call,
                self.messages,
            )
            yield str(tool_message.content)

    def clone(self) -> "ToolAgent":
        return ToolAgent(self.model, self.tools)

    async def shutdown(self) -> None:
        await self.engine.shutdown()
```

异步更新回调首先把合法的调用/结果消息对写入历史，再向 `_async_update_queue` 放入通知。长期运行的 `loop` 上下文持续等待该队列，每收到一次通知便调用 `_report_async_tool_updates()`。生成的文本以 `AgentOutput` 形式 `yield`，由此把主动汇报传回服务的输出管线。

`async_accept()` 处理用户请求时传入 `allow_tools=True`，使用 `bind()` 返回的模型，允许产生新工具调用。`_report_async_tool_updates()` 传入 `allow_tools=False`，使用 `bind_without_tool_calls()` 返回的模型；模型仍能理解历史中的工具 schema，但该轮不会继续调用工具。`_model_lock` 保证正常回复和主动汇报的生成及历史写入不会并发，`list(self.messages)` 则为每次模型调用提供稳定的历史快照。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[挂载工具 →](../tutorial/attach_tools.zh.md)** · Tutorial

-   **[引入 LLM Agent →](../tutorial/introduce_an_llm_agent.zh.md)** · Tutorial

-   **[生成的 tools 模块 API →](../api/server/xtalk/models/agents/tools/index.zh.md)** · Reference

</div>
