# 工具调用设计

X-Talk 支持三类 Agent 工具：LangChain `BaseTool`、X-Talk 原生同步工具
`SyncTool`，以及可在后台持续产生更新的 `AsyncTool`。`ToolEngine` 负责统一
绑定和调用这些工具，并维护异步调用的生命周期与消息历史协议。

## 设计目标

- 同步工具只返回一次最终结果。
- 异步工具启动后立即返回一条 `Running`，避免阻塞 Agent，并满足
  `AIMessage(tool_calls=...)` 后必须出现匹配 `ToolMessage` 的模型协议。
- 异步工具可以产生过程更新，并最终产生一个 `Finished`。
- LLM 可以查询、订阅、取消订阅或停止异步工具。
- 每个工具调用 ID 在同一个 `ToolEngine` 中唯一。
- 工具结果进入消息历史时始终保持合法的 ToolCall/ToolMessage 配对。

实现位于
[`src/xtalk/models/agents/tools/core.py`](../../src/xtalk/models/agents/tools/core.py)，
公共类型从 `xtalk.models.agents.tools` 导出。

## 核心数据类型

```python
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class ToolInput(BaseModel):
    pass


class ToolOutput(BaseModel):
    def to_content(self) -> str:
        return self.model_dump_json()


@dataclass
class ToolState:
    call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Running:
    content: str


@dataclass(frozen=True)
class Finished:
    content: ToolOutput
```

这些类型的职责如下：

- `ToolInput`：由 LLM 生成并经 Pydantic 校验的结构化输入。
- `ToolOutput`：工具的结构化最终输出；默认以 JSON 写入 ToolMessage。
- `ToolState`：一次异步调用独享的可变状态。
- `Running`：异步工具尚未结束时的文本状态。
- `Finished`：异步工具的最终结构化结果。

`ToolEngineState` 是传给原生工具的会话级共享对象，目前类型为 `Any`。
`ToolEngine` 保留传入对象的引用，不对其进行浅复制或深复制。

## 创建同步工具

同步工具继承 `SyncTool`，并实现带完整类型注解的 `invoke()`：

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
    name = "add"

    @classmethod
    def invoke(
        cls,
        tool_input: AddInput,
        global_state: ToolEngineState,
    ) -> AddOutput:
        del global_state
        return AddOutput(value=tool_input.left + tool_input.right)
```

框架从 `invoke()` 的注解自动推断 `input_type` 和 `output_type`。`ainvoke()`
默认通过 `asyncio.to_thread()` 执行同步实现，避免阻塞事件循环。

没有设置 `name` 时，工具类名会成为对 LLM 暴露的名称。

## 创建异步工具

异步工具继承 `AsyncTool`，至少实现 `emit_initial()` 和
`emit_updates()`。框架从生命周期方法的注解推断输入、状态和输出类型。

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


class SearchInput(ToolInput):
    query: str


@dataclass
class SearchState(ToolState):
    pages_done: int = 0
    subscribed: bool = False
    stopped: bool = False


class SearchOutput(ToolOutput):
    content: str


class SearchTool(AsyncTool):
    name = "search"
    subscribe_by_default = False

    @classmethod
    def emit_initial(
        cls,
        tool_call_id: str,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Running:
        del tool_input, global_state
        tool_state.call_id = tool_call_id
        return Running(content=f"搜索任务已启动，调用编号：{tool_call_id}")

    @classmethod
    def emit_updates(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> Iterator[ToolResult[SearchOutput]]:
        del global_state
        time.sleep(2)  # 模拟耗时检索；默认异步桥接会在线程中执行此方法
        tool_state.pages_done = 1
        yield Running(content="正在检索第一页")
        time.sleep(2)
        yield Finished(
            content=SearchOutput(content=f"搜索完成：{tool_input.query}")
        )

    # 可选 Hook：查询状态、停止任务，以及同步订阅状态。
    @classmethod
    def status(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> str:
        del tool_input, global_state
        return f"已检索 {tool_state.pages_done} 页"

    @classmethod
    def stop(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.stopped = True

    @classmethod
    def subscribe(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = True

    @classmethod
    def unsubscribe(
        cls,
        tool_input: SearchInput,
        tool_state: SearchState,
        global_state: ToolEngineState,
    ) -> None:
        del tool_input, global_state
        tool_state.subscribed = False
```

`emit_initial()` 必须快速返回 `Running`，且当前协议要求 `content` 包含
`tool_call_id`。`emit_updates()` 可以产生多个 `Running`，但必须以一个
`Finished` 结束；未产生 `Finished` 就结束会被视为运行错误。

默认的 `aemit_initial()` 和 `aemit_updates()` 会在线程中桥接同步实现。底层
SDK 原生支持 async 时，可以覆写对应的 `a*` 方法。

### 可选生命周期Hook

`AsyncTool` 还提供下列同步/异步Hook：

- `status()` / `astatus()`：返回当前可读状态。
- `stop()` / `astop()`：终止外部任务并释放资源。
- `subscribe()` / `asubscribe()`：开始订阅过程更新时执行额外逻辑。
- `unsubscribe()` / `aunsubscribe()`：取消订阅时执行额外逻辑。

## 状态与并发约束

`ToolState` 属于单次调用，应优先把进度、外部任务句柄和调用级锁放在这里。
不同调用不会共享同一个 `ToolState`。

`ToolEngineState` 是多个工具共享的会话级对象。同步调用和短生命周期Hook由
ToolEngine 协调，但 ToolEngine **不会** 在整个 `aemit_updates()` 等待期间
持有全局锁。异步更新可能长时间等待网络、队列或外部事件；长期持锁会导致
`status` 和 `stop` 无法执行。

如果 `aemit_updates()` 会修改共享状态，工具实现必须使用短临界区、自己的
`asyncio.Lock`，或线程安全的数据结构。不要把一次网络请求或一次完整更新流
包在共享状态锁内。

## ToolEngine

支持的工具联合类型为：

```python
Tool = BaseTool | type[SyncTool] | type[AsyncTool]
```

创建并绑定工具：

```python
from langchain_core.messages import ToolCall
from xtalk.models.agents.tools import ToolEngine

engine = ToolEngine(
    tools=[AddTool, SearchTool],
    state={},
)
model_with_tools = engine.bind(model)

message = await engine.ainvoke(
    ToolCall(
        id="call-add-1",
        name="add",
        args={"left": 2, "right": 3},
    )
)
```

`bind()` 只负责向模型暴露工具 schema。原生工具必须由 ToolEngine 执行，
不能绕过引擎直接调用绑定包装器。

### 调用语义

- `BaseTool`：调用 LangChain 的 `ainvoke()`，并保存文本化结果。
- `SyncTool`：校验输入、等待最终输出并返回 ToolMessage。
- `AsyncTool`：校验输入、调用 `aemit_initial()`、保存 `AsyncToolRun`、启动
  后台更新任务，然后立即返回初始 ToolMessage。

`invoke()` 只适用于同步上下文中的 `BaseTool` 和 `SyncTool`。已有事件循环时
应使用 `await ainvoke()`；`AsyncTool` 必须使用持久事件循环中的
`await ainvoke()`，否则后台更新无法继续运行。

每个调用 ID 会在执行前预留。并发使用同一个 ID，或重复使用已完成调用的
ID，都会抛出 `ValueError`。

### 异步运行记录

`AsyncToolRun` 保存：

- 最新的 `Running` 或 `Finished`；
- 工具类、输入和调用级状态；
- `subscribed`、`running` 标志；
- 后台 `task`；
- 生命周期锁和后台异常。

后台任务异常会记录到运行对象，并被读取以避免未处理 Task 警告。

## 辅助工具

只要存在一个 `AsyncTool`，ToolEngine 就会注册五个辅助工具：

- `async_tool_updated`：仅供系统写入异步更新，不向 LLM 暴露调用 schema。
- `id_to_async_tool_status`：查询运行状态、工具状态和错误。
- `subscribe_async_tool`：订阅过程更新，并返回最新已知结果。
- `unsubscribe_async_tool`：停止接收过程更新；最终结果仍会发送。
- `stop_async_tool`：调用 `astop()` 并取消后台任务。

后四个控制工具会绑定给模型。`async_tool_updated` 只作为 ToolCall/ToolMessage
协议名称保留，避免模型把系统内部通知误当成可主动调用的工具。
状态查询只用于用户明确要求查看当前状态的场景，不应被模型用于轮询。订阅
成功后，模型应停止调用工具并等待系统主动推送更新。

`subscribe_by_default=True` 会在异步任务启动前调用订阅Hook。未订阅的调用只
主动反馈 `Finished`；已订阅调用会反馈 `Running` 和 `Finished`。

## 消息历史协议

对于每条 ToolMessage，历史中必须存在 ID 匹配的 AI ToolCall。ToolEngine 的
`append_tool_message()` 和 `ainvoke_and_append()` 会维护这一约束，并拒绝：

- 空调用 ID；
- ToolCall 与 ToolMessage ID 不一致；
- 同一 ID 的重复 ToolMessage；
- 同一 ID 对应不同名称或参数。

异步更新使用新的 `async_tool_updated` ToolCall。其参数包含原始
`source_call_id`，消息内容为：

```json
{
  "running": true,
  "tool_output": "正在检索第一页"
}
```

最终更新把 `running` 设为 `false`，并把 `ToolOutput.to_content()` 的结果写入
`tool_output`。

## DefaultAgent 接入

`DefaultAgent` 为每个会话创建独立 ToolEngine，绑定模型，并注册异步
更新回调。没有模型生成时，回调把合法的 ToolCall/ToolMessage 写入历史并
通知会话 loop；模型正在生成时，更新会暂存到当前生成结束后再写入，避免
同一结果被当前工具调用链和后续 loop 重复消费。

当多条工具更新在模型开始生成前连续到达时，Agent 会保留每条下游事件，
但合并成一次模型生成。生成过程中到达的新更新会触发下一次生成。
用户请求使用允许调用工具的模型。后台更新触发主动汇报时仍会绑定完整工具
schema，使模型能够正确理解历史中的系统 ToolCall，但通过 `tool_choice="none"`
禁止模型在该轮主动调用工具。

### 与 ASR partial 并发

ASR partial 会作为 HumanMessage 进入历史。如果工具更新插入到未完成的用户
消息后面，后续 partial：

- 以前一条 partial 为前缀时，只追加新增后缀；
- 不再以前一条 partial 为前缀时，追加完整的新文本；
- final 与最后一条 partial 相同时，不追加空 HumanMessage；
- ASR final 生成结束、异常或取消时，都会复位生成状态。

用户尚未说完时，异步更新只写入历史，不立即触发模型生成。ASR final 到达
后，模型会同时看到工具更新和完整用户输入。

## 关闭

会话关闭时必须调用：

```python
await engine.shutdown()
```

`shutdown()` 会阻止新调用，对仍运行的异步工具执行 `astop()`，取消后台
任务并等待任务退出。重复调用 `shutdown()` 是安全的。

## 当前限制

- 尚不支持 LLM 对运行中的工具提供“二次输入”。
- `ToolEngineState` 的具体并发策略由应用和工具实现共同约定。
- 真实模型对工具描述和主动更新的理解可能不同，应在单元测试之外增加可选的
  端到端集成测试。
