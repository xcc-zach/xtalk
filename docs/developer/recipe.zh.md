# Recipe

以下几个例子说明如何通过直接修改该框架开发：

## 引入新的ASR模型

假设要引入Qwen3ASRFlashRealtime，目前实现已经在`src/xtalk/models/asr/qwen3_asr_flash_realtime.py`。

1. 在`src/xtalk/models/asr`下创建`qwen3_asr_flash_realtime.py`
2. 准备骨架，并实现对应方法（各类模型接口参考`src/xtalk/models/*/interfaces.py`，模型接口细节参考[ASR设计](../docs/asr_design.zh.md)等文档）
```python
from xtalk import model

from ..interfaces import ASR

@model
class Qwen3ASRFlashRealtime(ASR):
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        config: Optional[Qwen3ASRFlashConfig] = None,
    ) -> None:
        ...
    def recognize(self, audio: bytes) -> str:
        ...
    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        ...
    def stream_chunk_bytes_hint(self) -> int | None:
        ...
    def reset(self) -> None:
        ...
    def clone(self) -> "ASR":
        ...
    async def async_recognize(self, audio: bytes) -> str:
        ...
    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        ...
```
3. 用 `@model` 装饰实现类，使其可以从配置中被发现
4. 配置中使用
```json
"asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "your key"
        }
    }
```

## 引入新的Agent

参考`src/xtalk/models/agents/experimental.py`；实现与配置方法类似`引入新的ASR模型`章节：继承接口，并用 `@model` 装饰实现类。

### accept的逻辑

```python
async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]:
    pass
```

`accept`方法订阅外部输入并启动相关处理逻辑；`AgentContext`来自`src/xtalk/serving/modules/llm_agent_context_manager.py`，目前较为稳定的类型有`asr_partial`、`asr_final`、`loop`。其中`loop`在连接建立时触发一次，可以用于处理任何主动触发逻辑，或者是启动输出循环。`src/xtalk/models/agents/experimental.py`用于触发主动对话。

`AgentOutput`为字符串、工具调用或工具调用结果；工具调用返回后可用于`Manager`触发相关逻辑，例如`src/xtalk/serving/modules/llm_agent_context_manager.py`中`direct_audio`的工具调用触发下游`src/xtalk/serving/modules/direct_audio_manager.py`生成直接播放音频的事件。

`Agent`从设计哲学上期望成为整个系统的思考核心，负责整合其他块的信息输出。

## 引入新的Manager

上个章节的`Agent`要求引入一个新的`src/xtalk/serving/modules/direct_audio_manager.py`转发工具调用的输出到音频事件。所有`Manager`直接在`src/xtalk/serving/modules`下创建即可，之后要在`src/xtalk/serving/service.py`与`src/xtalk/serving/module_types.py`中注册。`Manager`采用观察者模式进行事件订阅与发布。所有事件在`src/xtalk/serving/events.py`。`src/xtalk/serving/modules/input_gateway.py`和`src/xtalk/serving/modules/output_gateway.py`较为特殊，负责接收前端的输入和向前端输出。

在`Manager`中调用模型可参考`src/xtalk/serving/modules/asr_manager.py`，使用`pipeline.get_asr_model`等方法即可。

注意事件发布`publish`方法中`wait_for_completion`代表`await`时是否等待该事件直接触发的监听函数处理完成。沿事件链条启用`wait_for_completion`可以保证该链条上所有事件处理完毕后才回到事件发送源头继续执行。

## 引入新的类型的模型

创建`src/xtalk/models/<model_type>`文件夹，在`src/xtalk/models/<model_type>/interfaces.py`中创建接口，并用从 `xtalk` 导入的 `@model_type(aliases=[...])` 装饰它，然后在同一文件夹下创建对应模型文件即可。文件夹名会成为主配置 key，alias 用于保留为配置键设置别名。
