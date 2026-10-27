# 定制服务

*实验中的功能*

完成本教程后，前端收到的回复文本会经过您自己的代码改写。您将向服务添加一个新类型的模型、定义一个携带其输出的事件、编写一个在每条回复结束时触发的 `Manager`，并重新安排由哪个处理函数把结束事件发给前端。最终效果是每条最终回复文本前都加上 `Assistant response: `。

开始前请先完成[挂载工具](attach_tools.zh.md)，熟悉分阶段配置和 `Agent` 模型类型。

> **Note**
> 完整示例请参阅 [`examples/sample_app/custom_service.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_service.py)。其中向 X-Talk 添加了一个哑的 `LLMOutputRefactorModel`，用于在发送到前端的最终 LLM 响应文本前附加 `Assistant response: `。

## 引入新类型模型

引入新类型模型有两种常见做法：

1. 直接在服务代码里定义模型，并手动放入 `Models`。[`examples/sample_app/custom_service.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_service.py) 使用的是这种方式。
2. 先定义新的模型类型，再注册具体模型实现，让模型可以通过配置文件创建。

### 做法一：直接引入新模型

这种做法适合服务内辅助模型：模型只服务于当前自定义服务，不需要作为配置文件中的通用模型类型暴露。

#### 定义自定义模型

示例中的 `LLMOutputRefactorModel` 会在最终 LLM 回复前添加一段前缀：

```python
class LLMOutputRefactorModel:
    def refactor(self, llm_output: str) -> str:
        return "Assistant response: " + llm_output

    def clone(self):
        return LLMOutputRefactorModel()
```

如果模型内部有需要按会话隔离的状态，应实现 `clone`，让每个会话使用独立的模型实例。

#### 创建 Models 并注入自定义模型

内建模型仍然从配置文件创建：

```python
from xtalk import Xtalk


models = Xtalk.create_models_from_config(config_path_or_dict=args.config)
```

随后把自定义模型放入同一个 `Models` 容器：

```python
models.set(LLMOutputRefactorModel, LLMOutputRefactorModel())
```

此时 `models` 中包含配置的 ASR、Agent、TTS 以及您的模型。还没有任何代码调用 `refactor()`；下文的 Manager 会调用它。

### 做法二：引入新的模型类型，再引入新模型

如果希望模型可以通过配置文件创建，就需要先定义模型类型接口，并用 `@model_type` 注册；再定义具体模型实现，并用 `@model` 注册。

模型类型和模型实现所在模块必须在调用 `Xtalk.create_models_from_config(...)` 之前被 import。如果它们直接写在同一个服务文件中，只要定义位置在创建 `Models` 之前即可。

```python
from abc import ABC, abstractmethod

from xtalk import model, model_type


@model_type(aliases=["llm_output_refactor_model"])
class LLMOutputRefactor(ABC):
    @abstractmethod
    def refactor(self, llm_output: str) -> str:
        pass

    @abstractmethod
    def clone(self) -> "LLMOutputRefactor":
        pass


@model
class PrefixLLMOutputRefactor(LLMOutputRefactor):
    def __init__(self, prefix: str = "Assistant response: "):
        self.prefix = prefix

    def refactor(self, llm_output: str) -> str:
        return self.prefix + llm_output

    def clone(self):
        return PrefixLLMOutputRefactor(prefix=self.prefix)
```

`clone()` 不是 `@model_type` 注册本身要求的字段，但服务为每个会话克隆 `Models` 时，会自动调用模型实例上的 `clone()`。因此，如果这个模型类型的实现不应该跨会话共享状态，建议把 `clone()` 写进接口契约。

随后在配置文件中以上面声明的别名添加这个模型：

```json
{
  "llm_output_refactor_model": {
    "type": "PrefixLLMOutputRefactor",
    "params": {
      "prefix": "Assistant response: "
    }
  }
}
```

采用这种做法时，创建 `Models` 不再需要手动 `set` 这个模型：

```python
models = Xtalk.create_models_from_config(config_path_or_dict=args.config)
```

## 创建自定义事件

改写后的文本需要传给前端输出组件，因此示例创建了一个新的事件类型：

```python
from xtalk import create_event_class


LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal",
    fields={"response_id": "", "text": ""},
)
```

该事件携带原回复的 `response_id` 和改写后的 `text`，使前端能够把结束事件关联到正确的回复。

## 定义自定义 Manager

`LLMOutputRefactorManager` 监听 TTS 播放完成后发布的 `ResponseFinish`，取出自定义模型，改写文本后发布 `LLMOutputRefactoredFinal`。

下面的代码继续沿用做法一中的 `LLMOutputRefactorModel`。如果采用做法二，应把 `self.models.get(LLMOutputRefactorModel)` 换成 `self.models.get(LLMOutputRefactor)`。

```python
from typing import Any

from xtalk import EventBus, Manager, Models
from xtalk.events import ResponseFinish


class LLMOutputRefactorManager(Manager):
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any],
    ):
        self.event_bus = event_bus
        self.models = models

    @Manager.event_handler(ResponseFinish)
    async def handle_response_finish(self, event: ResponseFinish):
        refactor_model = self.models.get(LLMOutputRefactorModel)
        if refactor_model:
            refactored_output = refactor_model.refactor(event.text)
            new_event = LLMOutputRefactoredFinal(
                session_id=event.session_id,
                response_id=event.response_id,
                text=refactored_output,
            )
            await self.event_bus.publish(new_event)

    async def shutdown(self):
        pass
```

构造函数中的 `models` 由服务在创建会话时传入。示例使用 `models.get(...)`，因此模型不存在时会跳过改写；如果你的自定义逻辑必须依赖该模型，可以改为直接要求模型存在，让配置错误尽早暴露。

单独定义这个类还没有任何效果：只有在下一步把 Manager 注册到服务之后，`handle_response_finish` 才会运行。

## 注册 Manager

创建服务时直接传入 `models`，然后注册自定义 `Manager`：

```python
from xtalk import DefaultService


custom_service = DefaultService(models=models)
custom_service.register_manager(LLMOutputRefactorManager)
```

这样每个会话都会创建一个 `LLMOutputRefactorManager`，并使用该会话克隆出的模型容器。从现在起，每个 `ResponseFinish` 之后事件总线上都会出现一个 `LLMOutputRefactoredFinal`，但还没有组件消费这个新事件，前端看到的仍是原始文本。

## 调整已有事件订阅

默认情况下，`OutputGateway` 会处理原始的 `ResponseFinish`。示例希望前端收到改写后的文本，因此先取消旧订阅：

```python
from xtalk.serving.module_types import OutputGateway


custom_service.unsubscribe_event(
    event_listener_cls=OutputGateway,
    event_type=ResponseFinish,
)
```

在这一步和下一步之间，前端不会收到已结束回复的 `response.output_audio.playback.done`。下面的注册用新事件把它恢复：

```python
async def output_gateway_llm_output_refactored_final_handler(
    self: OutputGateway,
    event,
):
    await self.send_signal(
        {
            "type": "response.output_audio.playback.done",
            "response_id": event.response_id, "text": event.text,
        }
    )


custom_service.subscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMOutputRefactoredFinal,
    method_or_handler=output_gateway_llm_output_refactored_final_handler,
)
```

## 启动服务

最后把自定义服务传给 `Xtalk`，创建 ASGI 应用：

```python
xtalk_instance = Xtalk(service_prototype=custom_service)

app = xtalk_instance.create_asgi_app()
```

按[连接客户端](connect_a_client.zh.md)的方式运行。每条回复播放完毕后，客户端收到的 `response.output_audio.playback.done` 事件中，`text` 以 `Assistant response: ` 开头。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[Bot2Bot →](bot2bot.zh.md)** · Tutorial

-   **[整体架构：事件、Manager 与网关 →](../explanation/system_design.zh.md)** · Explanation

-   **[模型对象的 `clone()` 与 `reset()` →](../reference/model_clone_reset.zh.md)** · Reference

</div>
