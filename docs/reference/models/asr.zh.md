# ASR 接口

```python
class ASR(ABC):
    """Abstract interface for automatic speech recognition."""

    @abstractmethod
    def recognize(self, audio: bytes) -> str: ...

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str: ...

    def stream_chunk_bytes_hint(self) -> int | None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clone(self) -> "ASR": ...

    async def async_recognize(self, audio: bytes) -> str: ...

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str: ...
```

`ASR` 将 PCM 16-bit 单声道音频转换为文本。服务层调用 `async_recognize_stream`；其默认实现在线程池中运行 `recognize_stream`，而默认的 `recognize_stream` 通过 `MockStreamRecognizer` 把 `async_recognize` 适配为流式调用。

## `recognize`

识别一段完整音频。

### 参数

- `audio`：PCM 16-bit 单声道音频字节。

### 返回值

- `str`：识别文本。

## `async_recognize_stream`

增量识别音频。

### 参数

- `audio`：本次调用新到来的音频，PCM 16-bit 单声道字节。
- `is_final`：当前位置是否出现声学停顿，例如一句话结束。该参数只是解码提示；实现可以据此输出缓冲中的尾部音频，但必须保留已累积的识别文本，后续音频在其基础上继续。只有 `reset` 会清空累积文本。
- `chat_history`：实验性字段；当前会话序列化后的聊天历史，无法获取时不含进行中的轮次。

### 返回值

- `str`：从上次 `reset` 到当前时刻累积的完整识别文本。

## `stream_chunk_bytes_hint`

返回期望的流式块大小。

### 返回值

- `int | None`：每累积多少字节音频触发一次流式识别；`None` 表示无偏好，服务层会频繁触发识别。

## `clone` 与 `reset`

`reset` 清空识别状态与累积文本。`clone` 返回共享权重、运行时状态独立的实例。参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[包装异步识别器或离线模型 →](../../how_to/adapt_async_models.zh.md)** · How-to

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[可用的 ASR 实现 →](../supported_models.zh.md)** · Reference

</div>
