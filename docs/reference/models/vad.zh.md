# VAD 接口

```python
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def set_threshold(self, *, threshold: float) -> None: ...

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool: ...

    async def async_is_speech(self, frame: bytes) -> bool: ...
```

`VAD` 判定一帧音频是否为语音。服务层在逐帧模式下调用 `async_is_speech`，在批量模式下调用 `async_detect_frames`；后者默认按顺序调用 `async_is_speech`。轮次级别的说话开始/结束由 `VADManager` 根据连续帧结果推导。

## `async_is_speech`

### 参数

- `frame`：当前输入音频帧，PCM 16-bit、单声道、16 kHz 字节流。

### 返回值

- `bool`：`True` 表示当前帧为语音，`False` 表示非语音。实现内部维护上下文时，返回值仍表示最新完整帧的判定结果。

## `set_threshold`

更新逐帧判定使用的语音概率阈值。

### 参数

- `threshold`：取值范围 `[0, 1]` 的概率阈值。

### 异常

- `NotImplementedError`：不支持调整阈值的模型显式抛出。该方法是抽象方法，所有子类都必须实现。

客户端通过 Realtime 的 `session.update` 事件驱动该方法；参见[会话配置](../realtime_api.zh.md#会话配置)。

## `clone` 与 `reset`

`VAD` 要求实现 `clone`。参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[逐帧结果如何变成轮次事件 →](../../explanation/vad.zh.md)** · Explanation

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[在 `is_speech` 中包装异步检测 →](../../how_to/adapt_async_models.zh.md)** · How-to

</div>
