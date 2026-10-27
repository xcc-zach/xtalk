# 语音增强接口

```python
class SpeechEnhancer(ABC):
    """语音增强引擎抽象基类。"""

    @abstractmethod
    def enhance(self, audio: bytes, far: bytes) -> bytes: ...

    def flush(self) -> bytes:
        return b""

    async def async_enhance(self, audio: bytes, far: bytes) -> bytes: ...

    async def async_flush(self) -> bytes: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def clone(self) -> "SpeechEnhancer": ...
```

`SpeechEnhancer` 处理近端麦克风音频，可结合远端参考做声学回声消除。服务层调用 `async_enhance`；其默认实现在线程池中运行 `enhance`。

## 音频格式

输入和输出均为 PCM 16-bit、单声道、16000 Hz 的裸字节流，不含 WAV 头。

## `async_enhance`

### 参数

- `audio`：近端麦克风音频。
- `far`：远端参考音频，来自正在播放给用户的 TTS 音频。服务层保证 `len(far) == len(audio)`、音频格式相同、对应同一时间窗口；实现无需重复检查。没有播放时传入等长静音。不需要参考的实现忽略该参数，收到静音 `far` 时仍正常返回。

### 返回值

- `bytes`：增强后的音频。输出长度与输入 `audio` 保持一致，避免下游 VAD/ASR 时间轴漂移。

## `async_flush`

排出增强器内部缓冲的尾部音频。服务层在 `VADSpeechEnd` 后插入 flush barrier，确保更早的帧先完成增强和下游分发。

### 返回值

- `bytes`：剩余的增强音频，或 `b""`。

## `reset` 与 `clone`

`reset` 清空当前会话的流式状态：模型缓存、输入输出缓冲、远程连接和 pending audio。权重保持加载。

`clone` 为新会话创建独立的运行时实例。多个克隆可以共享权重、配置和只读资源；不共享流式缓冲、远程连接、pending audio 或会话状态。

参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[远端参考如何生成 →](../../explanation/speech_enhancer.zh.md)** · Explanation

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[在 `enhance` 中包装异步增强 →](../../how_to/adapt_async_models.zh.md)** · How-to

</div>
