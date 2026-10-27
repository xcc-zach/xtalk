# TTS 接口

X-Talk 定义两个 TTS 接口：`TTS` 接收完整句子；`StreamingTextTTS` 通过长连接接收增量文本。支持增量文本输入的模型同时继承两者。

## `TTS`

```python
class TTS(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes: ...

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes: ...

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]: ...

    @abstractmethod
    def clone(self) -> "TTS": ...

    def set_voice(self, voice_names: list[str]) -> None: ...

    def set_emotion(self, emotion: str | list[float]) -> None: ...
```

`synthesize` 是所有实现都必须提供的基线接口。服务层调用 `async_synthesize_stream`；其默认实现包装 `synthesize_stream`，后者默认产出 `synthesize` 的结果。

### 参数

- `text`：待合成的文本片段。通常是一句完整句子，也可能是服务层 flush 时剩余的最后一段文本。
- `**kwargs`：模型自定义扩展参数。`TTSManager` 不传入额外参数；实现可自行保留该扩展点。

### 返回值

- `synthesize` / `async_synthesize`：完整音频 `bytes`。
- `synthesize_stream` / `async_synthesize_stream`：逐块产出的音频 `bytes`，连续且顺序正确。块大小不限；服务层在发送前会把音频再切成约 100 ms 的小块。

音频格式：PCM 16-bit、单声道、48000 Hz。

### `set_voice` 与 `set_emotion`

可选的控制方法，实验性接口，由 `TTSManager` 通过事件调用。

- `set_voice(voice_names)`：切换当前音色。`voice_names` 通常只含一个名称；`list[str]` 类型为多参考音色预留。
- `set_emotion(emotion)`：切换当前情绪。`emotion` 为字符串标签或模型自定义向量。

语速控制位于 `TTS` 接口之外；服务层的 speed controller 在合成后处理音频。

## `StreamingTextTTS`

```python
class StreamingTextTTS(ABC):
    """支持流式文本输入的 TTS 抽象基类。"""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def append_text(self, text: str) -> None: ...

    @abstractmethod
    async def flush(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    def audio_stream(self) -> AsyncIterator[bytes]: ...

    @abstractmethod
    def clone(self) -> "StreamingTextTTS": ...
```

### 方法

- `start()`：开始一次上游流式 TTS 会话，例如建立 WebSocket 连接并发送上游 start 事件。
- `append_text(text)`：LLM 增量文本一到就送入上游，不等待完整句子。
- `flush()`：请求合成已接收但尚未输出的文本。服务层只在收到 `TurnTTSFlushRequested` 时调用；句子边界不触发 flush。
- `stop()`：结束上游会话并释放连接资源。`stop()` 不执行 flush；需要合成残留文本时 `TTSManager` 先调用 `flush()`。
- `audio_stream()`：模型一生成音频就产出 PCM 块。`TTSManager` 将其包装为 `TTSChunkReady` 事件。
- `clone()`：返回上游连接状态、缓冲区和后台任务状态都独立的实例。克隆实例不复用其他会话的 live 连接。

### 音频格式

`audio_stream()` 产出连续、顺序正确的 PCM 16-bit、单声道、48000 Hz 音频。上游只支持其他采样率时，例如 Fish Audio 的 44100 Hz PCM，实现在产出前内部重采样到 48000 Hz。

## `clone`

参见[模型对象的 `clone()` 与 `reset()`](../model_clone_reset.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[服务层如何消费 TTS 音频 →](../../explanation/tts.zh.md)** · Explanation

-   **[注册新模型 →](../../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[在 `synthesize` 中包装异步合成 →](../../how_to/adapt_async_models.zh.md)** · How-to

-   **[强制对齐接口 →](forced_aligner.zh.md)** · Reference

</div>
