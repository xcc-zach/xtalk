```python
class TTS(ABC):
    """Abstract base class for text-to-speech engines."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        ...

    def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]:
        yield self.synthesize(text)

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        ...

    async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    ) -> AsyncIterator[bytes]:
        ...

    @abstractmethod
    def clone(self) -> "TTS":
        ...

    def set_voice(self, voice_names: list[str]) -> None:
        ...

    def set_emotion(self, emotion: str | list[float]) -> None:
        ...
```

## `synthesize`实现的最佳实践

`synthesize` 是所有 TTS 实现都必须提供的基线接口。

框架中实际优先调用的是 `async_synthesize_stream`。因此，实现新的 TTS 时，最佳实践是首先实现 `async_synthesize_stream`，然后采用如下方式实现 `synthesize`：

```python
import asyncio

def synthesize(self, text: str) -> bytes:
    return self._run_coro(self._collect_stream(text))

async def _collect_stream(self, text: str) -> bytes:
    chunks: list[bytes] = []
    async for chunk in self.async_synthesize_stream(text):
        chunks.append(chunk)
    return b"".join(chunks)

def _run_coro(self, coro: "asyncio.Future[bytes]") -> bytes:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

如果底层实现本身已经是同步流式的，也可以首先实现 `synthesize_stream`，再复用基类提供的 `async_synthesize_stream` 默认包装。

## `synthesize`、`synthesize_stream`入参与返回值说明

### 输入参数

- `text`：当前待合成的文本片段。通常是一句完整句子，也可能是服务层 flush 时剩余的最后一段文本。
- `**kwargs`：模型自定义扩展参数。当前框架默认不会从 `TTSManager` 传入额外参数，但实现可自行保留扩展能力。

### 返回值

- `synthesize`：返回完整音频 `bytes`
- `synthesize_stream` / `async_synthesize_stream`：逐块产出音频 `bytes`

音频格式约定为：

- PCM 16-bit
- 单声道
- 48000 Hz

## 服务层如何消费 TTS 输出

`TTSManager` 不会把整个 LLM 回复一次性直接丢给 TTS。它会先缓存文本、按句切分，再逐句调用 TTS。

这一层有两个和模型实现强相关的语义：

- 模型返回的 chunk 是“合成侧 chunk”，不等同于最终发给前端的 chunk。
- 服务层会再次把音频切成固定约 100 ms 的 `TTSChunkReady` 小块后再向外发送。

因此：

- 模型无需自行对齐前端发送粒度。
- 只要保证输出是连续、顺序正确的 PCM 音频即可。
- 如果模型天然输出很大的块，也不会破坏前端播放协议，因为服务层还会再切分一次。

## `set_voice`与`set_emotion`（实验接口）

这两个方法是可选控制接口，由 `TTSManager` 通过事件调用：

- `set_voice(voice_names)`：切换当前音色
- `set_emotion(emotion)`：切换当前情绪

其中：

- `voice_names` 当前通常只包含一个音色名，但接口保留为 `list[str]`，便于未来支持多参考音色。
- `emotion` 当前既可以是字符串标签，也可以是模型自定义的向量表示。

语速调整不属于 `TTS` 接口本身。当前仓库中的语速控制由服务层的 speed controller 在 TTS 输出音频之后单独处理。

## `clone`

请参阅[模型对象的 `clone()` 与 `reset()` 语义](model_clone_reset.zh.md)。
