# http TTS

该TTS用于输入非流式的情况

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

## TTS 播放文本跟踪

`TTSPlaybackManager` 会根据前端回传的 `TTSChunkPlayed` 事件推进“已经播放的音频时长”，并发布 `ResponseUpdate`。配置 `forced_aligner` 后，`TTSManager` 会先缓存每个完整句子的 TTS 音频，而不是立即发送；`TTSPlaybackManager` 使用最终将要播放的、已经过语速处理的 PCM 完成强制对齐后，才允许该句音频发送到前端，并依据字/词级时间戳计算已经说到的文本前缀。该流程会增加句子开始播放前的延迟，但能保证播放确认到达前对齐结果已经就绪；对齐失败时是否使用按音频比例估算的回退逻辑由配置决定。

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

# websockets TTS

该TTS用于流式输入文本的情况

```python
class StreamingTextTTS(ABC):
    """支持流式文本输入的 TTS 抽象基类。"""

    @abstractmethod
    async def start(self) -> None:
        """启动一次流式 TTS 会话。"""
        ...

    @abstractmethod
    async def append_text(self, text: str) -> None:
        """向当前 TTS 会话追加增量文本。"""
        ...

    @abstractmethod
    async def flush(self) -> None:
        """请求模型合成当前已接收但尚未输出的文本。"""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止当前 TTS 会话并释放连接资源。"""
        ...

    @abstractmethod
    def audio_stream(self) -> AsyncIterator[bytes]:
        """在模型生成音频时持续产出 PCM 音频块。"""
        ...

    @abstractmethod
    def clone(self) -> "StreamingTextTTS":
        """为新的服务会话克隆一个独立的流式 TTS 实例。"""
        ...
```

`StreamingTextTTS` 是独立于 `TTS` 的能力接口。普通非流式 TTS 只需要实现
`TTS`；支持文本流式输入的模型可以同时继承 `TTS` 和 `StreamingTextTTS`。

## 方法语义

- `start()`：开始一次上游流式 TTS 会话，例如建立 WebSocket 连接并发送上游
  `start` 事件。
- `append_text(text)`：每次收到 LLM 增量文本时立即调用。该方法只负责把文本送
  入上游 TTS，不等待完整句子。
- `flush()`：只在服务层收到 `TurnTTSFlushRequested` 时调用。当前设计不在完整
  句子边界自动触发 flush。
- `stop()`：结束当前上游 TTS 会话并释放连接资源。`stop()` 不隐式承担 flush
  语义；如果需要合成残留文本，`TTSManager` 应先显式调用 `flush()`。
- `audio_stream()`：模型一旦生成音频就产出音频块。`TTSManager` 会把这些音频块
  包装成 `TTSChunkReady` 事件。
- `clone()`：为新的服务会话克隆一个实例。克隆后的实例必须拥有独立的上游连接
  状态、缓冲区和后台任务状态，不能复用其他会话中的 live TTS 连接。

## 服务层如何消费 StreamingTextTTS

当当前 TTS 模型是 `StreamingTextTTS` 时，`TTSManager` 不再使用普通 TTS 的
`pending_sentence_buffer` 和按句合成路径，而是按如下事件流工作：

```text
TurnTTSStartRequested
  -> StreamingTextTTS.start()
  -> 启动后台 audio_stream 读取任务

TurnTTSTextAppendRequested(text)
  -> StreamingTextTTS.append_text(text)

TurnTTSFlushRequested
  -> StreamingTextTTS.flush()
  -> StreamingTextTTS.stop()

TurnTTSStopRequested / shutdown
  -> StreamingTextTTS.stop()
```

后台 `audio_stream` 读取任务应当在收到模型音频后立即发布现有事件：

```text
StreamingTextTTS.audio_stream() 产出 PCM
  -> TTSManager 切分为约 100 ms 的块
  -> 发布 TTSChunkReady(audio_chunk=chunk, sample_rate=...)
  -> OutputGateway 发送给前端
```

因此，`StreamingTextTTS` 的核心目标是：文本一到就通过 `append_text` 送入 TTS
上游；TTS 一有音频生成，`TTSManager` 就立即包装为 `TTSChunkReady`。

## 音频格式约定

`audio_stream()` 应产出连续、顺序正确的 PCM 音频。推荐输出格式仍与普通 `TTS`
保持一致：

- PCM 16-bit
- 单声道
- 48000 Hz

如果上游 WebSocket TTS 只支持其他采样率，例如 Fish Audio 的 PCM 输出支持
44100 Hz 但不支持 48000 Hz，则模型实现应在内部重采样到 48000 Hz 后再从
`audio_stream()` 产出，避免改变现有前端二进制音频协议。
