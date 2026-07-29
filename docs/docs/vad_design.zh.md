```python
class VAD(ABC):
    """Abstract base class for voice activity detection engines."""

    @abstractmethod
    def is_speech(self, frame: bytes) -> bool:
        ...

    async def async_is_speech(self, frame: bytes) -> bool:
        ...
```

## `is_speech`实现的最佳实践

框架中实际调用的是 `async_is_speech`。因此，如果底层实现本身是异步的，最佳实践是首先实现 `async_is_speech`，然后采用如下方式实现 `is_speech`：

```python
import asyncio

def is_speech(self, frame: bytes) -> bool:
    return self._run_coro(self.async_is_speech(frame))

def _run_coro(self, coro: "asyncio.Future[bool]") -> bool:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

## `async_is_speech`入参与返回值说明

每次调用应返回当前这一帧音频是否包含语音的布尔结果。

`frame` 为当前输入音频帧，格式为 PCM 16-bit、单声道、16 kHz 字节流；返回值 `bool` 中，`True` 表示当前帧为语音，`False` 表示当前帧为非语音。

需要注意：

- 当前服务层中的 `VADManager` 会把逐帧布尔结果再平滑成说话开始/结束事件。
- 如果实现内部维护上下文状态，也应保证返回值表达的是“当前最新完整帧”的语音判定结果。

## 服务层如何消费 VAD 输出

`VADManager` 会消费 `EnhancedAudioFrameReceived` 事件，并在后端启用 VAD 时执行如下流程：

- 将输入音频缓存并切分为固定长度 frame
- 对每个 frame 调用一次 `async_is_speech`
- 基于连续语音帧和连续静音帧数量推进状态机
- 在满足阈值时发布 `VADSpeechStart` / `VADSpeechEnd`

当前默认参数：

- `vad_sample_rate = 16000`
- `vad_frame_samples = 512`
- 每帧约 `32 ms`
- `vad_min_speech_ms = 250`
- `vad_redemption_ms = 500`

其中：

- 连续语音帧累计超过 `vad_min_speech_ms` 时，服务层认为用户开始说话
- 连续静音帧累计超过 `vad_redemption_ms` 时，服务层认为用户结束说话

也就是说，VAD 模型输出的是“原始逐帧判定”，而 turn 级别的 start/end 语义由 `VADManager` 完成。

## 前端VAD逻辑

Web 前端默认使用 Silero VAD v5。输入音频默认采用 16 kHz、PCM 16-bit、单声道格式，并按 512 samples 切帧，每帧约 32 ms。当前默认参数为：

- `positiveSpeechThreshold = 0.1`
- `negativeSpeechThreshold = 0.02`
- `minSpeechMs = 250`
- `redemptionMs = 500`

前端使用双阈值处理模型输出的语音概率：

- 概率大于等于 `0.1` 时，当前帧被认为是语音。第一帧满足条件时，`FrameProcessor` 立即产生 `SpeechStart`，前端随后发送 `vad_speech_start`。
- 概率低于 `0.02` 时，当前帧累计到 speech-end redemption 计数。
- 概率大于等于 `0.02` 时，前端立即清零 redemption 计数。因此只有连续低于 `0.02` 的帧才能触发 `SpeechEnd`。
- 概率位于 `[0.02, 0.1)` 时不会触发语音开始；如果已经处于语音状态，则会阻止语音结束。

在默认 16 kHz / 512 samples 配置下，`FrameProcessor` 将 500 ms 转换为 15 帧，因此实际需要约 480 ms 的连续低概率帧才会产生 `SpeechEnd`。前端随后发送 `vad_speech_end`。

`minSpeechMs` 不会延迟当前集成使用的 `SpeechStart`。它用于 `FrameProcessor` 的 `SpeechRealStart` 和短语音 misfire 判断，但当前前端没有转发这两类事件。

启用前端 VAD 时，麦克风 PCM 帧会立即上传，不等待 VAD 推理。VAD 使用增强后的帧进行判定，并通过独立队列串行处理；当推理落后时，该队列可以丢弃旧帧，但不会阻塞或丢弃上传给 ASR 的麦克风帧。

`inputConfig.vadRedemptionMs` 可以覆盖默认的 `redemptionMs`。该值仅调整结束等待时间，不改变正负概率阈值。

## 前端 VAD 与后端 VAD 的关系

X-Talk 支持前端 VAD。后端 `VAD` 主要用于前端无法运行 VAD，或你明确希望把 VAD 放到服务端执行的场景。

通常不建议同时启用前端和后端 VAD，否则可能产生重复的 turn 事件。

## `clone`与`reset`

VAD要求实现`clone`。

请参阅[模型对象的 `clone()` 与 `reset()` 语义](model_clone_reset.zh.md)。
