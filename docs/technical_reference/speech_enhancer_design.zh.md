# 语音增强设计

```python
class SpeechEnhancer(ABC):
    """语音增强引擎抽象基类。"""

    @abstractmethod
    def enhance(self, audio: bytes, far: bytes) -> bytes:
        ...

    def flush(self) -> bytes:
        return b""

    async def async_enhance(
        self,
        audio: bytes,
        far: bytes,
    ) -> bytes:
        ...

    async def async_flush(self) -> bytes:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def clone(self) -> "SpeechEnhancer":
        ...
```

## 音频格式约定

`SpeechEnhancer` 的输入和输出均使用：

- PCM 16-bit
- 单声道
- 16000 Hz
- 裸 PCM 字节流，不包含 WAV 头

`audio` 表示近端麦克风音频。`far` 表示远端参考音频，通常来自系统正在播放给用户的 TTS 音频，用于声学回声消除。

上游音频管线总是提供 `far`，并保证：

- `far` 与 `audio` 使用相同音频格式
- `len(far) == len(audio)`
- 二者描述同一段时间窗口内的近端输入与远端参考

## `enhance` 与 `async_enhance`

服务层实际优先调用 `async_enhance`。如果底层实现只有同步 API，可以实现 `enhance`，并复用基类提供的 `async_enhance` 线程池包装。

如果底层实现本身是异步或远程服务，最佳实践是直接实现 `async_enhance`，再用同步包装实现 `enhance`。

```python
import asyncio

def enhance(self, audio: bytes, far: bytes) -> bytes:
    return self._run_coro(self.async_enhance(audio, far=far))

def _run_coro(self, coro: "asyncio.Future[bytes]") -> bytes:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

## `far` 的语义

`far` 是必传接口参数，用于支持带远端参考的语音增强或回声消除。

- 没有远端播放时，服务层会传入与 `audio` 等长的静音 `far`。
- 有 TTS 播放时，服务层会从 TTS 参考缓冲中取出与当前 `audio` 等长的片段。
- 参考缓冲不足时，服务层会在右侧补静音，仍然保证长度一致。
- FastEnhancer 会接收接口参数，但本地模式和远程模式都会忽略 `far`。
- 面向回声消除服务的具体 enhancer 实现可以使用 `far`。

因此，语音增强实现可以安全地假设：`far` 与 `audio` 等长。该约束由上游服务管线负责，具体 enhancer 实现不需要重复检查。

## 服务层如何生成远端参考

`EnhancerManager` 消费 `AudioFrameReceived`，并发布 `EnhancedAudioFrameReceived`。当存在 `SpeechEnhancer` 模型时，它会在处理每个用户音频帧时调用：

```python
far = far_reference.take(len(audio))
enhanced = await enhancer.async_enhance(audio, far=far)
```

远端参考来自 `TTSChunkReady` 事件。服务层会把将要发送给客户端的 TTS chunk 复制一份，转换为语音增强接口要求的格式后写入参考缓冲。

当前转换规则为：

- 输入 TTS chunk 视为 PCM 16-bit 单声道
- 根据 `TTSChunkReady.sample_rate` 重采样到 16000 Hz
- 写入一个有限长度的 FIFO 缓冲
- 默认缓冲上限为 5 秒，可通过 `far_reference_buffer_seconds` 配置

当用户音频到达时，服务层从该缓冲取出与当前 `audio` 等长的远端参考。没有可用参考时返回等长静音。

## `TTSChunkPlayed` 的作用

`TTSChunkPlayed` 表示前端确认某个 TTS chunk 已播放完成。当前事件不携带 `chunk_id`、客户端播放时间戳或样本偏移，因此它不能提供样本级对齐。

服务层将其用于清理已经播放但尚未被麦克风帧消耗的旧参考音频，避免用户等 TTS 播完后再说话时仍然拿旧 TTS 作为 `far`。

更精确的对齐应由未来的客户端播放参考回传或带时间戳的播放事件实现。

## FastEnhancer 行为

`FastEnhancer` 实现 `SpeechEnhancer` 接口，但不使用 `far`。本地 ONNX 模式和远程 FastEnhancer WebSocket 模式都是如此。远程 FastEnhancer 仍然使用旧的二进制 PCM 协议。

`PyWebRTCAudio` 是面向 pywebrtc-audio 服务的具体适配器。它的构造参数仅包含 `base_url`，并把上游提供的 `audio` 和 `far` 发送到服务的 `/v1/stream` JSON WebSocket 接口。

## `flush`

`flush` / `async_flush` 用于排出增强器内部缓冲的尾部音频。服务层会在 `VADSpeechEnd` 后插入 flush barrier，确保所有更早的音频帧先完成增强和下游分发。

对远程 FastEnhancer 模式，`flush` 会发送本地 pending audio 的补齐帧，然后使用远程 FastEnhancer 的 flush 命令。

## `reset` 与 `clone`

`reset` 应清空当前会话的流式状态，例如模型 cache、输入输出缓冲、远程连接和 pending audio，但不应重新加载权重。

`clone` 应为新会话创建独立运行时实例。多个克隆可以共享权重、配置或只读资源，但不能共享流式缓冲、远程连接、pending audio 或会话状态。

请参阅[模型对象的 `clone()` 与 `reset()` 语义](model_clone_reset.zh.md)。

## 实现建议

- 输出长度应尽量与输入 `audio` 长度一致，避免下游 VAD/ASR 时间轴漂移。
- 不要在模型接口层自行猜测 TTS 对齐；长度补齐和 far 选择由服务层负责。
- 如果实现需要 `far`，可以假设上游管线已经将其长度与 `audio` 对齐。
- 如果实现不支持 `far`，可以忽略该参数，但不应因为收到静音 `far` 而失败。
- 远程实现应在 `reset` 和连接重建时清空 pending audio。
