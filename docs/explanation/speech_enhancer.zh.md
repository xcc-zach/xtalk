# 语音增强流程

回声消除需要知道麦克风录音时扬声器正在播放什么。X-Talk 把这一信息放在服务层：`EnhancerManager` 生成远端参考，并向每个增强器传入等长的 `audio` / `far` 对，模型实现无需自行推断 TTS 时序。

## 服务层如何生成远端参考

`EnhancerManager` 消费 `AudioFrameReceived`，发布 `EnhancedAudioFrameReceived`。配置 `SpeechEnhancer` 后，每个用户音频帧按如下方式处理：

```python
far = far_reference.take(len(audio))
enhanced = await enhancer.async_enhance(audio, far=far)
```

参考来自 `TTSChunkReady`。服务层复制每个将要发送给客户端的 TTS chunk，转换为增强接口要求的格式后写入参考缓冲：

- 输入 TTS chunk 视为 PCM 16-bit 单声道
- 根据 `TTSChunkReady.sample_rate` 重采样到 16000 Hz
- 转换后的音频写入有限长度的 FIFO 缓冲
- 缓冲默认保留 5 秒，可通过 `far_reference_buffer_seconds` 配置

用户音频到达时，服务层从缓冲取出等长片段。没有播放或缓冲不足时，返回或在右侧补齐静音，使 `far` 始终与 `audio` 等长。该不变量由管线负责，因此增强器可以直接依赖它。

## `TTSChunkPlayed` 的作用

`TTSChunkPlayed` 表示前端确认某个 TTS chunk 已播放完成。该事件不携带 `chunk_id`、客户端播放时间戳或样本偏移，因此无法做到样本级对齐。服务层用它清理已播放但尚未被麦克风帧消耗的参考音频；否则用户等播放结束后再说话时，会以过期的 TTS 音频作为参考进行增强。更精确的对齐依赖未来带时间戳的客户端播放反馈。

## FastEnhancer 行为

`FastEnhancer` 实现 `SpeechEnhancer`，本地 ONNX 模式和远程 WebSocket 模式都忽略 `far`；远程 FastEnhancer 沿用旧的二进制 PCM 协议。远程模式下，`flush` 先发送本地 pending audio 的补齐帧，再发出远程 flush 命令。

`PyWebRTCAudio` 适配 pywebrtc-audio 服务。其构造参数仅有 `base_url`，并把上游提供的 `audio` 和 `far` 发送到服务的 `/v1/stream` JSON WebSocket 接口。

## 实现建议

- 输出长度与输入 `audio` 保持一致；下游 VAD 和 ASR 的时间轴依赖它。
- TTS 对齐交给服务层；长度补齐和 far 选择都在那里完成。
- 使用 `far` 的实现可以依赖长度不变量；忽略 `far` 的实现收到静音参考时仍正常返回。
- 远程实现在 `reset` 和连接重建时清空 pending audio。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[语音增强接口 →](../reference/models/speech_enhancer.zh.md)** · Reference

-   **[注册新模型 →](../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[TTS 流程 →](tts.zh.md)** · Explanation

</div>
