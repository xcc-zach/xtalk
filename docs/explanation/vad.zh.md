# VAD 流程

VAD 模型每帧只回答一个问题：语音还是静音。轮次级别的含义，即用户开始和结束说话的时刻，由周围的流程产生。X-Talk 默认在浏览器中运行该流程，并为无法运行 VAD 的客户端提供等价的后端实现。

## 服务层如何消费 VAD 输出

`VADManager` 消费 `EnhancedAudioFrameReceived` 事件。后端 VAD 启用时，它将输入音频缓存并切分为固定长度的帧，在逐帧模式下调用 `async_is_speech`、在批量模式下调用 `async_detect_frames` 按输入顺序获得每帧的布尔结果，基于连续语音帧和连续静音帧推进状态机，并在满足阈值时发布 `VADSpeechStart` / `VADSpeechEnd`。

默认参数：

- `vad_sample_rate = 16000`
- `vad_frame_samples = 512`，每帧约 32 ms
- `vad_min_speech_ms = 250`：连续语音帧超过该值时判定用户开始说话
- `vad_redemption_ms = 500`：连续静音帧超过该值时判定用户结束说话

因此模型对轮次保持无状态；平滑逻辑位于 `VADManager`，所有模型实现共享同一套开始/结束行为。

## 前端 VAD 逻辑

Web 前端默认使用 Silero VAD v5，输入为 16 kHz、PCM 16-bit 单声道音频，按 512 个采样点切帧（约 32 ms）。默认参数：

- `positiveSpeechThreshold = 0.1`
- `negativeSpeechThreshold = 0.02`
- `minSpeechMs = 250`
- `redemptionMs = 500`

前端对语音概率应用双阈值：

- 概率大于等于 `0.1` 时当前帧判定为语音。第一帧满足条件时，`FrameProcessor` 立即产生 `SpeechStart`，前端随后发送 `vad_speech_start`。
- 概率低于 `0.02` 时推进 speech-end redemption 计数。
- 概率大于等于 `0.02` 时清零 redemption 计数，因此只有连续低于 `0.02` 的帧才会产生 `SpeechEnd`。
- 概率位于 `[0.02, 0.1)` 时不会触发语音开始；处于语音状态时则阻止语音结束。

默认配置下，`FrameProcessor` 将 500 ms 转换为 15 帧，因此需要约 480 ms 的连续低概率帧才会产生 `SpeechEnd`，前端随后发送 `vad_speech_end`。

`minSpeechMs` 不影响 `SpeechStart`。它驱动 `FrameProcessor` 的 `SpeechRealStart` 和短语音 misfire 判断，前端不转发这两类事件。

启用前端 VAD 时，麦克风 PCM 帧立即上传；VAD 推理通过独立队列在增强后的帧上串行执行。推理落后时该队列可以丢弃旧帧，上传给 ASR 的帧不受影响。

`inputConfig.vadRedemptionMs` 覆盖默认的 `redemptionMs`。它只改变结束等待时间，两个概率阈值保持不变。

## 前端 VAD 与后端 VAD

后端 `VAD` 面向无法运行 VAD 的客户端，以及有意把 VAD 放到服务端的部署。两侧同时启用会产生重复的轮次事件，因此每个部署只选一侧。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[VAD 接口 →](../reference/models/vad.zh.md)** · Reference

-   **[注册新模型 →](../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[通过 `session.update` 设置后端阈值 →](../reference/realtime_api.zh.md#会话配置)** · Reference

</div>
