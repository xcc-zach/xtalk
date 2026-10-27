# TTS 流程

TTS 模型负责产出音频；`TTSManager` 决定何时请求音频以及如何交付给客户端。分离这两项职责后，模型可以按其后端偏好的粒度产出音频，前端协议保持固定。

## 服务层如何消费 `TTS`

`TTSManager` 缓存 LLM 回复、按句切分，再逐句调用 TTS。模型返回的 chunk 是合成侧 chunk；服务层在发送前把音频再切成约 100 ms 的 `TTSChunkReady` 小块。因此模型无需对齐前端传输粒度，只要输出连续、顺序正确的 PCM 即可，天然输出大块音频的模型同样符合播放协议。

## 服务层如何消费 `StreamingTextTTS`

当前模型是 `StreamingTextTTS` 时，`TTSManager` 跳过 `pending_sentence_buffer` 和按句合成路径，按如下事件流工作：

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

后台读取任务在音频到达后立即发布：

```text
StreamingTextTTS.audio_stream() 产出 PCM
  -> TTSManager 切分为约 100 ms 的块
  -> 发布 TTSChunkReady(audio_chunk=chunk, sample_rate=...)
  -> OutputGateway 发送给前端
```

文本一生成就通过 `append_text` 送入上游，生成的音频立即包装为 `TTSChunkReady`。flush 只在 `TurnTTSFlushRequested` 时发生；句子边界不带 flush 语义，上游会话可以自行决定韵律边界。

## 播放进度跟踪

`TTSPlaybackManager` 根据前端回传的 `TTSChunkPlayed` 事件推进已播放音频时长，并发布 `ResponseUpdate`。配置 `forced_aligner` 后可进一步校准已播放到的文本位置；参见[强制对齐流程](forced_aligner.zh.md)。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[TTS 接口 →](../reference/models/tts.zh.md)** · Reference

-   **[注册新模型 →](../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[强制对齐流程 →](forced_aligner.zh.md)** · Explanation

</div>
