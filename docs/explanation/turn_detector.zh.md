# 轮次检测流程

轮次检测器判断用户何时说完、何时打断。X-Talk 通过同一个 `async_detect` 方法向它提供多种信号，使音频驱动和文本驱动的检测器共享同一套服务层契约。

## 输入组合

`TurnDetectorManager` 根据刚发生的事件，以不同的参数子集调用 `async_detect`：

- 仅传入 `audio`：纯音频判定路径。
- 仅传入 `text` 和 `speech_pause`：文本语义判定路径。
- 仅传入 `assistant_text`：用已播放的 AI 回复文本更新检测器上下文。
- 仅传入 `speech_start=True`：通知检测器当前说话轮次开始。

仓库中的两个实现分别代表两条主要路径。`SoulxDuplug` 以音频路径为主，并在文本停顿信号上提供 fallback；`LLMTurnDetector` 以文本语义路径为主，依赖 `text` 与 `speech_pause`。检测器可以接收但不使用 `assistant_text`；需要感知 AI 回复的检测器把累计值保存为会话状态，并在后续判断中使用。

## AI 回复文本更新

`TurnDetectorManager` 订阅 `ResponseUpdate` 和 `ResponseFinish`。每收到一个事件，就单独调用一次 `async_detect(assistant_text=event.text)`。这个纯上下文调用的返回值被忽略，不会变成 `STOP_SPEAKING` 或 `START_GENERATION`。

`ResponseUpdate.text` 是已确认播放给用户的累计文本前缀；`ResponseFinish.text` 是最终完整回复。保存该上下文的检测器覆盖旧值而非追加。最后一次 `ResponseUpdate` 和 `ResponseFinish` 可能包含相同文本，因此更新保持幂等。

## 监听状态

`listening` 标志告诉检测器当前在回答哪个问题。系统等待用户时（`listening = True`），检测器决定何时发出 `START_GENERATION`；输出播放中（`listening = False`），检测器判断用户输入是否构成需要 `STOP_SPEAKING` 的打断。`TurnDetectorManager` 在 TTS 开始播放时翻转该标志，在播放结束或被中断后恢复。

## 实现建议

- 每次返回结果时保证 `action` 与 `semantic` 一致。
- 检测器内部的会话状态只属于当前实例；克隆实例之间不共享状态。
- 把 `speech_pause` 视为“当前出现停顿”的提示；轮次结束后它不带重置语义。

## 下一步

<div class="grid cards next-step-cards" markdown>

-   **[Turn Detector 接口 →](../reference/models/turn_detector.zh.md)** · Reference

-   **[注册新模型 →](../tutorial/introduce_a_new_model.zh.md)** · Tutorial

-   **[VAD 流程 →](vad.zh.md)** · Explanation

</div>
