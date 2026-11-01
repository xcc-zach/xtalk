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

## XTurnixLLM 等待窗口

`XTurnixLLM` 保留 XTurnix 的全部判定，并在 XTurnix 继续等待时提供第二次开始回复的机会。XTurnix 只有在 VAD 给出停顿后才会开始回复，因此 VAD 一直报告有语音、或模型一直预测 `<|keep|>` 时，已经说完的用户可能一直等不到回复。

- **开启。** 每次被采纳且未开始回复的 listening 判定都会以当前时刻为起点开启新窗口，包括 `<|keep|>`，以及因尚未出现停顿而被挡下的 `<|start|>`。VAD 的说话开始与停顿事件既不开启也不关闭窗口。
- **关闭。** 用户转写变化、XTurnix 有新的被采纳判定、助手开始说话或已开始回复时，窗口关闭；关闭时取消该窗口所有未完成的 LLM 请求，并丢弃迟到的结果。
- **调度。** manager 不为检测器提供定时器，因此由音频帧调用驱动调度。距窗口起点每到 `llm_interval` 的整数倍、且不超过 `llm_timeout` 时，检测器在后台发起一次请求，输入完整对话历史和已等待的秒数；错过的时刻不补发。
- **并发。** 不论之前的请求是否返回都照常发起新请求。当前窗口内最先回答 `respond` 的请求在下一个音频帧开始回复，其余请求被取消。`wait`、无法解析的回复或失败的请求只结束该请求本身。
- **判定。** 只有停顿时长足以说明用户在等待回复、且最新发言是对助手而不是对其他人说的，LLM 才回答 `respond`。设置 `llm_output_reason: false` 时，提示词不再要求 `reason` 字段，适用于自带推理的 LLM。

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
