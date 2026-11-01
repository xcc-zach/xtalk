2026-11-01

# 需求变更

- 使用 XTurnix 时，用户已经说完、XTurnix 却继续等待的情况下，助手需要在用户停顿一段时间后主动开始回复，避免对话卡住。
    - 语音活动检测误判用户仍在说话、迟迟不给出停顿时，同样需要能开始回复。
- 用户的话不是对助手说的（例如在和身边的人说话）时，助手不应仅因为停顿足够长而开始回复。
- 使用者需要按所接入 LLM 是否带推理能力，选择是否让 LLM 额外输出判断理由。

# spec变更

- 新增回合检测器 `XTurnixLLM`。
    - 配置 `type` 为 `XTurnixLLM`，通过现有模型发现机制加载，无需额外注册或替换 manager。
    - 对话历史、ASR 修订、`<|pause|>` 处理、listening/speaking 状态切换、打断时的 stop 判定，以及 XTurnix 自身的 start 判定，均与 `XTurnix` 完全一致。
    - 在 XTurnix 判定之外，仅在 listening 状态下增加定时 LLM 判定，用于决定是否开始回复；speaking 状态不调用 LLM。
- 配置参数。
    - `base_url`、`timeout`、`max_model_len`：含义与默认值与 `XTurnix` 相同，只作用于 XTurnix 请求。
    - `llm`：必填，`ChatOpenAI` 构造参数 dict 或 `BaseChatModel` 实例；是否开启推理及请求自身的超时，由该配置按所接服务的方式控制，检测器不解释。
    - `llm_interval`：默认 1.0，单位秒，正数；相邻两次 LLM 请求的发起间隔，也是窗口开启后首次请求的延迟。
    - `llm_timeout`：默认 10.0，单位秒，正数；从窗口起点算起超过该时长后不再发起新的 LLM 请求，不限制单个请求的执行时长。
    - `llm_output_reason`：默认 true，bool；决定提示词要求的输出格式是否包含 `reason` 字段。
    - 参数不合法时构造失败并报错。
- 等待窗口。
    - 开启：XTurnix 在 listening 状态完成一次被采纳的判定且未开始回复时开启新窗口，起点为该判定被采纳的时刻。
        - 包括模型输出 `<|keep|>`。
        - 包括模型输出 `<|start|>` 但因尚未出现语音停顿而未开始回复。
        - 每次新的被采纳判定都关闭旧窗口，并按上述规则重新开启窗口、重新计时。
    - 关闭：以下任一情况立即关闭当前窗口，并取消该窗口全部未完成的 LLM 请求。
        - 用户 ASR 文本发生变化。
        - XTurnix 有新的被采纳判定。
        - 检测器进入 speaking 状态。
        - 已开始回复（无论由 XTurnix 还是 LLM 触发）。
    - VAD 的说话开始、停顿事件既不开启也不关闭窗口。
- LLM 请求调度。
    - 由音频帧驱动：每次 `async_detect(audio=...)` 调用时检查当前窗口；距起点达到 `llm_interval` 的整数倍、且该倍数时刻不超过 `llm_timeout` 时，发起对应的那一次请求；每个倍数时刻最多发起一次，错过的时刻不补发。
    - 请求在后台任务中执行，不阻塞音频帧处理；不论之前的请求是否返回都按时发起新请求，同一窗口内的请求可并发，数量不设上限。
    - 每个请求在发起时刻取当前对话历史和距窗口起点的秒数作为输入。
    - 请求结果为 `respond`、且其所属窗口在交付时仍未关闭时，下一次 `async_detect(audio=...)` 返回 `START_GENERATION` / `COMPLETE`，并取消同一窗口其余未完成请求；后续 XTurnix 的 start 判定按既有防重复规则不再重复开始回复。
    - 同一窗口内哪个请求先给出 `respond` 就采用哪个，与请求发起顺序无关；超过 `llm_timeout` 后仍在执行的请求给出的 `respond` 同样有效。
    - 结果为 `wait`、无法解析、或请求抛出异常时只结束该请求，异常记录日志，不影响同一窗口的其他请求和后续请求。
    - 所属窗口已关闭的请求结果一律丢弃。
- 结果解析。
    - 从 LLM 回复文本中查找 `"judgement"` 字段值；值为 `respond` 视为回复，值为 `wait` 或找不到该字段视为等待；`reason` 字段只用于日志，不影响判定。
- LLM 提示词。
    - 每次请求发送两条消息：一条 system 消息和一条 user 消息。
    - system 消息为以下文本，其中 `{output_format}` 按 `llm_output_reason` 替换：

        ```text
        You are the turn-taking judge of a voice assistant in a live spoken conversation.
        The user has gone quiet. Decide whether the assistant should start replying now.

        Output "respond" only if BOTH conditions hold:
        1. Finished: given the silence duration, the user has most likely finished and is waiting for the assistant.
           - A complete question, request, or statement usually needs only a short silence (about 1-2 seconds).
           - An utterance that sounds unfinished needs a longer silence, about 3 seconds or more. Examples: trailing conjunctions, fillers or hesitations such as "and", "so", "嗯…", "然后", "就是"; a cut-off clause; a list in progress.
           - If the user asked the assistant to wait or hold on (e.g. "wait", "等一下", "我想想"), the user has not finished.
        2. Addressed to the assistant: the latest user speech is directed at the assistant, not at another person nearby or on the phone.
           - Signs of speech aimed at someone else: calling another person's name; talking about the assistant in the third person; side talk unrelated to the conversation; instructions or replies clearly meant for people in the room.
           - If the latest speech is unrelated to the conversation and has no sign of being aimed at the assistant, treat it as not addressed to the assistant.

        Otherwise output "wait".

        Additional rules:
        - User text comes from streaming speech recognition and may contain recognition errors or missing punctuation. Judge by meaning, not by punctuation.
        - Isolated fillers, noise-like fragments, or bare acknowledgements (e.g. "嗯", "哦", "uh") with no request are "wait".
        - Assistant turns contain only the part of the reply the user actually heard; a reply may have been cut off when the user interrupted.

        {output_format}
        ```

    - `llm_output_reason` 为 true 时 `{output_format}` 为：

        ```text
        Return exactly one JSON object and nothing else:
        {"reason": "<one short sentence>", "judgement": "respond" | "wait"}
        ```

    - `llm_output_reason` 为 false 时 `{output_format}` 为：

        ```text
        Return exactly one JSON object and nothing else:
        {"judgement": "respond" | "wait"}
        ```

    - user 消息格式如下：对话历史按时间从旧到新排列，每条消息一行，前缀为 `User: ` 或 `Assistant: `，内容去掉 `<|pause|>` 标记，包含完整历史；秒数保留一位小数。

        ```text
        Conversation so far (oldest first):
        User: 帮我查一下明天
        Assistant: 好的，请问您想查明天的什么？
        User: 明天上海的天气

        Seconds since the user's last words: 2.0
        ```

- 会话隔离。
    - `clone()` 返回参数相同、复用同一 LLM 客户端、但不含对话历史、窗口和未完成请求的新实例。

# 设计变更

```text
xtalk/
├── src/xtalk/models/turn_detector/
│   └── + xturnix_llm.py                    # 新增 XTurnixLLM：继承 XTurnix，增加等待窗口与定时 LLM 判定
├── tests/
│   └── + test_xturnix_llm.py               # 窗口、调度、并发、提示词与解析测试
└── docs/
    ├── reference/
    │   ├── c supported_models.md           # 在 Turn Detection 下新增 XTurnixLLM 条目与配置示例
    │   └── c supported_models.zh.md        # 同上中文版
    └── explanation/
        ├── c turn_detector.md              # 说明 XTurnixLLM 的等待窗口与 LLM 判定机制
        └── c turn_detector.zh.md           # 同上中文版
```

- 类职责与扩展方式。
    - `XTurnixLLM` 继承 `XTurnix`，`xturnix.py` 不修改；XTurnix 部分的行为通过继承原样保留，子类可替代父类使用。
    - 子类覆盖父类内部方法接入窗口逻辑，不新增公有接口。
        - 覆盖 `_map_action_locked`：在被采纳的 listening 判定未开始回复时开启窗口，其他判定关闭窗口。
        - 覆盖 `_update_stream_text_locked`：用户文本变化时关闭窗口。
        - 覆盖 `listening` setter：进入 speaking 状态时关闭窗口。
        - 覆盖 `async_detect`：带 `audio` 的调用执行窗口调度并交付待返回的 start 结果，其余调用交给父类。
        - 覆盖 `clone`。
    - 窗口状态（窗口代次、起点、已发起的倍数时刻、未完成任务集合、待交付结果）为子类私有字段；窗口代次用于判定请求结果是否过期。
    - 只依赖 `TurnDetector` 既有接口与 `BaseChatModel` 抽象；`TurnDetectorManager`、`TurnDetector` 接口与事件均不修改。
- 后台任务与异常。
    - 每个 LLM 请求为一个 `asyncio` 任务，通过 `ainvoke` 调用；窗口关闭或开始回复时对未完成任务调用 `cancel`。
    - 任务异常通过任务完成回调读取并记录日志，不在请求代码中包裹 try/except。
    - 结果解析使用正则匹配 `judgement` 字段，不依赖 JSON 严格解析，可容忍代码块包裹或多余文本。
    - 会话结束时检测器没有关闭钩子，仍在执行的请求在 LLM 返回或客户端自身超时后结束，其结果因窗口不再被消费而无效。
- 文档。
    - Supported Models 中英文页面新增 `XTurnixLLM` 条目，给出关闭推理（`llm_output_reason: true`）与开启推理（`llm_output_reason: false`）两份配置示例；推理开关写法示例包括 vLLM 的 `extra_body.chat_template_kwargs.enable_thinking` 与 DashScope 的 `extra_body.enable_thinking`。
    - Turn Detection Explanation 中英文页面说明窗口开启/关闭条件、音频帧驱动调度、并发请求先到先得，以及不依赖 VAD 停顿的原因。
- 验证。
    - 用替身替换 `_infer_action`、LLM 和模块内时钟，测试不依赖真实等待。
    - 覆盖窗口开启：`<|keep|>` 开启；`<|start|>` 因缺少停顿未开始回复时开启；VAD 说话开始、停顿事件不开启也不关闭窗口。
    - 覆盖调度：首次请求在 `llm_interval` 后发起；按倍数时刻发起且不补发；超过 `llm_timeout` 不再发起；请求未返回时照常发起新请求。
    - 覆盖结果：后发请求先给出 `respond` 时开始回复且只开始一次，其余请求被取消；`wait`、无法解析、异常不开始回复且不影响其他请求；超过 `llm_timeout` 后返回的 `respond` 仍有效。
    - 覆盖窗口关闭：用户文本变化、XTurnix 新判定、进入 speaking、XTurnix 自身开始回复后，未完成请求被取消，迟到结果被丢弃。
    - 覆盖提示词：两种 `llm_output_reason` 下的 system 消息，以及 user 消息的历史格式、`<|pause|>` 去除和秒数格式。
    - 覆盖 `clone` 隔离、参数校验，以及通过模型发现加载 `XTurnixLLM`。
