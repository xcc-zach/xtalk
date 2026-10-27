2026-10-27

# 需求变更

- 读者需要按自己的处境找到文档：第一次接触 xtalk 的人跟着教程完成一个可运行的成果，带着明确任务的人直接查操作步骤，有经验的人查接口事实，想深入的人读设计原理。
    - 现有导航按「Tutorial / App Tutorial / Technical Reference / API」划分，同一页面常混有操作步骤、参数说明和原理讨论，读者无法预期一页会给出什么。
    - 多数扩展点（模型、Agent、工具、事件与 Manager、Bot2Bot）对读者是首次接触，这些页面应作为教程串成学习路径，而非零散的操作清单。
- 每篇教程需要交代前置条件、每步的可观察结果和预期中的报错，并在结尾推荐下一步阅读。
- 文档站保持中英双语，`README.md` 不在本次变更范围内。

# spec变更

- 文档分区。
    - `docs/` 下按 Diátaxis 四类组织：`tutorial/`、`how_to/`、`reference/`、`explanation/`；`index.md`、`quickstart.md` 保留在根目录；`api/` 保持原路径与生成脚本不变，导航上归入 Reference。
    - 每个页面只属于一类；一页同时包含操作步骤、参数事实和原理讨论时按下文拆分。
    - 每个页面以一级标题开头，标题与 `mkdocs.yml` 中的导航标签一致。
    - 所有页面保持 `.md` 与 `.zh.md` 成对存在于同一目录，两者结构等价、事实一致。
    - 不保留旧路径、重定向、旧名称对照或迁移说明。
- Tutorial（`docs/tutorial/`）。
    - 导航分三组，组内按顺序衔接，后一篇默认读者已完成前一篇。
        - Basics：Start the Service → Configure the Service → Connect a Client。
        - Extend the Service：Introduce a New Model → Introduce an LLM Agent → Attach Tools → Customize the Service → Bot2Bot。
        - Desktop App：Model Configuration → Add Tools。
    - 每篇教程满足以下结构。
        - 开头一段说明本篇最终产出的可运行成果，以及前置条件（链接到前一篇教程或 Quickstart）。
        - 每个步骤说明目的，执行后给出可观察的结果（终端输出、浏览器行为、日志行或响应内容）。
        - 步骤中会先出现再被后续步骤修复的报错，在报错出现前说明其为预期行为。
        - 结尾为「Next Steps」卡片区，按 [next-step 要求](../../.claude/skills/kkspec-documentation/references/requirements/next-step.md) 编写，只推荐相关页面，最推荐的放第一项。
        - 不影响当前操作与结果判断的实现机制、完整行为清单移至 Reference 或 Explanation 页面并链接。
    - 页面内容来源。
        - `start_the_service.md`：保留 Client setup、Server setup、Full example；成果为浏览器中可对话的页面。
        - `config_the_service.md`：保留 Configuration file、Staged configuration、Customize service behavior；Frontend configuration 一节移出到 `reference/client_session_config.md`。
        - `connect_a_client.md`（新建）：由原 `tutorial/realtime_api.md` 的 Run an ASGI application 与 Minimal audio client 组成；成果为用 `examples/sample_client.py` 与独立运行的 ASGI 应用完成一轮语音对话。
        - `introduce_a_new_model.md`、`introduce_an_llm_agent.md`、`customize_the_service.md`：内容保留，按教程结构补充前置条件、可观察结果与 Next Steps。
        - `attach_tools.md`（新建）：由原 `tutorial/tool_design.md` 的 Attach Tools 一节组成；成果为 Agent 成功调用一个工具。
        - `bot2bot.md`：保留 Create the bridge、Configure a bot session、Publish bot output back into the bridge、Let the bridge capture real user audio、Full example、Stop the bridge；Create the bridge 中的桥接实例类型声明移出到 `reference/audio_bridge.md`。
        - `desktop_app/model_configuration.md`、`desktop_app/add_tools.md`：由 `app_tutorial/` 对应页面移入。
- How-to（`docs/how_to/`）。
    - 每页开头一句说明完成的任务与前提，正文直接给出操作，理论内容以链接指向 Reference 或 Explanation。
    - 页面内容来源。
        - `mount_in_fastapi.md`（新建）：原 `tutorial/realtime_api.md` 的 Mount in FastAPI 一节。
        - `deploy_fully_local.md`：原 `tutorial/sample_config_for_fully_local_deployment.md`。
        - `adapt_async_models.md`（新建）：合并原 `asr_design`、`tts_design`、`vad_design`、`turn_detector_design`、`speech_enhancer_design` 中「Best Practice for Implementing …」的同步包装模式，按模型类型各给一个代码块；并入原 `asr_design` 的 Pseudo-streaming Integration 一节。
        - `logging.md`、`testing.md`、`text_input.md`：由 `tutorial/` 移入。
        - `desktop_app/web_search.md`、`desktop_app/voice_wake.md`：由 `app_tutorial/` 移入。
- Reference（`docs/reference/`）。
    - 每页按 [reference 要求](../../.claude/skills/kkspec-documentation/references/requirements/reference.md) 组织：接口签名、简短描述、参数、返回值、异常、示例，结尾为 Next Steps。
    - 页面内容来源。
        - `service_config.md`：原 `tutorial/service_config.md`。
        - `client_session_config.md`（新建）：原 `config_the_service.md` 的 Frontend configuration 一节（`inputConfig`、`outputConfig`、`serviceURLs`），链接到 `api/client/interfaces/SessionConfig.md`。
        - `realtime_api.md`：原 `technical_reference/realtime_api.md`；并入原 `vad_design` 的 Configuring Backend VAD Through Realtime 示例到 Session settings 一节。
        - `supported_models.md`、`model_clone_reset.md`：由 `technical_reference/` 移入。
        - `tools.md`（新建）：原 `tutorial/tool_design.md` 的 Built-in Tools、Tool Types、ToolEngine。
        - `audio_bridge.md`（新建）：原 `bot2bot.md` 的桥接实例类型声明与 Current limitations。
        - `models/asr.md`：原 `asr_design` 的接口代码、`async_recognize_stream` 参数与返回值、`stream_chunk_bytes_hint`、`clone`/`reset` 链接。
        - `models/tts.md`：原 `tts_design` 的 http TTS 与 websockets TTS 接口代码、参数与返回值、Method Semantics、Audio Format、`set_voice`/`set_emotion`、`clone`。
        - `models/vad.md`：原 `vad_design` 的接口代码、`async_is_speech` 参数与返回值、Updating the Threshold at Runtime、`clone`/`reset`。
        - `models/turn_detector.md`：原 `turn_detector_design` 的类型定义、`async_detect` 参数、返回值、三个枚举含义、`listening` 含义、`clone`。
        - `models/speech_enhancer.md`：原 `speech_enhancer_design` 的接口代码、Audio Format、`far` 的长度约定、`flush`、`reset`/`clone`。
        - `models/forced_aligner.md`：原 `forced_aligner` 的接口代码、`ForcedAlignmentUnit`、`align`、`async_align`、`clone`。
- Explanation（`docs/explanation/`）。
    - 每页讨论设计取舍与运行机制，不包含操作步骤；结尾为 Next Steps。
    - 页面内容来源。
        - `system_design.md`：原 `technical_reference/system_design.md`。
        - `vad.md`（新建）：原 `vad_design` 的 How the Service Layer Consumes VAD Output、Frontend VAD Logic、Relationship Between Frontend VAD and Backend VAD。
        - `tts.md`（新建）：原 `tts_design` 的 How the Service Layer Consumes TTS Output、How the Service Layer Consumes StreamingTextTTS、TTS Playback Progress Tracking。
        - `speech_enhancer.md`（新建）：原 `speech_enhancer_design` 的 Meaning of `far` 中的服务层行为、How the Service Layer Builds the Far-End Reference、Role of `TTSChunkPlayed`、FastEnhancer Behavior、Implementation Suggestions。
        - `turn_detector.md`（新建）：原 `turn_detector_design` 的输入组合与两类实现路径、Assistant response updates、Implementation Suggestions。
        - `forced_aligner.md`（新建）：原 `forced_aligner` 的 Play First, Calibrate Later 及其子节。
        - `audio_bridge.md`（新建）：原 `bot2bot.md` 的引言、When to use it、Shared stream model。
- 首页与 Quickstart。
    - `index.md` 保留现有提示与链接，增加一段产品说明和四个分区的入口卡片。
    - `quickstart.md` 结尾增加 Next Steps 卡片，内部链接指向新路径。
- 站点配置与辅助文件。
    - `mkdocs.yml` 的 `nav` 按上述分区重写；删除指向不存在文件的 `Recipe: developer/recipe.md`；`nav_translations` 与新标签一一对应；增加 `extra_css: [stylesheets/extra.css]`。
    - 新增 `docs/stylesheets/extra.css`，内容为 next-step 要求中的卡片样式。
    - `docs/AGENTS.md` 的 Scope 一节按新分区命名。
    - 所有页面间的相对链接指向新路径；`mkdocs build --strict` 无警告。
- 范围外。
    - `README.md` 不在本次修改；其中指向文档站的链接假定在新结构部署后更新为新路径（`tutorial/config_the_service/` 不变，`tutorial/sample_config_for_fully_local_deployment/` 改为 `how_to/deploy_fully_local/`）。
    - `docs/api/` 生成内容、`docs/generate_server_docs.py`、`frontend/typedoc.json`、`docs/README.md` 不变。

# 设计变更

```text
docs/
├── c index.md                              # 增加产品说明与四分区入口卡片
├── c quickstart.md                         # 更新内部链接，增加 Next Steps
├── c AGENTS.md                             # Scope 按新分区命名
├── + stylesheets/extra.css                 # Next Steps 卡片样式
├── tutorial/
│   ├── c start_the_service.md              # 补前置条件、可观察结果、Next Steps
│   ├── c config_the_service.md             # 移出 Frontend configuration，补教程结构
│   ├── + connect_a_client.md               # 独立运行 ASGI 应用并用 sample_client.py 对话
│   ├── c introduce_a_new_model.md          # 补教程结构
│   ├── c introduce_an_llm_agent.md         # 补教程结构
│   ├── + attach_tools.md                   # 原 tool_design 的 Attach Tools
│   ├── c customize_the_service.md          # 补教程结构
│   ├── c bot2bot.md                        # 只保留操作步骤
│   ├── - tool_design.md                    # 拆分到 attach_tools 与 reference/tools
│   ├── - realtime_api.md                   # 拆分到 connect_a_client 与 how_to/mount_in_fastapi
│   ├── - service_config.md                 # 移至 reference/
│   ├── - sample_config_for_fully_local_deployment.md  # 移至 how_to/deploy_fully_local
│   ├── - logging.md                        # 移至 how_to/
│   ├── - testing.md                        # 移至 how_to/
│   ├── - text_input.md                     # 移至 how_to/
│   └── desktop_app/
│       ├── + model_configuration.md        # 原 app_tutorial/model_configuration
│       └── + add_tools.md                  # 原 app_tutorial/add_tools
├── - app_tutorial/                         # 内容分别移入 tutorial/desktop_app 与 how_to/desktop_app
├── how_to/
│   ├── + mount_in_fastapi.md               # 原 tutorial/realtime_api 的 Mount in FastAPI
│   ├── + deploy_fully_local.md             # 原 sample_config_for_fully_local_deployment
│   ├── + adapt_async_models.md             # 各模型类型的同步包装模式与 ASR 伪流式接入
│   ├── + logging.md                        # 原 tutorial/logging
│   ├── + testing.md                        # 原 tutorial/testing
│   ├── + text_input.md                     # 原 tutorial/text_input
│   └── desktop_app/
│       ├── + web_search.md                 # 原 app_tutorial/web_search
│       └── + voice_wake.md                 # 原 app_tutorial/voice_wake
├── reference/
│   ├── + service_config.md                 # 原 tutorial/service_config
│   ├── + client_session_config.md          # 原 config_the_service 的 Frontend configuration
│   ├── + realtime_api.md                   # 原 technical_reference/realtime_api，并入 VAD 阈值示例
│   ├── + supported_models.md               # 原 technical_reference/supported_models
│   ├── + model_clone_reset.md              # 原 technical_reference/model_clone_reset
│   ├── + tools.md                          # 原 tool_design 的 Built-in Tools、Tool Types、ToolEngine
│   ├── + audio_bridge.md                   # 桥接实例类型与限制
│   └── models/
│       ├── + asr.md                        # ASR 接口与参数语义
│       ├── + tts.md                        # TTS / StreamingTextTTS 接口与参数语义
│       ├── + vad.md                        # VAD 接口与参数语义
│       ├── + turn_detector.md              # TurnDetector 类型与参数语义
│       ├── + speech_enhancer.md            # SpeechEnhancer 接口与音频格式
│       └── + forced_aligner.md             # ForcedAligner 接口
├── explanation/
│   ├── + system_design.md                  # 原 technical_reference/system_design
│   ├── + vad.md                            # 服务层与前端 VAD 机制
│   ├── + tts.md                            # 服务层消费 TTS 输出与播放进度跟踪
│   ├── + speech_enhancer.md                # 远端参考构建与 FastEnhancer 行为
│   ├── + turn_detector.md                  # 输入组合、实现路径与助手文本更新
│   ├── + forced_aligner.md                 # 先播放后校准
│   └── + audio_bridge.md                   # 共享音频流模型
├── - technical_reference/                  # 内容拆分到 reference/ 与 explanation/
└── api/                                    # 不变
mkdocs.yml                                  # c 重写 nav、nav_translations，增加 extra_css
```

上述每个 `.md` 均同时包含对应的 `.zh.md`。

- 页面迁移。
    - 用 `git mv` 移动整页保留的文件，再按分区要求修改内容。
    - 拆分页面时，先新建目标页面并复制对应小节，再删除原文件；英文与中文同步进行。
- Tutorial 页面改写。
    - 为缺少一级标题的页面补标题：Start the Service、Introduce a New Model、Bot2Bot、Attach Tools、Connect a Client。
    - 开头段落写明成果与前置条件；Basics 组第一篇以 Quickstart 为前置，Extend the Service 组第一篇以 Configure the Service 为前置，Desktop App 组第一篇无前置。
    - 为每个步骤补充可观察结果；`customize_the_service.md` 中注册 Manager 之前事件无人消费的现象在对应步骤前说明。
    - Next Steps 卡片使用 `<div class="grid cards next-step-cards" markdown>` 结构，每张卡片一个链接。
- 拆分规则。
    - 原 `*_design.md` 中以「Best Practice for Implementing」命名的小节合并进 `how_to/adapt_async_models.md`，每个模型类型一个二级标题。
    - 原 `*_design.md` 中描述参数、返回值、枚举含义与接口代码的小节进入 `reference/models/<type>.md`。
    - 原 `*_design.md` 中描述服务层行为、前端行为、设计建议的小节进入 `explanation/<type>.md`。
    - `reference/models/<type>.md` 的 Next Steps 第一项指向对应 `explanation/<type>.md`（ASR 指向 `how_to/adapt_async_models.md`），第二项指向 `tutorial/introduce_a_new_model.md`。
- 站点配置。
    - `nav` 顶层顺序：Home、Quickstart、Tutorial、How-to、Reference、Explanation；Reference 末尾包含 API 子组（Client、Server）。
    - `nav_translations` 删除不再使用的标签，新增：How-to → 操作指南、Reference → 参考、Explanation → 原理说明、Basics → 基础、Extend the Service → 扩展服务、Desktop App → 桌面应用、Connect a Client → 连接客户端、Attach Tools → 挂载工具、Mount in FastAPI → 挂载到 FastAPI、Deploy Fully Local → 完全本地部署、Adapt Async Models → 适配异步模型、Client Session Config → 客户端会话配置、Tools → 工具、Audio Bridge → 音频桥接、Model Interfaces → 模型接口、ASR → ASR、TTS → TTS、VAD → VAD、Turn Detector → Turn Detector、Speech Enhancer → 语音增强、Forced Aligner → 强制对齐。
    - 删除 `Recipe: developer/recipe.md`。
- 验证。
    - 在 `docs/requirements.txt` 环境下执行 `mkdocs build --strict`，确认无断链与缺页警告。
    - 逐页核对 `.md` 与 `.zh.md` 的标题层级一致。
