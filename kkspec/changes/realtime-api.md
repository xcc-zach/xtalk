2026-10-14

# 需求变更

- 开发者需要通过 xtalk 后端 SDK 配置模型并启动服务，使外部应用可以进行实时文本和语音对话。
    - 独立服务可以直接启动。
    - 已有应用可以嵌入服务能力。
- 客户端开发者需要采用 OpenAI Realtime 形式的接口，明确知道可用能力及兼容范围。
    - 支持文本输入、音频输入和流式回复。
    - 支持手动控制轮次及自动检测讲话结束。
    - 支持取消回复和用户插话，后续对话不会使用用户未听到的内容。
- xtalk 仅负责对话能力，业务用户管理和长期历史存储交给外部；已有应用按照 ASGI 迁移 spec 调整集成。

# spec变更

- 后端 SDK 服务入口。
    - 复用 `Xtalk.from_config()`、`Xtalk.configure().build()`、`Models` 和模型注册机制；一个实例提供一个对外模型组合名称，默认 `xtalk`，不新增模型 Provider 体系。
    - 复用 ASGI 迁移 spec 定义的 `Xtalk.create_asgi_app()` 返回完整 ASGI 应用；不提供 Xtalk.run()，不增加 xtalk[server] 安装项，使用者自行选择后端框架、ASGI 服务器及启动方式。
    - 嵌入方挂载 ASGI 应用并在宿主生命周期中显式驱动子应用的 ASGI lifespan；不公开 Xtalk.asgi_lifespan，不新增框架专用路由或生命周期方法。
    - 独立启动和嵌入模式共享同一套资源管理；停止接收新连接后取消会话任务、释放会话模型和发送队列，复用现有模型 clone/reset 约定。
    - mount_routes/connect 删除和业务接口外置由 ASGI 迁移 spec 负责；ASGI 应用默认提供 Realtime 路由，不需要启用开关或 realtime_api 配置。
- Realtime 连接及配置。
    - 提供 `GET /v1/models` 和 `WS /v1/realtime?model=<name>`；模型列表仅返回本实例的对外模型组合，model 查询参数省略时使用服务端配置的模型名（缺省为 xtalk），显式提供未知模型在创建会话前拒绝连接；健康检查由宿主提供。
    - 业务用户鉴权由外部应用完成，xtalk 不依赖 user_id；仅提供可选 API Key 访问控制，api_keys 缺省或为空列表时无需凭证，非空列表时 GET /v1/models 与 WebSocket 在创建会话前验证配置列表中的任意一个 Key。服务端使用 Authorization: Bearer <key>，浏览器 WebSocket 使用 openai-insecure-api-key.<key> 子协议，缺失或错误凭证拒绝访问，不记录原始 Key。
    - 不提供 /v1/realtime/client_secrets；短期凭证由外部应用签发和验证，这是相对 OpenAI 托管 API 的明确差异；默认独立启动无需 API Key，配置 api_keys 后执行相同的可选访问控制。
    - 连接就绪后主动发送 `session.created`，不等待 `attach_session`；每条连接创建独立临时 Session 和 Conversation，不读取数据库或按 session_id 恢复；外部通过 conversation.item.create 按顺序注入历史，再启动新回复。
    - `session.update` 支持 `instructions`、`output_modalities`、`audio.input.format`、`audio.input.transcription`、`audio.input.turn_detection`、`audio.output.format`、`audio.output.voice` 和 `audio.output.speed`；缺省字段不改变配置，`turn_detection: null` 关闭自动轮次检测，更新成功返回完整有效配置 `session.updated`。
    - 输出模式允许 `["text"]` 或 `["audio"]`，音频模式附带音频转写；模型名称连接后固定，音色允许通过同一 audio.output.voice 字段动态修改（xtalk 扩展），速度允许在活动 Response 期间按现有机制更新（xtalk 扩展），范围为 0.25 至 1.5；其他配置受模型能力和服务端允许值约束，不支持的参数返回结构化错误。
    - Realtime API 不支持内部 TTS/LLM 模型动态切换；不接收 change_tts_model/change_llm_model 或 session.xtalk.tts_model/llm_model，事件及字段分别返回明确不支持错误。模型由服务端启动配置确定；此项为明确批准的功能移除，迁移一致性测试不要求保留网络切换能力。
    - 客户端不能配置 Provider 地址、内部凭证或任意后端类；会话限制复用现有并发限制能力，新增接口不依赖旧队列的 action 消息。
- 输入及 Conversation 行为。
    - 会话运行不自动写聊天数据库、知识向量目录或录音文件；知识入库与音频保存由 ASGI 迁移 spec 定义的外部宿主负责，检索工具调用及结果注入仍属于对话能力。
    - xtalk 只保留当前会话内存上下文，不保存业务数据库，不增加 load_context/on_event 存储回调；外部客户端或代理消费协议事件保存历史，连接断开释放运行状态。
    - 外部按 Item ID 应用完成、取消和 response.output_audio.playback.done 事件，不能只保存初始生成文本；重连注入更新后的文本历史，不恢复旧音频或未完成生成。
    - 后端原 vad_speech_start/end 分别使用 input_audio_buffer.speech_started/stopped 并保留 origin；两者的非 server VAD 来源属于 xtalk 扩展语义，前端仅更新状态，不回发 commit。
    - 客户端 vad_speech_start 保留为 C→S 扩展 input_audio_buffer.speech_start，按现有路径通知后端讲话开始；音频不论 VAD 状态持续 append，客户端 vad_speech_end 改为 commit。speech_start 不提交缓冲区，不新增测量或配置机制；对照测试覆盖开始通知及结束提交的原轮次行为。
    - WebSocket 控制事件使用 JSON `type`，JSON 音频字段采用 Base64；同时默认接受裸 PCM 二进制消息作为 Xtalk 扩展，无需配置或协商开关。每条二进制消息等价于一次 input_audio_buffer.append，可与 JSON Base64 输入在同一连接中按接收顺序混用，共享缓冲、重采样及轮次处理。两种输入均为单声道 signed PCM16 little-endian、24 kHz 原始音频，不含 WAV 文件头；内部采样率在边界转换，不支持 G.711。
    - `input_audio_buffer.append` 或二进制消息追加音频；单条消息最大 1 MiB（JSON 按 UTF-8 字节计，二进制按 PCM 字节计），两种输入共享最大 60 秒的输入缓冲区，此限制不代表连接或对话时长；空缓冲区 commit、非法 Base64、奇数 PCM 字节及超限输入均返回错误且不修改有效状态；append 不发送逐包确认。二进制消息不隐式 commit，沿用自动 VAD 或 JSON commit/clear；输出音频默认使用 Base64 JSON；显式协商 binary_pcm 后使用 binary-audio-output.md 定义的互斥二进制输出。非法二进制输入返回 invalid_value 错误且 error.event_id 为 null，连接保持可用。测试覆盖两种输入的等价处理、混合输入、手动 commit/clear、自动 VAD 分发、1 MiB 边界、奇数字节及 60 秒累计上限，拒绝输入不得改变缓冲和重采样状态。
    - `input_audio_buffer.commit` 创建 input_audio Item、发送 `input_audio_buffer.committed` 并清空缓冲区，启动转写并交由现有轮次机制决定回复，commit 本身不直接执行模型推理；`input_audio_buffer.clear` 清空未提交缓冲区并发送 `input_audio_buffer.cleared`。
    - `turn_detection` 首期支持 `server_vad` 和 null；server_vad 支持 `threshold`、`prefix_padding_ms`、`silence_duration_ms`、`create_response`、`interrupt_response`，默认 create_response 和 interrupt_response 为 true；讲话开始/结束分别发送 `input_audio_buffer.speech_started/stopped`，检测结束自动提交且仅在 create_response 为 true 时创建回复。
    - `conversation.item.create` 首期接收 user input_text、system input_text 和 assistant text 消息；添加上下文不隐式触发生成，返回 `conversation.item.added/done`；客户端指定的 item ID 在本 Conversation 内唯一，重复 ID 返回错误。
    - 每个输入及输出 Item 有稳定 ID 和 content_index；转写通过 `conversation.item.input_audio_transcription.delta/completed/failed` 关联输入 Item，Response 和转写可以异步推进，不能依赖转写完成事件与回复之间的全局顺序。
    - ASR 中间结果使用 conversation.item.input_audio_transcription.delta，始终包含 delta 及 xtalk.mode/revision；新文本以完整旧文本为前缀时 mode=append，delta 为新增后缀，否则 mode=replace，delta 为完整新文本。首次按旧文本为空处理，文本不变可跳过，修订为空发送 replace 和空 delta；revision 按 Item/content_index 单调递增，客户端忽略过期预览，completed 后忽略所有预览。此替换语义是标准事件扩展，纯追加式 OpenAI 客户端不能直接兼容。
- 对话功能一致性。
    - 迁移前旧协议与迁移后 Realtime 协议使用相同确定性模型和输入，对照文本输入、语音转写、回复内容、检索工具结果、连续多轮上下文、用户插话后的实际播放前缀及后续上下文。
    - 以协议适配后的用户可见结果和最终会话状态判定一致，不要求旧 action 消息与新 type 事件逐条相同；commit 后由原轮次机制自动回复等已批准扩展语义按新契约单独验证。
- 自定义输入输出事件 API。
    - Service 新增 register_input_handler(*, type: str, handler: InputHandler) 和 register_output_handler(*, source_event: type[Event], type: str, handler: OutputHandler)，均只允许显式命名参数；配置随 Service.clone 复制，每个会话使用自己的网关及状态。
    - 输入注册按外部 type 分发，handler 接收当前 InputGateway 和解析后的完整消息，负责自定义字段校验与逻辑，可通过 gateway.event_bus.publish 发布内部事件；统一错误转换及 event_id 关联由输入网关处理。
    - 输出注册订阅 source_event，type 独立于 handler 指定；handler 接收内部事件，可同步或异步返回载荷字典，返回 None 时不发送；输出网关补 type/event_id 并进入统一发送队列，载荷不得覆盖这两个保留字段。
    - 保留现有 subscribe_event/unsubscribe_event 的优先级、启用条件及自由逻辑能力，也保留公共 OutputGateway.send_signal()；原输出订阅处理器可以在同一函数中发送消息和 event_bus.publish，新注册 API 是便利入口，不替代或削弱旧 API。
    - 不为覆盖或重复注册增加额外门控：输入以 type 为键，输出以 (source_event, type) 为键，后注册替换前注册；不同输出键分别保留。内置默认处理器对不支持事件返回错误，但用户可按同一注册规则覆盖。用户事件类型建议使用自己的命名空间，SDK 只提供一种 Realtime 协议，不开放网关实现注入。
- 终端声卡示例客户端。
    - 新增 examples/sample_client.py，CLI 必填 --base-url 和 --model，可选 --api-key；启动后建立单个 Realtime Session，持续采集麦克风并播放服务端音频，Ctrl+C 停止并释放声卡、连接和任务。
    - base_url 接受 http/https/ws/wss 服务根地址及可选挂载前缀，转换为 ws/wss 并追加 /v1/realtime，model 经查询参数编码；api_key 仅通过 Authorization: Bearer 传递，不输出到终端或拼入 URL。
    - 客户端只做 WebSocket 通信及单声道 PCM16 音频输入输出，使用协议支持的 24 kHz；不加载本地 VAD/enhancer/ASR/TTS、不调用 HTTP 登录或历史接口、不保存录音或历史、不提供文本输入。
    - 等待 session.created 并通过 session.update/updated 配置音频输出和 server_vad 自动轮次后开始采集；不论音量持续 append，不发送客户端 speech_start、commit 或 response.create，VAD/enhancer 由后端负责。
    - 实现当前必需的块播放确认、播放完成、暂停/恢复和停止增量回执扩展；保留接收块边界，实际播放完成才确认，读取 WebSocket 与声卡播放并发执行，排队及协议错误只做终端状态提示。
- 示例客户端端到端验收。
    - 实现完成后必须启动实际后端，并通过 examples/sample_client.py 完成单 Session 对话验收；不要求真实声卡，允许合成语音 PCM 替代麦克风输入，并以按音频时长消费的输出替代扬声器。必须使用该脚本的同一 WebSocket、事件处理和播放确认路径，不得用假 WebSocket、模型替身或其他客户端替代实际后端验收。
    - 验收至少覆盖连续两轮语音输入与回复播放、客户端无本地 VAD/enhancer 时的后端处理、块播放确认释放预算，以及讲话打断后的停播和停止增量结算；确认对话不因未释放预算卡住。
    - 验证无 api_key 的默认访问，以及配置 api_keys 后使用 --api-key 成功访问和错误凭证拒绝访问；记录命令、服务/模型配置、测试结果及相关日志，不记录真实凭证或不必要的录音数据。
    - 没有可用实际模型服务、合成语音输入或运行环境时，明确记录该验收未完成及缺失条件，不宣称验证通过；全部必需验收通过后才将实现任务标记完成。
- 知识、录音与运行支撑。
    - Realtime 服务不提供知识上传、向量入库或录音保存接口，不自动管理相关业务目录或文件；这些能力由外部宿主实现。
    - 服务端检索工具及检索结果参与对话，检索状态通过 conversation.retrieval.updated 输出；工具所需知识资源由宿主配置和维护。
    - 客户端或外部代理负责保存实际采集/播放音频；服务端音频生成事件仅表达生成数据与状态，不承诺代表完整实际播放录音。
    - 对话所需音频处理、模型管理、并发控制、资源清理和延迟观测保留；本变更不删除显式模型缓存或普通运行日志。
- Response 和播放行为。
    - 不支持 C→S response.create，收到返回 unsupported_event 且不启动生成；回复统一由现有输入及轮次判断机制触发，同一 Conversation 的输出生命周期仍发送标准 response.created/done，不删除 S→C response.created。
    - 输出事件按所属对象保证生命周期顺序：`response.created`、`response.output_item.added`、`response.content_part.added`、对应 delta/done、`response.content_part.done`、`response.output_item.done`、`response.done`；output_index 和 content_index 从零开始，所有增量携带 response_id 与 item_id。
    - 文本模式使用 `response.output_text.delta/done`；音频模式使用 `response.output_audio.delta/done` 与 `response.output_audio_transcript.delta/done`，转写描述实际合成内容，不能以尚未合成的 LLM 文本替代；音频 delta 默认含 Base64 音频字节；binary_pcm 扩展协商、二进制封装及控制事件行为以 binary-audio-output.md 为准。
    - 音频转写 delta 仅追加明确已被合成的文本：分句 TTS 按每个成功完成句子发送新增句子文本；流式 TTS 在整轮合成成功且完整音频确认产生后，仅发送一次完整文本作为 delta，因为此前没有输出该轮转写增量；随后 output_audio_transcript.done 携带同一份完整转写。
    - 流式文本接受不代表合成完成，TTSStreamingTextAccepted 不直接映射转写 delta；中途取消或失败且不能确认文本对应已合成音频时，不将全部已接受文本作为转写输出，不推测部分文本。
    - `response.done` 是生成终态，status 为 completed、cancelled、failed 或 incomplete；每个 Response 恰好一个终态，音频模式需等文本及音频生成均结束才 completed，不等待客户端播放确认。
    - `response.cancel` 支持指定 response_id 或取消当前活动 Response；取消生成和待发送旧数据，迟到结果不得进入后续回复；没有可取消的 Response 返回错误但保持连接。
    - 不支持 conversation.item.truncate，收到该标准事件返回明确 unsupported_event 错误且不改变上下文；打断沿用原块播放确认与停止回执，由客户端停播清队列后发送 response.output_audio.playback.stopped，携带 response_id 和未确认增量 played_audio_ms。
    - Conversation 生成记录与已播放记录分开；xtalk 客户端沿用现有块确认和停止增量结算，将实际播放文本用于后续上下文；纯标准客户端不具备这套打断结算能力，文档必须明确该限制，不宣称标准 truncate 兼容。
    - 停止回执的 played_audio_ms 仍是未确认增量，复用原等待 stop 回执及活动 response_id 检查；不新增累计播放位置计数或历史 Item 截断状态。
- 协议健壮性与交付范围。
    - 服务端事件具有唯一 event_id；客户端 event_id 可选，错误通过 `error.type/code/message/param/event_id` 关联客户端事件，不将 event_id 当作通用重试幂等键。
    - 无效 JSON、未知事件和不支持的字段返回 error，不自动断开正常连接；发送队列设字节上限且按顺序发送，持续慢客户端导致超限时终止该连接并清理资源。
    - 首期不提供 WebRTC、SIP、客户端函数调用闭环、MCP、out-of-band Response、Item 删除/查询、音频历史注入、conversation.item.truncate、原生推理信息或完整 OpenAI SDK 兼容承诺；既有服务端工具维持服务端执行，不映射为要求客户端执行的 function_call。
    - 除明确支持的裸 PCM 二进制输入及 binary-audio-output.md 定义的二进制输出扩展外，扩展严格对应已有能力和字段，不新增非 OpenAI 标准支持的功能；不能等价映射的既有对话能力通过下方转换表的扩展保留；持续配置放入 session.update/updated 的 session.xtalk，完整有效配置同时在 session.created 中回显；瞬时事件独立发送，不用配置回显触发命令。
    - 当前 frontend 随 ASGI 迁移 spec 同步切换到 Realtime 协议；本变更交付独立文本和语音客户端示例，不保留旧 /ws 端点。
    - 文档列出相对官方 Realtime 的字段、事件、限制和偏差；提供 Python SDK 独立启动、ASGI 嵌入、文本、语音及打断示例，并同步中英文文档。

# 设计变更

```text
src/xtalk/
├── c api.py                                  # SDK 构建、固定 ASGI 路由及私有生命周期
├── + realtime/
│   ├── + __init__.py                         # 包边界，不导出无消费者的内部 API
│   ├── + config.py                           # 服务配置校验，仅 model/api_keys
│   ├── + auth.py                             # 可选 API Key 校验，不处理用户身份
│   ├── + protocol.py                         # 仅定义对外事件形状、字段类型和错误结构
│   └── + connection.py                       # ASGI 传输适配、收发队列与关闭
├── c serving/interfaces.py                   # 框架无关连接协议，Gateway 不依赖 FastAPI
├── c serving/service.py                      # 固定网关、输入/输出注册 API 及克隆配置
├── c serving/service_manager.py              # 临时会话、任务与资源所有权
├── c serving/session_limiter.py              # 保留 FIFO 准入，通知通过输出接口发送
├── c serving/events.py                       # 内部事件，不依赖 Realtime schema
├── c serving/modules/input_gateway.py        # 原位置改为 Realtime 输入校验、转换及执行
├── c serving/modules/output_gateway.py       # 原位置改为 Realtime 事件构造、编码及输出
├── c serving/modules/turn_taking_manager.py   # 唯一分配音频输入 Item ID，沿轮次事件传递
├── c serving/modules/asr_manager.py           # 输入 Item 关联、ASR 预览及 revision
├── c serving/modules/llm_agent_context_manager.py # 历史注入/实时输入及上下文同步
├── c serving/modules/llm_agent_generation_manager.py # 唯一生成任务状态及终态
├── c serving/modules/tts_manager.py           # 合成、音频投递、块确认预算及动态配置
├── c serving/modules/tts_response_coordinator.py # 唯一准备/投递协调状态
├── c serving/modules/tts_playback_manager.py  # 唯一播放计时、对齐及结算状态
└── c models/agents/interfaces.py             # 仅补齐实际需要的上下文能力
examples/
└── + sample_client.py                        # 单 Session WebSocket 与音频输入输出
```


- SDK 与 HTTP 集成。
    - Realtime API 默认作为 xtalk 对外服务提供；`service_config.realtime_api` 仅允许 `model` 和 `api_keys` 两个配置字段，缺省等价于 {"model": "xtalk", "api_keys": []}。model 必须为非空字符串，api_keys 必须为非空字符串组成的列表（允许空列表），非法值或未知字段启动时报配置错误；不提供 enabled 开关，不读取顶层 realtime 字段；复用 Xtalk 持有的模型原型，避免第二次加载模型。
    - `create_asgi_app()` 返回包含对话路由及内部 ASGI lifespan 的应用；使用者直接部署或挂载应用，SDK 内部不依赖 FastAPI 或其他具体框架、不启动服务器、不绑定宿主框架、不覆盖宿主路由。
    - SDK 不安装或选择 ASGI 服务器，不新增 server extra；24 kHz 边界转换依赖按照实际音频实现归入 SDK 所需依赖，与部署框架及服务器依赖分开。
- 协议与运行时边界。
    - serving 为唯一对话运行时，不新增 realtime/session.py、response.py 或第二套生成/播放引擎；protocol.py 仅定义暴露给外部的事件形状、字段类型及错误结构，不实现解析、校验流程、编码或执行逻辑；connection 只负责收发及关闭，原 serving/modules/input_gateway.py 和 output_gateway.py 分别负责输入校验/转换/执行、输出构造/编码/发送。
    - api.py 直接实现固定 ASGI 路由和私有 lifespan，connection 将 ASGI receive/send 适配为 serving/interfaces.py 的窄传输接口；输入/输出网关不依赖宿主 app，也不操作 FastAPI Request/WebSocket，不新增 asgi.py 或 routes.py。
    - 不新增 realtime/conversation.py 或独立 Item/ID 管理层；输入 item_id 由现有输入轮次管理的唯一入口赋予，ASRManager 持有该轮标识及预览/revision；输出 response_id 由现有生成 Manager 创建，单回复单 Item 时确定性生成输出 item_id，并沿已有内部事件传递。文本增量由生成 Manager 的原始 chunk/累计状态产生，网关不另存文本游标。
    - Service 固定创建原 serving/modules 路径的 Realtime 输入/输出网关，仅支持一种协议；不公开网关类/工厂注入或协议选择配置。自定义注册配置由 Service 持有并复制给会话网关，直接复用已有 EventBus 订阅，不另建事件总线或独立 registry 模块；protocol.py 仅定义内置外部事件形状，用户自定义形状放在用户模块。
    - OutputGateway.send_signal 保持公共，接收完整对外事件字典并走统一发送队列；缺少 event_id 时补齐，显式提供时校验形状，不自动再包 action/data。注册输出 handler 返回的载荷由网关添加注册 type 及生成 event_id，然后调用同一发送入口。
    - 生成任务及一次终态归 llm_agent_generation_manager，合成和未确认音频预算归 tts_manager，投递协调归 tts_response_coordinator，播放位置和对齐结算归 tts_playback_manager；网关消费这些状态变化，不重复维护生命周期决策。
    - 依赖方向为 api 到 Realtime 适配及 Service，网关到 protocol 和内部事件接口；serving/modules 下输入/输出网关可引用 protocol 的事件类型；其他 Manager 只依赖内部事件及模型抽象，不导入 Realtime schema。Service 固定组装原位置网关，不将组装依赖扩散到其他 Manager。并发限制发送通知通过运行时输出接口，不直接序列化协议。
    - 接收循环与音频投递任务分开；输入命令按既有状态入口有序处理，投递等待块确认时仍能接收确认和取消，禁止在接收循环同步等待全部生成/播放结束导致死锁。
    - gateway 通过现有 EventBus 连接模型管线；对外服务默认使用 Realtime 网关，不自动装配业务持久化、知识入库或录音写入路径，模型资源与运行观测保持可用。
    - 新服务默认使用 Realtime 运行模式，沿用输入轮次判断及播放确认结算，新增标准生成事件输出；取消、VAD 和手动提交通过同一状态入口串行处理。
    - 对现有 Agent 增加只供 Realtime 调用的窄能力接口，支持历史注入但不生成、原轮次启动生成、同步播放结算和会话指令；实现不具备能力时启动报错，历史注入不调用会自动回复的 async_accept 路径；具体 Agent 实现依据调用链补齐这些能力，不要求无关模型实现接口。
    - 分句转写从 TTSTextDeliveryFinished(succeeded=True) 的已合成句子产生；流式转写缓存文本接受事件，在确认正常合成完成且 TTSTextSynthesized 提供完整文本及音频后一次输出，取消/失败不得仅凭缓存完整文本判定成功。两个路径分开处理，整轮汇总不能重复追加已输出句子。
    - item_id/response_id 由 Manager 创建并通过内部事件传递，音频时间轴和转写对齐数据由现有播放 Manager 持有；全局不可变模型资源可复用，输入、生成和历史状态必须连接隔离。
    - 新增/修改的跨模块 API 使用 NumPy 风格 docstring，明确类型后直接属性访问，仅在解析、网络边界和清理确有必要时捕获异常。
- 提示词。
    - 本变更不新增或重写任何固定 LLM/深度学习提示词。
    - 客户端 instructions 作为会话级指令送入 Agent，默认沿用开发者已有配置；空字符串清空会话指令，不偷偷拼接新指令或改变轮次检测提示词。
- 测试与示例。
    - 注册 API 测试覆盖仅命名参数、自定义输入解析/分发及 publish、自定义输出同步/异步载荷和 None、保留字段拒绝、重复及内置注册被后注册覆盖且不重复执行、会话克隆隔离，以及旧订阅处理器组合 send_signal 与 publish 的能力；验证不新增第二套总线或协议。
    - 增加依赖边界及集成测试，验证除网关外的 Manager 不导入 Realtime schema、网关不访问框架对象、单一生成/播放状态，以及预算阻塞时仍能处理播放确认和取消；不为单纯文件拆分添加镜像测试。
    - 新增 examples/sample_client.py，声卡和 websockets 依赖及安装命令在文件使用说明和中英文教程列出，由示例使用者安装，不新增 xtalk[server]；示例无需 frontend 包。
    - 实际验收运行 sample_client.py 并记录结果；合成或真实音频与实际后端验收是单独的必需检查，下面 CI 替身测试仅负责自动回归。
    - 新增 tests/test_sample_client.py，使用假 WebSocket 与假声卡验证 CLI/URL/鉴权、单 Session、持续音频输入、后端 VAD 配置、块边界确认、播放结束与停止增量及 Ctrl+C/断线清理，不要求 CI 声卡或真实模型。

    - 转换表逐项建立迁移对照用例，验证扩展无新增定期上报、订阅或确认机制；ASR 覆盖追加、部分共同前缀修订、缩短、清空、重复 revision 和 completed 后迟到预览；验证仅按标准追加的客户端不被声明为完全兼容。

    - 复用 ASGI 迁移 spec 的修改前基线 fixtures，增加 Realtime 对照用例；通过客户端正常路径验证功能等价，不只单测内部事件映射，覆盖模型失败、断开连接及打断迟到输出。

    - 新增 `tests/test_realtime_sdk.py`，覆盖独立部署与宿主挂载生命周期、固定 /v1/realtime 路径、省略 model 使用默认值、未知 model 拒绝连接和部署服务器由使用者提供。
    - 新增 `tests/test_realtime_protocol.py`，覆盖有效配置回显、默认无凭证访问、配置 Key 后的 HTTP/WS 校验、外部中间件拦截、超限、未知事件及错误关联。
    - 新增 `tests/test_realtime_session.py`，使用确定性模型替身覆盖会话无业务落盘、历史注入不生成、断线后新上下文、commit 轮次回复、response.create 被拒绝、VAD 自动轮次、转写修订和连接隔离。
    - 转写测试覆盖分句成功逐句追加、流式成功仅一次完整 delta 及相同 done、接受文本但没有音频、流式取消/失败不输出全部缓存文本，以及分句/整轮汇总不重复追加。
    - 新增 `tests/test_realtime_response.py`，覆盖文本/音频事件顺序、一次终态、取消后迟到输出、生成结束后播放中的停止结算、停止增量时长及不支持标准 truncate及慢客户端清理。
    - 新增 `examples/realtime/server.py`、`text_client.py`；声卡语音示例统一为 examples/sample_client.py，不重复新增 audio_client.py；示例直连新端点，音频示例实现块播放确认、播放完成与停止增量回执，验证原预算及连续两轮对话；文本示例无需音频播放事件。
- 文档。
    - 新增 `docs/tutorial/realtime_api.md` 和 `realtime_api.zh.md`，记录支持矩阵、启动和嵌入方式、生命周期责任、采样率、限制、扩展及打断步骤。
    - 更新 `docs/tutorial/start_the_service.md`、`start_the_service.zh.md`、`docs/index.md` 和 `index.zh.md`，添加新启动方式和教程链接。
    - 兼容契约依据本次讨论检索的官方文档并在实现时固定参考版本，不宣称实现所有官方能力。
        - https://developers.openai.com/api/docs/guides/realtime-conversations
        - https://developers.openai.com/api/reference/resources/realtime/client-events
        - https://developers.openai.com/api/reference/resources/realtime/server-events

**通信事件转换表**

标准新增能力与原播放机制分开：生成链路输出 response.output_text.delta/done 或 response.output_audio_transcript.delta/done，delta 仅为追加内容；原 update_resp/finish_resp 仍由播放确认链路触发并转换为播放扩展，不挪用为生成事件。同一 Item 的标准生成文本须满足前缀一致性测试，打断通过原块确认及停止增量回执结算上下文，不支持标准 truncate。


下表为开发契约；C→S 为客户端输入，S→C 为服务端输出。标准事件仍补齐 event_id、Item/Response 标识及生命周期。扩展表中自定义事件以及带 xtalk 字段的行为是明确扩展，不宣称纯 OpenAI 客户端无需适配。此表覆盖前端实际事件及后端额外支持事件，未实际发送的前端处理器不得被当作现有完整能力。

**OpenAI Realtime 标准事件映射**

| 当前事件/数据 | 方向 | 新事件或字段 | 转换行为 |
|---|---|---|---|
| session_attached | S→C | session.created | 返回运行 Session 和有效配置，不绑定业务历史 |
| submit_text | C→S | conversation.item.create | 不发送 response.create；实时文本输入交给原文本轮次机制自动回复，历史注入不得自动回复，实时输入带 xtalk.generate_response=true，历史缺省不触发 |
| 二进制输入音频 | C→S | input_audio_buffer.append | 不论客户端 VAD 状态持续发送；PCM 转为 Base64 或直接发送裸 PCM 二进制消息（Xtalk 扩展），匹配协商格式 |
| vad_speech_end | C→S | input_audio_buffer.commit | 客户端结束通知改为标准提交事件；提交本轮输入并清空对应缓冲区，沿用现有轮次判断决定自动回复，不再提供 response.create |
| finish_asr | S→C | conversation.item.input_audio_transcription.completed | transcript 为最终完整文本；文本输入不伪装成转写 |
| start_tts | S→C | response.content_part.added | 先建立 Response/输出 Item，再建立音频 part |
| 二进制输出音频 | S→C | response.output_audio.delta | 默认 Base64 和 Response/Item/index 关联；显式 binary_pcm 扩展见 binary-audio-output.md |
| TTSTextDeliveryFinished（分句，succeeded=True） | S→C | response.output_audio_transcript.delta | 追加每个成功完成句子的文本，不等待播放确认 |
| TTSTextSynthesized（流式整轮合成成功） | S→C | response.output_audio_transcript.delta + response.output_audio_transcript.done | 确认完整音频生成后，先发送一次完整文本 delta，再用同一完整文本发送 done |
| tts_finished | S→C | response.output_audio.done | 生成结束，播放结束另记；所有 part 结束才发送 response.done |
| change_voice / voice_changed（首次音频输出前） | C→S / S→C | session.update/updated：session.audio.output.voice | 使用标准音色配置和完整有效配置回显 |
| error | S→C | error | 结构化错误并关联客户端 event_id，生成失败另关闭 Response |

**xtalk 对 OpenAI Realtime 的扩展**

本表所有条目均为自定义事件或标准事件的自定义字段，即使使用标准对象前缀，也不属于官方事件契约。前缀仅用于归类，不改变已有触发条件和字段。ASR 复用官方事件名但扩展替换语义，客户端必须识别 xtalk.mode。

| 当前事件/数据 | 方向 | 新事件或字段 | 转换行为 |
|---|---|---|---|
| vad_speech_start | C→S | input_audio_buffer.speech_start | 保留原客户端讲话开始通知及后端处理；不同于 S→C speech_started，不提交输入或自行启动生成；原可选 timestamp 毫秒字段继续支持，不要求现有前端新增时间戳采集 |
| vad_speech_start | S→C | input_audio_buffer.speech_started（xtalk 扩展语义） | 保留现有后端触发与 origin；不限制为官方 server VAD 来源，非标准来源在兼容文档明确标注 |
| vad_speech_end | S→C | input_audio_buffer.speech_stopped（xtalk 扩展语义） | 保留现有后端触发与 origin，不限制为官方 server VAD 来源；仅通知前端状态，收到后不得再次发送 commit 或触发生成 |
| update_asr | S→C | conversation.item.input_audio_transcription.delta | delta 始终提供；xtalk.mode=append 时追加，replace 时用完整 delta 替换；xtalk.revision 单调递增 |
| update_resp | S→C | response.output_audio.playback.updated | 保留播放确认触发时机，发送完整已播放文本前缀和 response_id，不作为标准生成 delta 的来源 |
| finish_resp | S→C | response.output_audio.playback.done | 保留最终播放结算，发送完整实际播放文本和 response_id；不等同生成完成 |
| pause_tts/resume_tts | S→C | response.output_audio.playback.pause/resume | 瞬时控制客户端指定 Response/Item 播放队列 |
| stop_tts | S→C | response.output_audio.playback.stop | 清队列并停止播放，客户端随后发送 playback.stopped 停止增量回执；活动生成取消另发标准终态 |
| tts_playback_stopped | C→S | response.output_audio.playback.stopped | 原样保留 response_id 和未确认增量 played_audio_ms，按现有机制结算，不换算累计位置 |
| tts_chunk_played | C→S | response.output_audio.playback.chunk_played | 保留每块播放完确认，仅上报 response_id；不新增定期进度、块序号或累计时长字段 |
| tts_playback_finished | C→S | response.output_audio.playback.finished | 表示实际播放完成，不等于 response.done |
| change_voice / voice_changed（动态切换） | C→S / S→C | session.update/updated：session.audio.output.voice | 首次音频输出后仍允许修改是 xtalk 扩展；生效时机沿用现有 TTSVoiceChange 实现，成功后回显实际有效音色，不引入第二音色字段 |
| change_tts_speed | C→S / S→C | session.update/updated：session.audio.output.speed | 范围 0.25–1.5；允许活动回复期间更新是 xtalk 扩展，已处理音频不重写，后续尚未速度处理的音频按现有路径使用新值 |
| change_emotion / emotion_changed | C→S / S→C | session.update/updated：session.audio.output.emotion | 自定义音频输出配置，保留 name/vector；名称非空时优先于向量，生效沿用原 TTS 情绪更新机制 |
| tool_called | S→C | response.tool.called | xtalk 自定义 Response 前缀通知；保留 name/args 和后端执行，仅补统一 event_id，不强制添加 response_id/call_id，不要求客户端执行或提交 function_call_output |
| retrieval_updated | S→C | conversation.retrieval.updated | 完整快照，顶层 text/is_final；前端替换展示，现有工具结果路径 is_final=true，不使用 delta 或新增 done |
| caption_updated | S→C | conversation.caption.updated | 完整快照，顶层 text/is_final/reason；前端替换展示，不使用 delta 或新增 done，不冒充 ASR |
| thought_updated | S→C（仅前端接收能力） | conversation.thought.updated | text 为完整快照，替换 thought 展示；统一 event_id，不新增后端生产链路、delta/done 或关联 ID，不列为后端已支持输出 |
| speaker_updated | S→C | conversation.speaker.updated | 顶层保留 speaker_id/reason，仅替换包装并补 event_id；保留识别结果和发送时机，不强制关联 Item，不新增前端展示 |
| latency_metrics | S→C | session.latency.updated | 顶层保留 network_latency_ms/asr_latency_ms/llm_first_token_ms/llm_sentence_ms/tts_first_chunk_ms，单位毫秒，原计算和触发不变，不映射为 rate_limits.updated |
| queue_status/queue_granted | S→C | session.queue.updated/granted | updated 统一顶层 position/limit/active，granted 仅 type/event_id；保留 FIFO 和每 5 秒更新，获准后初始化完成再发 session.created |
| full_audio_frame | S→C | session.audio.mix.delta | 沿用 full_audio_frame 的触发、分块和混合机制，audio_base64 改名为 delta（仍是 Base64 音频块），保留 sample_rate/channels/format；不新增订阅、done 或时间位置字段，文件保存由外部负责 |

**移除、外置及前端内部动作**

| 当前事件/数据 | 方向 | 新事件或字段 | 转换行为 |
|---|---|---|---|
| attach_session | C→S | 移除；conversation.item.create 注入历史 | 新连接自动创建运行会话，业务历史由外部提供 |
| change_tts_model | C→S | Realtime API 不支持 | TTS 模型由服务端启动配置确定，不提供会话内切换字段或事件 |
| change_llm_model | C→S | Realtime API 不支持 | LLM 模型、Provider 地址及凭证由服务端配置，不提供会话内切换字段或事件 |
| ping/pong、clock_sync | C→S / S→C | Realtime API 首期不支持 | 不提供应用层时钟同步扩展或前端交互；收到对应旧事件或拟议 session.clock.* 返回明确不支持错误 |
| session_config.recording_path | C→S | 宿主配置接口 | 文件路径外置，核心不提供录音存储命令 |
| client_* | 前端内部 | 保留本地动作并转换对应协议事件 | 不直接暴露为网络协议 |

- 配置扩展的更新规则。
    - 情绪配置使用 session.audio.output.emotion={name, vector}，属于标准 Session 音频前缀下的自定义字段；不提供 session.xtalk.emotion。原有 name 非空时优先于 vector 的选择逻辑保留，两者均空时返回参数错误，不改变已有情绪；通过 session.updated 回显实际配置，生效时机沿用原模型更新实现；不新增 input_turn_mode、playback_reporting、audio_mix 或能力查询接口；不放入 service_config.realtime_api，后者仍仅包含 model 和 api_keys。
    - 未提供的配置不变，未知字段和能力不支持返回 error；session.updated 回显完整有效配置，不新增 xtalk.session.update/updated 事件。
    - 初始设置和动态换音色统一使用 session.audio.output.voice，不提供 session.xtalk.voice；首次音频输出后允许修改超出官方限制，归入扩展表，成功通过 session.updated 回显实际生效值，失败保留原音色。生效时机以原实现及迁移对照测试为准，不强制改为下一轮。
- 扩展瞬时事件的约束。
    - ASR 按 Item/content_index 关联；播放扩展保留现有 response_id，混合音频和工具通知保留原字段，不添加无现有用途的关联 ID；排队事件允许会话就绪前发送。
    - 服务端扩展事件必须有 event_id，客户端可提供 event_id；消费方不将 session.updated 的完整配置回显重复执行为一次性命令。
    - 标准 response.done 不等待播放确认；使用块确认的 xtalk 客户端沿用原已播放文本展示与结算机制。外部实际播放录音以客户端实际播放为准。
    - 不新增定期 progress、控制执行确认、对齐精度回显、播放配置开关、音频订阅或工具完成通知；仅标准 OpenAI Realtime 支持的能力允许作为新增功能。

**特别说明**

**OpenAI Realtime 风格的 API，包含标准子集和 xtalk 扩展，需要适配客户端**

| 差异 | 对标准客户端的影响 |
|---|---|
| 音频投递等待块播放确认 | 客户端不发送 response.output_audio.playback.chunk_played，可能阻塞后续音频投递 |
| ASR delta 支持 replace | 客户端若只追加 delta，会显示错误文本，需识别 xtalk.mode |
| commit 后可能自动回复 | 与标准手动模式不同；现有轮次机制可自动生成，不应额外请求回复 |
| 不支持 conversation.item.truncate | 标准打断后的上下文同步不可用，需使用 xtalk 块确认及停止增量回执 |
| 不支持 C→S response.create | 显式生成请求返回 unsupported_event；回复由现有输入和轮次机制触发，S→C response.created 保留 |

- `response.output_audio.playback.chunk_played` 保留现有音频块播放确认能力，对需要限制客户端音频积压的 xtalk 客户端具有必要性。
    - 网络发送成功只说明数据被传输层接受，不代表音频已播放；网络速度快于播放速度时，仅靠连接发送队列背压不能限制客户端未播放音频积压。
    - xtalk 客户端在每块实际播放完成后发送该扩展，后端沿用未确认音频预算控制投递；达到预算时等待块确认释放额度，从而限制已发送但未播放的音频量，并保留播放文本更新和打断结算功能。
    - 不新增周期进度事件或新的播放确认字段；该事件仍仅携带 response_id，按现有 FIFO 音频块完成机制处理。
    - 音频客户端必须适配原块确认机制；不提供无需播放确认的纯标准客户端投递路径，不新增 chunk_playback_ack 或模式协商字段。有界网络发送队列仅作为传输资源保护，不能替代播放预算。
    - 所有音频连接采用原播放预算，不探测是否支持确认，不等首个确认选择模式；未发送确认可能阻塞后续投递，客户端适配文档明确此要求。

- xtalk Realtime API 不支持标准 `conversation.item.truncate`。
    - 现有音频块播放确认与停止时未确认增量回执已满足 xtalk 的实际播放文本结算需求；保留 response.output_audio.playback.chunk_played/stopped/finished 及服务端播放文本 updated/done，不增加历史 Item 截断或累计位置换算功能。
    - 客户端应实现这些扩展以保留原打断上下文一致性；仅使用 OpenAI 标准事件的客户端不能通过 truncate 完成该功能，服务端收到时返回 unsupported_event，不静默忽略或假装成功。

- 动态音色沿用标准事件和字段，是修改约束的 xtalk 扩展。
    - OpenAI 标准在首次音频输出后锁定音色；xtalk 保留当前会话内动态换音色能力，继续接收 session.update 的 session.audio.output.voice 并通过 session.updated 回显。
    - 不新增音色事件或 session.xtalk.voice；验证首次输出前后修改及合成期间更新的实际生效行为与原实现一致，不把已经生成的音频重新合成。

- Realtime API 不提供内部 LLM/TTS 模型动态切换。
    - 原 change_llm_model/change_tts_model 不映射为扩展；不增加替代事件或 Session 字段，服务端配置仍可选择模型。
    - 测试覆盖旧切换事件及拟议的 session.xtalk.llm_model/tts_model 字段被明确拒绝，失败不影响当前模型、上下文和活动任务。

- 结构化情绪配置位于 session.audio.output.emotion，属于 xtalk 扩展，不是官方标准字段。
    - C→S 使用 session.update，S→C 使用 session.updated；不新增独立情绪事件，不限制为下一轮才生效，已经生成的音频不重写。
    - 验证名称、向量、两者同时提供、空参数和模型不支持情绪配置的场景；实际名称/向量维度约束沿用配置模型能力，不增加新的情绪类型。

- 速度采用标准 session.audio.output.speed 字段和 0.25–1.5 范围，活动回复期间可修改属于 xtalk 扩展。
    - 保留 current_speed 更新及后续音频速度处理路径；已生成且完成速度处理或已发送的音频不重新处理，不能承诺正在播放的块立即变速。
    - 非数值、非有限值和超出范围返回参数错误且保留原配置；旧 0.5 下限调整为 0.25，所有输入入口和模型能力校验同步，不通过旧 clamp 静默变为 0.5。
    - 测试覆盖 0.25、1.5 边界、越界、活动回复更新、分句/流式速度处理及处理后 PCM 时长与播放确认的一致性；已生成音频不被二次变速。

- 工具通知采用 xtalk 扩展 `response.tool.called`。
    - response 前缀仅表示能力归类，不保证关联某个标准 Response；现有事件缺少 response_id，不能推断或虚构归属。
    - 工具仍由后端执行，客户端只消费 name/args 展示，不重复执行；不映射为标准 function_call Item 或 response.function_call_arguments 事件，不新增工具完成/失败通知。
    - 测试验证事件字段及现有过滤规则（tool_call_result/direct_audio 不发送），工具执行次数和后续对话行为与原实现一致。

- 混合音频扩展 `session.audio.mix.delta` 的音频字段统一命名为 delta。
    - 默认模式的 delta 保持原 audio_base64 的 Base64 数据及分块内容，binary_pcm 模式改用 binary-audio-output.md 定义的二进制混合音频消息；sample_rate、channels 和 format 沿用原值；此事件不代表单个 Response 的 TTS 输出。
    - 迁移对照测试验证解码后的音频字节、分块顺序、采样率、声道和格式一致，不引入订阅、额外完成事件或时间位置字段。

- 排队扩展统一消息结构，不扩展排队功能。
    - session.queue.updated 在首次排队及原每 5 秒周期更新时发送，position、limit、active 统一位于顶层，不再混用 data 包装；每条消息有唯一 event_id。
    - session.queue.granted 仅包含 type 和 event_id，表示获得会话名额；随后初始化会话，成功后发送标准 session.created，不能把获得名额等同于会话就绪。
    - 保留 FIFO、断开/取消清理和名额释放机制，不新增等待超时、队列容量或新的配置项；对照测试覆盖首次/周期字段一致及事件顺序。

- 客户端讲话开始通知保留为 input_audio_buffer.speech_start（xtalk 自定义事件）。
    - C→S speech_start 与 S→C speech_started 方向及名称不同；前者调用原客户端 VAD 开始处理，后者保持后端状态通知，不造成通知回环。
    - 原可选 timestamp 仍按毫秒解析，缺省时不虚构客户端开始时间；网络延迟测量仅在已有所需时间数据时计算，不新增前端测量功能。

- 应用层时钟同步暂不纳入 Realtime API。
    - 不迁移 ping/pong/clock_sync，不提供 session.clock.ping/pong/sync，不新增前端时钟同步链路；与 WebSocket 传输层控制帧无关。
    - 保留内部 ASR/LLM/TTS 延迟观测和 session.latency.updated；无法计算的客户端网络延迟保持原缺失行为，不伪造测量值。原 speech_start 的可选 timestamp 支持不因此删除。
    - 此项为已批准的 API 能力排除，功能一致性测试不要求保留时钟同步端点。

- 延迟扩展 session.latency.updated 保留原五项指标。
    - 字段 network_latency_ms、asr_latency_ms、llm_first_token_ms、llm_sentence_ms、tts_first_chunk_ms 放在顶层，统一毫秒单位；仅替换旧 action/data 包装为 type/event_id，不新增指标或时钟同步流程。
    - 无有效客户端网络测量时沿用原输出值及缺失处理，不能将回退的零值解释为测得零延迟；文档明确这一限制。
    - 对照测试验证原计算结果、整数转换、消息触发时机与新字段一致。

- Caption 与检索展示扩展均采用完整快照。
    - conversation.caption.updated 保留 text、is_final、reason，conversation.retrieval.updated 保留 text、is_final，字段位于顶层；每次替换对应展示内容，不拼接多次结果。
    - 检索沿用本地/网页搜索工具完整结果的触发路径，现有实现 is_final=true；不新增流式检索、delta 或独立 done 事件。
    - 对照测试验证连续两次更新替换而非追加，空文本及最终状态沿用原网关行为。

- 说话人识别扩展 conversation.speaker.updated 保留原事件能力。
    - speaker_id 和 reason 位于顶层，沿用原识别结果与发送时机；只增加统一 event_id，不增加置信度、说话人列表或 Item 关联字段。
    - 当前前端没有对应处理器，本变更不新增展示功能；对照测试验证事件字段与原输出一致。

- Thought 扩展仅保留当前前端接收能力。
    - 前端处理器迁移为 conversation.thought.updated，读取顶层 text 完整快照并替换展示；消息契约包含统一 event_id。
    - 当前后端没有对应发送链路，不新增生产逻辑、推理内容、模型或提示词，不增加 delta/done 或关联 ID；支持矩阵注明仅存在前端处理器。
    - 测试注入消息验证替换行为，不以人工构造事件测试宣称后端具备该输出能力。

- Realtime API 不支持客户端 response.create。
    - 生成由现有文本/音频输入及轮次机制触发；客户端不得在 commit 后再发送 response.create，收到该事件明确报错，不创建第二次回复。
    - 标准 S→C response.created 和其他生成生命周期事件仍保留；移除仅依赖 response.create 的逐次输出模式覆盖参数，输出模式由 session.update 配置。
    - conversation.item.create 默认仅注入历史/上下文；实时用户文本在事件顶层显式提供 xtalk.generate_response=true，沿用原文本轮次流程自动回复，非 user 消息不允许该标记；create_response=false 时服务端音频轮次只转写并注入上下文，不自动生成；显式 xtalk.generate_response=true 的用户文本仍可触发回复，不宣称提供标准 response.create 手动生成能力。

- 纯声卡客户端不依赖客户端讲话通知。
    - 后端 server_vad 自动轮次必须在客户端只发送 append 时正常工作，保留 speech_start/commit 扩展输入路径但不将其作为所有音频会话的必需事件。
    - 验收使用可替换音频输入/输出边界：测试驱动器提供合成语音及静音、按实际音频时长消费输出，触发相同块完成及停止回执；不新增客户端 VAD、周期进度或另套网络实现。默认 CLI 仍为声卡对话。
    - sample_client.py 不新增重连、多 Session 或存储功能；缺少声卡/依赖、认证失败和连接关闭给出可操作的错误并清理资源。

- 内置及自定义事件采用同一注册机制，具体组装由 Service 负责。
    - Service 先通过 register_input_handler 注册内置事件，例如 type="input_audio_buffer.append"、handler=InputGateway._handle_audio_append；注册的是未绑定方法，网关执行时调用 await handler(current_gateway, message)，不会捕获原型或其他会话网关。
    - 输入网关接收注册配置副本，内部 _dispatch_message 按 message["type"] 查表执行；不反向导入 Service，也不自行调用 Service 注册方法。输出简单映射通过 register_output_handler 注册，复杂状态处理使用原 subscribe_event，由 Service 组装后交给当前会话 EventBus。
    - 注册配置随 Service.clone 复制；内置注册仅在原型初始化时建立，会话克隆复用配置而不重复追加订阅。后注册替换语义必须同时适用于内置和自定义项，输出替换不得遗留旧订阅导致重复发送。
    - 输入/输出事件 type 常量、消息形状及字段约束统一位于 serving/events.py；realtime/protocol.py 仅兼容导出这些定义并保留处理器类型契约，不重复定义形状，不导入 Service 或具体 Gateway。处理器注解需要引用 Gateway 时仅在 TYPE_CHECKING 中导入，并使用延迟注解。
    - 运行时依赖为 Service 到 Gateway 和协议类型、Gateway 到协议类型和内部 EventBus；Gateway 不导入 Service，realtime/__init__.py 不反向导入 Xtalk/Service，避免包初始化循环。类型定义仅单向引用基础内部 Event 类型，不引入框架组装逻辑。
- 公共和内部方法命名。
    - 仅框架内部使用的方法以下划线开头，例如 _register_builtin_handlers、_dispatch_message、_handle_audio_append、_build_audio_payload 和 _close_connection；不因内部注册引用而将这些方法公开。
    - 用户调用的 register_input_handler、register_output_handler、subscribe_event、unsubscribe_event、send_signal 和 create_asgi_app 保持无下划线；所有跨模块引用的类型/函数仍按仓库要求添加 NumPy 风格 docstring，包括私有但跨模块使用的处理方法。
    - 测试覆盖未绑定内置处理器使用当前网关、注册替换不重复执行、克隆状态隔离及公共/私有入口边界；不新增网关注入或第二套注册总线。

- Realtime 传输采用原生 ASGI 适配，具体框架由宿主决定。
    - Connection 不公开 send_text，结构化控制事件统一使用 send_json，由连接内部序列化；不为已不支持的应用层 ping/pong 保留单独文本发送入口。Connection 接口及 Service/Gateway 不引用 FastAPI 类型，events.py 统一定义消息形状与约束，protocol.py 保留兼容导出；发送队列与断线任务唯一归 connection，原网关不再重复启动另一套收发/关闭任务。
    - FastAPI 挂载方式和生命周期驱动示例以 asgi-app-entrypoint.md 为准；默认协议及块播放确认要求不因宿主框架不同而改变。

- Item 与 Response 标识融入现有事件系统。
    - response_id 保留现有生成、TTS、播放及取消关联用途；音频输入 item_id 由 TurnTakingManager 在 _turn_open 从 false 变为 true、分配新 turn_id 时按 item_input_{session_id}_{turn_id} 创建一次；同轮多个 segment 复用，结束或中止后下一轮创建新 ID。TurnASRStartRequested/TurnASREndRequested 携带该 ID，ASRManager 保存并原样写入 Partial/Final/Failed，OutputGateway 仅使用事件携带的 ID，不生成或回退推导输入 ID；ASR 部分、最终、提交和通知复用该 ID。客户端已提供的有效 Item ID 在历史/文本注入入口沿用，不重新分配。
    - 当前输出一回复一 Item 时，输出 item_id 按固定规则由 response_id 确定性生成，不新增映射字典；后续实际出现多 Item 需求再扩展，不提前设计。
    - 仅在关联输入/输出内容的内部事件增加 item_id，基础 Event 不统一附加；生成、合成和播放事件沿链携带同一 response_id/item_id，取消或迟到事件不得换成当前新轮次 ID。
    - ASR 预览、append/replace 和 revision 放在现有 ASRManager，生成文本 delta 放在现有 llm_agent_generation_manager；不新增独立预览/文本 Manager，OutputGateway 只构造消息，不重复计算状态。
    - 删除拟议 conversation.py；历史上下文仍由现有 Agent/上下文 Manager 管理，输入缓冲由原输入/ASR 管线持有，不为协议创建另一份完整历史。
    - 测试覆盖连续轮次、ASR 修订、输出音频/转写、停止结算与迟到结果的 ID 一致性及隔离；protocol 类型不依赖具体 Manager。

- frontend 客户端库与示例鉴权分离。
    - createSession(url) 不执行登录流程；可直接连接默认无 API Key 的 xtalk Realtime 服务。示例前端登录后在 URL 中附加业务 access_token，示例后端先验证再转交 ASGI 应用。
    - 业务登录和历史/上传接口不属于 Realtime API，客户端库不提供 business 配置；具体迁移、URL 凭证脱敏及测试以 asgi-app-entrypoint.md 为准。

- 现有 frontend 麦克风模式的本地 VAD/Enhancer 行为保留。
    - 默认两者开启，inputConfig 显式开关及四种组合按 asgi-app-entrypoint.md 的行为契约执行；Realtime 消息转换不改变原始/增强音频上传路径，不新增动态开关。
    - sample_client.py 无本地 VAD/enhancer 的设备场景独立成立，不据此删除浏览器客户端现有处理能力。

- 前端协议类型集中放在 frontend/src/action-handler-functions/protocol.ts，原处理器 types.ts 改名 handler-types.ts，沿用现有 SDK 分层，不新增 frontend/src/realtime/ 运行时或分发体系；具体源码树和业务迁出范围以 ASGI 迁移 spec 为准。

- 前端 runtime 不依赖具体 Realtime 事件类型，协议解析/构造及识别由 ActionHandler 和现有处理器承担；不拆前端 protocol.ts，不新增客户端/服务端类型目录。详细依赖与文件迁移以 ASGI spec 为准。


**实现验证记录（2026-10-15）**

- 原协议快照基线：从提交 `88d877a2336bac0a05d1bc13699857aaa1792952` 的原 OutputGateway 实际执行提取 7 类输出，保存于 `tests/fixtures/migration/gateway.json`；`tests/test_migration_behavior.py` 对照旧 data 与新 Realtime 载荷，记录源文件与提交，不由新实现生成期望。
- 自动验证：核心协议、前缀挂载、独占生命周期、活动连接关闭、慢 ASR 的 commit 顺序、网关注册、示例宿主身份/历史、知识库重载和录音样本测试；桌面业务链见 `app/tests/integration/test_migration_parity.py`；前端协议与示例业务编排见 `frontend/tests/`。
- 实际模型配置：`server_configs/sii.json`，先导入 `server_configs/modelset_clients.py` 注册模型；不在记录中写入推理凭证。服务使用本机临时端口 18765，验收结束关闭。

```sh
python3 -m server_configs.modelset_server --port 18765
python3 examples/sample_client.py --base-url http://127.0.0.1:18765 --model xtalk --input-wav /tmp/xtalk-realtime-input.wav --no-speaker --rounds 2 --timeout 90
python3 examples/sample_client.py --base-url http://127.0.0.1:18765 --model xtalk --input-wav /tmp/xtalk-realtime-input.wav --interrupt-wav /tmp/xtalk-realtime-interrupt.wav --no-speaker --rounds 2 --timeout 90 --interrupt-after .1
```

- 真实验收结果：连续两轮音频输入/输出及逐块播放确认通过；明确的第二问题插话触发停播、增量停止结算及新轮回复，通过 sample_client 的 `Verified playback interruption` 检查。客户端未运行本地 VAD/enhancer，无真实声卡。
- 鉴权验收：临时配置 api_keys 后，sample_client 携带测试 Key 的两轮对话通过；缺失或错误 Key 被拒绝。另验证 `turn_detection: null` 下手动 commit 后完整转写及文本回复。
- 本机验收日志：`/tmp/xtalk-real-latest-two-rounds.txt`、`/tmp/xtalk-real-latest-interruption.txt`、`/tmp/xtalk-real-final-auth.txt`；音频及测试凭证均为临时材料，不加入仓库。Python 3.14 环境会输出既有 Pydantic V1/asyncio 弃用警告，测试通过不代表已消除这些兼容警告。


**输入网关职责重构（2026-10-16）**

- 事件和约束的唯一事实源为 serving/events.py。
    - 协议消息形状、字段必填性、枚举及数值范围随类型定义；运行时严格校验不将字符串或布尔值隐式转为数值，拒绝非有限数。
    - 部分会话配置从完整配置类型派生，省略字段保持原值，显式 null 仅在类型允许时接受；不再另写重复的字段范围判断。
    - JSON 在 InputGateway 边界提取、校验并转换为内部事件；形状或范围错误不分发，不改变有效状态。
- InputGateway 保留事件分发、组合请求执行顺序及配置更新协调，不新增会话协调层。
    - VAD 阈值、TTS 音色/情绪/速度、Agent instructions 和上下文写入由对应现有 Manager 消费明确类型的请求事件。
    - 全部组合参数构造校验及能力预检成功后才应用；只应用实际变化字段，逐项保存已成功生效的快照，不承诺跨模型事务回滚；执行失败不发送 session.updated。
    - EventBus 新增单消费者 request 路径，等待处理完成并向调用者传播异常；缺失或重复处理者明确拒绝，不改变 publish 的既有异常处理语义。
    - 输入网关不读取输出网关的私有响应表或调用其私有终态方法；取消由内部请求协调已有生成停止链路，输出层负责协议终态去重。
- 错误采用 Realtime error 事件，连接继续接收后续请求。
    - 校验失败返回 invalid_request_error，携带字段 param 与能够提取的客户端 event_id；服务端错误事件拥有独立 event_id。
    - 未预期执行异常记录日志并返回不暴露内部细节的 server_error；不把任意内部异常解释为参数错误。
- 变更文件与验证。
    - 修改 serving/events.py、event_bus.py、modules/input_gateway.py、output_gateway.py、vad_manager.py、tts_manager.py、llm_agent_context_manager.py 及 realtime/protocol.py；显式声明校验库依赖。
    - 覆盖类型/范围/缺省/null、组合校验原子性、缺失处理者、业务拒绝传播、部分执行失败快照、无变化配置及错误后连接可继续使用；运行现有 Realtime 与相关 Manager 回归测试，同步中英文技术文档。
    - 本重构不修改 LLM 或深度学习提示词，不改变客户端协议能力范围。

**输入网关重构验证记录（2026-10-16）**

- `kkspec validate`：0 个问题。
- 自动回归：`python3 -m pytest` 运行完整 tests/realtime、VAD 批处理、ASGI、事件分发、持久生成流、TTS 播放对齐、文本输入及多说话人上下文测试，共 125 项通过，无失败或跳过。
- 静态检查：变更源码及新增边界测试通过 Ruff E4/E7/E9/F/I 检查；`git diff --check` 通过。
- 本轮为协议边界及职责重构，未重新执行实际模型端到端验收；此前实际验收记录保留，不将本轮替身测试表述为真实模型验收。
