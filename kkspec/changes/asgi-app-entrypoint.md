2026-10-14

# 需求变更

- 开发者需要将 xtalk 服务接入不同后端框架，而不必让 SDK 操作宿主框架的路由对象。
    - 可以独立部署 xtalk 应用。
    - 可以在已有支持应用挂载的异步服务中接入 xtalk。
- xtalk 仅负责对话能力，业务用户鉴权、历史持久化和业务文件管理由外部应用承担；API Key 访问控制由 Realtime spec 定义为可选且默认关闭；当前客户端的业务接口由宿主接管。
- 已有集成需要明确迁移步骤，保留宿主的接口、安全策略和资源管理行为。

# spec变更

- SDK 应用入口。
    - 新增 `Xtalk.create_asgi_app()`，返回可调用的 ASGI 应用；无需传入宿主应用，不启动服务器、不监听端口、不重新加载模型。
    - 返回类型使用 ASGI scope/receive/send 标准协议，支持 HTTP、WebSocket 和 lifespan；公共签名及文档不要求使用 FastAPI 类型，SDK 内部直接实现 ASGI 协议，不依赖 FastAPI 或其他具体后端框架。
    - SDK 不新增 `create_app()` 别名；同一个 Xtalk 实例只创建一个应用，首次调用缓存应用，再次调用返回同一对象，避免重复路由和资源所有权。
    - 删除公共 `mount_routes()`，仓库内所有调用迁移为创建应用及宿主挂载；这是 SDK 破坏性入口变更，发行说明明确替换方式，不静默改变旧方法语义。
    - 删除公共 `connect()`，连接处理收敛为 ASGI 应用内部逻辑；并发准入、消息循环、取消和清理仍由内部实现负责，不增加框架专用连接入口。
- 路径与协议兼容。
    - 返回应用默认包含 Realtime WebSocket 和模型列表接口；登录、会话列表、详情和上传迁到宿主。WebSocket 固定为 `/v1/realtime`，使用 realtime-api.md 定义的事件协议；移除旧 `/ws`，不在新路径承载旧 action 协议。
    - 端点路径相对子应用根路径解释；挂载在 `/xtalk` 时WebSocket 地址为 `/xtalk/v1/realtime?model=<model>`，宿主业务 HTTP 地址由宿主配置；SDK 不硬编码宿主前缀，正确处理 ASGI root_path。
    - 移除 ws_path 及其他路由路径参数；路由路径固定为 `/v1/realtime`，`model` 是连接查询参数而非路径的一部分，值匹配 `service_config.realtime_api.model`；省略 model 时使用配置模型名（缺省 xtalk），未知 model 在创建会话前拒绝连接，具体错误由 Realtime spec 定义。
    - 宿主在挂载子应用之前注册自身 API、静态页面和健康检查；在根路径末尾挂载时客户端使用 `/v1/realtime?model=<model>`，根挂载之后的宿主路由不保证可访问，示例不得依赖这种顺序。
    - 子应用不默认增加宿主级 CORS、桌面启动令牌、健康检查或根静态页面；原安全中间件继续放在宿主，必须同时覆盖子应用 HTTP 和 WebSocket 请求。
- 对话与存储职责。
    - xtalk 不自动创建或写入聊天数据库、会话知识目录、持久化向量库或录音文件；用户历史、知识入库、文件上传和录音保存全部由宿主提供。
    - xtalk 保留对话内检索工具调用、结果注入和检索状态输出；宿主管理知识资源及访问权限，核心不管理知识库归属或存储生命周期。
    - xtalk 保留对话所需的音频重采样、缓冲、播放位置及时间轴处理；宿主根据实际采集和播放数据保存录音，不能将生成完成视为播放完成。
    - 模型配置与加载、并发限制、连接取消与清理、日志和延迟指标仍由 xtalk 负责；显式模型缓存与运行日志不属于本次移除的业务持久化。
    - 迁移后宿主上传、检索和录音功能保持可用；正常启动及连续文本/语音对话不得自动生成业务数据库、向量目录或录音文件。
- 迁移前后功能一致性。
    - 为迁移前的实际功能建立可重复基线，使用固定模型替身、输入及临时时间/存储环境；迁移后通过宿主与 xtalk 的完整集成运行相同场景，对比用户可见结果与业务状态。
    - 对照覆盖登录身份与权限隔离、会话列表/详情及消息保存、历史恢复后的模型上下文、文件上传及知识检索结果、录音保存及打断处理、文本/语音对话和资源释放。
    - 已批准的公共入口删除、端点及事件协议变更、业务职责迁移、可选 API Key 默认关闭不要求接口字节一致；其余可见功能不得因迁移丢失，移到宿主的功能必须通过集成测试证明保留。
- 前端本地 VAD 与 Enhancer 行为保持。
    - createSession 的 inputConfig.enableVAD 和 enableEnhancer 保留，麦克风模式未指定时两者默认 true；作为初始化配置，不新增运行中开关 API。
    - enableVAD=true 时持续上传原始音频，同时本地检测讲话边界；enableEnhancer=true 时增强音频用于本地 VAD 检测链路，不将上传路径静默改为增强音频。
    - enableVAD=false 且 enableEnhancer=true 时上传增强音频，不产生本地讲话开始/结束；两者 false 时上传原始音频，不产生本地讲话边界。客户端关闭本地 VAD 时，后端按已配置的 VAD/轮次能力处理，不伪造客户端通知。
    - Enhancer 加载或推理失败沿用现有降级关闭并继续处理音频；muted 保留停止输入帧处理的语义，不作为 VAD/Enhancer 配置开关。
    - 浏览器模型、缓存、推理和设备操作继续限定在 frontend/src/platforms；验证四种开关组合、初始化默认值、失败降级及静音行为，按迁移前实际音频路径进行对照。
- 生命周期契约。
    - 不公开 `Xtalk.asgi_lifespan()` 或替代的框架专用生命周期方法；生命周期统一由 create_asgi_app 返回应用的 ASGI lifespan 协议管理，内部资源管理保持私有。独立部署由 ASGI 服务器驱动，挂载场景由宿主显式发送并等待 lifespan.startup/complete 和 lifespan.shutdown/complete，不假设普通挂载自动转发。
    - create_asgi_app 仅构造路由；服务开始接受请求前必须完成 ASGI lifespan 启动，未启动返回 HTTP 503 或关闭 WebSocket，不隐式按请求启动后台资源。
    - 宿主转发失败时启动失败，不将未启动的子应用标记为 ready；shutdown 必须等待完成后再关闭宿主事件循环，驱动器不依赖子应用的具体框架类型。
    - 同一实例生命周期只允许一个活动所有者；嵌套或重复启动报错，防止一个宿主退出时关闭另一个仍在运行的集成。
    - 正常关闭先停止接收新会话，再取消并等待活动会话任务、关闭连接及释放会话资源；复用现有模型释放约定，不关闭仍由外部持有的模型资源；初始化失败也清理已经启动的任务。
    - 应用实例关闭后不得重新启动；需要重启时创建新 Xtalk 实例。宿主使用者负责在自身 lifespan 中驱动子应用的 ASGI lifespan，SDK 不覆盖宿主已配置的生命周期。
- 交付边界。
    - 本变更实现对话服务的 ASGI 应用入口及鉴权/持久化分离，Realtime 事件由 realtime-api.md 实现，两份变更共同交付可用新端点；不保留旧 /ws 过渡端点，不提供短期凭证接口。
    - `realtime-api.md` 的 SDK 入口设计以本 spec 为准：其 create_app 替换为 create_asgi_app，不新增 mount_realtime_routes/realtime_lifespan；该子应用默认提供 Realtime 路由，service_config.realtime_api 仅配置模型名与可选 API Key 列表；不提供 Xtalk.run() 或 xtalk[server]，部署框架与服务器由使用者选择。
    - Realtime spec 同步采用本 spec 的入口和职责边界；Realtime 事件及历史 Item 注入由 realtime-api.md 实现。
    - 支持范围是直接 ASGI 部署及宿主提供的 ASGI 挂载；不声称可直接挂载到 WSGI 应用，不新增 Flask/Django 专用适配器。

# 设计变更

```text
src/xtalk/
├── c api.py                            # create_asgi_app、固定路由与框架适配、私有生命周期，移除旧公共及业务接口
├── - auth.py                           # 用户 JWT 鉴权迁出核心
├── - persistence.py                    # 业务历史存储迁出核心
├── c serving/interfaces.py             # 不依赖框架的窄连接协议
├── c serving/service.py                # 不装配历史保存、知识入库及录音文件写入
├── - serving/modules/embeddings_manager.py # 知识入库迁移到宿主 knowledge_store.py
├── c serving/modules/recording_manager.py # 移除自动文件写入，按实际调用保留音频处理
├── c serving/modules/retrieval_manager.py # 保留检索工具结果进入对话的逻辑
├── - serving/modules/persistence_manager.py # 历史保存迁出核心
└── c serving/service_manager.py        # 临时会话创建与统一关闭，不依赖用户和数据库
frontend/src/
├── c index.ts                         # 保留小型 SDK 公共入口，移除业务导出
├── c conversation.ts                  # 当前对话状态、Item 关联及播放展示
├── c action-handler.ts                # 保留唯一事件分发器，改按 type 分发
├── c session/
│   ├── c create.ts                    # 纯对话编排与公开操作
│   ├── c runtime.ts                   # WebSocket/音频生命周期与配置握手
│   ├── c types.ts                     # 对话 API 类型，不包含业务服务配置
│   ├── - auth.ts                      # 登录迁到应用
│   ├── - api.ts                       # 列表/历史/上传迁到应用
│   └── - snapshot.ts                  # 业务持久化及恢复策略迁到应用
├── c action-handler-functions/
│   ├── c index.ts                     # 原有处理器注册入口
│   ├── + protocol.ts                  # 全部网络事件形状与共用协议类型
│   ├── + handler-types.ts             # 从 types.ts 迁移处理器签名与映射
│   ├── - types.ts                     # 改名，区分协议与处理器契约
│   ├── c session.ts                   # Session 配置和运行标识
│   ├── c queue.ts                     # 沿用现有排队处理模块
│   ├── c messages.ts                  # ASR 修订与生成文本
│   ├── c output.ts                    # 播放控制及生成音频终态
│   ├── c input.ts                     # 后端讲话通知与混合音频
│   ├── c client.ts                    # 本地音频/VAD 回调转网络消息
│   ├── c client-operations.ts         # 文本、音色及其他既有操作
│   ├── c meta.ts                      # 完整快照和工具展示
│   ├── c latency.ts                   # 延迟指标
│   └── c utils.ts                     # 复用状态转换，不建第二分发层
├── c websocket.ts                     # 保留平台工厂薄入口
├── c bases/websocket.ts               # 框架无关传输契约及 JSON 发送
├── c bases/audio-session.ts           # 原音频生命周期/回调契约
├── - http.ts                          # 业务迁出后删除无消费者抽象
├── - persistence.ts                   # 业务持久化迁出
├── - bases/http.ts                    # 业务 HTTP 契约迁出
├── - bases/persistence.ts             # 业务存储契约迁出
└── c platforms/
    ├── c index.ts                     # 平台装配，移除业务 HTTP/存储工厂
    ├── c web.ts                       # WebSocket 凭证、采样率；保留 VAD/enhancer
    └── c web-audio-bridge.ts           # 保留桥接输入，适配音频契约
app/backend/
├── c xtalk_adapter.py                  # 创建 SDK 子应用，替换旧路由注册调用
├── c runtime.py                        # 组合生命周期、安全策略及业务路由
├── c desktop_service.py                # 匿名登录和存储业务迁出 SDK 子类
├── + conversation_host.py              # 桌面业务路由与 Realtime 连接组合
├── + user_auth.py                      # 桌面业务 JWT
├── + conversation_store.py             # 桌面侧业务历史存储
├── + knowledge_store.py                # 宿主上传、向量入库及知识资源管理
└── + recording_store.py                # 宿主音频文件保存
examples/
├── + sample_app/static/js/business-api.js # 示例登录、列表/详情和上传
├── + sample_app/static/js/conversation-store.js # 业务历史/快照和恢复编排
├── c sample_app/static/js/index.js     # 组合业务模块和纯对话 Session
├── c sample_app/configurable_server.py # 保留宿主页面/静态路由后挂载子应用
├── c sample_app/custom_service.py      # 迁移自定义服务集成
├── c sample_app/custom_model.py        # 迁移自定义模型集成
├── c sample_app/custom_async_tool.py   # 迁移异步工具集成
├── c sample_app/mental_consultant_server.py # 迁移咨询服务集成
└── c bot2bot/server.py                 # 动态子应用创建、启动及停止的显式资源管理
```

- 应用构造与框架边界。
    - 采用 realtime-api.md 的单一运行时结构，原 serving/modules/input_gateway.py 和 output_gateway.py 原位改为 Realtime 网关，protocol.py 仅定义外部事件形状，具体逻辑由网关实现；不新增 Session/Response 状态引擎或公共网关注入能力。保留公共 send_signal 和原事件订阅，新增仅命名参数的输入/输出事件注册便利接口由 Realtime spec 定义。两份 spec 各负责协议与宿主迁移，不重复定义同一模块实现。
    - api.py 只向开发者暴露返回 ASGI 应用的 create_asgi_app 入口；固定路由和 ASGI 生命周期放在 api.py 的私有 ASGI callable，连接收发由 realtime/connection.py 适配 ASGI receive/send，不新增 asgi.py，只调用核心内部对话能力；DesktopXtalk 的匿名登录覆写迁到宿主，不再覆写已删除业务接口。
    - ASGI 类型协议不新增依赖；从 SDK 运行依赖中移除 FastAPI；桌面应用及示例需要的框架由各自依赖声明，宿主框架和服务器由使用者选择。
    - 路由构造函数为模块私有，外部只调用 create_asgi_app；应用不管理宿主 app.state，不从宿主提取模型或凭证。
- 业务职责分离。
    - 核心不创建数据库、不签发或验证业务用户凭证；仅保留 Realtime spec 定义的可选 API Key 校验。废弃的用户鉴权及持久化配置报迁移错误，不静默生效。保留模型所需的内存上下文，不改变显式模型资源接口。
    - 不新增数据库读取/保存回调；Realtime 历史通过 Item 注入，外部消费协议事件保存完成、取消和播放结算结果。
    - 桌面侧及示例宿主接管原登录、会话列表、详情、上传与存储逻辑，并提供自己的业务会话 ID；示例及桌面应用前端请求这些宿主业务端点，frontend 库不自动调用，但实时连接同步迁移为 Realtime 事件及 Item 历史注入，不能只换 URL 保留旧 action 协议。
    - 宿主中间件或网关在创建对话前验证连接权限，独立部署由外部包装层负责；xtalk 不接收 user_id，也不通过客户端 session_id 判断用户权限。
- 知识与音频存储边界。
    - 文件上传、知识切分与入库编排、向量库目录创建和持久化由宿主负责；移除核心按 session_id/data_dir 自动创建知识目录和调用持久化向量库的路径，不影响可单独使用的模型计算接口。
    - 对话中的检索工具调用、结果注入和检索状态事件保留；宿主配置工具及其知识资源，核心不维护知识库归属或生命周期，不新增未被调用的存储 Provider 抽象。
    - 录音文件创建、写入和关闭由宿主负责；音频重采样、缓冲、播放位置和时间轴处理按对话实际需求保留，停止自动写录音及最终汇总文件。宿主需要录音时使用客户端实际采集/播放的音频及时间信息，不以生成结束替代播放结束，也不新增尚无消费者的音频协议扩展。
    - 模型配置与加载、会话并发限制、连接取消与清理、日志及延迟指标留在 xtalk；普通运行日志与模型缓存不属于业务历史/知识/录音存储，不因本变更删除。
    - 桌面及示例现有上传/录音功能迁到宿主后保持可用；宿主存储失败由宿主返回或记录，不伪装成核心生成失败，迁移测试覆盖这些实际调用链。
- 调用方迁移。
    - 独立部署示例采用 `app = Xtalk.from_config("config.json").create_asgi_app()`，由 Uvicorn 驱动完整应用 lifespan。
    - 宿主挂载示例先创建 xtalk_app，在宿主 lifespan 中通过 ASGI lifespan 驱动器启动并关闭 xtalk_app（文档示例使用 asgi-lifespan 的 LifespanManager，作为集成方可选依赖，不新增 SDK 公共封装），最后执行 `app.mount("/", xtalk_app)`；自定义前缀示例改为 `/xtalk` 并展示完整客户端 URL。
    - 桌面侧保留原安全中间件、宿主业务 HTTP 路径和 readiness 响应，实时地址改为 /v1/realtime?model=<model>，准备好子应用后才置 ready；应用生命周期结束时关闭 xtalk 会话，冻结打包依赖检查覆盖新增 realtime 包。
    - Bot2Bot 按运行管理动态创建的 Xtalk 实例，启动时显式驱动每个应用的 ASGI lifespan，停止时等待 shutdown 完成；不假设运行中动态挂载触发服务器 lifespan，不遗留旧运行的连接任务。
- 验证。
    - 新增迁移前基线 fixtures 和 `tests/test_migration_behavior.py`（核心对话对照）及 `app/tests/integration/test_migration_parity.py`（宿主业务对照）；fixtures 记录确定性期望及原实现来源，先在修改前验证，再在修改后复用，避免仅由新实现生成期望。
    - 对照测试比较规范化的消息、转写、检索结果、录音样本/时间轴和上下文；忽略随机 ID 与真实耗时，仍验证 ID 关联、事件相对顺序、取消/播放结算及所有权边界，不能用忽略字段掩盖行为差异。
    - 迁移前后各执行同一组场景并记录结果；出现差异必须修复或对应到 spec 明确批准的变更，不以快照直接更新替代验证；不调用真实模型或生产存储。

    - 新增 `tests/test_asgi_app.py`，使用模型替身验证独立 lifespan、业务 HTTP 路由不存在和 固定 WebSocket 路径、拒绝 ws_path 参数和省略 model 的默认值和未知 model、缓存构造、重复启动及启动失败清理。
    - 新增 `tests/test_asgi_mount.py`，通过非 FastAPI 的 ASGI 宿主组合验证公共入口无宿主类型要求，以及根路径/前缀挂载、未启动拒绝请求和关闭活动连接。
    - 更新 `app/tests/unit/test_xtalk_adapter.py` 与 `test_runtime.py` 的替身及断言，验证宿主接口不被根挂载吞掉、安全中间件保护子应用，以及外部匿名身份和 readiness 行为。
    - 检查所有 mount_routes/connect 公共调用迁移完毕；鉴权、上传和历史存储测试迁到外部应用，保留桌面冻结服务相关测试，按影响范围运行，不调用真实模型服务。
    - 增加核心无业务落盘验证：正常启动及两轮文本/语音对话不会自动创建历史数据库、会话向量目录或录音文件；宿主知识上传/检索和录音写入另测，使用模型替身且允许模型自身显式配置的缓存。
- 文档与提示词。
    - 同步修改 `docs/tutorial/start_the_service.md/.zh.md`、`customize_the_service.md/.zh.md`、`introduce_a_new_model.md/.zh.md`，展示独立部署与生命周期组合、知识/录音存储迁移边界；重新生成 `docs/api/server/xtalk/index.md/.zh.md` 的 SDK API 文档。
    - 在启动教程明确说明挂载路径、路由顺序、宿主生命周期责任及 WSGI 限制，并说明 mount_routes/connect 删除、业务接口外置和实时协议及连接地址迁移的破坏性迁移。
    - 本变更不增加或修改任何 LLM 或深度学习提示词。

- 原生 ASGI 服务边界。
    - create_asgi_app 返回支持 scope/receive/send 的应用，HTTP 路由、WebSocket 握手、查询参数、可选 API Key 及 lifespan 使用 ASGI 标准消息实现；不返回或要求 FastAPI 对象。
    - realtime/connection.py 将 ASGI receive/send 适配为 serving/interfaces.py 的窄连接协议，提供握手接受、消息接收、JSON 发送和关闭，不公开 send_text；JSON 由连接内部序列化，二进制音频输出增加的 send_bytes 及共用队列行为以 binary-audio-output.md 为准；断开以统一内部结果表达，不能向运行时传播 FastAPI WebSocketDisconnect。
    - Service、ServiceManager、SessionLimiter 和 Gateway 移除 FastAPI WebSocket 类型及导入，统一依赖连接协议；输入/输出处理逻辑仍在原网关，connection 只承担传输任务、发送队列和关闭。
    - SDK 不安装服务器或框架；pyproject.toml 移除 FastAPI 运行依赖，宿主应用补齐自己的 FastAPI 依赖，验证纯 SDK 环境无需 FastAPI 也能导入及运行 ASGI 应用。
- FastAPI 宿主集成示例。
    - 宿主安装 fastapi、uvicorn、asgi-lifespan，在自身 lifespan 中通过 LifespanManager 显式启动和关闭子应用；挂载本身不能替代子应用生命周期。
    - 宿主先注册业务和健康路由，再在根路径挂载 xtalk ASGI 应用；prefix 挂载按 root_path 处理。根挂载连接路径为 /v1/realtime?model=xtalk，前缀 /xtalk 挂载路径为 /xtalk/v1/realtime?model=xtalk。

```python
from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from xtalk import Xtalk

xtalk_app = Xtalk.from_config("config.json").create_asgi_app()

@asynccontextmanager
async def lifespan(app):
    async with LifespanManager(xtalk_app) as manager:
        yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok"}

app.mount("/", xtalk_app)
# 使用者启动：uvicorn server:app --host 0.0.0.0 --port 8000
```

- ASGI 解耦验证。
    - 验证 SDK 源码无 FastAPI 导入、无框架异常泄漏；通过直接 ASGI 测试及 FastAPI 宿主挂载测试验证 HTTP、WebSocket、握手拒绝、root_path、生命周期和清理。
    - 示例代码和新增接口补齐 NumPy 风格文档；asgi-lifespan 是宿主可选集成工具，不作为 SDK 依赖或公共 API。

- 前端连接和身份配置。
    - SessionConfig 增加 model（缺省 xtalk）和可选 Realtime 连接凭证配置，宿主业务 serviceURLs/登录令牌与之分开；SDK 不把宿主登录 token 自动当作 API Key，不将 Key 写入 URL、日志或持久化快照。
    - createSession 继续以外部指定 WebSocket URL 为连接目标，不静默覆盖宿主挂载前缀；调用示例使用 /v1/realtime?model=xtalk，缺省 model 由配置补入查询参数，平台 URL 和浏览器子协议构造限定在 platforms。
    - open 连接后处理可能先到达的 queue.updated/granted，等待 session.created，再发送 session.update 并等待配置生效，随后开启音频；移除 attach_session/session_attached 和必须先登录才能对话的限制。
    - frontend 客户端库只提供对话连接和状态，不提供 business/serviceURLs 登录、历史或上传配置；业务应用前端自行完成这些 HTTP 调用及业务会话管理，再把连接 URL 交给 createSession。
- 前端音频和事件迁移。
    - 保留当前 ActionHandler 分发结构，网络消息由 action/data 改为 type 与顶层字段，action-handler-functions/protocol.ts 仅定义形状；client_* 仍为本地动作，不作为网络事件直接发送。
    - 输入不论 VAD 状态持续发送 Base64 append；本地讲话开始映射 speech_start、结束映射 commit，不发送 response.create；后端 speech_started/stopped 仅更新状态，不回发开始/结束事件。
    - 默认音频配置与协商格式对齐 24 kHz PCM16 单声道，平台采集/播放必要时转换；默认 Base64 编解码复用平台编码抽象；显式二进制输出协商及解析以 binary-audio-output.md 为准，不在 session 或处理器中引入浏览器实现。
    - 输出每条 audio.delta 解码后保持原块边界，沿用实际块完成、全部播放完成及 stop 未确认增量回调，分别发送 playback.chunk_played/finished/stopped；不新增周期进度、累计位置或 truncate。
    - content_part.added 及关联 ID 建立播放流，audio.done 仅标记后端无更多音频；暂停/恢复/停止扩展调用原音频 API，接收与播放并发，不因等待播放完成阻塞网络接收。
- 前端状态和展示。
    - 用户消息按 item_id/content_index 关联；ASR append/replace 与 revision 由后端提供，前端只应用且忽略过期预览，completed 后不接受迟到预览。
    - assistant 保存生成内容与原播放展示内容的独立状态，标准 delta 追加，done 确认最终内容；playback.updated/done 按 response_id 替换完整已播放文本。保留原随播放展示能力，不由 response.done 提前结束 speaking 状态。
    - caption/retrieval/thought 继续完整快照替换，工具通知保留 name/args；混合音频读取 delta 并调用原音频观察回调；speaker 不新增原本没有的展示，未知扩展不因缺少 handler 导致连接失败。
    - error 读取结构化对象，保持原用户提示及待完成消息结算；只在相关生成终态时处理对应回复，队列/参数错误不能一律关闭全部对话。
- 前端业务恢复和待定契约。
    - 登录、列表/详情/上传及快照逻辑迁到 examples/sample_app 的前端业务模块，桌面应用自行管理；业务恢复和重连策略由应用组合，frontend 只提供连接、关闭及必要状态通知，旧连接音频和等待器必须清理。
    - conversation.item.create 默认只注入上下文；实时 user 文本使用已批准的事件顶层 xtalk.generate_response=true 沿用原自动轮次，不发送 response.create。
    - 不为了网络协议新增前端说话人展示、时钟同步、模型切换或音频订阅能力；公共类型/函数按 TypeDoc 文档要求维护。
- 前端验收。
    - 补充前端协议回归测试，覆盖 created/updated/排队顺序、凭证与业务身份分离、ASR 修订/最终、ID 关联、生成与播放终态分离、元信息快照及未知扩展处理。
    - 音频测试保留块边界，验证逐块确认和停止增量、暂停恢复、24 kHz 解码时长；使用替身时不模拟为另一套协议路径。
    - 宿主集成测试覆盖默认无登录对话、业务登录/会话切换/上传、断线恢复和历史注入不触发生成；这些测试须待历史输入契约明确后完成。

- 示例应用鉴权与纯对话客户端边界。
    - createSession(url) 直接使用应用提供的 WebSocket URL，默认可直连 /v1/realtime?model=xtalk；不自动调用 /api/auth/login，不要求业务用户身份，不自动附加业务 token。
    - 示例前端先调用示例后端登录接口取得 access_token，通过 URL.searchParams 将其附加到 Realtime URL，再调用 createSession(url) 和 session.open；沿用当前查询参数令牌传递方式，不要求核心新增登录 API 或 business 配置。
    - 示例后端在 Realtime WebSocket 握手进入 xtalk ASGI 应用前验证 access_token 及应用权限；无效 token 拒绝连接，不创建对话。xtalk 核心不读取或验证这个业务参数；可选 xtalk API Key 配置与业务 token 保持独立。
    - 示例后端负责凭证生命周期、Origin 策略和日志脱敏；示例前端负责刷新凭证及重新构造 URL，frontend 不持久化或记录含凭证的完整 URL。不在本次变更新增异步 URL 提供函数，凭证刷新由应用显式重新创建/连接 Session。
    - frontend 的登录、历史、上传、持久化与业务自动恢复迁到示例或应用；清理迁出后无消费者的 HTTP/持久化抽象及平台实现，通过引用检查确定删除范围，音频、WebSocket、编码等平台抽象保留。
    - 迁移测试覆盖无登录直连、示例登录后 URL token 连接、缺失/无效 token 被宿主拒绝、业务 token 不当作 API Key、旧协议业务功能由应用层保留，以及凭证不进入快照和日志。

```typescript
const { access_token } = await exampleAPI.login();
const url = new URL("/v1/realtime", backendURL);
url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
url.searchParams.set("model", "xtalk");
url.searchParams.set("access_token", access_token);
const session = createSession(url);
await session.open();
```

- 前端文件划分沿用既有 SDK 架构。
    - 网络契约采用单个 action-handler-functions/protocol.ts，不拆 client-events/server-events 或协议共用类型文件；原 types.ts 改名为 handler-types.ts，不新增 realtime/ 目录、第二套 receive/send/registry 或 Session 运行时；createSession、ActionHandler、Conversation 和 platforms 继续分别承担编排、分发、状态及平台实现。
    - 保留 audio-session.ts、encoding.ts、task-scheduler.ts 等现有平台工厂薄入口及其 bases 契约；协议编码调用现有编码抽象，浏览器 WebSocket、URL 凭证、设备及模型操作限定在 platforms。
    - session/create.ts 只组装当前对话，不做业务 HTTP 或落盘；runtime.ts 管理当前连接与音频资源，不承担用户身份或业务历史 ID。输入输出处理继续按原 action-handler-functions 分类，不为网络事件重新组织另一套处理器。
    - 删除 HTTP/持久化抽象前验证核心消费者已全部迁出；应用自己的 HTTP/存储属于应用平台适配，不从 xtalk-client 内部路径导入。业务 token URL 由示例业务模块构造，核心只连接传入 URL。
    - 公共 Session API 的列表/上传/业务切换方法迁到示例业务控制器，保留对话操作及状态通知；API 破坏性迁移及前后功能对照测试同步文档，不以保留无业务实现的空方法替代迁移。
    - 桌面应用在自身现有适配层接管同类职责，示例业务模块不反向成为 frontend 核心依赖；测试验证平台实现不泄漏到 session/事件处理器，以及无需业务服务也能运行纯对话。

- 前端协议依赖收敛在事件处理层。
    - action-handler-functions/protocol.ts 定义全部 C→S/S→C 网络事件形状及 Session/音频等协议共用类型；不实现解析、序列化、分发或业务逻辑，不依赖平台实现。
    - action-handler-functions/handler-types.ts 由原 types.ts 改名，定义处理函数签名、上下文和映射类型，引用必要的协议类型；所有原处理器 import 同步迁移，避免与 Session 类型混淆。
    - ActionHandler 负责原始网络消息解析和 type 分发，具体处理器负责字段处理和输出消息构造；runtime.ts 只维护连接/音频生命周期，将原始消息与本地动作交给处理层，不导入具体协议事件或其联合/派生类型。
    - runtime 的音频帧、讲话边界和播放完成回调沿用本地动作参数，经 client/client-operations 等处理器构造网络消息；session.created/updated 的识别与配置消息构造在处理层，runtime 仅通过等待/通知接口协调就绪，不按网络 type 分支。
    - session/create.ts 的公开操作委托本地动作或处理层；业务层不通过内部协议类型直接操作网关。平台采样率等资源配置仍由通用配置提供，网络配置到资源配置的转换在处理层完成。
    - 测试验证 runtime 无具体协议类型导入、协议类型集中且没有第二套消息分发；单独验证握手通知、输入编码与输出控制能够通过原处理链协作。
