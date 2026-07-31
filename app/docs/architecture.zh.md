# XTalk 桌面应用 Phase 0 架构

## 边界

所有桌面专用源码、测试、资源和构建脚本都位于 `app/`。后端使用已安装包公开的
runtime builder 与工具 API，包括原生工具所需、文档已公开的
`xtalk.models.agents.tools` API；UI 只从 `xtalk-client` 包根入口导入。两个适配层
都不导入带下划线的核心私有 API，也不复制核心实现。

## 启动协议

1. Tauri 读取用户此前选择的外部模型配置路径，但不在原生 setup 阶段启动用户选择的
   进程。
2. 没有有效路径时，窗口会在不启动 sidecar 的情况下打开，WebView 随即弹出原生
   JSON 文件选择器。
3. WebView 订阅 managed 模型进度后，请求 Tauri 确保所选后端正在运行；若已有受管且
   健康的实例则直接复用，否则先启动配置引用的全部 managed 服务，再启动 Python
   sidecar。
4. Tauri 生成本次启动随机 token、创建应用数据目录并启动已打包的 Python sidecar，
   秘密不进入 argv。
5. Tauri 通过 sidecar stdin 写入一条有大小上限的 JSON 启动消息。
6. sidecar 加载指定的 XTalk 配置，绑定由操作系统分配的 loopback 端口，启动
   FastAPI，然后输出一行 readiness JSON。
7. Tauri 严格校验 readiness 协议，并通过最小 command 向 WebView 提供 endpoint
   和 token。
8. UI 创建 `xtalk-client` Session，并关闭前端 VAD 与增强器。
9. Tauri 将随包 Silero 模型作为顶层配置回退项传给 sidecar；当所选配置没有显式
   声明 VAD 时，由 Python sidecar 产生语音边界。

sidecar 关闭 HTTP access log，避免公共 SDK 使用的 query capability 出现在 URL
日志中。

## 本地界面

WebView 沿用 `examples/sample_app` 的布局层级和视觉语言，但不导入示例实现代码。
界面包含默认收起的左侧聊天记录栏、上下文状态栏、Orb/对话双视图、底部玻璃控制坞
和右侧“设置与诊断”抽屉。左侧栏可创建新聊天，并通过公开客户端 API 切换全部持久
化会话。“新聊天”下方的同款“工具”按钮会打开居中浮窗，用于将工具目录复制到
AppData、修改内置或用户工具的启用状态、删除用户工具，并通过重启 sidecar 应用
变更。内置工具单独分组显示，且不提供删除入口。“设置与诊
断”顶部显示本地服务状态，并将语言、外部模型配置、运行状态、本地服务诊断和恢复
组织为可独立展开的选项卡。用户可更换模型配置并重新探测本地服务。“语言”选项默
认按操作系统首选语言自动选择界面语言，也可持久化指定简体中文或英文。静态内容、
动态状态、无障碍标签和原生文件选择器统一使用解析后的语言。浅色、深色和窄窗口
布局共用同一桌面适配器和离线状态模型。macOS bundle 包含用户开始语音对话时所需
的麦克风用途说明和 audio-input entitlement。

会话历史以服务端数据为准，存储在强制配置给 sidecar 的应用数据目录下的
`chat_history.sqlite3` 中。桌面端私有启动协议会传递固定的匿名用户标识，该标识不
进入公开的 `service_config`，并由启动 token 与 Origin 边界确保此单用户身份仅供
本应用使用。因此左侧栏中的会话标题和消息可跨应用及 sidecar 重启保留，无需再维护
一份 WebView 自有的历史记录。

## 认证契约

所有 HTTP 请求都要求启动 token 和批准的 Origin。app 自有客户端使用
`X-XTalk-App-Token`；由于公共 `xtalk-client` 没有自定义 header 扩展点，适配器只在
显式 HTTP service URL 上附加 token。

WebSocket URL 不包含启动 token，避免 SDK 把 token 作为 local-storage key 的一部分
持久化。外层中间件仅在 WebSocket 携带启动 token，或携带从受启动 token 保护的登录
路由取得的 XTalk access token 时放行；随后仍由 XTalk 完成权威 access-token 校验。
loopback peer 与 Origin 检查始终生效。

文本回合不再经过 app 自有 HTTP endpoint；公共客户端直接通过已经认证的 XTalk
WebSocket 发送。

## 文本输入与计时器链路

桌面适配器直接调用公共 `Session.sendText()`。SDK 会裁剪首尾空白，把消息限制为
2,048 个 JavaScript 字符，通过已连接的 WebSocket 发送 `submit_text`，并等待最多
10 秒，直到收到文本完全匹配且 `origin="text"` 的 `finish_asr`。核心后端验证
UTF-8 编码不超过 8 KiB，再通过普通 VAD、ASR、Agent、Tool、TTS 与持久化管线处理
合成文本回合；UI 不会在本地乐观伪造消息。

每个客户端 Session 同一时间只能有一条文本等待确认。断线、关闭、重新打开或切换
Session 都会取消待确认请求；客户端不会排队或自动重试可能执行有副作用工具的回合。

app 通过公共 runtime builder 注册已启用的内置与用户工具。随包异步计时器与
`examples/sample_app/custom_async_tool.py` 契约一致，并作为普通内置 manifest 工具
加载；若用户工具导出相同名称，则优先使用用户实现。单元测试覆盖 `Running`、进度、
`Finished`、stop、工具
入口加载，并通过公共 `ToolEngine` 验证最终更新；模型 smoke 则独立从文本入口发起
请求，观察 `tool_called: timer`，确认真实的助手回复与 TTS 音频播放。测试不强制要求
第二次 LLM 主动报告，因为模型驱动的报告可能与首轮响应重叠；app 不为这一时序增加
timer 专用的 serving 修补。

文本输入要求已有活动的 XTalk Session。公共 SDK 的 `open()` 仍会初始化麦克风采集，
因此即使随后只输入文字，启动 Session 也需要麦克风权限。

## 本地语音管线

WebView 通过公开 `xtalk-client` WebSocket 发送 16 kHz 单声道 PCM，前端 VAD 与
增强器保持关闭。Tauri 解析随包的 `models/audio/silero_vad.onnx` 资源，sidecar
再通过 XTalk 普通公开模型配置加载它。由服务端产生的语音开始、结束边界负责启动和
结束配置中的远程 ASR 回合。

模型的上游 commit 与 SHA-256 固定在
`resources/manifests/audio-models.lock.json`。资源校验会拒绝缺失或被修改的文件；
安装后的应用不会为这项基础运行资源执行在线下载。

## 可选原生模型运行时

按需下载的可选语音模型运行在独立的 Rust HTTP sidecar
`app/local-model-runtime` 中。首个引擎使用 SentencePiece 和五个 ONNX Session
直接实现 MOSS-TTS-Nano：参考音频 codec encode、prefill、自回归 decode、本地帧
采样和 codec decode。主接口 `POST /api/generate` 与官方 Python/FastAPI 服务一致：
输入 multipart `text` 和 `prompt_audio`，输出包含 base64 WAV 的 JSON。参考音频会在
编码前转换为 48 kHz，生成结果固定为 48 kHz 单声道 PCM16。

Apple Silicon 安装包还包含 `app/local-model-runtime-mlx` 中的 Swift sidecar。
它通过固定版本的 `mlx-audio-swift` 从本地加载 SenseVoice 与 MOSS safetensors
快照，并保持同一套离线 ASR WebSocket 数据包与 MOSS multipart HTTP 协议，因此
Python 模型客户端不需要按推理后端分支。MLX 的 MOSS 输出同样固定为 48 kHz
单声道 PCM16。

ONNX Runtime 是 App 自带资源，不要求用户另行安装。sidecar 通过
`--ort-dylib` 加载 App 解析后传入的精确动态库；模型权重不放进安装包。用户配置
`managed://sensevoice-small` 或 `managed://moss-tts-nano` 后，Tauri 会读取不可变
的 `managed-models.lock.json`，仅下载对应服务的固定版本文件到
`AppData/models/managed/<id>/<version>/`，校验文件大小和 SHA-256，再原子写入完成
标记。此后每次启动仍会重新校验已安装快照。

managed URL 支持 `?backend=cpu`、`?backend=cuda` 和 `?backend=mlx`。不带查询参数
时，Tauri 依次选择 NVIDIA 设备上随包可用的 CUDA provider、Apple Silicon MLX，
最后回退 CPU。显式选择不可用后端会报错，不会静默降级。CUDA 与 CPU 共用 ONNX
快照，MLX 则选择单独固定的 safetensors 快照。

用户选择配置后，Tauri 会在应用配置前先做预检。包含 managed 服务的配置会打开阻塞
式进度窗口；原生进度事件会报告模型校验、逐文件下载字节数、服务启动和就绪状态。
Python 后端通过健康检查前，界面其余区域保持不可交互；成功后进度窗口自动关闭。
启动失败时，窗口会保留错误信息和关闭操作。

ONNX 模式下，Tauri 通过随包的原生 `sherpa-onnx-offline-websocket-server` 启动
SenseVoice，并通过 Rust sidecar 启动 MOSS；MLX 模式下，每项服务分别启动一个
Swift sidecar。它等待 TCP/readiness 就绪边界，再把真实临时 loopback
地址和解析后的 AppData 音色路径深度合并进 Python 启动 overlay；外部配置因此不含
运行时端口。安装、模型进程或 Python 启动任一步骤失败时，刚启动的全部子进程都会
被停止，并恢复此前配置。managed 子进程意外退出后，后端连接也会立即变为不可用。

完整本地示例见
[`../examples/local_models.json`](../examples/local_models.json)。其中 LLM 与
`server_configs/sample.json` 一致，但 `api_key` 特意留空。SenseVoice 继续通过现有
离线 WebSocket 客户端接收 16 kHz PCM；MOSS 参考音频和生成结果均使用 48 kHz。
[`../examples/local_models_mlx.json`](../examples/local_models_mlx.json) 则显式选择
MLX。

## 配置

release 包不包含默认 XTalk 模型配置。原生文件选择器接收一个外部 JSON 文件；
Tauri 要求根节点为对象、限制大小不超过 1 MiB、规范化路径，并且只在 AppConfig
中持久化该路径。配置内容和 provider 凭据仍保留在外部文件中。更换配置时先验证
新文件，再停止当前 sidecar、启动新 sidecar、持久化成功的选择，最后让 WebView
重新探测本地服务。

启动消息选择该 JSON 配置，也可以携带顶层 fallback 和 JSON overlay。fallback
只填充所选配置中完全缺失的 key；显式模型槽位作为完整值保留。随后以 overlay 为
最高优先级执行通用深度合并。Python 端仍不检查模型类型名称，唯一强制项是把
`service_config.data_dir` 指向 AppData。配置中的模型缺失、未知或参数错误仍作为
XTalk 原始配置错误返回。

模型集成测试以 `server_configs/sample.json` 为基础。可选回合检测测试从外部临时
配置注入完整标准 `type`/`params` 对象。桌面 VAD fallback 使未修改的 sample 配置
也能完成语音回合。私有模型仓库和凭据属于测试输入，不进入提交的应用配置。

由于核心模型发现机制是动态的，冻结构建会收集已安装的公共 `xtalk.models` 命名空间
及其包数据。所需可选依赖组通过 `--xtalk-extra` 显式传入构建过程，不在应用行为中
引入模型类型分支。

## 内置与用户工具

两种来源的目录都包含 Python 文件和 `xtalk_tool.json`。`display_name` 可以是单个
字符串，也可以是语言字典；`ui` 为可选配置：

```json
{
  "display_name": {
    "zh": "计时器",
    "en": "Timer"
  },
  "entrypoint": "timer_tool:create_tools",
  "ui": {
    "entrypoint": "ui/index.html",
    "update_every_s": 0.5
  }
}
```

对于用户工具，Tauri 生成内部标识，将目录递归复制到
`AppData/tools/<id>/`，并在 `AppData/tools/registry.json` 中保存该标识和启用状态。
内置工具保留在只读的 `resources/tools/` 包资源中，由
`resources/tools/builtin_tools.json` 建立索引；来源、删除权限和默认启用状态属于
App 元数据，不进入 manifest。内置工具 ID 使用 `builtin:<id>` 命名空间，其启用
覆盖保存到 `AppData/tool_preferences.json`。

原生删除命令会自行解析工具来源并拒绝内置 ID，删除保护不依赖 WebView。两种工具
都可以禁用。Python sidecar 使用同一套 `module:factory` 入口加载每个已启用目录；
用户工具与内置工具导出同名时优先使用用户实现。工厂必须返回一个列表，其中元素
可直接交给 `XtalkBuilder.add_agent_tools()`。

配置中的 Agent 会在读取该注册表后构建，因此工具变更通过受控重启 sidecar 生效。
单个开发者工厂加载失败时会被忽略，不会阻止其余本地服务启动。

Codex 内置工具由一个 catalog 条目和一个 manifest 表示，其工厂一次返回查询、新建、
继续、切换模型和删除五个原生异步工具。因此它只能整体启用或禁用，不存在按导出
工具保存的偏好；默认状态为禁用。所有业务线程与 turn 都通过官方 Python SDK 使用
`Sandbox.full_access` 与 SDK 的无提示审批模式，`cwd` 可以是任意真实存在的本地
目录，并在每次 turn 显式重用该 session 保存的模型与推理强度。条件式路由规则直接
写入工具描述，不向 Agent 的 developer instructions 注入 Codex 专用指令。

App 侧 session 索引是
`AppData/tool-data/codex/codex_sessions.sqlite3`。其中只保存 SDK thread ID、精简标题与
摘要、工作目录、模型、推理强度、状态和时间戳；真实 thread 对话记录仍由 SDK 负责。
自然语言查询先把活动索引机械限制到最多 30 个候选，再仅把该 JSON 快照交给使用严格
输出 schema 的临时 ephemeral 查询线程；返回 ID 必须是候选 ID 的子集。删除操作先
归档 SDK thread，再在 App 索引中将其移出活动池；每个 session 的异步锁会串行化
继续、改配置和归档操作。

可选 UI 入口是最大一 MiB 的自包含 HTML，与 Python 工具入口保持分离。UI 通过注册
`window.xtalkToolUI.status(callback)` 和/或
`window.xtalkToolUI.emit(callback)` 声明能力；没有注册某个 hook，就表示该工具没有
对应的 live UI 或 history UI。`update_every_s` 控制 live 状态轮询，默认为一秒；
`-1` 表示禁用周期轮询，其余取值范围为 0.1 到 3600 秒。

App 在注册表加载时仅包装原生 `AsyncTool` 类。包装层保持原生命周期调用和返回值
不变，只观察 `astatus()`，再通过受启动 token 保护的 App WebSocket 发布只读事件。
live 事件跟随正在运行的调用；原工具的 initial emit 和每次 update emit 都会产生独立
的 history 事件，其中保存当次 emit 消息和当时的 status。history 快照不会变化，
每个会话最多保留 200 条，并按持久化 session ID 保存到 WebView AppData；live 状态
只保存在内存中。

对话顶部不再重复显示 XTalk 产品名。没有支持 live UI 的工具运行时，顶部中心保持
空白；有工具运行时显示一条默认收起的紧凑状态栏，其中包含运行数量和最新状态。
用户点击后可展开查看所有当前 live UI。live 卡片不再插入消息时间线，history 卡片
仍固定在对应 emit 的历史位置。

每张卡片使用独立的 `sandbox="allow-scripts"` iframe。注入的 CSP 阻止外部资源和
网络 API，bridge 还会拦截链接和表单行为；frame 的不透明 origin 不具备 Tauri
权限。frame 只能接收 status/emit 数据并报告期望高度，不能调用、停止或以其他方式
操作工具。为保留 App 顶层的严格 CSP，host 会把每个准备好的文档放在一个高熵、
30 秒有效且只能使用一次的 loopback ticket 后面；启动 token 不会进入 iframe URL
或文档。host 控制卡片完整可用宽度，并把 live 卡片高度限制在 120–420 px、
history 卡片限制在 80–600 px；两者还同时受窗口高度 60% 的上限约束。

## 关闭

UI 先关闭 XTalk Session，再请求关闭。Tauri 调用受认证的 Python sidecar shutdown
endpoint，在有限时间内等待；若优雅关闭失败，只终止它自己启动的子进程。随后按
启动顺序的逆序停止 managed 模型进程。

## Phase 0 限制

- 外部模型配置必须一直保留在所选路径。若未来进入平台沙箱分发，还需要保存安全
  作用域书签，才能继续使用这种路径持久化方式。
- 开发环境和常规 bundle 中，PyInstaller `onedir` 支持文件与 sidecar 相邻；macOS
  app bundle 按 bootloader 要求将其放入 `Contents/Frameworks`。运行时验证会拒绝
  不完整布局。
- 工具额外依赖管理、本地增强器和 provider 设置属于后续阶段；开发者工具目录当前
  可使用冻结 sidecar 中已经存在的 Python 包。
