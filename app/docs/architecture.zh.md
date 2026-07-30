# XTalk 桌面应用 Phase 0 架构

## 边界

所有桌面专用源码、测试、资源和构建脚本都位于 `app/`。后端使用已安装包公开的
runtime builder 与工具 API，包括原生工具所需、文档已公开的
`xtalk.models.agents.tools` API；UI 只从 `xtalk-client` 包根入口导入。两个适配层
都不导入带下划线的核心私有 API，也不复制核心实现。

## 启动协议

1. Tauri 读取用户此前选择的外部模型配置路径。
2. 没有有效路径时，窗口会在不启动 sidecar 的情况下打开，WebView 随即弹出原生
   JSON 文件选择器。
3. Tauri 生成本次启动随机 token，并创建应用数据目录。
4. Tauri 启动已打包的 Python sidecar，秘密不进入 argv。
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
界面包含居中品牌栏、Orb/对话双视图、底部玻璃控制坞和右侧“设置与诊断”抽屉。
抽屉显示当前外部模型配置与已安装开发者工具，可重新选择配置、将工具目录复制到
AppData、修改启用状态、重启 sidecar 并重新探测本地服务。浅色、深色与窄窗口布局
共用同一桌面适配器和离线状态模型。macOS bundle 包含用户开始语音对话时所需的
麦克风用途说明和 audio-input entitlement。

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

app 通过公共 runtime builder 注册已启用的开发者工具。若没有已启用工具声明
`timer` 名称，则继续注册契约与 `examples/sample_app/custom_async_tool.py` 相同的
随包异步计时器作为回退。单元测试覆盖 `Running`、进度、`Finished`、stop、开发者
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

## 开发者工具

用户选择的目录包含 Python 文件，以及字段严格限定为两个的 `xtalk_tool.json`：

```json
{
  "display_name": "Timer",
  "entrypoint": "timer_tool:create_tools"
}
```

Tauri 生成内部标识，将目录递归复制到 `AppData/tools/<id>/`，并仅在
`AppData/tools/registry.json` 中保存该标识和启用状态。Python sidecar 为每个已启用
目录解析 `module:factory` 入口；工厂必须返回一个列表，其中元素可直接交给
`XtalkBuilder.add_agent_tools()`。

配置中的 Agent 会在读取该注册表后构建，因此工具变更通过受控重启 sidecar 生效。
单个开发者工厂加载失败时会被忽略，不会阻止其余本地服务启动。

## 关闭

UI 先关闭 XTalk Session，再请求关闭。Tauri 调用受认证的 sidecar shutdown endpoint，
在有限时间内等待；若优雅关闭失败，只终止它自己启动的子进程。

## Phase 0 限制

- 外部模型配置必须一直保留在所选路径。若未来进入平台沙箱分发，还需要保存安全
  作用域书签，才能继续使用这种路径持久化方式。
- 开发环境和常规 bundle 中，PyInstaller `onedir` 支持文件与 sidecar 相邻；macOS
  app bundle 按 bootloader 要求将其放入 `Contents/Frameworks`。运行时验证会拒绝
  不完整布局。
- 工具额外依赖管理、本地增强器、provider 设置和可选组件 supervisor 属于后续阶段；
  开发者工具目录当前可使用冻结 sidecar 中已经存在的 Python 包。
