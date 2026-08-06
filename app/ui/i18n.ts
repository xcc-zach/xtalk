/**
 * User-selectable language preference stored by the desktop WebView.
 */
export type LanguagePreference = "auto" | SupportedLanguage;

/**
 * UI languages currently shipped with the desktop application.
 */
export type SupportedLanguage = "en" | "zh-CN";

const LANGUAGE_STORAGE_KEY = "xtalk.ui-language";

const zhCN = {
  "app.description":
    "XTalk 本地桌面客户端。远程服务离线时仍可查看本地运行状态和诊断。",
  "sidebar.close": "关闭聊天侧栏",
  "sidebar.collapse": "收起聊天侧栏",
  "sidebar.expand": "展开聊天侧栏",
  "sidebar.newChat": "新聊天",
  "sidebar.chats": "聊天",
  "sidebar.sessions": "聊天会话",
  "sidebar.loading": "正在读取聊天记录",
  "sidebar.empty": "还没有聊天记录",
  "sidebar.newConversation": "新对话",
  "sidebar.waitingForConfig": "选择模型配置后显示聊天记录",
  "sidebar.waitingForService": "本地服务可用后显示聊天记录",
  "sidebar.creating": "正在创建新聊天",
  "sidebar.switching": "正在载入聊天记录",
  "sidebar.readFailed": "聊天记录读取失败：{{error}}",
  "sidebar.createFailed": "新聊天创建失败：{{error}}",
  "sidebar.switchFailed": "聊天切换失败：{{error}}",
  "sidebar.delete": "删除",
  "sidebar.cancel": "取消",
  "sidebar.deleteTitle": "删除聊天",
  "sidebar.deleteConfirm": "确定删除“{{title}}”吗？该操作无法撤销。",
  "sidebar.deleteSessionAria": "删除这条聊天记录",
  "sidebar.deleteFailed": "聊天删除失败：{{error}}",
  "settings.title": "设置与诊断",
  "settings.open": "打开设置与诊断",
  "settings.close": "关闭设置与诊断",
  "language.title": "语言",
  "language.label": "界面语言",
  "language.auto": "自动（跟随系统）",
  "language.zhCN": "简体中文",
  "language.en": "English",
  "model.title": "模型配置",
  "model.current": "当前配置",
  "model.none": "未选择",
  "model.choose": "选择模型配置文件",
  "model.notSelected": "尚未选择模型配置",
  "model.choosePrompt": "等待选择 JSON 配置",
  "model.firstLaunch": "首次启动需要选择模型配置文件",
  "model.firstLaunchTitle": "首次启动配置",
  "model.firstLaunchDescription":
    "选择推荐配置可立即使用本地语音模型与 DeepSeek 大语言模型；也可以选择自己的 JSON 配置文件。",
  "model.recommended": "推荐配置",
  "model.recommendedDescription":
    "本地语音模型（SenseVoice + Matcha）搭配 DeepSeek 大语言模型，无需选择文件。",
  "model.custom": "自定义配置",
  "model.customDescription": "选择自己的外部 JSON 模型配置文件。",
  "model.cancel": "取消",
  "model.cancelCurrent": "已取消；继续使用当前配置",
  "model.cancelNone": "已取消；尚未选择模型配置",
  "model.restarting": "正在重启本地服务",
  "model.appliedRestarted": "配置已应用，本地服务已重启",
  "model.applied": "配置已应用",
  "model.applyFailed": "配置应用失败",
  "model.applyFailedDetail": "模型配置应用失败：{{error}}",
  "model.readFailed": "无法读取模型配置状态",
  "model.dialogTitle": "选择 XTalk 模型配置",
  "model.dialogFilter": "JSON 配置",
  "llm.title": "配置 LLM API Key",
  "llm.description":
    "输入 DeepSeek API Key 后即可开始对话。Key 将保存在系统凭据管理器中，不会写入 AppData。",
  "llm.linkLabel": "获取 DeepSeek API Key",
  "llm.placeholder": "粘贴 API Key",
  "llm.skip": "跳过",
  "llm.saveAndContinue": "保存并继续",
  "llm.saving": "正在保存 API Key",
  "llm.saved": "API Key 已保存",
  "llm.saveFailed": "API Key 保存失败",
  "llm.saveFailedDetail": "API Key 保存失败：{{error}}",
  "managed.title": "准备本地模型",
  "managed.preparing": "正在检查模型文件",
  "managed.progress": "本地模型准备进度",
  "managed.services": "本地模型服务",
  "managed.checking": "正在检查 {{service}} 的模型文件",
  "managed.downloading": "正在下载 {{service}}",
  "managed.starting": "正在启动 {{service}}",
  "managed.ready": "{{service}} 已就绪",
  "managed.finalizing": "模型服务已就绪，正在启动 XTalk 本地服务",
  "managed.failed": "本地模型准备失败：{{error}}",
  "managed.close": "关闭",
  "tools.title": "工具",
  "tools.close": "关闭工具",
  "tools.description":
    "内置工具与用户工具使用相同的 xtalk_tool.json 协议。",
  "tools.installedList": "可用工具",
  "tools.builtinGroup": "内置工具",
  "tools.userGroup": "用户工具",
  "tools.builtinBadge": "内置",
  "tools.builtinImmutable": "内置工具不能删除，但可以禁用。",
  "tools.install": "从目录安装工具",
  "tools.apply": "应用并重启本地服务",
  "tools.loading": "正在读取已安装工具",
  "tools.none": "没有可用工具",
  "tools.enabled": "启用",
  "tools.enableName": "启用{{name}}",
  "tools.disableName": "禁用{{name}}",
  "tools.requiredName": "{{name}}是必需工具",
  "tools.removeName": "删除{{name}}",
  "tools.removeTitle": "删除已复制的工具",
  "tools.pending": "工具配置已修改；应用并重启本地服务后生效",
  "tools.count": "共 {{count}} 个可用工具",
  "tools.choosePrompt": "等待选择工具目录",
  "tools.cancelled": "已取消安装工具",
  "tools.copying": "正在复制工具目录到 AppData",
  "tools.installed": "{{name}} 已安装；重启本地服务后生效",
  "tools.installFailed": "工具目录安装失败",
  "tools.installFailedDetail": "工具目录安装失败：{{error}}",
  "tools.updating": "正在更新工具状态",
  "tools.updated": "{{name}} 已{{state}}；重启本地服务后生效",
  "tools.stateEnabled": "启用",
  "tools.stateDisabled": "禁用",
  "tools.updateFailed": "工具状态更新失败",
  "tools.updateFailedDetail": "工具状态更新失败：{{error}}",
  "tools.removing": "正在删除已复制的工具",
  "tools.generic": "工具",
  "tools.removed": "{{name}}已删除；重启本地服务后生效",
  "tools.removeFailed": "工具删除失败",
  "tools.removeFailedDetail": "工具删除失败：{{error}}",
  "tools.restarting": "正在重启本地服务并加载工具",
  "tools.applying": "正在应用工具",
  "tools.applied": "工具配置已应用，本地服务已重启",
  "tools.applyFailed": "工具配置应用失败",
  "tools.applyFailedDetail": "工具配置应用失败：{{error}}",
  "tools.dialogTitle": "选择 XTalk 工具目录",
  "tools.uiFrameTitle": "{{name}}工具界面",
  "tools.uiUnavailable": "无法加载工具自定义界面",
  "tools.liveCount": "{{count}} 个工具正在运行",
  "tools.liveExpand": "展开正在运行的工具界面",
  "tools.liveCollapse": "收起正在运行的工具界面",
  "tools.required": "必需",
  "credentials.title": "服务凭据",
  "credentials.description":
    "API Key 保存在操作系统凭据管理器中。环境变量优先，Key 不会写入 AppData、工具目录或启动配置。",
  "credentials.listLabel": "服务凭据",
  "credentials.none": "没有已注册的服务凭据",
  "credentials.count": "共 {{count}} 个服务凭据",
  "credentials.environment": "由环境变量提供（只读且优先）",
  "credentials.system": "已保存在系统凭据管理器",
  "credentials.missing": "未配置",
  "credentials.unavailable": "当前平台的系统凭据管理器不可用",
  "credentials.configure": "配置",
  "credentials.replace": "替换",
  "credentials.delete": "删除",
  "credentials.apply": "应用并重启本地服务",
  "credentials.pending": "凭据已修改；应用并重启本地服务后生效",
  "credentials.dialogTitle": "保存服务凭据",
  "credentials.dialogDescription":
    "Key 将持久保存在操作系统凭据管理器中，不会写入 AppData 或工具目录。",
  "credentials.keyFor": "{{name}} API Key",
  "credentials.keyPlaceholder": "输入 API Key",
  "credentials.keyValidation": "请输入 API Key",
  "credentials.cancel": "取消",
  "credentials.save": "保存",
  "credentials.saving": "正在保存到系统凭据管理器",
  "credentials.saved": "凭据已保存；应用并重启本地服务后生效",
  "credentials.saveFailed": "凭据保存失败",
  "credentials.saveFailedDetail": "凭据保存失败：{{error}}",
  "credentials.deleting": "正在从系统凭据管理器删除",
  "credentials.deleted": "凭据已删除；相关工具已关闭，重启后生效",
  "credentials.deleteFailed": "凭据删除失败",
  "credentials.deleteFailedDetail": "凭据删除失败：{{error}}",
  "credentials.required": "请先在“服务凭据”中配置此工具所需的 API Key",
  "credentials.storeUnavailable": "系统凭据管理器不可用；请改用环境变量",
  "runtime.title": "运行状态",
  "runtime.connection": "连接",
  "runtime.stream": "音频流",
  "runtime.muted": "静音",
  "runtime.network": "网络",
  "runtime.session": "会话",
  "runtime.user": "用户",
  "runtime.onlineTitle": "远程 provider 可尝试连接",
  "runtime.offlineTitle": "远程网络不可用；本地界面仍可使用",
  "runtime.true": "是",
  "runtime.false": "否",
  "runtime.disconnected": "未连接",
  "runtime.connecting": "连接中",
  "runtime.connected": "已连接",
  "runtime.reconnecting": "重新连接中",
  "runtime.idle": "空闲",
  "runtime.listening": "聆听中",
  "runtime.processing": "处理中",
  "runtime.speaking": "播放中",
  "service.title": "本地服务",
  "service.starting": "正在启动",
  "service.chooseConfig": "请选择模型配置",
  "service.searching": "正在查找本地服务",
  "service.applyingConfig": "正在应用模型配置",
  "service.ready": "本地服务已就绪",
  "service.unavailable": "本地服务不可用",
  "service.runtimeUnavailable": "桌面运行时不可用",
  "service.notStarted": "尚未启动",
  "service.waitingEndpoint": "等待 Tauri 提供 endpoint",
  "service.notConfigured": "未配置",
  "service.offlineMode": "离线模式",
  "service.connectFailed": "无法连接本地服务：{{error}}",
  "service.summary.loading": "启动中",
  "service.summary.ready": "已就绪",
  "service.summary.offline": "离线",
  "service.summary.unconfigured": "未配置",
  "recovery.title": "恢复",
  "recovery.retry": "重启本地服务",
  "orb.showChat": "显示聊天记录",
  "orb.startingTitle": "本地服务启动中",
  "orb.offlineAvailable": "界面可离线使用",
  "orb.needsConfig": "需要模型配置",
  "orb.chooseConfig": "请在设置与诊断中选择 JSON 配置",
  "orb.unavailable": "本地服务不可用",
  "orb.reconfigure": "打开设置与诊断后可重新配置",
  "orb.reconnecting": "正在恢复连接",
  "orb.resumeAfterConnect": "对话将在连接恢复后继续",
  "orb.muted": "麦克风已静音",
  "orb.unmuteHint": "点击底部麦克风按钮恢复",
  "orb.listening": "正在聆听",
  "orb.processing": "正在思考",
  "orb.speaking": "正在播放",
  "orb.connected": "对话已连接",
  "orb.openChatHint": "点击圆球查看对话",
  "chat.history": "对话记录",
  "chat.empty": "暂无对话记录",
  "chat.backToOrb": "返回语音圆球模式",
  "composer.form": "发送文本消息",
  "composer.label": "文本消息",
  "composer.send": "发送消息",
  "composer.sending": "正在发送消息",
  "composer.enterHint": "按 Enter 发送，按 Shift+Enter 换行",
  "composer.connectFirst": "连接对话后可发送文本消息",
  "composer.connectFirstShort": "连接对话后可发送文字",
  "composer.chooseConfig": "请先选择模型配置",
  "composer.serviceUnavailable": "本地服务不可用",
  "composer.connecting": "正在连接对话",
  "composer.updating": "正在更新连接",
  "composer.reconnecting": "正在恢复连接",
  "composer.initializing": "正在初始化会话",
  "composer.input": "输入消息",
  "composer.sendFailed": "消息发送失败：{{error}}",
  "voice.controls": "语音控制",
  "voice.mute": "关闭麦克风",
  "voice.unmute": "打开麦克风",
  "voice.starting": "正在开始对话",
  "voice.stopping": "正在结束对话",
  "voice.start": "开始对话",
  "voice.stop": "结束对话",
  "voice.connectFailed": "会话连接失败：{{error}}",
  "voice.closeFailed": "会话关闭失败：{{error}}",
  "whiteboard.show": "显示白板",
  "whiteboard.hide": "隐藏白板",
  "whiteboard.windowTitle": "白板",
  "whiteboard.revision": "版本",
  "whiteboard.empty": "白板内容为空，等待助手写入内容。",
  "whiteboard.updated": "更新于 {{time}}",
  "whiteboard.unavailable": "白板暂不可用：{{error}}",
  "message.user": "你",
  "message.system": "系统",
  "message.copy": "复制消息",
  "message.copied": "已复制",
  "message.copyFailed": "复制消息失败：{{error}}",
  "app.initializeFailed": "无法初始化应用：{{error}}",
  "native.runtimeUnavailable": "桌面运行时不可用；当前界面已进入离线模式。",
} as const;

const en: Record<keyof typeof zhCN, string> = {
  "app.description":
    "XTalk local desktop client. Local status and diagnostics remain available when remote services are offline.",
  "sidebar.close": "Close conversation sidebar",
  "sidebar.collapse": "Collapse conversation sidebar",
  "sidebar.expand": "Expand conversation sidebar",
  "sidebar.newChat": "New chat",
  "sidebar.chats": "Chats",
  "sidebar.sessions": "Conversation sessions",
  "sidebar.loading": "Loading conversation history",
  "sidebar.empty": "No conversations yet",
  "sidebar.newConversation": "New conversation",
  "sidebar.waitingForConfig": "Choose a model configuration to view conversations",
  "sidebar.waitingForService": "Conversations will appear when the local service is available",
  "sidebar.creating": "Creating a new chat",
  "sidebar.switching": "Loading conversation history",
  "sidebar.readFailed": "Could not load conversation history: {{error}}",
  "sidebar.createFailed": "Could not create the new chat: {{error}}",
  "sidebar.switchFailed": "Could not switch conversations: {{error}}",
  "sidebar.delete": "Delete",
  "sidebar.cancel": "Cancel",
  "sidebar.deleteTitle": "Delete conversation",
  "sidebar.deleteConfirm": "Delete \"{{title}}\"? This cannot be undone.",
  "sidebar.deleteSessionAria": "Delete this conversation",
  "sidebar.deleteFailed": "Failed to delete conversation: {{error}}",
  "settings.title": "Settings & diagnostics",
  "settings.open": "Open settings and diagnostics",
  "settings.close": "Close settings and diagnostics",
  "language.title": "Language",
  "language.label": "Interface language",
  "language.auto": "Automatic (system)",
  "language.zhCN": "简体中文",
  "language.en": "English",
  "model.title": "Model configuration",
  "model.current": "Current configuration",
  "model.none": "Not selected",
  "model.choose": "Choose model configuration",
  "model.notSelected": "No model configuration selected",
  "model.choosePrompt": "Waiting for a JSON configuration",
  "model.firstLaunch": "Choose a model configuration to finish first-time setup",
  "model.firstLaunchTitle": "First-time setup",
  "model.firstLaunchDescription":
    "Choose the recommended configuration to start with local voice models and the DeepSeek LLM, or select your own JSON configuration file.",
  "model.recommended": "Recommended configuration",
  "model.recommendedDescription":
    "Local voice models (SenseVoice + Matcha) with the DeepSeek LLM; no file selection needed.",
  "model.custom": "Custom configuration",
  "model.customDescription": "Choose your own external JSON model configuration file.",
  "model.cancel": "Cancel",
  "model.cancelCurrent": "Cancelled; continuing with the current configuration",
  "model.cancelNone": "Cancelled; no model configuration selected",
  "model.restarting": "Restarting the local service",
  "model.appliedRestarted": "Configuration applied and local service restarted",
  "model.applied": "Configuration applied",
  "model.applyFailed": "Could not apply the configuration",
  "model.applyFailedDetail": "Could not apply the model configuration: {{error}}",
  "model.readFailed": "Could not read the model configuration status",
  "model.dialogTitle": "Choose XTalk model configuration",
  "model.dialogFilter": "JSON configuration",
  "llm.title": "Configure the LLM API key",
  "llm.description":
    "Enter your DeepSeek API key to start chatting. The key is stored in the system credential manager and never written to AppData.",
  "llm.linkLabel": "Get a DeepSeek API key",
  "llm.placeholder": "Paste the API key",
  "llm.skip": "Skip",
  "llm.saveAndContinue": "Save and continue",
  "llm.saving": "Saving the API key",
  "llm.saved": "API key saved",
  "llm.saveFailed": "Could not save the API key",
  "llm.saveFailedDetail": "Could not save the API key: {{error}}",
  "managed.title": "Preparing local models",
  "managed.preparing": "Checking model files",
  "managed.progress": "Local model preparation progress",
  "managed.services": "Local model services",
  "managed.checking": "Checking model files for {{service}}",
  "managed.downloading": "Downloading {{service}}",
  "managed.starting": "Starting {{service}}",
  "managed.ready": "{{service}} is ready",
  "managed.finalizing":
    "Model services are ready; starting the XTalk local service",
  "managed.failed": "Could not prepare local models: {{error}}",
  "managed.close": "Close",
  "tools.title": "Tools",
  "tools.close": "Close tools",
  "tools.description":
    "Built-in and user tools use the same xtalk_tool.json protocol.",
  "tools.installedList": "Available tools",
  "tools.builtinGroup": "Built-in tools",
  "tools.userGroup": "User tools",
  "tools.builtinBadge": "Built-in",
  "tools.builtinImmutable": "Built-in tools cannot be removed, but can be disabled.",
  "tools.install": "Install tool from directory",
  "tools.apply": "Apply and restart local service",
  "tools.loading": "Loading installed tools",
  "tools.none": "No tools available",
  "tools.enabled": "Enabled",
  "tools.enableName": "Enable {{name}}",
  "tools.disableName": "Disable {{name}}",
  "tools.requiredName": "{{name}} is required",
  "tools.removeName": "Remove {{name}}",
  "tools.removeTitle": "Remove copied tool",
  "tools.pending": "Tool settings changed; apply and restart the local service",
  "tools.count": "{{count}} tools available",
  "tools.choosePrompt": "Waiting for a tool directory",
  "tools.cancelled": "Tool installation cancelled",
  "tools.copying": "Copying tool directory to AppData",
  "tools.installed": "{{name}} installed; restart the local service to apply",
  "tools.installFailed": "Could not install the tool directory",
  "tools.installFailedDetail": "Could not install the tool directory: {{error}}",
  "tools.updating": "Updating tool status",
  "tools.updated": "{{name}} {{state}}; restart the local service to apply",
  "tools.stateEnabled": "enabled",
  "tools.stateDisabled": "disabled",
  "tools.updateFailed": "Could not update the tool status",
  "tools.updateFailedDetail": "Could not update the tool status: {{error}}",
  "tools.removing": "Removing copied tool",
  "tools.generic": "Tool",
  "tools.removed": "{{name}} removed; restart the local service to apply",
  "tools.removeFailed": "Could not remove the tool",
  "tools.removeFailedDetail": "Could not remove the tool: {{error}}",
  "tools.restarting": "Restarting the local service and loading tools",
  "tools.applying": "Applying tools",
  "tools.applied": "Tool settings applied and local service restarted",
  "tools.applyFailed": "Could not apply tool settings",
  "tools.applyFailedDetail": "Could not apply tool settings: {{error}}",
  "tools.dialogTitle": "Choose XTalk tool directory",
  "tools.uiFrameTitle": "{{name}} tool UI",
  "tools.uiUnavailable": "Could not load the custom tool UI",
  "tools.liveCount": "{{count}} tools running",
  "tools.liveExpand": "Expand running tool UI",
  "tools.liveCollapse": "Collapse running tool UI",
  "tools.required": "Required",
  "credentials.title": "Service credentials",
  "credentials.description":
    "API keys are stored in the operating system credential manager. Environment variables take precedence, and keys are never written to AppData, tool directories, or startup configuration.",
  "credentials.listLabel": "Service credentials",
  "credentials.none": "No service credentials are registered",
  "credentials.count": "{{count}} service credentials",
  "credentials.environment":
    "Provided by environment variable (read-only and preferred)",
  "credentials.system": "Saved in the system credential manager",
  "credentials.missing": "Not configured",
  "credentials.unavailable":
    "The system credential manager is unavailable on this platform",
  "credentials.configure": "Configure",
  "credentials.replace": "Replace",
  "credentials.delete": "Delete",
  "credentials.apply": "Apply and restart local service",
  "credentials.pending":
    "Credentials changed; apply and restart the local service to use them",
  "credentials.dialogTitle": "Save service credential",
  "credentials.dialogDescription":
    "The key is persisted in the operating system credential manager and is never written to AppData or tool directories.",
  "credentials.keyFor": "{{name}} API key",
  "credentials.keyPlaceholder": "Enter API key",
  "credentials.keyValidation": "Enter an API key",
  "credentials.cancel": "Cancel",
  "credentials.save": "Save",
  "credentials.saving": "Saving to the system credential manager",
  "credentials.saved":
    "Credential saved; apply and restart the local service to use it",
  "credentials.saveFailed": "Could not save credential",
  "credentials.saveFailedDetail": "Could not save credential: {{error}}",
  "credentials.deleting": "Deleting from the system credential manager",
  "credentials.deleted":
    "Credential deleted; dependent tools were disabled and the change applies after restart",
  "credentials.deleteFailed": "Could not delete credential",
  "credentials.deleteFailedDetail": "Could not delete credential: {{error}}",
  "credentials.required":
    "Configure the API key required by this tool under Service credentials first",
  "credentials.storeUnavailable":
    "The system credential manager is unavailable; use an environment variable instead",
  "runtime.title": "Runtime status",
  "runtime.connection": "Connection",
  "runtime.stream": "Stream",
  "runtime.muted": "Muted",
  "runtime.network": "Network",
  "runtime.session": "Session",
  "runtime.user": "User",
  "runtime.onlineTitle": "The remote provider can be reached",
  "runtime.offlineTitle": "Remote network unavailable; the local interface remains available",
  "runtime.true": "Yes",
  "runtime.false": "No",
  "runtime.disconnected": "Disconnected",
  "runtime.connecting": "Connecting",
  "runtime.connected": "Connected",
  "runtime.reconnecting": "Reconnecting",
  "runtime.idle": "Idle",
  "runtime.listening": "Listening",
  "runtime.processing": "Processing",
  "runtime.speaking": "Speaking",
  "service.title": "Local service",
  "service.starting": "Starting",
  "service.chooseConfig": "Choose a model configuration",
  "service.searching": "Finding the local service",
  "service.applyingConfig": "Applying model configuration",
  "service.ready": "Local service is ready",
  "service.unavailable": "Local service unavailable",
  "service.runtimeUnavailable": "Desktop runtime unavailable",
  "service.notStarted": "Not started",
  "service.waitingEndpoint": "Waiting for Tauri to provide the endpoint",
  "service.notConfigured": "Not configured",
  "service.offlineMode": "Offline mode",
  "service.connectFailed": "Could not connect to the local service: {{error}}",
  "service.summary.loading": "Starting",
  "service.summary.ready": "Ready",
  "service.summary.offline": "Offline",
  "service.summary.unconfigured": "Not configured",
  "recovery.title": "Recovery",
  "recovery.retry": "Restart local service",
  "orb.showChat": "Show conversation history",
  "orb.startingTitle": "Local service is starting",
  "orb.offlineAvailable": "The interface remains available offline",
  "orb.needsConfig": "Model configuration required",
  "orb.chooseConfig": "Choose a JSON file in Settings & diagnostics",
  "orb.unavailable": "Local service unavailable",
  "orb.reconfigure": "Open Settings & diagnostics to reconfigure",
  "orb.reconnecting": "Restoring connection",
  "orb.resumeAfterConnect": "The conversation will continue after reconnection",
  "orb.muted": "Microphone muted",
  "orb.unmuteHint": "Use the microphone button below to unmute",
  "orb.listening": "Listening",
  "orb.processing": "Thinking",
  "orb.speaking": "Playing",
  "orb.connected": "Conversation connected",
  "orb.openChatHint": "Select the orb to view the conversation",
  "chat.history": "Conversation history",
  "chat.empty": "No messages yet",
  "chat.backToOrb": "Return to voice orb",
  "composer.form": "Send a text message",
  "composer.label": "Text message",
  "composer.send": "Send message",
  "composer.sending": "Sending message",
  "composer.enterHint": "Press Enter to send; Shift+Enter inserts a new line",
  "composer.connectFirst": "Connect a conversation to send text messages",
  "composer.connectFirstShort": "Connect a conversation to send text",
  "composer.chooseConfig": "Choose a model configuration first",
  "composer.serviceUnavailable": "Local service unavailable",
  "composer.connecting": "Connecting conversation",
  "composer.updating": "Updating connection",
  "composer.reconnecting": "Restoring connection",
  "composer.initializing": "Initializing session",
  "composer.input": "Type a message",
  "composer.sendFailed": "Could not send message: {{error}}",
  "voice.controls": "Voice controls",
  "voice.mute": "Mute microphone",
  "voice.unmute": "Unmute microphone",
  "voice.starting": "Starting conversation",
  "voice.stopping": "Ending conversation",
  "voice.start": "Start conversation",
  "voice.stop": "End conversation",
  "voice.connectFailed": "Could not connect the conversation: {{error}}",
  "voice.closeFailed": "Could not close the conversation: {{error}}",
  "whiteboard.show": "Show whiteboard",
  "whiteboard.hide": "Hide whiteboard",
  "whiteboard.windowTitle": "Whiteboard",
  "whiteboard.revision": "Revision",
  "whiteboard.empty": "The whiteboard is empty. Ask the assistant to write to it.",
  "whiteboard.updated": "Updated {{time}}",
  "whiteboard.unavailable": "Whiteboard unavailable: {{error}}",
  "message.user": "You",
  "message.system": "System",
  "message.copy": "Copy message",
  "message.copied": "Copied",
  "message.copyFailed": "Could not copy message: {{error}}",
  "app.initializeFailed": "Could not initialize the application: {{error}}",
  "native.runtimeUnavailable":
    "Desktop runtime unavailable; the interface is now in offline mode.",
};

export type TranslationKey = keyof typeof zhCN;

const dictionaries: Record<SupportedLanguage, Record<TranslationKey, string>> = {
  "zh-CN": zhCN,
  en,
};

let preference = readLanguagePreference();
let language = resolveLanguage(preference);

/**
 * Returns the persisted language choice, including automatic mode.
 *
 * @returns Current language preference.
 */
export function getLanguagePreference(): LanguagePreference {
  return preference;
}

/**
 * Returns the concrete language currently used by the interface.
 *
 * @returns Resolved supported language.
 */
export function getResolvedLanguage(): SupportedLanguage {
  return language;
}

/**
 * Persists and applies a language preference.
 *
 * @param nextPreference Automatic mode or a supported explicit language.
 * @returns Resolved language after applying the preference.
 */
export function setLanguagePreference(
  nextPreference: LanguagePreference,
): SupportedLanguage {
  preference = nextPreference;
  localStorage.setItem(LANGUAGE_STORAGE_KEY, nextPreference);
  language = resolveLanguage(nextPreference);
  document.documentElement.lang = language;
  return language;
}

/**
 * Translates one interface message and interpolates named parameters.
 *
 * @param key Stable translation key.
 * @param parameters Named values inserted into `{{name}}` placeholders.
 * @returns Localized interface text.
 */
export function t(
  key: TranslationKey,
  parameters: Readonly<Record<string, string | number>> = {},
): string {
  return Object.entries(parameters).reduce(
    (message, [name, value]) =>
      message.split(`{{${name}}}`).join(String(value)),
    dictionaries[language][key],
  );
}

/**
 * Applies translation attributes found in the current HTML document.
 */
export function translateDocument(): void {
  document.documentElement.lang = language;
  document
    .querySelectorAll<HTMLElement>("[data-i18n]")
    .forEach((element) => {
      const key = element.dataset.i18n as TranslationKey | undefined;
      if (key) {
        element.textContent = t(key);
      }
    });
  for (const attribute of ["aria-label", "title", "placeholder"] as const) {
    const dataAttribute = `data-i18n-${attribute}`;
    document
      .querySelectorAll<HTMLElement>(`[${dataAttribute}]`)
      .forEach((element) => {
        const key = element.getAttribute(dataAttribute) as TranslationKey | null;
        if (key) {
          element.setAttribute(attribute, t(key));
        }
      });
  }
  const description = document.querySelector<HTMLMetaElement>(
    'meta[name="description"]',
  );
  description?.setAttribute("content", t("app.description"));
  document.documentElement.style.setProperty(
    "--empty-conversation-label",
    JSON.stringify(t("chat.empty")),
  );
}

/**
 * Re-evaluates automatic language selection after a system language change.
 *
 * @returns Whether the resolved language changed.
 */
export function refreshAutomaticLanguage(): boolean {
  if (preference !== "auto") {
    return false;
  }
  const nextLanguage = resolveLanguage(preference);
  if (nextLanguage === language) {
    return false;
  }
  language = nextLanguage;
  document.documentElement.lang = language;
  return true;
}

/**
 * Re-localizes a known UI-originated error after the interface language changes.
 *
 * @param message Error text produced in any supported UI language.
 * @returns The current-language equivalent when known, otherwise the original.
 */
export function localizeKnownError(message: string): string {
  const translatedKeys: TranslationKey[] = ["native.runtimeUnavailable"];
  for (const key of translatedKeys) {
    if (Object.values(dictionaries).some((dictionary) => dictionary[key] === message)) {
      return t(key);
    }
  }
  const nativeErrors: Readonly<Record<string, TranslationKey>> = {
    "the selected tool requires a configured service credential":
      "credentials.required",
    "the system credential store is unavailable":
      "credentials.storeUnavailable",
  };
  const key = nativeErrors[message];
  return key === undefined ? message : t(key);
}

function readLanguagePreference(): LanguagePreference {
  const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
  return stored === "en" || stored === "zh-CN" || stored === "auto"
    ? stored
    : "auto";
}

function resolveLanguage(
  nextPreference: LanguagePreference,
): SupportedLanguage {
  if (nextPreference !== "auto") {
    return nextPreference;
  }
  const locales =
    navigator.languages.length > 0 ? navigator.languages : [navigator.language];
  return locales.some((locale) => locale.toLowerCase().startsWith("zh"))
    ? "zh-CN"
    : "en";
}
