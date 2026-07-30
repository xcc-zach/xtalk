import "./styles.css";

import {
  applyNativeModelConfig,
  applyNativeToolChanges,
  chooseNativeModelConfigFile,
  chooseNativeToolDirectory,
  getNativeBackendConnection,
  getNativeInstalledTools,
  getNativeModelConfigSelection,
  installNativeToolDirectory,
  removeNativeInstalledTool,
  setNativeToolEnabled,
  type NativeModelConfigSelection,
  type NativeToolDefinition,
} from "./adapters/native-capabilities";
import {
  XtalkClientAdapter,
  type DesktopSessionSnapshot,
} from "./adapters/xtalk-client-adapter";

const EMPTY_SNAPSHOT: DesktopSessionSnapshot = {
  connectionState: "disconnected",
  streamState: "idle",
  sessionId: null,
  userId: null,
  muted: false,
  messages: [],
};

type BackendState = "loading" | "ready" | "offline" | "unconfigured";

const elements = {
  backendStatusButton: requireElement<HTMLButtonElement>(
    "backend-status-button",
  ),
  backendStatusDot: requireElement<HTMLElement>("backend-status-dot"),
  backendStatusLabel: requireElement<HTMLElement>("backend-status-label"),
  backendSummary: requireElement<HTMLElement>("backend-summary"),
  backendDetail: requireElement<HTMLElement>("backend-detail"),
  websocketDetail: requireElement<HTMLElement>("websocket-detail"),
  tokenDetail: requireElement<HTMLElement>("token-detail"),
  networkDetail: requireElement<HTMLElement>("network-detail"),
  sessionDetail: requireElement<HTMLElement>("session-detail"),
  userDetail: requireElement<HTMLElement>("user-detail"),
  connectionStateDetail: requireElement<HTMLElement>(
    "connection-state-detail",
  ),
  streamStateDetail: requireElement<HTMLElement>("stream-state-detail"),
  mutedStateDetail: requireElement<HTMLElement>("muted-state-detail"),
  modelConfigDetail: requireElement<HTMLElement>("model-config-detail"),
  modelConfigStatus: requireElement<HTMLElement>("model-config-status"),
  selectModelConfigButton: requireElement<HTMLButtonElement>(
    "select-model-config-button",
  ),
  developerToolsList: requireElement<HTMLElement>("developer-tools-list"),
  developerToolsStatus: requireElement<HTMLElement>(
    "developer-tools-status",
  ),
  installToolDirectoryButton: requireElement<HTMLButtonElement>(
    "install-tool-directory-button",
  ),
  applyToolChangesButton: requireElement<HTMLButtonElement>(
    "apply-tool-changes-button",
  ),
  messages: requireElement<HTMLElement>("messages"),
  textComposer: requireElement<HTMLFormElement>("text-composer"),
  messageInput: requireElement<HTMLTextAreaElement>("message-input"),
  sendTextButton: requireElement<HTMLButtonElement>("send-text-button"),
  composerStatus: requireElement<HTMLElement>("composer-status"),
  errorBanner: requireElement<HTMLElement>("error-banner"),
  orbView: requireElement<HTMLElement>("orb-view"),
  chatView: requireElement<HTMLElement>("chat-view"),
  showChatButton: requireElement<HTMLButtonElement>("show-chat-button"),
  showOrbButton: requireElement<HTMLButtonElement>("show-orb-button"),
  orbTitle: requireElement<HTMLElement>("orb-title"),
  orbCaption: requireElement<HTMLElement>("orb-caption"),
  debugDrawer: requireElement<HTMLElement>("debug-drawer"),
  drawerBackdrop: requireElement<HTMLButtonElement>("drawer-backdrop"),
  toggleDebugButton: requireElement<HTMLButtonElement>(
    "toggle-debug-button",
  ),
  closeDebugButton: requireElement<HTMLButtonElement>("close-debug-button"),
  callButton: requireElement<HTMLButtonElement>("call-button"),
  muteButton: requireElement<HTMLButtonElement>("mute-button"),
  retryButton: requireElement<HTMLButtonElement>("retry-button"),
};

let adapter: XtalkClientAdapter | null = null;
let unsubscribe: (() => void) | null = null;
let discoveringBackend = false;
let sessionOperation = false;
let sendingText = false;
let modelConfigOperation = false;
let toolOperation = false;
let toolChangesPending = false;
let diagnosticsOpen = false;
let backendState: BackendState = "loading";
let modelConfigPath: string | null = null;
let installedTools: NativeToolDefinition[] = [];
let latestSnapshot = EMPTY_SNAPSHOT;

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Required UI element #${id} is missing.`);
  }
  return element as T;
}

function formatError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  return message.replace(
    /([?&]app_token=)[^&\s]+/giu,
    "$1[hidden]",
  );
}

function showError(message: string | null): void {
  elements.errorBanner.hidden = !message;
  elements.errorBanner.textContent = message ?? "";
}

function setBackendStatus(
  state: BackendState,
  label: string,
): void {
  backendState = state;
  elements.backendStatusDot.dataset.state = state;
  elements.backendStatusLabel.textContent = label;
  elements.backendSummary.textContent = state;
  updateOrbPresentation(latestSnapshot);
}

function setDiagnosticsOpen(open: boolean): void {
  diagnosticsOpen = open;
  elements.debugDrawer.classList.toggle("is-open", open);
  elements.drawerBackdrop.classList.toggle("is-visible", open);
  elements.debugDrawer.setAttribute("aria-hidden", String(!open));
  elements.backendStatusButton.setAttribute("aria-expanded", String(open));
  elements.toggleDebugButton.setAttribute("aria-expanded", String(open));

  if (open) {
    elements.closeDebugButton.focus();
  }
}

function setMainView(view: "orb" | "chat"): void {
  const showChat = view === "chat";
  elements.orbView.classList.toggle("is-hidden", showChat);
  elements.chatView.classList.toggle("is-hidden", !showChat);
  elements.orbView.setAttribute("aria-hidden", String(showChat));
  elements.chatView.setAttribute("aria-hidden", String(!showChat));

  if (showChat) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
}

function updateNetworkStatus(): void {
  const online = navigator.onLine;
  elements.networkDetail.textContent = online ? "online" : "offline";
  elements.networkDetail.dataset.state = online ? "ready" : "warning";
  elements.networkDetail.title = online
    ? "远程 provider 可尝试连接"
    : "远程网络不可用；本地界面仍可使用";
}

function updateControls(snapshot: DesktopSessionSnapshot): void {
  const live =
    snapshot.connectionState === "connected" ||
    snapshot.connectionState === "reconnecting";
  const hasBackend = adapter !== null;
  const callAction = live ? "stop" : "start";
  const callLabel = sessionOperation
    ? live
      ? "正在结束对话"
      : "正在开始对话"
    : live
      ? "结束对话"
      : "开始对话";

  elements.callButton.disabled =
    !hasBackend ||
    backendState !== "ready" ||
    discoveringBackend ||
    modelConfigOperation ||
    toolOperation ||
    sessionOperation ||
    sendingText;
  elements.callButton.dataset.action = callAction;
  elements.callButton.classList.toggle("is-loading", sessionOperation);
  elements.callButton.setAttribute("aria-label", callLabel);
  elements.callButton.setAttribute("aria-busy", String(sessionOperation));
  elements.callButton.title = callLabel;

  elements.muteButton.disabled =
    !hasBackend ||
    modelConfigOperation ||
    toolOperation ||
    sessionOperation ||
    !live;
  elements.muteButton.classList.toggle("is-muted", snapshot.muted);
  elements.muteButton.setAttribute("aria-pressed", String(snapshot.muted));
  elements.muteButton.setAttribute(
    "aria-label",
    snapshot.muted ? "打开麦克风" : "关闭麦克风",
  );
  elements.muteButton.title = snapshot.muted ? "打开麦克风" : "关闭麦克风";

  elements.retryButton.disabled =
    discoveringBackend ||
    modelConfigOperation ||
    toolOperation ||
    sessionOperation ||
    sendingText;
  elements.selectModelConfigButton.disabled =
    discoveringBackend ||
    modelConfigOperation ||
    toolOperation ||
    sessionOperation ||
    sendingText;
  updateToolControls();
  updateComposer(snapshot);
}

function updateComposer(snapshot: DesktopSessionSnapshot): void {
  const connected =
    adapter !== null &&
    backendState === "ready" &&
    snapshot.connectionState === "connected" &&
    snapshot.sessionId !== null;
  const available =
    connected &&
    !discoveringBackend &&
    !modelConfigOperation &&
    !toolOperation &&
    !sessionOperation &&
    !sendingText;
  const hasText = elements.messageInput.value.trim().length > 0;
  const placeholder = composerPlaceholder(snapshot);

  elements.messageInput.disabled = !available;
  elements.messageInput.placeholder = placeholder;
  elements.sendTextButton.disabled = !available || !hasText;
  elements.sendTextButton.classList.toggle("is-loading", sendingText);
  elements.sendTextButton.setAttribute("aria-busy", String(sendingText));
  elements.sendTextButton.setAttribute(
    "aria-label",
    sendingText ? "正在发送消息" : "发送消息",
  );
  elements.sendTextButton.title = sendingText ? "正在发送消息" : "发送消息";
  elements.textComposer.dataset.state = available ? "ready" : "unavailable";
  elements.textComposer.setAttribute("aria-busy", String(sendingText));
  if (elements.composerStatus.textContent !== placeholder) {
    elements.composerStatus.textContent = placeholder;
  }
}

function composerPlaceholder(snapshot: DesktopSessionSnapshot): string {
  if (sendingText) {
    return "正在发送消息";
  }
  if (discoveringBackend || backendState === "loading") {
    return "本地服务启动中";
  }
  if (backendState === "unconfigured") {
    return "请先选择模型配置";
  }
  if (backendState === "offline" || adapter === null) {
    return "本地服务不可用";
  }
  if (sessionOperation) {
    return snapshot.connectionState === "disconnected"
      ? "正在连接对话"
      : "正在更新连接";
  }
  if (snapshot.connectionState === "reconnecting") {
    return "正在恢复连接";
  }
  if (snapshot.connectionState === "disconnected") {
    return "连接对话后可发送文字";
  }
  if (snapshot.sessionId === null) {
    return "正在初始化会话";
  }
  return "输入消息";
}

function resizeMessageInput(): void {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(
    elements.messageInput.scrollHeight,
    120,
  )}px`;
}

function updateOrbPresentation(snapshot: DesktopSessionSnapshot): void {
  for (const orb of [elements.showChatButton, elements.showOrbButton]) {
    orb.dataset.connectionState = snapshot.connectionState;
    orb.dataset.streamState = snapshot.streamState;
    orb.classList.toggle("is-muted", snapshot.muted);
  }

  if (backendState === "loading") {
    elements.orbTitle.textContent = "本地服务启动中";
    elements.orbCaption.textContent = "界面可离线使用";
    return;
  }

  if (backendState === "unconfigured") {
    elements.orbTitle.textContent = "需要模型配置";
    elements.orbCaption.textContent = "请在设置与诊断中选择 JSON 配置";
    return;
  }

  if (backendState === "offline") {
    elements.orbTitle.textContent = "本地服务不可用";
    elements.orbCaption.textContent = "打开设置与诊断后可重新配置";
    return;
  }

  if (snapshot.connectionState === "disconnected") {
    elements.orbTitle.textContent = "准备开始对话";
    elements.orbCaption.textContent = "点击下方波形按钮连接";
    return;
  }

  if (snapshot.connectionState === "reconnecting") {
    elements.orbTitle.textContent = "正在恢复连接";
    elements.orbCaption.textContent = "对话将在连接恢复后继续";
    return;
  }

  if (snapshot.muted) {
    elements.orbTitle.textContent = "麦克风已静音";
    elements.orbCaption.textContent = "点击底部麦克风按钮恢复";
    return;
  }

  switch (snapshot.streamState) {
    case "listening":
      elements.orbTitle.textContent = "正在聆听";
      break;
    case "processing":
      elements.orbTitle.textContent = "正在思考";
      break;
    case "speaking":
      elements.orbTitle.textContent = "正在播放";
      break;
    case "idle":
      elements.orbTitle.textContent = "对话已连接";
      break;
  }
  elements.orbCaption.textContent = "点击圆球查看对话";
}

function renderSnapshot(snapshot: DesktopSessionSnapshot): void {
  latestSnapshot = snapshot;
  elements.connectionStateDetail.textContent = snapshot.connectionState;
  elements.streamStateDetail.textContent = snapshot.streamState;
  elements.mutedStateDetail.textContent = String(snapshot.muted);
  elements.sessionDetail.textContent = snapshot.sessionId ?? "--";
  elements.userDetail.textContent = snapshot.userId ?? "--";

  const messageElements = snapshot.messages.map((message) => {
    const row = document.createElement("article");
    row.className = `message-row message-row-${message.role}`;
    row.setAttribute("aria-label", messageRoleLabel(message.role));

    const body = document.createElement("div");
    body.className = `message message-${message.role}`;
    body.dataset.final = String(message.final);

    const content = document.createElement("span");
    content.className = "message-content";
    content.textContent = message.content;

    body.append(content);
    row.append(body);
    return row;
  });

  elements.messages.replaceChildren(...messageElements);
  if (messageElements.length > 0) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }

  updateOrbPresentation(snapshot);
  updateControls(snapshot);
}

function messageRoleLabel(
  role: DesktopSessionSnapshot["messages"][number]["role"],
): string {
  switch (role) {
    case "user":
      return "你";
    case "assistant":
      return "XTalk";
    case "info":
      return "系统";
  }
}

function renderModelConfigSelection(
  selection: NativeModelConfigSelection,
): void {
  modelConfigPath = selection.configPath;
  elements.modelConfigDetail.textContent =
    selection.configPath ?? "未选择";
  elements.modelConfigDetail.title = selection.configPath ?? "";
  elements.modelConfigStatus.textContent = selection.configPath
    ? "已选择；更换文件会重启本地服务"
    : "尚未选择模型配置";
  updateControls(latestSnapshot);
}

function renderInstalledTools(tools: NativeToolDefinition[]): void {
  installedTools = tools;
  if (tools.length === 0) {
    const empty = document.createElement("p");
    empty.className = "developer-tools-empty";
    empty.textContent = "尚未安装开发者工具";
    elements.developerToolsList.replaceChildren(empty);
    updateToolControls();
    return;
  }

  const rows = tools.map((tool) => {
    const row = document.createElement("article");
    row.className = "developer-tool-row";

    const copy = document.createElement("div");
    copy.className = "developer-tool-copy";

    const name = document.createElement("strong");
    name.textContent = tool.displayName;

    const entrypoint = document.createElement("code");
    entrypoint.textContent = tool.entrypoint;

    copy.append(name, entrypoint);

    const actions = document.createElement("div");
    actions.className = "developer-tool-actions";

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "developer-tool-toggle";

    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = tool.enabled;
    toggle.setAttribute(
      "aria-label",
      `${tool.enabled ? "禁用" : "启用"}${tool.displayName}`,
    );
    toggle.addEventListener("change", () => {
      void updateInstalledToolEnabled(tool.id, toggle.checked);
    });

    const toggleText = document.createElement("span");
    toggleText.textContent = "启用";
    toggleLabel.append(toggle, toggleText);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "developer-tool-remove";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `删除${tool.displayName}`);
    remove.title = "删除已复制的工具";
    remove.addEventListener("click", () => {
      void removeInstalledTool(tool.id);
    });

    actions.append(toggleLabel, remove);
    row.append(copy, actions);
    return row;
  });

  elements.developerToolsList.replaceChildren(...rows);
  updateToolControls();
}

function updateToolControls(): void {
  const busy =
    toolOperation ||
    modelConfigOperation ||
    discoveringBackend ||
    sessionOperation ||
    sendingText;
  elements.installToolDirectoryButton.disabled = busy;
  elements.applyToolChangesButton.disabled =
    busy || !toolChangesPending || modelConfigPath === null;
  for (const control of elements.developerToolsList.querySelectorAll<
    HTMLInputElement | HTMLButtonElement
  >("input, button")) {
    control.disabled = busy;
  }
}

function updateDeveloperToolsStatus(message?: string): void {
  if (message) {
    elements.developerToolsStatus.textContent = message;
    return;
  }
  if (toolChangesPending) {
    elements.developerToolsStatus.textContent =
      "工具配置已修改；应用并重启本地服务后生效";
    return;
  }
  elements.developerToolsStatus.textContent =
    installedTools.length === 0
      ? "尚未安装开发者工具"
      : `已安装 ${installedTools.length} 个开发者工具`;
}

async function refreshModelConfigSelection(): Promise<NativeModelConfigSelection> {
  const selection = await getNativeModelConfigSelection();
  renderModelConfigSelection(selection);
  return selection;
}

async function refreshInstalledTools(): Promise<NativeToolDefinition[]> {
  const tools = await getNativeInstalledTools();
  renderInstalledTools(tools);
  updateDeveloperToolsStatus();
  return tools;
}

async function detachCurrentAdapter(): Promise<void> {
  const previousAdapter = adapter;
  unsubscribe?.();
  unsubscribe = null;
  adapter = null;
  renderSnapshot(EMPTY_SNAPSHOT);
  if (previousAdapter) {
    await previousAdapter.disconnect().catch(() => undefined);
  }
}

async function discoverBackend(): Promise<void> {
  if (discoveringBackend) {
    return;
  }
  if (modelConfigPath === null) {
    setBackendStatus("unconfigured", "请选择模型配置");
    elements.backendDetail.textContent = "尚未启动";
    elements.websocketDetail.textContent = "未配置";
    elements.tokenDetail.textContent = "未获取";
    updateControls(latestSnapshot);
    return;
  }

  discoveringBackend = true;
  showError(null);
  setBackendStatus("loading", "正在查找本地服务");
  elements.backendDetail.textContent = "等待 Tauri 提供 endpoint";
  elements.websocketDetail.textContent = "未配置";
  elements.tokenDetail.textContent = "未获取";
  updateControls(latestSnapshot);

  await detachCurrentAdapter();

  try {
    const connection = await getNativeBackendConnection();
    const nextAdapter = new XtalkClientAdapter(connection);
    adapter = nextAdapter;
    unsubscribe = nextAdapter.subscribe(renderSnapshot);

    elements.backendDetail.textContent = nextAdapter.diagnostics.origin;
    elements.websocketDetail.textContent = nextAdapter.diagnostics.websocketURL;
    elements.tokenDetail.textContent =
      nextAdapter.diagnostics.httpEndpointsAuthenticated
        ? "已配置（值已隐藏）"
        : "缺失";
    setBackendStatus("ready", "本地服务已就绪");
  } catch (error) {
    adapter = null;
    setBackendStatus("offline", "本地服务不可用");
    elements.backendDetail.textContent = "离线模式";
    showError(`无法连接本地服务：${formatError(error)}`);
    setDiagnosticsOpen(true);
  } finally {
    discoveringBackend = false;
    updateControls(latestSnapshot);
  }
}

async function chooseAndApplyModelConfig(required: boolean): Promise<void> {
  if (modelConfigOperation) {
    return;
  }

  modelConfigOperation = true;
  showError(null);
  elements.modelConfigStatus.textContent = "等待选择 JSON 配置";
  updateControls(latestSnapshot);

  try {
    const selectedPath = await chooseNativeModelConfigFile();
    if (selectedPath === null) {
      if (required && modelConfigPath === null) {
        setBackendStatus("unconfigured", "请选择模型配置");
        elements.modelConfigStatus.textContent =
          "首次启动需要选择模型配置文件";
      } else {
        elements.modelConfigStatus.textContent = modelConfigPath
          ? "已取消；继续使用当前配置"
          : "已取消；尚未选择模型配置";
      }
      return;
    }

    elements.modelConfigStatus.textContent = "正在重启本地服务";
    setBackendStatus("loading", "正在应用模型配置");
    await detachCurrentAdapter();
    await applyNativeModelConfig(selectedPath);
    toolChangesPending = false;
    const selection = await refreshModelConfigSelection();
    updateDeveloperToolsStatus();
    elements.modelConfigStatus.textContent = selection.configPath
      ? "配置已应用，本地服务已重启"
      : "配置已应用";
    await discoverBackend();
  } catch (error) {
    const message = `模型配置应用失败：${formatError(error)}`;
    await refreshModelConfigSelection().catch(() => undefined);
    if (modelConfigPath === null) {
      setBackendStatus("unconfigured", "请选择模型配置");
    } else {
      await discoverBackend();
    }
    elements.modelConfigStatus.textContent = "配置应用失败";
    showError(message);
    setDiagnosticsOpen(true);
  } finally {
    modelConfigOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

async function chooseAndInstallToolDirectory(): Promise<void> {
  if (toolOperation) {
    return;
  }

  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus("等待选择工具目录");
  updateControls(latestSnapshot);

  try {
    const selectedPath = await chooseNativeToolDirectory();
    if (selectedPath === null) {
      updateDeveloperToolsStatus("已取消安装工具");
      return;
    }

    updateDeveloperToolsStatus("正在复制工具目录到 AppData");
    const installed = await installNativeToolDirectory(selectedPath);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      `${installed.displayName} 已安装；重启本地服务后生效`,
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus("工具目录安装失败");
    showError(`工具目录安装失败：${formatError(error)}`);
  } finally {
    toolOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

async function updateInstalledToolEnabled(
  toolId: string,
  enabled: boolean,
): Promise<void> {
  if (toolOperation) {
    return;
  }

  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus("正在更新工具状态");
  updateControls(latestSnapshot);

  try {
    const updated = await setNativeToolEnabled(toolId, enabled);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      `${updated.displayName} 已${updated.enabled ? "启用" : "禁用"}；重启本地服务后生效`,
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus("工具状态更新失败");
    showError(`工具状态更新失败：${formatError(error)}`);
  } finally {
    toolOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

async function removeInstalledTool(toolId: string): Promise<void> {
  if (toolOperation) {
    return;
  }

  const tool = installedTools.find((candidate) => candidate.id === toolId);
  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus("正在删除已复制的工具");
  updateControls(latestSnapshot);

  try {
    await removeNativeInstalledTool(toolId);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      `${tool?.displayName ?? "工具"}已删除；重启本地服务后生效`,
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus("工具删除失败");
    showError(`工具删除失败：${formatError(error)}`);
  } finally {
    toolOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

async function applyInstalledToolChanges(): Promise<void> {
  if (toolOperation || !toolChangesPending) {
    return;
  }
  if (modelConfigPath === null) {
    updateDeveloperToolsStatus("请先选择模型配置");
    return;
  }

  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus("正在重启本地服务并加载工具");
  setBackendStatus("loading", "正在应用开发者工具");
  updateControls(latestSnapshot);

  try {
    await detachCurrentAdapter();
    await applyNativeToolChanges();
    toolChangesPending = false;
    updateDeveloperToolsStatus("工具配置已应用，本地服务已重启");
    await discoverBackend();
  } catch (error) {
    const message = `工具配置应用失败：${formatError(error)}`;
    updateDeveloperToolsStatus("工具配置应用失败");
    await discoverBackend();
    showError(message);
    setDiagnosticsOpen(true);
  } finally {
    toolOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

async function initializeApplication(): Promise<void> {
  try {
    await refreshInstalledTools();
    const selection = await refreshModelConfigSelection();
    if (selection.configPath === null) {
      setBackendStatus("unconfigured", "请选择模型配置");
      setDiagnosticsOpen(true);
      await chooseAndApplyModelConfig(true);
      return;
    }
    await discoverBackend();
  } catch (error) {
    setBackendStatus("offline", "桌面运行时不可用");
    elements.modelConfigStatus.textContent = "无法读取模型配置状态";
    showError(`无法初始化应用：${formatError(error)}`);
    setDiagnosticsOpen(true);
  }
}

async function connectSession(): Promise<void> {
  const activeAdapter = adapter;
  if (!activeAdapter || sessionOperation) {
    return;
  }

  sessionOperation = true;
  showError(null);
  updateControls(activeAdapter.snapshot);
  try {
    await activeAdapter.connect();
  } catch (error) {
    showError(`会话连接失败：${formatError(error)}`);
  } finally {
    sessionOperation = false;
    updateControls(activeAdapter.snapshot);
  }
}

async function disconnectSession(): Promise<void> {
  const activeAdapter = adapter;
  if (!activeAdapter || sessionOperation) {
    return;
  }

  sessionOperation = true;
  showError(null);
  updateControls(activeAdapter.snapshot);
  try {
    await activeAdapter.disconnect();
  } catch (error) {
    showError(`会话关闭失败：${formatError(error)}`);
  } finally {
    sessionOperation = false;
    updateControls(activeAdapter.snapshot);
  }
}

async function sendTextMessage(): Promise<void> {
  const activeAdapter = adapter;
  const text = elements.messageInput.value.trim();
  if (
    !activeAdapter ||
    sendingText ||
    discoveringBackend ||
    sessionOperation ||
    backendState !== "ready" ||
    activeAdapter.snapshot.connectionState !== "connected" ||
    activeAdapter.snapshot.sessionId === null ||
    text.length === 0
  ) {
    return;
  }

  sendingText = true;
  showError(null);
  updateControls(activeAdapter.snapshot);
  try {
    await activeAdapter.sendText(text);
    elements.messageInput.value = "";
    resizeMessageInput();
  } catch (error) {
    showError(`消息发送失败：${formatError(error)}`);
  } finally {
    sendingText = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
    if (
      adapter === activeAdapter &&
      activeAdapter.snapshot.connectionState === "connected"
    ) {
      elements.messageInput.focus();
    }
  }
}

elements.backendStatusButton.addEventListener("click", () => {
  setDiagnosticsOpen(!diagnosticsOpen);
});
elements.toggleDebugButton.addEventListener("click", () => {
  setDiagnosticsOpen(!diagnosticsOpen);
});
elements.closeDebugButton.addEventListener("click", () => {
  setDiagnosticsOpen(false);
});
elements.drawerBackdrop.addEventListener("click", () => {
  setDiagnosticsOpen(false);
});
elements.showChatButton.addEventListener("click", () => {
  setMainView("chat");
});
elements.showOrbButton.addEventListener("click", () => {
  setMainView("orb");
});
elements.callButton.addEventListener("click", () => {
  if (
    adapter?.snapshot.connectionState === "connected" ||
    adapter?.snapshot.connectionState === "reconnecting"
  ) {
    void disconnectSession();
  } else {
    void connectSession();
  }
});
elements.muteButton.addEventListener("click", () => {
  if (adapter) {
    adapter.setMuted(!adapter.snapshot.muted);
  }
});
elements.selectModelConfigButton.addEventListener("click", () => {
  void chooseAndApplyModelConfig(false);
});
elements.installToolDirectoryButton.addEventListener("click", () => {
  void chooseAndInstallToolDirectory();
});
elements.applyToolChangesButton.addEventListener("click", () => {
  void applyInstalledToolChanges();
});
elements.textComposer.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendTextMessage();
});
elements.messageInput.addEventListener("input", () => {
  resizeMessageInput();
  updateComposer(latestSnapshot);
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!elements.sendTextButton.disabled) {
      void sendTextMessage();
    }
  }
});
elements.retryButton.addEventListener("click", () => {
  if (modelConfigPath === null) {
    void chooseAndApplyModelConfig(true);
  } else {
    void discoverBackend();
  }
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && diagnosticsOpen) {
    setDiagnosticsOpen(false);
  }
});
window.addEventListener("online", updateNetworkStatus);
window.addEventListener("offline", updateNetworkStatus);

updateNetworkStatus();
renderSnapshot(EMPTY_SNAPSHOT);
resizeMessageInput();
void initializeApplication();
