import "./styles.css";

import {
  applyNativeModelConfig,
  applyNativeToolChanges,
  chooseNativeModelConfigFile,
  chooseNativeToolDirectory,
  getNativeManagedModelPlan,
  getNativeBackendConnection,
  getNativeInstalledTools,
  getNativeModelConfigSelection,
  installNativeToolDirectory,
  listenNativeManagedModelProgress,
  removeNativeInstalledTool,
  setNativeToolEnabled,
  type NativeManagedModelProgress,
  type NativeModelConfigSelection,
  type NativeToolDefinition,
} from "./adapters/native-capabilities";
import {
  XtalkClientAdapter,
  type DesktopSessionSnapshot,
  type DesktopSessionSummary,
} from "./adapters/xtalk-client-adapter";
import {
  getLanguagePreference,
  getResolvedLanguage,
  localizeKnownError,
  refreshAutomaticLanguage,
  setLanguagePreference,
  t,
  translateDocument,
  type LanguagePreference,
  type TranslationKey,
} from "./i18n";

const EMPTY_SNAPSHOT: DesktopSessionSnapshot = {
  connectionState: "disconnected",
  streamState: "idle",
  sessionId: null,
  userId: null,
  muted: false,
  messages: [],
};

type BackendState = "loading" | "ready" | "offline" | "unconfigured";

const BACKEND_SUMMARY_KEYS: Record<BackendState, TranslationKey> = {
  loading: "service.summary.loading",
  ready: "service.summary.ready",
  offline: "service.summary.offline",
  unconfigured: "service.summary.unconfigured",
};

const elements = {
  app: requireElement<HTMLElement>("app"),
  backendStatusDot: requireElement<HTMLElement>("backend-status-dot"),
  backendStatusLabel: requireElement<HTMLElement>("backend-status-label"),
  backendSummary: requireElement<HTMLElement>("backend-summary"),
  backendDetail: requireElement<HTMLElement>("backend-detail"),
  websocketDetail: requireElement<HTMLElement>("websocket-detail"),
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
  chatSidebar: requireElement<HTMLElement>("chat-sidebar"),
  sidebarBackdrop: requireElement<HTMLButtonElement>("sidebar-backdrop"),
  toggleSidebarButton: requireElement<HTMLButtonElement>(
    "toggle-sidebar-button",
  ),
  newChatButton: requireElement<HTMLButtonElement>("new-chat-button"),
  openToolsButton: requireElement<HTMLButtonElement>("open-tools-button"),
  chatSessionList: requireElement<HTMLElement>("chat-session-list"),
  chatSessionListStatus: requireElement<HTMLElement>(
    "chat-session-list-status",
  ),
  debugDrawer: requireElement<HTMLElement>("debug-drawer"),
  drawerBackdrop: requireElement<HTMLButtonElement>("drawer-backdrop"),
  toggleDebugButton: requireElement<HTMLButtonElement>(
    "toggle-debug-button",
  ),
  closeDebugButton: requireElement<HTMLButtonElement>("close-debug-button"),
  toolsDialog: requireElement<HTMLElement>("tools-dialog"),
  toolsDialogBackdrop: requireElement<HTMLButtonElement>(
    "tools-dialog-backdrop",
  ),
  closeToolsButton: requireElement<HTMLButtonElement>("close-tools-button"),
  managedProgressBackdrop: requireElement<HTMLElement>(
    "managed-progress-backdrop",
  ),
  managedProgressDialog: requireElement<HTMLElement>(
    "managed-progress-dialog",
  ),
  managedProgressMessage: requireElement<HTMLElement>(
    "managed-progress-message",
  ),
  managedProgressBar: requireElement<HTMLProgressElement>(
    "managed-progress-bar",
  ),
  managedProgressDetail: requireElement<HTMLElement>(
    "managed-progress-detail",
  ),
  managedProgressPercent: requireElement<HTMLElement>(
    "managed-progress-percent",
  ),
  managedProgressServices: requireElement<HTMLOListElement>(
    "managed-progress-services",
  ),
  managedProgressError: requireElement<HTMLElement>(
    "managed-progress-error",
  ),
  closeManagedProgressButton: requireElement<HTMLButtonElement>(
    "close-managed-progress-button",
  ),
  callButton: requireElement<HTMLButtonElement>("call-button"),
  muteButton: requireElement<HTMLButtonElement>("mute-button"),
  retryButton: requireElement<HTMLButtonElement>("retry-button"),
  languageSelect: requireElement<HTMLSelectElement>("language-select"),
  languageSummary: requireElement<HTMLElement>("language-summary"),
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
let toolsDialogOpen = false;
let managedProgressState: "closed" | "running" | "failed" = "closed";
let managedProgressServiceIds: string[] = [];
let latestManagedProgress: NativeManagedModelProgress | null = null;
let managedProgressFailure: unknown = null;
let sidebarOpen = false;
let sessionListOperation = false;
let backendState: BackendState = "loading";
let modelConfigPath: string | null = null;
let installedTools: NativeToolDefinition[] = [];
let persistedSessions: DesktopSessionSummary[] = [];
let sessionListError: string | null = null;
let latestSnapshot = EMPTY_SNAPSHOT;
let sessionRefreshTimer: ReturnType<typeof setTimeout> | null = null;
let sessionActivityKey = "";
let backendStatusKey: TranslationKey = "service.starting";
let visibleError:
  | {
      key: TranslationKey;
      parameters: Readonly<Record<string, unknown>>;
    }
  | null = null;

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Required UI element #${id} is missing.`);
  }
  return element as T;
}

function formatError(error: unknown): string {
  if (!("__TAURI_INTERNALS__" in globalThis)) {
    return t("native.runtimeUnavailable");
  }
  const message = error instanceof Error ? error.message : String(error);
  return localizeKnownError(
    message.replace(
      /([?&]app_token=)[^&\s]+/giu,
      "$1[hidden]",
    ),
  );
}

function showError(
  key: TranslationKey | null,
  parameters: Readonly<Record<string, unknown>> = {},
): void {
  visibleError = key === null ? null : { key, parameters };
  renderVisibleError();
}

function renderVisibleError(): void {
  elements.errorBanner.hidden = visibleError === null;
  if (visibleError === null) {
    elements.errorBanner.textContent = "";
    return;
  }
  const parameters = Object.fromEntries(
    Object.entries(visibleError.parameters).map(([name, value]) => [
      name,
      name === "error" ? formatError(value) : String(value),
    ]),
  );
  elements.errorBanner.textContent = t(visibleError.key, parameters);
}

function setBackendStatus(
  state: BackendState,
  labelKey: TranslationKey,
): void {
  backendState = state;
  backendStatusKey = labelKey;
  elements.backendStatusDot.dataset.state = state;
  elements.backendStatusLabel.textContent = t(labelKey);
  elements.backendSummary.textContent = t(BACKEND_SUMMARY_KEYS[state]);
  updateOrbPresentation(latestSnapshot);
}

function applyUiLanguage(): void {
  translateDocument();
  renderVisibleError();
  const preference = getLanguagePreference();
  elements.languageSelect.value = preference;
  elements.languageSummary.textContent =
    preference === "auto"
      ? t("language.auto")
      : t(getResolvedLanguage() === "zh-CN" ? "language.zhCN" : "language.en");

  setBackendStatus(backendState, backendStatusKey);
  if (backendState === "unconfigured") {
    elements.backendDetail.textContent = t("service.notStarted");
    elements.websocketDetail.textContent = t("service.notConfigured");
  } else if (backendState === "loading") {
    elements.backendDetail.textContent = t("service.waitingEndpoint");
    elements.websocketDetail.textContent = t("service.notConfigured");
  } else if (backendState === "offline") {
    elements.backendDetail.textContent = t("service.offlineMode");
  }

  updateNetworkStatus();
  renderModelConfigSelection({ configPath: modelConfigPath });
  renderInstalledTools(installedTools);
  updateDeveloperToolsStatus();
  renderSnapshot(latestSnapshot);
  renderManagedProgress();
}

function isCompactLayout(): boolean {
  return window.matchMedia("(max-width: 760px)").matches;
}

function setSidebarOpen(open: boolean, moveFocus = true): void {
  sidebarOpen = open;
  elements.app.classList.toggle("sidebar-open", open);
  elements.chatSidebar.setAttribute("aria-hidden", String(!open));
  elements.toggleSidebarButton.setAttribute("aria-expanded", String(open));
  elements.toggleSidebarButton.setAttribute(
    "aria-label",
    t(open ? "sidebar.collapse" : "sidebar.expand"),
  );

  if (open && isCompactLayout()) {
    setDiagnosticsOpen(false);
  }
  if (moveFocus) {
    elements.toggleSidebarButton.focus();
  }
}

function setDiagnosticsOpen(open: boolean): void {
  if (open) {
    setToolsDialogOpen(false, false);
  }
  diagnosticsOpen = open;
  elements.debugDrawer.classList.toggle("is-open", open);
  elements.drawerBackdrop.classList.toggle("is-visible", open);
  elements.debugDrawer.setAttribute("aria-hidden", String(!open));
  elements.toggleDebugButton.setAttribute("aria-expanded", String(open));

  if (open && isCompactLayout()) {
    setSidebarOpen(false, false);
  }
  if (open) {
    elements.closeDebugButton.focus();
  }
}

function setToolsDialogOpen(open: boolean, moveFocus = true): void {
  toolsDialogOpen = open;
  elements.toolsDialog.classList.toggle("is-open", open);
  elements.toolsDialogBackdrop.classList.toggle("is-visible", open);
  elements.toolsDialog.setAttribute("aria-hidden", String(!open));
  elements.openToolsButton.setAttribute("aria-expanded", String(open));

  if (open) {
    setDiagnosticsOpen(false);
    if (isCompactLayout()) {
      setSidebarOpen(false, false);
    }
    elements.closeToolsButton.focus();
  } else if (moveFocus) {
    elements.openToolsButton.focus();
  }
}

function managedServiceName(serviceId: string): string {
  switch (serviceId) {
    case "sensevoice-small":
      return "SenseVoice Small";
    case "sensevoice-small-mlx":
      return "SenseVoice Small (MLX)";
    case "moss-tts-nano":
      return "MOSS-TTS-Nano";
    case "moss-tts-nano-mlx":
      return "MOSS-TTS-Nano (MLX)";
    default:
      return serviceId;
  }
}

function setApplicationInert(inert: boolean): void {
  for (const child of elements.app.children) {
    if (
      child === elements.managedProgressBackdrop ||
      child === elements.managedProgressDialog
    ) {
      continue;
    }
    if (child instanceof HTMLElement) {
      child.inert = inert;
    }
  }
}

function openManagedProgress(serviceIds: string[]): void {
  setToolsDialogOpen(false, false);
  managedProgressState = "running";
  managedProgressServiceIds = serviceIds;
  latestManagedProgress = null;
  managedProgressFailure = null;
  elements.managedProgressDialog.dataset.state = "running";
  elements.managedProgressError.hidden = true;
  elements.closeManagedProgressButton.hidden = true;
  elements.managedProgressBar.removeAttribute("value");
  elements.managedProgressDetail.textContent = "";
  elements.managedProgressPercent.textContent = "";
  elements.managedProgressBackdrop.classList.add("is-visible");
  elements.managedProgressDialog.classList.add("is-open");
  elements.managedProgressBackdrop.setAttribute("aria-hidden", "false");
  elements.managedProgressDialog.setAttribute("aria-hidden", "false");
  setApplicationInert(true);
  renderManagedProgress();
  elements.managedProgressDialog.focus();
}

function closeManagedProgress(): void {
  managedProgressState = "closed";
  managedProgressServiceIds = [];
  latestManagedProgress = null;
  managedProgressFailure = null;
  elements.managedProgressBackdrop.classList.remove("is-visible");
  elements.managedProgressDialog.classList.remove("is-open");
  elements.managedProgressBackdrop.setAttribute("aria-hidden", "true");
  elements.managedProgressDialog.setAttribute("aria-hidden", "true");
  setApplicationInert(false);
}

function failManagedProgress(error: unknown): void {
  managedProgressState = "failed";
  managedProgressFailure = error;
  elements.managedProgressDialog.dataset.state = "failed";
  elements.managedProgressBar.removeAttribute("value");
  elements.managedProgressDetail.textContent = "";
  elements.managedProgressPercent.textContent = "";
  elements.closeManagedProgressButton.hidden = false;
  renderManagedProgress();
  elements.closeManagedProgressButton.focus();
}

function updateManagedProgress(progress: NativeManagedModelProgress): void {
  if (managedProgressState !== "running") {
    return;
  }
  latestManagedProgress = progress;
  renderManagedProgress();
}

function renderManagedProgress(): void {
  if (managedProgressState === "closed") {
    return;
  }

  const progress = latestManagedProgress;
  let activeIndex = progress?.serviceIndex ?? 0;
  if (progress?.phase === "complete") {
    activeIndex = managedProgressServiceIds.length;
  }
  const rows = managedProgressServiceIds.map((serviceId, index) => {
    const row = document.createElement("li");
    row.className = "managed-progress-service";
    row.textContent = managedServiceName(serviceId);
    const oneBasedIndex = index + 1;
    const isReady =
      progress?.phase === "complete" ||
      oneBasedIndex < activeIndex ||
      (oneBasedIndex === activeIndex && progress?.phase === "ready");
    row.dataset.state = isReady
      ? "ready"
      : oneBasedIndex === activeIndex
        ? "active"
        : "pending";
    return row;
  });
  elements.managedProgressServices.replaceChildren(...rows);

  if (managedProgressState === "failed") {
    elements.managedProgressMessage.textContent = t("model.applyFailed");
    elements.managedProgressError.textContent = t("managed.failed", {
      error: formatError(managedProgressFailure),
    });
    elements.managedProgressError.hidden = false;
    return;
  }

  elements.managedProgressError.hidden = true;
  if (progress === null) {
    elements.managedProgressMessage.textContent = t("managed.preparing");
    return;
  }

  const service = progress.serviceId
    ? managedServiceName(progress.serviceId)
    : "";
  switch (progress.phase) {
    case "checking":
      elements.managedProgressMessage.textContent = t("managed.checking", {
        service,
      });
      break;
    case "downloading":
      elements.managedProgressMessage.textContent = t("managed.downloading", {
        service,
      });
      break;
    case "starting":
      elements.managedProgressMessage.textContent = t("managed.starting", {
        service,
      });
      break;
    case "ready":
      elements.managedProgressMessage.textContent = t("managed.ready", {
        service,
      });
      break;
    case "complete":
      elements.managedProgressMessage.textContent = t("managed.finalizing");
      break;
  }

  const serviceCount = Math.max(progress.serviceCount, 1);
  let fraction = Math.max(progress.serviceIndex - 1, 0) / serviceCount;
  if (progress.phase === "downloading" && progress.totalBytes > 0) {
    fraction =
      (progress.serviceIndex -
        1 +
        Math.min(progress.completedBytes / progress.totalBytes, 1) * 0.82) /
      serviceCount;
  } else if (progress.phase === "starting") {
    fraction = (progress.serviceIndex - 0.12) / serviceCount;
  } else if (progress.phase === "ready") {
    fraction = progress.serviceIndex / serviceCount;
  } else if (progress.phase === "complete") {
    fraction = 1;
  }
  const percent = Math.max(0, Math.min(100, Math.round(fraction * 100)));
  elements.managedProgressBar.value = percent;
  elements.managedProgressPercent.textContent = `${percent}%`;

  if (progress.phase === "downloading") {
    const filename = progress.filePath?.split("/").slice(-1)[0] ?? "";
    elements.managedProgressDetail.textContent = [
      filename,
      `${formatBytes(progress.completedBytes)} / ${formatBytes(progress.totalBytes)}`,
    ]
      .filter(Boolean)
      .join(" · ");
  } else {
    elements.managedProgressDetail.textContent =
      progress.phase === "complete"
        ? ""
        : `${progress.serviceIndex} / ${progress.serviceCount}`;
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} ${unit}`;
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

function renderChatSessions(): void {
  const activeSessionId = latestSnapshot.sessionId;
  const rows = persistedSessions.map((session) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chat-session-button";
    button.dataset.sessionId = session.id;
    button.classList.toggle("is-active", session.id === activeSessionId);
    button.setAttribute(
      "aria-current",
      session.id === activeSessionId ? "page" : "false",
    );

    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("aria-hidden", "true");
    const iconPath = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    iconPath.setAttribute("d", "M5 5h14v11H9l-4 3Z");
    icon.append(iconPath);

    const title = document.createElement("span");
    title.textContent = session.title?.trim() || t("sidebar.newConversation");
    title.title = title.textContent;

    button.append(icon, title);
    button.addEventListener("click", () => {
      void switchChatSession(session.id);
    });
    return button;
  });

  elements.chatSessionList.replaceChildren(...rows);
  elements.newChatButton.classList.toggle(
    "is-active",
    activeSessionId === null,
  );
  elements.newChatButton.setAttribute(
    "aria-current",
    activeSessionId === null ? "page" : "false",
  );

  if (!sessionListOperation) {
    elements.chatSessionListStatus.textContent =
      sessionListError ?? (rows.length === 0 ? t("sidebar.empty") : "");
  }
  updateSessionControls();
}

function updateSessionControls(): void {
  const unavailable =
    adapter === null ||
    backendState !== "ready" ||
    discoveringBackend ||
    modelConfigOperation ||
    toolOperation ||
    sessionOperation ||
    sendingText;
  elements.newChatButton.disabled = unavailable;
  for (const button of elements.chatSessionList.querySelectorAll<HTMLButtonElement>(
    "button",
  )) {
    button.disabled = unavailable || sessionListOperation;
  }
}

async function refreshChatSessions(): Promise<void> {
  const activeAdapter = adapter;
  if (!activeAdapter || sessionListOperation) {
    if (!activeAdapter) {
      elements.chatSessionListStatus.textContent =
        backendState === "unconfigured"
          ? t("sidebar.waitingForConfig")
          : t("sidebar.waitingForService");
      updateSessionControls();
    }
    return;
  }

  sessionListOperation = true;
  sessionListError = null;
  elements.chatSessionListStatus.textContent = t("sidebar.loading");
  updateSessionControls();
  try {
    const sessions = await activeAdapter.getSessions();
    if (adapter === activeAdapter) {
      persistedSessions = sessions;
      renderChatSessions();
    }
  } catch (error) {
    if (adapter === activeAdapter) {
      sessionListError = t("sidebar.readFailed", {
        error: formatError(error),
      });
    }
  } finally {
    sessionListOperation = false;
    if (adapter === activeAdapter) {
      renderChatSessions();
    }
  }
}

function scheduleChatSessionsRefresh(): void {
  if (sessionRefreshTimer !== null) {
    clearTimeout(sessionRefreshTimer);
  }
  sessionRefreshTimer = setTimeout(() => {
    sessionRefreshTimer = null;
    void refreshChatSessions();
  }, 350);
}

async function switchChatSession(sessionId: string | null): Promise<void> {
  const activeAdapter = adapter;
  if (
    !activeAdapter ||
    backendState !== "ready" ||
    sessionOperation ||
    sendingText
  ) {
    return;
  }

  if (sessionId === latestSnapshot.sessionId) {
    setMainView(sessionId === null ? "orb" : "chat");
    if (isCompactLayout()) {
      setSidebarOpen(false);
    }
    return;
  }

  sessionOperation = true;
  showError(null);
  elements.chatSessionListStatus.textContent =
    t(sessionId === null ? "sidebar.creating" : "sidebar.switching");
  updateControls(activeAdapter.snapshot);
  try {
    await activeAdapter.switchSession(sessionId);
    setMainView(sessionId === null ? "orb" : "chat");
    await refreshChatSessions();
    if (isCompactLayout()) {
      setSidebarOpen(false);
    }
  } catch (error) {
    showError(
      sessionId === null ? "sidebar.createFailed" : "sidebar.switchFailed",
      { error },
    );
  } finally {
    sessionOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
    renderChatSessions();
  }
}

function updateNetworkStatus(): void {
  const online = navigator.onLine;
  elements.networkDetail.textContent = online ? "online" : "offline";
  elements.networkDetail.dataset.state = online ? "ready" : "warning";
  elements.networkDetail.title = online
    ? t("runtime.onlineTitle")
    : t("runtime.offlineTitle");
}

function updateControls(snapshot: DesktopSessionSnapshot): void {
  const live =
    snapshot.connectionState === "connected" ||
    snapshot.connectionState === "reconnecting";
  const hasBackend = adapter !== null;
  const callAction = live ? "stop" : "start";
  const callLabel = sessionOperation
    ? live
      ? t("voice.stopping")
      : t("voice.starting")
    : live
      ? t("voice.stop")
      : t("voice.start");

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
    t(snapshot.muted ? "voice.unmute" : "voice.mute"),
  );
  elements.muteButton.title = t(
    snapshot.muted ? "voice.unmute" : "voice.mute",
  );

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
  updateSessionControls();
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
    t(sendingText ? "composer.sending" : "composer.send"),
  );
  elements.sendTextButton.title = t(
    sendingText ? "composer.sending" : "composer.send",
  );
  elements.textComposer.dataset.state = available ? "ready" : "unavailable";
  elements.textComposer.setAttribute("aria-busy", String(sendingText));
  if (elements.composerStatus.textContent !== placeholder) {
    elements.composerStatus.textContent = placeholder;
  }
}

function composerPlaceholder(snapshot: DesktopSessionSnapshot): string {
  if (sendingText) {
    return t("composer.sending");
  }
  if (discoveringBackend || backendState === "loading") {
    return t("orb.startingTitle");
  }
  if (backendState === "unconfigured") {
    return t("composer.chooseConfig");
  }
  if (backendState === "offline" || adapter === null) {
    return t("composer.serviceUnavailable");
  }
  if (sessionOperation) {
    return snapshot.connectionState === "disconnected"
      ? t("composer.connecting")
      : t("composer.updating");
  }
  if (snapshot.connectionState === "reconnecting") {
    return t("composer.reconnecting");
  }
  if (snapshot.connectionState === "disconnected") {
    return t("composer.connectFirstShort");
  }
  if (snapshot.sessionId === null) {
    return t("composer.initializing");
  }
  return t("composer.input");
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
    elements.orbTitle.textContent = t("orb.startingTitle");
    elements.orbCaption.textContent = t("orb.offlineAvailable");
    return;
  }

  if (backendState === "unconfigured") {
    elements.orbTitle.textContent = t("orb.needsConfig");
    elements.orbCaption.textContent = t("orb.chooseConfig");
    return;
  }

  if (backendState === "offline") {
    elements.orbTitle.textContent = t("orb.unavailable");
    elements.orbCaption.textContent = t("orb.reconfigure");
    return;
  }

  if (snapshot.connectionState === "disconnected") {
    elements.orbTitle.textContent = "";
    elements.orbCaption.textContent = "";
    return;
  }

  if (snapshot.connectionState === "reconnecting") {
    elements.orbTitle.textContent = t("orb.reconnecting");
    elements.orbCaption.textContent = t("orb.resumeAfterConnect");
    return;
  }

  if (snapshot.muted) {
    elements.orbTitle.textContent = t("orb.muted");
    elements.orbCaption.textContent = t("orb.unmuteHint");
    return;
  }

  switch (snapshot.streamState) {
    case "listening":
      elements.orbTitle.textContent = t("orb.listening");
      break;
    case "processing":
      elements.orbTitle.textContent = t("orb.processing");
      break;
    case "speaking":
      elements.orbTitle.textContent = t("orb.speaking");
      break;
    case "idle":
      elements.orbTitle.textContent = t("orb.connected");
      break;
  }
  elements.orbCaption.textContent = t("orb.openChatHint");
}

function renderSnapshot(snapshot: DesktopSessionSnapshot): void {
  const nextSessionActivityKey = `${snapshot.sessionId ?? ""}:${
    snapshot.messages.filter((message) => message.final).length
  }`;
  const shouldRefreshSessions =
    snapshot.sessionId !== null &&
    nextSessionActivityKey !== sessionActivityKey;
  sessionActivityKey = nextSessionActivityKey;
  latestSnapshot = snapshot;
  elements.connectionStateDetail.textContent = t(
    `runtime.${snapshot.connectionState}` as TranslationKey,
  );
  elements.streamStateDetail.textContent = t(
    `runtime.${snapshot.streamState}` as TranslationKey,
  );
  elements.mutedStateDetail.textContent = t(
    snapshot.muted ? "runtime.true" : "runtime.false",
  );
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
  renderChatSessions();
  if (shouldRefreshSessions) {
    scheduleChatSessionsRefresh();
  }
}

function messageRoleLabel(
  role: DesktopSessionSnapshot["messages"][number]["role"],
): string {
  switch (role) {
    case "user":
      return t("message.user");
    case "assistant":
      return "XTalk";
    case "info":
      return t("message.system");
  }
}

function renderModelConfigSelection(
  selection: NativeModelConfigSelection,
): void {
  modelConfigPath = selection.configPath;
  elements.modelConfigDetail.textContent =
    selection.configPath ?? t("model.none");
  elements.modelConfigDetail.title = selection.configPath ?? "";
  elements.modelConfigStatus.textContent =
    selection.configPath === null ? t("model.notSelected") : "";
  updateControls(latestSnapshot);
}

function renderInstalledTools(tools: NativeToolDefinition[]): void {
  installedTools = tools;
  if (tools.length === 0) {
    const empty = document.createElement("p");
    empty.className = "developer-tools-empty";
    empty.textContent = t("tools.none");
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
      t(tool.enabled ? "tools.disableName" : "tools.enableName", {
        name: tool.displayName,
      }),
    );
    toggle.addEventListener("change", () => {
      void updateInstalledToolEnabled(tool.id, toggle.checked);
    });

    const toggleText = document.createElement("span");
    toggleText.textContent = t("tools.enabled");
    toggleLabel.append(toggle, toggleText);

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "developer-tool-remove";
    remove.textContent = "×";
    remove.setAttribute(
      "aria-label",
      t("tools.removeName", { name: tool.displayName }),
    );
    remove.title = t("tools.removeTitle");
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
    elements.developerToolsStatus.textContent = t("tools.pending");
    return;
  }
  elements.developerToolsStatus.textContent =
    installedTools.length === 0
      ? t("tools.none")
      : t("tools.count", { count: installedTools.length });
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
    setBackendStatus("unconfigured", "service.chooseConfig");
    elements.backendDetail.textContent = t("service.notStarted");
    elements.websocketDetail.textContent = t("service.notConfigured");
    updateControls(latestSnapshot);
    return;
  }

  discoveringBackend = true;
  showError(null);
  setBackendStatus("loading", "service.searching");
  elements.backendDetail.textContent = t("service.waitingEndpoint");
  elements.websocketDetail.textContent = t("service.notConfigured");
  updateControls(latestSnapshot);

  await detachCurrentAdapter();

  try {
    const connection = await getNativeBackendConnection();
    const nextAdapter = new XtalkClientAdapter(connection);
    adapter = nextAdapter;
    unsubscribe = nextAdapter.subscribe(renderSnapshot);

    elements.backendDetail.textContent = nextAdapter.diagnostics.origin;
    elements.websocketDetail.textContent = nextAdapter.diagnostics.websocketURL;
    setBackendStatus("ready", "service.ready");
    await refreshChatSessions();
  } catch (error) {
    adapter = null;
    setBackendStatus("offline", "service.unavailable");
    elements.backendDetail.textContent = t("service.offlineMode");
    showError("service.connectFailed", { error });
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
  elements.modelConfigStatus.textContent = t("model.choosePrompt");
  updateControls(latestSnapshot);
  let stopManagedProgress: (() => void) | null = null;
  let managedProgressOpened = false;

  try {
    const selectedPath = await chooseNativeModelConfigFile();
    if (selectedPath === null) {
      if (required && modelConfigPath === null) {
        setBackendStatus("unconfigured", "service.chooseConfig");
        elements.modelConfigStatus.textContent =
          t("model.firstLaunch");
      } else {
        elements.modelConfigStatus.textContent = modelConfigPath
          ? t("model.cancelCurrent")
          : t("model.cancelNone");
      }
      return;
    }

    const managedPlan = await getNativeManagedModelPlan(selectedPath);
    if (managedPlan.services.length > 0) {
      managedProgressOpened = true;
      openManagedProgress(managedPlan.services);
      stopManagedProgress = await listenNativeManagedModelProgress(
        updateManagedProgress,
      );
    }

    elements.modelConfigStatus.textContent = t("model.restarting");
    setBackendStatus("loading", "service.applyingConfig");
    await detachCurrentAdapter();
    await applyNativeModelConfig(selectedPath);
    toolChangesPending = false;
    const selection = await refreshModelConfigSelection();
    updateDeveloperToolsStatus();
    elements.modelConfigStatus.textContent = selection.configPath
      ? t("model.appliedRestarted")
      : t("model.applied");
    await discoverBackend();
    if (managedProgressOpened) {
      closeManagedProgress();
    }
  } catch (error) {
    await refreshModelConfigSelection().catch(() => undefined);
    if (modelConfigPath === null) {
      setBackendStatus("unconfigured", "service.chooseConfig");
    } else {
      await discoverBackend();
    }
    elements.modelConfigStatus.textContent = t("model.applyFailed");
    showError("model.applyFailedDetail", { error });
    setDiagnosticsOpen(true);
    if (managedProgressOpened) {
      failManagedProgress(error);
    }
  } finally {
    stopManagedProgress?.();
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
  updateDeveloperToolsStatus(t("tools.choosePrompt"));
  updateControls(latestSnapshot);

  try {
    const selectedPath = await chooseNativeToolDirectory();
    if (selectedPath === null) {
      updateDeveloperToolsStatus(t("tools.cancelled"));
      return;
    }

    updateDeveloperToolsStatus(t("tools.copying"));
    const installed = await installNativeToolDirectory(selectedPath);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.installed", { name: installed.displayName }),
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus(t("tools.installFailed"));
    showError("tools.installFailedDetail", { error });
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
  updateDeveloperToolsStatus(t("tools.updating"));
  updateControls(latestSnapshot);

  try {
    const updated = await setNativeToolEnabled(toolId, enabled);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.updated", {
        name: updated.displayName,
        state: t(
          updated.enabled ? "tools.stateEnabled" : "tools.stateDisabled",
        ),
      }),
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus(t("tools.updateFailed"));
    showError("tools.updateFailedDetail", { error });
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
  updateDeveloperToolsStatus(t("tools.removing"));
  updateControls(latestSnapshot);

  try {
    await removeNativeInstalledTool(toolId);
    toolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.removed", {
        name: tool?.displayName ?? t("tools.generic"),
      }),
    );
  } catch (error) {
    await refreshInstalledTools().catch(() => undefined);
    updateDeveloperToolsStatus(t("tools.removeFailed"));
    showError("tools.removeFailedDetail", { error });
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
    updateDeveloperToolsStatus(t("composer.chooseConfig"));
    return;
  }

  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus(t("tools.restarting"));
  setBackendStatus("loading", "tools.applying");
  updateControls(latestSnapshot);

  try {
    await detachCurrentAdapter();
    await applyNativeToolChanges();
    toolChangesPending = false;
    updateDeveloperToolsStatus(t("tools.applied"));
    await discoverBackend();
  } catch (error) {
    updateDeveloperToolsStatus(t("tools.applyFailed"));
    await discoverBackend();
    showError("tools.applyFailedDetail", { error });
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
      setBackendStatus("unconfigured", "service.chooseConfig");
      setDiagnosticsOpen(true);
      await chooseAndApplyModelConfig(true);
      return;
    }
    await discoverBackend();
  } catch (error) {
    setBackendStatus("offline", "service.runtimeUnavailable");
    elements.modelConfigStatus.textContent = t("model.readFailed");
    showError("app.initializeFailed", { error });
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
    scheduleChatSessionsRefresh();
  } catch (error) {
    showError("voice.connectFailed", { error });
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
    showError("voice.closeFailed", { error });
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
    scheduleChatSessionsRefresh();
  } catch (error) {
    showError("composer.sendFailed", { error });
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

elements.toggleSidebarButton.addEventListener("click", () => {
  setSidebarOpen(!sidebarOpen);
});
elements.sidebarBackdrop.addEventListener("click", () => {
  setSidebarOpen(false);
});
elements.newChatButton.addEventListener("click", () => {
  void switchChatSession(null);
});
elements.openToolsButton.addEventListener("click", () => {
  setToolsDialogOpen(true);
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
elements.closeToolsButton.addEventListener("click", () => {
  setToolsDialogOpen(false);
});
elements.closeManagedProgressButton.addEventListener("click", () => {
  if (managedProgressState === "failed") {
    closeManagedProgress();
  }
});
elements.toolsDialogBackdrop.addEventListener("click", () => {
  setToolsDialogOpen(false);
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
elements.languageSelect.addEventListener("change", () => {
  const preference = elements.languageSelect.value as LanguagePreference;
  setLanguagePreference(preference);
  applyUiLanguage();
});
window.addEventListener("keydown", (event) => {
  if (managedProgressState === "running") {
    event.preventDefault();
    event.stopImmediatePropagation();
    return;
  }
  if (event.key === "Escape" && managedProgressState === "failed") {
    closeManagedProgress();
  } else if (event.key === "Escape" && toolsDialogOpen) {
    setToolsDialogOpen(false);
  } else if (event.key === "Escape" && diagnosticsOpen) {
    setDiagnosticsOpen(false);
  } else if (event.key === "Escape" && sidebarOpen && isCompactLayout()) {
    setSidebarOpen(false);
  }
});
window.addEventListener("online", updateNetworkStatus);
window.addEventListener("offline", updateNetworkStatus);
window.addEventListener("languagechange", () => {
  if (refreshAutomaticLanguage()) {
    applyUiLanguage();
  }
});

applyUiLanguage();
updateNetworkStatus();
setSidebarOpen(false, false);
renderSnapshot(EMPTY_SNAPSHOT);
resizeMessageInput();
void initializeApplication();
