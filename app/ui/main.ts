import "./styles.css";

import {
  applyNativeModelConfig,
  applyNativeToolChanges,
  chooseNativeModelConfigFile,
  chooseNativeToolDirectory,
  deleteNativeCredential,
  ensureNativeBackendStarted,
  getNativeCredentials,
  getNativeManagedModelPlan,
  getNativeBackendConnection,
  getNativeInstalledTools,
  getNativeModelConfigSelection,
  getNativeRecommendedModelConfig,
  getNativeToolUiSource,
  getNativeWakeWordSettings,
  installNativeToolDirectory,
  listenNativeAppBackgrounding,
  listenNativeManagedModelProgress,
  listenNativeWakeWordDetected,
  listenNativeWakeWordStatus,
  pauseNativeWakeWord,
  removeNativeInstalledTool,
  resumeNativeWakeWord,
  saveNativeCredential,
  setNativeToolEnabled,
  setNativeWakeWordEnabled,
  setNativeWakeWordPhrase,
  setNativeWakeWordThreshold,
  type NativeBackendConnection,
  type NativeManagedModelProgress,
  type NativeCredentialDefinition,
  type NativeModelConfigSelection,
  type NativeToolDefinition,
  type NativeWakeWordSettings,
} from "./adapters/native-capabilities";
import {
  XtalkClientAdapter,
  type DesktopMessage,
  type DesktopSessionSnapshot,
  type DesktopSessionSummary,
} from "./adapters/xtalk-client-adapter";
import {
  ToolUIAdapter,
  type ToolUIEmitEvent,
  type ToolUIEvent,
  type ToolUIStatusEvent,
} from "./adapters/tool-ui-adapter";
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
import {
  createToolUIFrameDocument,
  ToolUIFrame,
  type ToolUICapabilities,
} from "./tool-ui-frame";
import {
  listenWhiteboardWindowHidden,
  persistWhiteboardVisiblePreference,
  setWhiteboardWindowVisible,
} from "./whiteboard-window";

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

const WAKE_WORD_SUMMARY_KEYS: Record<
  NativeWakeWordSettings["state"],
  TranslationKey
> = {
  disabled: "wakeWord.stateDisabled",
  starting: "wakeWord.stateStarting",
  listening: "wakeWord.stateListening",
  paused: "wakeWord.statePaused",
  error: "wakeWord.stateError",
};
const WAKE_WORD_PHRASE_DEBOUNCE_MS = 500;
const WHITEBOARD_TOOL_ID = "builtin:whiteboard";

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
  firstLaunchDialog: requireElement<HTMLDialogElement>(
    "first-launch-dialog",
  ),
  recommendedConfigButton: requireElement<HTMLButtonElement>(
    "recommended-config-button",
  ),
  customConfigButton: requireElement<HTMLButtonElement>(
    "custom-config-button",
  ),
  firstLaunchCancelButton: requireElement<HTMLButtonElement>(
    "first-launch-cancel-button",
  ),
  llmKeyDialog: requireElement<HTMLDialogElement>("llm-key-dialog"),
  llmKeyForm: requireElement<HTMLFormElement>("llm-key-form"),
  llmKeyInput: requireElement<HTMLInputElement>("llm-key-input"),
  llmKeySkipButton: requireElement<HTMLButtonElement>(
    "llm-key-skip-button",
  ),
  credentialsList: requireElement<HTMLElement>("credentials-list"),
  credentialsStatus: requireElement<HTMLElement>("credentials-status"),
  credentialDialog: requireElement<HTMLDialogElement>("credential-dialog"),
  credentialForm: requireElement<HTMLFormElement>("credential-form"),
  credentialDialogLabel: requireElement<HTMLElement>("credential-dialog-label"),
  credentialDialogInput: requireElement<HTMLInputElement>(
    "credential-dialog-input",
  ),
  credentialCancelButton: requireElement<HTMLButtonElement>(
    "credential-cancel-button",
  ),
  deleteSessionDialog: requireElement<HTMLDialogElement>(
    "delete-session-dialog",
  ),
  deleteSessionForm: requireElement<HTMLFormElement>("delete-session-form"),
  deleteSessionDialogBody: requireElement<HTMLElement>(
    "delete-session-dialog-body",
  ),
  deleteSessionCancelButton: requireElement<HTMLButtonElement>(
    "delete-session-cancel-button",
  ),
  applyCredentialChangesButton: requireElement<HTMLButtonElement>(
    "apply-credential-changes-button",
  ),
  wakeWordEnabledToggle: requireElement<HTMLInputElement>(
    "wake-word-enabled-toggle",
  ),
  wakeWordPhraseInput: requireElement<HTMLInputElement>(
    "wake-word-phrase-input",
  ),
  wakeWordThresholdInput: requireElement<HTMLInputElement>(
    "wake-word-threshold-input",
  ),
  wakeWordSummary: requireElement<HTMLElement>("wake-word-summary"),
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
  liveToolPanel: requireElement<HTMLElement>("live-tool-panel"),
  liveToolStatusToggle: requireElement<HTMLButtonElement>(
    "live-tool-status-toggle",
  ),
  liveToolStatusTitle: requireElement<HTMLElement>(
    "live-tool-status-title",
  ),
  liveToolStatusSummary: requireElement<HTMLElement>(
    "live-tool-status-summary",
  ),
  liveToolContent: requireElement<HTMLElement>("live-tool-content"),
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
  whiteboardButton: requireElement<HTMLButtonElement>("whiteboard-button"),
  muteButton: requireElement<HTMLButtonElement>("mute-button"),
  retryButton: requireElement<HTMLButtonElement>("retry-button"),
  languageSelect: requireElement<HTMLSelectElement>("language-select"),
  languageSummary: requireElement<HTMLElement>("language-summary"),
};

// Diagnostic trace for the whiteboard window flow. The desktop shell reads
// this key to see exactly where a toggle stops when the window does not open.
const WHITEBOARD_TRACE_KEY = "xtalk.whiteboard.trace";

function traceWhiteboard(step: string, details?: unknown): void {
  try {
    localStorage.setItem(
      WHITEBOARD_TRACE_KEY,
      JSON.stringify({
        step,
        details: details === undefined ? null : String(details),
        at: Date.now(),
      }),
    );
  } catch {
    // The trace must never break the whiteboard flow itself.
  }
}

window.addEventListener("error", (event) => {
  traceWhiteboard(
    "window-error",
    event.error instanceof Error ? event.error.message : event.message,
  );
});

let adapter: XtalkClientAdapter | null = null;
let unsubscribe: (() => void) | null = null;
let toolUIAdapter: ToolUIAdapter | null = null;
let unsubscribeToolUI: (() => void) | null = null;
let discoveringBackend = false;
let sessionOperation = false;
let sendingText = false;
let modelConfigOperation = false;
let toolOperation = false;
let credentialOperation = false;
let developerToolChangesPending = false;
let credentialChangesPending = false;
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
let wakeWordSettings: NativeWakeWordSettings | null = null;
let wakeWordOperation = false;
let wakeWordPhraseComposing = false;
let wakeWordPhraseUpdateTimer: number | null = null;
let pendingWakeWordActivation = false;
let backgroundingRequested = false;
let recommendedConfigPath: string | null = null;
let credentials: NativeCredentialDefinition[] = [];
let selectedCredentialId: string | null = null;
let installedTools: NativeToolDefinition[] = [];
let whiteboardVisible = false;
let whiteboardAutoShown = false;
let activeToolUISessionId: string | null = null;
let toolUIOrder = 0;
let toolUIHistory: ToolUIHistoryItem[] = [];
let toolUILiveExpanded = false;
const toolUILive = new Map<string, ToolUILiveItem>();
const toolUIRows = new Map<string, ToolUIRow>();
const toolUISourceCache = new Map<string, Promise<string>>();
let toolUIFrameLanguage = getResolvedLanguage();
const toolUICapabilities = new Map<
  string,
  Partial<ToolUICapabilities>
>();
let persistedSessions: DesktopSessionSummary[] = [];
let sessionListError: string | null = null;
let backendConnection: NativeBackendConnection | null = null;
let pendingDeleteSessionId: string | null = null;
let latestSnapshot = EMPTY_SNAPSHOT;
let sessionRefreshTimer: ReturnType<typeof setTimeout> | null = null;
let sessionActivityKey = "";
const ACTIVE_SESSION_STORAGE_KEY = "xtalk.desktop.active-session.v1";
let backendStatusKey: TranslationKey = "service.starting";
let visibleError:
  | {
      key: TranslationKey;
      parameters: Readonly<Record<string, unknown>>;
    }
  | null = null;

interface ToolUIHistoryItem {
  kind: "history";
  id: string;
  anchorMessageIndex: number;
  order: number;
  event: ToolUIEmitEvent;
}

interface ToolUILiveItem {
  kind: "live";
  id: string;
  anchorMessageIndex: number;
  order: number;
  event: ToolUIStatusEvent;
}

type ToolUITimelineItem = ToolUIHistoryItem | ToolUILiveItem;

interface ToolUIRow {
  element: HTMLElement;
  frame: ToolUIFrame | null;
  mode: "live" | "history";
}

interface MessageRowState {
  row: HTMLElement;
  body: HTMLElement;
  contentHost: HTMLElement;
  spans: HTMLSpanElement[];
  actions: HTMLElement | null;
}

interface ChatSessionRowState {
  row: HTMLElement;
  button: HTMLButtonElement;
  title: HTMLSpanElement;
}

type MessageContentPart =
  | { kind: "text"; text: string }
  | { kind: "tool"; id: string; element: HTMLElement };

/**
 * Reused message row elements keyed by stable desktop message identity.
 *
 * Tool UI rows are embedded inside assistant messages, and WKWebView reloads
 * an iframe whenever its element is removed and reinserted into the document.
 * Keeping message rows stable means embedded tool iframes stay mounted across
 * snapshot renders instead of flashing on every text or status update.
 */
const messageRowStates = new Map<string, MessageRowState>();

/** Reused sidebar rows keyed by persisted conversation identity. */
const chatSessionRowStates = new Map<string, ChatSessionRowState>();

const TOOL_UI_HISTORY_PREFIX = "xtalk.tool-ui-history.v1:";
const MAX_TOOL_UI_HISTORY_ITEMS = 200;
const MESSAGE_COPY_CONFIRMATION_MS = 1_600;
const copiedMessageIds = new Set<string>();
const copiedMessageTimers = new Map<string, number>();
let messageInputCompositionActive = false;
let messageInputCompositionEnterHeld = false;

/** Return whether Enter is confirming an IME edit or is still the same held key. */
function isMessageInputCompositionEnter(event: KeyboardEvent): boolean {
  if (event.key !== "Enter") {
    return false;
  }
  if (
    event.isComposing ||
    messageInputCompositionActive ||
    event.keyCode === 229
  ) {
    messageInputCompositionEnterHeld = true;
    return true;
  }
  return messageInputCompositionEnterHeld;
}

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
  resetToolUIRowsIfLanguageChanged();
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
  if (wakeWordSettings !== null) {
    renderWakeWordSettings(wakeWordSettings);
  }
  renderCredentials(credentials);
  updateDeveloperToolsStatus();
  renderSnapshot(latestSnapshot);
  renderManagedProgress();
  updateWhiteboardButton();
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
    void refreshCredentials().catch(() => undefined);
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
    case "qwen3-asr-0.6b-int8":
      return "Qwen3-ASR 0.6B INT8";
    case "agentic-asr-refiner":
      return "AgenticASR Refiner";
    case "agentic-asr-refiner-mlx":
      return "AgenticASR Refiner (MLX)";
    case "moss-tts-nano":
      return "MOSS-TTS-Nano";
    case "moss-tts-nano-mlx":
      return "MOSS-TTS-Nano (MLX)";
    case "matcha-icefall-zh-en":
      return "Matcha Icefall (ZH/EN)";
    case "moss-transcribe-diarize":
      return "MOSS Transcribe Diarize";
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
    let state = chatSessionRowStates.get(session.id);
    if (state !== undefined) {
      state.row.classList.toggle("is-active", session.id === activeSessionId);
      state.button.classList.toggle("is-active", session.id === activeSessionId);
      state.button.setAttribute(
        "aria-current",
        session.id === activeSessionId ? "page" : "false",
      );
      state.title.textContent =
        session.title?.trim() || t("sidebar.newConversation");
      state.title.title = state.title.textContent;
      return state.row;
    }

    const row = document.createElement("div");
    row.className = "chat-session-row";
    row.dataset.sessionId = session.id;
    row.classList.toggle("is-active", session.id === activeSessionId);

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
    row.append(button);

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "chat-session-delete";
    deleteButton.setAttribute(
      "aria-label",
      t("sidebar.deleteSessionAria"),
    );
    deleteButton.title = t("sidebar.delete");
    const trash = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "svg",
    );
    trash.setAttribute("viewBox", "0 0 24 24");
    trash.setAttribute("aria-hidden", "true");
    trash.setAttribute("fill", "none");
    trash.setAttribute("stroke", "currentColor");
    trash.setAttribute("stroke-width", "2");
    trash.setAttribute("stroke-linecap", "round");
    trash.setAttribute("stroke-linejoin", "round");
    const trashPath = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    trashPath.setAttribute(
      "d",
      "M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6",
    );
    trash.append(trashPath);
    deleteButton.append(trash);
    deleteButton.addEventListener("click", () => {
      const currentSession = persistedSessions.find(
        (candidate) => candidate.id === session.id,
      );
      openDeleteSessionDialog(session.id, currentSession?.title ?? null);
    });
    row.append(deleteButton);
    state = { row, button, title };
    chatSessionRowStates.set(session.id, state);
    return row;
  });

  const activeSessionIds = new Set(persistedSessions.map((session) => session.id));
  for (const sessionId of chatSessionRowStates.keys()) {
    if (!activeSessionIds.has(sessionId)) {
      chatSessionRowStates.delete(sessionId);
    }
  }
  reconcileStableChildren(elements.chatSessionList, rows);
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

function openDeleteSessionDialog(
  sessionId: string,
  title: string | null,
): void {
  pendingDeleteSessionId = sessionId;
  const displayTitle = title?.trim() || t("sidebar.newConversation");
  elements.deleteSessionDialogBody.textContent = t(
    "sidebar.deleteConfirm",
    { title: displayTitle },
  );
  elements.deleteSessionDialog.showModal();
  elements.deleteSessionCancelButton.focus();
}

function cancelDeleteSessionDialog(): void {
  pendingDeleteSessionId = null;
  elements.deleteSessionDialog.close();
}

async function deletePendingSession(): Promise<void> {
  const sessionId = pendingDeleteSessionId;
  pendingDeleteSessionId = null;
  elements.deleteSessionDialog.close();
  if (sessionId === null || backendConnection === null) {
    return;
  }
  try {
    const response = await fetch(
      `${backendConnection.origin}/app/api/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: "DELETE",
        headers: {
          Accept: "application/json",
          "X-XTalk-App-Token": backendConnection.launchToken,
          Origin: "tauri://localhost",
        },
      },
    );
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    persistedSessions = persistedSessions.filter(
      (session) => session.id !== sessionId,
    );
    try {
      localStorage.removeItem(`${TOOL_UI_HISTORY_PREFIX}${sessionId}`);
    } catch {
      // Keep the in-memory cleanup even when storage is unavailable.
    }
    if (latestSnapshot.sessionId === sessionId) {
      await switchChatSession(null);
    } else {
      renderChatSessions();
    }
  } catch (error) {
    showError("sidebar.deleteFailed", { error });
    renderChatSessions();
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

  if (sessionId !== null && sessionId === latestSnapshot.sessionId) {
    setMainView("chat");
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
    await resumeWakeWordAfterConversation();
    persistActiveSessionId(sessionId);
    // Switching conversations collapses the whiteboard so the previous
    // conversation's board never follows into the newly opened chat. The
    // first whiteboard tool emit in the new conversation auto-opens it again.
    whiteboardAutoShown = false;
    await setWhiteboardVisible(false);
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
    if (backgroundingRequested) {
      void disconnectSession();
    }
  }
}

/**
 * Reads the app-owned active session independently of the sidecar's random port.
 *
 * @returns The last selected persisted session, or `null` for a new chat.
 */
function readActiveSessionId(): string | null {
  const sessionId = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
  return sessionId?.trim() || null;
}

/**
 * Persists the selected session across full application and sidecar restarts.
 *
 * @param sessionId Persisted session identifier, or `null` for a new chat.
 */
function persistActiveSessionId(sessionId: string | null): void {
  if (sessionId === null) {
    localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
    return;
  }
  localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, sessionId);
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

function whiteboardToolAvailable(): boolean {
  return installedTools.some(
    (tool) => tool.id === WHITEBOARD_TOOL_ID && tool.enabled,
  );
}

function updateWhiteboardButton(): void {
  elements.whiteboardButton.hidden = !whiteboardToolAvailable();
  elements.whiteboardButton.classList.toggle("is-active", whiteboardVisible);
  elements.whiteboardButton.setAttribute(
    "aria-pressed",
    String(whiteboardVisible),
  );
  const labelKey = whiteboardVisible ? "whiteboard.hide" : "whiteboard.show";
  elements.whiteboardButton.setAttribute("aria-label", t(labelKey));
  elements.whiteboardButton.title = t(labelKey);
}

async function setWhiteboardVisible(visible: boolean): Promise<void> {
  whiteboardVisible = visible;
  persistWhiteboardVisiblePreference(visible);
  traceWhiteboard("persisted", visible);
  try {
    updateWhiteboardButton();
    traceWhiteboard("button-updated");
    await setWhiteboardWindowVisible(visible);
    traceWhiteboard("invoke-ok", visible);
  } catch (error) {
    // The native whiteboard window is unavailable. Keep the button state
    // consistent and surface the failure so it can be diagnosed.
    traceWhiteboard(
      "invoke-error",
      error instanceof Error ? error.message : String(error),
    );
    console.error("Failed to toggle the whiteboard window.", error);
  }
}

function handleWhiteboardEmit(): void {
  if (whiteboardAutoShown && whiteboardVisible) {
    return;
  }
  whiteboardAutoShown = true;
  void setWhiteboardVisible(true);
}

function handleToolUIEvent(event: ToolUIEvent): void {
  const sessionId = event.sessionId ?? latestSnapshot.sessionId;
  if (sessionId === null || !toolHasUI(event.toolId)) {
    return;
  }
  if (event.toolId === WHITEBOARD_TOOL_ID) {
    if (event.type === "tool_ui.emit") {
      handleWhiteboardEmit();
    }
    return;
  }
  const anchorMessageIndex =
    sessionId === latestSnapshot.sessionId
      ? findToolUIAnchorMessageIndex(latestSnapshot.messages)
      : Number.MAX_SAFE_INTEGER;
  if (event.type === "tool_ui.emit") {
    if (!event.running && sessionId === activeToolUISessionId) {
      toolUILive.delete(event.callId);
      removeToolUIRow(`live:${event.callId}`);
    }
    const item: ToolUIHistoryItem = {
      kind: "history",
      id: `history:${event.callId}`,
      anchorMessageIndex,
      order: ++toolUIOrder,
      event: { ...event, sessionId },
    };
    const history = appendToolUIHistory(sessionId, item);
    if (sessionId === activeToolUISessionId) {
      toolUIHistory = history;
      toolUIRows.get(item.id)?.frame?.emit(item.event);
    }
  } else if (sessionId === activeToolUISessionId) {
    const id = `live:${event.callId}`;
    if (event.running) {
      const existing = toolUILive.get(event.callId);
      toolUILive.set(event.callId, {
        kind: "live",
        id,
        anchorMessageIndex:
          existing?.anchorMessageIndex ?? anchorMessageIndex,
        order: existing?.order ?? ++toolUIOrder,
        event: { ...event, sessionId },
      });
      const row = toolUIRows.get(id);
      row?.frame?.status(event);
    } else {
      toolUILive.delete(event.callId);
      removeToolUIRow(id);
    }
  }
  renderSnapshot(latestSnapshot);
}

/**
 * Finds the timeline boundary before the current turn's first AI reply.
 *
 * Tool UI observations arrive on an independent polling channel and may be
 * delivered after the assistant response they preceded. Anchoring to the
 * current message count would therefore place completed tool UI after that
 * response. The latest user message identifies the current turn, and the
 * first following assistant message is the stable insertion boundary.
 *
 * @param messages Conversation messages in display order.
 * @returns Message index at which history UI should be inserted.
 */
function findToolUIAnchorMessageIndex(messages: DesktopMessage[]): number {
  let latestUserIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === "user") {
      latestUserIndex = index;
      break;
    }
  }
  for (let index = latestUserIndex + 1; index < messages.length; index += 1) {
    if (messages[index]?.role === "assistant") {
      return index;
    }
  }
  return messages.length;
}

function switchToolUISession(sessionId: string | null): void {
  activeToolUISessionId = sessionId;
  messageRowStates.clear();
  toolUILiveExpanded = false;
  toolUILive.clear();
  for (const row of toolUIRows.values()) {
    row.frame?.destroy();
  }
  toolUIRows.clear();
  toolUIHistory = sessionId === null ? [] : readToolUIHistory(sessionId);
  toolUIOrder = toolUIHistory.reduce(
    (maximum, item) => Math.max(maximum, item.order),
    toolUIOrder,
  );
}

/** Recreates sandboxed tool documents after the resolved language changes. */
function resetToolUIRowsIfLanguageChanged(): void {
  const language = getResolvedLanguage();
  if (toolUIFrameLanguage === language) {
    return;
  }
  toolUIFrameLanguage = language;
  for (const row of toolUIRows.values()) {
    row.frame?.destroy();
    row.element.remove();
  }
  toolUIRows.clear();
}

function renderLiveToolPanel(): void {
  const items = [...toolUILive.values()]
    .filter(
      (item) =>
        toolUICapabilities.get(item.event.toolId)?.status !== false,
    )
    .sort((left, right) => left.order - right.order);
  if (items.length === 0) {
    toolUILiveExpanded = false;
    elements.liveToolPanel.hidden = true;
    elements.liveToolStatusToggle.setAttribute("aria-expanded", "false");
    elements.liveToolContent.hidden = true;
    elements.liveToolContent.replaceChildren();
    return;
  }

  const latest = items[items.length - 1]!;
  const tool = installedTools.find(
    (candidate) => candidate.id === latest.event.toolId,
  );
  const toolName =
    tool === undefined
      ? latest.event.toolName
      : resolveToolDisplayName(tool.displayName);
  elements.liveToolPanel.hidden = false;
  elements.liveToolStatusTitle.textContent = t("tools.liveCount", {
    count: items.length,
  });
  elements.liveToolStatusSummary.textContent =
    `${toolName} · ${latest.event.status}`;
  elements.liveToolStatusToggle.setAttribute(
    "aria-expanded",
    String(toolUILiveExpanded),
  );
  elements.liveToolStatusToggle.setAttribute(
    "aria-label",
    t(
      toolUILiveExpanded
        ? "tools.liveCollapse"
        : "tools.liveExpand",
    ),
  );
  elements.liveToolContent.hidden = !toolUILiveExpanded;
  reconcileStableChildren(
    elements.liveToolContent,
    items.map((item) => getOrCreateToolUIRow(item).element),
  );
}

/**
 * Reconciles children without detaching unchanged iframes.
 *
 * WKWebView resets an iframe browsing context when `replaceChildren` removes
 * and reinserts the same node, leaving an already-mounted live UI at
 * `about:blank`. Incremental reconciliation preserves existing frame contexts
 * and only inserts, moves, or removes changed rows.
 *
 * @param container Element whose children should match the requested order.
 * @param desiredChildren Child elements in display order.
 */
function reconcileStableChildren(
  container: HTMLElement,
  desiredChildren: HTMLElement[],
): void {
  const desired = new Set(desiredChildren);
  for (const child of [...container.children]) {
    if (!desired.has(child as HTMLElement)) {
      child.remove();
    }
  }
  for (const [index, child] of desiredChildren.entries()) {
    const current = container.children.item(index);
    if (current !== child) {
      container.insertBefore(child, current);
    }
  }
}

function getOrCreateToolUIRow(item: ToolUITimelineItem): ToolUIRow {
  const existing = toolUIRows.get(item.id);
  if (existing !== undefined) {
    if (item.kind === "live") {
      existing.frame?.status(item.event);
    }
    return existing;
  }

  const row = document.createElement("article");
  row.className = `tool-ui-row tool-ui-row-${item.kind}`;
  row.dataset.toolUiId = item.id;

  const fallback = document.createElement("div");
  fallback.className = "tool-ui-fallback";
  const fallbackTitle = document.createElement("strong");
  const tool = installedTools.find(
    (candidate) => candidate.id === item.event.toolId,
  );
  fallbackTitle.textContent =
    tool === undefined
      ? item.event.toolName
      : resolveToolDisplayName(tool.displayName);
  const fallbackBody = document.createElement("span");
  fallbackBody.textContent =
    item.kind === "history" ? item.event.message : item.event.status;
  fallback.append(fallbackTitle, fallbackBody);
  row.append(fallback);

  const record: ToolUIRow = {
    element: row,
    frame: null,
    mode: item.kind === "live" ? "live" : "history",
  };
  toolUIRows.set(item.id, record);
  void hydrateToolUIRow(record, item, fallback);
  return record;
}

async function hydrateToolUIRow(
  row: ToolUIRow,
  item: ToolUITimelineItem,
  fallback: HTMLElement,
): Promise<void> {
  const capability = toolUICapabilities.get(item.event.toolId);
  const requiredCapability = item.kind === "live" ? "status" : "emit";
  if (capability?.[requiredCapability] === false) {
    row.element.hidden = true;
    return;
  }
  try {
    const source = await loadToolUISource(item.event.toolId);
    if (toolUIRows.get(item.id) !== row) {
      return;
    }
    const tool = installedTools.find(
      (candidate) => candidate.id === item.event.toolId,
    );
    const title =
      tool === undefined
        ? item.event.toolName
        : resolveToolDisplayName(tool.displayName);
    const frameAdapter = toolUIAdapter;
    if (frameAdapter === null) {
      throw new Error("Tool UI adapter is unavailable.");
    }
    const channelId = crypto.randomUUID();
    const frameUrl = await frameAdapter.createFrame(
      createToolUIFrameDocument(
        source,
        channelId,
        row.mode,
        getResolvedLanguage(),
      ),
    );
    if (
      toolUIRows.get(item.id) !== row ||
      toolUIAdapter !== frameAdapter
    ) {
      return;
    }
    const frame = new ToolUIFrame(
      frameUrl,
      channelId,
      row.mode,
      t("tools.uiFrameTitle", { name: title }),
      (capabilities) => {
        toolUICapabilities.set(item.event.toolId, {
          ...toolUICapabilities.get(item.event.toolId),
          [requiredCapability]: capabilities[requiredCapability],
        });
        if (!capabilities[requiredCapability]) {
          row.element.hidden = true;
          if (item.kind === "live") {
            renderLiveToolPanel();
          }
          return;
        }
        row.element.hidden = false;
        if (item.kind === "live") {
          renderLiveToolPanel();
        }
      },
      undefined,
    );
    row.frame = frame;
    fallback.replaceWith(frame.element);
    frame.mount();
    if (item.kind === "live") {
      frame.status(item.event);
    } else {
      frame.emit(item.event);
    }
  } catch {
    fallback.classList.add("is-error");
    const error = document.createElement("span");
    error.textContent = t("tools.uiUnavailable");
    fallback.append(error);
  }
}

function loadToolUISource(toolId: string): Promise<string> {
  const cached = toolUISourceCache.get(toolId);
  if (cached !== undefined) {
    return cached;
  }
  const source = getNativeToolUiSource(toolId).then((payload) => payload.source);
  toolUISourceCache.set(toolId, source);
  return source;
}

function removeToolUIRow(id: string): void {
  const row = toolUIRows.get(id);
  row?.frame?.destroy();
  row?.element.remove();
  toolUIRows.delete(id);
}

function toolHasUI(toolId: string): boolean {
  return installedTools.some(
    (tool) => tool.id === toolId && tool.ui !== null,
  );
}

function resolveToolDisplayName(
  displayName: NativeToolDefinition["displayName"],
): string {
  if (typeof displayName === "string") {
    return displayName;
  }
  const language = getResolvedLanguage().toLowerCase();
  const primaryLanguage = language.split("-", 1)[0] ?? language;
  return (
    displayName[language] ??
    displayName[primaryLanguage] ??
    displayName.en ??
    displayName.zh ??
    Object.entries(displayName).sort(([left], [right]) =>
      left.localeCompare(right),
    )[0]?.[1] ??
    t("tools.generic")
  );
}

function appendToolUIHistory(
  sessionId: string,
  item: ToolUIHistoryItem,
): ToolUIHistoryItem[] {
  const history = readToolUIHistory(sessionId);
  const existingIndex = history.findIndex(
    (candidate) => candidate.id === item.id,
  );
  if (existingIndex !== -1) {
    const existing = history[existingIndex]!;
    item.anchorMessageIndex = Math.min(
      item.anchorMessageIndex,
      existing.anchorMessageIndex,
    );
    history[existingIndex] = item;
    try {
      localStorage.setItem(
        `${TOOL_UI_HISTORY_PREFIX}${sessionId}`,
        JSON.stringify(history),
      );
    } catch {
      // Keep the in-memory update even when persistence is unavailable.
    }
    return history;
  }
  const callHistory = history.filter(
    (candidate) => candidate.event.callId === item.event.callId,
  );
  if (callHistory.length > 0) {
    item.anchorMessageIndex = Math.min(
      item.anchorMessageIndex,
      ...callHistory.map((candidate) => candidate.anchorMessageIndex),
    );
  }
  const converged = item.event.running
    ? history
    : history.filter(
        (candidate) => candidate.event.callId !== item.event.callId,
      );
  converged.push(item);
  let bounded = converged
    .sort((left, right) => left.order - right.order)
    .slice(-MAX_TOOL_UI_HISTORY_ITEMS);
  const key = `${TOOL_UI_HISTORY_PREFIX}${sessionId}`;
  while (bounded.length > 0) {
    try {
      localStorage.setItem(key, JSON.stringify(bounded));
      return bounded;
    } catch {
      bounded = bounded.slice(1);
    }
  }
  return [item];
}

function readToolUIHistory(sessionId: string): ToolUIHistoryItem[] {
  const serialized = localStorage.getItem(
    `${TOOL_UI_HISTORY_PREFIX}${sessionId}`,
  );
  if (serialized === null) {
    return [];
  }
  try {
    const payload: unknown = JSON.parse(serialized);
    if (!Array.isArray(payload)) {
      return [];
    }
    return payload
      .filter(isStoredToolUIHistoryItem)
      .slice(-MAX_TOOL_UI_HISTORY_ITEMS);
  } catch {
    return [];
  }
}

function isStoredToolUIHistoryItem(
  value: unknown,
): value is ToolUIHistoryItem {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const candidate = value as Partial<ToolUIHistoryItem>;
  const event = candidate.event as Partial<ToolUIEmitEvent> | undefined;
  return (
    candidate.kind === "history" &&
    typeof candidate.id === "string" &&
    typeof candidate.anchorMessageIndex === "number" &&
    Number.isInteger(candidate.anchorMessageIndex) &&
    candidate.anchorMessageIndex >= 0 &&
    typeof candidate.order === "number" &&
    Number.isInteger(candidate.order) &&
    event?.type === "tool_ui.emit" &&
    typeof event.toolId === "string" &&
    typeof event.toolName === "string" &&
    typeof event.callId === "string" &&
    typeof event.sequence === "number" &&
    typeof event.message === "string" &&
    typeof event.status === "string" &&
    typeof event.running === "boolean" &&
    (event.outcome === undefined ||
      event.outcome === "running" ||
      event.outcome === "complete" ||
      event.outcome === "cancelled") &&
    typeof event.emittedAt === "string"
  );
}

function renderSnapshot(snapshot: DesktopSessionSnapshot): void {
  const previousConnectionState = latestSnapshot.connectionState;
  const nextSessionActivityKey = `${snapshot.sessionId ?? ""}:${
    snapshot.messages.filter((message) => message.final).length
  }`;
  const shouldRefreshSessions =
    snapshot.sessionId !== null &&
    nextSessionActivityKey !== sessionActivityKey;
  sessionActivityKey = nextSessionActivityKey;
  latestSnapshot = snapshot;
  if (
    snapshot.connectionState === "disconnected" &&
    (previousConnectionState === "connected" ||
      previousConnectionState === "reconnecting")
  ) {
    void resumeWakeWordAfterConversation();
  }
  if (snapshot.sessionId !== null) {
    persistActiveSessionId(snapshot.sessionId);
  }
  toolUIAdapter?.bindSession(snapshot.sessionId);
  if (activeToolUISessionId !== snapshot.sessionId) {
    switchToolUISession(snapshot.sessionId);
  }
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

  const timelineItems = [...toolUIHistory].sort(
    (left, right) =>
      left.anchorMessageIndex - right.anchorMessageIndex ||
      left.order - right.order,
  );

  const embeddedToolItemIds = new Set<string>();
  const messageElements = snapshot.messages.map((message, messageIndex) => {
    let state = messageRowStates.get(message.id);
    if (state === undefined) {
      state = createMessageRowState(message);
      messageRowStates.set(message.id, state);
    }
    updateMessageRowState(
      state,
      message,
      timelineItems,
      messageIndex,
      embeddedToolItemIds,
    );
    return state.row;
  });

  const timelineElements: HTMLElement[] = [];
  for (let index = 0; index <= messageElements.length; index += 1) {
    if (index > 0) {
      timelineElements.push(messageElements[index - 1]!);
    }
    for (const item of timelineItems) {
      if (
        !embeddedToolItemIds.has(item.id) &&
        Math.min(item.anchorMessageIndex, messageElements.length) === index
      ) {
        timelineElements.push(getOrCreateToolUIRow(item).element);
      }
    }
  }

  reconcileStableChildren(elements.messages, timelineElements);
  if (
    timelineElements.length > 0 &&
    isScrolledNearBottom(elements.messages)
  ) {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  }
  renderLiveToolPanel();

  updateOrbPresentation(snapshot);
  updateControls(snapshot);
  renderChatSessions();
  if (shouldRefreshSessions) {
    scheduleChatSessionsRefresh();
  }
}

/**
 * Returns whether a scroll container is at or near its bottom edge.
 *
 * Renders happen frequently while tools stream status updates. Auto-scrolling
 * only when the user is already near the bottom prevents the chat from yanking
 * the viewport down while they scroll up through earlier messages.
 *
 * @param container Scrollable messages container.
 * @returns Whether the container is within the auto-scroll threshold.
 */
function isScrolledNearBottom(container: HTMLElement): boolean {
  return (
    container.scrollHeight - container.scrollTop - container.clientHeight < 64
  );
}

/**
 * Creates one plain-text content span for an assistant message part.
 *
 * @param text Text rendered inside the span.
 * @returns Span element carrying the message-content class.
 */
function createMessageContentSpan(text: string): HTMLSpanElement {
  const content = document.createElement("span");
  content.className = "message-content";
  content.textContent = text;
  return content;
}

/**
 * Clamps a tool-call character offset into the current message bounds.
 *
 * @param offset Raw offset delivered by the backend.
 * @param length Current message content length.
 * @returns Integer offset between zero and the message length.
 */
function clampMessageOffset(offset: number, length: number): number {
  if (!Number.isFinite(offset)) {
    return 0;
  }
  return Math.max(0, Math.min(length, Math.trunc(offset)));
}

/**
 * Returns whether one character ends a spoken sentence.
 *
 * @param character Candidate sentence-terminating character.
 * @returns Whether the character is a sentence boundary.
 */
function isSentenceBoundary(character: string): boolean {
  return "。！？；!?;\n".includes(character);
}

/**
 * Advances a tool-call offset to the end of the sentence containing it.
 *
 * Assistant text streams while a tool call is already emitted, so the exact
 * recorded offset often lands mid-sentence. Rendering the tool card after the
 * sentence keeps the chat readable: the acknowledgment is shown first, then
 * the tool card, then the tool result.
 *
 * @param offset Recorded character offset inside the message.
 * @param text Complete assistant message content.
 * @returns Offset moved past the sentence boundary, when one exists.
 */
function advanceToSentenceBoundary(offset: number, text: string): number {
  if (offset >= text.length) {
    return text.length;
  }
  if (offset > 0 && isSentenceBoundary(text[offset - 1]!)) {
    return offset;
  }
  let index = offset;
  while (index < text.length) {
    if (isSentenceBoundary(text[index]!)) {
      return index + 1;
    }
    index += 1;
  }
  return text.length;
}

function createMessageRowState(message: DesktopMessage): MessageRowState {
  const row = document.createElement("article");
  row.className = `message-row message-row-${message.role}`;
  row.setAttribute("aria-label", messageRoleLabel(message.role));

  const body = document.createElement("div");
  body.className = `message message-${message.role}`;

  const contentHost = document.createElement("div");
  contentHost.className = "message-content-group";

  body.append(contentHost);
  row.append(body);
  return { row, body, contentHost, spans: [], actions: null };
}

function updateMessageRowState(
  state: MessageRowState,
  message: DesktopMessage,
  timelineItems: ToolUITimelineItem[],
  messageIndex: number,
  embeddedToolItemIds: Set<string>,
): void {
  state.body.dataset.final = String(message.final);

  const embeddable = message.role === "assistant"
    ? timelineItems.filter(
        (item): item is ToolUIHistoryItem =>
          item.kind === "history" &&
          item.anchorMessageIndex === messageIndex &&
          typeof item.event.textOffset === "number",
      )
    : [];
  const desiredParts: MessageContentPart[] = [];
  if (embeddable.length > 0) {
    const ordered = [...embeddable].sort(
      (left, right) =>
        (left.event.textOffset ?? 0) - (right.event.textOffset ?? 0) ||
        left.order - right.order,
    );
    let cursor = 0;
    for (const item of ordered) {
      const offset = advanceToSentenceBoundary(
        clampMessageOffset(
          item.event.textOffset ?? 0,
          message.content.length,
        ),
        message.content,
      );
      if (offset > cursor) {
        desiredParts.push({
          kind: "text",
          text: message.content.slice(cursor, offset),
        });
        cursor = offset;
      }
      desiredParts.push({
        kind: "tool",
        id: item.id,
        element: getOrCreateToolUIRow(item).element,
      });
      embeddedToolItemIds.add(item.id);
    }
    if (cursor < message.content.length) {
      desiredParts.push({ kind: "text", text: message.content.slice(cursor) });
    }
  } else {
    desiredParts.push({ kind: "text", text: message.content });
  }
  reconcileMessageContent(state, desiredParts);

  if (message.role === "assistant" && message.content.length > 0) {
    if (state.actions === null) {
      state.actions = document.createElement("div");
      state.actions.className = "message-actions";
      state.row.append(state.actions);
    }
    state.actions.replaceChildren(
      createMessageCopyButton(message.id, message.content),
    );
  } else if (state.actions !== null) {
    state.actions.remove();
    state.actions = null;
  }
}

function reconcileMessageContent(
  state: MessageRowState,
  desiredParts: MessageContentPart[],
): void {
  const host = state.contentHost;
  const remainingSpans = state.spans.slice();
  const desiredElements: HTMLElement[] = [];
  for (const part of desiredParts) {
    if (part.kind === "text") {
      const span = remainingSpans.shift() ?? createMessageContentSpan("");
      span.textContent = part.text;
      desiredElements.push(span);
    } else {
      desiredElements.push(part.element);
    }
  }
  const usedSpans = desiredElements.filter(
    (element): element is HTMLSpanElement =>
      element.classList.contains("message-content"),
  );
  for (const span of remainingSpans) {
    span.remove();
  }
  state.spans = usedSpans;

  const desired = new Set(desiredElements);
  for (const child of [...host.children]) {
    if (!desired.has(child as HTMLElement)) {
      child.remove();
    }
  }
  for (const [index, element] of desiredElements.entries()) {
    const current = host.children.item(index);
    if (current !== element) {
      host.insertBefore(element, current);
    }
  }
}

function createMessageActionIcon(kind: "copy" | "check"): SVGSVGElement {
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.setAttribute("viewBox", "0 0 24 24");
  icon.setAttribute("aria-hidden", "true");
  icon.classList.add(`message-action-icon-${kind}`);

  if (kind === "check") {
    const path = document.createElementNS(
      "http://www.w3.org/2000/svg",
      "path",
    );
    path.setAttribute("d", "m5 12.5 4.1 4.1L19 6.7");
    icon.append(path);
    return icon;
  }

  const rear = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "rect",
  );
  rear.setAttribute("x", "8");
  rear.setAttribute("y", "3.5");
  rear.setAttribute("width", "12.5");
  rear.setAttribute("height", "12.5");
  rear.setAttribute("rx", "3");
  const front = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "rect",
  );
  front.classList.add("message-copy-icon-front");
  front.setAttribute("x", "3.5");
  front.setAttribute("y", "8");
  front.setAttribute("width", "12.5");
  front.setAttribute("height", "12.5");
  front.setAttribute("rx", "3");
  icon.append(rear, front);
  return icon;
}

function updateMessageCopyButton(
  button: HTMLButtonElement,
  copied: boolean,
): void {
  const label = t(copied ? "message.copied" : "message.copy");
  button.dataset.copied = String(copied);
  button.setAttribute("aria-label", label);
  button.title = label;
  button.replaceChildren(createMessageActionIcon(copied ? "check" : "copy"));
}

function createMessageCopyButton(
  messageId: string,
  content: string,
): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-copy-button";
  button.dataset.messageId = messageId;
  updateMessageCopyButton(button, copiedMessageIds.has(messageId));
  button.addEventListener("click", () => {
    void copyAssistantMessage(button, messageId, content);
  });
  return button;
}

async function copyAssistantMessage(
  button: HTMLButtonElement,
  messageId: string,
  content: string,
): Promise<void> {
  if (button.dataset.copying === "true") {
    return;
  }
  button.dataset.copying = "true";
  try {
    await writeClipboardText(content);
    copiedMessageIds.add(messageId);
    updateMessageCopyButton(button, true);

    const previousTimer = copiedMessageTimers.get(messageId);
    if (previousTimer !== undefined) {
      window.clearTimeout(previousTimer);
    }
    const timer = window.setTimeout(() => {
      copiedMessageTimers.delete(messageId);
      copiedMessageIds.delete(messageId);
      for (const currentButton of elements.messages.querySelectorAll<HTMLButtonElement>(
        ".message-copy-button",
      )) {
        if (currentButton.dataset.messageId === messageId) {
          updateMessageCopyButton(currentButton, false);
        }
      }
    }, MESSAGE_COPY_CONFIRMATION_MS);
    copiedMessageTimers.set(messageId, timer);
  } catch (error) {
    showError("message.copyFailed", { error });
  } finally {
    delete button.dataset.copying;
  }
}

async function writeClipboardText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    const copied = await navigator.clipboard.writeText(text).then(
      () => true,
      () => false,
    );
    if (copied) {
      return;
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.append(textarea);
  textarea.select();
  const copied = document.execCommand("copy");
  textarea.remove();
  if (!copied) {
    throw new Error("Clipboard API is unavailable");
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
  updateWhiteboardButton();
  if (tools.length === 0) {
    const empty = document.createElement("p");
    empty.className = "developer-tools-empty";
    empty.textContent = t("tools.none");
    elements.developerToolsList.replaceChildren(empty);
    updateToolControls();
    return;
  }

  const createRow = (tool: NativeToolDefinition): HTMLElement => {
    const row = document.createElement("article");
    row.className = "developer-tool-row";
    row.dataset.origin = tool.origin;

    const copy = document.createElement("div");
    copy.className = "developer-tool-copy";

    const name = document.createElement("strong");
    const nameText = document.createElement("span");
    nameText.textContent = resolveToolDisplayName(tool.displayName);
    name.append(nameText);
    if (tool.origin === "builtin") {
      const badge = document.createElement("small");
      badge.className = "developer-tool-origin";
      badge.textContent = t("tools.builtinBadge");
      name.append(badge);
    }

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
    toggle.disabled = !tool.canDisable;
    toggle.dataset.required = String(!tool.canDisable);
    toggle.setAttribute(
      "aria-label",
      t(
        !tool.canDisable
          ? "tools.requiredName"
          : tool.enabled
            ? "tools.disableName"
            : "tools.enableName",
        {
          name: resolveToolDisplayName(tool.displayName),
        },
      ),
    );
    if (tool.canDisable) {
      toggle.addEventListener("change", () => {
        void updateInstalledToolEnabled(tool.id, toggle.checked);
      });
    }

    const toggleText = document.createElement("span");
    toggleText.textContent = t(
      tool.canDisable ? "tools.enabled" : "tools.required",
    );
    toggleLabel.append(toggle, toggleText);

    actions.append(toggleLabel);
    if (tool.canDelete) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "developer-tool-remove";
      remove.textContent = "×";
      remove.setAttribute(
        "aria-label",
        t("tools.removeName", {
          name: resolveToolDisplayName(tool.displayName),
        }),
      );
      remove.title = t("tools.removeTitle");
      remove.addEventListener("click", () => {
        void removeInstalledTool(tool.id);
      });
      actions.append(remove);
    }

    row.append(copy, actions);
    return row;
  };

  const sections = (["builtin", "user"] as const)
    .map((origin) => {
      const matching = tools.filter((tool) => tool.origin === origin);
      if (matching.length === 0) {
        return null;
      }
      const section = document.createElement("section");
      section.className = "developer-tool-section";
      const heading = document.createElement("h3");
      heading.textContent = t(
        origin === "builtin" ? "tools.builtinGroup" : "tools.userGroup",
      );
      section.append(heading, ...matching.map(createRow));
      return section;
    })
    .filter((section): section is HTMLElement => section !== null);

  elements.developerToolsList.replaceChildren(...sections);
  updateToolControls();
}

function renderWakeWordSettings(settings: NativeWakeWordSettings): void {
  wakeWordSettings = settings;
  elements.wakeWordEnabledToggle.checked = settings.enabled;
  elements.wakeWordPhraseInput.value = settings.phrase;
  elements.wakeWordThresholdInput.value = String(settings.threshold);
  elements.wakeWordSummary.textContent = t(
    WAKE_WORD_SUMMARY_KEYS[settings.state],
  );
  elements.wakeWordEnabledToggle.disabled = wakeWordOperation;
  elements.wakeWordPhraseInput.disabled = wakeWordOperation;
  elements.wakeWordThresholdInput.disabled = wakeWordOperation;
}

function renderCredentials(items: NativeCredentialDefinition[]): void {
  credentials = items;
  const rows = items.map((credential) => {
    const row = document.createElement("article");
    row.className = "credential-row";

    const copy = document.createElement("div");
    copy.className = "credential-copy";
    const name = document.createElement("strong");
    name.textContent = resolveToolDisplayName(credential.displayName);
    const status = document.createElement("span");
    status.textContent = t(
      credential.source === "environment"
        ? "credentials.environment"
        : credential.source === "system"
          ? "credentials.system"
          : credential.storageAvailable
            ? "credentials.missing"
            : "credentials.unavailable",
    );
    copy.append(name, status);

    const actions = document.createElement("div");
    actions.className = "credential-actions";
    if (credential.source !== "environment" && credential.storageAvailable) {
      const configure = document.createElement("button");
      configure.type = "button";
      configure.className = "credential-action";
      configure.textContent = t(
        credential.source === "system"
          ? "credentials.replace"
          : "credentials.configure",
      );
      configure.addEventListener("click", () => {
        openCredentialDialog(credential.id);
      });
      actions.append(configure);
    }
    if (credential.source === "system" && credential.storageAvailable) {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.className = "credential-action credential-action-danger";
      remove.textContent = t("credentials.delete");
      remove.addEventListener("click", () => {
        void removeCredential(credential.id);
      });
      actions.append(remove);
    }

    row.append(copy, actions);
    return row;
  });
  if (rows.length === 0) {
    const empty = document.createElement("p");
    empty.className = "credential-empty";
    empty.textContent = t("credentials.none");
    elements.credentialsList.replaceChildren(empty);
  } else {
    elements.credentialsList.replaceChildren(...rows);
  }
  updateCredentialStatus();
  updateToolControls();
}

function updateToolControls(): void {
  const busy =
    toolOperation ||
    modelConfigOperation ||
    discoveringBackend ||
    sessionOperation ||
    sendingText ||
    credentialOperation;
  const hasPendingChanges =
    developerToolChangesPending || credentialChangesPending;
  elements.installToolDirectoryButton.disabled = busy;
  elements.applyToolChangesButton.disabled =
    busy || !hasPendingChanges || modelConfigPath === null;
  elements.applyCredentialChangesButton.disabled =
    busy || !hasPendingChanges || modelConfigPath === null;
  elements.wakeWordEnabledToggle.disabled =
    wakeWordOperation || wakeWordSettings === null;
  elements.wakeWordPhraseInput.disabled =
    wakeWordOperation || wakeWordSettings === null;
  elements.wakeWordThresholdInput.disabled =
    wakeWordOperation || wakeWordSettings === null;
  for (const control of elements.developerToolsList.querySelectorAll<
    HTMLInputElement | HTMLButtonElement
  >("input, button")) {
    control.disabled =
      busy ||
      (control instanceof HTMLInputElement && control.dataset.required === "true");
  }
  for (const control of elements.credentialsList.querySelectorAll<
    HTMLButtonElement
  >("button")) {
    control.disabled = busy;
  }
}

function updateCredentialStatus(message?: string): void {
  if (message) {
    elements.credentialsStatus.textContent = message;
    return;
  }
  if (credentialChangesPending) {
    elements.credentialsStatus.textContent = t("credentials.pending");
    return;
  }
  elements.credentialsStatus.textContent = credentials.length
    ? t("credentials.count", { count: credentials.length })
    : t("credentials.none");
}

function updateDeveloperToolsStatus(message?: string): void {
  if (message) {
    elements.developerToolsStatus.textContent = message;
    return;
  }
  if (developerToolChangesPending) {
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

async function refreshCredentials(): Promise<NativeCredentialDefinition[]> {
  const items = await getNativeCredentials();
  renderCredentials(items);
  return items;
}

async function refreshWakeWordSettings(): Promise<NativeWakeWordSettings> {
  const settings = await getNativeWakeWordSettings();
  renderWakeWordSettings(settings);
  return settings;
}

async function updateWakeWordEnabled(enabled: boolean): Promise<void> {
  if (wakeWordOperation) {
    return;
  }
  if (!enabled) {
    pendingWakeWordActivation = false;
  }
  wakeWordOperation = true;
  elements.wakeWordEnabledToggle.disabled = true;
  try {
    const conversationActive =
      latestSnapshot.connectionState === "connected" ||
      latestSnapshot.connectionState === "reconnecting";
    renderWakeWordSettings(
      await setNativeWakeWordEnabled(enabled, !conversationActive),
    );
  } catch (error) {
    await refreshWakeWordSettings().catch(() => undefined);
    showError("wakeWord.updateFailed", { error });
  } finally {
    wakeWordOperation = false;
    if (wakeWordSettings !== null) {
      renderWakeWordSettings(wakeWordSettings);
    }
  }
}

async function updateWakeWordPhrase(phrase: string): Promise<void> {
  if (wakeWordOperation || wakeWordSettings === null) {
    return;
  }
  if (phrase.trim() === wakeWordSettings.phrase) {
    elements.wakeWordPhraseInput.value = wakeWordSettings.phrase;
    return;
  }
  wakeWordOperation = true;
  elements.wakeWordEnabledToggle.disabled = true;
  elements.wakeWordPhraseInput.disabled = true;
  try {
    const conversationActive =
      latestSnapshot.connectionState === "connected" ||
      latestSnapshot.connectionState === "reconnecting";
    renderWakeWordSettings(
      await setNativeWakeWordPhrase(phrase, !conversationActive),
    );
  } catch (error) {
    await refreshWakeWordSettings().catch(() => undefined);
    showError("wakeWord.updateFailed", { error });
  } finally {
    wakeWordOperation = false;
    if (wakeWordSettings !== null) {
      renderWakeWordSettings(wakeWordSettings);
    }
  }
}

async function updateWakeWordThreshold(rawThreshold: string): Promise<void> {
  if (wakeWordOperation || wakeWordSettings === null) {
    return;
  }
  if (!rawThreshold.trim()) {
    elements.wakeWordThresholdInput.value = String(wakeWordSettings.threshold);
    return;
  }
  const threshold = Number(rawThreshold);
  if (
    !Number.isFinite(threshold) ||
    threshold < 0 ||
    threshold > 1
  ) {
    elements.wakeWordThresholdInput.value = String(wakeWordSettings.threshold);
    return;
  }
  if (threshold === wakeWordSettings.threshold) {
    elements.wakeWordThresholdInput.value = String(wakeWordSettings.threshold);
    return;
  }
  wakeWordOperation = true;
  elements.wakeWordEnabledToggle.disabled = true;
  elements.wakeWordPhraseInput.disabled = true;
  elements.wakeWordThresholdInput.disabled = true;
  try {
    const conversationActive =
      latestSnapshot.connectionState === "connected" ||
      latestSnapshot.connectionState === "reconnecting";
    renderWakeWordSettings(
      await setNativeWakeWordThreshold(threshold, !conversationActive),
    );
  } catch (error) {
    await refreshWakeWordSettings().catch(() => undefined);
    showError("wakeWord.updateFailed", { error });
  } finally {
    wakeWordOperation = false;
    if (wakeWordSettings !== null) {
      renderWakeWordSettings(wakeWordSettings);
    }
  }
}

function scheduleWakeWordPhraseUpdate(): void {
  if (wakeWordPhraseComposing) {
    return;
  }
  if (wakeWordPhraseUpdateTimer !== null) {
    window.clearTimeout(wakeWordPhraseUpdateTimer);
  }
  wakeWordPhraseUpdateTimer = window.setTimeout(() => {
    wakeWordPhraseUpdateTimer = null;
    void updateWakeWordPhrase(elements.wakeWordPhraseInput.value);
  }, WAKE_WORD_PHRASE_DEBOUNCE_MS);
}

function submitWakeWordPhraseUpdate(): void {
  if (wakeWordPhraseUpdateTimer !== null) {
    window.clearTimeout(wakeWordPhraseUpdateTimer);
    wakeWordPhraseUpdateTimer = null;
  }
  void updateWakeWordPhrase(elements.wakeWordPhraseInput.value);
}

async function detachCurrentAdapter(): Promise<void> {
  const previousAdapter = adapter;
  unsubscribeToolUI?.();
  unsubscribeToolUI = null;
  toolUIAdapter?.close();
  toolUIAdapter = null;
  unsubscribe?.();
  unsubscribe = null;
  adapter = null;
  renderSnapshot(EMPTY_SNAPSHOT);
  if (previousAdapter) {
    await previousAdapter.disconnect().catch(() => undefined);
    await resumeWakeWordAfterConversation();
  }
}

async function discoverBackend(
  sessionIdToRestore?: string | null,
): Promise<void> {
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
    backendConnection = connection;
    const nextAdapter = new XtalkClientAdapter(connection);
    const nextToolUIAdapter = new ToolUIAdapter(connection);
    toolUIAdapter = nextToolUIAdapter;
    unsubscribeToolUI = nextToolUIAdapter.subscribe(handleToolUIEvent);
    nextToolUIAdapter.connect();
    adapter = nextAdapter;
    unsubscribe = nextAdapter.subscribe(renderSnapshot);

    let restoredSessionId = sessionIdToRestore;
    if (restoredSessionId === undefined) {
      restoredSessionId =
        latestSnapshot.sessionId ?? readActiveSessionId();
      if (restoredSessionId === null) {
        const sessions = await nextAdapter.getSessions();
        restoredSessionId = sessions[0]?.id ?? null;
      }
    }

    let sessionRestoreError: unknown = null;
    if (restoredSessionId !== null) {
      try {
        await nextAdapter.switchSession(restoredSessionId);
        persistActiveSessionId(restoredSessionId);
      } catch (error) {
        sessionRestoreError = error;
      }
    }

    elements.backendDetail.textContent = nextAdapter.diagnostics.origin;
    elements.websocketDetail.textContent = nextAdapter.diagnostics.websocketURL;
    setBackendStatus("ready", "service.ready");
    await refreshChatSessions();
    if (sessionRestoreError !== null) {
      showError("sidebar.switchFailed", { error: sessionRestoreError });
    }
  } catch (error) {
    adapter = null;
    setBackendStatus("offline", "service.unavailable");
    elements.backendDetail.textContent = t("service.offlineMode");
    showError("service.connectFailed", { error });
    setDiagnosticsOpen(true);
  } finally {
    discoveringBackend = false;
    updateControls(latestSnapshot);
    if (pendingWakeWordActivation) {
      void activatePendingWakeWordConversation();
    }
  }
}

async function applyModelConfigPath(selectedPath: string): Promise<void> {
  const sessionIdToRestore = latestSnapshot.sessionId;
  let stopManagedProgress: (() => void) | null = null;
  let managedProgressOpened = false;

  try {
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
    developerToolChangesPending = false;
    credentialChangesPending = false;
    const selection = await refreshModelConfigSelection();
    await refreshCredentials();
    updateDeveloperToolsStatus();
    elements.modelConfigStatus.textContent = selection.configPath
      ? t("model.appliedRestarted")
      : t("model.applied");
    await discoverBackend(sessionIdToRestore);
    if (managedProgressOpened) {
      closeManagedProgress();
    }
  } catch (error) {
    await refreshModelConfigSelection().catch(() => undefined);
    if (modelConfigPath === null) {
      setBackendStatus("unconfigured", "service.chooseConfig");
    } else {
      await discoverBackend(sessionIdToRestore);
    }
    elements.modelConfigStatus.textContent = t("model.applyFailed");
    showError("model.applyFailedDetail", { error });
    setDiagnosticsOpen(true);
    if (managedProgressOpened) {
      failManagedProgress(error);
    }
  } finally {
    stopManagedProgress?.();
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
    await applyModelConfigPath(selectedPath);
  } finally {
    modelConfigOperation = false;
    updateControls(adapter?.snapshot ?? latestSnapshot);
  }
}

function openFirstLaunchDialog(): void {
  if (modelConfigPath !== null || elements.firstLaunchDialog.open) {
    return;
  }
  elements.firstLaunchDialog.showModal();
  elements.recommendedConfigButton.focus();
}

function cancelFirstLaunchChoice(): void {
  elements.firstLaunchDialog.close();
  setBackendStatus("unconfigured", "service.chooseConfig");
  elements.modelConfigStatus.textContent = t("model.firstLaunch");
}

async function chooseCustomConfig(): Promise<void> {
  elements.firstLaunchDialog.close();
  await chooseAndApplyModelConfig(true);
}

async function chooseRecommendedConfig(): Promise<void> {
  if (modelConfigPath !== null) {
    return;
  }
  elements.firstLaunchDialog.close();
  showError(null);
  elements.modelConfigStatus.textContent = t("model.choosePrompt");
  try {
    recommendedConfigPath = await getNativeRecommendedModelConfig();
    elements.llmKeyInput.value = "";
    elements.llmKeyInput.setCustomValidity("");
    elements.llmKeyDialog.showModal();
    elements.llmKeyInput.focus();
  } catch (error) {
    recommendedConfigPath = null;
    setBackendStatus("unconfigured", "service.chooseConfig");
    elements.modelConfigStatus.textContent = t("model.firstLaunch");
    showError("model.applyFailedDetail", { error });
    setDiagnosticsOpen(true);
  }
}

async function applyRecommendedConfig(): Promise<void> {
  const configPath = recommendedConfigPath;
  recommendedConfigPath = null;
  if (configPath === null) {
    setBackendStatus("unconfigured", "service.chooseConfig");
    return;
  }
  await applyModelConfigPath(configPath);
}

function skipLlmKey(): void {
  elements.llmKeyDialog.close();
  elements.llmKeyInput.value = "";
  void applyRecommendedConfig();
}

function cancelLlmKeyDialog(): void {
  elements.llmKeyDialog.close();
  recommendedConfigPath = null;
  setBackendStatus("unconfigured", "service.chooseConfig");
  elements.modelConfigStatus.textContent = t("model.firstLaunch");
}

async function saveLlmKeyAndContinue(): Promise<void> {
  const value = elements.llmKeyInput.value.trim();
  if (!value) {
    elements.llmKeyInput.setCustomValidity(t("credentials.keyValidation"));
    elements.llmKeyInput.reportValidity();
    return;
  }
  if (credentialOperation) {
    return;
  }

  credentialOperation = true;
  showError(null);
  updateCredentialStatus(t("llm.saving"));
  updateToolControls();
  try {
    await saveNativeCredential("llm", value);
    elements.llmKeyDialog.close();
    elements.llmKeyInput.value = "";
    credentialChangesPending = true;
    await refreshCredentials();
    updateCredentialStatus(t("llm.saved"));
    await applyRecommendedConfig();
  } catch (error) {
    showError("llm.saveFailedDetail", { error });
  } finally {
    credentialOperation = false;
    updateToolControls();
  }
}

async function restartCurrentModelConfig(): Promise<void> {
  if (modelConfigOperation || modelConfigPath === null) {
    return;
  }

  modelConfigOperation = true;
  showError(null);
  elements.modelConfigStatus.textContent = t("model.restarting");
  updateControls(latestSnapshot);
  try {
    await applyModelConfigPath(modelConfigPath);
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
    developerToolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.installed", {
        name: resolveToolDisplayName(installed.displayName),
      }),
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
    developerToolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.updated", {
        name: resolveToolDisplayName(updated.displayName),
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
  if (tool?.canDelete === false) {
    showError("tools.builtinImmutable");
    return;
  }
  toolOperation = true;
  showError(null);
  updateDeveloperToolsStatus(t("tools.removing"));
  updateControls(latestSnapshot);

  try {
    await removeNativeInstalledTool(toolId);
    developerToolChangesPending = true;
    await refreshInstalledTools();
    updateDeveloperToolsStatus(
      t("tools.removed", {
        name:
          tool === undefined
            ? t("tools.generic")
            : resolveToolDisplayName(tool.displayName),
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

function openCredentialDialog(credentialId: string): void {
  const credential = credentials.find((item) => item.id === credentialId);
  if (
    credential === undefined ||
    credential.source === "environment" ||
    !credential.storageAvailable
  ) {
    return;
  }
  selectedCredentialId = credentialId;
  elements.credentialDialogLabel.textContent = t("credentials.keyFor", {
    name: resolveToolDisplayName(credential.displayName),
  });
  elements.credentialDialogInput.value = "";
  elements.credentialDialogInput.setCustomValidity("");
  elements.credentialDialog.showModal();
  elements.credentialDialogInput.focus();
}

function cancelCredentialDialog(): void {
  selectedCredentialId = null;
  elements.credentialDialogInput.value = "";
  elements.credentialDialog.close();
}

async function saveCredential(): Promise<void> {
  const credentialId = selectedCredentialId;
  const value = elements.credentialDialogInput.value.trim();
  if (credentialId === null) {
    cancelCredentialDialog();
    return;
  }
  if (!value) {
    elements.credentialDialogInput.setCustomValidity(
      t("credentials.keyValidation"),
    );
    elements.credentialDialogInput.reportValidity();
    return;
  }

  credentialOperation = true;
  showError(null);
  updateCredentialStatus(t("credentials.saving"));
  updateToolControls();
  try {
    await saveNativeCredential(credentialId, value);
    credentialChangesPending = true;
    cancelCredentialDialog();
    await refreshCredentials();
    updateCredentialStatus(t("credentials.saved"));
  } catch (error) {
    updateCredentialStatus(t("credentials.saveFailed"));
    showError("credentials.saveFailedDetail", { error });
  } finally {
    credentialOperation = false;
    updateToolControls();
  }
}

async function removeCredential(credentialId: string): Promise<void> {
  if (credentialOperation) {
    return;
  }
  credentialOperation = true;
  showError(null);
  updateCredentialStatus(t("credentials.deleting"));
  updateToolControls();
  try {
    await deleteNativeCredential(credentialId);
    credentialChangesPending = true;
    await Promise.all([refreshCredentials(), refreshInstalledTools()]);
    updateCredentialStatus(t("credentials.deleted"));
  } catch (error) {
    updateCredentialStatus(t("credentials.deleteFailed"));
    showError("credentials.deleteFailedDetail", { error });
  } finally {
    credentialOperation = false;
    updateToolControls();
  }
}

async function applyToolChanges(): Promise<void> {
  if (
    toolOperation ||
    (!developerToolChangesPending && !credentialChangesPending)
  ) {
    return;
  }
  if (modelConfigPath === null) {
    updateDeveloperToolsStatus(t("composer.chooseConfig"));
    return;
  }

  const sessionIdToRestore = latestSnapshot.sessionId;
  toolOperation = true;
  showError(null);
  updateCredentialStatus(t("tools.restarting"));
  updateDeveloperToolsStatus(t("tools.restarting"));
  setBackendStatus("loading", "tools.applying");
  updateControls(latestSnapshot);

  try {
    await detachCurrentAdapter();
    await applyNativeToolChanges();
    developerToolChangesPending = false;
    credentialChangesPending = false;
    await refreshCredentials();
    updateDeveloperToolsStatus(t("tools.applied"));
    await discoverBackend(sessionIdToRestore);
  } catch (error) {
    updateCredentialStatus(t("tools.applyFailed"));
    updateDeveloperToolsStatus(t("tools.applyFailed"));
    await discoverBackend(sessionIdToRestore);
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
      openFirstLaunchDialog();
      return;
    }
    const managedPlan = await getNativeManagedModelPlan(selection.configPath);
    let stopManagedProgress: (() => void) | null = null;
    if (managedPlan.services.length > 0) {
      openManagedProgress(managedPlan.services);
      stopManagedProgress = await listenNativeManagedModelProgress(
        updateManagedProgress,
      );
    }
    try {
      await ensureNativeBackendStarted();
      if (managedPlan.services.length > 0) {
        closeManagedProgress();
      }
    } catch (error) {
      if (managedPlan.services.length > 0) {
        failManagedProgress(error);
      }
      throw error;
    } finally {
      stopManagedProgress?.();
    }
    await discoverBackend();
  } catch (error) {
    setBackendStatus("offline", "service.runtimeUnavailable");
    elements.modelConfigStatus.textContent = t("model.readFailed");
    showError("app.initializeFailed", { error });
    setDiagnosticsOpen(true);
  }
}

async function initializeNativeWakeWord(): Promise<void> {
  await listenNativeWakeWordStatus(renderWakeWordSettings);
  await listenNativeWakeWordDetected(() => {
    pendingWakeWordActivation = true;
    void activatePendingWakeWordConversation();
  });
  await listenNativeAppBackgrounding(() => {
    backgroundingRequested = true;
    pendingWakeWordActivation = false;
    void disconnectSession();
  });
  await refreshWakeWordSettings();
}

async function activatePendingWakeWordConversation(): Promise<void> {
  if (!pendingWakeWordActivation) {
    return;
  }
  if (discoveringBackend || backendState === "loading" || sessionOperation) {
    return;
  }
  if (adapter === null || backendState !== "ready") {
    pendingWakeWordActivation = false;
    await resumeWakeWordAfterConversation();
    return;
  }
  pendingWakeWordActivation = false;
  backgroundingRequested = false;
  await connectSession(true);
}

async function resumeWakeWordAfterConversation(): Promise<void> {
  if (!wakeWordSettings?.enabled) {
    return;
  }
  try {
    renderWakeWordSettings(await resumeNativeWakeWord());
  } catch (error) {
    await refreshWakeWordSettings().catch(() => undefined);
    showError("wakeWord.updateFailed", { error });
  }
}

async function connectSession(startNewConversation = false): Promise<void> {
  const activeAdapter = adapter;
  if (!activeAdapter || sessionOperation) {
    return;
  }

  sessionOperation = true;
  showError(null);
  updateControls(activeAdapter.snapshot);
  try {
    renderWakeWordSettings(await pauseNativeWakeWord());
    if (startNewConversation) {
      await activeAdapter.switchSession(null);
    }
    await activeAdapter.connect();
    scheduleChatSessionsRefresh();
  } catch (error) {
    showError("voice.connectFailed", { error });
    await resumeWakeWordAfterConversation();
  } finally {
    sessionOperation = false;
    updateControls(activeAdapter.snapshot);
    if (backgroundingRequested) {
      void disconnectSession();
    }
  }
}

async function disconnectSession(): Promise<void> {
  const activeAdapter = adapter;
  if (sessionOperation) {
    return;
  }
  if (!activeAdapter) {
    backgroundingRequested = false;
    await resumeWakeWordAfterConversation();
    return;
  }
  if (activeAdapter.snapshot.connectionState === "disconnected") {
    backgroundingRequested = false;
    await resumeWakeWordAfterConversation();
    return;
  }

  sessionOperation = true;
  backgroundingRequested = false;
  showError(null);
  updateControls(activeAdapter.snapshot);
  try {
    await activeAdapter.disconnect();
  } catch (error) {
    showError("voice.closeFailed", { error });
  } finally {
    backgroundingRequested = false;
    sessionOperation = false;
    updateControls(activeAdapter.snapshot);
    await resumeWakeWordAfterConversation();
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
elements.liveToolStatusToggle.addEventListener("click", () => {
  toolUILiveExpanded = !toolUILiveExpanded;
  renderLiveToolPanel();
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
elements.whiteboardButton.addEventListener("click", () => {
  traceWhiteboard("clicked", whiteboardVisible);
  void setWhiteboardVisible(!whiteboardVisible);
});
elements.selectModelConfigButton.addEventListener("click", () => {
  void chooseAndApplyModelConfig(false);
});
elements.recommendedConfigButton.addEventListener("click", () => {
  void chooseRecommendedConfig();
});
elements.customConfigButton.addEventListener("click", () => {
  void chooseCustomConfig();
});
elements.firstLaunchCancelButton.addEventListener("click", () => {
  cancelFirstLaunchChoice();
});
elements.firstLaunchDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  cancelFirstLaunchChoice();
});
elements.llmKeyForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void saveLlmKeyAndContinue();
});
elements.llmKeySkipButton.addEventListener("click", () => {
  skipLlmKey();
});
elements.llmKeyDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  cancelLlmKeyDialog();
});
elements.llmKeyInput.addEventListener("input", () => {
  elements.llmKeyInput.setCustomValidity("");
});
elements.installToolDirectoryButton.addEventListener("click", () => {
  void chooseAndInstallToolDirectory();
});
elements.wakeWordEnabledToggle.addEventListener("change", () => {
  void updateWakeWordEnabled(elements.wakeWordEnabledToggle.checked);
});
elements.wakeWordPhraseInput.addEventListener("change", () => {
  submitWakeWordPhraseUpdate();
});
elements.wakeWordPhraseInput.addEventListener("input", () => {
  scheduleWakeWordPhraseUpdate();
});
elements.wakeWordPhraseInput.addEventListener("compositionstart", () => {
  wakeWordPhraseComposing = true;
  if (wakeWordPhraseUpdateTimer !== null) {
    window.clearTimeout(wakeWordPhraseUpdateTimer);
    wakeWordPhraseUpdateTimer = null;
  }
});
elements.wakeWordPhraseInput.addEventListener("compositionend", () => {
  wakeWordPhraseComposing = false;
  scheduleWakeWordPhraseUpdate();
});
elements.wakeWordPhraseInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    elements.wakeWordPhraseInput.blur();
  }
});
elements.wakeWordThresholdInput.addEventListener("change", () => {
  void updateWakeWordThreshold(elements.wakeWordThresholdInput.value);
});
elements.wakeWordThresholdInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    elements.wakeWordThresholdInput.blur();
  }
});
elements.credentialForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void saveCredential();
});
elements.credentialDialogInput.addEventListener("input", () => {
  elements.credentialDialogInput.setCustomValidity("");
});
elements.credentialCancelButton.addEventListener("click", () => {
  cancelCredentialDialog();
});
elements.credentialDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  cancelCredentialDialog();
});
elements.deleteSessionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  void deletePendingSession();
});
elements.deleteSessionCancelButton.addEventListener("click", () => {
  cancelDeleteSessionDialog();
});
elements.deleteSessionDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  cancelDeleteSessionDialog();
});
elements.applyToolChangesButton.addEventListener("click", () => {
  void applyToolChanges();
});
elements.applyCredentialChangesButton.addEventListener("click", () => {
  void applyToolChanges();
});
elements.textComposer.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendTextMessage();
});
elements.messageInput.addEventListener("input", () => {
  resizeMessageInput();
  updateComposer(latestSnapshot);
});
elements.messageInput.addEventListener("compositionstart", () => {
  messageInputCompositionActive = true;
  messageInputCompositionEnterHeld = false;
});
elements.messageInput.addEventListener("compositionend", () => {
  messageInputCompositionActive = false;
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (isMessageInputCompositionEnter(event)) {
    if (
      !event.isComposing &&
      !messageInputCompositionActive &&
      event.keyCode !== 229
    ) {
      event.preventDefault();
    }
    return;
  }
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (!elements.sendTextButton.disabled) {
      void sendTextMessage();
    }
  }
});
elements.messageInput.addEventListener("keyup", (event) => {
  if (event.key === "Enter") {
    messageInputCompositionEnterHeld = false;
  }
});
elements.messageInput.addEventListener("blur", () => {
  messageInputCompositionActive = false;
  messageInputCompositionEnterHeld = false;
});
elements.retryButton.addEventListener("click", () => {
  if (modelConfigPath === null) {
    openFirstLaunchDialog();
  } else {
    void restartCurrentModelConfig();
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
  if (
    elements.credentialDialog.open ||
    elements.firstLaunchDialog.open ||
    elements.llmKeyDialog.open
  ) {
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
void listenWhiteboardWindowHidden(() => {
  whiteboardVisible = false;
  persistWhiteboardVisiblePreference(false);
  updateWhiteboardButton();
});
// The whiteboard always starts hidden after a service restart. Only the dock
// button or the first whiteboard tool emit opens the window, so a previously
// visible board never resurfaces an unwanted window on launch.
whiteboardVisible = false;
persistWhiteboardVisiblePreference(false);

async function initializeDesktopApplication(): Promise<void> {
  try {
    await initializeNativeWakeWord();
  } catch (error) {
    showError("wakeWord.updateFailed", { error });
  }
  await initializeApplication();
}

void initializeDesktopApplication();
