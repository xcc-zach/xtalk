import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

import { t } from "../i18n";

/**
 * Connection bootstrap data returned by the trusted Tauri layer.
 */
export interface NativeBackendConnection {
  /** Canonical loopback HTTP origin of the Python sidecar. */
  origin: string;
  /** Per-launch credential used only for explicit sidecar HTTP endpoints. */
  launchToken: string;
}

/**
 * External model configuration currently selected by the desktop user.
 */
export interface NativeModelConfigSelection {
  /** Canonical JSON configuration path, or `null` before first selection. */
  configPath: string | null;
}

/** One App-owned external service credential without its secret value. */
export interface NativeCredentialDefinition {
  /** Stable credential identifier from the packaged App registry. */
  id: string;
  /** Localized human-readable service name. */
  displayName: string | Record<string, string>;
  /** Whether an environment variable or system credential is available. */
  configured: boolean;
  /** Active environment-first credential source. */
  source: "environment" | "system" | "missing";
  /** Whether the platform credential manager could be accessed. */
  storageAvailable: boolean;
}

/**
 * Managed services referenced by one selected model configuration.
 */
export interface NativeManagedModelPlan {
  /** Stable managed service identifiers in startup order. */
  services: string[];
}

/**
 * Progress emitted while managed model files and services are prepared.
 */
export interface NativeManagedModelProgress {
  /** Current native preparation phase. */
  phase: "checking" | "downloading" | "starting" | "ready" | "complete";
  /** Stable service identifier, or `null` while finalizing the backend. */
  serviceId: string | null;
  /** One-based index of the active service. */
  serviceIndex: number;
  /** Number of managed services requested by the configuration. */
  serviceCount: number;
  /** Verified download bytes for the active service. */
  completedBytes: number;
  /** Total download bytes for the active service. */
  totalBytes: number;
  /** Current manifest-relative file path, when downloading. */
  filePath: string | null;
}

/**
 * One built-in or user-installed tool exposed by the native shell.
 */
export interface NativeToolDefinition {
  /** Stable identifier assigned by the App. */
  id: string;
  /** App-owned source classification, never supplied by the tool manifest. */
  origin: "builtin" | "user";
  /** Whether the native layer permits deleting this tool directory. */
  canDelete: boolean;
  /** Whether the native layer permits disabling this tool. */
  canDisable: boolean;
  /** Human-readable name declared by the developer tool. */
  displayName: string | Record<string, string>;
  /** Python `module:factory` entrypoint declared by the tool. */
  entrypoint: string;
  /** Optional read-only custom UI configuration. */
  ui: {
    /** Self-contained HTML entrypoint relative to the installed tool. */
    entrypoint: string;
    /** Live status polling interval, or `-1` when polling is disabled. */
    updateEveryS: number;
  } | null;
  /** Whether the sidecar loads this tool during its next restart. */
  enabled: boolean;
}

/**
 * Self-contained HTML returned for one installed tool UI.
 */
export interface NativeToolUiSource {
  /** Installed tool identifier owning the source. */
  toolId: string;
  /** Untrusted self-contained HTML loaded only inside a sandbox iframe. */
  source: string;
}

const APPLY_MODEL_CONFIG_COMMAND = "apply_model_config";
const APPLY_TOOL_CHANGES_COMMAND = "apply_tool_changes";
const BACKEND_CONNECTION_COMMAND = "get_backend_connection";
const CREDENTIALS_COMMAND = "get_credentials";
const DELETE_CREDENTIAL_COMMAND = "delete_credential";
const ENSURE_BACKEND_STARTED_COMMAND = "ensure_backend_started";
const INSTALLED_TOOLS_COMMAND = "get_installed_tools";
const INSTALL_TOOL_DIRECTORY_COMMAND = "install_tool_directory";
const MANAGED_MODEL_PLAN_COMMAND = "get_managed_model_plan";
const MANAGED_MODEL_PROGRESS_EVENT = "managed-model-progress";
const MODEL_CONFIG_SELECTION_COMMAND = "get_model_config_selection";
const RECOMMENDED_MODEL_CONFIG_COMMAND = "get_recommended_model_config";
const REMOVE_INSTALLED_TOOL_COMMAND = "remove_installed_tool";
const SAVE_CREDENTIAL_COMMAND = "save_credential";
const SET_TOOL_ENABLED_COMMAND = "set_tool_enabled";
const TOOL_UI_SOURCE_COMMAND = "get_tool_ui_source";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "[::1]", "::1", "localhost"]);

/**
 * Requests the current sidecar endpoint and launch credential from Tauri.
 *
 * @returns Validated bootstrap data for the local XTalk adapters.
 * @throws When Tauri is unavailable or returns a non-loopback endpoint.
 */
export async function getNativeBackendConnection(): Promise<NativeBackendConnection> {
  if (!("__TAURI_INTERNALS__" in globalThis)) {
    throw new Error(t("native.runtimeUnavailable"));
  }

  const payload = await invoke<unknown>(BACKEND_CONNECTION_COMMAND);
  return parseNativeBackendConnection(payload);
}

/**
 * Starts the selected backend and managed services unless they are healthy.
 *
 * @returns Validated bootstrap data for the running local backend.
 */
export async function ensureNativeBackendStarted(): Promise<NativeBackendConnection> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(ENSURE_BACKEND_STARTED_COMMAND);
  return parseNativeBackendConnection(payload);
}

/**
 * Reads the persisted model configuration selection from Tauri.
 *
 * @returns Current canonical JSON path, or `null` when no file is selected.
 */
export async function getNativeModelConfigSelection(): Promise<NativeModelConfigSelection> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(MODEL_CONFIG_SELECTION_COMMAND);
  if (!isRecord(payload)) {
    throw new Error("Tauri returned an invalid model configuration payload.");
  }

  const configPath = payload.configPath;
  if (configPath !== null && typeof configPath !== "string") {
    throw new Error("Model configuration payload contains an invalid path.");
  }
  if (typeof configPath === "string" && !configPath.trim()) {
    throw new Error("Model configuration payload contains an empty path.");
  }
  return { configPath };
}

/**
 * Returns the validated bundled recommended model configuration path.
 *
 * @returns Absolute JSON configuration path bundled with the App.
 */
export async function getNativeRecommendedModelConfig(): Promise<string> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(RECOMMENDED_MODEL_CONFIG_COMMAND);
  if (typeof payload !== "string" || !payload.trim()) {
    throw new Error("Tauri returned an invalid recommended configuration path.");
  }
  return payload;
}

/** Lists external service credentials without returning secret values. */
export async function getNativeCredentials(): Promise<NativeCredentialDefinition[]> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(CREDENTIALS_COMMAND);
  if (!Array.isArray(payload)) {
    throw new Error("Tauri returned an invalid credential list.");
  }
  return payload.map(parseNativeCredentialDefinition);
}

/** Saves one external service credential in the platform credential manager. */
export async function saveNativeCredential(
  credentialId: string,
  value: string,
): Promise<NativeCredentialDefinition> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(SAVE_CREDENTIAL_COMMAND, {
    credentialId,
    value,
  });
  return parseNativeCredentialDefinition(payload);
}

/** Deletes one external service credential from the platform credential manager. */
export async function deleteNativeCredential(
  credentialId: string,
): Promise<NativeCredentialDefinition> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(DELETE_CREDENTIAL_COMMAND, {
    credentialId,
  });
  return parseNativeCredentialDefinition(payload);
}

/**
 * Opens the native JSON file picker for a model configuration.
 *
 * @returns Selected filesystem path, or `null` when the user cancels.
 */
export async function chooseNativeModelConfigFile(): Promise<string | null> {
  requireTauriRuntime();
  const selection = await open({
    directory: false,
    multiple: false,
    title: t("model.dialogTitle"),
    filters: [
      {
        name: t("model.dialogFilter"),
        extensions: ["json"],
      },
    ],
  });
  if (selection === null) {
    return null;
  }
  if (Array.isArray(selection)) {
    throw new Error("Model configuration picker returned multiple files.");
  }
  if (!selection.trim()) {
    throw new Error("Model configuration picker returned an empty path.");
  }
  return selection;
}

/**
 * Persists a selected model configuration and restarts the native sidecar.
 *
 * @param configPath Absolute JSON configuration path selected by the user.
 * @returns Validated connection details for the restarted sidecar.
 */
export async function applyNativeModelConfig(
  configPath: string,
): Promise<NativeBackendConnection> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(APPLY_MODEL_CONFIG_COMMAND, {
    configPath,
  });
  return parseNativeBackendConnection(payload);
}

/**
 * Inspects a selected configuration for managed model services.
 *
 * @param configPath Absolute JSON configuration path selected by the user.
 * @returns Managed services in native startup order.
 */
export async function getNativeManagedModelPlan(
  configPath: string,
): Promise<NativeManagedModelPlan> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(MANAGED_MODEL_PLAN_COMMAND, {
    configPath,
  });
  if (!isRecord(payload) || !Array.isArray(payload.services)) {
    throw new Error("Tauri returned an invalid managed model plan.");
  }
  const services = payload.services;
  if (!services.every((service) => typeof service === "string" && service)) {
    throw new Error("Managed model plan contains an invalid service identifier.");
  }
  return { services };
}

/**
 * Subscribes to native managed-model preparation progress.
 *
 * @param listener Callback invoked for each validated progress update.
 * @returns Function that removes the native event subscription.
 */
export async function listenNativeManagedModelProgress(
  listener: (progress: NativeManagedModelProgress) => void,
): Promise<UnlistenFn> {
  requireTauriRuntime();
  return listen<unknown>(MANAGED_MODEL_PROGRESS_EVENT, (event) => {
    listener(parseManagedModelProgress(event.payload));
  });
}

/**
 * Opens the native directory picker for a developer tool.
 *
 * @returns Selected directory path, or `null` when the user cancels.
 */
export async function chooseNativeToolDirectory(): Promise<string | null> {
  requireTauriRuntime();
  const selection = await open({
    directory: true,
    multiple: false,
    title: t("tools.dialogTitle"),
  });
  if (selection === null) {
    return null;
  }
  if (Array.isArray(selection)) {
    throw new Error("Tool directory picker returned multiple paths.");
  }
  if (!selection.trim()) {
    throw new Error("Tool directory picker returned an empty path.");
  }
  return selection;
}

/**
 * Lists built-in and user-installed tools known to the native shell.
 *
 * @returns Unified tool definitions sorted by display name.
 */
export async function getNativeInstalledTools(): Promise<NativeToolDefinition[]> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(INSTALLED_TOOLS_COMMAND);
  if (!Array.isArray(payload)) {
    throw new Error("Tauri returned an invalid installed tools payload.");
  }
  return payload.map(parseNativeToolDefinition);
}

/**
 * Copies one selected developer tool directory into application data.
 *
 * @param sourcePath Directory containing `xtalk_tool.json`.
 * @returns Installed tool definition.
 */
export async function installNativeToolDirectory(
  sourcePath: string,
): Promise<NativeToolDefinition> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(INSTALL_TOOL_DIRECTORY_COMMAND, {
    sourcePath,
  });
  return parseNativeToolDefinition(payload);
}

/**
 * Persists whether one built-in or user tool should load at sidecar startup.
 *
 * @param toolId Stable identifier returned by the native shell.
 * @param enabled Desired enabled state.
 * @returns Updated tool definition.
 */
export async function setNativeToolEnabled(
  toolId: string,
  enabled: boolean,
): Promise<NativeToolDefinition> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(SET_TOOL_ENABLED_COMMAND, {
    toolId,
    enabled,
  });
  return parseNativeToolDefinition(payload);
}

/**
 * Deletes one copied user tool directory from application data.
 *
 * @param toolId User-tool identifier returned by the native shell.
 */
export async function removeNativeInstalledTool(toolId: string): Promise<void> {
  requireTauriRuntime();
  await invoke(REMOVE_INSTALLED_TOOL_COMMAND, { toolId });
}

/**
 * Restarts the sidecar so persisted tool changes become active.
 *
 * @returns Validated connection details for the restarted sidecar.
 */
export async function applyNativeToolChanges(): Promise<NativeBackendConnection> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(APPLY_TOOL_CHANGES_COMMAND);
  return parseNativeBackendConnection(payload);
}

/**
 * Reads one built-in or user tool's self-contained UI entrypoint.
 *
 * @param toolId Tool identifier returned by Tauri.
 * @returns Untrusted HTML source for sandboxed rendering.
 */
export async function getNativeToolUiSource(
  toolId: string,
): Promise<NativeToolUiSource> {
  requireTauriRuntime();
  const payload = await invoke<unknown>(TOOL_UI_SOURCE_COMMAND, { toolId });
  if (
    !isRecord(payload) ||
    payload.toolId !== toolId ||
    typeof payload.source !== "string"
  ) {
    throw new Error("Tauri returned an invalid tool UI source.");
  }
  return {
    toolId,
    source: payload.source,
  };
}

function requireTauriRuntime(): void {
  if (!("__TAURI_INTERNALS__" in globalThis)) {
    throw new Error(t("native.runtimeUnavailable"));
  }
}

function parseNativeBackendConnection(
  payload: unknown,
): NativeBackendConnection {
  if (!isRecord(payload)) {
    throw new Error("Tauri returned an invalid backend connection payload.");
  }
  const rawOrigin = payload.origin;
  const launchToken = payload.launchToken;
  if (typeof rawOrigin !== "string" || typeof launchToken !== "string") {
    throw new Error("Backend connection payload is missing origin or launchToken.");
  }

  const origin = normalizeLoopbackOrigin(rawOrigin);
  const normalizedToken = launchToken.trim();
  if (!normalizedToken) {
    throw new Error("Backend connection payload contains an empty launchToken.");
  }

  return { origin, launchToken: normalizedToken };
}

function parseManagedModelProgress(
  payload: unknown,
): NativeManagedModelProgress {
  if (!isRecord(payload)) {
    throw new Error("Tauri returned invalid managed model progress.");
  }
  const phase = payload.phase;
  const serviceId = payload.serviceId;
  const filePath = payload.filePath;
  const numericFields = [
    payload.serviceIndex,
    payload.serviceCount,
    payload.completedBytes,
    payload.totalBytes,
  ];
  if (
    !["checking", "downloading", "starting", "ready", "complete"].includes(
      String(phase),
    ) ||
    (serviceId !== null && typeof serviceId !== "string") ||
    (filePath !== null && typeof filePath !== "string") ||
    !numericFields.every(
      (value) => typeof value === "number" && Number.isFinite(value) && value >= 0,
    )
  ) {
    throw new Error("Tauri returned malformed managed model progress.");
  }
  return {
    phase: phase as NativeManagedModelProgress["phase"],
    serviceId,
    serviceIndex: payload.serviceIndex as number,
    serviceCount: payload.serviceCount as number,
    completedBytes: payload.completedBytes as number,
    totalBytes: payload.totalBytes as number,
    filePath,
  };
}

function parseNativeToolDefinition(payload: unknown): NativeToolDefinition {
  if (!isRecord(payload)) {
    throw new Error("Tauri returned an invalid tool definition.");
  }
  const id = payload.id;
  const displayName = payload.displayName;
  const entrypoint = payload.entrypoint;
  const ui = payload.ui;
  const enabled = payload.enabled;
  const origin = payload.origin;
  const canDelete = payload.canDelete;
  const canDisable = payload.canDisable;
  if (
    typeof id !== "string" ||
    !isDisplayName(displayName) ||
    typeof entrypoint !== "string" ||
    typeof enabled !== "boolean" ||
    (origin !== "builtin" && origin !== "user") ||
    typeof canDelete !== "boolean" ||
    typeof canDisable !== "boolean" ||
    canDelete !== (origin === "user") ||
    !id.trim() ||
    !entrypoint.trim() ||
    !isToolUiConfig(ui)
  ) {
    throw new Error("Tool definition contains invalid fields.");
  }
  const normalizedUi =
    ui === null
      ? null
      : (ui as {
          entrypoint: string;
          update_every_s: number;
        });
  return {
    id,
    origin,
    canDelete,
    canDisable,
    displayName,
    entrypoint,
    ui:
      normalizedUi === null
        ? null
        : {
            entrypoint: normalizedUi.entrypoint,
            updateEveryS: normalizedUi.update_every_s,
          },
    enabled,
  };
}

function parseNativeCredentialDefinition(
  payload: unknown,
): NativeCredentialDefinition {
  if (!isRecord(payload)) {
    throw new Error("Tauri returned an invalid credential definition.");
  }
  const id = payload.id;
  const displayName = payload.displayName;
  const configured = payload.configured;
  const source = payload.source;
  const storageAvailable = payload.storageAvailable;
  if (
    typeof id !== "string" ||
    !id.trim() ||
    !isDisplayName(displayName) ||
    typeof configured !== "boolean" ||
    (source !== "environment" && source !== "system" && source !== "missing") ||
    configured !== (source !== "missing") ||
    typeof storageAvailable !== "boolean"
  ) {
    throw new Error("Credential definition contains invalid fields.");
  }
  return { id, displayName, configured, source, storageAvailable };
}

function isDisplayName(value: unknown): value is string | Record<string, string> {
  if (typeof value === "string") {
    return Boolean(value.trim());
  }
  if (!isRecord(value) || Object.keys(value).length === 0) {
    return false;
  }
  return Object.entries(value).every(
    ([language, name]) =>
      Boolean(language.trim()) && typeof name === "string" && Boolean(name.trim()),
  );
}

function isToolUiConfig(value: unknown): boolean {
  if (value === null) {
    return true;
  }
  if (!isRecord(value)) {
    return false;
  }
  const entrypoint = value.entrypoint;
  const updateEveryS = value.update_every_s;
  return (
    typeof entrypoint === "string" &&
    Boolean(entrypoint.trim()) &&
    typeof updateEveryS === "number" &&
    Number.isFinite(updateEveryS) &&
    (updateEveryS === -1 || (updateEveryS >= 0.1 && updateEveryS <= 3600))
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function normalizeLoopbackOrigin(rawOrigin: string): string {
  const parsedOrigin = new URL(rawOrigin);
  if (!["http:", "https:"].includes(parsedOrigin.protocol)) {
    throw new Error("Backend origin must use HTTP or HTTPS.");
  }
  if (!LOOPBACK_HOSTS.has(parsedOrigin.hostname)) {
    throw new Error("Backend origin must resolve to a loopback host.");
  }
  if (parsedOrigin.username || parsedOrigin.password || parsedOrigin.search || parsedOrigin.hash) {
    throw new Error("Backend origin must not contain credentials, a query, or a fragment.");
  }
  if (parsedOrigin.pathname !== "/" && parsedOrigin.pathname !== "") {
    throw new Error("Backend origin must not contain a path.");
  }
  return parsedOrigin.origin;
}
