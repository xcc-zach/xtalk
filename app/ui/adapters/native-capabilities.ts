import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";

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

const APPLY_MODEL_CONFIG_COMMAND = "apply_model_config";
const BACKEND_CONNECTION_COMMAND = "get_backend_connection";
const MODEL_CONFIG_SELECTION_COMMAND = "get_model_config_selection";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "[::1]", "::1", "localhost"]);

/**
 * Requests the current sidecar endpoint and launch credential from Tauri.
 *
 * @returns Validated bootstrap data for the local XTalk adapters.
 * @throws When Tauri is unavailable or returns a non-loopback endpoint.
 */
export async function getNativeBackendConnection(): Promise<NativeBackendConnection> {
  if (!("__TAURI_INTERNALS__" in globalThis)) {
    throw new Error("桌面运行时不可用；当前界面已进入离线模式。");
  }

  const payload = await invoke<unknown>(BACKEND_CONNECTION_COMMAND);
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
 * Opens the native JSON file picker for a model configuration.
 *
 * @returns Selected filesystem path, or `null` when the user cancels.
 */
export async function chooseNativeModelConfigFile(): Promise<string | null> {
  requireTauriRuntime();
  const selection = await open({
    directory: false,
    multiple: false,
    title: "选择 XTalk 模型配置",
    filters: [
      {
        name: "JSON 配置",
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

function requireTauriRuntime(): void {
  if (!("__TAURI_INTERNALS__" in globalThis)) {
    throw new Error("桌面运行时不可用；当前界面已进入离线模式。");
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
