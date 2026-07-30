import { invoke } from "@tauri-apps/api/core";

/**
 * Connection bootstrap data returned by the trusted Tauri layer.
 */
export interface NativeBackendConnection {
  /** Canonical loopback HTTP origin of the Python sidecar. */
  origin: string;
  /** Per-launch credential used only for explicit sidecar HTTP endpoints. */
  launchToken: string;
}

const BACKEND_CONNECTION_COMMAND = "get_backend_connection";
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
