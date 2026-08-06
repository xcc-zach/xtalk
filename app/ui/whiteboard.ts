import { getCurrentWebviewWindow } from "@tauri-apps/api/webviewWindow";

import {
  getNativeBackendConnection,
  type NativeBackendConnection,
} from "./adapters/native-capabilities";
import { t } from "./i18n";
import { renderMarkdown } from "./markdown";

/**
 * Normalized snapshot returned by the whiteboard sidecar endpoint.
 */
interface WhiteboardSnapshot {
  version: number;
  text: string;
  revision: number;
  updated_at: string;
}

const POLL_INTERVAL_MS = 1_000;
const RETRY_CONNECTION_INTERVAL_MS = 3_000;
const ACTIVE_SESSION_STORAGE_KEY = "xtalk.desktop.active-session.v1";

const elements = {
  board: requireElement<HTMLElement>("board"),
  empty: requireElement<HTMLElement>("empty"),
  status: requireElement<HTMLElement>("status"),
};

let connection: NativeBackendConnection | null = null;
let lastRenderedRevision = -1;
let lastRenderedSession: string | null = null;
let polling = false;
let connectionFailedAt = 0;

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`Required UI element #${id} is missing.`);
  }
  return element as T;
}

/** Returns the active conversation id persisted by the main window. */
function activeSessionId(): string | null {
  const sessionId = localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY);
  return sessionId?.trim() || null;
}

/**
 * Applies the localized window title through the native webview window.
 */
function applyWindowTitle(): void {
  const title = t("whiteboard.windowTitle");
  document.title = title;
  void getCurrentWebviewWindow().setTitle(title);
}

/**
 * Renders one snapshot into the window, skipping unchanged revisions.
 *
 * @param sessionId Conversation owning the snapshot.
 * @param snapshot Whiteboard snapshot to display.
 */
function renderSnapshot(sessionId: string, snapshot: WhiteboardSnapshot): void {
  if (
    sessionId === lastRenderedSession &&
    snapshot.revision === lastRenderedRevision &&
    elements.board.childElementCount > 0
  ) {
    return;
  }
  lastRenderedSession = sessionId;
  lastRenderedRevision = snapshot.revision;
  const text = typeof snapshot.text === "string" ? snapshot.text : "";
  elements.board.innerHTML = renderMarkdown(text);
  elements.empty.hidden = text.trim().length !== 0;
  elements.empty.textContent = t("whiteboard.empty");
  const updated = new Date(snapshot.updated_at);
  elements.status.textContent =
    Number.isNaN(updated.getTime())
      ? ""
      : t("whiteboard.updated", { time: updated.toLocaleTimeString() });
}

/**
 * Fetches the active conversation's whiteboard snapshot from the sidecar.
 */
async function refresh(): Promise<void> {
  if (polling) {
    return;
  }
  polling = true;
  try {
    const sessionId = activeSessionId();
    if (sessionId === null) {
      elements.board.replaceChildren();
      elements.empty.hidden = false;
      elements.empty.textContent = t("whiteboard.empty");
      elements.status.textContent = "";
      return;
    }
    if (connection === null) {
      connection = await getNativeBackendConnection();
    }
    const url = new URL("/app/api/whiteboard", connection.origin);
    url.searchParams.set("app_token", connection.launchToken);
    url.searchParams.set("session_id", sessionId);
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const snapshot = (await response.json()) as WhiteboardSnapshot;
    if (
      typeof snapshot !== "object" ||
      snapshot === null ||
      typeof snapshot.text !== "string" ||
      typeof snapshot.revision !== "number"
    ) {
      throw new Error("Invalid whiteboard snapshot.");
    }
    renderSnapshot(sessionId, snapshot);
    elements.status.hidden = false;
  } catch (error) {
    elements.status.hidden = false;
    elements.status.textContent = t("whiteboard.unavailable", {
      error: error instanceof Error ? error.message : String(error),
    });
    const now = Date.now();
    if (
      connection !== null &&
      now - connectionFailedAt >= RETRY_CONNECTION_INTERVAL_MS
    ) {
      connection = null;
      connectionFailedAt = now;
    }
  } finally {
    polling = false;
  }
}

applyWindowTitle();
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    void refresh();
  }
});
window.addEventListener("storage", (event) => {
  if (event.key === ACTIVE_SESSION_STORAGE_KEY) {
    lastRenderedSession = null;
    void refresh();
  }
});

void refresh();
setInterval(() => {
  if (!document.hidden) {
    void refresh();
  }
}, POLL_INTERVAL_MS);
