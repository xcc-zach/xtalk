import type { NativeBackendConnection } from "./native-capabilities";

/**
 * Live status emitted by an App-observed asynchronous developer tool.
 */
export interface ToolUIStatusEvent {
  type: "tool_ui.status";
  toolId: string;
  toolName: string;
  callId: string;
  sessionId: string | null;
  sequence: number;
  status: string;
  running: boolean;
  updatedAt: string;
}

/**
 * Immutable emit observation produced by a developer tool lifecycle hook.
 */
export interface ToolUIEmitEvent {
  type: "tool_ui.emit";
  toolId: string;
  toolName: string;
  callId: string;
  sessionId: string | null;
  sequence: number;
  message: string;
  status: string;
  running: boolean;
  emittedAt: string;
}

/** Event delivered through the App-only read-only Tool UI channel. */
export type ToolUIEvent = ToolUIStatusEvent | ToolUIEmitEvent;

/** Listener for validated Tool UI observations. */
export type ToolUIListener = (event: ToolUIEvent) => void;

const APP_TOKEN_QUERY_PARAMETER = "app_token";

/**
 * Maintains the independent read-only Tool UI WebSocket.
 */
export class ToolUIAdapter {
  readonly #origin: URL;
  readonly #launchToken: string;
  readonly #url: URL;
  readonly #listeners = new Set<ToolUIListener>();
  #websocket: WebSocket | null = null;
  #closed = false;
  #sessionId: string | null = null;
  #reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Creates an adapter from trusted loopback connection data.
   *
   * @param connection Tauri-validated sidecar origin and launch token.
   */
  constructor(connection: NativeBackendConnection) {
    this.#origin = new URL(connection.origin);
    this.#launchToken = connection.launchToken;
    this.#url = new URL("/app/tool-ui/ws", connection.origin);
    this.#url.protocol = this.#url.protocol === "https:" ? "wss:" : "ws:";
    this.#url.searchParams.set(
      APP_TOKEN_QUERY_PARAMETER,
      connection.launchToken,
    );
  }

  /** Opens the read-only Tool UI channel. */
  connect(): void {
    if (this.#closed || this.#websocket !== null) {
      return;
    }
    const websocket = new WebSocket(this.#url);
    this.#websocket = websocket;
    websocket.addEventListener("open", () => {
      this.#sendSessionBinding();
    });
    websocket.addEventListener("message", (event) => {
      if (typeof event.data !== "string") {
        return;
      }
      try {
        const payload: unknown = JSON.parse(event.data);
        const toolEvent = parseToolUIEvent(payload);
        for (const listener of this.#listeners) {
          listener(toolEvent);
        }
      } catch {
        // Ignore malformed untrusted tool UI observations.
      }
    });
    websocket.addEventListener("close", () => {
      if (this.#websocket === websocket) {
        this.#websocket = null;
      }
      if (!this.#closed) {
        this.#reconnectTimer = setTimeout(() => {
          this.#reconnectTimer = null;
          this.connect();
        }, 1000);
      }
    });
  }

  /**
   * Binds future tool calls to the active persisted chat session.
   *
   * @param sessionId Current session identifier or `null` while detached.
   */
  bindSession(sessionId: string | null): void {
    if (this.#sessionId === sessionId) {
      return;
    }
    this.#sessionId = sessionId;
    this.#sendSessionBinding();
  }

  /**
   * Subscribes to validated read-only Tool UI events.
   *
   * @param listener Event listener.
   * @returns Function removing the listener.
   */
  subscribe(listener: ToolUIListener): () => void {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  /**
   * Publishes a prepared document behind a short-lived one-time frame URL.
   *
   * @param source Complete sandbox frame HTML prepared by the trusted App.
   * @returns Loopback URL that can load the document exactly once.
   */
  async createFrame(source: string): Promise<string> {
    const response = await fetch(
      new URL("/app/api/tool-ui/frames", this.#origin),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-XTalk-App-Token": this.#launchToken,
        },
        body: JSON.stringify({ source }),
      },
    );
    if (!response.ok) {
      throw new Error(`Could not prepare tool UI frame (${response.status}).`);
    }
    const payload: unknown = await response.json();
    if (
      !isRecord(payload) ||
      typeof payload.ticket !== "string" ||
      !/^[A-Za-z0-9_-]{32,128}$/u.test(payload.ticket)
    ) {
      throw new Error("Tool UI frame ticket is invalid.");
    }
    return new URL(
      `/tool-ui-frame/${payload.ticket}`,
      this.#origin,
    ).toString();
  }

  /** Permanently closes this adapter and cancels reconnection. */
  close(): void {
    this.#closed = true;
    if (this.#reconnectTimer !== null) {
      clearTimeout(this.#reconnectTimer);
      this.#reconnectTimer = null;
    }
    this.#websocket?.close();
    this.#websocket = null;
    this.#listeners.clear();
  }

  #sendSessionBinding(): void {
    if (this.#websocket?.readyState !== WebSocket.OPEN) {
      return;
    }
    this.#websocket.send(
      JSON.stringify({
        type: "bind_session",
        sessionId: this.#sessionId,
      }),
    );
  }
}

function parseToolUIEvent(payload: unknown): ToolUIEvent {
  if (!isRecord(payload)) {
    throw new Error("Tool UI event root must be an object.");
  }
  const baseValid =
    typeof payload.toolId === "string" &&
    Boolean(payload.toolId) &&
    typeof payload.toolName === "string" &&
    Boolean(payload.toolName) &&
    typeof payload.callId === "string" &&
    Boolean(payload.callId) &&
    (payload.sessionId === null || typeof payload.sessionId === "string") &&
    isNonNegativeInteger(payload.sequence) &&
    typeof payload.status === "string" &&
    typeof payload.running === "boolean";
  if (!baseValid) {
    throw new Error("Tool UI event contains invalid shared fields.");
  }
  if (
    payload.type === "tool_ui.status" &&
    typeof payload.updatedAt === "string"
  ) {
    return payload as unknown as ToolUIStatusEvent;
  }
  if (
    payload.type === "tool_ui.emit" &&
    typeof payload.message === "string" &&
    typeof payload.emittedAt === "string"
  ) {
    return payload as unknown as ToolUIEmitEvent;
  }
  throw new Error("Tool UI event type is not supported.");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}
