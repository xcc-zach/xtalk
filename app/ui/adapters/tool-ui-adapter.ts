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
  /** Explicit lifecycle outcome; absent only in legacy persisted events. */
  outcome?: "running" | "complete" | "cancelled";
  /**
   * Character offset inside the anchored assistant message where the tool
   * call happened. Absent for legacy events and events without a UI binding.
   */
  textOffset?: number;
  emittedAt: string;
}

/** Event delivered through the App-only read-only Tool UI channel. */
export type ToolUIEvent = ToolUIStatusEvent | ToolUIEmitEvent;

/** Listener for validated Tool UI observations. */
export type ToolUIListener = (event: ToolUIEvent) => void;

const TOOL_UI_POLL_INTERVAL_MS = 350;
const TOOL_UI_RETRY_INTERVAL_MS = 1_000;
const MAX_DELIVERED_TOOL_UI_EVENTS = 1_000;

/**
 * Maintains the independent read-only Tool UI WebSocket.
 */
export class ToolUIAdapter {
  readonly #origin: URL;
  readonly #launchToken: string;
  readonly #listeners = new Set<ToolUIListener>();
  readonly #deliveredEvents = new Set<string>();
  #closed = false;
  #sessionId: string | null = null;
  #pollTimer: ReturnType<typeof setTimeout> | null = null;
  #pollAbortController: AbortController | null = null;
  #polling = false;

  /**
   * Creates an adapter from trusted loopback connection data.
   *
   * @param connection Tauri-validated sidecar origin and launch token.
   */
  constructor(connection: NativeBackendConnection) {
    this.#origin = new URL(connection.origin);
    this.#launchToken = connection.launchToken;
  }

  /** Starts polling the authenticated read-only Tool UI event snapshot. */
  connect(): void {
    if (this.#closed || this.#polling || this.#pollTimer !== null) {
      return;
    }
    void this.#poll();
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
    this.#deliveredEvents.clear();
    this.#pollAbortController?.abort();
    this.#pollAbortController = null;
    if (this.#pollTimer !== null) {
      clearTimeout(this.#pollTimer);
      this.#pollTimer = null;
    }
    this.connect();
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
   * Publishes a prepared document behind a runtime-scoped frame URL.
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
    if (this.#pollTimer !== null) {
      clearTimeout(this.#pollTimer);
      this.#pollTimer = null;
    }
    this.#pollAbortController?.abort();
    this.#pollAbortController = null;
    this.#deliveredEvents.clear();
    this.#listeners.clear();
  }

  async #poll(): Promise<void> {
    if (this.#closed || this.#polling) {
      return;
    }
    const sessionId = this.#sessionId;
    if (sessionId === null) {
      return;
    }

    this.#polling = true;
    const controller = new AbortController();
    this.#pollAbortController = controller;
    let delay = TOOL_UI_POLL_INTERVAL_MS;
    try {
      const url = new URL("/app/api/tool-ui/events", this.#origin);
      url.searchParams.set("session_id", sessionId);
      const response = await fetch(url, {
        headers: {
          "X-XTalk-App-Token": this.#launchToken,
        },
        cache: "no-store",
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`Could not read tool UI events (${response.status}).`);
      }
      const payload: unknown = await response.json();
      for (const toolEvent of parseToolUIEvents(payload)) {
        const eventKey = `${toolEvent.type}:${toolEvent.callId}:${toolEvent.sequence}`;
        if (this.#deliveredEvents.has(eventKey)) {
          continue;
        }
        this.#deliveredEvents.add(eventKey);
        if (this.#deliveredEvents.size > MAX_DELIVERED_TOOL_UI_EVENTS) {
          const oldest = this.#deliveredEvents.values().next().value;
          if (oldest !== undefined) {
            this.#deliveredEvents.delete(oldest);
          }
        }
        for (const listener of this.#listeners) {
          listener(toolEvent);
        }
      }
    } catch {
      if (!controller.signal.aborted) {
        delay = TOOL_UI_RETRY_INTERVAL_MS;
      }
    } finally {
      if (this.#pollAbortController === controller) {
        this.#pollAbortController = null;
      }
      this.#polling = false;
    }

    if (!this.#closed && this.#sessionId !== null) {
      this.#pollTimer = setTimeout(() => {
        this.#pollTimer = null;
        void this.#poll();
      }, delay);
    }
  }
}

function parseToolUIEvents(payload: unknown): ToolUIEvent[] {
  if (!isRecord(payload) || !Array.isArray(payload.events)) {
    throw new Error("Tool UI event snapshot must contain an events array.");
  }
  return payload.events.map(parseToolUIEvent);
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
    typeof payload.emittedAt === "string" &&
    (payload.outcome === undefined ||
      payload.outcome === "running" ||
      payload.outcome === "complete" ||
      payload.outcome === "cancelled")
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
