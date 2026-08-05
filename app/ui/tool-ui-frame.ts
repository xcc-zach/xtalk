import type {
  ToolUIEmitEvent,
  ToolUIStatusEvent,
} from "./adapters/tool-ui-adapter";
import type { SupportedLanguage } from "./i18n";

/** Tool UI frame rendering mode. */
export type ToolUIFrameMode = "live" | "history";

/** Hooks registered by one untrusted tool UI entrypoint. */
export interface ToolUICapabilities {
  status: boolean;
  emit: boolean;
}

/**
 * Owns one sandboxed, display-only user-defined tool UI.
 */
export class ToolUIFrame {
  readonly element: HTMLIFrameElement;
  readonly #frameUrl: string;
  readonly #channelId: string;
  readonly #mode: ToolUIFrameMode;
  readonly #onCapabilities: (capabilities: ToolUICapabilities) => void;
  #mounted = false;
  #loaded = false;
  #pendingStatus: ToolUIStatusEvent | null = null;
  #pendingEmit: ToolUIEmitEvent | null = null;
  #lastStatus: ToolUIStatusEvent | null = null;
  #lastEmit: ToolUIEmitEvent | null = null;
  #retryTimer: number | null = null;
  #retryAttempts = 0;
  readonly #maxHeight: number | undefined;

  /**
   * Creates a sandboxed frame from self-contained untrusted HTML.
   *
   * @param frameUrl One-time loopback URL serving the prepared document.
   * @param channelId Random channel identifier embedded in that document.
   * @param mode Live or immutable history mode.
   * @param title Accessible iframe title.
   * @param onCapabilities Called when the UI registers display hooks.
   * @param maxHeight Optional per-frame height cap replacing the mode default.
   */
  constructor(
    frameUrl: string,
    channelId: string,
    mode: ToolUIFrameMode,
    title: string,
    onCapabilities: (capabilities: ToolUICapabilities) => void,
    maxHeight: number | undefined = undefined,
  ) {
    this.#frameUrl = frameUrl;
    this.#channelId = channelId;
    this.#mode = mode;
    this.#onCapabilities = onCapabilities;
    this.#maxHeight = maxHeight;
    this.element = document.createElement("iframe");
    this.element.className = "tool-ui-frame";
    this.element.title = title;
    this.element.sandbox.add("allow-scripts");
    this.element.referrerPolicy = "no-referrer";
    window.addEventListener("message", this.#handleMessage);
  }

  /** Loads the runtime-scoped frame URL after the iframe enters the document. */
  mount(): void {
    if (this.#mounted || !this.element.isConnected) {
      return;
    }
    this.#mounted = true;
    this.element.src = this.#frameUrl;
  }

  /**
   * Delivers the latest status to a live UI.
   *
   * @param event Validated status event.
   */
  status(event: ToolUIStatusEvent): void {
    this.#pendingStatus = event;
    this.#lastStatus = event;
    this.#retryAttempts = 0;
    this.#flush();
  }

  /**
   * Delivers one immutable emit to a history UI.
   *
   * @param event Validated emit event.
   * Tools that declare structured content can read it from
   * `event.payload`; the raw JSON text remains available in `event.message`.
   */
  emit(event: ToolUIEmitEvent): void {
    this.#pendingEmit = event;
    this.#lastEmit = event;
    this.#retryAttempts = 0;
    this.#flush();
  }

  /** Removes global listeners owned by this frame. */
  destroy(): void {
    window.removeEventListener("message", this.#handleMessage);
    if (this.#retryTimer !== null) {
      window.clearTimeout(this.#retryTimer);
      this.#retryTimer = null;
    }
  }

  readonly #handleMessage = (event: MessageEvent<unknown>): void => {
    if (
      event.source !== this.element.contentWindow ||
      !isRecord(event.data) ||
      event.data.channelId !== this.#channelId
    ) {
      return;
    }
    if (event.data.type === "tool_ui.capabilities") {
      this.#loaded = true;
      this.#onCapabilities({
        status: event.data.status === true,
        emit: event.data.emit === true,
      });
      this.#flush();
      return;
    }
    if (event.data.type === "tool_ui.received") {
      const pending =
        event.data.eventType === "tool_ui.status"
          ? this.#pendingStatus
          : event.data.eventType === "tool_ui.emit"
            ? this.#pendingEmit
            : null;
      if (
        pending !== null &&
        event.data.callId === pending.callId &&
        event.data.sequence === pending.sequence
      ) {
        if (event.data.eventType === "tool_ui.status") {
          this.#pendingStatus = null;
        } else {
          this.#pendingEmit = null;
        }
      }
      if (this.#pendingStatus === null && this.#pendingEmit === null) {
        this.#retryAttempts = 0;
        if (this.#retryTimer !== null) {
          window.clearTimeout(this.#retryTimer);
          this.#retryTimer = null;
        }
      }
      return;
    }
    if (
      event.data.type === "tool_ui.resize" &&
      typeof event.data.height === "number" &&
      Number.isFinite(event.data.height)
    ) {
      this.#applyHeight(event.data.height);
    }
  };

  #flush(): void {
    if (!this.#loaded || this.element.contentWindow === null) {
      return;
    }
    if (this.#pendingStatus !== null) {
      this.element.contentWindow.postMessage(
        {
          channelId: this.#channelId,
          type: "tool_ui.status",
          event: this.#pendingStatus,
        },
        "*",
      );
    } else if (this.#lastStatus !== null) {
      this.element.contentWindow.postMessage(
        {
          channelId: this.#channelId,
          type: "tool_ui.status",
          event: this.#lastStatus,
        },
        "*",
      );
    }
    if (this.#pendingEmit !== null) {
      this.element.contentWindow.postMessage(
        {
          channelId: this.#channelId,
          type: "tool_ui.emit",
          event: this.#pendingEmit,
        },
        "*",
      );
    } else if (this.#lastEmit !== null) {
      this.element.contentWindow.postMessage(
        {
          channelId: this.#channelId,
          type: "tool_ui.emit",
          event: this.#lastEmit,
        },
        "*",
      );
    }
    if (
      this.#retryTimer === null &&
      this.#retryAttempts < 20 &&
      (this.#pendingStatus !== null || this.#pendingEmit !== null)
    ) {
      this.#retryAttempts += 1;
      this.#retryTimer = window.setTimeout(() => {
        this.#retryTimer = null;
        this.#flush();
      }, 100);
    }
  }

  #applyHeight(requestedHeight: number): void {
    const minimum = this.#mode === "live" ? 120 : 80;
    const configuredMaximum =
      this.#maxHeight ?? (this.#mode === "live" ? 420 : 600);
    const maximum = Math.min(
      configuredMaximum,
      Math.max(minimum, window.innerHeight * 0.6),
    );
    const height = Math.round(
      Math.max(minimum, Math.min(requestedHeight, maximum)),
    );
    this.element.style.height = `${height}px`;
  }
}

/**
 * Injects the read-only bridge into one self-contained tool UI document.
 *
 * @param source Self-contained installed tool HTML.
 * @param channelId Random per-frame message channel identifier.
 * @param mode Live or immutable history mode.
 * @param language Resolved language selected by the desktop application.
 * @returns Complete HTML served from the restricted loopback frame route.
 */
export function createToolUIFrameDocument(
  source: string,
  channelId: string,
  mode: ToolUIFrameMode,
  language: SupportedLanguage,
): string {
  const escapedChannelId = JSON.stringify(channelId);
  const escapedMode = JSON.stringify(mode);
  const escapedLanguage = JSON.stringify(language);
  const bridge = `
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; base-uri 'none'; connect-src 'none'; font-src 'none'; form-action 'none'; frame-src 'none'; img-src data: blob:; media-src 'none'; object-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<script>
(() => {
  "use strict";
  const channelId = ${escapedChannelId};
  const mode = ${escapedMode};
  const language = ${escapedLanguage};
  const statusListeners = new Set();
  const emitListeners = new Set();
  let capabilitiesReady = false;
  let capabilitiesQueued = false;
  const send = (payload) => parent.postMessage({ channelId, ...payload }, "*");
  const reportCapabilities = () => {
    if (!capabilitiesReady || capabilitiesQueued) return;
    capabilitiesQueued = true;
    queueMicrotask(() => {
      capabilitiesQueued = false;
      send({
        type: "tool_ui.capabilities",
        status: statusListeners.size > 0,
        emit: emitListeners.size > 0,
      });
    });
  };
  const api = Object.freeze({
    context: Object.freeze({ mode, language }),
    status(callback) {
      if (typeof callback !== "function") throw new TypeError("status callback must be a function");
      statusListeners.add(callback);
      reportCapabilities();
      return () => {
        statusListeners.delete(callback);
        reportCapabilities();
      };
    },
    emit(callback) {
      if (typeof callback !== "function") throw new TypeError("emit callback must be a function");
      emitListeners.add(callback);
      reportCapabilities();
      return () => {
        emitListeners.delete(callback);
        reportCapabilities();
      };
    },
    reportHeight(height) {
      if (Number.isFinite(height)) send({ type: "tool_ui.resize", height });
    },
  });
  Object.defineProperty(window, "xtalkToolUI", {
    value: api,
    configurable: false,
    writable: false,
  });
  document.documentElement.lang = language;
  addEventListener("click", (event) => event.preventDefault(), true);
  addEventListener("submit", (event) => event.preventDefault(), true);
  addEventListener("message", (message) => {
    const payload = message.data;
    if (!payload || payload.channelId !== channelId) return;
    const listeners =
      payload.type === "tool_ui.status"
        ? statusListeners
        : payload.type === "tool_ui.emit"
          ? emitListeners
          : null;
    if (!listeners) return;
    for (const listener of listeners) {
      try { listener(payload.event); } catch (_) {}
    }
    send({
      type: "tool_ui.received",
      eventType: payload.type,
      callId: payload.event && payload.event.callId,
      sequence: payload.event && payload.event.sequence,
    });
  });
  addEventListener("load", () => {
    capabilitiesReady = true;
    reportCapabilities();
    const measuredElement = document.body || document.documentElement;
    const reportHeight = () => api.reportHeight(
      Math.ceil(measuredElement.getBoundingClientRect().height),
    );
    reportHeight();
    new ResizeObserver(reportHeight).observe(measuredElement);
  });
})();
</script>`;
  const head = /<head(?:\s[^>]*)?>/iu;
  if (head.test(source)) {
    return source.replace(head, (match) => `${match}${bridge}`);
  }
  return `<!doctype html><html><head>${bridge}</head><body>${source}</body></html>`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
