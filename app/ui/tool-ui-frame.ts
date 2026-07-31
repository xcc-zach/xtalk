import type {
  ToolUIEmitEvent,
  ToolUIStatusEvent,
} from "./adapters/tool-ui-adapter";

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
  readonly #channelId: string;
  readonly #mode: ToolUIFrameMode;
  readonly #onCapabilities: (capabilities: ToolUICapabilities) => void;
  #loaded = false;
  #pendingStatus: ToolUIStatusEvent | null = null;
  #pendingEmit: ToolUIEmitEvent | null = null;

  /**
   * Creates a sandboxed frame from self-contained untrusted HTML.
   *
   * @param frameUrl One-time loopback URL serving the prepared document.
   * @param channelId Random channel identifier embedded in that document.
   * @param mode Live or immutable history mode.
   * @param title Accessible iframe title.
   * @param onCapabilities Called when the UI registers display hooks.
   */
  constructor(
    frameUrl: string,
    channelId: string,
    mode: ToolUIFrameMode,
    title: string,
    onCapabilities: (capabilities: ToolUICapabilities) => void,
  ) {
    this.#channelId = channelId;
    this.#mode = mode;
    this.#onCapabilities = onCapabilities;
    this.element = document.createElement("iframe");
    this.element.className = "tool-ui-frame";
    this.element.title = title;
    this.element.sandbox.add("allow-scripts");
    this.element.referrerPolicy = "no-referrer";
    this.element.src = frameUrl;
    window.addEventListener("message", this.#handleMessage);
    this.element.addEventListener("load", () => {
      this.#loaded = true;
      this.#flush();
    });
  }

  /**
   * Delivers the latest status to a live UI.
   *
   * @param event Validated status event.
   */
  status(event: ToolUIStatusEvent): void {
    this.#pendingStatus = event;
    this.#flush();
  }

  /**
   * Delivers one immutable emit to a history UI.
   *
   * @param event Validated emit event.
   */
  emit(event: ToolUIEmitEvent): void {
    this.#pendingEmit = event;
    this.#flush();
  }

  /** Removes global listeners owned by this frame. */
  destroy(): void {
    window.removeEventListener("message", this.#handleMessage);
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
      this.#onCapabilities({
        status: event.data.status === true,
        emit: event.data.emit === true,
      });
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
      this.#pendingStatus = null;
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
      this.#pendingEmit = null;
    }
  }

  #applyHeight(requestedHeight: number): void {
    const minimum = this.#mode === "live" ? 120 : 80;
    const configuredMaximum = this.#mode === "live" ? 420 : 600;
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
 * @returns Complete HTML served from the restricted loopback frame route.
 */
export function createToolUIFrameDocument(
  source: string,
  channelId: string,
  mode: ToolUIFrameMode,
): string {
  const escapedChannelId = JSON.stringify(channelId);
  const escapedMode = JSON.stringify(mode);
  const bridge = `
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; base-uri 'none'; connect-src 'none'; font-src 'none'; form-action 'none'; frame-src 'none'; img-src data: blob:; media-src 'none'; object-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
<script>
(() => {
  "use strict";
  const channelId = ${escapedChannelId};
  const mode = ${escapedMode};
  const statusListeners = new Set();
  const emitListeners = new Set();
  const send = (payload) => parent.postMessage({ channelId, ...payload }, "*");
  const reportCapabilities = () => send({
    type: "tool_ui.capabilities",
    status: statusListeners.size > 0,
    emit: emitListeners.size > 0,
  });
  const api = Object.freeze({
    context: Object.freeze({ mode }),
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
  });
  addEventListener("load", () => {
    reportCapabilities();
    const reportHeight = () => api.reportHeight(
      Math.max(document.documentElement.scrollHeight, document.body?.scrollHeight || 0),
    );
    reportHeight();
    new ResizeObserver(reportHeight).observe(document.documentElement);
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
