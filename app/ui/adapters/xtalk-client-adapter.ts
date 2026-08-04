import {
  createSession,
  type Session,
  type SessionConfig,
} from "xtalk-client";

import type { NativeBackendConnection } from "./native-capabilities";

/**
 * Connection states exposed to the desktop UI store.
 */
export type DesktopConnectionState =
  | "connected"
  | "reconnecting"
  | "disconnected";

/**
 * Audio pipeline states exposed to the desktop UI store.
 */
export type DesktopStreamState =
  | "idle"
  | "listening"
  | "processing"
  | "speaking";

/**
 * Safe message representation owned by the desktop application.
 */
export interface DesktopMessage {
  /** Stable UI key scoped to the current adapter snapshot. */
  id: string;
  /** Source role used to style and label the message. */
  role: "user" | "assistant" | "info";
  /** Plain-text message body. */
  content: string;
  /** Whether the backend marked the message as complete. */
  final: boolean;
}

/**
 * Minimal XTalk state consumed by the Phase 0 desktop UI.
 */
export interface DesktopSessionSnapshot {
  /** Current WebSocket connection state. */
  connectionState: DesktopConnectionState;
  /** Current microphone and playback pipeline state. */
  streamState: DesktopStreamState;
  /** Active persisted conversation identifier, when one exists. */
  sessionId: string | null;
  /** Authenticated XTalk user identifier, when login has completed. */
  userId: string | null;
  /** Whether microphone frames are currently suppressed. */
  muted: boolean;
  /** Plain-text conversation messages in display order. */
  messages: DesktopMessage[];
}

/**
 * Persisted conversation metadata displayed by the desktop session sidebar.
 */
export interface DesktopSessionSummary {
  /** Backend identifier stored in the application-data conversation database. */
  id: string;
  /** Human-readable title derived from the conversation, when available. */
  title: string | null;
}

/**
 * Redacted endpoint information suitable for the diagnostics UI.
 */
export interface XtalkEndpointDiagnostics {
  /** Validated sidecar HTTP origin. */
  origin: string;
  /** Token-free WebSocket URL used to create the SDK session. */
  websocketURL: string;
  /** Whether all HTTP service endpoints carry the app token. */
  httpEndpointsAuthenticated: boolean;
}

/**
 * Listener invoked whenever the desktop session snapshot changes.
 */
export type DesktopSessionListener = (snapshot: DesktopSessionSnapshot) => void;

const APP_TOKEN_QUERY_PARAMETER = "app_token";
let desktopNoiseSuppressionInstalled = false;

/**
 * Finalize assistant messages that can no longer be the active response.
 *
 * An interrupted response may miss its normal finish event while the next
 * turn has already started. The desktop timeline must still expose at most
 * one streaming assistant message: the latest assistant after the latest user
 * message. This keeps stale responses from retaining a blinking cursor.
 *
 * @param messages Desktop messages in conversation order.
 * @returns A normalized copy suitable for rendering in the desktop timeline.
 */
function finalizeSupersededAssistantMessages(
  messages: DesktopMessage[],
): DesktopMessage[] {
  let activeAssistantIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "user") {
      break;
    }
    if (message?.role === "assistant") {
      if (!message.final) {
        activeAssistantIndex = index;
      }
      break;
    }
  }

  return messages.map((message, index) => {
    if (
      message.role === "assistant" &&
      !message.final &&
      index !== activeAssistantIndex
    ) {
      return { ...message, final: true };
    }
    return message;
  });
}

/** Force browser-native microphone noise suppression inside the desktop WebView. */
function installDesktopNoiseSuppression(): void {
  if (desktopNoiseSuppressionInstalled) {
    return;
  }
  const mediaDevices = navigator.mediaDevices;
  const originalGetUserMedia = mediaDevices?.getUserMedia?.bind(mediaDevices);
  if (!mediaDevices || !originalGetUserMedia) {
    return;
  }
  mediaDevices.getUserMedia = (constraints?: MediaStreamConstraints) => {
    const requestedAudio = constraints?.audio;
    if (!requestedAudio) {
      return originalGetUserMedia(constraints ?? {});
    }
    const audioConstraints: MediaTrackConstraints = requestedAudio === true
      ? {}
      : requestedAudio;
    return originalGetUserMedia({
      ...constraints,
      audio: {
        ...audioConstraints,
        noiseSuppression: true,
      },
    });
  };
  desktopNoiseSuppressionInstalled = true;
}

/**
 * Adapts the public xtalk-client session API to desktop-owned state and URLs.
 */
export class XtalkClientAdapter {
  readonly #session: Session;
  readonly #diagnostics: XtalkEndpointDiagnostics;
  readonly #listeners = new Set<DesktopSessionListener>();
  #snapshot: DesktopSessionSnapshot;

  /**
   * Creates a desktop session around one validated Tauri bootstrap payload.
   *
   * @param connection Sidecar origin and per-launch credential from Tauri.
   */
  constructor(connection: NativeBackendConnection) {
    installDesktopNoiseSuppression();
    const endpoints = createEndpoints(connection);
    const sessionConfig: SessionConfig = {
      inputConfig: {
        enableVAD: false,
        enableEnhancer: false,
      },
      serviceURLs: {
        login: endpoints.login,
        sessions: endpoints.sessions,
        sessionDetail: (sessionId: string) => endpoints.sessionDetail(sessionId),
        upload: endpoints.upload,
      },
    };

    this.#session = createSession(endpoints.websocket, sessionConfig);
    this.#diagnostics = {
      origin: connection.origin,
      websocketURL: endpoints.websocket.toString(),
      httpEndpointsAuthenticated: true,
    };
    this.#snapshot = {
      connectionState: "disconnected",
      streamState: "idle",
      sessionId: null,
      userId: null,
      muted: false,
      messages: [],
    };

    this.#session.onStateChange((state) => {
      const messages = state.messages.map((message, index) => ({
        id: `${state.sessionId ?? "pending"}:${index}`,
        role: message.role,
        content: message.content,
        final: message.final ?? false,
      }));
      this.#snapshot = {
        connectionState: state.connectionState,
        streamState: state.streamState,
        sessionId: state.sessionId,
        userId: state.user?.id ?? null,
        muted: this.#session.muted,
        messages: finalizeSupersededAssistantMessages(messages),
      };
      this.#notify();
    });
  }

  /**
   * Returns the latest desktop-owned state snapshot.
   */
  get snapshot(): DesktopSessionSnapshot {
    return this.#snapshot;
  }

  /**
   * Returns endpoint metadata with the launch credential omitted.
   */
  get diagnostics(): XtalkEndpointDiagnostics {
    return this.#diagnostics;
  }

  /**
   * Subscribes to session state and immediately emits the current snapshot.
   *
   * @param listener Desktop state listener.
   * @returns A function that removes the listener from this adapter.
   */
  subscribe(listener: DesktopSessionListener): () => void {
    this.#listeners.add(listener);
    listener(this.#snapshot);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  /**
   * Authenticates, opens audio resources, and connects the XTalk WebSocket.
   */
  async connect(): Promise<void> {
    await this.#session.open();
  }

  /**
   * Closes audio resources and the active XTalk WebSocket.
   */
  async disconnect(): Promise<void> {
    await this.#session.close();
  }

  /**
   * Enables or suppresses outgoing microphone frames.
   *
   * @param muted Desired microphone mute state.
   */
  setMuted(muted: boolean): void {
    this.#session.muted = muted;
    this.#snapshot = { ...this.#snapshot, muted };
    this.#notify();
  }

  /**
   * Sends a plain-text user turn through the public realtime session.
   *
   * The SDK waits for the matching server-confirmed `finish_asr` action, so
   * message state, tools, synthesis, and persistence remain server-authoritative.
   *
   * @param text Non-empty user message containing at most 2,048 characters.
   */
  async sendText(text: string): Promise<void> {
    await this.#session.sendText(text);
  }

  /**
   * Lists conversations persisted by the sidecar in the application data directory.
   *
   * @returns Persisted conversations in backend activity order.
   */
  async getSessions(): Promise<DesktopSessionSummary[]> {
    const sessions = await this.#session.getSessions();
    return sessions.map((session) => ({
      id: session.session_id,
      title: session.title,
    }));
  }

  /**
   * Selects a persisted conversation or resets the client for a new one.
   *
   * Switching closes the active realtime connection. The caller can reconnect
   * through {@link connect} when the user is ready to continue the conversation.
   *
   * @param sessionId Persisted session identifier, or `null` for a new chat.
   */
  async switchSession(sessionId: string | null): Promise<void> {
    await this.#session.switchSession(sessionId);
  }

  #notify(): void {
    for (const listener of this.#listeners) {
      listener(this.#snapshot);
    }
  }
}

interface XtalkEndpoints {
  websocket: URL;
  login: URL;
  sessions: URL;
  sessionDetail(sessionId: string): URL;
  upload: URL;
}

function createEndpoints(connection: NativeBackendConnection): XtalkEndpoints {
  const websocket = new URL("/ws", connection.origin);
  websocket.protocol = websocket.protocol === "https:" ? "wss:" : "ws:";

  const authenticatedURL = (path: string): URL => {
    const url = new URL(path, connection.origin);
    url.searchParams.set(APP_TOKEN_QUERY_PARAMETER, connection.launchToken);
    return url;
  };

  return {
    websocket,
    login: authenticatedURL("/api/auth/login"),
    sessions: authenticatedURL("/api/sessions"),
    sessionDetail: (sessionId: string) =>
      authenticatedURL(`/api/sessions/${encodeURIComponent(sessionId)}`),
    upload: authenticatedURL("/api/upload"),
  };
}
