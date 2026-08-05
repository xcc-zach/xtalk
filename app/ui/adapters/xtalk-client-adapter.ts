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
let desktopAudioConstraintsInstalled = false;

/**
 * Maps SDK messages into desktop-owned display entries.
 *
 * The public SDK ``messages`` array is the authoritative conversation state:
 * each entry corresponds to one visible turn and streaming assistant updates
 * mutate that same entry in place. Desktop renders the entries verbatim, one
 * row per SDK message, matching the sample application's rendering semantics.
 *
 * @param sessionId Active SDK session identifier.
 * @param messages Public SDK messages in conversation order.
 * @returns Desktop messages with one entry per SDK message.
 */
function mapDesktopMessages(
  sessionId: string | null,
  messages: ReadonlyArray<{
    role: DesktopMessage["role"];
    content: string;
    final?: boolean;
  }>,
): DesktopMessage[] {
  return messages.map((message, index) => ({
    id: `${sessionId ?? "pending"}:${index}`,
    role: message.role,
    content: message.content,
    final: message.final === true,
  }));
}

/** Force browser-native microphone echo and noise suppression in the WebView. */
function installDesktopAudioConstraints(): void {
  if (desktopAudioConstraintsInstalled) {
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
        echoCancellation: true,
        noiseSuppression: true,
      },
    });
  };
  desktopAudioConstraintsInstalled = true;
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
    installDesktopAudioConstraints();
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
      this.#snapshot = {
        connectionState: state.connectionState,
        streamState: state.streamState,
        sessionId: state.sessionId,
        userId: state.user?.id ?? null,
        muted: this.#session.muted,
        messages: mapDesktopMessages(
          state.sessionId,
          state.messages,
        ),
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
   * Switching replaces the in-memory conversation and leaves the realtime
   * connection closed. The desktop application starts a conversation only
   * when the user presses the chat-bar start button, so the session is
   * explicitly closed after the switch to reflect that state.
   *
   * @param sessionId Persisted session identifier, or `null` for a new chat.
   */
  async switchSession(sessionId: string | null): Promise<void> {
    await this.#session.switchSession(sessionId);
    await this.#session.close();
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
