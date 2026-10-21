import { createSession, type Session, type SessionConfig } from "xtalk-client";

/** Application-owned endpoints for authentication and stored conversations. */
export interface BusinessEndpoints {
  websocket: URL;
  login: URL;
  sessions: URL;
  sessionDetail(id: string): URL;
}

/** Application identity combined with pure conversation state. */
export type BusinessState = Session["state"] & {user: {id: string} | null};

/** Pure Realtime session composed with desktop-owned history operations. */
export interface BusinessSession extends Session {
  readonly state: BusinessState;
  onStateChange(callback: (state: BusinessState) => void): void;
  getSessions(): Promise<Array<{session_id: string; title: string | null}>>;
  switchSession(id: string | null): Promise<void>;
}

/** Compose application HTTP authentication with a new temporary Realtime connection. */
export function createBusinessSession(endpoints: BusinessEndpoints, config: SessionConfig): BusinessSession {
  let session = createSession(endpoints.websocket, config);
  let token: string | null = null;
  let user: BusinessState["user"] = null;
  let conversationId: string | null = null;
  let history: Array<{role: "user" | "assistant"; content: string}> = [];
  let muted = false;
  const listeners = new Set<(state: BusinessState) => void>();
  const inputCallbacks = new Set<Parameters<Session["onInputAudioChunk"]>[0]>();
  const outputCallbacks = new Set<Parameters<Session["onOutputAudioChunk"]>[0]>();
  const fullCallbacks = new Set<Parameters<Session["onFullAudioChunk"]>[0]>();
  async function request(url: URL, method = "GET") {
    const response = await fetch(url, {method, headers: token ? {Authorization: `Bearer ${token}`} : {}});
    if (!response.ok) throw new Error(`Conversation request failed: ${response.status}`);
    return response.json();
  }
  async function login() {
    if (token) return;
    const result = await request(endpoints.login, "POST");
    token = result.access_token;
    user = result.user;
  }
  function state(): BusinessState {
    return {...session.state, sessionId: conversationId, user};
  }
  return {
    async open() {
      await session.close();
      await login();
      if (!conversationId) conversationId = (await request(endpoints.sessions, "POST")).session_id;
      const url = new URL(endpoints.websocket);
      url.searchParams.set("access_token", token!);
      url.searchParams.set("conversation_id", conversationId!);
      session = createSession(url, config);
      session.muted = muted;
      session.onStateChange(() => { for (const callback of listeners) callback(state()); });
      for (const callback of inputCallbacks) session.onInputAudioChunk(callback);
      for (const callback of outputCallbacks) session.onOutputAudioChunk(callback);
      for (const callback of fullCallbacks) session.onFullAudioChunk(callback);
      if (history.length) await session.restoreHistory(history);
      await session.open();
    },
    close: () => session.close(),
    get state() { return state(); },
    get muted() { return muted; },
    set muted(value) { muted = value; session.muted = value; },
    onStateChange(callback) { listeners.add(callback); callback(state()); },
    onInputAudioChunk(callback) { inputCallbacks.add(callback); session.onInputAudioChunk(callback); },
    onOutputAudioChunk(callback) { outputCallbacks.add(callback); session.onOutputAudioChunk(callback); },
    onFullAudioChunk(callback) { fullCallbacks.add(callback); session.onFullAudioChunk(callback); },
    changeVoice: voice => session.changeVoice(voice),
    sendText: text => session.sendText(text),
    restoreHistory: messages => session.restoreHistory(messages),
    async getSessions() { await login(); return (await request(endpoints.sessions)).sessions; },
    async switchSession(id) {
      await session.close();
      await login();
      conversationId = id;
      history = id ? (await request(endpoints.sessionDetail(id))).messages : [];
      session = createSession(endpoints.websocket, config);
      const snapshot = {...state(), messages: history};
      for (const callback of listeners) callback(snapshot);
    },
  };
}
