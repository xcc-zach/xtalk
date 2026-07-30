import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import type { SessionServiceURLConfig } from "../bases/http";
import { Conversation } from "../conversation";

export type {
    AudioChunkCallback,
    Session,
    SessionConfig,
    SessionDetail,
    SessionState,
    SessionSummary,
};

/**
 * Snapshot of the current session state exposed by the conversation store.
 */
type SessionState = Conversation["state"];

/**
 * Receives a PCM audio chunk alongside its sample rate.
 */
type AudioChunkCallback = (pcmChunkInt16: ArrayBuffer, sampleRate: number) => void;

/**
 * Summary information for a stored session returned by the sessions endpoint.
 */
type SessionSummary = {
    /**
     * Backend identifier of the session.
     */
    session_id: string;
    /**
     * Human-readable title generated or assigned for the session.
     */
    title: string | null;
};

/**
 * Detailed session payload including historical messages.
 */
type SessionDetail = {
    /**
     * Backend identifier of the session.
     */
    session_id: string;
    /**
     * Human-readable title generated or assigned for the session.
     */
    title: string | null;
    /**
     * Ordered messages stored in the session.
     */
    messages: Array<{
        /**
         * Role of the message author.
         */
        role: "user" | "assistant" | "info";
        /**
         * Text content associated with the message.
         */
        content: string;
    }>;
};

/**
 * Configuration overrides used when creating a session.
 */
interface SessionConfig {
    /**
     * Input audio session overrides such as capture sample rate.
     */
    inputConfig?: Partial<InputAudioSessionConfig>;
    /**
     * Output audio session overrides such as playback sample rate.
     */
    outputConfig?: Partial<OutputAudioSessionConfig>;
    /**
     * Optional overrides for auxiliary HTTP service endpoints.
     */
    serviceURLs?: SessionServiceURLConfig;
}

/**
 * Public session controller exposed by the frontend entrypoint.
 */
interface Session {
    /**
     * Opens the session runtime and performs authentication if needed.
     */
    open(): Promise<void>;
    /**
     * Closes the active runtime connection and audio resources.
     */
    close(): Promise<void>;
    /**
     * Registers a callback that runs whenever the conversation state changes.
     *
     * @param callback State change listener.
     */
    onStateChange(callback: (state: Conversation["state"]) => void): void;
    /**
     * Current conversation state snapshot.
     */
    readonly state: Conversation["state"];
    /**
     * Registers a callback for microphone input PCM chunks.
     *
     * @param callback Input audio listener.
     */
    onInputAudioChunk(callback: AudioChunkCallback): void;
    /**
     * Registers a callback for speaker output PCM chunks.
     *
     * @param callback Output audio listener.
     */
    onOutputAudioChunk(callback: AudioChunkCallback): void;
    /**
     * Registers a callback for merged full-duplex PCM chunks.
     *
     * @param callback Full audio listener.
     */
    onFullAudioChunk(callback: AudioChunkCallback): void;
    /**
     * Whether microphone capture is currently muted.
     */
    muted: boolean;
    /**
     * Requests a voice change for subsequent assistant synthesis.
     *
     * @param voiceName Target voice identifier.
     */
    changeVoice(voiceName: string): Promise<void>;
    /**
     * Submits a finalized text turn through the connected realtime session.
     *
     * The promise resolves after a `finish_asr` action echoes the normalized
     * text with `origin` set to `text`.
     *
     * @param text User-authored text for the next turn.
     */
    sendText(text: string): Promise<void>;
    /**
     * Uploads a file into the current session context.
     *
     * @param file File blob to upload.
     * @param endpoint Optional upload endpoint override.
     */
    uploadFile(file: Blob, endpoint?: string | URL): Promise<void>;
    /**
     * Fetches available persisted sessions for the current user.
     */
    getSessions(): Promise<SessionSummary[]>;
    /**
     * Switches the active conversation to a persisted session or starts a new one.
     *
     * @param sessionId Target session identifier, or `null` to start a new session.
     */
    switchSession(sessionId: string | null): Promise<void>;
}
