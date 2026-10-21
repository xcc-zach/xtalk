import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import { Conversation } from "../conversation";

export type {
    AudioChunkCallback,
    Session,
    SessionConfig,
    SessionState,
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
 * Configuration overrides used when creating a session.
 */
interface SessionConfig {
    /** Request binary PCM output; defaults to compatible Base64 JSON. */
    outputAudioEncoding?: "base64" | "binary_pcm";
    /**
     * Input audio session overrides such as capture sample rate.
     */
    inputConfig?: Partial<InputAudioSessionConfig>;
    /**
     * Output audio session overrides such as playback sample rate.
     */
    outputConfig?: Partial<OutputAudioSessionConfig>;
    /** Optional Realtime API key; business credentials belong to the URL owner. */
    apiKey?: string;
}

/**
 * Public session controller exposed by the frontend entrypoint.
 */
interface Session {
    /**
     * Opens the Realtime connection and its audio resources.
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
     * The promise resolves after the matching `conversation.item.done` receipt.
     * Validation failures and disconnection reject the promise.
     *
     * @param text User-authored text for the next turn.
     */
    sendText(text: string): Promise<void>;
    /** Injects external text history without generating a reply. */
    restoreHistory(messages: Array<{role: "user" | "assistant" | "system"; content: string}>): Promise<void>;
}
