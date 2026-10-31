export { BaseInputAudioSession, BaseOutputAudioSession };
export type { InputAudioSessionConfig, OutputAudioSessionConfig, OutputAudioStopResult, AudioStreamProgress };

/** Cumulative playback counters for one continuous PCM stream. */
interface AudioStreamProgress {
    stream_id: string;
    played_samples: number;
}

interface InputAudioSessionConfig {
    /**
     * Target input sample rate forwarded to the session runtime.
     */
    sampleRate: number;
    /**
     * Selects the input source implementation.
     */
    mode?: "microphone" | "web_bridge";
    /**
     * Participant identifier used by bridge-backed input implementations.
     */
    participantId?: string;
    /**
     * Bridge instance consumed by bridge-backed input implementations.
     */
    bridge?: unknown;
    /**
     * Whether the source should auto-broadcast frontend VAD when publishing output
     * back into a shared bridge stream.
     */
    autoEmitVad?: boolean;
    /**
     * VAD redemption window used by bridge-backed input sources that emit frontend
     * VAD automatically.
     */
    vadRedemptionMs?: number;
    [key: string]: any;
}
abstract class BaseInputAudioSession {
    abstract open(): Promise<void>;
    abstract close(): Promise<void>;
    abstract get muted(): boolean;
    abstract set muted(value: boolean);

    onFrame(callback: (pcmChunkInt16: ArrayBuffer) => void | Promise<void>) {
        this.frameCallback = callback;
    }
    onSpeechStart(callback: () => void | Promise<void>) {
        this.speechStartCallback = callback;
    }
    onSpeechEnd(callback: () => void | Promise<void>) {
        this.speechEndCallback = callback;
    }
    protected frameCallback(_pcmChunkInt16: ArrayBuffer): void | Promise<void> {

    };
    /**
     * Used only when VAD enabled
     */
    protected speechStartCallback(): void | Promise<void> {

    };
    /**
     * Used only when VAD enabled
     */
    protected speechEndCallback(): void | Promise<void> {

    };
}

interface OutputAudioSessionConfig {
    sampleRate: number;
    [key: string]: any;
}
interface OutputAudioStopResult {
    /** Playback time not yet confirmed by completed-chunk callbacks. */
    unconfirmedPlayedMs: number;
}
abstract class BaseOutputAudioSession {
    /** Whether this platform implements continuous PCM playback. */
    get supportsContinuousAudio(): boolean { return false; }
    /** Begin one negotiated continuous stream. */
    startAudioStream(_streamId: string): void { throw new Error("Continuous audio unavailable on this platform"); }
    /** Admit a continuous PCM frame, independent of TTS response boundaries. */
    async pushStreamAudioChunk(_pcm: ArrayBuffer): Promise<void> { throw new Error("Continuous audio unavailable"); }
    /** Stop continuous output and report samples actually played. */
    endAudioStream(): void | Promise<void> { }
    /** Register a continuous playback counter observer. */
    onAudioStreamProgress(callback: (progress: AudioStreamProgress) => void): void { this.streamProgressCallback = callback; }
    protected streamProgressCallback(_progress: AudioStreamProgress): void { }
    abstract open(): Promise<void>;
    abstract close(): Promise<void>;
    abstract pause(): Promise<void>;
    abstract resume(): Promise<void>;
    /** Open local playback state for one server response. */
    abstract startTTS(responseId: string): void;
    /** Stop only the identified response, or all playback during shutdown. */
    abstract stop(responseId?: string): Promise<OutputAudioStopResult>;
    abstract pushAudioChunk(pcmChunkInt16: ArrayBuffer): Promise<void>;
    /**
     * Marks that the server has finished producing TTS audio for the current turn.
     * Implementations should combine this signal with local playback state before
     * reporting that playback is fully finished.
     */
    abstract notifyTTSFinished(responseId: string): Promise<void> | void;

    onChunkStarted(callback: (responseId: string, pcmChunkInt16: ArrayBuffer) => void | Promise<void>) {
        this.chunkStartedCallback = callback;
    }
    onChunkPlayed(callback: (responseId: string, pcmChunkInt16: ArrayBuffer) => void | Promise<void>) {
        this.chunkPlayedCallback = callback;
    }
    onAllChunksPlayed(callback: (responseId: string) => void | Promise<void>) {
        this.allChunksPlayedCallback = callback;
    }
    protected chunkStartedCallback(_responseId: string, _pcmChunkInt16: ArrayBuffer): void | Promise<void> {

    }
    protected chunkPlayedCallback(_responseId: string, _pcmChunkInt16: ArrayBuffer): void | Promise<void> {

    }
    protected allChunksPlayedCallback(_responseId: string): void | Promise<void> {

    }
}
