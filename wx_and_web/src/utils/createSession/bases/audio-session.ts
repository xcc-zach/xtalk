export { BaseInputAudioSession, BaseOutputAudioSession };
export type { InputAudioSessionConfig, OutputAudioSessionConfig };

interface InputAudioSessionConfig {
    sampleRate: number;
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
abstract class BaseOutputAudioSession {
    abstract open(): Promise<void>;
    abstract close(): Promise<void>;
    abstract pause(): Promise<void>;
    abstract resume(): Promise<void>;
    abstract stop(): Promise<void>;
    abstract pushAudioChunk(pcmChunkInt16: ArrayBuffer): Promise<void>;

    onChunkStarted(callback: (pcmChunkInt16: ArrayBuffer) => void | Promise<void>) {
        this.chunkStartedCallback = callback;
    }
    onChunkPlayed(callback: (pcmChunkInt16: ArrayBuffer) => void | Promise<void>) {
        this.chunkPlayedCallback = callback;
    }
    onAllChunksPlayed(callback: () => void | Promise<void>) {
        this.allChunksPlayedCallback = callback;
    }
    protected chunkStartedCallback(_pcmChunkInt16: ArrayBuffer): void | Promise<void> {

    }
    protected chunkPlayedCallback(_pcmChunkInt16: ArrayBuffer): void | Promise<void> {

    }
    protected allChunksPlayedCallback(): void | Promise<void> {

    }
}