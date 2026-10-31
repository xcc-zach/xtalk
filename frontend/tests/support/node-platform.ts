/**
 * A platform runtime for Node: a real WebSocket plus a scripted microphone
 * and a virtual speaker that tests can drive and observe.
 */

import {
    BaseDeferredTaskScheduler,
    BaseEncoding,
    BaseInputAudioSession,
    BaseOutputAudioSession,
    BaseWebSocket,
} from "../../dist/index.js";
import type {
    BaseWebSocketCloseEvent,
    BaseWebSocketMessageEvent,
    InputAudioSessionConfig,
    OutputAudioSessionConfig,
    OutputAudioStopResult,
    PlatformRuntime,
} from "../../dist/index.js";

const OUTPUT_BYTES_PER_MS = 48; // 24 kHz PCM16

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Wraps the Node global WebSocket. */
export class NodeWebSocket extends BaseWebSocket {
    private readonly socket: WebSocket;

    constructor(url: string | URL, protocols?: string | string[]) {
        super();
        this.socket = new WebSocket(url, protocols);
        this.socket.binaryType = "arraybuffer";
    }

    ready(): boolean {
        return this.socket.readyState === WebSocket.OPEN;
    }

    send(data: string | ArrayBuffer): void {
        this.socket.send(data);
    }

    close(): void {
        this.socket.close();
    }

    addEventListener(type: "open" | "error", listener: () => any): void;
    addEventListener(type: "message", listener: (evt: BaseWebSocketMessageEvent) => any): void;
    addEventListener(type: "close", listener: (evt: BaseWebSocketCloseEvent) => any): void;
    addEventListener(type: "open" | "message" | "close" | "error", listener: (...args: any[]) => any): void {
        if (type === "message") {
            this.socket.addEventListener("message", (event) => {
                listener({ data: event.data as string | ArrayBuffer });
            });
            return;
        }
        if (type === "close") {
            this.socket.addEventListener("close", (event) => {
                listener({ code: event.code, reason: event.reason, wasClean: event.wasClean });
            });
            return;
        }
        this.socket.addEventListener(type, () => listener());
    }
}

/** A microphone whose audio and speech boundaries are pushed by the test. */
export class ScriptedInputAudioSession extends BaseInputAudioSession {
    private mutedState = false;
    private opened = false;
    readonly config: InputAudioSessionConfig;

    constructor(config: InputAudioSessionConfig) {
        super();
        this.config = config;
    }

    async open(): Promise<void> {
        this.opened = true;
    }

    async close(): Promise<void> {
        this.opened = false;
    }

    get isOpen(): boolean {
        return this.opened;
    }

    get muted(): boolean {
        return this.mutedState;
    }

    set muted(value: boolean) {
        this.mutedState = value;
    }

    /** Feed PCM16 at the configured sample rate as consecutive frames. */
    async pushPcm(pcm16: ArrayBuffer, options: { frameMs?: number; realtime?: boolean } = {}): Promise<void> {
        const frameMs = options.frameMs ?? 32;
        const realtime = options.realtime ?? true;
        const frameBytes = Math.round((this.config.sampleRate * frameMs) / 1000) * 2;
        const started = Date.now();
        let index = 0;
        for (let start = 0; start < pcm16.byteLength; start += frameBytes) {
            const frame = pcm16.slice(start, Math.min(start + frameBytes, pcm16.byteLength));
            if (!this.mutedState) {
                await this.frameCallback(frame);
            }
            index += 1;
            if (realtime) {
                await sleep(Math.max(0, started + index * frameMs - Date.now()));
            }
        }
    }

    /** Report a client side speech onset. */
    async speechStart(): Promise<void> {
        if (!this.mutedState) await this.speechStartCallback();
    }

    /** Report a client side speech end. */
    async speechEnd(): Promise<void> {
        if (!this.mutedState) await this.speechEndCallback();
    }
}

interface QueuedChunk {
    responseId: string;
    pcm: ArrayBuffer;
    totalMs: number;
    playedMs: number;
    started: boolean;
}

/** A speaker with a virtual clock advanced by the test. */
export class VirtualOutputAudioSession extends BaseOutputAudioSession {
    /** Whether queued audio is treated as heard as soon as it arrives. */
    autoPlay: boolean;
    private readonly queue: QueuedChunk[] = [];
    private readonly heard = new Map<string, number>();
    private activeResponseId: string | null = null;
    private serverFinished = false;
    private paused = false;
    private opened = false;
    readonly config: OutputAudioSessionConfig;

    constructor(config: OutputAudioSessionConfig, options: { autoPlay?: boolean } = {}) {
        super();
        this.config = config;
        this.autoPlay = options.autoPlay ?? true;
    }

    async open(): Promise<void> {
        this.opened = true;
    }

    async close(): Promise<void> {
        await this.stop();
        this.opened = false;
    }

    get isOpen(): boolean {
        return this.opened;
    }

    async pause(): Promise<void> {
        this.paused = true;
    }

    async resume(): Promise<void> {
        this.paused = false;
        if (this.autoPlay) await this.advance(Number.POSITIVE_INFINITY);
    }

    startTTS(responseId: string): void {
        if (this.activeResponseId && this.activeResponseId !== responseId) {
            throw new Error("Another TTS response is still active");
        }
        this.activeResponseId = responseId;
        this.serverFinished = false;
    }

    async stop(responseId?: string): Promise<OutputAudioStopResult> {
        if (responseId && responseId !== this.activeResponseId) {
            return { unconfirmedPlayedMs: 0 };
        }
        let unconfirmedPlayedMs = 0;
        for (const chunk of this.queue) {
            unconfirmedPlayedMs += chunk.playedMs;
            this.heard.set(chunk.responseId, (this.heard.get(chunk.responseId) ?? 0) + chunk.playedMs);
        }
        this.queue.length = 0;
        this.activeResponseId = null;
        this.serverFinished = false;
        return { unconfirmedPlayedMs };
    }

    async pushAudioChunk(pcm16: ArrayBuffer): Promise<void> {
        if (!this.activeResponseId) {
            throw new Error("Received TTS audio without an active response");
        }
        this.queue.push({
            responseId: this.activeResponseId,
            pcm: pcm16,
            totalMs: pcm16.byteLength / OUTPUT_BYTES_PER_MS,
            playedMs: 0,
            started: false,
        });
        if (this.autoPlay && !this.paused) {
            await this.advance(Number.POSITIVE_INFINITY);
        }
    }

    async notifyTTSFinished(responseId: string): Promise<void> {
        if (responseId !== this.activeResponseId) return;
        this.serverFinished = true;
        await this.maybeFinish();
    }

    /** Response currently owning the playback stream. */
    get activeResponse(): string | null {
        return this.activeResponseId;
    }

    /** Total audio of one response that finished playing. */
    heardMs(responseId: string): number {
        return this.heard.get(responseId) ?? 0;
    }

    /** Audio of the active response received but not yet played. */
    get queuedMs(): number {
        return this.queue.reduce((sum, chunk) => sum + chunk.totalMs - chunk.playedMs, 0);
    }

    /** Advance the virtual clock, completing queued chunks in order. */
    async advance(ms: number): Promise<number> {
        let remaining = ms;
        let consumed = 0;
        while (remaining > 0 && this.queue.length > 0 && !this.paused) {
            const chunk = this.queue[0]!;
            if (!chunk.started) {
                chunk.started = true;
                await this.chunkStartedCallback(chunk.responseId, chunk.pcm);
            }
            const step = Math.min(remaining, chunk.totalMs - chunk.playedMs);
            chunk.playedMs += step;
            remaining -= step;
            consumed += step;
            if (chunk.playedMs + 1e-6 >= chunk.totalMs) {
                this.queue.shift();
                this.heard.set(chunk.responseId, (this.heard.get(chunk.responseId) ?? 0) + chunk.totalMs);
                await this.chunkPlayedCallback(chunk.responseId, chunk.pcm);
            }
        }
        await this.maybeFinish();
        return consumed;
    }

    private async maybeFinish(): Promise<void> {
        if (!this.serverFinished || this.queue.length > 0 || !this.activeResponseId) return;
        const responseId = this.activeResponseId;
        this.activeResponseId = null;
        this.serverFinished = false;
        await this.allChunksPlayedCallback(responseId);
    }
}

/** Base64 and PCM encoding backed by Node buffers. */
export class BufferEncoding extends BaseEncoding {
    encodePCM(data: ArrayBuffer, sampleRate: number, targetRate: number): string {
        if (sampleRate === targetRate) return this.encodeBase64(data);
        const input = new Int16Array(data);
        const output = new Int16Array(Math.round((input.length * targetRate) / sampleRate));
        for (let i = 0; i < output.length; i++) {
            const position = (i * sampleRate) / targetRate;
            const left = Math.min(Math.floor(position), input.length - 1);
            const right = Math.min(left + 1, input.length - 1);
            output[i] = Math.round(input[left]! + (input[right]! - input[left]!) * (position - left));
        }
        return this.encodeBase64(output.buffer);
    }

    encodeBase64(data: ArrayBuffer): string {
        return Buffer.from(data).toString("base64");
    }

    decodeBase64(base64: string): ArrayBuffer {
        const bytes = Buffer.from(base64, "base64");
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    }
}

/** Runs deferred tasks on the microtask queue. */
export class ImmediateScheduler extends BaseDeferredTaskScheduler {
    private disposed = false;

    schedule(task: () => void): void {
        if (this.disposed) return;
        queueMicrotask(() => {
            if (!this.disposed) task();
        });
    }

    dispose(): void {
        this.disposed = true;
    }
}

export interface NodeTestPlatform {
    /** Runtime to pass to `registerPlatformRuntime`. */
    runtime: PlatformRuntime;
    /** Microphone created for the most recently opened session. */
    readonly microphone: ScriptedInputAudioSession | null;
    /** Speaker created for the most recently opened session. */
    readonly speaker: VirtualOutputAudioSession | null;
    /** Scheduler created for the most recently opened session. */
    readonly scheduler: ImmediateScheduler | null;
}

/** Create a platform whose devices are controlled by the test. */
export function createNodeTestPlatform(options: { autoPlay?: boolean } = {}): NodeTestPlatform {
    let microphone: ScriptedInputAudioSession | null = null;
    let speaker: VirtualOutputAudioSession | null = null;
    let scheduler: ImmediateScheduler | null = null;
    const runtime: PlatformRuntime = {
        createInputAudioSession(config) {
            microphone = new ScriptedInputAudioSession(config);
            return microphone;
        },
        createOutputAudioSession(config) {
            speaker = new VirtualOutputAudioSession(config, options);
            return speaker;
        },
        createWebSocket(url, protocols) {
            return new NodeWebSocket(url, protocols);
        },
        delay(milliseconds) {
            return sleep(milliseconds);
        },
        createDeferredTaskScheduler() {
            scheduler = new ImmediateScheduler();
            return scheduler;
        },
        createEncoding() {
            return new BufferEncoding();
        },
    };
    return {
        runtime,
        get microphone() { return microphone; },
        get speaker() { return speaker; },
        get scheduler() { return scheduler; },
    };
}
