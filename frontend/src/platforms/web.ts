import { BaseWebSocket } from "../bases/websocket";
import type { BaseWebSocketCloseEvent, BaseWebSocketMessageEvent } from "../bases/websocket";
import { BaseInputAudioSession, BaseOutputAudioSession } from "../bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import { BaseHTTPClient, HTTPRequestError } from "../bases/http";
import type { ResolvableURL, SessionServiceURLConfig, SessionServiceURLs } from "../bases/http";
import { BaseEncoding } from "../bases/encoding";
import { BasePersistenceStore } from "../bases/persistence";
import { BaseDeferredTaskScheduler } from "../bases/task-scheduler";

import vadProcessorUrl from "../../worklets/vad-processor.worklet.js";

const FRONTEND_UTILITIES_BASE_URL = "/xtalk/frontend-utilities";
const ONNXRUNTIME_WEB_CDN_BASE_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web";
const VAD_WEB_VERSION = "0.0.27";
const VAD_WEB_CDN_BASE_URL = `https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@${VAD_WEB_VERSION}/dist`;
export {
    WebDeferredTaskScheduler,
    WebWebSocket,
    WebInputAudioSession,
    WebOutputAudioSession,
    WebHTTPClient,
    WebEncoding,
    WebPersistenceStore,
    resolveWebServiceURLs,
    buildWebSocketURLWithAccessToken,
};

/**
 * Web implementation of deferred task scheduling for non-blocking callbacks.
 */
class WebDeferredTaskScheduler extends BaseDeferredTaskScheduler {
    private readonly taskQueue: Array<() => void> = [];
    private readonly messageChannel: MessageChannel | null;
    private timeoutId: number | null = null;
    private scheduled = false;
    private disposed = false;

    constructor() {
        super();
        this.messageChannel = typeof MessageChannel === "function" ? new MessageChannel() : null;
        if (this.messageChannel) {
            this.messageChannel.port1.onmessage = () => {
                this.flush();
            };
        }
    }

    schedule(task: () => void): void {
        if (this.disposed) {
            return;
        }
        this.taskQueue.push(task);
        if (this.scheduled) {
            return;
        }
        this.scheduled = true;
        if (this.messageChannel) {
            this.messageChannel.port2.postMessage(null);
            return;
        }
        this.timeoutId = window.setTimeout(() => {
            this.timeoutId = null;
            this.flush();
        }, 0);
    }

    dispose(): void {
        this.disposed = true;
        this.taskQueue.length = 0;
        this.scheduled = false;
        if (this.timeoutId !== null) {
            window.clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
        if (this.messageChannel) {
            this.messageChannel.port1.onmessage = null;
            this.messageChannel.port1.close();
            this.messageChannel.port2.close();
        }
    }

    private flush(): void {
        if (this.disposed) {
            return;
        }
        this.scheduled = false;
        const queuedTasks = this.taskQueue.splice(0, this.taskQueue.length);
        for (const task of queuedTasks) {
            task();
        }
    }
}

class WebWebSocket extends BaseWebSocket {
    private instance: WebSocket;
    constructor(url: string | URL, protocols?: string | string[]) {
        super();
        this.instance = new WebSocket(url, protocols);
        this.instance.binaryType = 'arraybuffer';
    }
    ready(): boolean {
        return this.instance.readyState === WebSocket.OPEN;
    }
    send(data: string | ArrayBuffer): void {
        this.instance.send(data);
    }
    close(): void {
        this.instance.close();
    }
    addEventListener(type: "open" | "error", listener: () => any): void;
    addEventListener(type: "message", listener: (evt: BaseWebSocketMessageEvent) => any): void;
    addEventListener(type: "close", listener: (evt: BaseWebSocketCloseEvent) => any): void;
    addEventListener(
        type: "open" | "message" | "close" | "error",
        listener: (() => any) | ((evt: BaseWebSocketMessageEvent) => any) | ((evt: BaseWebSocketCloseEvent) => any),
    ): void {
        if (type === "message") {
            this.instance.addEventListener("message", (event: MessageEvent<string | ArrayBuffer>) => {
                (listener as (evt: BaseWebSocketMessageEvent) => any)({
                    data: event.data,
                });
            });
            return;
        }
        if (type === "close") {
            this.instance.addEventListener("close", (event: CloseEvent) => {
                (listener as (evt: BaseWebSocketCloseEvent) => any)({
                    code: event.code,
                    reason: event.reason,
                    wasClean: event.wasClean,
                });
            });
            return;
        }
        this.instance.addEventListener(type, () => {
            (listener as () => any)();
        });
    }
}

function resolveBaseURL(rawURL: ResolvableURL): URL {
    return new URL(rawURL.toString(), window.location.href);
}

function createAuthorizedHeaders(accessToken: string | null): Headers {
    const headers = new Headers();
    if (accessToken) {
        headers.set("Authorization", `Bearer ${accessToken}`);
    }
    return headers;
}

class WebHTTPClient extends BaseHTTPClient {
    async postJSON<T>(url: ResolvableURL, accessToken: string | null): Promise<T> {
        const response = await fetch(url, {
            method: "POST",
            headers: createAuthorizedHeaders(accessToken),
        });
        if (!response.ok) {
            throw new HTTPRequestError(response.status);
        }
        return await response.json() as T;
    }

    async getJSON<T>(url: ResolvableURL, accessToken: string): Promise<T> {
        const response = await fetch(url, {
            method: "GET",
            headers: createAuthorizedHeaders(accessToken),
        });
        if (!response.ok) {
            throw new HTTPRequestError(response.status);
        }
        return await response.json() as T;
    }

    async postFile(
        url: ResolvableURL,
        accessToken: string,
        sessionId: string,
        file: Blob,
    ): Promise<void> {
        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("file", file);
        const response = await fetch(url, {
            method: "POST",
            headers: createAuthorizedHeaders(accessToken),
            body: formData,
        });
        if (!response.ok) {
            throw new HTTPRequestError(response.status);
        }
    }
}

class WebEncoding extends BaseEncoding {
    decodeBase64(base64: string): ArrayBuffer {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes.buffer;
    }
}

class WebPersistenceStore extends BasePersistenceStore {
    private supportsStorage(): boolean {
        return typeof window !== "undefined" && typeof window.localStorage !== "undefined";
    }

    resolveKey(websocketURL: ResolvableURL): string | null {
        if (!this.supportsStorage()) {
            return null;
        }
        const resolvedURL = new URL(websocketURL.toString(), window.location.href);
        return `xtalk:session:${resolvedURL.toString()}`;
    }

    load(key: string | null): string | null {
        if (!key || !this.supportsStorage()) {
            return null;
        }
        try {
            return window.localStorage.getItem(key);
        } catch {
            return null;
        }
    }

    save(key: string | null, value: string): void {
        if (!key || !this.supportsStorage()) {
            return;
        }
        try {
            window.localStorage.setItem(key, value);
        } catch {
            // Ignore storage failures so realtime usage continues normally.
        }
    }

    clear(key: string | null): void {
        if (!key || !this.supportsStorage()) {
            return;
        }
        try {
            window.localStorage.removeItem(key);
        } catch {
            // Ignore storage failures so realtime usage continues normally.
        }
    }
}

function resolveWebServiceURLs(
    websocketURL: ResolvableURL,
    config?: SessionServiceURLConfig,
): SessionServiceURLs {
    const httpURL = resolveBaseURL(websocketURL);
    if (httpURL.protocol === "ws:") {
        httpURL.protocol = "http:";
    } else if (httpURL.protocol === "wss:") {
        httpURL.protocol = "https:";
    }

    const defaultSessionDetail = (sessionId: string) =>
        new URL(`./api/sessions/${encodeURIComponent(sessionId)}`, httpURL);
    const configuredSessionDetail = config?.sessionDetail;

    return {
        login: config?.login ?? new URL("./api/auth/login", httpURL),
        sessions: config?.sessions ?? new URL("./api/sessions", httpURL),
        sessionDetail: typeof configuredSessionDetail === "function"
            ? configuredSessionDetail
            : (sessionId: string) => configuredSessionDetail ?? defaultSessionDetail(sessionId),
        upload: config?.upload ?? new URL("./api/upload", httpURL),
    };
}

function buildWebSocketURLWithAccessToken(
    websocketURL: ResolvableURL,
    accessToken: string,
): URL {
    const url = resolveBaseURL(websocketURL);
    url.searchParams.set("access_token", accessToken);
    return url;
}

interface WebInputAudioSessionConfig extends InputAudioSessionConfig {
    // Whether to enable VAD on client. Defaults to true.
    enableVAD?: boolean;
    // Whether to enable enhancer on client. Defaults to true.
    enableEnhancer?: boolean;
    // Base URL for locally hosted browser-side runtime and model assets.
    frontendUtilitiesBaseUrl?: string;
    // VAD redemption window in milliseconds.
    vadRedemptionMs?: number;
}
declare global {
    interface Window {
        ort?: any;
        vad?: any;
    }
}
class WebInputAudioSession extends BaseInputAudioSession {
    readonly VAD_PARAMS = {
        vadFrameSamples: 512,
        vadNegativeFramesBeforeEnd: 50,
        vadConfig: {
            positiveSpeechThreshold: 0.8,
            negativeSpeechThreshold: 0.2,
            preSpeechPadMs: 30,
            redemptionMs: 500,
            minSpeechMs: 250,
            submitUserSpeechOnPause: false
        }
    }
    readonly MAX_VAD_QUEUE_FRAMES = 8;
    readonly ENHANCER_PARAMS = {
        hopSize: 512,
        nFFT: 512,
    }
    private config: WebInputAudioSessionConfig;
    private _muted = false;
    private audioContext: AudioContext | null = null;
    constructor(config: WebInputAudioSessionConfig) {
        super()
        // Default config
        config = { ...config };
        if (config.enableVAD === undefined) {
            config.enableVAD = true;
        }
        if (config.enableEnhancer === undefined) {
            config.enableEnhancer = true;
        }
        if (
            typeof config.vadRedemptionMs === 'number'
            && Number.isFinite(config.vadRedemptionMs)
            && config.vadRedemptionMs >= 0
        ) {
            this.VAD_PARAMS.vadConfig.redemptionMs = config.vadRedemptionMs;
        }
        this.config = config;
    }
    async open(): Promise<void> {
        if (this.audioContext !== null) {
            throw new Error('Session already started');
        }
        await this.ensureModelsEnv();

        const { enhanceFrame, resetEnhancer } = await this.setupEnhancer();
        const { audioContext, frameProcessNode } = await this.setupAudioPipeline();

        if (this.config.enableVAD) {
            await this.setupVAD(frameProcessNode, enhanceFrame, resetEnhancer);
        } else {
            this.setupDirectProcessing(frameProcessNode, enhanceFrame);
        }

        this.audioContext = audioContext;
    }
    async close(): Promise<void> {
        if (!this.audioContext) {
            throw new Error('Session not started');
        };
        this.audioContext.close();
        this.audioContext = null;
    }

    get muted(): boolean {
        return this._muted;
    }
    set muted(value: boolean) {
        this._muted = value;
    }

    private setupDirectProcessing(
        frameProcessNode: AudioWorkletNode,
        enhanceFrame: (frame: Float32Array) => Promise<Float32Array>
    ): void {
        frameProcessNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                if (this.muted) return;
                const frame = event.data.frame;
                const enhancedFrame = await enhanceFrame(frame);
                this.frameCallback(this.float32ToInt16(enhancedFrame));
            }
        };
    }

    private async setupVAD(
        frameProcessNode: AudioWorkletNode,
        enhanceFrame: (frame: Float32Array) => Promise<Float32Array>,
        resetEnhancer: () => void
    ): Promise<void> {
        const vadArrayBuffer = await this.fetchArrayBufferWithFallback(
            this.frontendUtilityURL(`vad-web/${VAD_WEB_VERSION}/dist/silero_vad_v5.onnx`),
            `${VAD_WEB_CDN_BASE_URL}/silero_vad_v5.onnx`
        );
        const vadSession = await window.ort.InferenceSession.create(vadArrayBuffer);
        const vadStateZeros = Array(2 * 128).fill(0);
        let vadState = new window.ort.Tensor('float32', vadStateZeros, [2, 1, 128]);
        const vadSr = new window.ort.Tensor('int64', [BigInt(this.config.sampleRate)]);
        const vadHelpers = {
            negEndCounterEnabled: false,
            negEndCounter: 0
        };

        const frameProcessorProcess = async (frame: Float32Array) => {
            const enhancedFrame = await enhanceFrame(frame);
            const audioTensor = new window.ort.Tensor('float32', enhancedFrame, [1, enhancedFrame.length]);
            const inputs = { input: audioTensor, state: vadState, sr: vadSr };
            const out = await vadSession.run(inputs);
            vadState = out.stateN;
            const isSpeech = out.output.data[0];
            return { isSpeech, notSpeech: 1 - isSpeech };
        };
        const frameProcessorReset = () => {
            vadState = new window.ort.Tensor('float32', vadStateZeros, [2, 1, 128]);
            resetEnhancer();
        };
        const frameProcessor = new window.vad.FrameProcessor(
            frameProcessorProcess,
            frameProcessorReset,
            this.VAD_PARAMS.vadConfig,
            this.VAD_PARAMS.vadFrameSamples / this.config.sampleRate * 1000
        );
        const onFrameProcessorEvent = (ev: { msg: any; frame: Float32Array; probs: { notSpeech: number; }; }) => {
            switch (ev.msg) {
                case window.vad.Message.FrameProcessed:
                    if (vadHelpers.negEndCounterEnabled) {
                        const ns = Number(ev?.probs?.notSpeech ?? 0);
                        const nsHigh = ns > (1 - this.VAD_PARAMS.vadConfig.negativeSpeechThreshold);
                        vadHelpers.negEndCounter = nsHigh ? (vadHelpers.negEndCounter + 1) : 0;
                        if (vadHelpers.negEndCounter > this.VAD_PARAMS.vadNegativeFramesBeforeEnd) {
                            this.speechEndCallback();
                            vadHelpers.negEndCounterEnabled = false;
                            vadHelpers.negEndCounter = 0;
                        }
                    }
                    break;

                case window.vad.Message.SpeechStart:
                    this.speechStartCallback();
                    vadHelpers.negEndCounterEnabled = true;
                    vadHelpers.negEndCounter = 0;
                    break;

                case window.vad.Message.SpeechEnd:
                    this.speechEndCallback();
                    vadHelpers.negEndCounterEnabled = false;
                    vadHelpers.negEndCounter = 0;
                    break;
            }
        };
        const frameQueue: Float32Array[] = [];
        let isProcessingFrameQueue = false;
        frameProcessNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                if (this.muted) return;
                const frame = event.data.frame as Float32Array;
                // Keep microphone upload real-time; VAD must not be allowed to stall ASR input.
                this.frameCallback(this.float32ToInt16(frame));
                if (frameQueue.length >= this.MAX_VAD_QUEUE_FRAMES) {
                    frameQueue.splice(0, frameQueue.length - this.MAX_VAD_QUEUE_FRAMES + 1);
                }
                frameQueue.push(frame);
                if (isProcessingFrameQueue) return;
                isProcessingFrameQueue = true;
                try {
                    while (frameQueue.length > 0) {
                        const queuedFrame = frameQueue.shift();
                        if (!queuedFrame) {
                            continue;
                        }
                        await frameProcessor.process(queuedFrame, onFrameProcessorEvent);
                    }
                } finally {
                    isProcessingFrameQueue = false;
                }
            }
        };
        frameProcessor.resume();
    }

    private async setupAudioPipeline(): Promise<{
        audioContext: AudioContext;
        frameProcessNode: AudioWorkletNode;
    }> {
        const audioContext = new window.AudioContext({ sampleRate: this.config.sampleRate });
        const inputStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: false
            }
        });
        const sourceNode = audioContext.createMediaStreamSource(inputStream);
        await audioContext.audioWorklet.addModule(vadProcessorUrl);
        const frameProcessNode = new AudioWorkletNode(audioContext, 'vad-processor', {
            processorOptions: {
                targetSampleRate: this.config.sampleRate,
                targetFrameSize: this.VAD_PARAMS.vadFrameSamples
            }
        });
        const silentGainNode = audioContext.createGain();
        silentGainNode.gain.value = 0;
        sourceNode.connect(frameProcessNode);
        frameProcessNode.connect(silentGainNode);
        silentGainNode.connect(audioContext.destination);
        return { audioContext, frameProcessNode };
    }

    private async setupEnhancer(): Promise<{
        enhanceFrame: (frame: Float32Array) => Promise<Float32Array>;
        resetEnhancer: () => void;
    }> {
        const identityMap = async (frame: Float32Array) => frame;
        if (!this.config.enableEnhancer) {
            return { enhanceFrame: identityMap, resetEnhancer: () => { } };
        }

        try {
            const enhancerURL = this.frontendUtilityURL('xtalk/models/fastenhancer_s.onnx');
            const enhancerArrayBuffer = await this.fetchArrayBuffer(enhancerURL);
            const enhancerSession = await window.ort.InferenceSession.create(enhancerArrayBuffer);
            const enhancerCache: Record<string, any> = {
                'cache_in_0': new window.ort.Tensor('float32', new Float32Array(1 * 512).fill(0), [1, 512]),
                'cache_in_1': new window.ort.Tensor('float32', new Float32Array(1 * 512).fill(0), [1, 512]),
                'cache_in_2': new window.ort.Tensor('float32', new Float32Array(1 * 48 * 48).fill(0), [1, 48, 48]),
                'cache_in_3': new window.ort.Tensor('float32', new Float32Array(1 * 48 * 48).fill(0), [1, 48, 48]),
                'cache_in_4': new window.ort.Tensor('float32', new Float32Array(1 * 48 * 48).fill(0), [1, 48, 48])
            };
            let enhancerInputBuffer: number[] = [];
            let enhancerOutputBuffer: number[] = [];
            let isFirstEnhancerFrame = true;
            let enhancerRuntimeDisabled = false;

            const disableEnhancerRuntime = (error: unknown) => {
                enhancerRuntimeDisabled = true;
                this.config.enableEnhancer = false;
                enhancerInputBuffer = [];
                enhancerOutputBuffer = [];
                console.error(
                    'FastEnhancer inference failed. FastEnhancer has been disabled.',
                    error
                );
            };

            const enhanceFrame = async (frame: Float32Array) => {
                if (enhancerRuntimeDisabled) {
                    return frame;
                }
                try {
                    for (let i = 0; i < frame.length; i++) {
                        enhancerInputBuffer.push(frame[i]!);
                    }
                    while (enhancerInputBuffer.length >= this.ENHANCER_PARAMS.hopSize) {
                        const chunk = enhancerInputBuffer.splice(0, this.ENHANCER_PARAMS.hopSize);
                        const chunkArray = new Float32Array(chunk);
                        const wavIn = new window.ort.Tensor('float32', chunkArray, [1, this.ENHANCER_PARAMS.hopSize]);
                        const inputs: Record<string, any> = { wav_in: wavIn };
                        for (const inputName of Object.keys(enhancerCache)) {
                            inputs[inputName] = enhancerCache[inputName];
                        }
                        const outputs = await enhancerSession.run(inputs);
                        const outputNames = enhancerSession.outputNames;
                        const enhancedChunk = outputs[outputNames[0]].data;
                        for (let i = 1; i < outputNames.length; i++) {
                            const cacheName = `cache_in_${i - 1}`;
                            enhancerCache[cacheName] = outputs[outputNames[i]];
                        }
                        for (let i = 0; i < enhancedChunk.length; i++) {
                            enhancerOutputBuffer.push(enhancedChunk[i]);
                        }
                        if (isFirstEnhancerFrame && enhancerOutputBuffer.length >= (this.ENHANCER_PARAMS.nFFT - this.ENHANCER_PARAMS.hopSize)) {
                            enhancerOutputBuffer.splice(0, this.ENHANCER_PARAMS.nFFT - this.ENHANCER_PARAMS.hopSize);
                            isFirstEnhancerFrame = false;
                        }
                    }
                    if (enhancerOutputBuffer.length >= frame.length) {
                        const output = enhancerOutputBuffer.splice(0, frame.length);
                        return new Float32Array(output);
                    } else {
                        return frame;
                    }
                } catch (error) {
                    disableEnhancerRuntime(error);
                    return frame;
                }
            };

            const resetEnhancer = () => {
                enhancerInputBuffer = [];
                enhancerOutputBuffer = [];
                isFirstEnhancerFrame = true;
                for (const cacheName of Object.keys(enhancerCache)) {
                    const shape = enhancerCache[cacheName].dims;
                    const zeros = new Float32Array(shape.reduce((a: number, b: number) => a * b, 1)).fill(0);
                    enhancerCache[cacheName] = new window.ort.Tensor('float32', zeros, shape);
                }
            };

            return { enhanceFrame, resetEnhancer };
        } catch (error) {
            this.config.enableEnhancer = false;
            console.error(
                'Failed to load FastEnhancer from the server. FastEnhancer has been disabled.',
                error
            );
            return { enhanceFrame: identityMap, resetEnhancer: () => { } };
        }
    }

    private float32ToInt16(frame: Float32Array): ArrayBuffer {
        const int16 = new Int16Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const s = Math.max(-1, Math.min(1, frame[i]!));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return int16.buffer;
    }

    private frontendUtilityURL(path: string): string {
        const baseURL = this.config.frontendUtilitiesBaseUrl ?? FRONTEND_UTILITIES_BASE_URL;
        return `${baseURL.replace(/\/+$/, '')}/${path.replace(/^\/+/, '')}`;
    }

    private async fetchArrayBuffer(url: string): Promise<ArrayBuffer> {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${url}: ${response.status}`);
        }
        return await response.arrayBuffer();
    }

    private async fetchArrayBufferWithFallback(
        localURL: string,
        fallbackURL: string
    ): Promise<ArrayBuffer> {
        try {
            const response = await fetch(localURL);
            if (response.ok) {
                return await response.arrayBuffer();
            }
        } catch {
            // Fall back to the public URL below.
        }
        const fallbackResponse = await fetch(fallbackURL);
        if (!fallbackResponse.ok) {
            throw new Error(`Failed to fetch ${fallbackURL}: ${fallbackResponse.status}`);
        }
        return await fallbackResponse.arrayBuffer();
    }

    private injectScript(src: string): Promise<void> {
        return new Promise<void>((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = () => resolve();
            script.onerror = () => {
                script.remove();
                reject(new Error(`Failed to load script ${src}`));
            };
            document.head.appendChild(script);
        });
    }

    private formatLoadError(error: unknown): string {
        return error instanceof Error ? error.message : String(error);
    }

    private async injectScriptWithFallback(
        localURL: string,
        fallbackURL: string
    ): Promise<'local' | 'fallback'> {
        try {
            await this.injectScript(localURL);
            return 'local';
        } catch (localError) {
            try {
                await this.injectScript(fallbackURL);
                return 'fallback';
            } catch (fallbackError) {
                throw new Error(
                    `Failed to load script from local URL ${localURL} `
                    + `(${this.formatLoadError(localError)}) and fallback URL `
                    + `${fallbackURL} (${this.formatLoadError(fallbackError)}).`
                );
            }
        }
    }

    private async ensureModelsEnv() {
        if (!this.config.enableEnhancer && !this.config.enableVAD) {
            // No need to load models
            return;
        }
        if (!window.ort) {
            // Pick ORT version by UA (only iOS stays on 1.17.0)
            const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
            const ortVersion = isIOS ? '1.17.0' : '1.22.0';
            const localOrtBaseURL = this.frontendUtilityURL(`onnxruntime-web/${ortVersion}/dist/`);
            const fallbackOrtBaseURL = `${ONNXRUNTIME_WEB_CDN_BASE_URL}@${ortVersion}/dist/`;
            const ortSource = await this.injectScriptWithFallback(
                `${localOrtBaseURL}ort.js`,
                `${fallbackOrtBaseURL}ort.js`
            );

            window.ort.env.wasm.wasmPaths = ortSource === 'local'
                ? localOrtBaseURL
                : fallbackOrtBaseURL;
        }
        if (this.config.enableVAD && !window.vad) {
            await this.injectScriptWithFallback(
                this.frontendUtilityURL(`vad-web/${VAD_WEB_VERSION}/dist/bundle.min.js`),
                `${VAD_WEB_CDN_BASE_URL}/bundle.min.js`
            );
        }
    }
}

function createPausableTimeout(
    callback: () => void,
    delay: number
): {
    pause: () => void;
    resume: () => void;
    cancel: () => void;
} {
    let timerId: ReturnType<typeof setTimeout> | null = null;
    let startTime = 0;
    let remaining = delay;
    let running = false;
    let cancelled = false;

    function start(ms: number): void {
        startTime = Date.now();
        running = true;
        timerId = setTimeout(() => {
            running = false;
            timerId = null;
            remaining = 0;
            if (!cancelled) {
                callback();
            }
        }, ms);
    }

    function pause(): void {
        if (!running || timerId === null) return;
        clearTimeout(timerId);
        timerId = null;
        remaining -= Date.now() - startTime;
        running = false;
    }

    function resume(): void {
        if (running || cancelled || remaining <= 0) return;
        start(remaining);
    }

    function cancel(): void {
        if (timerId !== null) {
            clearTimeout(timerId);
            timerId = null;
        }
        running = false;
        cancelled = true;
        remaining = 0;
    }

    start(delay);

    return {
        pause,
        resume,
        cancel,
    };
}
class WebOutputAudioSession extends BaseOutputAudioSession {
    private audioContext: AudioContext | null = null;
    private audioBufferSources: AudioBufferSourceNode[] = [];
    private audioBufferSourceTimes = new Map<
        AudioBufferSourceNode,
        { start: number; end: number }
    >();
    private audioTimeToPlay = 0;
    private audioChunkStartedTimeouts: ReturnType<typeof createPausableTimeout>[] = [];
    private audioChunksPaused: ArrayBuffer[] = [];
    private serverTtsFinished = false;
    constructor(private config: OutputAudioSessionConfig) {
        super();
    }
    async open(): Promise<void> {
        if (this.audioContext !== null) {
            throw new Error('Session already started');
        }
        this.audioContext = new window.AudioContext({ sampleRate: this.config.sampleRate });
        await this.audioContext.resume();
    }
    async close(): Promise<void> {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        await this.stop();
        this.audioContext.close();
        this.audioContext = null;
        this.audioTimeToPlay = 0;
    }
    async pause(): Promise<void> {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        if (this.audioContext.state == 'suspended') {
            throw new Error('Session already paused');
        }
        this.audioChunkStartedTimeouts.forEach(timeout => {
            timeout.pause();
        });
        await this.audioContext.suspend();
    }
    async resume(): Promise<void> {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        if (this.audioContext.state == 'running') {
            throw new Error('Session not paused');
        }
        this.audioChunkStartedTimeouts.forEach(timeout => {
            timeout.resume();
        });
        await this.audioContext.resume();
        // Play the paused chunks immediately
        for (const chunk of this.audioChunksPaused) {
            await this.pushAudioChunk(chunk);
        }
        this.audioChunksPaused.length = 0;
    }
    async stop(): Promise<{ unconfirmedPlayedMs: number }> {
        const currentTime = this.audioContext?.currentTime ?? 0;
        let unconfirmedPlayedMs = 0;
        this.audioChunkStartedTimeouts.forEach(timeout => {
            timeout.cancel();
        });
        this.audioChunkStartedTimeouts.length = 0;
        this.audioBufferSources.forEach(source => {
            const timing = this.audioBufferSourceTimes.get(source);
            if (timing && currentTime > timing.start) {
                unconfirmedPlayedMs += Math.max(
                    0,
                    Math.min(currentTime, timing.end) - timing.start,
                ) * 1000;
            }
            source.onended = null;
            try { source.stop(); } catch { }
            try { source.disconnect(); } catch { }
        });
        this.audioBufferSources.length = 0;
        this.audioBufferSourceTimes.clear();
        this.audioTimeToPlay = 0;
        this.audioChunksPaused.length = 0;
        this.serverTtsFinished = false;
        // DO NOT suspend to avoid pop sounds after restart
        // await this.audioContext?.suspend();
        return { unconfirmedPlayedMs };
    }
    async notifyTTSFinished(): Promise<void> {
        this.serverTtsFinished = true;
        await this.maybeNotifyPlaybackFinished();
    }
    private async maybeNotifyPlaybackFinished(): Promise<void> {
        if (!this.serverTtsFinished || this.audioBufferSources.length !== 0) {
            return;
        }
        this.serverTtsFinished = false;
        await this.allChunksPlayedCallback();
    }
    async pushAudioChunk(pcm_chunk_int16: ArrayBuffer): Promise<void> {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        // If suspended, meaning that the session is paused; cache the incoming audio for future
        if (this.audioContext.state === 'suspended') {
            this.audioChunksPaused.push(pcm_chunk_int16);
            return;
        }
        const int16 = new Int16Array(pcm_chunk_int16);
        if (int16.length === 0) return;
        const float32 = new Float32Array(int16.length);
        int16.forEach((value, index) => {
            float32[index] = value / 32768;
        });

        // Schedule audio play
        const buffer = this.audioContext.createBuffer(1, float32.length, this.config.sampleRate);
        buffer.getChannelData(0).set(float32);
        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        // Mount onended callback before starting
        source.onended = () => {
            this.chunkPlayedCallback(int16.buffer);
            // Remove this source from the list
            const idx = this.audioBufferSources.indexOf(source);
            if (idx !== -1) this.audioBufferSources.splice(idx, 1);
            this.audioBufferSourceTimes.delete(source);
            void this.maybeNotifyPlaybackFinished();
        };

        // Add to buffer sources list before starting
        this.audioBufferSources.push(source);

        // Schedule time to play
        const currentTime = this.audioContext.currentTime;
        if (this.audioTimeToPlay < currentTime) {
            this.audioTimeToPlay = currentTime;
        }
        const chunkStartTime = this.audioTimeToPlay;
        const chunkEndTime = chunkStartTime + buffer.duration / source.playbackRate.value;
        this.audioBufferSourceTimes.set(source, {
            start: chunkStartTime,
            end: chunkEndTime,
        });
        source.start(chunkStartTime);

        // Mount onstarted
        const msForChunkStart = (this.audioTimeToPlay - currentTime) * 1000;
        if (msForChunkStart <= 0) {
            this.chunkStartedCallback(int16.buffer);
        } else {
            const timeout = createPausableTimeout(() => {
                this.chunkStartedCallback(int16.buffer);
                // Remove this timeout from the list
                const idx = this.audioChunkStartedTimeouts.indexOf(timeout);
                if (idx !== -1) this.audioChunkStartedTimeouts.splice(idx, 1);
            }, msForChunkStart);
            this.audioChunkStartedTimeouts.push(timeout);
        }
        // Update time to play for next chunk
        this.audioTimeToPlay += buffer.duration / source.playbackRate.value;
    }
}
