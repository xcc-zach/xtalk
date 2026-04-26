import { BaseWebSocket } from "../bases/websocket";
import { BaseInputAudioSession, BaseOutputAudioSession } from "../bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";

console.log('web.ts');

// import vadProcessorUrl from "/worklets/vad-processor.worklet.js";
// import fastEnhancerOnnxUrl from "/models/fastenhancer_s.onnx";
export { WebWebSocket, WebInputAudioSession, WebOutputAudioSession };

declare const uni: any;

class WebWebSocket extends BaseWebSocket {
    private instance: any;
    private opened = false;
    constructor(url: string | URL, protocols?: string | string[]) {
        super();
        const urlText = typeof url === 'string' ? url : url.toString();
        const uniApi = typeof uni !== 'undefined' ? uni : undefined;
        if (!uniApi || typeof uniApi.connectSocket !== 'function') {
            throw new Error('uni.connectSocket is not available in current environment');
        }

        const protocolList = typeof protocols === 'string' ? [protocols] : protocols;
        this.instance = uniApi.connectSocket({
            url: urlText,
            protocols: protocolList,
            complete: () => {}
        });

        if (!this.instance
            || typeof this.instance.onOpen !== 'function'
            || typeof this.instance.onMessage !== 'function'
            || typeof this.instance.onClose !== 'function'
            || typeof this.instance.onError !== 'function') {
            throw new Error('uni.connectSocket did not return a valid SocketTask');
        }

        this.instance.onOpen(() => {
            this.opened = true;
        });
        this.instance.onClose(() => {
            this.opened = false;
        });
        this.instance.onError(() => {
            this.opened = false;
        });
    }
    ready(): boolean {
        return this.opened;
    }
    send(data: string | ArrayBuffer): void {
        this.instance.send({ data });
    }
    close(): void {
        this.opened = false;
        this.instance.close({});
    }
    addEventListener(type: "open" | "message" | "close" | "error", listener: (evt?: any) => any): void {
        switch (type) {
            case 'open':
                this.instance.onOpen((evt: any) => listener(evt));
                break;
            case 'message':
                this.instance.onMessage((evt: any) => listener(evt));
                break;
            case 'close':
                this.instance.onClose((evt: any) => listener(evt));
                break;
            case 'error':
                this.instance.onError((evt: any) => listener(evt));
                break;
        }
    }
}

interface WebInputAudioSessionConfig extends InputAudioSessionConfig {
    // Whether to enable VAD on client. Defaults to true.
    enableVAD?: boolean;
    // Whether to enable enhancer on client. Defaults to true if enhancer model is available.
    enableEnhancer?: boolean;
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
    readonly ENHANCER_PARAMS = {
        hopSize: 256,
        nFFT: 512,
    }
    readonly DIRECT_VAD_PARAMS = {
        speechEnergyThreshold: 0.015,
        positiveFramesBeforeStart: 2,
        negativeFramesBeforeEnd: 20,
    }
    private config: WebInputAudioSessionConfig;
    private _muted = false;
    private audioContext: AudioContext | null = null;
    private directSpeechActive = false;
    private directPositiveFrameCount = 0;
    private directNegativeFrameCount = 0;
    constructor(config: WebInputAudioSessionConfig) {
        super()
        // Default config
        config = { ...config };
        if (config.enableVAD === undefined) {
            config.enableVAD = false;
            // config.enableVAD = false; // Disable VAD by default as it may cause issues on some devices
        }
        if (config.enableEnhancer === undefined) {
            config.enableEnhancer = true;
            // config.enableEnhancer = false; // Disable enhancer by default as it may cause issues on some devices
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
        this.endDirectSpeechIfNeeded();
        this.audioContext.close();
        this.audioContext = null;
    }

    get muted(): boolean {
        return this._muted;
    }
    set muted(value: boolean) {
        if (value) {
            this.endDirectSpeechIfNeeded();
        }
        this._muted = value;
    }

    private setupDirectProcessing(
        frameProcessNode: AudioWorkletNode,
        enhanceFrame: (frame: Float32Array) => Promise<Float32Array>
    ): void {
        this.resetDirectProcessingState();
        frameProcessNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                if (this.muted) return;
                const frame = event.data.frame;
                const enhancedFrame = await enhanceFrame(frame);
                this.updateDirectSpeechState(enhancedFrame);
                this.frameCallback(this.float32ToInt16(enhancedFrame));
            }
        };
    }

    private updateDirectSpeechState(frame: Float32Array): void {
        const rms = this.calculateFrameRms(frame);
        const isSpeechFrame = rms >= this.DIRECT_VAD_PARAMS.speechEnergyThreshold;

        if (isSpeechFrame) {
            this.directPositiveFrameCount += 1;
            this.directNegativeFrameCount = 0;
            if (!this.directSpeechActive && this.directPositiveFrameCount >= this.DIRECT_VAD_PARAMS.positiveFramesBeforeStart) {
                this.directSpeechActive = true;
                this.speechStartCallback();
            }
            return;
        }

        this.directPositiveFrameCount = 0;
        if (!this.directSpeechActive) {
            return;
        }

        this.directNegativeFrameCount += 1;
        if (this.directNegativeFrameCount >= this.DIRECT_VAD_PARAMS.negativeFramesBeforeEnd) {
            this.endDirectSpeechIfNeeded();
        }
    }

    private calculateFrameRms(frame: Float32Array): number {
        if (frame.length === 0) {
            return 0;
        }
        let energy = 0;
        for (let i = 0; i < frame.length; i++) {
            const sample = frame[i] ?? 0;
            energy += sample * sample;
        }
        return Math.sqrt(energy / frame.length);
    }

    private resetDirectProcessingState(): void {
        this.directSpeechActive = false;
        this.directPositiveFrameCount = 0;
        this.directNegativeFrameCount = 0;
    }

    private endDirectSpeechIfNeeded(): void {
        if (!this.directSpeechActive) {
            this.resetDirectProcessingState();
            return;
        }
        this.directSpeechActive = false;
        this.directPositiveFrameCount = 0;
        this.directNegativeFrameCount = 0;
        this.speechEndCallback();
    }

    private async setupVAD(
        frameProcessNode: AudioWorkletNode,
        enhanceFrame: (frame: Float32Array) => Promise<Float32Array>,
        resetEnhancer: () => void
    ): Promise<void> {
        const vadURL = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.27/dist/silero_vad_v5.onnx';
        const vadArrayBuffer = await fetch(vadURL).then(r => r.arrayBuffer());
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
                    const frame = ev.frame;
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
                    if (!this.muted) {
                        this.frameCallback(this.float32ToInt16(frame));
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
                frameQueue.push(event.data.frame);
                if (isProcessingFrameQueue) return;
                isProcessingFrameQueue = true;
                while (frameQueue.length > 0) {
                    const frame = frameQueue.shift();
                    await frameProcessor.process(frame, onFrameProcessorEvent);
                }
                isProcessingFrameQueue = false;
            }
        };
        frameProcessor.resume();
    }

    private async setupAudioPipeline(): Promise<{
        audioContext: AudioContext;
        frameProcessNode: AudioWorkletNode;
    }> {
        const AudioContextCtor = (window as any).AudioContext || (window as any).webkitAudioContext;
        if (!AudioContextCtor) {
            throw new Error('AudioContext is not supported in current browser');
        }
        const audioContext = new AudioContextCtor({ sampleRate: this.config.sampleRate }) as AudioContext;
        const inputStream = await this.requestMicrophoneStream();
        const sourceNode = audioContext.createMediaStreamSource(inputStream);
        await audioContext.audioWorklet.addModule('/static/worklets/vad-processor.worklet.js');
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

    private async requestMicrophoneStream(): Promise<MediaStream> {
        const constraints: MediaStreamConstraints = {
            audio: {
                channelCount: 1,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: false
            }
        };

        const isLocalhost = /^(localhost|127\.0\.0\.1)$/i.test(window.location.hostname);
        if (!window.isSecureContext && !isLocalhost) {
            throw new Error('Microphone access requires HTTPS (or localhost). Please use https:// on mobile browsers.');
        }

        if (navigator.mediaDevices && typeof navigator.mediaDevices.getUserMedia === 'function') {
            return navigator.mediaDevices.getUserMedia(constraints);
        }

        const nav = navigator as any;
        const legacyGetUserMedia = nav.getUserMedia || nav.webkitGetUserMedia || nav.mozGetUserMedia || nav.msGetUserMedia;
        if (typeof legacyGetUserMedia === 'function') {
            return new Promise<MediaStream>((resolve, reject) => {
                legacyGetUserMedia.call(navigator, constraints, resolve, reject);
            });
        }

        throw new Error('getUserMedia is not supported in this browser');
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
            const enhancerArrayBuffer = await fetch('/static/models/fastenhancer_s.onnx').then(r => r.arrayBuffer());
            const enhancerSession = await window.ort.InferenceSession.create(enhancerArrayBuffer);
            const enhancerCache: Record<string, any> = {
                'cache_in_0': new window.ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_1': new window.ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_2': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_3': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_4': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48])
            };
            let enhancerInputBuffer: number[] = [];
            let enhancerOutputBuffer: number[] = [];
            let isFirstEnhancerFrame = true;

            const enhanceFrame = async (frame: Float32Array) => {
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
        } catch {
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

    private async ensureModelsEnv() {
        if (!this.config.enableEnhancer && !this.config.enableVAD) {
            // No need to load models
            return;
        }
        // Inject window.ort and window.vad
        const inject = (src: string) => new Promise<void>((resolve, reject) => {
            const s = document.createElement('script');
            s.src = src;
            s.onload = () => resolve();
            s.onerror = (e) => reject(e);
            document.head.appendChild(s);
        });
        if (!window.ort) {
            // Pick ORT version by UA (only iOS stays on 1.17.0)
            const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
            const ortVersion = isIOS ? '1.17.0' : '1.22.0';

            await inject(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.js`);
            window.ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;
        }
        if (this.config.enableVAD && !window.vad) {
            await inject('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.27/dist/bundle.min.js');
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
    private audioTimeToPlay = 0;
    private audioChunkStartedTimeouts: ReturnType<typeof createPausableTimeout>[] = [];
    private audioChunksPaused: ArrayBuffer[] = [];
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
    async stop(): Promise<void> {
        this.audioChunkStartedTimeouts.forEach(timeout => {
            timeout.cancel();
        });
        this.audioChunkStartedTimeouts.length = 0;
        this.audioBufferSources.forEach(source => {
            source.onended = null;
            source.disconnect();
        });
        this.audioBufferSources.length = 0;
        this.audioTimeToPlay = 0;
        this.audioChunksPaused.length = 0;
        // DO NOT suspend to avoid pop sounds after restart
        // await this.audioContext?.suspend();
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
            // If this is the last scheduled chunk, trigger onAllChunksPlayed
            if (this.audioBufferSources.length === 0) {
                this.allChunksPlayedCallback();
            }
        };

        // Add to buffer sources list before starting
        this.audioBufferSources.push(source);

        // Schedule time to play
        const currentTime = this.audioContext.currentTime;
        if (this.audioTimeToPlay < currentTime) {
            this.audioTimeToPlay = currentTime;
        }
        source.start(this.audioTimeToPlay);

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
