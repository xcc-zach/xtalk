import { BaseInputAudioSession, BaseOutputAudioSession } from "../bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";

console.log('wx.ts');


declare const uni: any;
declare const wx: any;

export { WxInputAudioSession, WxOutputAudioSession };
export type { WxInputAudioSessionConfig, WxOutputAudioSessionConfig };

interface WxInputAudioSessionConfig extends InputAudioSessionConfig {
    enableVAD?: boolean;
    vadAmplitudeThreshold?: number;
    vadModelUrl?: string;
    vadModelCachePath?: string;
}

interface WxOutputAudioSessionConfig extends OutputAudioSessionConfig {
}

class WxInputAudioSession extends BaseInputAudioSession {
    readonly MODEL_VAD_PARAMS = {
        modelUrl: 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.27/dist/silero_vad_v5.onnx',
        modelCacheFileName: 'silero_vad_v5.onnx',
        positiveSpeechThreshold: 0.8,
        positiveFramesBeforeStart: 2,
        negativeSpeechThreshold: 0.2,
        negativeFramesBeforeEnd: 50,
    }
    readonly ENERGY_VAD_PARAMS = {
        positiveFramesBeforeStart: 2,
        negativeFramesBeforeEnd: 20,
    }
    private config: WxInputAudioSessionConfig;
    private _muted = false;
    private recorder: any = null;
    private started = false;
    private speaking = false;
    private targetFrameSize = 512;
    private nativeSampleRate: number;
    private inputBuffer: number[] = [];
    private outputBuffer: number[] = [];
    private vadSession: any = null;
    private vadState: any = null;
    private vadSampleRateTensor: any = null;
    private vadNegativeFrameCount = 0;
    private vadPositiveFrameCount = 0;
    private vadFrameQueue: ArrayBuffer[] = [];
    private isProcessingVadFrames = false;
    private useNativeModelVad = false;

    constructor(config: WxInputAudioSessionConfig) {
        super();
        this.config = { ...config };
        this.nativeSampleRate = this.config.sampleRate;
        if (this.config.enableVAD === undefined) {
            this.config.enableVAD = true;
        }
        if (this.config.vadAmplitudeThreshold === undefined) {
            this.config.vadAmplitudeThreshold = 0.012;
        }
        if (!this.config.vadModelUrl) {
            this.config.vadModelUrl = this.MODEL_VAD_PARAMS.modelUrl;
        }
        if (!this.config.vadModelCachePath) {
            this.config.vadModelCachePath = `${this.getUserDataPath()}/${this.MODEL_VAD_PARAMS.modelCacheFileName}`;
        }
    }

    async open(): Promise<void> {
        if (this.started) {
            throw new Error("Session already started");
        }
        const uniApi = typeof uni !== "undefined" ? uni : undefined;
        if (!uniApi || typeof uniApi.getRecorderManager !== "function") {
            throw new Error("uni.getRecorderManager is not available in current environment");
        }

        this.recorder = uniApi.getRecorderManager();
        if (!this.recorder || typeof this.recorder.start !== "function") {
            throw new Error("RecorderManager is not available");
        }

        if (this.config.enableVAD) {
            try {
                await this.ensureModelVadSession();
                this.useNativeModelVad = true;
            } catch (error) {
                this.useNativeModelVad = false;
                console.warn("Falling back to energy-based VAD on mp-weixin:", error);
            }
        } else {
            this.useNativeModelVad = false;
        }

        this.recorder.onFrameRecorded((res: any) => {
            const frameBuffer = res?.frameBuffer;
            if (!(frameBuffer instanceof ArrayBuffer) || frameBuffer.byteLength === 0) {
                return;
            }
            if (this._muted) {
                return;
            }
            const nativeSampleRate = typeof res?.sampleRate === "number" && Number.isFinite(res.sampleRate) && res.sampleRate > 0
                ? res.sampleRate
                : this.config.sampleRate;
            const frames = this.processToFixedFrames(frameBuffer, nativeSampleRate);
            for (let i = 0; i < frames.length; i++) {
                const frame = frames[i]!;
                if (!this.config.enableVAD) {
                    this.startSpeechIfNeeded();
                    this.frameCallback(frame);
                    continue;
                }

                if (this.useNativeModelVad) {
                    this.enqueueVadFrame(frame);
                } else {
                    this.frameCallback(frame);
                    this.handleVad(frame);
                }
            }

        });

        this.recorder.onStop(() => {
            this.endSpeechIfNeeded();
            console.log("onStop回调函数执行,录音停止");
        });

        this.recorder.start({
            duration: 600000,
            sampleRate: this.config.sampleRate,
            numberOfChannels: 1,
            encodeBitRate: 96000,
            format: "PCM",
            frameSize: 1,
        });//录音开始

        this.started = true;
        console.log("open函数调用成功，config:", this.config);
    }

    async close(): Promise<void> {
        if (!this.started || !this.recorder) {
            throw new Error("Session not started");
        }
        this.recorder.stop();
        this.endSpeechIfNeeded();
        this.started = false;
        this.recorder = null;
        this.inputBuffer = [];
        this.outputBuffer = [];
        this.vadFrameQueue = [];
        this.isProcessingVadFrames = false;
        this.useNativeModelVad = false;
        if (this.vadSession && typeof this.vadSession.destroy === "function") {
            try {
                this.vadSession.destroy();
            } catch {
            }
        }
        this.vadSession = null;
        this.vadState = null;
        this.vadSampleRateTensor = null;
        this.vadNegativeFrameCount = 0;
        console.log("close函数调用成功");
    }

    get muted(): boolean {
        return this._muted;
    }

    set muted(value: boolean) {
        if (value) {
            this.endSpeechIfNeeded();
        }
        this._muted = value;
    }

    private processToFixedFrames(frameBuffer: ArrayBuffer, nativeSampleRate: number): ArrayBuffer[] {
        this.nativeSampleRate = nativeSampleRate;
        const pcm = new Int16Array(frameBuffer);
        for (let i = 0; i < pcm.length; i++) {
            this.inputBuffer.push((pcm[i] ?? 0) / 32768);
        }

        const frames: ArrayBuffer[] = [];
        const minInputSamples = Math.ceil(this.targetFrameSize * this.nativeSampleRate / this.config.sampleRate);

        while (this.inputBuffer.length >= minInputSamples) {
            const chunk = this.inputBuffer.splice(0, minInputSamples);
            const resampled = this.resample(chunk);

            for (let i = 0; i < resampled.length; i++) {
                this.outputBuffer.push(resampled[i] ?? 0);
            }

            while (this.outputBuffer.length >= this.targetFrameSize) {
                const frame = this.outputBuffer.splice(0, this.targetFrameSize);
                frames.push(this.float32ToInt16Buffer(frame));
            }
        }

        return frames;
    }

    private resample(inputData: number[]): Float32Array {
        if (this.nativeSampleRate === this.config.sampleRate) {
            return new Float32Array(inputData);
        }

        const ratio = this.nativeSampleRate / this.config.sampleRate;
        const outputLength = Math.floor(inputData.length / ratio);
        const output = new Float32Array(outputLength);

        for (let i = 0; i < outputLength; i++) {
            const pos = i * ratio;
            const index = Math.floor(pos);
            const frac = pos - index;

            const sample1 = inputData[index] ?? 0;
            const sample2 = inputData[Math.min(index + 1, inputData.length - 1)] ?? sample1;
            output[i] = sample1 + (sample2 - sample1) * frac;
        }

        return output;
    }

    private float32ToInt16Buffer(frame: number[]): ArrayBuffer {
        const int16 = new Int16Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const s = Math.max(-1, Math.min(1, frame[i] ?? 0));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return int16.buffer;
    }

    private startSpeechIfNeeded(): void {
        if (this.speaking) {
            return;
        }
        this.speaking = true;
        this.vadNegativeFrameCount = 0;
        this.vadPositiveFrameCount = 0;
        this.speechStartCallback();
    }

    private endSpeechIfNeeded(): void {
        if (!this.speaking) {
            this.vadNegativeFrameCount = 0;
            this.vadPositiveFrameCount = 0;
            if (this.config.enableVAD && this.useNativeModelVad) {
                this.resetModelVadState();
            }
            return;
        }
        this.speaking = false;
        this.vadNegativeFrameCount = 0;
        this.vadPositiveFrameCount = 0;
        if (this.config.enableVAD && this.useNativeModelVad) {
            this.resetModelVadState();
        }
        this.speechEndCallback();
    }

    private enqueueVadFrame(frameBuffer: ArrayBuffer): void {
        this.vadFrameQueue.push(frameBuffer.slice(0));
        if (this.isProcessingVadFrames) {
            return;
        }
        this.isProcessingVadFrames = true;
        void this.drainVadFrameQueue();
    }

    private async drainVadFrameQueue(): Promise<void> {
        try {
            while (this.vadFrameQueue.length > 0) {
                const frameBuffer = this.vadFrameQueue.shift();
                if (!frameBuffer) {
                    continue;
                }
                await this.processVadFrame(frameBuffer);
            }
        } finally {
            this.isProcessingVadFrames = false;
            if (this.vadFrameQueue.length > 0) {
                this.isProcessingVadFrames = true;
                void this.drainVadFrameQueue();
            }
        }
    }

    private async processVadFrame(frameBuffer: ArrayBuffer): Promise<void> {
        if (!this.vadSession) {
            this.frameCallback(frameBuffer);
            return;
        }

        const frameFloat32 = this.int16BufferToFloat32(frameBuffer);
        const inputs = {
            input: this.createInferenceTensor("float32", frameFloat32, [1, frameFloat32.length]),
            state: this.vadState,
            sr: this.vadSampleRateTensor,
        };
        const outputs = await this.runModelSession(inputs);
        const outputTensor = outputs?.output ?? outputs?.prob ?? outputs?.probs;
        const stateTensor = outputs?.stateN ?? outputs?.state ?? outputs?.hn;

        if (stateTensor) {
            this.vadState = stateTensor;
        }

        const speechScore = Number(this.getTensorValue(outputTensor, 0));
        const notSpeechScore = 1 - speechScore;
        const notSpeechHigh = notSpeechScore > (1 - this.MODEL_VAD_PARAMS.negativeSpeechThreshold);
        const isSpeechFrame = speechScore >= this.MODEL_VAD_PARAMS.positiveSpeechThreshold;

        if (isSpeechFrame) {
            this.vadPositiveFrameCount += 1;
            this.vadNegativeFrameCount = 0;
            if (!this.speaking && this.vadPositiveFrameCount >= this.MODEL_VAD_PARAMS.positiveFramesBeforeStart) {
                this.startSpeechIfNeeded();
            }
        } else if (!this.speaking) {
            this.vadPositiveFrameCount = 0;
        }

        if (this.speaking) {
            this.vadNegativeFrameCount = notSpeechHigh ? (this.vadNegativeFrameCount + 1) : 0;
            if (this.vadNegativeFrameCount > this.MODEL_VAD_PARAMS.negativeFramesBeforeEnd) {
                this.endSpeechIfNeeded();
            }
        }

        this.frameCallback(frameBuffer);
    }

    private async ensureModelVadSession(): Promise<void> {
        const wxApi = this.getWxApi();
        if (typeof wxApi.createInferenceSession !== "function") {
            throw new Error("wx.createInferenceSession is not available in current environment");
        }

        const modelPath = await this.ensureVadModelCached();
        this.vadSession = await this.createModelSession(wxApi, modelPath);
        this.resetModelVadState();
    }

    private getWxApi(): any {
        if (typeof wx !== "undefined") {
            return wx;
        }
        throw new Error("wx runtime is not available in current environment");
    }

    private async ensureVadModelCached(): Promise<string> {
        const fsManager = this.getWxApi().getFileSystemManager?.();
        const cachePath = this.config.vadModelCachePath;
        if (!fsManager || typeof fsManager.access !== "function") {
            throw new Error("wx.getFileSystemManager.access is not available in current environment");
        }
        if (!cachePath) {
            throw new Error("VAD model cache path is not configured");
        }

        try {
            await new Promise<void>((resolve, reject) => {
                fsManager.access({
                    path: cachePath,
                    success: () => resolve(),
                    fail: (error: any) => reject(error),
                });
            });
            return cachePath;
        } catch {
            return this.downloadVadModelToCache();
        }
    }

    private async downloadVadModelToCache(): Promise<string> {
        const wxApi = this.getWxApi();
        const modelUrl = this.config.vadModelUrl;
        const cachePath = this.config.vadModelCachePath;
        if (!modelUrl || !cachePath) {
            throw new Error("VAD model url or cache path is not configured");
        }

        const tempFilePath = await new Promise<string>((resolve, reject) => {
            wxApi.downloadFile({
                url: modelUrl,
                success: (result: any) => {
                    if (result?.statusCode >= 200 && result?.statusCode < 300 && result?.tempFilePath) {
                        resolve(result.tempFilePath);
                        return;
                    }
                    reject(new Error(`Failed to download VAD model, status code: ${result?.statusCode ?? "unknown"}`));
                },
                fail: (error: any) => reject(error),
            });
        });

        await this.copyFile(tempFilePath, cachePath);
        return cachePath;
    }

    private async copyFile(srcPath: string, destPath: string): Promise<void> {
        const fsManager = this.getWxApi().getFileSystemManager?.();
        if (!fsManager || typeof fsManager.copyFile !== "function") {
            throw new Error("wx.getFileSystemManager.copyFile is not available in current environment");
        }
        await new Promise<void>((resolve, reject) => {
            fsManager.copyFile({
                srcPath,
                destPath,
                success: () => resolve(),
                fail: (error: any) => reject(error),
            });
        });
    }

    private getUserDataPath(): string {
        const wxApi = this.getWxApi();
        return wxApi.env?.USER_DATA_PATH ?? "";
    }

    private async createModelSession(wxApi: any, modelPath: string): Promise<any> {
        const session = wxApi.createInferenceSession({
            model: modelPath,
            precisionLevel: 4,
        });
        if (!session) {
            throw new Error("Failed to create wx inference session for VAD model");
        }

        await new Promise<void>((resolve, reject) => {
            const handleLoad = () => {
                if (typeof session.offLoad === "function") {
                    session.offLoad(handleLoad);
                }
                if (typeof session.offError === "function") {
                    session.offError(handleError);
                }
                resolve();
            };
            const handleError = (error: any) => {
                if (typeof session.offLoad === "function") {
                    session.offLoad(handleLoad);
                }
                if (typeof session.offError === "function") {
                    session.offError(handleError);
                }
                reject(error);
            };

            if (typeof session.onLoad === "function") {
                session.onLoad(handleLoad);
            }
            if (typeof session.onError === "function") {
                session.onError(handleError);
            }
        });

        return session;
    }

    private resetModelVadState(): void {
        if (!this.vadSession) {
            return;
        }
        this.vadState = this.createInferenceTensor(
            "float32",
            new Float32Array(2 * 128).fill(0),
            [2, 1, 128]
        );
        this.vadSampleRateTensor = this.createInferenceTensor(
            "int64",
            [BigInt(this.config.sampleRate)],
            [1]
        );
        this.vadNegativeFrameCount = 0;
        this.vadPositiveFrameCount = 0;
    }

    private createInferenceTensor(type: string, data: any, shape: number[]): any {
        const wxApi = this.getWxApi();
        const candidates = [
            () => wxApi.createInferenceTensor({ type, data, shape }),
            () => wxApi.createInferenceTensor(type, data, shape),
            () => new wxApi.InferenceTensor(type, data, shape),
            () => new wxApi.InferenceTensor({ type, data, shape }),
        ];

        for (const candidate of candidates) {
            try {
                return candidate();
            } catch {
            }
        }

        return { type, data, shape };
    }

    private async runModelSession(inputs: Record<string, any>): Promise<any> {
        if (!this.vadSession || typeof this.vadSession.run !== "function") {
            throw new Error("wx inference session is not ready");
        }
        return this.vadSession.run(inputs);
    }

    private getTensorValue(tensor: any, index: number): number {
        if (!tensor) {
            return 0;
        }
        const data: any = tensor.data ?? tensor.value ?? tensor;
        if (ArrayBuffer.isView(data)) {
            const view = data as any;
            return Number(view[index] ?? 0);
        }
        if (Array.isArray(data)) {
            return Number(data[index] ?? 0);
        }
        return Number(data?.[index] ?? 0);
    }

    private int16BufferToFloat32(frameBuffer: ArrayBuffer): Float32Array {
        const pcm = new Int16Array(frameBuffer);
        const float32 = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) {
            float32[i] = (pcm[i] ?? 0) / 32768;
        }
        return float32;
    }

    private handleVad(frameBuffer: ArrayBuffer): void {
        const pcm = new Int16Array(frameBuffer);
        if (pcm.length === 0) {
            return;
        }
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) {
            const v = pcm[i]! / 32768;
            sum += v * v;
        }
        const rms = Math.sqrt(sum / pcm.length);
        const threshold = this.config.vadAmplitudeThreshold ?? 0.012;
        const isSpeech = rms >= threshold;
        if (isSpeech) {
            this.vadPositiveFrameCount += 1;
            this.vadNegativeFrameCount = 0;
            if (!this.speaking && this.vadPositiveFrameCount >= this.ENERGY_VAD_PARAMS.positiveFramesBeforeStart) {
                this.startSpeechIfNeeded();
            }
            return;
        }

        this.vadPositiveFrameCount = 0;
        if (!this.speaking) {
            return;
        }

        this.vadNegativeFrameCount += 1;
        if (this.vadNegativeFrameCount >= this.ENERGY_VAD_PARAMS.negativeFramesBeforeEnd) {
            this.endSpeechIfNeeded();
        }
    }
}

class WxOutputAudioSession extends BaseOutputAudioSession {
    readonly PLAYBACK_FINISH_GRACE_MS = 400;
    private config: WxOutputAudioSessionConfig;
    private sampleRate: number;
    private player: any = null;
    private opened = false;
    private paused = false;
    private queue: ArrayBuffer[] = [];
    private currentBatch: ArrayBuffer[] = [];
    private playing = false;
    private playbackFinishedTimer: ReturnType<typeof setTimeout> | null = null;

    constructor(config: WxOutputAudioSessionConfig) {
        super();
        this.config = { ...config };
        this.sampleRate = typeof config.sampleRate === "number" ? config.sampleRate : 48000;
    }

    async open(): Promise<void> {
        if (this.opened) {
            throw new Error("Session already started");
        }
        const uniApi = typeof uni !== "undefined" ? uni : undefined;
        if (!uniApi || typeof uniApi.createInnerAudioContext !== "function") {
            throw new Error("uni.createInnerAudioContext is not available in current environment");
        }
        this.player = uniApi.createInnerAudioContext();
        this.player.autoplay = false;
        this.player.obeyMuteSwitch = false;
        this.player.onEnded(() => {
            const playedBatch = this.currentBatch;
            this.currentBatch = [];
            this.playing = false;
            for (let i = 0; i < playedBatch.length; i++) {
                this.chunkPlayedCallback(playedBatch[i]!);
            }
            if (this.queue.length === 0) {
                this.schedulePlaybackFinishedCheck();
            }
            this.playNext();
        });
        this.player.onError(() => {
            this.cancelPlaybackFinishedCheck();
            this.currentBatch = [];
            this.playing = false;
            this.playNext();
        });
        this.opened = true;
    }

    async close(): Promise<void> {
        if (!this.opened) {
            throw new Error("Session not started");
        }
        await this.stop();
        this.cancelPlaybackFinishedCheck();
        if (this.player && typeof this.player.destroy === "function") {
            this.player.destroy();
        }
        this.player = null;
        this.opened = false;
    }

    async pause(): Promise<void> {
        if (!this.opened) {
            throw new Error("Session not started");
        }
        if (this.paused) {
            throw new Error("Session already paused");
        }
        this.paused = true;
        if (this.player && typeof this.player.pause === "function") {
            this.player.pause();
        }
    }

    async resume(): Promise<void> {
        if (!this.opened) {
            throw new Error("Session not started");
        }
        if (!this.paused) {
            throw new Error("Session not paused");
        }
        this.paused = false;
        if (this.player && this.playing && typeof this.player.play === "function") {
            this.player.play();
            return;
        }
        this.playNext();
    }

    async stop(): Promise<void> {
        this.cancelPlaybackFinishedCheck();
        this.queue = [];
        this.currentBatch = [];
        this.playing = false;
        this.paused = false;
        if (this.player && typeof this.player.stop === "function") {
            this.player.stop();
        }
    }

    async pushAudioChunk(pcmChunkInt16: ArrayBuffer): Promise<void> {
        if (!this.opened) {
            throw new Error("Session not started");
        }
        if (!(pcmChunkInt16 instanceof ArrayBuffer) || pcmChunkInt16.byteLength === 0) {
            return;
        }
        this.cancelPlaybackFinishedCheck();
        this.queue.push(pcmChunkInt16.slice(0));
        this.playNext();
    }

    private playNext(): void {
        if (!this.opened || this.paused || this.playing || this.queue.length === 0 || !this.player) {
            return;
        }
        const batch = this.collectQueuedChunks();
        if (batch.length === 0) {
            return;
        }
        this.currentBatch = batch;
        this.chunkStartedCallback(batch[0]!);
        const wavDataUri = this.pcmToWavDataUri(this.concatPcmChunks(batch), this.sampleRate);
        this.player.src = wavDataUri;
        this.playing = true;
        this.player.play();
    }

    private collectQueuedChunks(): ArrayBuffer[] {
        const batch: ArrayBuffer[] = [];
        while (this.queue.length > 0) {
            const chunk = this.queue.shift();
            if (chunk) {
                batch.push(chunk);
            }
        }
        return batch;
    }

    private concatPcmChunks(chunks: ArrayBuffer[]): ArrayBuffer {
        let totalBytes = 0;
        for (let i = 0; i < chunks.length; i++) {
            totalBytes += chunks[i]!.byteLength;
        }

        const merged = new Uint8Array(totalBytes);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
            const bytes = new Uint8Array(chunks[i]!);
            merged.set(bytes, offset);
            offset += bytes.byteLength;
        }
        return merged.buffer;
    }

    private schedulePlaybackFinishedCheck(): void {
        this.cancelPlaybackFinishedCheck();
        this.playbackFinishedTimer = setTimeout(() => {
            this.playbackFinishedTimer = null;
            if (!this.playing && this.queue.length === 0 && this.currentBatch.length === 0) {
                this.allChunksPlayedCallback();
            }
        }, this.PLAYBACK_FINISH_GRACE_MS);
    }

    private cancelPlaybackFinishedCheck(): void {
        if (this.playbackFinishedTimer !== null) {
            clearTimeout(this.playbackFinishedTimer);
            this.playbackFinishedTimer = null;
        }
    }

    private pcmToWavDataUri(pcmChunk: ArrayBuffer, sampleRate: number): string {
        const channels = 1;
        const bitsPerSample = 16;
        const blockAlign = channels * bitsPerSample / 8;
        const byteRate = sampleRate * blockAlign;
        const dataSize = pcmChunk.byteLength;
        const wavBuffer = new ArrayBuffer(44 + dataSize);
        const view = new DataView(wavBuffer);

        this.writeAscii(view, 0, "RIFF");
        view.setUint32(4, 36 + dataSize, true);
        this.writeAscii(view, 8, "WAVE");
        this.writeAscii(view, 12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, channels, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, byteRate, true);
        view.setUint16(32, blockAlign, true);
        view.setUint16(34, bitsPerSample, true);
        this.writeAscii(view, 36, "data");
        view.setUint32(40, dataSize, true);

        new Uint8Array(wavBuffer, 44).set(new Uint8Array(pcmChunk));

        const uniApi = typeof uni !== "undefined" ? uni : undefined;
        let base64 = "";
        if (uniApi && typeof uniApi.arrayBufferToBase64 === "function") {
            base64 = uniApi.arrayBufferToBase64(wavBuffer);
        } else {
            const bytes = new Uint8Array(wavBuffer);
            let binary = "";
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]!);
            }
            base64 = btoa(binary);
        }
        return `data:audio/wav;base64,${base64}`;
    }

    private writeAscii(view: DataView, offset: number, text: string): void {
        for (let i = 0; i < text.length; i++) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
    }
}