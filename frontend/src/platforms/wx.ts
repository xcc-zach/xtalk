import { BaseInputAudioSession, BaseOutputAudioSession } from "../bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";

declare const uni: any;

export { WxInputAudioSession, WxOutputAudioSession };
export type { WxInputAudioSessionConfig, WxOutputAudioSessionConfig };

interface WxInputAudioSessionConfig extends InputAudioSessionConfig {
    enableVAD?: boolean;
    vadAmplitudeThreshold?: number;
}

interface WxOutputAudioSessionConfig extends OutputAudioSessionConfig {
}

class WxInputAudioSession extends BaseInputAudioSession {
    private config: WxInputAudioSessionConfig;
    private _muted = false;
    private recorder: any = null;
    private started = false;
    private speaking = false;
    private targetFrameSize = 512;
    private nativeSampleRate: number;
    private inputBuffer: number[] = [];
    private outputBuffer: number[] = [];

    constructor(config: WxInputAudioSessionConfig) {
        super();
        this.config = { ...config };
        this.nativeSampleRate = this.config.sampleRate;
        if (this.config.enableVAD === undefined) {
            this.config.enableVAD = false;
        }
        if (this.config.vadAmplitudeThreshold === undefined) {
            this.config.vadAmplitudeThreshold = 0.012;
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

                this.frameCallback(frame);
                this.handleVad(frame);
            }
        });

        this.recorder.onStop(() => {
            this.endSpeechIfNeeded();
        });

        this.recorder.start({
            duration: 600000,
            sampleRate: this.config.sampleRate,
            numberOfChannels: 1,
            encodeBitRate: 96000,
            format: "PCM",
            frameSize: 5,
        });

        this.started = true;
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
            const sample = Math.max(-1, Math.min(1, frame[i] ?? 0));
            int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }
        return int16.buffer;
    }

    private startSpeechIfNeeded(): void {
        if (this.speaking) {
            return;
        }
        this.speaking = true;
        this.speechStartCallback();
    }

    private endSpeechIfNeeded(): void {
        if (!this.speaking) {
            return;
        }
        this.speaking = false;
        this.speechEndCallback();
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
        if (isSpeech && !this.speaking) {
            this.startSpeechIfNeeded();
            return;
        }
        if (!isSpeech && this.speaking) {
            this.endSpeechIfNeeded();
        }
    }
}

class WxOutputAudioSession extends BaseOutputAudioSession {
    private config: WxOutputAudioSessionConfig;
    private sampleRate: number;
    private player: any = null;
    private opened = false;
    private paused = false;
    private queue: ArrayBuffer[] = [];
    private playing = false;
    private currentBatch: ArrayBuffer[] = [];

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
                this.allChunksPlayedCallback();
            }
            this.playNext();
        });
        this.player.onError(() => {
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
        const mergedChunk = this.concatPcmChunks(batch);
        this.currentBatch = batch;
        this.chunkStartedCallback(batch[0]!);
        const wavDataUri = this.pcmToWavDataUri(mergedChunk, this.sampleRate);
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