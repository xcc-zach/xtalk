/******/ // The require scope
/******/ var __webpack_require__ = {};
/******/ 
/************************************************************************/
/******/ /* webpack/runtime/define property getters */
/******/ (() => {
/******/ 	// define getter functions for harmony exports
/******/ 	__webpack_require__.d = (exports, definition) => {
/******/ 		for(var key in definition) {
/******/ 			if(__webpack_require__.o(definition, key) && !__webpack_require__.o(exports, key)) {
/******/ 				Object.defineProperty(exports, key, { enumerable: true, get: definition[key] });
/******/ 			}
/******/ 		}
/******/ 	};
/******/ })();
/******/ 
/******/ /* webpack/runtime/hasOwnProperty shorthand */
/******/ (() => {
/******/ 	__webpack_require__.o = (obj, prop) => (Object.prototype.hasOwnProperty.call(obj, prop))
/******/ })();
/******/ 
/******/ /* webpack/runtime/publicPath */
/******/ (() => {
/******/ 	__webpack_require__.p = "/src/js/";
/******/ })();
/******/ 
/************************************************************************/
var __webpack_exports__ = {};

;// ./src/utils.ts

var Platform;
(function (Platform) {
    Platform[Platform["MpWeixin"] = 0] = "MpWeixin";
    Platform[Platform["Web"] = 1] = "Web";
})(Platform || (Platform = {}));
function getPlatform() {
    if (typeof window !== "undefined" && typeof document !== "undefined")
        return Platform.Web;
    const hasUniRuntime = typeof uni !== "undefined" && typeof uni.getSystemInfoSync === "function";
    if (hasUniRuntime) {
        return Platform.MpWeixin;
    }
    throw new Error("getPlatform: Unknown platform");
}

;// ./src/bases/websocket.ts

class BaseWebSocket {
    sendJson(data) {
        this.send(JSON.stringify(data));
    }
    sendAudioChunk(pcm_chunk_int16) {
        this.send(pcm_chunk_int16);
    }
}

;// ./src/bases/audio-session.ts

class BaseInputAudioSession {
    onFrame(callback) {
        this.frameCallback = callback;
    }
    onSpeechStart(callback) {
        this.speechStartCallback = callback;
    }
    onSpeechEnd(callback) {
        this.speechEndCallback = callback;
    }
    frameCallback(_pcmChunkInt16) {
    }
    ;
    /**
     * Used only when VAD enabled
     */
    speechStartCallback() {
    }
    ;
    /**
     * Used only when VAD enabled
     */
    speechEndCallback() {
    }
    ;
}
class BaseOutputAudioSession {
    onChunkStarted(callback) {
        this.chunkStartedCallback = callback;
    }
    onChunkPlayed(callback) {
        this.chunkPlayedCallback = callback;
    }
    onAllChunksPlayed(callback) {
        this.allChunksPlayedCallback = callback;
    }
    chunkStartedCallback(_pcmChunkInt16) {
    }
    chunkPlayedCallback(_pcmChunkInt16) {
    }
    allChunksPlayedCallback() {
    }
}

;// ./worklets/vad-processor.worklet.js
const vad_processor_worklet_namespaceObject = __webpack_require__.p + "worklets/vad-processor.worklet.646a7957dbad113d1114.js";
;// ./models/fastenhancer_s.onnx
const fastenhancer_s_namespaceObject = __webpack_require__.p + "models/fastenhancer_s.0a43b8234398af92ea20.onnx";
;// ./src/platforms/web.ts





class WebWebSocket extends BaseWebSocket {
    constructor(url, protocols) {
        super();
        this.opened = false;
        const urlText = typeof url === 'string' ? url : url.toString();
        const uniApi = typeof uni !== 'undefined' ? uni : undefined;
        if (!uniApi || typeof uniApi.connectSocket !== 'function') {
            throw new Error('uni.connectSocket is not available in current environment');
        }
        const protocolList = typeof protocols === 'string' ? [protocols] : protocols;
        this.instance = uniApi.connectSocket({
            url: urlText,
            protocols: protocolList,
            complete: () => { }
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
    ready() {
        return this.opened;
    }
    send(data) {
        this.instance.send({ data });
    }
    close() {
        this.opened = false;
        this.instance.close({});
    }
    addEventListener(type, listener) {
        switch (type) {
            case 'open':
                this.instance.onOpen((evt) => listener(evt));
                break;
            case 'message':
                this.instance.onMessage((evt) => listener(evt));
                break;
            case 'close':
                this.instance.onClose((evt) => listener(evt));
                break;
            case 'error':
                this.instance.onError((evt) => listener(evt));
                break;
        }
    }
}
class WebInputAudioSession extends BaseInputAudioSession {
    constructor(config) {
        super();
        this.VAD_PARAMS = {
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
        };
        this.ENHANCER_PARAMS = {
            hopSize: 256,
            nFFT: 512,
        };
        this._muted = false;
        this.audioContext = null;
        // Default config
        config = { ...config };
        if (config.enableVAD === undefined) {
            config.enableVAD = true;
        }
        if (config.enableEnhancer === undefined) {
            config.enableEnhancer = true;
        }
        if (typeof config.vadRedemptionMs === 'number'
            && Number.isFinite(config.vadRedemptionMs)
            && config.vadRedemptionMs >= 0) {
            this.VAD_PARAMS.vadConfig.redemptionMs = config.vadRedemptionMs;
        }
        this.config = config;
    }
    async open() {
        if (this.audioContext !== null) {
            throw new Error('Session already started');
        }
        await this.ensureModelsEnv();
        const { enhanceFrame, resetEnhancer } = await this.setupEnhancer();
        const { audioContext, frameProcessNode } = await this.setupAudioPipeline();
        if (this.config.enableVAD) {
            await this.setupVAD(frameProcessNode, enhanceFrame, resetEnhancer);
        }
        else {
            this.setupDirectProcessing(frameProcessNode, enhanceFrame);
        }
        this.audioContext = audioContext;
    }
    async close() {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        ;
        this.audioContext.close();
        this.audioContext = null;
    }
    get muted() {
        return this._muted;
    }
    set muted(value) {
        this._muted = value;
    }
    setupDirectProcessing(frameProcessNode, enhanceFrame) {
        frameProcessNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                if (this.muted)
                    return;
                const frame = event.data.frame;
                const enhancedFrame = await enhanceFrame(frame);
                this.frameCallback(this.float32ToInt16(enhancedFrame));
            }
        };
    }
    async setupVAD(frameProcessNode, enhanceFrame, resetEnhancer) {
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
        const frameProcessorProcess = async (frame) => {
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
        const frameProcessor = new window.vad.FrameProcessor(frameProcessorProcess, frameProcessorReset, this.VAD_PARAMS.vadConfig, this.VAD_PARAMS.vadFrameSamples / this.config.sampleRate * 1000);
        const onFrameProcessorEvent = (ev) => {
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
        const frameQueue = [];
        let isProcessingFrameQueue = false;
        frameProcessNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                if (this.muted)
                    return;
                frameQueue.push(event.data.frame);
                if (isProcessingFrameQueue)
                    return;
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
    async setupAudioPipeline() {
        const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextCtor) {
            throw new Error('AudioContext is not supported in current browser');
        }
        const audioContext = new AudioContextCtor({ sampleRate: this.config.sampleRate });
        const inputStream = await this.requestMicrophoneStream();
        const sourceNode = audioContext.createMediaStreamSource(inputStream);
        await audioContext.audioWorklet.addModule(vad_processor_worklet_namespaceObject);
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
    async requestMicrophoneStream() {
        const constraints = {
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
        const nav = navigator;
        const legacyGetUserMedia = nav.getUserMedia || nav.webkitGetUserMedia || nav.mozGetUserMedia || nav.msGetUserMedia;
        if (typeof legacyGetUserMedia === 'function') {
            return new Promise((resolve, reject) => {
                legacyGetUserMedia.call(navigator, constraints, resolve, reject);
            });
        }
        throw new Error('getUserMedia is not supported in this browser');
    }
    async setupEnhancer() {
        const identityMap = async (frame) => frame;
        if (!this.config.enableEnhancer) {
            return { enhanceFrame: identityMap, resetEnhancer: () => { } };
        }
        try {
            const enhancerArrayBuffer = await fetch(fastenhancer_s_namespaceObject).then(r => r.arrayBuffer());
            const enhancerSession = await window.ort.InferenceSession.create(enhancerArrayBuffer);
            const enhancerCache = {
                'cache_in_0': new window.ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_1': new window.ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_2': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_3': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_4': new window.ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48])
            };
            let enhancerInputBuffer = [];
            let enhancerOutputBuffer = [];
            let isFirstEnhancerFrame = true;
            const enhanceFrame = async (frame) => {
                for (let i = 0; i < frame.length; i++) {
                    enhancerInputBuffer.push(frame[i]);
                }
                while (enhancerInputBuffer.length >= this.ENHANCER_PARAMS.hopSize) {
                    const chunk = enhancerInputBuffer.splice(0, this.ENHANCER_PARAMS.hopSize);
                    const chunkArray = new Float32Array(chunk);
                    const wavIn = new window.ort.Tensor('float32', chunkArray, [1, this.ENHANCER_PARAMS.hopSize]);
                    const inputs = { wav_in: wavIn };
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
                }
                else {
                    return frame;
                }
            };
            const resetEnhancer = () => {
                enhancerInputBuffer = [];
                enhancerOutputBuffer = [];
                isFirstEnhancerFrame = true;
                for (const cacheName of Object.keys(enhancerCache)) {
                    const shape = enhancerCache[cacheName].dims;
                    const zeros = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(0);
                    enhancerCache[cacheName] = new window.ort.Tensor('float32', zeros, shape);
                }
            };
            return { enhanceFrame, resetEnhancer };
        }
        catch {
            return { enhanceFrame: identityMap, resetEnhancer: () => { } };
        }
    }
    float32ToInt16(frame) {
        const int16 = new Int16Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const s = Math.max(-1, Math.min(1, frame[i]));
            int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        return int16.buffer;
    }
    async ensureModelsEnv() {
        if (!this.config.enableEnhancer && !this.config.enableVAD) {
            // No need to load models
            return;
        }
        // Inject window.ort and window.vad
        const inject = (src) => new Promise((resolve, reject) => {
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
function createPausableTimeout(callback, delay) {
    let timerId = null;
    let startTime = 0;
    let remaining = delay;
    let running = false;
    let cancelled = false;
    function start(ms) {
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
    function pause() {
        if (!running || timerId === null)
            return;
        clearTimeout(timerId);
        timerId = null;
        remaining -= Date.now() - startTime;
        running = false;
    }
    function resume() {
        if (running || cancelled || remaining <= 0)
            return;
        start(remaining);
    }
    function cancel() {
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
    constructor(config) {
        super();
        this.config = config;
        this.audioContext = null;
        this.audioBufferSources = [];
        this.audioTimeToPlay = 0;
        this.audioChunkStartedTimeouts = [];
        this.audioChunksPaused = [];
    }
    async open() {
        if (this.audioContext !== null) {
            throw new Error('Session already started');
        }
        this.audioContext = new window.AudioContext({ sampleRate: this.config.sampleRate });
        await this.audioContext.resume();
    }
    async close() {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        await this.stop();
        this.audioContext.close();
        this.audioContext = null;
        this.audioTimeToPlay = 0;
    }
    async pause() {
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
    async resume() {
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
    async stop() {
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
    async pushAudioChunk(pcm_chunk_int16) {
        if (!this.audioContext) {
            throw new Error('Session not started');
        }
        // If suspended, meaning that the session is paused; cache the incoming audio for future
        if (this.audioContext.state === 'suspended') {
            this.audioChunksPaused.push(pcm_chunk_int16);
            return;
        }
        const int16 = new Int16Array(pcm_chunk_int16);
        if (int16.length === 0)
            return;
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
            if (idx !== -1)
                this.audioBufferSources.splice(idx, 1);
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
        }
        else {
            const timeout = createPausableTimeout(() => {
                this.chunkStartedCallback(int16.buffer);
                // Remove this timeout from the list
                const idx = this.audioChunkStartedTimeouts.indexOf(timeout);
                if (idx !== -1)
                    this.audioChunkStartedTimeouts.splice(idx, 1);
            }, msForChunkStart);
            this.audioChunkStartedTimeouts.push(timeout);
        }
        // Update time to play for next chunk
        this.audioTimeToPlay += buffer.duration / source.playbackRate.value;
    }
}

;// ./src/websocket.ts




function createWebSocket(url, protocols) {
    switch (getPlatform()) {
        case Platform.Web:
        case Platform.MpWeixin:
            return new WebWebSocket(url, protocols);
        default:
            throw new Error("createWebSocket: Unknown platform");
    }
}

;// ./src/platforms/wx.ts


class WxInputAudioSession extends BaseInputAudioSession {
    constructor(config) {
        super();
        this._muted = false;
        this.recorder = null;
        this.started = false;
        this.speaking = false;
        this.targetFrameSize = 512;
        this.inputBuffer = [];
        this.outputBuffer = [];
        this.config = { ...config };
        this.nativeSampleRate = this.config.sampleRate;
        if (this.config.enableVAD === undefined) {
            this.config.enableVAD = false;
        }
        if (this.config.vadAmplitudeThreshold === undefined) {
            this.config.vadAmplitudeThreshold = 0.012;
        }
    }
    async open() {
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
        this.recorder.onFrameRecorded((res) => {
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
                const frame = frames[i];
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
    async close() {
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
    get muted() {
        return this._muted;
    }
    set muted(value) {
        if (value) {
            this.endSpeechIfNeeded();
        }
        this._muted = value;
    }
    processToFixedFrames(frameBuffer, nativeSampleRate) {
        this.nativeSampleRate = nativeSampleRate;
        const pcm = new Int16Array(frameBuffer);
        for (let i = 0; i < pcm.length; i++) {
            this.inputBuffer.push((pcm[i] ?? 0) / 32768);
        }
        const frames = [];
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
    resample(inputData) {
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
    float32ToInt16Buffer(frame) {
        const int16 = new Int16Array(frame.length);
        for (let i = 0; i < frame.length; i++) {
            const sample = Math.max(-1, Math.min(1, frame[i] ?? 0));
            int16[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
        }
        return int16.buffer;
    }
    startSpeechIfNeeded() {
        if (this.speaking) {
            return;
        }
        this.speaking = true;
        this.speechStartCallback();
    }
    endSpeechIfNeeded() {
        if (!this.speaking) {
            return;
        }
        this.speaking = false;
        this.speechEndCallback();
    }
    handleVad(frameBuffer) {
        const pcm = new Int16Array(frameBuffer);
        if (pcm.length === 0) {
            return;
        }
        let sum = 0;
        for (let i = 0; i < pcm.length; i++) {
            const v = pcm[i] / 32768;
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
    constructor(config) {
        super();
        this.player = null;
        this.opened = false;
        this.paused = false;
        this.queue = [];
        this.playing = false;
        this.currentBatch = [];
        this.config = { ...config };
        this.sampleRate = typeof config.sampleRate === "number" ? config.sampleRate : 48000;
    }
    async open() {
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
                this.chunkPlayedCallback(playedBatch[i]);
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
    async close() {
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
    async pause() {
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
    async resume() {
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
    async stop() {
        this.queue = [];
        this.currentBatch = [];
        this.playing = false;
        this.paused = false;
        if (this.player && typeof this.player.stop === "function") {
            this.player.stop();
        }
    }
    async pushAudioChunk(pcmChunkInt16) {
        if (!this.opened) {
            throw new Error("Session not started");
        }
        if (!(pcmChunkInt16 instanceof ArrayBuffer) || pcmChunkInt16.byteLength === 0) {
            return;
        }
        this.queue.push(pcmChunkInt16.slice(0));
        this.playNext();
    }
    playNext() {
        if (!this.opened || this.paused || this.playing || this.queue.length === 0 || !this.player) {
            return;
        }
        const batch = this.collectQueuedChunks();
        if (batch.length === 0) {
            return;
        }
        const mergedChunk = this.concatPcmChunks(batch);
        this.currentBatch = batch;
        this.chunkStartedCallback(batch[0]);
        const wavDataUri = this.pcmToWavDataUri(mergedChunk, this.sampleRate);
        this.player.src = wavDataUri;
        this.playing = true;
        this.player.play();
    }
    collectQueuedChunks() {
        const batch = [];
        while (this.queue.length > 0) {
            const chunk = this.queue.shift();
            if (chunk) {
                batch.push(chunk);
            }
        }
        return batch;
    }
    concatPcmChunks(chunks) {
        let totalBytes = 0;
        for (let i = 0; i < chunks.length; i++) {
            totalBytes += chunks[i].byteLength;
        }
        const merged = new Uint8Array(totalBytes);
        let offset = 0;
        for (let i = 0; i < chunks.length; i++) {
            const bytes = new Uint8Array(chunks[i]);
            merged.set(bytes, offset);
            offset += bytes.byteLength;
        }
        return merged.buffer;
    }
    pcmToWavDataUri(pcmChunk, sampleRate) {
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
        }
        else {
            const bytes = new Uint8Array(wavBuffer);
            let binary = "";
            for (let i = 0; i < bytes.length; i++) {
                binary += String.fromCharCode(bytes[i]);
            }
            base64 = btoa(binary);
        }
        return `data:audio/wav;base64,${base64}`;
    }
    writeAscii(view, offset, text) {
        for (let i = 0; i < text.length; i++) {
            view.setUint8(offset + i, text.charCodeAt(i));
        }
    }
}

;// ./src/audio-session.ts





function createInputAudioSession(config) {
    switch (getPlatform()) {
        case Platform.Web:
            return new WebInputAudioSession(config);
        case Platform.MpWeixin:
            return new WxInputAudioSession(config);
        default:
            throw new Error("createInputAudioSession: Unknown platform");
    }
}
function createOutputAudioSession(config) {
    switch (getPlatform()) {
        case Platform.Web:
            return new WebOutputAudioSession(config);
        case Platform.MpWeixin:
            return new WxOutputAudioSession(config);
        default:
            throw new Error("createOutputAudioSession: Unknown platform");
    }
}

;// ./src/conversation.ts

function defaultConversation() {
    return {
        streamState: "idle",
        sessionId: null,
        latency: {},
        messages: [],
        thought: "",
        caption: "",
        retrieval: "",
    };
}
class Conversation {
    constructor() {
        this._state = defaultConversation();
        this.stateChangeCallback = () => { };
        this.fullAudioChunkCallback = (_chunk, _sr) => { };
    }
    onStateChange(callback) {
        callback(this._state);
        this.stateChangeCallback = callback;
    }
    onFullAudioChunk(callback) {
        this.fullAudioChunkCallback = callback;
    }
    get state() {
        return new Proxy(this._state, {
            set: (target, key, value) => {
                target[key] = value;
                this.stateChangeCallback(target);
                return true;
            },
            get: (target, key) => {
                return key in target ? target[key] : undefined;
            }
        });
    }
    appendMessage(message) {
        // If is an info, directly append
        if (message.role === "info") {
            this.state.messages.push(message);
            this.stateChangeCallback(this._state);
            return;
        }
        // Find the latest message with same role and turnId to replace
        for (let i = this.state.messages.length - 1; i >= 0; i--) {
            const msg = this.state.messages[i];
            if (msg.role === message.role && msg.turnId === message.turnId) {
                msg.content = message.content;
                // If last message is an info, put that message in front of the updated message
                const lastMsg = this.state.messages[this.state.messages.length - 1];
                if (lastMsg.role === "info") {
                    this.state.messages.splice(this.state.messages.length - 1, 1);
                    this.state.messages.splice(i, 0, lastMsg);
                }
                this.stateChangeCallback(this._state);
                return;
            }
        }
        // Otherwise, add as new message
        this.state.messages.push(message);
        this.stateChangeCallback(this._state);
    }
    updateLatency(latency) {
        this.state.latency = { ...latency };
    }
    emitFullAudioChunk(pcmChunkInt16, sampleRate) {
        this.fullAudioChunkCallback(pcmChunkInt16, sampleRate);
    }
}

;// ./src/action-handler-functions/utils.ts

const onVadSpeechStart = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'listening';
};
const onVadSpeechEnd = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'processing';
};

;// ./src/action-handler-functions/client.ts

const clientMap = {
    "client_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_start" });
    },
    "client_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_end" });
    },
    "client_audio_chunk_started": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = 'speaking';
    },
    "client_audio_playback_finished": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = 'idle';
        websocket.sendJson({ action: "tts_playback_finished" });
    },
    "client_audio_chunk_played": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({ action: "tts_chunk_played" });
    }
};
/* harmony default export */ const client = (clientMap);

;// ./src/action-handler-functions/messages.ts
const messagesMap = {
    "update_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "user",
            content: data.text,
            turnId: data.turn_id
        });
    },
    "finish_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "user",
            content: data.text,
            turnId: data.turn_id
        });
    },
    "update_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            turnId: data.turn_id
        });
    },
    "finish_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            turnId: data.turn_id
        });
    },
};
/* harmony default export */ const messages = (messagesMap);

;// ./src/action-handler-functions/output.ts
const outputMap = {
    "start_tts": async (data, websocket, conversation, outputAudioSession) => {
        // Leave blank, no use
    },
    "pause_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "stop_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.stop();
    },
    "resume_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
};
/* harmony default export */ const output = (outputMap);

;// ./src/action-handler-functions/session.ts
const sessionMap = {
    "session_info": async (data, websocket, conversation, outputAudioSession) => {
        const sid = data.session_id || null;
        conversation.state.sessionId = sid;
    },
};
/* harmony default export */ const session = (sessionMap);

;// ./src/action-handler-functions/client-operations.ts
const clientOperationMap = {
    "client_change_voice": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            action: "change_voice",
            voice_name: data.voiceName,
        });
    },
    "client_upload_file": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = "processing";
        const file = data.file;
        const endpoint = data.endpoint;
        const formData = new FormData();
        formData.append("session_id", conversation.state.sessionId);
        formData.append("file", file);
        const resp = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });
        if (!resp.ok) {
            conversation.state.streamState = "idle";
        }
    },
};
/* harmony default export */ const client_operations = (clientOperationMap);

;// ./src/action-handler-functions/meta.ts
const metaMap = {
    "thought_updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.thought = data.text;
    },
    "caption_updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.caption = data.text;
    },
    "retrieval_updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.retrieval = data.text;
    },
};
/* harmony default export */ const meta = (metaMap);

;// ./src/action-handler-functions/latency.ts
const latencyMap = {
    "latency_metrics": async (data, websocket, conversation, outputAudioSession) => {
        conversation.updateLatency({
            network: Number(data.network_latency_ms) || 0,
            asr: Number(data.asr_latency_ms) || 0,
            llmFirstToken: Number(data.llm_first_token_ms) || 0,
            llmSentence: Number(data.llm_sentence_ms) || 0,
            ttsFirstChunk: Number(data.tts_first_chunk_ms) || 0,
        });
    },
};
/* harmony default export */ const latency = (latencyMap);

;// ./src/action-handler-functions/input.ts

function decodeBase64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
}
const inputMap = {
    "vad_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
    },
    "vad_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
    },
    "full_audio_frame": async (data, websocket, conversation, outputAudioSession) => {
        const audioBase64 = typeof data?.audio_base64 === "string" ? data.audio_base64 : "";
        if (!audioBase64) {
            return;
        }
        const sampleRate = typeof data?.sample_rate === "number" ? data.sample_rate : 48000;
        const pcmChunkInt16 = decodeBase64ToArrayBuffer(audioBase64);
        conversation.emitFullAudioChunk(pcmChunkInt16, sampleRate);
    }
};
/* harmony default export */ const input = (inputMap);

;// ./src/action-handler-functions/index.ts

const ACTION_TO_FUNCTION = {};
function registerMap(partial_map) {
    for (const key in partial_map) {
        if (partial_map[key]) {
            ACTION_TO_FUNCTION[key] = partial_map[key];
        }
    }
}
// ------------ Import and register action handler functions here ------------








registerMap(client);
registerMap(messages);
registerMap(output);
registerMap(session);
registerMap(client_operations);
registerMap(meta);
registerMap(latency);
registerMap(input);

;// ./src/action-handler.ts





class ActionHandler {
    constructor() {
        this.ACTION_TO_FUNCTION = ACTION_TO_FUNCTION;
    }
    async handleAction(action, data, websocket, conversation, outputAudioSession) {
        const handler = this.ACTION_TO_FUNCTION[action];
        if (handler) {
            await handler(data, websocket, conversation, outputAudioSession);
        }
        else {
            throw new Error(`No handler found for action: ${action}`);
        }
    }
}

;// ./src/core.ts





function createSession(websocketURL, { inputConfig = {}, outputConfig = {}, } = {}) {
    const resolvedInputConfig = {
        sampleRate: 16000,
        ...inputConfig,
    };
    const resolvedOutputConfig = {
        sampleRate: 48000,
        ...outputConfig,
    };
    const conversation = new Conversation();
    const actionHandler = new ActionHandler();
    let websocket;
    let inputAudioSession;
    let outputAudioSession;
    let inputAudioChunkCallback = (_chunk, _sr) => { };
    let outputAudioChunkCallback = (_chunk, _sr) => { };
    function initialize() {
        websocket = createWebSocket(websocketURL);
        inputAudioSession = createInputAudioSession(resolvedInputConfig);
        outputAudioSession = createOutputAudioSession(resolvedOutputConfig);
        // Subscribe actions and audio chunks
        websocket.addEventListener("message", async (event) => {
            if (typeof event.data === "string") {
                const message = JSON.parse(event.data);
                try {
                    await actionHandler.handleAction(message.action, message.data, websocket, conversation, outputAudioSession);
                }
                catch (error) {
                    //TODO: Handle unknown action error
                }
            }
            else if (event.data instanceof ArrayBuffer) {
                await outputAudioSession.pushAudioChunk(event.data);
            }
        });
        // Bind audio input handling
        inputAudioSession.onFrame(async (audioChunk) => {
            inputAudioChunkCallback(audioChunk, resolvedInputConfig.sampleRate);
            websocket.sendAudioChunk(audioChunk);
        });
        inputAudioSession.onSpeechStart(async () => {
            await actionHandler.handleAction("client_speech_start", null, websocket, conversation, outputAudioSession);
        });
        inputAudioSession.onSpeechEnd(async () => {
            await actionHandler.handleAction("client_speech_end", null, websocket, conversation, outputAudioSession);
        });
        // Bind audio output handling
        outputAudioSession.onChunkStarted(async (audioChunk) => {
            outputAudioChunkCallback(audioChunk, resolvedOutputConfig.sampleRate);
            await actionHandler.handleAction("client_audio_chunk_started", null, websocket, conversation, outputAudioSession);
        });
        outputAudioSession.onChunkPlayed(async (_audioChunk) => {
            await actionHandler.handleAction("client_audio_chunk_played", null, websocket, conversation, outputAudioSession);
        });
        outputAudioSession.onAllChunksPlayed(async () => {
            await actionHandler.handleAction("client_audio_playback_finished", null, websocket, conversation, outputAudioSession);
        });
    }
    // Create API for external use
    const session = {
        open: async () => {
            initialize();
            await inputAudioSession.open();
            await outputAudioSession.open();
        },
        close: async () => {
            await inputAudioSession.close();
            await outputAudioSession.close();
            websocket.close();
        },
        onStateChange: (callback) => {
            conversation.onStateChange(callback);
        },
        get state() {
            return conversation.state;
        },
        onInputAudioChunk: (callback) => {
            inputAudioChunkCallback = callback;
        },
        onOutputAudioChunk: (callback) => {
            outputAudioChunkCallback = callback;
        },
        onFullAudioChunk: (callback) => {
            conversation.onFullAudioChunk(callback);
        },
        get muted() {
            return inputAudioSession.muted;
        },
        set muted(value) {
            inputAudioSession.muted = value;
        },
        async changeVoice(voiceName) {
            await actionHandler.handleAction("client_change_voice", { voiceName }, websocket, conversation, outputAudioSession);
        },
        async uploadFile(file, endpoint = "./api/upload") {
            await actionHandler.handleAction("client_upload_file", { file, endpoint }, websocket, conversation, outputAudioSession);
        }
    };
    return session;
}

;// ./src/index.ts



export { createSession };
