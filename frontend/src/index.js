import fastEnhancerOnnxUrl from "../models/fastenhancer_s.onnx";
import vadProcessorUrl from "./vad-processor.worklet.js";
// Create low-level audio/WebSocket session.
//
// This file now includes on-demand ORT/VAD loading logic:
// - Removed script injection and ensureOrtVad from the HTML template;
// - Paths that require VAD call ensureOrtVad() the first time recording starts
//   to make sure window.ort and window.vad are available;
// - Pure front-end mode (pure_frontend) skips loading to avoid extra payload and requests.

// ------------------------------
// ORT/VAD Loader (lazy load)
// ------------------------------
let __ensureOrtVadPromise = null;
/**
 * Ensure onnxruntime-web and @ricky0123/vad-web are loaded.
 * - Use ORT 1.17.0 on iOS devices and 1.22.0 elsewhere to match the old HTML injection logic.
 * - Multiple calls only trigger one network fetch.
 */
async function ensureOrtVad() {
    // Exit early when already loaded
    if (window.ort && window.vad) return;
    // TODO: remove this redundant logic
    if (__ensureOrtVadPromise) {
        await __ensureOrtVadPromise; return;
    }

    // Pick ORT version by UA (only iOS stays on 1.17.0)
    const isIOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
    const ver = isIOS ? '1.17.0' : '1.22.0';
    // Remember the version so downstream code can reuse it for wasm paths
    try { window.__ortVersion = ver; } catch (_) { }

    const inject = (src) => new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = src;
        s.onload = () => resolve();
        s.onerror = (e) => reject(e);
        document.head.appendChild(s);
    });

    __ensureOrtVadPromise = (async () => {
        if (!window.ort) {
            await inject(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ver}/dist/ort.js`);
        }
        if (!window.vad) {
            await inject('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.27/dist/bundle.min.js');
        }
    })();

    try {
        await __ensureOrtVadPromise;
    } finally {
        // Release the reference so failures do not block retries; success returns immediately next time
        __ensureOrtVadPromise = null;
    }
}

// The second argument accepts a boolean or options object:
// - A boolean maps to { simGen: <bool> }
// - Objects may include { simGen?: bool, sim_gen?: bool, pureFrontend?: bool, pure_frontend?: bool }
//   pureFrontend = true enables "pure front-end mode": skip VAD/enhancer and only capture+forward raw audio.
// simGen: when true, never auto-stop/pause TTS except via toggleStreaming().
function createAudioSession(onIncomingJson, websocketURL = null, opts = null) {
    const scriptUrl = new URL(import.meta.url);
    if (typeof onIncomingJson !== 'function') {
        throw new Error('onIncomingJson must be a function');
    }
    // Back-compat: boolean second argument is treated as simGen
    const simGen = opts?.simGen ?? false;
    const pureFrontend = opts?.pureFrontend ?? false;
    // Server PCM sample rate
    const TTS_SAMPLE_RATE = opts?.ttsSampleRate ?? 48000;
    let ws = null;
    // Whether to suppress the "Lost connection" notice on the next WS close (one-shot)
    let suppressNextCloseLog = false;
    let vad = null; // In pure front-end mode this encapsulates the raw mic capture lifecycle
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    let audioCtx = null;
    let newAudioOutputCallback = null;
    function onNewAudioOutput(cb) {
        newAudioOutputCallback = cb;
    }
    const playingSources = [];
    // Frontend TTS playback rate (local playback only, server synthesis unaffected), default 1.0x
    // The UI adjusts via convo.changeTTSSpeed(x); recommended range [0.5, 1.5]
    let ttsPlaybackRate = 1.0;
    const audioQueue = [];
    let streaming = false;
    let pendingTTSStreamFinished = false;
    // Whether mic input is muted (no capture/report)
    let micMuted = false;

    // Simple frontend log forwarding so model-related logs appear as conversation messages
    // Note: log entries use English for easier debugging and search
    const modelLog = (msg) => {
        try { onIncomingJson && onIncomingJson({ action: 'client_log', data: msg }); } catch (_) { }
    };

    // Heartbeat management
    let heartbeatInterval = null;
    let heartbeatTimeout = null;
    let missedHeartbeats = 0;
    const HEARTBEAT_INTERVAL_MS = 15000;
    const HEARTBEAT_TIMEOUT_MS = 10000;
    const MAX_MISSED_HEARTBEATS = 3;

    // Clock synchronization
    let pingTimestamp = 0; // Timestamp when ping was sent
    let clockOffset = null; // Clock offset (ms): client_time - server_time
    let clockSyncSamples = []; // Clock offset samples
    const MAX_CLOCK_SYNC_SAMPLES = 5; // Keep the latest 5 samples 

    // TTS stream flags
    let ttsStreamActive = false;
    let ttsStreamFinished = false;

    let nextScheduledTime = 0;
    // ===========================================

    // Audio context local sample rate
    const LOCAL_SAMPLE_RATE = 44100;

    // VAD params
    const NEGATIVE_SPEECH_THRESHOLD = 0.2;
    const NEGATIVE_FRAMES_BEFORE_END = 50;

    // AudioContext helper and streaming resampler state
    function ensureAudioCtx() {
        if (!audioCtx) {
            try {
                audioCtx = new AudioContext({ sampleRate: LOCAL_SAMPLE_RATE });
            } catch (_e) {
                audioCtx = new AudioContext();
            }
        }
    }

    // Use a stateless linear resampler per chunk
    function resampleChunkPCM(src, fromRate, toRate) {
        if (!src || src.length === 0) return src || new Float32Array(0);
        if (fromRate === toRate) return src;
        const outLen = Math.max(1, Math.round(src.length * toRate / fromRate));
        const out = new Float32Array(outLen);
        const ratio = fromRate / toRate; // input samples per output sample
        for (let j = 0; j < outLen; j++) {
            const pos = j * ratio;
            const i = Math.floor(pos);
            const frac = pos - i;
            const s0 = src[i] ?? 0;
            const i1 = i + 1 < src.length ? i + 1 : i;
            const s1 = src[i1] ?? s0;
            out[j] = s0 + (s1 - s0) * frac;
        }
        return out;
    }

    // Poll timer for "wait for next chunk" when active
    let playPollTimer = null;

    // Unified send helpers
    function sendJson(obj) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send(JSON.stringify(obj)); } catch (_e) { }
        }
    }
    function sendPCM(int16) {
        if (ws && ws.readyState === WebSocket.OPEN) {
            try { ws.send(int16.buffer); } catch (_e) { }
        }
    }

    // Heartbeat functions
    function startHeartbeat() {
        stopHeartbeat();
        missedHeartbeats = 0;

        console.log('[Heartbeat] Starting heartbeat mechanism');

        heartbeatInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                console.log('[Heartbeat] Sending ping');
                pingTimestamp = Date.now(); // Record send time
                sendJson({ action: 'ping', timestamp: pingTimestamp });

                if (heartbeatTimeout) clearTimeout(heartbeatTimeout);
                heartbeatTimeout = setTimeout(() => {
                    missedHeartbeats++;
                    console.warn(`[Heartbeat] Pong timeout! Missed heartbeats: ${missedHeartbeats}/${MAX_MISSED_HEARTBEATS}`);

                    if (missedHeartbeats >= MAX_MISSED_HEARTBEATS) {
                        console.error('[Heartbeat] Too many missed heartbeats, reconnecting...');
                        handleHeartbeatFailure();
                    }
                }, HEARTBEAT_TIMEOUT_MS);
            }
        }, HEARTBEAT_INTERVAL_MS);
    }

    function stopHeartbeat() {
        if (heartbeatInterval) {
            clearInterval(heartbeatInterval);
            heartbeatInterval = null;
        }
        if (heartbeatTimeout) {
            clearTimeout(heartbeatTimeout);
            heartbeatTimeout = null;
        }
        missedHeartbeats = 0;
        console.log('[Heartbeat] Stopped heartbeat mechanism');
    }

    function onPongReceived(pongData) {
        if (heartbeatTimeout) {
            clearTimeout(heartbeatTimeout);
            heartbeatTimeout = null;
        }
        missedHeartbeats = 0;

        // Clock sync: compute offset
        // Use server_recv_timestamp (server received ping) for more accurate NTP sync
        const serverRecvTs = pongData && (pongData.server_recv_timestamp || pongData.server_timestamp);
        if (serverRecvTs && pingTimestamp > 0) {
            const clientRecvTs = Date.now(); // Client receives pong
            const clientSendTs = pingTimestamp; // Client sent ping

            // Compute round-trip time (RTT)
            const rtt = clientRecvTs - clientSendTs;

            // Compute clock offset (NTP formula)
            // offset = server_recv_time - (client_send_time + client_recv_time) / 2
            // Positive offset => server clock ahead of client; negative => server behind
            const offset = serverRecvTs - (clientSendTs + clientRecvTs) / 2;

            // Store the sample
            clockSyncSamples.push(offset);
            if (clockSyncSamples.length > MAX_CLOCK_SYNC_SAMPLES) {
                clockSyncSamples.shift();
            }

            // Use the median to stabilize the offset
            if (clockSyncSamples.length >= 3) {
                const sorted = [...clockSyncSamples].sort((a, b) => a - b);
                clockOffset = sorted[Math.floor(sorted.length / 2)];
            } else {
                clockOffset = offset;
            }

            console.log(`[Clock Sync] RTT: ${rtt.toFixed(1)}ms, Offset: ${clockOffset.toFixed(1)}ms (samples: ${clockSyncSamples.length})`);

            // Notify the backend about these clock sync samples
            sendJson({
                action: 'clock_sync',
                client_send_ts: clientSendTs / 1000, // Convert to seconds
                server_recv_ts: serverRecvTs / 1000, // Convert to seconds
                client_recv_ts: clientRecvTs / 1000 // Convert to seconds
            });
        }

        console.log('[Heartbeat] Received pong');
    }

    function handleHeartbeatFailure() {
        stopHeartbeat();
        if (ws) {
            console.log('[Heartbeat] Closing dead connection');
            ws.close();
            ws = null;
        }

        onIncomingJson({
            action: 'error',
            data: 'Connection lost. Please refresh the page or restart streaming.'
        });
    }


    // Init Silero VAD manually (default mode)
    async function initVAD() {
        // Lazily load ORT/VAD (moved here from HTML)
        await ensureOrtVad();
        // Counter for consecutive non-speech frames (enabled at SpeechStart, disabled/reset after triggering)
        let negEndCounterEnabled = false;
        let negEndCounter = 0;

        // Check VAD lib (if still missing it's likely a network/CDN failure)
        if (!window.vad || !window.vad.FrameProcessor) {
            throw new Error('VAD library not available after ensureOrtVad(). Check network/CDN access.');
        }

        function sendFrame(frame) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const int16 = new Int16Array(frame.length);
            for (let i = 0; i < frame.length; i++) {
                const s = Math.max(-1, Math.min(1, frame[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            sendPCM(int16);
        }

        function onSpeechEnd() {
            sendJson({ action: "vad_speech_end", timestamp: Date.now() });
            onIncomingJson({ action: 'client_vad_speech_end', data: { timestamp: Date.now() } });
        }

        // Initialize ONNX Runtime
        const ort = window.ort;

        // Optional fast enhancer state (falls back to passthrough on failure)
        let useEnhancer = false;
        // Default enhancer function: passthrough
        let enhanceSpeech = async (audioFrame) => audioFrame; // No enhancement on failure

        // Variables needed when the enhancer is enabled so reset/cleanup can access them
        let ENHANCER_HOP_SIZE = 256;
        let ENHANCER_N_FFT = 512;
        let enhancerSession = null;
        let enhancerCache = null;
        let enhancerInputBuffer = [];
        let enhancerOutputBuffer = [];
        let isFirstEnhancerFrame = true;
        let enhancementErrorNotified = false; // Only notify the session once about enhancer errors

        try {
            // Throw if ORT or its APIs are missing so fallback logic runs
            if (!ort || !ort.env || !ort.InferenceSession) {
                throw new Error('onnxruntime-web not available');
            }

            // Set ORT wasm path: only iOS stays on 1.17.0, others use 1.22.0
            // HTML injection is no longer needed; ensureOrtVad() already set window.__ortVersion.
            const isIOSUA = /iPhone|iPad|iPod/i.test(navigator.userAgent);
            const ortVersion = (window.__ortVersion) || (isIOSUA ? '1.17.0' : '1.22.0');
            ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/`;

            // Show FastEnhancer loading info inside the conversation
            const enhancerArrayBuffer = await fetch(fastEnhancerOnnxUrl).then(r => r.arrayBuffer());
            enhancerSession = await ort.InferenceSession.create(enhancerArrayBuffer);

            // FastEnhancer parameters
            ENHANCER_HOP_SIZE = 256;  // Samples per frame
            ENHANCER_N_FFT = 512;     // Window size

            enhancerCache = {
                'cache_in_0': new ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_1': new ort.Tensor('float32', new Float32Array(1 * 256).fill(0), [1, 256]),
                'cache_in_2': new ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_3': new ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48]),
                'cache_in_4': new ort.Tensor('float32', new Float32Array(1 * 36 * 48).fill(0), [1, 36, 48])
            };

            // Buffers for accumulating input/output
            enhancerInputBuffer = [];
            enhancerOutputBuffer = [];
            isFirstEnhancerFrame = true;

            modelLog('Audio enhancer loaded.');

            // Enable the enhancer
            useEnhancer = true;

            // Speech enhancement function (adapted from the Python version)
            enhanceSpeech = async (audioFrame) => {
                try {
                    // Push into the input buffer
                    for (let i = 0; i < audioFrame.length; i++) {
                        enhancerInputBuffer.push(audioFrame[i]);
                    }

                    // Process data in hop_size chunks
                    while (enhancerInputBuffer.length >= ENHANCER_HOP_SIZE) {
                        const chunk = enhancerInputBuffer.splice(0, ENHANCER_HOP_SIZE);
                        const chunkArray = new Float32Array(chunk);

                        // wav_in: [1, hop_size]
                        const wavIn = new ort.Tensor('float32', chunkArray, [1, ENHANCER_HOP_SIZE]);

                        // Build inputs including cache
                        const inputs = { wav_in: wavIn };
                        for (const inputName of Object.keys(enhancerCache)) {
                            inputs[inputName] = enhancerCache[inputName];
                        }

                        // Run inference
                        const outputs = await enhancerSession.run(inputs);

                        // First output is the enhanced waveform; the rest update caches
                        const outputNames = enhancerSession.outputNames;
                        const enhancedChunk = outputs[outputNames[0]].data;

                        for (let i = 1; i < outputNames.length; i++) {
                            const cacheName = `cache_in_${i - 1}`;
                            enhancerCache[cacheName] = outputs[outputNames[i]];
                        }

                        // Write into the output buffer
                        for (let i = 0; i < enhancedChunk.length; i++) {
                            enhancerOutputBuffer.push(enhancedChunk[i]);
                        }

                        // Drop the overlapping section from the first frame
                        if (isFirstEnhancerFrame && enhancerOutputBuffer.length >= (ENHANCER_N_FFT - ENHANCER_HOP_SIZE)) {
                            enhancerOutputBuffer.splice(0, ENHANCER_N_FFT - ENHANCER_HOP_SIZE);
                            isFirstEnhancerFrame = false;
                        }
                    }

                    // Return enhanced audio with the same length or fall back to original if insufficient
                    if (enhancerOutputBuffer.length >= audioFrame.length) {
                        const output = enhancerOutputBuffer.splice(0, audioFrame.length);
                        return new Float32Array(output);
                    } else {
                        return audioFrame;
                    }
                } catch (error) {
                    // Log errors to console and notify the conversation once
                    console.error('Enhancement error:', error);
                    if (!enhancementErrorNotified) {
                        modelLog('Enhancement error. Falling back to raw audio.');
                        enhancementErrorNotified = true;
                    }
                    // Fall back to raw audio on error
                    return audioFrame;
                }
            };
        } catch (e) {
            // Enhancer load failed: continue without it.
            console.error('FastEnhancer initialization failed, disabling enhancer.', e);
            modelLog('Enhancer initialization failed and disabled. Maybe you are using a mobile device.');
            useEnhancer = false;
            enhanceSpeech = async (audioFrame) => audioFrame; // Passthrough
        }

        // Load Silero VAD model (always v5)
        const vadURL = 'https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.27/dist/silero_vad_v5.onnx';
        const vadArrayBuffer = await fetch(vadURL).then(r => r.arrayBuffer());
        const vadSession = await ort.InferenceSession.create(vadArrayBuffer);

        // Create VAD model state for V5
        const stateZeros = Array(2 * 128).fill(0);
        let state = new ort.Tensor('float32', stateZeros, [2, 1, 128]);
        const sr = new ort.Tensor('int64', [16000n]);

        modelLog('VAD loaded.');

        // Store enhanced frames temporarily
        let lastEnhancedFrame = null;

        // VAD model process function (with enhancement)
        const modelProcess = async (frame) => {
            // Step 1: Enhance audio using FastEnhancer
            const enhancedFrame = await enhanceSpeech(frame);

            // Store enhanced frame for transmission
            lastEnhancedFrame = enhancedFrame;

            // Step 2: Run VAD on enhanced audio (fixed v5 I/O names)
            const audioTensor = new ort.Tensor('float32', enhancedFrame, [1, enhancedFrame.length]);
            const inputs = { input: audioTensor, state, sr };
            const out = await vadSession.run(inputs);
            state = out.stateN;
            const isSpeech = out.output.data[0];
            return { isSpeech, notSpeech: 1 - isSpeech };
        };

        // Model reset function
        const modelReset = () => {
            // Reset VAD state
            state = new ort.Tensor('float32', stateZeros, [2, 1, 128]);

            // Also reset enhancer buffers/caches when enabled
            if (useEnhancer && enhancerCache) {
                enhancerInputBuffer = [];
                enhancerOutputBuffer = [];
                isFirstEnhancerFrame = true;

                for (const cacheName of Object.keys(enhancerCache)) {
                    const shape = enhancerCache[cacheName].dims;
                    const zeros = new Float32Array(shape.reduce((a, b) => a * b, 1)).fill(0);
                    enhancerCache[cacheName] = new ort.Tensor('float32', zeros, shape);
                }
            }
        };

        // Frame size for V5 model
        const frameSamples = 512;
        const msPerFrame = frameSamples / 16; // 16kHz sample rate


        // Create FrameProcessor
        const frameProcessor = new window.vad.FrameProcessor(
            modelProcess,
            modelReset,
            {
                positiveSpeechThreshold: 0.8,
                negativeSpeechThreshold: NEGATIVE_SPEECH_THRESHOLD,
                preSpeechPadMs: 30,
                redemptionMs: 500,
                minSpeechMs: 250,
                submitUserSpeechOnPause: false
            },
            msPerFrame
        );

        // Handle frame processor events
        const handleFrameProcessorEvent = (ev) => {
            switch (ev.msg) {
                case window.vad.Message.FrameProcessed:
                    const frame = ev.frame; // Original frame from input
                    // Use consecutive notSpeech samples to trigger early speech end
                    if (negEndCounterEnabled) {
                        const ns = Number(ev?.probs?.notSpeech ?? 0);
                        const nsHigh = ns > (1 - NEGATIVE_SPEECH_THRESHOLD);
                        negEndCounter = nsHigh ? (negEndCounter + 1) : 0;
                        if (negEndCounter > NEGATIVE_FRAMES_BEFORE_END) {
                            // Trigger speech end and disable/reset the counter
                            onSpeechEnd();
                            negEndCounterEnabled = false;
                            negEndCounter = 0;
                        }
                    }
                    // const enhancedFrame = lastEnhancedFrame || frame; // Use enhanced frame if available

                    // Only send frames to the backend when not muted
                    if (!micMuted) {
                        sendFrame(frame);
                    }
                    break;

                case window.vad.Message.SpeechStart:
                    // Pause TTS on barge-in unless in simultaneous generation mode
                    if (!simGen) {
                        pauseTTSPlayback();
                    }

                    // Enable the negative sample counter when speech starts
                    negEndCounterEnabled = true;
                    negEndCounter = 0;

                    // Notify server
                    const nowTs = Date.now();
                    sendJson({ action: "vad_speech_start", timestamp: nowTs });
                    onIncomingJson({ action: 'client_vad_speech_start', data: { timestamp: nowTs } });
                    break;

                case window.vad.Message.SpeechEnd:
                    // Disable/reset the counter when VAD reports speech end
                    negEndCounterEnabled = false;
                    negEndCounter = 0;
                    onSpeechEnd();
                    break;
            }
        };

        // Get microphone stream
        // TODO: add another input stream for ASR with echoCancellation and noiceSuppression?
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: false
            }
        });

        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

        // Load AudioWorklet processor
        await audioContext.audioWorklet.addModule(vadProcessorUrl);

        const sourceNode = audioContext.createMediaStreamSource(stream);

        // Create AudioWorkletNode with resampling built-in
        const workletNode = new AudioWorkletNode(audioContext, 'vad-processor', {
            processorOptions: {
                targetSampleRate: 16000,
                targetFrameSize: frameSamples
            }
        });

        // Queue for frame processing to avoid dropping frames
        const frameQueue = [];
        let isProcessingQueue = false;

        // Process frames from queue sequentially
        const processFrameQueue = async () => {
            if (isProcessingQueue) return;
            isProcessingQueue = true;

            while (frameQueue.length > 0) {
                const frame = frameQueue.shift();
                try {
                    await frameProcessor.process(frame, handleFrameProcessorEvent);
                } catch (error) {
                    console.error('Error processing audio frame:', error);
                }
            }

            isProcessingQueue = false;
        };

        // Handle audio data from worklet (already resampled to 16kHz)
        workletNode.port.onmessage = async (event) => {
            if (event.data.type === 'audioFrame') {
                // Drop frames immediately while muted without VAD processing
                if (micMuted) return;
                frameQueue.push(event.data.frame);
                processFrameQueue();
            }
        };

        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        sourceNode.connect(workletNode);
        workletNode.connect(silentGain);
        silentGain.connect(audioContext.destination);

        // Start frame processor
        frameProcessor.resume();

        // Store VAD instance
        vad = {
            stream,
            audioContext,
            sourceNode,
            workletNode,
            frameProcessor,
            // Mute toggle: pause/resume VAD and discard frames while muted
            setMuted(flag) {
                micMuted = !!flag;
            },
            async stop() {
                // Clear pending frames before stopping
                frameQueue.length = 0;

                frameProcessor.pause();
                workletNode.disconnect();
                sourceNode.disconnect();
                stream.getTracks().forEach(track => track.stop());
                await audioContext.close();
            }
        };
    }

    // Pure front-end mode: initialize raw mic capture (resample + forward, no VAD/enhancer)
    async function initRawMic() {
        // Note: reuse the vad-processor resampler to stay aligned with the server's 16 kHz sample rate
        // But skip onnxruntime/vad-web entirely and stream frames continuously without boundaries.
        const frameSamples = 512; // Match VAD frame size so the backend buffer stays aligned

        function sendFrame(frame) {
            if (!ws || ws.readyState !== WebSocket.OPEN) return;
            const int16 = new Int16Array(frame.length);
            for (let i = 0; i < frame.length; i++) {
                const s = Math.max(-1, Math.min(1, frame[i]));
                int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
            }
            sendPCM(int16);
        }

        // Microphone media stream
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                autoGainControl: true,
                noiseSuppression: false
            }
        });

        // Audio context keeps the processing graph silent to avoid playback
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        await audioContext.audioWorklet.addModule(new URL("./vad-processor.js", scriptUrl).toString()
        );
        const sourceNode = audioContext.createMediaStreamSource(stream);
        const workletNode = new AudioWorkletNode(audioContext, 'vad-processor', {
            processorOptions: { targetSampleRate: 16000, targetFrameSize: frameSamples }
        });

        // Silent gain node keeps the graph active
        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        sourceNode.connect(workletNode);
        workletNode.connect(silentGain);
        silentGain.connect(audioContext.destination);

        // Send frames straight to the backend
        workletNode.port.onmessage = (event) => {
            if (event.data?.type === 'audioFrame') {
                if (micMuted) return; // Drop when muted
                sendFrame(event.data.frame);
            }
        };

        // Store a pseudo VAD handle so existing stop/mute management can be reused
        vad = {
            stream,
            audioContext,
            sourceNode,
            workletNode,
            setMuted(flag) { micMuted = !!flag; },
            async stop() {
                try { workletNode.disconnect(); } catch (_) { }
                try { sourceNode.disconnect(); } catch (_) { }
                try { stream.getTracks().forEach(t => t.stop()); } catch (_) { }
                try { await audioContext.close(); } catch (_) { }
            }
        };
    }

    function initWebSocket(onMessage = handleIncomingData) {
        if (typeof onMessage !== 'function') {
            throw new Error('onMessage must be a function');
        }
        if (ws) {
            console.log('[WS] Closing existing WebSocket before creating new one');
            // Mark this closure as expected to avoid showing disconnect prompts
            suppressNextCloseLog = true;
            stopHeartbeat();
            ws.close();
            ws = null;
        }
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';

        const wsPath = new URL('../../ws', scriptUrl);

        wsPath.protocol = wsProtocol;
        wsPath.host = window.location.host;

        ws = new WebSocket(websocketURL || wsPath.toString());
        ws.binaryType = 'arraybuffer';


        // Add event listeners for debugging
        ws.addEventListener('open', () => {
            console.log('[WS] WebSocket opened');
            startHeartbeat();
        });

        ws.addEventListener('close', (event) => {
            // Only show the notice when the disconnect was unexpected
            if (!suppressNextCloseLog) {
                modelLog('Websocket Lost connection.');
            }
            suppressNextCloseLog = false;
            stopHeartbeat();
        });

        ws.addEventListener('error', (event) => {
            console.error('[WS] WebSocket error', event);
            stopHeartbeat();
        });

        ws.addEventListener('message', (event) => {
            onMessage(event);
        });
    }

    function enableAllPlayback() {
        // Guard when stream marked finished/reset
        if (ttsStreamFinished) return;

        ensureAudioCtx();

        playNextAudio();
    }

    function pauseTTSPlayback() {
        // TODO: pause AudioContext on server signal instead of from client side immediately?
        if (audioCtx && audioCtx.state === 'running') {
            audioCtx.suspend().catch(() => {
                // Fallback: stop current sources but keep queue
                playingSources.forEach((src) => { try { src.stop(); } catch (_e2) { } });
                playingSources.length = 0;
                nextScheduledTime = 0;
            });
        } else {
            playingSources.forEach((src) => { try { src.stop(); } catch (_e) { } });
            playingSources.length = 0;
            nextScheduledTime = 0;
        }
    }

    function resumeTTSPlayback() {
        ensureAudioCtx();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume().then(() => {
                if (playingSources.length === 0 && audioQueue.length > 0) {
                    playNextAudio();
                } else if (playingSources.length === 0 && audioQueue.length === 0) {
                    // Queue empty means nothing is left to play, so finish playback
                    // If the stream is already finished or inactive, finish playback
                    if (ttsStreamFinished || !ttsStreamActive) {
                        // If the stream is marked finished, notify the backend (matches onended behavior)
                        if (ttsStreamFinished) sendJson({ action: 'tts_playback_finished', timestamp: Date.now() });
                        // Reset stream state and inform the UI to go idle
                        markTTSStreamState('reset');
                        onIncomingJson({ action: 'client_tts_playback_finished', data: { timestamp: Date.now() } });
                        nextScheduledTime = 0;
                    }
                }
            }).catch(() => { });
        } else if (playingSources.length === 0 && audioQueue.length > 0) {
            playNextAudio();
        } else if (playingSources.length === 0 && audioQueue.length === 0) {
            // Queue empty means nothing is left to play, so finish playback
            if (ttsStreamFinished || !ttsStreamActive) {
                if (ttsStreamFinished) sendJson({ action: 'tts_playback_finished', timestamp: Date.now() });
                markTTSStreamState('reset');
                onIncomingJson({ action: 'client_tts_playback_finished', data: { timestamp: Date.now() } });
                nextScheduledTime = 0;
            }
        }
    }

    function stopAllPlayback() {
        // Reset TTS flags and stop timers/sources
        markTTSStreamState('reset');

        if (playPollTimer) {
            clearTimeout(playPollTimer);
            playPollTimer = null;
        }

        playingSources.forEach((src) => { try { src.stop(); } catch (_e) { } });
        playingSources.length = 0;

        nextScheduledTime = 0;

        audioQueue.length = 0;

        if (audioCtx) {
            if (audioCtx.state !== 'suspended') {
                audioCtx.close();
            }
            audioCtx = null;
        }
    }

    function handleIncomingData(event) {
        if (typeof event.data === 'string') {
            try {
                const json_data = JSON.parse(event.data);
                if (json_data) {
                    if (json_data.action === 'pong') {
                        onPongReceived(json_data);
                        return;
                    }


                    // Handle queue events even when streaming has not started
                    if (json_data.action === 'queue_status' || json_data.action === 'queue_granted') {
                        onIncomingJson(json_data);
                        return;
                    }

                    // Always process session info so session_id is not lost before streaming
                    if (json_data.action === 'session_info') {
                        onIncomingJson(json_data);
                        return;
                    }

                    if (streaming) {
                        onIncomingJson(json_data);
                    }
                }
            } catch (_e) { }
            return;
        }

        if (event.data instanceof ArrayBuffer) {
            if (!streaming || ttsStreamFinished) return;

            const audioData = event.data;

            const int16 = new Int16Array(audioData);
            if (int16.length === 0) return;

            const float32 = new Float32Array(int16.length);
            for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;

            ensureAudioCtx();
            const targetRate = audioCtx ? audioCtx.sampleRate : LOCAL_SAMPLE_RATE;
            const resampled = resampleChunkPCM(float32, TTS_SAMPLE_RATE, targetRate);

            audioQueue.push(resampled);
            enableAllPlayback();
            return;
        }
    }

    // Serial playback with seamless audio scheduling
    function playNextAudio() {
        // Do not start when AudioContext is paused
        if (audioCtx && audioCtx.state === 'suspended') return;

        if (audioQueue.length === 0) {
            if (ttsStreamFinished) return;

            if (ttsStreamActive) {
                // Poll until next chunk arrives
                if (playPollTimer) clearTimeout(playPollTimer);
                playPollTimer = setTimeout(() => {
                    playPollTimer = null;
                    if (audioQueue.length > 0) playNextAudio();
                }, 100);
                return;
            }

            return;
        }

        // Dequeue audio data (chunks arrive and play in order)
        const float32 = audioQueue.shift();

        newAudioOutputCallback && newAudioOutputCallback(float32, audioCtx ? audioCtx.sampleRate : LOCAL_SAMPLE_RATE);

        // Use local AudioContext sample rate for playback
        const buffer = audioCtx.createBuffer(1, float32.length, audioCtx.sampleRate);
        buffer.getChannelData(0).set(float32);

        const src = audioCtx.createBufferSource();
        src.buffer = buffer;
        src.connect(audioCtx.destination);

        // Adjust local playback speed (affects pitch because BufferSource changes playback rate)
        try {
            if (typeof ttsPlaybackRate === 'number' && isFinite(ttsPlaybackRate) && ttsPlaybackRate > 0) {
                src.playbackRate.value = ttsPlaybackRate;
            } else {
                src.playbackRate.value = 1.0;
            }
        } catch (_) { }

        const currentTime = audioCtx.currentTime;

        // If nextScheduledTime hasn't been initialized yet or is already in the past, start from the current time
        if (nextScheduledTime < currentTime) {
            nextScheduledTime = currentTime;
        }

        // Compute the actual playback duration of this audio chunk (taking playback rate into account)
        const duration = buffer.duration / (src.playbackRate.value || 1.0);

        // Start playback using an exact scheduled time
        const startTime = nextScheduledTime;
        src.start(startTime);  // ← Key: use an exact start time, not src.start()

        // Update the next chunk's start time = the end time of the current chunk
        nextScheduledTime = startTime + duration;


        // ===================================================

        src.onended = () => {
            const idx = playingSources.indexOf(src);
            if (idx !== -1) playingSources.splice(idx, 1);

            // Notify backend that a TTS chunk finished playback (FIFO order)
            sendJson({ action: "tts_chunk_played", timestamp: Date.now() });

            if (playingSources.length === 0 && audioQueue.length === 0) {
                // Delayed finish handling for TTS stream
                if (pendingTTSStreamFinished) {
                    pendingTTSStreamFinished = false;
                    markTTSStreamState('finished');
                    nextScheduledTime = 0;
                    return;
                }
                // Case A: playback drained and backend already marked finished
                if (ttsStreamFinished) {
                    sendJson({ action: "tts_playback_finished", timestamp: Date.now() });
                    markTTSStreamState('reset');
                    onIncomingJson({ action: 'client_tts_playback_finished', data: { timestamp: Date.now() } });
                    nextScheduledTime = 0;
                }
                // Case B: stream no longer active (stop or inactive resume) so fall back to idle
                else if (!ttsStreamActive) {
                    markTTSStreamState('reset');
                    onIncomingJson({ action: 'client_tts_playback_finished', data: { timestamp: Date.now() } });
                    nextScheduledTime = 0;
                }
            }
        };

        src.onerror = () => {
            const idx = playingSources.indexOf(src);
            if (idx !== -1) playingSources.splice(idx, 1);
        };

        playingSources.push(src);

        try {
            onIncomingJson({ action: 'client_tts_playback_started', data: { timestamp: Date.now() } });
        } catch (_) { }

        if (audioQueue.length > 0) {
            playNextAudio();
        }
    }

    // Start capture with auto VAD mode (default) or run raw capture in pure front-end mode
    async function startStreaming() {
        if (streaming) return;

        if (pureFrontend) {
            await initRawMic();
            if (!vad) throw new Error('Mic not initialized');
            streaming = true;
            return;
        }

        await initVAD();
        if (!vad) throw new Error('VAD not initialized');
        streaming = true;
    }

    // Stop capture and cleanup
    async function stopStreaming() {
        // Reset TTS flags and stop playback first
        markTTSStreamState('reset');
        stopAllPlayback();

        if (!streaming) return;

        streaming = false;

        if (vad) {
            try {
                await vad.stop();
            } catch (_e) {
                console.error('Error stopping VAD:', _e);
            } finally {
                vad = null;
            }
        }
    }

    function closeWebSocket() {
        if (ws) {
            console.log('[WS] Explicitly closing WebSocket');
            // Mark this closure as expected to avoid the disconnect prompt
            suppressNextCloseLog = true;
            stopHeartbeat();
            ws.close();
            ws = null;
        }
    }

    // Change reference voice by name
    function changeVoice(voiceName) {
        sendJson({
            action: "change_voice",
            voice_name: voiceName,
            timestamp: Date.now()
        });
    }

    // Change TTS speed
    function changeTTSSpeed(speed) {
        sendJson({
            action: "change_tts_speed",
            speed: speed,
            timestamp: Date.now()
        });
    }

    // Switch TTS model (IndexTTS / IndexTTS2)
    function changeTTSModel(modelType, config = {}) {
        sendJson({
            action: "change_tts_model",
            model_type: modelType,
            config: config,
            timestamp: Date.now()
        });
    }

    // Switch LLM model/base_url configuration (ChatOpenAI style)
    function changeLLMModel(modelName, baseUrl = '', apiKey = '', extraBody = null) {
        sendJson({
            action: "change_llm_model",
            model_name: modelName,
            base_url: baseUrl,
            api_key: apiKey,
            extra_body: extraBody,
            timestamp: Date.now()
        });
    }

    // TTS state flags
    function markTTSStreamState(state) {
        switch (state) {
            case 'active':
            case 'start':
                ttsStreamActive = true;
                ttsStreamFinished = false;
                break;
            case 'finished':
            case 'end':
                ttsStreamActive = false;
                ttsStreamFinished = true;
                if (playingSources.length === 0 && audioQueue.length === 0) {
                    sendJson({ action: "tts_playback_finished", timestamp: Date.now() });
                    onIncomingJson?.({ action: 'client_tts_playback_finished', data: { timestamp: Date.now() } });
                    ttsStreamActive = false;
                    ttsStreamFinished = false;
                }
                break;
            case 'reset':
            case 'idle':
                ttsStreamActive = false;
                ttsStreamFinished = false;
                pendingChunkIndex = null;
                break;
            default:
                break;
        }
    }

    function pendTTSStreamFinished() {
        pendingTTSStreamFinished = true;
    }

    return {
        closeWebSocket,
        initWebSocket,
        stopAllPlayback,
        pauseTTSPlayback,
        resumeTTSPlayback,
        startStreaming,
        stopStreaming,
        markTTSStreamState,
        pendTTSStreamFinished,
        // Toggle mic mute
        setMicMuted(flag) {
            micMuted = !!flag;
            try { vad && vad.setMuted && vad.setMuted(micMuted); } catch (_) { }
            // Tell the backend to end the current segment while muted
            if (micMuted) {
                try { sendJson({ action: 'vad_speech_end', timestamp: Date.now(), reason: 'muted' }); } catch (_) { }
            }
        },
        isMicMuted() { return !!micMuted; },
        // Expose simGen for higher-level logic if needed
        get simGen() { return !!simGen; },
        changeVoice,
        changeTTSSpeed,
        changeTTSModel,
        changeLLMModel,
        get isWebSocketOpen() {
            return ws && ws.readyState === WebSocket.OPEN;
        },
        onNewAudioOutput,
        // Set the local-only TTS playback rate
        setTTSPlaybackRate(rate) {
            const r = Number(rate);
            if (!Number.isNaN(r) && r > 0) {
                // Clamp to a reasonable range to avoid poor UX
                ttsPlaybackRate = Math.min(2.0, Math.max(0.5, r));
            }
        }
    };
}

// Create the conversation controller.
// Backward compatible signature:
// - createConversation(true|false) => { sim_gen }
// New signature:
// - createConversation({ sim_gen?: bool, simGen?: bool, pure_frontend?: bool, pureFrontend?: bool })
//   With pure_frontend=true the frontend skips VAD/enhancer and keeps streaming raw audio frames.
function createConversation(websocketURL = null, opts = null) {
    const SIM_GEN = opts?.simGen ?? false;
    const PURE_FRONTEND = opts?.pureFrontend ?? false;
    // Helper vars
    let lastClientVadStartTs = null;
    let waitingFirstUpdateResp = false;
    let finishASRTs = null;
    // Track assistant cumulative text and the "interruption baseline" length
    // - assistantFullText: backend-provided full text (grows with update_resp)
    // - assistantBaseLen: when interrupted by Info, record the displayed length
    //   Later assistant messages show fullText.slice(assistantBaseLen)
    let assistantFullText = '';
    let assistantBaseLen = 0;
    let currentAssistantTurnId = 0;
    let currentUserTurnId = 0;
    // State
    const ASSISTANT = 'assistant';
    const USER = 'user';
    // Initialize latency metrics
    const createDefaultLatencyState = () => ({
        network: 0,
        asr: 0,
        llmFirstToken: 0,
        llmSentence: 0,
        ttsFirstChunk: 0,
    });
    const rawState = {
        queued: false,
        queuePosition: null,
        // Detailed latency metrics (ms)
        latency: createDefaultLatencyState(),
        messages: [],
        streaming: false,
        loading: false,
        streamState: 'idle', // 'idle' | 'listening' | 'processing' | 'speaking'
        currentVoiceName: null,
        currentVoicePath: null,
        currentSessionId: null,
        latestThought: '',
        latestCaption: '',
        latestRetrieval: '',
        micMuted: false,
        currentSpeakerId: null,
    };
    function appendMessage({ role, content, turnId = 0 }) {
        rawState.messages.push({ role, content, turnId });
        onChangeCallback && onChangeCallback({ ...rawState });
    }
    function updateMessageByTurnOrLast({ role, content, turnId = 0 }) {
        const messages = rawState.messages || (rawState.messages = []);
        const t = Number(turnId || 0);

        // helper: emit with new messages reference (avoid UI missing updates)
        const emit = () => {
            onChangeCallback?.({ ...rawState, messages: [...messages] });
        };

        // 1) Same (role, turnId) -> update existing (search from tail)
        for (let i = messages.length - 1; i >= 0; i--) {
            const m = messages[i];
            if (m && m.role === role && Number(m.turnId || 0) === t) {
                messages[i] = { ...m, role, content, turnId: t };
                emit();
                return;
            }
        }

        // 2) Non-user/assistant (system/info/tool/...) -> always append
        if (role !== USER && role !== ASSISTANT) {
            messages.push({ role, content, turnId: t });
            emit();
            return;
        }

        // 3) Insert strategy:
        // - ASSISTANT: NEVER insert before existing items (no "back insert"),
        //              only append, so it can't appear before the triggering user message.
        // - USER: if assistant(t) already exists (rare, out-of-order), insert right after it
        //         to keep same-turn messages adjacent; otherwise append (allow user 1,2,3 ...).
        if (role === ASSISTANT) {
            messages.push({ role, content, turnId: t });
            emit();
            return;
        }

        // role === USER
        const asstIdx = messages.findIndex(
            (m) => m?.role === ASSISTANT && Number(m.turnId || 0) === t
        );

        if (asstIdx !== -1) {
            // Keep same turn together: assistant(t) -> user(t)
            messages.splice(asstIdx + 1, 0, { role, content, turnId: t });
        } else {
            // Collapse trailing user bubbles so the view alternates user/assistant
            const last = messages[messages.length - 1];
            if (last && last.role === USER) {
                messages[messages.length - 1] = { role, content, turnId: t };
            } else {
                messages.push({ role, content, turnId: t });
            }
        }

        emit();
    }


    function removeLastUserMessage() {
        const messages = rawState.messages;
        if (messages.length > 0 && messages[messages.length - 1].role === USER) {
            messages.pop();
            onChangeCallback({ ...rawState });
        }
    }
    let onChangeCallback = null;
    const state = new Proxy(rawState, {
        set(target, prop, value) {
            target[prop] = value;
            if (onChangeCallback) onChangeCallback({ ...target });
            return true;
        }
    });
    function updateLatencyState(partial) {
        const prev = rawState.latency || createDefaultLatencyState();
        state.latency = { ...prev, ...(partial || {}) };
    }
    // Handle incoming JSON from server
    let audioSession = null;
    function onIncomingJson(json) {
        const normalizeDisplayText = (s) => {
            // Convert literal \"\n\" to real newlines while keeping actual newlines
            if (typeof s !== 'string') return '';
            return s.replace(/\\n/g, '\n');
        };
        const resolveTurnId = (payload) => {
            const v = Number(payload?.turn_id ?? payload?.data?.turn_id ?? 0);
            return Number.isFinite(v) ? v : 0;
        };
        switch (json.action) {
            case 'queue_status': {
                const position = json.position ?? 1;
                state.loading = true;
                state.queued = true;
                const queueMsg = `In queue... Your position: ${position}. Please wait.`;
                if (state.queuePosition !== position) {
                    state.queuePosition = position;
                    appendMessage({ role: 'info', content: queueMsg });
                }
                break;
            }
            // Queue granted: it is the user's turn
            case 'queue_granted': {
                state.loading = false;
                state.queued = false;
                state.queuePosition = null;
                appendMessage({ role: 'info', content: 'It\'s your turn!' });
                break;
            }
            case 'session_info': {
                // Store the current session_id for later uploads and actions
                const sid = json?.data?.session_id || null;
                state.currentSessionId = sid ? String(sid) : null;
                break;
            }
            case 'speaker_updated': {
                const sid = json?.data?.speaker_id ?? null;
                state.currentSpeakerId = (sid === null || sid === undefined || sid === '') ? null : String(sid);
                break;
            }
            case 'client_log': {
                // Surface model-related logs as conversation messages
                const text = typeof json.data === 'string' ? json.data : JSON.stringify(json.data);
                // If the last message is assistant, treat the Info log as an interruption and update the baseline
                const last = rawState.messages[rawState.messages.length - 1];
                if (last && last.role === ASSISTANT) {
                    assistantBaseLen = assistantFullText.length;
                }
                appendMessage({ role: 'info', content: text });
                break;
            }
            case 'voice_changed': {
                const name = json?.data?.name || '';
                state.currentVoiceName = name;
                break;
            }
            case 'tool_called': {
                const name = json?.data?.name || '';
                const args = json?.data?.args || {};
                let prettyArgs = '';
                try {
                    const keys = Object.keys(args || {});
                    if (keys.length > 0) {
                        // Compact view: k=v pairs joined by commas to avoid spam
                        prettyArgs = ' ' + keys.map(k => `${k}=${typeof args[k] === 'string' ? args[k] : JSON.stringify(args[k])}`).join(', ');
                    }
                } catch (_) { }
                const last = rawState.messages[rawState.messages.length - 1];
                if (last && last.role === ASSISTANT) {
                    assistantBaseLen = assistantFullText.length;
                }
                // Log each tool call as a standalone info entry to avoid being overwritten
                appendMessage({ role: 'info', content: `Tool call -> ${name}${prettyArgs}` });
                break;
            }
            case 'client_vad_speech_start': {
                const ts = (json.data && json.data.timestamp) || Date.now();
                lastClientVadStartTs = ts;
                state.streamState = 'listening';
                break;
            }
            case 'client_vad_speech_end':
                state.streamState = 'processing';
                break;
            case 'client_tts_playback_started':
                state.streamState = 'speaking';
                break;
            case 'client_tts_playback_finished':
                state.streamState = 'idle';
                break;
            case 'invalid_asr_result':
                state.streamState = 'idle';
                break;
            case 'start_tts':
                // Only mark the backend TTS stream as active; wait for actual audio playback before switching to speaking.
                audioSession.markTTSStreamState('active');
                break;
            case 'pause_tts': {
                // Pure front-end mode: when the backend asks to pause TTS, simulate the local interruption start,
                // Run the same local logic as window.vad.Message.SpeechStart without sending JSON.
                // Only do this outside of simultaneous generation; SIM_GEN never auto-pauses.
                if (!SIM_GEN) {
                    try { audioSession.pauseTTSPlayback(); } catch (_) { }
                    if (PURE_FRONTEND) {
                        const nowTs = Date.now();
                        // Trigger the same frontend state update as VADSpeechStart (no server call)
                        onIncomingJson({ action: 'client_vad_speech_start', data: { timestamp: nowTs } });
                    }
                }
                break;
            }
            case 'stop_tts':
                audioSession.stopAllPlayback();
                break;
            case 'resume_tts':
                removeLastUserMessage();
                state.streamState = 'speaking';
                // When resuming, check if there is audio left to play
                // If nothing is queued and no source is playing, return to idle
                audioSession.resumeTTSPlayback();
                // Note: resumeTTSPlayback triggers playback and onended handles the state changes
                break;
            case 'tts_finished':
                audioSession.pendTTSStreamFinished();
                break;
            case 'update_resp': {
                // Backend now computes synthesis latency and pushes it via latency_metrics
                // We no longer compute it from the first update_resp
                const fullText = normalizeDisplayText(json.data.text || '');
                const last = rawState.messages[rawState.messages.length - 1];
                const incomingTurn = resolveTurnId(json);
                const targetTurn = incomingTurn || currentAssistantTurnId || 0;
                if (incomingTurn && incomingTurn !== currentAssistantTurnId) {
                    // New assistant turn -> reset the baseline
                    assistantFullText = '';
                    assistantBaseLen = 0;
                    currentAssistantTurnId = incomingTurn;
                }
                // If backend text no longer extends the old prefix, fall back to showing the full text
                const safeBase = (assistantFullText && !fullText.startsWith(assistantFullText)) ? 0 : assistantBaseLen;
                const sliceFrom = Math.max(0, Math.min(safeBase, fullText.length));
                const toDisplay = fullText.slice(sliceFrom);

                if (SIM_GEN) {
                    appendMessage({ role: ASSISTANT, content: toDisplay, turnId: targetTurn });
                } else {
                    if (toDisplay) {
                        updateMessageByTurnOrLast({ role: ASSISTANT, content: toDisplay, turnId: targetTurn });
                    }
                }
                // Update the stored full text
                assistantFullText = fullText;
                break;
            }
            case 'finish_resp': {
                const fullText = normalizeDisplayText(json.data.text || '');
                const last = rawState.messages[rawState.messages.length - 1];
                const incomingTurn = resolveTurnId(json);
                const targetTurn = incomingTurn || currentAssistantTurnId || 0;
                if (incomingTurn && incomingTurn !== currentAssistantTurnId) {
                    assistantFullText = '';
                    assistantBaseLen = 0;
                    currentAssistantTurnId = incomingTurn;
                }
                const safeBase = (assistantFullText && !fullText.startsWith(assistantFullText)) ? 0 : assistantBaseLen;
                const sliceFrom = Math.max(0, Math.min(safeBase, fullText.length));
                const toDisplay = fullText.slice(sliceFrom);

                if (SIM_GEN) {
                    appendMessage({ role: ASSISTANT, content: toDisplay, turnId: targetTurn });
                } else {
                    if (toDisplay) {
                        updateMessageByTurnOrLast({ role: ASSISTANT, content: toDisplay, turnId: targetTurn });
                    }
                }
                // End of a reply: reset accumulation and baseline
                assistantFullText = '';
                assistantBaseLen = 0;
                if (incomingTurn) {
                    currentAssistantTurnId = incomingTurn;
                }
                break;
            }
            case 'update_asr': {
                if (lastClientVadStartTs) {
                    const now = Date.now();
                    lastClientVadStartTs = null;
                }
                const incomingTurn = resolveTurnId(json);
                if (incomingTurn && incomingTurn !== currentUserTurnId) {
                    currentUserTurnId = incomingTurn;
                }
                const targetTurn = incomingTurn || currentUserTurnId || 0;
                if (SIM_GEN) {
                    appendMessage({ role: USER, content: json.data.text, turnId: targetTurn });
                } else {
                    updateMessageByTurnOrLast({ role: USER, content: json.data.text, turnId: targetTurn });
                }
                break;
            }
            case 'refine_transcription': {
                if (SIM_GEN) {
                    // Remove all user messages and keep a single aggregated entry
                    const incomingTurn = resolveTurnId(json);
                    const filtered = rawState.messages.filter(m => m.role !== USER || (incomingTurn && Number(m.turnId || 0) !== incomingTurn));
                    filtered.push({ role: USER, content: normalizeDisplayText(json.data?.text || ''), turnId: incomingTurn || 0 });
                    rawState.messages = filtered;
                    onChangeCallback && onChangeCallback({ ...rawState });
                }
                break;
            }
            case 'finish_asr': {
                const incomingTurn = resolveTurnId(json);
                if (incomingTurn && incomingTurn !== currentUserTurnId) {
                    currentUserTurnId = incomingTurn;
                }
                const targetTurn = incomingTurn || currentUserTurnId || 0;
                if (SIM_GEN) {
                    appendMessage({ role: USER, content: json.data.text, turnId: targetTurn });
                } else {
                    updateMessageByTurnOrLast({ role: USER, content: json.data.text, turnId: targetTurn });
                }
                // Backend pushes latency_metrics after the first TTSStarted event
                // Frontend only resets the displayed metrics
                finishASRTs = null;
                waitingFirstUpdateResp = false;
                break;
            }
            case 'latency_metrics': {
                // Detailed latency metrics
                const data = json?.data || {};
                updateLatencyState({
                    network: Number(data.network_latency_ms) || 0,
                    asr: Number(data.asr_latency_ms) || 0,
                    llmFirstToken: Number(data.llm_first_token_ms) || 0,
                    llmSentence: Number(data.llm_sentence_ms) || 0,
                    ttsFirstChunk: Number(data.tts_first_chunk_ms) || 0,
                });
                break;
            }
            case 'thought_updated': {
                state.latestThought = (json.data && json.data.text) ? String(json.data.text) : '';
                break;
            }
            case 'caption_updated': {
                state.latestCaption = (json.data && json.data.text) ? String(json.data.text) : '';
                break;
            }
            case 'retrieval_updated': {
                state.latestRetrieval = (json.data && json.data.text) ? String(json.data.text) : '';
                break;
            }
            case 'error':
                // Treat errors as interruptions and update the baseline
                {
                    const last = rawState.messages[rawState.messages.length - 1];
                    if (last && last.role === ASSISTANT) {
                        assistantBaseLen = assistantFullText.length;
                    }
                }
                updateMessageByTurnOrLast({ role: 'info', content: `Error: ${json.data || 'unknown'}` });
        }
    }
    audioSession = createAudioSession(onIncomingJson, websocketURL, opts);
    audioSession.initWebSocket();
    state.loading = false;

    async function uploadFile(file) {
        if (!file) return;
        const sid = state.currentSessionId;
        if (!sid) {
            console.warn("[Upload] No active session_id, skip upload.");
            return;
        }
        // File upload counts as an offline request; enter processing and wait for backend TTS/embedding.
        state.streamState = 'processing';
        const form = new FormData();
        form.append("session_id", sid);
        form.append("file", file);
        try {
            const resp = await fetch("/api/upload", {
                method: "POST",
                body: form,
            });
            if (!resp.ok) {
                console.error("[Upload] Failed:", await resp.text());
                // On upload failure, return to idle instead of staying stuck in processing.
                state.streamState = 'idle';
                return;
            }
            const data = await resp.json().catch(() => ({}));
            console.log("[Upload] Done:", data);
        } catch (e) {
            console.error("[Upload] Error:", e);
            // Network or other exceptions also revert to idle.
            state.streamState = 'idle';
        }
    }
    // Outside control
    async function toggleStreaming() {
        if (state.loading) {
            return;
        }

        if (state.streaming) {
            await audioSession.stopStreaming();
            state.streaming = false;
            state.streamState = 'idle';
            // When stopping, also clear assistant tracking to avoid truncating the next turn
            assistantFullText = '';
            assistantBaseLen = 0;
            currentAssistantTurnId = 0;
            currentUserTurnId = 0;
        } else {
            state.loading = true;
            try {
                await audioSession.startStreaming();
                state.streaming = true;
                if (PURE_FRONTEND) {
                    // Pure front-end mode: without VAD start/end events treat it as "listening" by default,
                    // and let backend TTS events move it to speaking/idle.
                    state.streamState = 'listening';
                }
            } finally {
                state.loading = false;
            }
        }
    }
    function subscribe(cb) {
        onChangeCallback = cb;
        cb({ ...rawState });
    }

    function resetConversationState() {
        rawState.messages = [];
        // Reset latency metrics
        rawState.latency = createDefaultLatencyState();
        rawState.latestThought = '';
        rawState.latestCaption = '';
        rawState.latestRetrieval = '';
        rawState.currentSpeakerId = null;
        if (!rawState.streaming) {
            rawState.streamState = 'idle';
        }
        assistantFullText = '';
        assistantBaseLen = 0;
        onChangeCallback && onChangeCallback({ ...rawState });
        // Reset assistant baseline tracking
        assistantFullText = '';
        assistantBaseLen = 0;
        currentAssistantTurnId = 0;
        currentUserTurnId = 0;
    }

    // Restart: clear messages and rebuild the WebSocket connection (no auto recording)
    async function restart() {
        try {
            if (state.streaming) {
                await audioSession.stopStreaming();
                state.streaming = false;
            }
        } catch (_e) { }
        try { audioSession.stopAllPlayback(); } catch (_e) { }
        try { audioSession.closeWebSocket(); } catch (_e) { }
        try { audioSession.initWebSocket(); } catch (_e) { }

        resetConversationState();
    }

    function clearHistory() {
        resetConversationState();
    }

    function changeVoice(voiceName) {
        if (audioSession) {
            audioSession.changeVoice(voiceName);
        }
    }

    function changeTTSSpeed(speed) {
        if (audioSession) {
            audioSession.changeTTSSpeed(speed);
        }
    }

    function changeTTSModel(modelType, config = {}) {
        if (audioSession && audioSession.changeTTSModel) {
            audioSession.changeTTSModel(modelType, config);
        }
    }

    function changeLLMModel(modelName, baseUrl = '', apiKey = '') {
        if (audioSession && audioSession.changeLLMModel) {
            audioSession.changeLLMModel(modelName, baseUrl, apiKey);
        }
    }

    // Toggle/set the microphone mute state
    function toggleMicMute() {
        const next = !rawState.micMuted;
        if (audioSession) audioSession.setMicMuted(next);
        state.micMuted = next;
        // If muting while "listening", fall back to idle to avoid confusing the UI
        if (next && state.streamState === 'listening') {
            state.streamState = 'idle';
        }
    }

    return {
        subscribe,
        state,
        toggleStreaming,
        changeVoice,
        changeTTSSpeed,
        changeTTSModel,
        changeLLMModel,
        restart,
        clearHistory,
        onNewAudioOutput: audioSession.onNewAudioOutput,
        toggleMicMute,
        uploadFile,
    }
}

export { createConversation };
