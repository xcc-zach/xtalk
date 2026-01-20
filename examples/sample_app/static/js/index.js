import { createConversation } from "../../xtalk/index.js";

function getWebSocketURL(path = "ws") {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsPath = new URL("./ws", window.location.href);
    wsPath.protocol = proto;
    wsPath.host = window.location.host;
    return wsPath
}

const convo = createConversation(getWebSocketURL());

const $btnToggle = document.getElementById('btn-toggle');
const $btnView = document.getElementById('btn-view');
const $btnThought = document.getElementById('btn-thought');
const $btnRestart = document.getElementById('btn-restart');
const $btnStop = document.getElementById('btn-stop');
const $btnCaption = document.getElementById('btn-caption');
const $btnRetrieval = document.getElementById('btn-retrieval');
const $btnMic = document.getElementById('btn-mic');
const $btnUploadFile = document.getElementById('btn-upload-file');
const $fileInput = document.getElementById('file-input');
const $messages = document.getElementById('messages');
const $latNet = document.getElementById('latency-net');
const $latASR = document.getElementById('latency-asr');
const $latLLMToken = document.getElementById('latency-llm-token');
const $latLLMSentence = document.getElementById('latency-llm-sentence');
const $latTTS = document.getElementById('latency-tts');
const $latE2E = document.getElementById('latency-e2e');
const $streamingState = document.getElementById('streaming-state');
const $speakerId = document.getElementById('speaker-id');
const $waveform = document.getElementById('waveform');
const $waveformCard = document.getElementById('waveform-card');
const $voiceSelect = document.getElementById('voice-select');
const $speedSelect = document.getElementById('speed-select');
const $ttsModelSelect = document.getElementById('tts-model-select');
const $llmModelSelect = document.getElementById('llm-model-select');
const $thoughtCard = document.getElementById('thought-card');
const $thoughtContent = document.getElementById('thought-content');
const $captionCard = document.getElementById('caption-card');
const $captionContent = document.getElementById('caption-content');
const $retrievalCard = document.getElementById('retrieval-card');
const $retrievalContent = document.getElementById('retrieval-content');

// Load available reference audio files
let availableAudios = [];
function syncVoiceSelectValue(targetName) {
    if (!$voiceSelect) return;
    const desired = targetName || convo.state.currentVoiceName || '';
    if (!desired) return;
    if ($voiceSelect.value === desired) return;
    const hasOption = Array.from($voiceSelect.options).some(opt => opt.value === desired);
    if (hasOption) {
        $voiceSelect.value = desired;
    }
}
async function loadReferenceAudios() {
    try {
        const response = await fetch('/api/voices');
        const data = await response.json();
        availableAudios = data.audios || [];

        // Populate the dropdown
        $voiceSelect.innerHTML = '<option value="" selected disabled hidden></option>';
        availableAudios.forEach((audio, index) => {
            const voiceName = audio.name || audio.path || `voice_${index}`;
            const option = document.createElement('option');
            option.value = voiceName;
            option.textContent = voiceName;
            option.dataset.path = audio.path || '';
            $voiceSelect.appendChild(option);
        });

        // Enable the dropdown
        $voiceSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load reference audios:', error);
        $voiceSelect.innerHTML = '<option value=\"\">Load failed</option>';
    }
}

// Handle voice changes
$voiceSelect.addEventListener('change', (e) => {
    const selectedName = e.target.value;
    const selectedAudio = availableAudios.find(a => (a.name || a.path) === selectedName);

    if (selectedAudio) {
        // Switch voices and let the frontend session announce it in the chat log
        const voiceName = selectedAudio.name || selectedName;
        convo.changeVoice(voiceName);
        convo.state.currentVoiceName = voiceName;
        convo.state.currentVoicePath = selectedAudio.path || null;
        syncVoiceSelectValue(voiceName);
    }
});

// Load available TTS models
let availableTTSModels = [];
async function loadTTSModels() {
    try {
        const response = await fetch('/api/available-tts-models');
        const data = await response.json();
        availableTTSModels = data.models || [];

        // Populate the TTS dropdown
        $ttsModelSelect.innerHTML = '';
        if (availableTTSModels.length === 0) {
            $ttsModelSelect.innerHTML = '<option value=\"\">No available models</option>';
            return;
        }
        availableTTSModels.forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.type;
            option.textContent = model.name || model.type;
            option.dataset.config = JSON.stringify(model.config || {});
            if (index === 0) {
                option.selected = true;
            }
            $ttsModelSelect.appendChild(option);
        });

        $ttsModelSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load TTS models:', error);
        $ttsModelSelect.innerHTML = '<option value=\"\">Load failed</option>';
    }
}

// Load available LLM models
let availableLLMModels = [];
async function loadLLMModels() {
    try {
        const response = await fetch('/api/available-llm-models');
        const data = await response.json();
        availableLLMModels = data.models || [];

        // Populate the LLM dropdown
        $llmModelSelect.innerHTML = '';
        if (availableLLMModels.length === 0) {
            $llmModelSelect.innerHTML = '<option value=\"\">No available models</option>';
            return;
        }
        availableLLMModels.forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.model;
            option.textContent = model.display_name || model.model;
            option.dataset.baseUrl = model.base_url || '';
            option.dataset.apiKey = model.api_key || '';
            option.dataset.extraBody = JSON.stringify(model.extra_body || null);
            if (index === 0) {
                option.selected = true;
            }
            $llmModelSelect.appendChild(option);
        });

        $llmModelSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load LLM models:', error);
        $llmModelSelect.innerHTML = '<option value=\"\">Load failed</option>';
    }
}

// Handle speech-speed changes
$speedSelect.addEventListener('change', (e) => {
    const speed = parseFloat(e.target.value);
    if (!isNaN(speed) && speed >= 0.5 && speed <= 1.5) {
        convo.changeTTSSpeed(speed);
    }
});

// Handle TTS-model changes
$ttsModelSelect.addEventListener('change', (e) => {
    const selectedType = e.target.value;
    const selectedOption = e.target.options[e.target.selectedIndex];
    let config = {};
    try {
        config = JSON.parse(selectedOption.dataset.config || '{}');
    } catch (_) { }

    if (selectedType) {
        convo.changeTTSModel(selectedType, config);
    }
});

// Handle LLM-model changes
$llmModelSelect.addEventListener('change', (e) => {
    const selectedModelName = e.target.value;
    const selectedOption = e.target.options[e.target.selectedIndex];
    const baseUrl = selectedOption.dataset.baseUrl || '';
    const apiKey = selectedOption.dataset.apiKey || '';
    let extraBody = null;
    try {
        extraBody = JSON.parse(selectedOption.dataset.extraBody || 'null');
    } catch (_) { }

    if (selectedModelName) {
        // Support the newer session API by forwarding extraBody
        convo.changeLLMModel(selectedModelName, baseUrl, apiKey, extraBody);
    }
});

// Fetch available audio and model lists on page load
loadReferenceAudios();
loadTTSModels();
loadLLMModels();

// Waveform drawing state
let audioCtx = null;
let analyser = null; // input analyser (microphone)
let micStream = null;
let sourceNode = null;
let rafId = null;
let dataArray = null;
let bufferLength = 0;
let waveformActive = false;
let isStreaming = false;
let currentStreamState = 'idle';

// Output (TTS) analyser state
let outAnalyser = null; // output analyser (TTS/audio elements)
let outDataArray = null;
let outBufferLength = 0;
const attachedMediaElements = new WeakSet();
let outAnalyserConnected = false;
let silentGain = null; // zero-gain node to keep graph pulling without audible output

const COLOR_IN = '#a7f3d0';
const COLOR_OUT = '#93c5fd';

// Status color mapping (adjust per theme)
const STATE_COLORS = {
    idle: '#6b7280',        // gray-500
    listening: '#34d399',   // emerald-400
    processing: '#fbbf24',  // amber-400
    speaking: '#93c5fd'     // sky-300
};

function getWaveformColor() {
    return STATE_COLORS[currentStreamState] || COLOR_IN;
}

const canvasCtx = $waveform.getContext('2d');


function ensureAudioContext() {
    if (!audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();
    }
    return audioCtx;
}

function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const { clientWidth, clientHeight } = $waveform;
    const width = Math.max(1, Math.floor(clientWidth * dpr));
    const height = Math.max(1, Math.floor(clientHeight * dpr));
    if ($waveform.width !== width || $waveform.height !== height) {
        $waveform.width = width;
        $waveform.height = height;
    }
    canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
}

function clearCanvas() {
    const { width, height } = $waveform;
    canvasCtx.clearRect(0, 0, width, height);
    // Baseline
    canvasCtx.strokeStyle = '#1f2937';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, height / 2);
    canvasCtx.lineTo(width, height / 2);
    canvasCtx.stroke();
}

function drawSeries(uint8Array, length, color) {
    const w = $waveform.width;
    const h = $waveform.height;
    const sliceWidth = w / length;
    canvasCtx.strokeStyle = color;
    canvasCtx.lineWidth = 2;
    canvasCtx.beginPath();
    let x = 0;
    for (let i = 0; i < length; i++) {
        const v = uint8Array[i] / 128.0; // 0..255 -> ~0..2
        const y = (v * h) / 2;
        if (i === 0) canvasCtx.moveTo(x, y);
        else canvasCtx.lineTo(x, y);
        x += sliceWidth;
    }
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();
}

function drawWaveform() {
    if (!waveformActive) return;
    rafId = requestAnimationFrame(drawWaveform);

    // Background
    const w = $waveform.width;
    const h = $waveform.height;
    canvasCtx.fillStyle = '#0f172a';
    canvasCtx.fillRect(0, 0, w, h);
    // Baseline
    canvasCtx.strokeStyle = '#1f2937';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, h / 2);
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();

    // Only draw one series: speaking for output; else for input
    let streamState = convo.state.streamState;
    if (outAnalyser && outDataArray && outBufferLength && streamState === 'speaking') {
        outAnalyser.getByteTimeDomainData(outDataArray);
        drawSeries(outDataArray, outBufferLength, getWaveformColor());
    }
    if (dataArray && bufferLength && analyser && streamState !== 'speaking') {
        analyser.getByteTimeDomainData(dataArray);
        drawSeries(dataArray, bufferLength, getWaveformColor());
    }
}

async function startWaveform() {
    // If visualization exists but only lacks the input chain (e.g., output-only mode), upgrade to input + output
    if (waveformActive) {
        // When an output visualization exists and we are not muted, add the input chain
        if ((!analyser || !sourceNode) && !convo.state?.micMuted) {
            try {
                ensureAudioContext();
                await audioCtx.resume();

                micStream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
                    video: false,
                });
                sourceNode = audioCtx.createMediaStreamSource(micStream);
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 1024;
                analyser.smoothingTimeConstant = 0.7;
                bufferLength = analyser.fftSize;
                dataArray = new Uint8Array(bufferLength);
                sourceNode.connect(analyser);
            } catch (e) {
                console.error('Waveform upgrade to input failed:', e);
            }
        }
        return;
    }
    try {
        ensureAudioContext();
        await audioCtx.resume();

        micStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
            video: false,
        });

        sourceNode = audioCtx.createMediaStreamSource(micStream);
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.7;
        bufferLength = analyser.fftSize;
        dataArray = new Uint8Array(bufferLength);

        sourceNode.connect(analyser);

        // Prepare output analyser if not exists (will be fed when an audio element/node is attached)
        if (!outAnalyser) {
            outAnalyser = audioCtx.createAnalyser();
            outAnalyser.fftSize = 1024;
            outAnalyser.smoothingTimeConstant = 0.7;
            outBufferLength = outAnalyser.fftSize;
            outDataArray = new Uint8Array(outBufferLength);
        }

        resizeCanvas();
        clearCanvas();
        waveformActive = true;
        drawWaveform();
    } catch (err) {
        console.error('Waveform start failed:', err);
    }
}

// Waveform used for TTS-only visualization without opening the mic
async function startWaveformOutputOnly() {
    if (waveformActive) return;
    try {
        ensureAudioContext();
        await audioCtx.resume();
        // Do not create or connect input nodes; only keep the output analyzer alive
        if (!outAnalyser) {
            outAnalyser = audioCtx.createAnalyser();
            outAnalyser.fftSize = 1024;
            outAnalyser.smoothingTimeConstant = 0.7;
            outBufferLength = outAnalyser.fftSize;
            outDataArray = new Uint8Array(outBufferLength);
        }
        if (!outAnalyserConnected) {
            if (!silentGain) {
                silentGain = audioCtx.createGain();
                silentGain.gain.value = 0;
                silentGain.connect(audioCtx.destination);
            }
            outAnalyser.connect(silentGain);
            outAnalyserConnected = true;
        }
        resizeCanvas();
        clearCanvas();
        waveformActive = true;
        drawWaveform();
    } catch (err) {
        console.error('Waveform output-only start failed:', err);
    }
}

async function stopWaveform() {
    if (!waveformActive) return;
    waveformActive = false;
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    try {
        if (sourceNode && analyser) {
            try { sourceNode.disconnect(); } catch { }
        }
        if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
        }
        if (audioCtx) {
            // keep audioCtx for quick resume
        }
    } finally {
        clearCanvas();
        sourceNode = null;
        analyser = null;
        micStream = null;
    }
}

// Stop microphone input only while keeping waveform rendering and output analysis
function stopInputCapture() {
    try {
        if (sourceNode && analyser) {
            try { sourceNode.disconnect(); } catch { }
        }
        if (micStream) {
            micStream.getTracks().forEach(t => t.stop());
        }
    } finally {
        sourceNode = null;
        analyser = null;
        micStream = null;
        dataArray = null;
        bufferLength = 0;
    }
}

function attachOutputFromMediaElement(el) {
    try {
        ensureAudioContext();
        if (!(el instanceof HTMLMediaElement)) return;
        if (attachedMediaElements.has(el)) return;
        const src = audioCtx.createMediaElementSource(el);
        // connect to analyser only; do not alter playback path
        if (!outAnalyser) {
            outAnalyser = audioCtx.createAnalyser();
            outAnalyser.fftSize = 1024;
            outAnalyser.smoothingTimeConstant = 0.7;
            outBufferLength = outAnalyser.fftSize;
            outDataArray = new Uint8Array(outBufferLength);
        }
        // ensure silent pull chain: analyser -> silentGain(0) -> destination
        if (!outAnalyserConnected) {
            if (!silentGain) {
                silentGain = audioCtx.createGain();
                silentGain.gain.value = 0;
                silentGain.connect(audioCtx.destination);
            }
            outAnalyser.connect(silentGain);
            outAnalyserConnected = true;
        }
        src.connect(outAnalyser);
        attachedMediaElements.add(el);
    } catch (e) {
        // Creating MediaElementSource on same element more than once throws; guard above should avoid it
        // Ignore if cannot attach
    }
}

function attachOutputFromAudioNode(node) {
    try {
        ensureAudioContext();
        if (!node || typeof node.connect !== 'function') return;
        if (!outAnalyser) {
            outAnalyser = audioCtx.createAnalyser();
            outAnalyser.fftSize = 1024;
            outAnalyser.smoothingTimeConstant = 0.7;
            outBufferLength = outAnalyser.fftSize;
            outDataArray = new Uint8Array(outBufferLength);
        }
        // ensure silent pull chain
        if (!outAnalyserConnected) {
            if (!silentGain) {
                silentGain = audioCtx.createGain();
                silentGain.gain.value = 0;
                silentGain.connect(audioCtx.destination);
            }
            outAnalyser.connect(silentGain);
            outAnalyserConnected = true;
        }
        node.connect(outAnalyser);
    } catch (e) {
        // ignore attach errors
    }
}

// Auto-hook: whenever an <audio> element starts playing, try to attach for output visualization
document.addEventListener('play', (ev) => {
    const el = ev.target;
    if (el instanceof HTMLAudioElement) {
        attachOutputFromMediaElement(el);
    }
}, true);

// Expose manual hooks for external TTS pipeline integration
window.attachOutputMediaElement = attachOutputFromMediaElement;
window.attachOutputAudioNode = attachOutputFromAudioNode;

window.addEventListener('resize', () => {
    resizeCanvas();
});

// Pause drawing when page not visible, resume if still streaming
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopWaveform();
    } else if (isStreaming) {
        // Visualization policy: if muted but speaking, render output only; otherwise capture normally
        if (convo.state?.micMuted && (convo.state?.streamState === 'speaking')) {
            startWaveformOutputOnly();
        } else if (!convo.state?.micMuted) {
            startWaveform();
        }
    }
});

// Render callback for state changes (messages, metrics, waveforms, etc.)
convo.subscribe((state) => {
    $latNet.textContent = state.latency.network ?? '--';
    $latASR.textContent = state.latency.asr ?? '--';
    $latLLMToken.textContent = state.latency.llmFirstToken ?? '--';
    $latLLMSentence.textContent = state.latency.llmSentence ?? '--';
    $latTTS.textContent = state.latency.ttsFirstChunk ?? '--';
    $latE2E.textContent = state.latency.asr + state.latency.llmSentence + state.latency.ttsFirstChunk ?? '--';

    $messages.innerHTML = '';
    for (const m of state.messages) {
        const div = document.createElement('div');
        div.className = `msg ${m.role}`;
        const role = document.createElement('span');
        role.className = 'role';
        role.textContent = m.role;
        const content = document.createElement('span');
        content.textContent = '  ' + (m.content ?? '');
        div.appendChild(role);
        div.appendChild(content);
        $messages.appendChild(div);
    }

    // After updating the message list, always scroll the chat container itself to the bottom
    // Use rAF to wait for DOM writes before setting scrollTop
    if (!$messages.classList.contains('hidden')) {
        requestAnimationFrame(() => {
            try { $messages.scrollTop = $messages.scrollHeight; } catch { }
        });
    }

    // Update the Thought display while keeping card visibility
    if ($thoughtContent) {
        $thoughtContent.textContent = state.latestThought || '';
    }
    // Update the Caption display while keeping card visibility
    if ($captionContent) {
        $captionContent.textContent = state.latestCaption || '';
    }
    // Update the Retrieval display while keeping card visibility
    if ($retrievalContent) {
        $retrievalContent.textContent = state.latestRetrieval || '';
    }

    $btnToggle.textContent = state.loading ? 'Loading' : state.streaming ? '🛑 Stop' : '🎙️ Start';
    $streamingState.textContent = state.streamState; // 'idle' | 'listening' | 'processing' | 'speaking'
    if ($speakerId) $speakerId.textContent = state.currentSpeakerId || '--';

    $btnMic.textContent = state.micMuted ? 'Unmute' : 'Mute';

    currentStreamState = state.streamState ?? 'idle';
    isStreaming = !!state.streaming;

    // Start/stop waveform capture：
    // - Not muted and streaming: input + output
    // - Muted but "speaking": output only
    // - Otherwise: stop everything
    if (state.streaming && !state.micMuted) {
        startWaveform();
    } else if (state.streaming && state.micMuted && state.streamState === 'speaking') {
        startWaveformOutputOnly();
    } else {
        stopWaveform();
    }

    // Keep the voice dropdown synced with the backend state
    syncVoiceSelectValue(state.currentVoiceName);
});

convo.onNewAudioOutput((float32, sampleRate) => {
    try {
        ensureAudioContext();
        if (!audioCtx) return;

        // Ensure analyser exists and arrays are sized
        if (!outAnalyser) {
            outAnalyser = audioCtx.createAnalyser();
            outAnalyser.fftSize = 1024;
            outAnalyser.smoothingTimeConstant = 0.7;
            outBufferLength = outAnalyser.fftSize;
            outDataArray = new Uint8Array(outBufferLength);
        }
        // Ensure silent pull chain (no audible playback)
        if (!outAnalyserConnected) {
            if (!silentGain) {
                silentGain = audioCtx.createGain();
                silentGain.gain.value = 0;
                silentGain.connect(audioCtx.destination);
            }
            outAnalyser.connect(silentGain);
            outAnalyserConnected = true;
        }

        // Create and feed a BufferSource from raw PCM (silent due to zero-gain chain)
        const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
        buffer.getChannelData(0).set(float32);
        const src = audioCtx.createBufferSource();
        src.buffer = buffer;

        // Connect: src -> analyser -> silentGain(0) -> destination
        src.connect(outAnalyser);
        src.start();
        src.addEventListener('ended', () => {
            try { src.disconnect(); } catch { }
        });
    } catch (e) {
        console.log(e);
    }
});

$btnToggle.addEventListener('click', async () => {
    try {
        await convo.toggleStreaming();
    } catch (e) {
        alert('Failed to start: ' + (e?.message || e));
    }
});

// Toggle microphone capture (mute/unmute)
$btnMic.addEventListener('click', () => {
    try {
        const next = !convo.state.micMuted;
        convo.toggleMicMute();
        if (next) {
            // When muting: if capturing input stop only the input; if speaking without visualization, enable output visualization
            stopInputCapture();
            if (!waveformActive && convo.state.streamState === 'speaking') {
                startWaveformOutputOnly();
            }
        } else {
            // When unmuting: if a session is active, restore full capture
            if (convo.state.streaming) {
                startWaveform();
            }
        }
    } catch (e) {
        alert('Failed to toggle mic: ' + (e?.message || e));
    }
});

// Restart: clear history, reset metrics/thought/caption, and reconnect WebSocket
$btnRestart.addEventListener('click', async () => {
    try {
        await convo.restart();
    } catch (e) {
        alert('Failed to restart: ' + (e?.message || e));
    }
});

// Disconnect: stop streaming and close WebSocket
$btnStop.addEventListener('click', async () => {
    try {
        // Stop audio capture and the session
        if (convo.state.streaming) {
            await convo.toggleStreaming();
        }
        // Clear the history
        convo.clearHistory();
    } catch (e) {
        alert('Failed to disconnect: ' + (e?.message || e));
    }
});

// The upload button triggers the hidden file input
$btnUploadFile.addEventListener('click', () => {
    $fileInput.click();
});

// Handle file selection
$fileInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
        await convo.uploadFile(file);
    } catch (err) {
        alert('Failed to upload file: ' + (err?.message || err));
    }
    // Reset the input so the same file can be picked again
    $fileInput.value = '';
});

// Toggle chat history visibility
$btnView.addEventListener('click', () => {
    const hidden = $messages.classList.toggle('hidden');
    $btnView.textContent = hidden ? 'Show chat history' : 'Hide chat history';
    // When toggled to visible, only scroll the chat container to the bottom
    if (!hidden) {
        requestAnimationFrame(() => {
            try { $messages.scrollTop = $messages.scrollHeight; } catch { }
        });
    }
});

// Toggle Thought box visibility (hidden by default)
$btnThought.addEventListener('click', () => {
    const hidden = $thoughtCard.classList.toggle('hidden');
    $btnThought.textContent = hidden ? 'Show thought' : 'Hide thought';
});

// Toggle Caption box visibility (hidden by default)
$btnCaption.addEventListener('click', () => {
    const hidden = $captionCard.classList.toggle('hidden');
    $btnCaption.textContent = hidden ? 'Show caption' : 'Hide caption';
});

// Toggle Retrieval box visibility (hidden by default)
$btnRetrieval.addEventListener('click', () => {
    const hidden = $retrievalCard.classList.toggle('hidden');
    $btnRetrieval.textContent = hidden ? 'Show information' : 'Hide information';
});