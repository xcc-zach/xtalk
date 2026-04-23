async function loadXtalk() {
    try {
        return await import("../../xtalk/index.js");
    } catch (e) {
        console.log("Failed to load local xtalk-client, falling back to CDN:", e)
        return await import("https://unpkg.com/xtalk-client@latest/dist/index.js");
    }
}

const { createSession } = await loadXtalk();

function getWebSocketURL() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsPath = new URL("./ws", window.location.href);
    wsPath.protocol = proto;
    wsPath.host = window.location.host;
    return wsPath;
}

const session = createSession(getWebSocketURL());

const $btnStart = document.getElementById('btn-start');
const $btnStop = document.getElementById('btn-stop');
const $btnMute = document.getElementById('btn-mute');
const $voiceSelect = document.getElementById('voice-select');
const $btnUploadFile = document.getElementById('btn-upload-file');
const $fileInput = document.getElementById('file-input');
const $streamState = document.getElementById('stream-state');
const $sessionId = document.getElementById('session-id');
const $waveform = document.getElementById('waveform');
const $messages = document.getElementById('messages');
const $thoughtContent = document.getElementById('thought-content');
const $captionContent = document.getElementById('caption-content');
const $retrievalContent = document.getElementById('retrieval-content');
const $panelThought = document.getElementById('panel-thought');
const $panelCaption = document.getElementById('panel-caption');
const $panelRetrieval = document.getElementById('panel-retrieval');
const $btnToggleThought = document.getElementById('btn-toggle-thought');
const $btnToggleCaption = document.getElementById('btn-toggle-caption');
const $btnToggleRetrieval = document.getElementById('btn-toggle-retrieval');
const $latencyNetwork = document.getElementById('latency-network');
const $latencyAsr = document.getElementById('latency-asr');
const $latencyLlmFirst = document.getElementById('latency-llm-first');
const $latencyLlmSentence = document.getElementById('latency-llm-sentence');
const $latencyTts = document.getElementById('latency-tts');
const $latencyE2e = document.getElementById('latency-e2e');
const $btnToggleRecentAudio = document.getElementById('btn-toggle-recent-audio');
const $recentAudioCard = document.getElementById('recent-audio-card');
const $recentAudioStatus = document.getElementById('recent-audio-status');
const $recentAudioPlayer = document.getElementById('recent-audio-player');

let audioCtx = null;
let inputAnalyser = null;
let outputAnalyser = null;
let inputDataArray = null;
let outputDataArray = null;
let inputBufferLength = 0;
let outputBufferLength = 0;
let rafId = null;
let isActive = false;
let currentStreamState = 'idle';
let recentAudioObjectUrl = null;

const FULL_AUDIO_CHANNELS = 2;
const FULL_AUDIO_BYTES_PER_SAMPLE = 2;
const FULL_AUDIO_FRAME_BYTES = FULL_AUDIO_CHANNELS * FULL_AUDIO_BYTES_PER_SAMPLE;
const MAX_RECENT_AUDIO_SECONDS = 60;
let recentFullAudioSampleRate = 48000;
let recentFullAudioChunks = [];
let recentFullAudioTotalBytes = 0;
let recentAudioSnapshotDirty = false;

const canvasCtx = $waveform.getContext('2d');

const STATE_COLORS = {
    idle: '#6b7280',
    listening: '#34d399',
    processing: '#fbbf24',
    speaking: '#93c5fd'
};

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
}

function drawWaveform() {
    if (!isActive) return;
    rafId = requestAnimationFrame(drawWaveform);

    const w = $waveform.width;
    const h = $waveform.height;

    canvasCtx.fillStyle = '#0f172a';
    canvasCtx.fillRect(0, 0, w, h);

    canvasCtx.strokeStyle = '#1f2937';
    canvasCtx.lineWidth = 1;
    canvasCtx.beginPath();
    canvasCtx.moveTo(0, h / 2);
    canvasCtx.lineTo(w, h / 2);
    canvasCtx.stroke();

    const color = STATE_COLORS[currentStreamState] || '#6b7280';
    let dataArray = null;
    let bufferLength = 0;

    if (currentStreamState === 'speaking' && outputAnalyser && outputDataArray) {
        outputAnalyser.getByteTimeDomainData(outputDataArray);
        dataArray = outputDataArray;
        bufferLength = outputBufferLength;
    } else if (inputAnalyser && inputDataArray) {
        inputAnalyser.getByteTimeDomainData(inputDataArray);
        dataArray = inputDataArray;
        bufferLength = inputBufferLength;
    }

    if (dataArray && bufferLength) {
        const sliceWidth = w / bufferLength;
        canvasCtx.strokeStyle = color;
        canvasCtx.lineWidth = 2;
        canvasCtx.beginPath();
        let x = 0;
        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = (v * h) / 2;
            if (i === 0) canvasCtx.moveTo(x, y);
            else canvasCtx.lineTo(x, y);
            x += sliceWidth;
        }
        canvasCtx.lineTo(w, h / 2);
        canvasCtx.stroke();
    }
}

function startVisualization() {
    if (isActive) return;
    ensureAudioContext();
    resizeCanvas();
    isActive = true;
    drawWaveform();
}

function stopVisualization() {
    if (!isActive) return;
    isActive = false;
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    const w = $waveform.width;
    const h = $waveform.height;
    canvasCtx.fillStyle = '#0f172a';
    canvasCtx.fillRect(0, 0, w, h);
}

function updateRecentAudioStatus(text) {
    $recentAudioStatus.textContent = text;
}

function setRecentAudioVisible(visible) {
    $recentAudioCard.classList.toggle('is-hidden', !visible);
    $recentAudioCard.setAttribute('aria-hidden', visible ? 'false' : 'true');
    $btnToggleRecentAudio.textContent = visible ? 'Hide Recent Audio' : 'Recent 60s Audio';
}

function revokeRecentAudioUrl() {
    if (recentAudioObjectUrl) {
        URL.revokeObjectURL(recentAudioObjectUrl);
        recentAudioObjectUrl = null;
    }
}

function resetRecentAudioBuffer() {
    recentFullAudioSampleRate = 48000;
    recentFullAudioChunks = [];
    recentFullAudioTotalBytes = 0;
    recentAudioSnapshotDirty = false;
    revokeRecentAudioUrl();
    $recentAudioPlayer.removeAttribute('src');
    $recentAudioPlayer.load();
    updateRecentAudioStatus('Waiting for server full audio stream');
}

function trimRecentAudioBuffer(maxBytes) {
    while (recentFullAudioTotalBytes > maxBytes && recentFullAudioChunks.length > 0) {
        const overflowBytes = recentFullAudioTotalBytes - maxBytes;
        const firstChunk = recentFullAudioChunks[0];
        if (!firstChunk) {
            break;
        }
        if (overflowBytes >= firstChunk.length) {
            recentFullAudioChunks.shift();
            recentFullAudioTotalBytes -= firstChunk.length;
            continue;
        }
        const bytesToDrop = overflowBytes - (overflowBytes % FULL_AUDIO_FRAME_BYTES);
        if (bytesToDrop <= 0) {
            break;
        }
        recentFullAudioChunks[0] = firstChunk.slice(bytesToDrop);
        recentFullAudioTotalBytes -= bytesToDrop;
        break;
    }
}

function appendRecentFullAudioChunk(pcmChunkInt16, sampleRate) {
    if (!(pcmChunkInt16 instanceof ArrayBuffer) || pcmChunkInt16.byteLength === 0) {
        return;
    }
    if (recentFullAudioTotalBytes > 0 && sampleRate !== recentFullAudioSampleRate) {
        resetRecentAudioBuffer();
    }
    recentFullAudioSampleRate = sampleRate;
    const chunk = new Uint8Array(pcmChunkInt16.slice(0));
    recentFullAudioChunks.push(chunk);
    recentFullAudioTotalBytes += chunk.byteLength;
    const maxBytes = sampleRate * MAX_RECENT_AUDIO_SECONDS * FULL_AUDIO_FRAME_BYTES;
    trimRecentAudioBuffer(maxBytes);
    recentAudioSnapshotDirty = true;
    const bufferedSeconds = recentFullAudioTotalBytes / (sampleRate * FULL_AUDIO_FRAME_BYTES);
    updateRecentAudioStatus(`Buffered ${Math.min(MAX_RECENT_AUDIO_SECONDS, bufferedSeconds).toFixed(1)}s of server full audio`);
}

function flattenRecentAudioBuffer() {
    if (recentFullAudioTotalBytes <= 0) {
        return new Uint8Array(0);
    }
    const merged = new Uint8Array(recentFullAudioTotalBytes);
    let offset = 0;
    for (const chunk of recentFullAudioChunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    return merged;
}

function buildWavBlobFromPcm(pcmBytes, sampleRate, channels) {
    const header = new ArrayBuffer(44);
    const view = new DataView(header);
    const byteRate = sampleRate * channels * FULL_AUDIO_BYTES_PER_SAMPLE;
    const blockAlign = channels * FULL_AUDIO_BYTES_PER_SAMPLE;

    view.setUint32(0, 0x52494646, false);
    view.setUint32(4, 36 + pcmBytes.length, true);
    view.setUint32(8, 0x57415645, false);
    view.setUint32(12, 0x666d7420, false);
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true);
    view.setUint32(36, 0x64617461, false);
    view.setUint32(40, pcmBytes.length, true);

    return new Blob([header, pcmBytes], { type: 'audio/wav' });
}

function refreshRecentAudioSnapshot(force = false) {
    if (!force && !recentAudioSnapshotDirty) {
        return;
    }
    if (recentFullAudioTotalBytes <= 0) {
        updateRecentAudioStatus('Waiting for server full audio stream');
        return;
    }
    const pcmBytes = flattenRecentAudioBuffer();
    const wavBlob = buildWavBlobFromPcm(
        pcmBytes,
        recentFullAudioSampleRate,
        FULL_AUDIO_CHANNELS
    );
    const nextObjectUrl = URL.createObjectURL(wavBlob);
    const wasPlaying = !$recentAudioPlayer.paused && !$recentAudioPlayer.ended;
    revokeRecentAudioUrl();
    recentAudioObjectUrl = nextObjectUrl;
    $recentAudioPlayer.src = recentAudioObjectUrl;
    $recentAudioPlayer.load();
    recentAudioSnapshotDirty = false;
    if (wasPlaying) {
        $recentAudioPlayer.play().catch(() => { });
    }
}

session.onStateChange((state) => {
    $streamState.textContent = state.streamState;
    $sessionId.textContent = state.sessionId || '--';
    currentStreamState = state.streamState;

    $messages.innerHTML = '';
    for (const msg of state.messages) {
        const el = document.createElement('div');
        el.className = 'message message-' + msg.role;
        el.textContent = msg.content;
        $messages.appendChild(el);
    }
    $messages.scrollTop = $messages.scrollHeight;

    $thoughtContent.textContent = state.thought || '';
    $captionContent.textContent = state.caption || '';
    $retrievalContent.textContent = state.retrieval || '';

    const l = state.latency || {};
    $latencyNetwork.textContent = l.network ?? '--';
    $latencyAsr.textContent = l.asr ?? '--';
    $latencyLlmFirst.textContent = l.llmFirstToken ?? '--';
    $latencyLlmSentence.textContent = l.llmSentence ?? '--';
    $latencyTts.textContent = l.ttsFirstChunk ?? '--';
    const e2eParts = [l.network, l.asr, l.llmSentence, l.ttsFirstChunk];
    $latencyE2e.textContent = e2eParts.every(v => v != null) ? e2eParts.reduce((a, b) => a + b, 0) : '--';
});

session.onInputAudioChunk((pcmChunkInt16, sampleRate) => {
    try {
        ensureAudioContext();
        if (!inputAnalyser) {
            inputAnalyser = audioCtx.createAnalyser();
            inputAnalyser.fftSize = 1024;
            inputAnalyser.smoothingTimeConstant = 0.7;
            inputBufferLength = inputAnalyser.fftSize;
            inputDataArray = new Uint8Array(inputBufferLength);
        }

        const int16 = new Int16Array(pcmChunkInt16);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768;
        }

        const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
        buffer.getChannelData(0).set(float32);
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(inputAnalyser);
        const gain = audioCtx.createGain();
        gain.gain.value = 0;
        inputAnalyser.connect(gain);
        gain.connect(audioCtx.destination);
        source.start();
        source.addEventListener('ended', () => {
            try { source.disconnect(); } catch { }
        });
    } catch (e) {
        console.error('Input audio chunk error:', e);
    }
});

session.onOutputAudioChunk((pcmChunkInt16, sampleRate) => {
    try {
        ensureAudioContext();
        if (!outputAnalyser) {
            outputAnalyser = audioCtx.createAnalyser();
            outputAnalyser.fftSize = 1024;
            outputAnalyser.smoothingTimeConstant = 0.7;
            outputBufferLength = outputAnalyser.fftSize;
            outputDataArray = new Uint8Array(outputBufferLength);
        }

        const int16 = new Int16Array(pcmChunkInt16);
        const float32 = new Float32Array(int16.length);
        for (let i = 0; i < int16.length; i++) {
            float32[i] = int16[i] / 32768;
        }

        const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
        buffer.getChannelData(0).set(float32);
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(outputAnalyser);
        const gain = audioCtx.createGain();
        gain.gain.value = 0;
        outputAnalyser.connect(gain);
        gain.connect(audioCtx.destination);
        source.start();
        source.addEventListener('ended', () => {
            try { source.disconnect(); } catch { }
        });
    } catch (e) {
        console.error('Output audio chunk error:', e);
    }
});

session.onFullAudioChunk((pcmChunkInt16, sampleRate) => {
    appendRecentFullAudioChunk(pcmChunkInt16, sampleRate);
    const canRefreshSnapshot = $recentAudioPlayer.ended
        || ($recentAudioPlayer.paused && $recentAudioPlayer.currentTime === 0);
    if (canRefreshSnapshot) {
        refreshRecentAudioSnapshot();
    }
});

$btnStart.addEventListener('click', async () => {
    try {
        resetRecentAudioBuffer();
        await session.open();
        startVisualization();
        $btnStart.disabled = true;
        $btnStop.disabled = false;
    } catch (e) {
        alert('Failed to start: ' + (e?.message || e));
    }
});

$btnStop.addEventListener('click', async () => {
    try {
        await session.close();
        stopVisualization();
        $btnStart.disabled = false;
        $btnStop.disabled = true;
    } catch (e) {
        alert('Failed to stop: ' + (e?.message || e));
    }
});

$btnMute.addEventListener('click', () => {
    try {
        session.muted = !session.muted;
        $btnMute.textContent = session.muted ? 'Unmute' : 'Mute';
    } catch (e) {
        alert('Failed to toggle mute: ' + (e?.message || e));
    }
});

function setupToggle(btn, panel) {
    btn.addEventListener('click', () => {
        const active = btn.classList.toggle('active');
        panel.style.display = active ? '' : 'none';
    });
}
setupToggle($btnToggleThought, $panelThought);
setupToggle($btnToggleCaption, $panelCaption);
setupToggle($btnToggleRetrieval, $panelRetrieval);

$btnToggleRecentAudio.addEventListener('click', () => {
    const willOpen = $recentAudioCard.classList.contains('is-hidden');
    setRecentAudioVisible(willOpen);
    if (willOpen) {
        refreshRecentAudioSnapshot(true);
    }
});

window.addEventListener('resize', () => {
    resizeCanvas();
});

window.addEventListener('beforeunload', () => {
    revokeRecentAudioUrl();
});

$btnStop.disabled = true;
setRecentAudioVisible(false);

let availableAudios = [];

function syncVoiceSelectValue(targetName) {
    if (!$voiceSelect) return;
    const desired = targetName || session.state.currentVoiceName || '';
    if (!desired) return;
    if ($voiceSelect.value === desired) return;
    const hasOption = Array.from($voiceSelect.options).some(opt => opt.value === desired);
    if (hasOption) {
        $voiceSelect.value = desired;
    }
}

async function loadReferenceAudios() {
    try {
        const response = await fetch('./api/voices');
        const data = await response.json();
        availableAudios = data.audios || [];

        $voiceSelect.innerHTML = '<option value="" selected disabled hidden></option>';
        availableAudios.forEach((audio, index) => {
            const voiceName = audio.name || audio.path || `voice_${index}`;
            const option = document.createElement('option');
            option.value = voiceName;
            option.textContent = voiceName;
            option.dataset.path = audio.path || '';
            $voiceSelect.appendChild(option);
        });

        $voiceSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load reference audios:', error);
        $voiceSelect.innerHTML = '<option value="">Load failed</option>';
    }
}

$voiceSelect.addEventListener('change', (e) => {
    const selectedName = e.target.value;
    const selectedAudio = availableAudios.find(a => (a.name || a.path) === selectedName);
    if (selectedAudio) {
        const voiceName = selectedAudio.name || selectedName;
        session.changeVoice(voiceName);
        session.state.currentVoiceName = voiceName;
        session.state.currentVoicePath = selectedAudio.path || null;
        syncVoiceSelectValue(voiceName);
    }
});

$btnUploadFile.addEventListener('click', () => {
    $fileInput.click();
});

$fileInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
        await session.uploadFile(file);
    } catch (err) {
        alert('Failed to upload file: ' + (err?.message || err));
    }
    $fileInput.value = '';
});

loadReferenceAudios();
