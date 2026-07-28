const { createSession } = await import("../../xtalk/index.js");

function resolveAppURL(path) {
    const baseURL = new URL("./", window.location.href);
    return new URL(path.replace(/^\/+/, ""), baseURL);
}

function getWebSocketURL() {
    const url = resolveAppURL("ws");
    url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
    return url;
}

const frontendUtilitiesBaseUrl = resolveAppURL("xtalk/frontend-utilities").toString();
const inputConfig = {
    frontendUtilitiesBaseUrl,
    enableVAD: true,
};
const FRONTEND_VAD_ENABLED = inputConfig.enableVAD !== false;
const session = createSession(getWebSocketURL(), {
    inputConfig,
});

const $btnCall = document.getElementById('btn-call');
const $btnMute = document.getElementById('btn-mute');
const $btnNewSession = document.getElementById('btn-new-session');
const $btnRefreshSessions = document.getElementById('btn-refresh-sessions');
const $btnUploadFile = document.getElementById('btn-upload-file');
const $fileInput = document.getElementById('file-input');
const $voiceSelect = document.getElementById('voice-select');
const $sessionList = document.getElementById('session-list');
const $sessionListEmpty = document.getElementById('session-list-empty');
const $connectionState = document.getElementById('connection-state');
const $streamState = document.getElementById('stream-state');
const $mutedState = document.getElementById('muted-state');
const $sessionId = document.getElementById('session-id');
const $waveform = document.getElementById('waveform');
const $orbView = document.getElementById('orb-view');
const $chatView = document.getElementById('chat-view');
const $btnShowChat = document.getElementById('btn-show-chat');
const $btnShowOrb = document.getElementById('btn-show-orb');
const $messages = document.getElementById('messages');
const $sessionsDrawer = document.getElementById('sessions-drawer');
const $debugDrawer = document.getElementById('debug-drawer');
const $drawerBackdrop = document.getElementById('drawer-backdrop');
const $btnToggleSessions = document.getElementById('btn-toggle-sessions');
const $btnToggleDebug = document.getElementById('btn-toggle-debug');
const $btnCloseSessions = document.getElementById('btn-close-sessions');
const $btnCloseDebug = document.getElementById('btn-close-debug');
const $thoughtContent = document.getElementById('thought-content');
const $captionContent = document.getElementById('caption-content');
const $retrievalContent = document.getElementById('retrieval-content');
const $latencyNetwork = document.getElementById('latency-network');
const $latencyAsr = document.getElementById('latency-asr');
const $latencyLlmFirst = document.getElementById('latency-llm-first');
const $latencyLlmSentence = document.getElementById('latency-llm-sentence');
const $latencyTts = document.getElementById('latency-tts');
const $latencyE2e = document.getElementById('latency-e2e');
const $recentAudioDetails = document.getElementById('recent-audio-details');
const $recentAudioStatus = document.getElementById('recent-audio-status');
const $recentAudioPlayer = document.getElementById('recent-audio-player');
const $vadStatusDetails = document.getElementById('vad-status-details');
const $vadCurrentState = document.getElementById('vad-current-state');
const $vadTimelineClock = document.getElementById('vad-timeline-clock');
const $vadTimelineEvents = document.getElementById('vad-timeline-events');
const $vadRecordingCount = document.getElementById('vad-recording-count');
const $vadRecordingEmpty = document.getElementById('vad-recording-empty');
const $vadRecordingList = document.getElementById('vad-recording-list');
const $toastRegion = document.getElementById('toast-region');

let audioCtx = null;
let inputAnalyser = null;
let outputAnalyser = null;
let inputMonitorGain = null;
let outputMonitorGain = null;
let inputDataArray = null;
let outputDataArray = null;
let inputBufferLength = 0;
let outputBufferLength = 0;
let rafId = null;
let isActive = false;
let currentStreamState = 'idle';
let recentAudioObjectUrl = null;
let sessionsCache = [];
let previousSessionId = null;
let previousMessageCount = 0;
let isSessionListLoading = false;
let refreshSessionsTimer = null;
let timelineSessionId = null;
let toolCallCacheKey = '';
let chatTimeline = [];
let chatTimelineIndexByKey = new Map();
let isStarting = false;
let isStopping = false;
let isUploading = false;
let voiceOptionsLoaded = false;
let currentDrawer = null;
let currentMainView = 'orb';

const FULL_AUDIO_CHANNELS = 2;
const FULL_AUDIO_BYTES_PER_SAMPLE = 2;
const FULL_AUDIO_FRAME_BYTES = FULL_AUDIO_CHANNELS * FULL_AUDIO_BYTES_PER_SAMPLE;
const MAX_RECENT_AUDIO_SECONDS = 60;
let recentFullAudioSampleRate = 48000;
let recentFullAudioChunks = [];
let recentFullAudioTotalBytes = 0;
let recentAudioSnapshotDirty = false;

const VAD_TIMELINE_WINDOW_MS = 30_000;
const VAD_PRE_ROLL_MS = 300;
const MAX_VAD_RECORDINGS = 20;
let vadSegmentSequence = 0;
let vadStateActive = false;
let vadActiveRecording = null;
let vadTimelineSegments = [];
let vadRecordings = [];
let vadPreRollChunks = [];
let vadPreRollTotalBytes = 0;
let vadPreRollSampleRate = 16000;
let vadTimelineTimer = null;

const canvasCtx = $waveform.getContext('2d');

const STATE_COLORS = {
    idle: ['#7380ff', '#c8d0ff', '#f8fbff'],
    listening: ['#6978ff', '#aab8ff', '#f9fbff'],
    processing: ['#8b78ff', '#c2b8ff', '#fffaff'],
    speaking: ['#5e73ff', '#9eafff', '#f8fbff'],
};

function isLiveConnection() {
    return session.state.connectionState === 'connected'
        || session.state.connectionState === 'reconnecting';
}

function renderControls() {
    const isLive = isLiveConnection();
    const isMuted = session.muted;
    const isCallPending = isStarting || isStopping;
    const callAction = isLive ? 'stop' : 'start';
    const callLabel = isLive ? '停止对话' : '开始对话';
    const pendingLabel = isStopping ? '正在停止对话' : '正在开始对话';

    $btnCall.dataset.action = callAction;
    $btnCall.disabled = isCallPending;
    $btnCall.classList.toggle('is-loading', isCallPending);
    $btnCall.setAttribute('aria-label', isCallPending ? pendingLabel : callLabel);
    $btnCall.setAttribute('aria-busy', String(isCallPending));
    $btnCall.title = isCallPending ? pendingLabel : callLabel;

    $btnMute.disabled = !isLive || isCallPending;
    $btnMute.classList.toggle('is-muted', isMuted);
    $btnMute.setAttribute('aria-pressed', String(isMuted));
    $btnMute.setAttribute('aria-label', isMuted ? '打开麦克风' : '关闭麦克风');
    $btnMute.title = isMuted ? '打开麦克风' : '关闭麦克风';

    $btnUploadFile.disabled = isUploading || !session.state.sessionId;
    $btnUploadFile.classList.toggle('is-loading', isUploading);
    $voiceSelect.disabled = !voiceOptionsLoaded || !isLive;

    $connectionState.textContent = isStarting
        ? 'starting'
        : isStopping
            ? 'stopping'
            : session.state.connectionState;
    $streamState.textContent = session.state.streamState;
    $mutedState.textContent = String(isMuted);
    $sessionId.textContent = session.state.sessionId || '--';
    $btnShowChat.dataset.streamState = session.state.streamState;
    $btnShowChat.classList.toggle('is-muted', isMuted);
    $btnShowOrb.dataset.streamState = session.state.streamState;
    $btnShowOrb.classList.toggle('is-muted', isMuted);
}

function resetRealtimeUI() {
    stopVisualization();
    isStarting = false;
    isStopping = false;
    renderControls();
}

function showToast(message, type = 'error') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    $toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
}

function setMainView(view) {
    currentMainView = view === 'chat' ? 'chat' : 'orb';
    const showChat = currentMainView === 'chat';
    $orbView.classList.toggle('is-hidden', showChat);
    $chatView.classList.toggle('is-hidden', !showChat);
    $orbView.setAttribute('aria-hidden', String(showChat));
    $chatView.setAttribute('aria-hidden', String(!showChat));
    if (!showChat) {
        resizeCanvas();
        if (!isActive) drawWaveform(true);
    }
}

function setDrawer(drawer) {
    currentDrawer = drawer;
    const sessionsOpen = drawer === 'sessions';
    const debugOpen = drawer === 'debug';
    const anyOpen = sessionsOpen || debugOpen;

    document.body.dataset.drawer = drawer || '';
    $sessionsDrawer.classList.toggle('is-open', sessionsOpen);
    $debugDrawer.classList.toggle('is-open', debugOpen);
    $drawerBackdrop.classList.toggle('is-visible', anyOpen);
    $sessionsDrawer.setAttribute('aria-hidden', String(!sessionsOpen));
    $debugDrawer.setAttribute('aria-hidden', String(!debugOpen));
    $btnToggleSessions.setAttribute('aria-expanded', String(sessionsOpen));
    $btnToggleDebug.setAttribute('aria-expanded', String(debugOpen));
}

function formatSessionTitle(item) {
    const title = (item?.title || '').trim();
    if (title) {
        return title;
    }
    if (item?.session_id === session.state.sessionId) {
        return '新会话';
    }
    return `会话 ${String(item?.session_id || '').slice(0, 8) || '--'}`;
}

function renderSessions() {
    const activeSessionId = session.state.sessionId;
    $sessionList.innerHTML = '';
    $sessionListEmpty.style.display = sessionsCache.length === 0 ? '' : 'none';

    for (const item of sessionsCache) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'session-item';
        if (item.session_id === activeSessionId) {
            button.classList.add('active');
        }

        const title = document.createElement('div');
        title.className = 'session-title';
        title.textContent = formatSessionTitle(item);

        button.appendChild(title);
        button.addEventListener('click', async () => {
            if (item.session_id === session.state.sessionId) {
                setDrawer(null);
                return;
            }
            try {
                resetRecentAudioBuffer();
                resetRealtimeUI();
                await session.switchSession(item.session_id);
                renderSessions();
                setMainView('chat');
                setDrawer(null);
            } catch (error) {
                showToast('切换会话失败：' + (error?.message || error));
            }
        });

        $sessionList.appendChild(button);
    }
}

async function refreshSessions({ preserveSelection = true } = {}) {
    if (isSessionListLoading) {
        return;
    }
    isSessionListLoading = true;
    $btnRefreshSessions.disabled = true;
    try {
        const sessions = await session.getSessions();
        sessionsCache = Array.isArray(sessions) ? sessions : [];
        $sessionListEmpty.textContent = 'No sessions yet.';
        renderSessions();
        if (!preserveSelection && session.state.sessionId) {
            const exists = sessionsCache.some((item) => item.session_id === session.state.sessionId);
            if (!exists) {
                await session.switchSession(null);
            }
        }
    } catch (error) {
        console.error('Failed to load sessions:', error);
        $sessionListEmpty.textContent = 'Failed to load sessions.';
        $sessionListEmpty.style.display = '';
    } finally {
        isSessionListLoading = false;
        $btnRefreshSessions.disabled = false;
    }
}

function scheduleRefreshSessions() {
    if (refreshSessionsTimer) {
        clearTimeout(refreshSessionsTimer);
    }
    refreshSessionsTimer = setTimeout(() => {
        refreshSessionsTimer = null;
        refreshSessions().catch((error) => {
            console.error('Failed to refresh sessions:', error);
        });
    }, 300);
}

function ensureAudioContext() {
    if (!audioCtx) {
        const AC = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AC();
    }
    return audioCtx;
}

function resetChatTimeline(sessionId) {
    timelineSessionId = sessionId;
    toolCallCacheKey = '';
    chatTimeline = [];
    chatTimelineIndexByKey = new Map();
}

function getConversationMessageKey(message, index) {
    if (message.role === 'info') {
        return `info:${index}`;
    }
    return `${message.role}:index:${index}`;
}

function syncConversationMessages(messages) {
    messages.forEach((message, index) => {
        const key = getConversationMessageKey(message, index);
        const timelineIndex = chatTimelineIndexByKey.get(key);
        if (timelineIndex != null) {
            chatTimeline[timelineIndex] = {
                ...chatTimeline[timelineIndex],
                role: message.role,
                content: message.content,
                final: message.final,
            };
            return;
        }

        chatTimelineIndexByKey.set(key, chatTimeline.length);
        chatTimeline.push({
            kind: 'conversation',
            key,
            role: message.role,
            content: message.content,
            final: message.final,
        });
    });
}

function normalizeToolCall(toolCall) {
    return {
        name: typeof toolCall?.name === 'string' ? toolCall.name : '',
        args: toolCall && typeof toolCall.args === 'object' && toolCall.args !== null
            ? toolCall.args
            : {},
    };
}

function formatToolCallArgs(args) {
    try {
        return JSON.stringify(args ?? {}, null, 2);
    } catch {
        return String(args ?? '{}');
    }
}

function buildToolCallCacheKey(toolCall) {
    if (!toolCall.name) {
        return '';
    }
    return `${toolCall.name}\n${formatToolCallArgs(toolCall.args)}`;
}

function appendToolCallMessage(toolCall) {
    const argsText = formatToolCallArgs(toolCall.args);
    chatTimeline.push({
        kind: 'tool',
        key: `tool:${chatTimeline.length}:${toolCall.name}`,
        name: toolCall.name,
        argsText,
    });
}

function appendLocalInfoMessage(content) {
    chatTimeline.push({
        kind: 'conversation',
        key: `local-info:${chatTimeline.length}`,
        role: 'info',
        content,
    });
}

async function copyMessageText(text) {
    try {
        if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(text);
        } else {
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.setAttribute('readonly', '');
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            document.body.appendChild(textArea);
            textArea.select();
            const copied = document.execCommand('copy');
            textArea.remove();
            if (!copied) throw new Error('Clipboard API unavailable');
        }
        return true;
    } catch (error) {
        showToast('复制失败：' + (error?.message || error));
        return false;
    }
}

function createMessageCopyButton(text) {
    const button = document.createElement('button');
    let feedbackTimer = null;
    button.type = 'button';
    button.className = 'message-copy-button';
    button.setAttribute('aria-label', '复制消息');
    button.title = '复制消息';
    button.innerHTML = '<svg class="icon-copy" viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="2"></rect><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"></path></svg><svg class="icon-check" viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"></path></svg>';
    button.addEventListener('click', async () => {
        const copied = await copyMessageText(text);
        if (!copied) return;

        button.classList.add('is-copied');
        button.setAttribute('aria-label', '已复制');
        button.title = '已复制';
        if (feedbackTimer) window.clearTimeout(feedbackTimer);
        feedbackTimer = window.setTimeout(() => {
            button.classList.remove('is-copied');
            button.setAttribute('aria-label', '复制消息');
            button.title = '复制消息';
            feedbackTimer = null;
        }, 1600);
    });
    return button;
}

function renderChatTimeline() {
    $messages.innerHTML = '';
    for (const entry of chatTimeline) {
        const role = entry.kind === 'tool' ? 'tool' : entry.role;
        const row = document.createElement('div');
        row.className = 'message-row message-row-' + role;

        const message = document.createElement('div');
        let copyText = '';
        if (entry.kind === 'tool') {
            message.className = 'message message-tool';

            const label = document.createElement('div');
            label.className = 'message-tool-label';
            label.textContent = `Tool Call: ${entry.name}`;

            const args = document.createElement('pre');
            args.className = 'message-tool-args';
            args.textContent = entry.argsText;

            message.appendChild(label);
            message.appendChild(args);
            copyText = `Tool Call: ${entry.name}\n${entry.argsText}`;
        } else {
            message.className = 'message message-' + entry.role;
            message.textContent = entry.content;
            copyText = entry.content;
        }
        row.appendChild(message);

        if (role !== 'info') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            actions.appendChild(createMessageCopyButton(copyText));
            row.appendChild(actions);
        }
        $messages.appendChild(row);
    }
    $messages.scrollTop = $messages.scrollHeight;
}

function ensureInputAnalyser() {
    ensureAudioContext();
    if (!inputAnalyser) {
        inputAnalyser = audioCtx.createAnalyser();
        inputAnalyser.fftSize = 1024;
        inputAnalyser.smoothingTimeConstant = 0.7;
        inputBufferLength = inputAnalyser.fftSize;
        inputDataArray = new Uint8Array(inputBufferLength);
    }
    if (!inputMonitorGain) {
        inputMonitorGain = audioCtx.createGain();
        inputMonitorGain.gain.value = 0;
        inputAnalyser.connect(inputMonitorGain);
        inputMonitorGain.connect(audioCtx.destination);
    }
    return inputAnalyser;
}

function ensureOutputAnalyser() {
    ensureAudioContext();
    if (!outputAnalyser) {
        outputAnalyser = audioCtx.createAnalyser();
        outputAnalyser.fftSize = 1024;
        outputAnalyser.smoothingTimeConstant = 0.7;
        outputBufferLength = outputAnalyser.fftSize;
        outputDataArray = new Uint8Array(outputBufferLength);
    }
    if (!outputMonitorGain) {
        outputMonitorGain = audioCtx.createGain();
        outputMonitorGain.gain.value = 0;
        outputAnalyser.connect(outputMonitorGain);
        outputMonitorGain.connect(audioCtx.destination);
    }
    return outputAnalyser;
}

function playPcmChunkThroughAnalyser(pcmChunkInt16, sampleRate, analyser) {
    const int16 = new Int16Array(pcmChunkInt16);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) {
        float32[i] = int16[i] / 32768;
    }

    const buffer = audioCtx.createBuffer(1, float32.length, sampleRate);
    buffer.getChannelData(0).set(float32);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(analyser);
    source.onended = () => {
        source.onended = null;
        try { source.disconnect(); } catch { }
    };
    source.start();
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

function drawWaveform(renderOnce = false) {
    if (!isActive && !renderOnce) return;
    if (!renderOnce) {
        rafId = requestAnimationFrame(() => drawWaveform(false));
    }

    resizeCanvas();
    const w = $waveform.width;
    const h = $waveform.height;
    const dpr = window.devicePixelRatio || 1;
    const centerX = w / 2;
    const centerY = h / 2;
    const palette = STATE_COLORS[currentStreamState] || STATE_COLORS.idle;
    const time = performance.now() / 1000;
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

    let energy = 0;
    if (dataArray && bufferLength) {
        const stride = Math.max(1, Math.floor(bufferLength / 256));
        let samples = 0;
        for (let i = 0; i < bufferLength; i += stride) {
            const sample = (dataArray[i] - 128) / 128;
            energy += sample * sample;
            samples += 1;
        }
        energy = samples > 0 ? Math.sqrt(energy / samples) : 0;
    }

    const statePulse = currentStreamState === 'processing'
        ? 0.018 * Math.sin(time * 3.2)
        : currentStreamState === 'speaking'
            ? 0.014 * Math.sin(time * 5.2)
            : 0.008 * Math.sin(time * 1.8);
    const audioScale = Math.min(0.14, energy * 0.52);
    const radius = Math.min(w, h) * 0.39 * (1 + statePulse + audioScale);
    const isMuted = session.muted;

    canvasCtx.clearRect(0, 0, w, h);
    canvasCtx.save();
    canvasCtx.globalAlpha = isMuted ? 0.48 : 1;
    canvasCtx.shadowColor = 'rgba(104, 119, 255, 0.34)';
    canvasCtx.shadowBlur = (24 + energy * 70) * dpr;

    const baseGradient = canvasCtx.createLinearGradient(
        centerX - radius * 0.65,
        centerY - radius,
        centerX + radius * 0.55,
        centerY + radius,
    );
    baseGradient.addColorStop(0, palette[0]);
    baseGradient.addColorStop(0.56, palette[1]);
    baseGradient.addColorStop(1, palette[2]);
    canvasCtx.fillStyle = baseGradient;
    canvasCtx.beginPath();
    canvasCtx.arc(centerX, centerY, radius, 0, Math.PI * 2);
    canvasCtx.fill();

    canvasCtx.shadowBlur = 0;
    canvasCtx.clip();
    const cloudX = centerX + Math.sin(time * 0.48) * radius * 0.08;
    const cloudY = centerY + radius * 0.3 + Math.cos(time * 0.42) * radius * 0.05;
    const cloudGradient = canvasCtx.createRadialGradient(
        cloudX,
        cloudY,
        radius * 0.03,
        cloudX,
        cloudY,
        radius * 0.78,
    );
    cloudGradient.addColorStop(0, 'rgba(255, 255, 255, 0.92)');
    cloudGradient.addColorStop(0.32, 'rgba(255, 255, 255, 0.5)');
    cloudGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    canvasCtx.fillStyle = cloudGradient;
    canvasCtx.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);

    const sheenGradient = canvasCtx.createRadialGradient(
        centerX - radius * 0.42,
        centerY - radius * 0.45,
        0,
        centerX - radius * 0.42,
        centerY - radius * 0.45,
        radius * 0.72,
    );
    sheenGradient.addColorStop(0, 'rgba(255, 255, 255, 0.26)');
    sheenGradient.addColorStop(1, 'rgba(255, 255, 255, 0)');
    canvasCtx.fillStyle = sheenGradient;
    canvasCtx.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);
    canvasCtx.restore();
}

function startVisualization() {
    if (isActive) return;
    ensureAudioContext();
    resizeCanvas();
    isActive = true;
    drawWaveform();
}

function stopVisualization() {
    isActive = false;
    if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = null;
    }
    resizeCanvas();
    drawWaveform(true);
}

function updateRecentAudioStatus(text) {
    $recentAudioStatus.textContent = text;
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

function formatVadTimestamp(timestamp) {
    const date = new Date(timestamp);
    const time = date.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
    return `${time}.${String(date.getMilliseconds()).padStart(3, '0')}`;
}

function resetVadPreRoll(sampleRate = 16000) {
    vadPreRollChunks = [];
    vadPreRollTotalBytes = 0;
    vadPreRollSampleRate = sampleRate;
}

function trimVadPreRoll() {
    const maxBytes = Math.ceil(
        vadPreRollSampleRate * FULL_AUDIO_BYTES_PER_SAMPLE * VAD_PRE_ROLL_MS / 1000
    );
    while (vadPreRollTotalBytes > maxBytes && vadPreRollChunks.length > 0) {
        const overflowBytes = vadPreRollTotalBytes - maxBytes;
        const firstChunk = vadPreRollChunks[0];
        if (overflowBytes >= firstChunk.byteLength) {
            vadPreRollChunks.shift();
            vadPreRollTotalBytes -= firstChunk.byteLength;
            continue;
        }
        const bytesToDrop = overflowBytes - (overflowBytes % FULL_AUDIO_BYTES_PER_SAMPLE);
        if (bytesToDrop <= 0) {
            break;
        }
        vadPreRollChunks[0] = firstChunk.slice(bytesToDrop);
        vadPreRollTotalBytes -= bytesToDrop;
        break;
    }
}

function appendVadInputAudioChunk(pcmChunkInt16, sampleRate) {
    if (!FRONTEND_VAD_ENABLED
        || !(pcmChunkInt16 instanceof ArrayBuffer)
        || pcmChunkInt16.byteLength === 0) {
        return;
    }
    if (vadPreRollChunks.length > 0 && sampleRate !== vadPreRollSampleRate) {
        resetVadPreRoll(sampleRate);
    }
    vadPreRollSampleRate = sampleRate;
    const chunk = new Uint8Array(pcmChunkInt16.slice(0));
    if (vadActiveRecording && vadActiveRecording.sampleRate === sampleRate) {
        vadActiveRecording.chunks.push(chunk);
        vadActiveRecording.totalBytes += chunk.byteLength;
    }
    vadPreRollChunks.push(chunk);
    vadPreRollTotalBytes += chunk.byteLength;
    trimVadPreRoll();
}

function flattenVadRecording(recording) {
    const pcmBytes = new Uint8Array(recording.totalBytes);
    let offset = 0;
    for (const chunk of recording.chunks) {
        pcmBytes.set(chunk, offset);
        offset += chunk.byteLength;
    }
    return pcmBytes;
}

function renderVadTimeline() {
    if (!FRONTEND_VAD_ENABLED) {
        return;
    }
    const now = Date.now();
    const windowStart = now - VAD_TIMELINE_WINDOW_MS;
    vadTimelineSegments = vadTimelineSegments.filter(
        (segment) => (segment.endAt ?? now) >= windowStart
    );
    $vadTimelineClock.textContent = formatVadTimestamp(now);
    $vadCurrentState.textContent = vadStateActive ? 'speech' : 'idle';
    $vadCurrentState.classList.toggle('is-active', vadStateActive);
    $vadTimelineEvents.replaceChildren();

    for (const segment of vadTimelineSegments) {
        const visibleStart = Math.max(segment.startAt, windowStart);
        const visibleEnd = Math.min(segment.endAt ?? now, now);
        const startPercent = (visibleStart - windowStart) / VAD_TIMELINE_WINDOW_MS * 100;
        const endPercent = (visibleEnd - windowStart) / VAD_TIMELINE_WINDOW_MS * 100;
        const bar = document.createElement('span');
        bar.className = 'vad-timeline-segment';
        if (segment.endAt === null) {
            bar.classList.add('is-active');
        }
        bar.style.left = `${Math.max(0, startPercent)}%`;
        bar.style.width = `${Math.max(1.2, endPercent - startPercent)}%`;
        bar.title = segment.endAt === null
            ? `开始 ${formatVadTimestamp(segment.startAt)} · 进行中`
            : `开始 ${formatVadTimestamp(segment.startAt)} · 结束 ${formatVadTimestamp(segment.endAt)}`;
        $vadTimelineEvents.appendChild(bar);

        if (segment.startAt >= windowStart) {
            const startMarker = document.createElement('span');
            startMarker.className = 'vad-timeline-marker is-start';
            startMarker.style.left = `${Math.min(98.5, Math.max(1.5, startPercent))}%`;
            startMarker.dataset.label = 'S';
            startMarker.title = `VAD start ${formatVadTimestamp(segment.startAt)}`;
            $vadTimelineEvents.appendChild(startMarker);
        }
        if (segment.endAt !== null && segment.endAt >= windowStart) {
            const endMarker = document.createElement('span');
            endMarker.className = 'vad-timeline-marker is-end';
            endMarker.style.left = `${Math.min(98.5, Math.max(1.5, endPercent))}%`;
            endMarker.dataset.label = 'E';
            endMarker.title = `VAD end ${formatVadTimestamp(segment.endAt)}`;
            $vadTimelineEvents.appendChild(endMarker);
        }
    }
}

function renderVadRecordings() {
    $vadRecordingList.replaceChildren();
    $vadRecordingCount.textContent = `${vadRecordings.length} clips`;
    $vadRecordingEmpty.hidden = vadRecordings.length > 0;

    for (const recording of vadRecordings) {
        const item = document.createElement('li');
        item.className = 'vad-recording-item';

        const meta = document.createElement('div');
        meta.className = 'vad-recording-meta';
        const timestamp = document.createElement('span');
        timestamp.textContent = `${formatVadTimestamp(recording.startAt)}`
            + ` → ${formatVadTimestamp(recording.endAt)}`;
        const duration = document.createElement('span');
        duration.textContent = `${(recording.durationMs / 1000).toFixed(2)}s`;
        meta.append(timestamp, duration);
        item.appendChild(meta);

        if (recording.objectUrl) {
            const player = document.createElement('audio');
            player.className = 'vad-recording-player';
            player.controls = true;
            player.preload = 'metadata';
            player.src = recording.objectUrl;
            item.appendChild(player);
        } else {
            const unavailable = document.createElement('div');
            unavailable.className = 'vad-recording-unavailable';
            unavailable.textContent = 'No PCM captured';
            item.appendChild(unavailable);
        }
        $vadRecordingList.appendChild(item);
    }
}

function startVadSegment(timestamp = Date.now()) {
    if (!FRONTEND_VAD_ENABLED || vadStateActive) {
        return;
    }
    const timelineSegment = {
        id: ++vadSegmentSequence,
        startAt: timestamp,
        endAt: null,
    };
    vadTimelineSegments.push(timelineSegment);
    vadActiveRecording = {
        timelineSegment,
        sampleRate: vadPreRollSampleRate,
        chunks: vadPreRollChunks.map((chunk) => chunk.slice()),
        totalBytes: vadPreRollTotalBytes,
    };
    vadStateActive = true;
    renderVadTimeline();
}

function finishVadSegment(timestamp = Date.now()) {
    if (!vadStateActive || !vadActiveRecording) {
        return;
    }
    const recording = vadActiveRecording;
    recording.timelineSegment.endAt = Math.max(
        timestamp,
        recording.timelineSegment.startAt
    );
    const pcmBytes = flattenVadRecording(recording);
    const durationMs = pcmBytes.byteLength
        / (recording.sampleRate * FULL_AUDIO_BYTES_PER_SAMPLE) * 1000;
    const objectUrl = pcmBytes.byteLength > 0
        ? URL.createObjectURL(buildWavBlobFromPcm(pcmBytes, recording.sampleRate, 1))
        : null;
    vadRecordings.unshift({
        id: recording.timelineSegment.id,
        startAt: recording.timelineSegment.startAt,
        endAt: recording.timelineSegment.endAt,
        durationMs,
        objectUrl,
    });
    const expiredRecordings = vadRecordings.splice(MAX_VAD_RECORDINGS);
    for (const expired of expiredRecordings) {
        if (expired.objectUrl) {
            URL.revokeObjectURL(expired.objectUrl);
        }
    }
    vadActiveRecording = null;
    vadStateActive = false;
    renderVadTimeline();
    renderVadRecordings();
}

function syncFrontendVadState(streamState, connectionState) {
    if (!FRONTEND_VAD_ENABLED) {
        return;
    }
    const nextActive = connectionState !== 'disconnected' && streamState === 'listening';
    if (nextActive && !vadStateActive) {
        startVadSegment();
    } else if (!nextActive && vadStateActive) {
        finishVadSegment();
    }
}

function resetVadDiagnostics() {
    for (const recording of vadRecordings) {
        if (recording.objectUrl) {
            URL.revokeObjectURL(recording.objectUrl);
        }
    }
    vadSegmentSequence = 0;
    vadStateActive = false;
    vadActiveRecording = null;
    vadTimelineSegments = [];
    vadRecordings = [];
    resetVadPreRoll();
    renderVadTimeline();
    renderVadRecordings();
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
    currentStreamState = state.streamState;
    syncFrontendVadState(state.streamState, state.connectionState);
    renderControls();
    renderSessions();
    if (!isActive) {
        drawWaveform(true);
    }

    if (state.sessionId !== timelineSessionId) {
        resetChatTimeline(state.sessionId);
    }
    syncConversationMessages(state.messages);
    const nextToolCall = normalizeToolCall(state.tool_call);
    const nextToolCallKey = buildToolCallCacheKey(nextToolCall);
    if (nextToolCallKey && nextToolCallKey !== toolCallCacheKey) {
        appendToolCallMessage(nextToolCall);
    }
    toolCallCacheKey = nextToolCallKey;
    renderChatTimeline();

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

    if (state.sessionId !== previousSessionId) {
        previousSessionId = state.sessionId;
        scheduleRefreshSessions();
    } else if (state.sessionId && state.messages.length !== previousMessageCount) {
        scheduleRefreshSessions();
    }
    previousMessageCount = state.messages.length;
});

session.onInputAudioChunk((pcmChunkInt16, sampleRate) => {
    try {
        appendVadInputAudioChunk(pcmChunkInt16, sampleRate);
        const analyser = ensureInputAnalyser();
        playPcmChunkThroughAnalyser(pcmChunkInt16, sampleRate, analyser);
    } catch (e) {
        console.error('Input audio chunk error:', e);
    }
});

session.onOutputAudioChunk((pcmChunkInt16, sampleRate) => {
    try {
        const analyser = ensureOutputAnalyser();
        playPcmChunkThroughAnalyser(pcmChunkInt16, sampleRate, analyser);
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

$btnCall.addEventListener('click', async () => {
    if (isStarting) return;

    if (isLiveConnection()) {
        isStopping = true;
        renderControls();
        try {
            await session.close();
            stopVisualization();
        } catch (error) {
            showToast('停止对话失败：' + (error?.message || error));
        } finally {
            isStopping = false;
            renderControls();
        }
        return;
    }

    isStarting = true;
    renderControls();
    try {
        resetRecentAudioBuffer();
        resetVadDiagnostics();
        await session.open();
        startVisualization();
        setMainView('orb');
        await refreshSessions();
    } catch (error) {
        showToast('开始对话失败：' + (error?.message || error));
    } finally {
        isStarting = false;
        renderControls();
    }
});

$btnMute.addEventListener('click', () => {
    if (!isLiveConnection()) return;
    session.muted = !session.muted;
    renderControls();
});

$recentAudioDetails.addEventListener('toggle', () => {
    if ($recentAudioDetails.open) {
        refreshRecentAudioSnapshot(true);
    }
});

$btnRefreshSessions.addEventListener('click', async () => {
    await refreshSessions();
});

$btnNewSession.addEventListener('click', async () => {
    try {
        const hadActiveSession = session.state.connectionState === 'connected'
            || session.state.connectionState === 'reconnecting';
        if (hadActiveSession) {
            await session.close();
        }
        resetRealtimeUI();
        resetVadDiagnostics();
        await session.switchSession(null);
        if (hadActiveSession) {
            appendLocalInfoMessage('Previous session stopped.');
            renderChatTimeline();
        }
        renderSessions();
        setMainView('orb');
        setDrawer(null);
    } catch (error) {
        showToast('新建会话失败：' + (error?.message || error));
    }
});

$btnToggleSessions.addEventListener('click', () => {
    setDrawer(currentDrawer === 'sessions' ? null : 'sessions');
});

$btnToggleDebug.addEventListener('click', () => {
    setDrawer(currentDrawer === 'debug' ? null : 'debug');
});

$btnCloseSessions.addEventListener('click', () => setDrawer(null));
$btnCloseDebug.addEventListener('click', () => setDrawer(null));
$drawerBackdrop.addEventListener('click', () => setDrawer(null));
$btnShowChat.addEventListener('click', () => setMainView('chat'));
$btnShowOrb.addEventListener('click', () => setMainView('orb'));

document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && currentDrawer) {
        setDrawer(null);
    }
});

window.addEventListener('resize', () => {
    resizeCanvas();
    if (!isActive) drawWaveform(true);
});

window.addEventListener('beforeunload', () => {
    if (refreshSessionsTimer) {
        clearTimeout(refreshSessionsTimer);
        refreshSessionsTimer = null;
    }
    revokeRecentAudioUrl();
    if (vadTimelineTimer) {
        clearInterval(vadTimelineTimer);
        vadTimelineTimer = null;
    }
    for (const recording of vadRecordings) {
        if (recording.objectUrl) {
            URL.revokeObjectURL(recording.objectUrl);
        }
    }
});

$vadStatusDetails.hidden = !FRONTEND_VAD_ENABLED;
if (FRONTEND_VAD_ENABLED) {
    renderVadTimeline();
    renderVadRecordings();
    vadTimelineTimer = window.setInterval(renderVadTimeline, 200);
}
renderControls();
resizeCanvas();
drawWaveform(true);

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

        voiceOptionsLoaded = availableAudios.length > 0;
        renderControls();
    } catch (error) {
        console.error('Failed to load reference audios:', error);
        voiceOptionsLoaded = false;
        $voiceSelect.innerHTML = '<option value="">Load failed</option>';
        renderControls();
    }
}

$voiceSelect.addEventListener('change', async (event) => {
    const selectedName = event.target.value;
    const selectedAudio = availableAudios.find((audio) => (audio.name || audio.path) === selectedName);
    if (!selectedAudio) return;

    const voiceName = selectedAudio.name || selectedName;
    $voiceSelect.disabled = true;
    try {
        await session.changeVoice(voiceName);
        session.state.currentVoiceName = voiceName;
        session.state.currentVoicePath = selectedAudio.path || null;
        syncVoiceSelectValue(voiceName);
        showToast(`已切换语音：${voiceName}`, 'success');
    } catch (error) {
        showToast('切换语音失败：' + (error?.message || error));
        syncVoiceSelectValue(session.state.currentVoiceName);
    } finally {
        renderControls();
    }
});

$btnUploadFile.addEventListener('click', () => {
    if (!session.state.sessionId) {
        showToast('请先开始或选择一个会话。');
        return;
    }
    $fileInput.click();
});

$fileInput.addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    isUploading = true;
    renderControls();
    try {
        await session.uploadFile(file);
        showToast(`已上传：${file.name}`, 'success');
    } catch (error) {
        showToast('上传失败：' + (error?.message || error));
    } finally {
        isUploading = false;
        renderControls();
        $fileInput.value = '';
    }
});

loadReferenceAudios();
refreshSessions().catch((error) => {
    console.error('Initial session load failed:', error);
    showToast('加载会话列表失败：' + (error?.message || error));
});
