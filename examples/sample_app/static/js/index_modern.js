async function loadXtalk() {
    try {
        return await import("../../xtalk/index.js");
    } catch (e) {
        return await import("https://unpkg.com/xtalk-client@latest/dist/index.js");
    }
}

const { createConversation } = await loadXtalk();
function getWebSocketURL() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsPath = new URL("./ws", window.location.href);
    wsPath.protocol = proto;
    wsPath.host = window.location.host;
    return wsPath
}

const convo = createConversation(getWebSocketURL());

// --- UI Elements ---
const $orbArea = document.getElementById('orb-area');
const $orbVisual = document.getElementById('orb-visual');
const $statusDot = document.querySelector('.status-dot');
const $statusText = document.getElementById('status-text');
const $helperText = document.getElementById('helper-text');

const $liveCaption = document.getElementById('live-caption'); // Assistant (small, gray)
const $liveThought = document.getElementById('live-thought'); // User (big, white)

const $btnMain = document.getElementById('btn-main-mic');
const $iconPlay = document.getElementById('icon-play');
const $iconMic = document.getElementById('icon-mic');
const $iconMuted = document.getElementById('icon-muted');
const $micWaves = document.getElementById('mic-waves');
const $micLoading = document.getElementById('mic-loading');

const $btnSettingsToggle = document.getElementById('btn-settings-toggle');
const $settingsModal = document.getElementById('settings-modal');
const $btnUploadFile = document.getElementById('btn-upload-file');
const $fileInput = document.getElementById('file-input');
const $btnStop = document.getElementById('btn-stop');
const $btnRestart = document.getElementById('btn-restart');
const $voiceSelect = document.getElementById('voice-select');
const $speedSelect = document.getElementById('speed-select');
const $ttsModelSelect = document.getElementById('tts-model-select');
const $llmModelSelect = document.getElementById('llm-model-select');

const $overlay = document.getElementById('overlay-panel');
const $btnToggleChatTop = document.getElementById('btn-toggle-chat-top');
const $btnCloseOverlay = document.getElementById('btn-close-overlay');

const $tabBtns = document.querySelectorAll('.tab-btn');
const $tabPanes = document.querySelectorAll('.tab-pane');
const $chatList = document.getElementById('chat-list');
const $captionLogList = document.getElementById('caption-log-list');
const $thoughtList = document.getElementById('thought-list');
const $retrievalList = document.getElementById('retrieval-list');

// --- State ---
let currentVisualState = 'idle';
let rafId = null;
let isSettingsOpen = false;
let isOverlayOpen = false;
// Local log cache (avoid redundant renders)
const captionLog = [];
const thoughtLog = [];
const retrievalLog = [];
let lastCaption = '';
let lastThought = '';
let lastRetrieval = '';
let lastRenderedChatCount = null;

function resetHistoryLogs() {
    captionLog.length = 0;
    thoughtLog.length = 0;
    retrievalLog.length = 0;
    lastCaption = '';
    lastThought = '';
    lastRetrieval = '';
    lastRenderedChatCount = null;
}

// --- Logic ---

// 1. Settings Toggle
$btnSettingsToggle.addEventListener('click', (e) => {
    e.stopPropagation();
    isSettingsOpen = !isSettingsOpen;
    if (isSettingsOpen) $settingsModal.classList.add('open');
    else $settingsModal.classList.remove('open');
});
document.addEventListener('click', (e) => {
    if (isSettingsOpen && !$settingsModal.contains(e.target) && e.target !== $btnSettingsToggle) {
        isSettingsOpen = false;
        $settingsModal.classList.remove('open');
    }
});

$btnUploadFile.addEventListener('click', () => {
    if (!$fileInput) return;
    $fileInput.value = '';
    $fileInput.click();
});

$fileInput.addEventListener('change', async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    try {
        await convo.uploadFile(file);
    } catch (err) {
        console.error('Upload failed', err);
    }
});

// 2. Overlay Toggle (Orb Click & Top Button)
function toggleOverlay() {
    isOverlayOpen = !isOverlayOpen;
    if (isOverlayOpen) $overlay.classList.add('show');
    else $overlay.classList.remove('show');
}
$orbArea.addEventListener('click', (e) => { e.stopPropagation(); toggleOverlay(); });
$btnToggleChatTop.addEventListener('click', (e) => { e.stopPropagation(); toggleOverlay(); });
$btnCloseOverlay.addEventListener('click', (e) => { e.stopPropagation(); toggleOverlay(); });
// Tap the “Session History” title or blank content to return to the Orb view
const $overlayHeader = document.querySelector('.overlay-header');
$overlayHeader?.addEventListener('click', (e) => {
    // Prevent double-trigger when the close button is clicked
    if (e.target.closest('#btn-close-overlay')) return;
    e.stopPropagation();
    toggleOverlay();
});
// Clicking non-interactive elements (blank/text) in the overlay also closes it; interactive controls (tab/button/select) are exempt
$overlay.addEventListener('click', (e) => {
    if (
        e.target.closest('.tab-btn, button, select, option, input, textarea, a') ||
        e.target.closest('.overlay-tabs')
    ) {
        return; // Allow interactive controls to handle clicks
    }
    e.stopPropagation();
    toggleOverlay();
});

// 3. Tab Switching Logic (Mutually Exclusive)
$tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all
        $tabBtns.forEach(b => b.classList.remove('active'));
        $tabPanes.forEach(p => p.classList.remove('active'));

        // Activate current
        btn.classList.add('active');
        const tabId = btn.dataset.tab;
        document.getElementById(`tab-${tabId}`).classList.add('active');
    });
});

// 4. Main button logic (start/mute toggle)
$btnMain.addEventListener('click', async () => {
    try {
        if (convo.state.loading) {
            return;
        }
        if (!convo.state.streaming) {
            // Start a session (stay unmuted by default)
            await convo.toggleStreaming();
        } else {
            // Toggle microphone mute
            convo.toggleMicMute();
        }
    } catch (e) {
        alert('Failed: ' + (e?.message || e));
    }
});

// Settings Actions
$btnStop.addEventListener('click', async () => {
    try {
        if (convo.state.streaming) {
            await convo.toggleStreaming();
        }
        resetHistoryLogs();
        convo.clearHistory();
    } finally {
        isSettingsOpen = false; $settingsModal.classList.remove('open');
    }
});
$btnRestart.addEventListener('click', async () => {
    try {
        await convo.restart();
    } finally {
        isSettingsOpen = false; $settingsModal.classList.remove('open');
    }
});

// --- Visual Animation ---
function animateOrbMock() {
    rafId = requestAnimationFrame(animateOrbMock);
    let targetScale = 1;
    if (currentVisualState === 'listening') {
        const time = Date.now() / 1000;
        targetScale = 1 + Math.abs(Math.sin(time * 10) * 0.05 + Math.cos(time * 23) * 0.05);
    } else if (currentVisualState === 'speaking') {
        const time = Date.now() / 1000;
        targetScale = 1.1 + Math.abs(Math.sin(time * 15) * 0.15 + Math.cos(time * 40) * 0.1);
    } else if (currentVisualState === 'processing') {
        targetScale = 1.1;
    }
    $orbVisual.style.transform = `scale(${targetScale})`;
}
animateOrbMock();

// --- State Update ---
const STATE_CONFIG = {
    idle: { text: 'Ready', color: '#9ca3af', helper: 'Idle' },
    listening: { text: 'Listening', color: '#ef4444', helper: 'Listening...' },
    processing: { text: 'Thinking', color: '#a855f7', helper: 'Processing...' },
    speaking: { text: 'Speaking', color: '#10b981', helper: 'Speaking...' },
};

function pickLastByRole(messages, role) {
    for (let i = messages.length - 1; i >= 0; i--) {
        if (messages[i].role === role) return messages[i];
    }
    return null;
}

convo.subscribe((state) => {
    // 1) Top + center copy: use “latest user ASR + latest AI reply” instead of the caption
    if (state.streaming) {
        const lastUser = pickLastByRole(state.messages, 'user');
        const lastAssistant = pickLastByRole(state.messages, 'assistant');
        const lastMsg = state.messages[state.messages.length - 1] || null;
        $liveThought.textContent = lastUser?.content || '';
        // If the newest entry is info, show that info text on the Orb instead of the assistant caption
        $liveCaption.textContent = (lastMsg?.role === 'info')
            ? (lastMsg?.content || '')
            : (lastAssistant?.content || (state.micMuted ? 'Mic Muted' : (state.streamState === 'listening' ? 'Listening...' : '')));
    } else {
        $liveCaption.textContent = 'Tap Start to begin';
        $liveThought.textContent = '';
    }

    // 2) Main button and status capsule
    if (!state.streaming) {
        currentVisualState = 'idle';
        $orbArea.className = 'orb-container';
        $btnStop.style.opacity = 0.5;
        $btnStop.disabled = true;
        $btnMain.classList.remove('active', 'muted');
        $micWaves.classList.add('hidden');
        if (state.loading) {
            $btnMain.classList.add('mode-start', 'loading');
            $btnMain.setAttribute('disabled', 'disabled');
            $micLoading.classList.remove('hidden');
            $iconPlay.classList.add('hidden');
            $iconMic.classList.add('hidden');
            $iconMuted.classList.add('hidden');
            $helperText.textContent = 'Loading...';
            $statusText.textContent = 'Preparing';
            $statusDot.style.backgroundColor = '#3b82f6';
        } else {
            $btnMain.classList.add('mode-start');
            $btnMain.classList.remove('loading');
            $btnMain.removeAttribute('disabled');
            $micLoading.classList.add('hidden');
            $iconPlay.classList.remove('hidden');
            $iconMic.classList.add('hidden');
            $iconMuted.classList.add('hidden');
            $helperText.textContent = 'Tap to Start';
            $statusText.textContent = 'Disconnected';
            $statusDot.style.backgroundColor = '#6b7280';
        }
        if (state.queued) {
            $helperText.textContent = 'In Queue...';
            $statusText.textContent = 'In Queue';
            $statusDot.style.backgroundColor = '#f59e0b';
        }
    } else {
        $btnMain.removeAttribute('disabled');
        $btnMain.classList.remove('mode-start', 'loading');
        $micLoading.classList.add('hidden');
        $btnStop.style.opacity = 1;
        $btnStop.disabled = false;
        $iconPlay.classList.add('hidden');

        if (state.micMuted) {
            $btnMain.classList.add('muted');
            $btnMain.classList.remove('active');
            $iconMuted.classList.remove('hidden');
            $iconMic.classList.add('hidden');
            $micWaves.classList.add('hidden');
            $helperText.textContent = 'Mic Muted';
            currentVisualState = 'idle';
            $orbArea.className = 'orb-container';
            $statusText.textContent = 'Muted';
            $statusDot.style.backgroundColor = '#6b7280';
        } else {
            $btnMain.classList.remove('muted');
            $iconMuted.classList.add('hidden');

            let visualState = 'idle';
            if (state.streamState === 'listening') visualState = 'listening';
            else if (state.streamState === 'processing') visualState = 'processing';
            else if (state.streamState === 'speaking') visualState = 'speaking';

            currentVisualState = visualState;
            const config = STATE_CONFIG[visualState] || STATE_CONFIG.idle;

            $orbArea.className = `orb-container state-${visualState}`;
            $statusDot.style.backgroundColor = config.color;
            $statusText.textContent = config.text;
            $helperText.textContent = config.helper;

            if (visualState === 'listening') {
                $btnMain.classList.add('active');
                $iconMic.classList.add('hidden');
                $micWaves.classList.remove('hidden');
            } else {
                $btnMain.classList.remove('active');
                $iconMic.classList.remove('hidden');
                $micWaves.classList.add('hidden');
            }
        }
    }

    if (state.currentVoiceName) {
        syncVoiceSelectValue(state.currentVoiceName);
    }

    // Force the top status to Disconnected when an info log reports a websocket error/disconnect
    try {
        const lastMsg = state.messages[state.messages.length - 1] || null;
        const infoText = lastMsg?.role === 'info' ? String(lastMsg.content || '').toLowerCase() : '';
        if (infoText.includes('lost connection') || infoText.includes('websocket error') || infoText.includes('connection lost')) {
            $statusText.textContent = 'Disconnected';
            $statusDot.style.backgroundColor = '#6b7280';
        }
    } catch (_) { }

    // 3) Optional log accumulation (Thought/Caption/Retrieval)
    // Caption is no longer the primary display but can stay in the log (optional).
    if (state.latestCaption && state.latestCaption !== lastCaption) {
        lastCaption = state.latestCaption;
        captionLog.push({ role: state.streamState === 'speaking' ? 'assistant' : 'user', content: state.latestCaption });
        if (captionLog.length > 10) captionLog.splice(0, captionLog.length - 10); // Keep only the latest 10 entries
    }
    if (state.latestThought && state.latestThought !== lastThought) {
        lastThought = state.latestThought;
        thoughtLog.push(state.latestThought);
        if (thoughtLog.length > 10) thoughtLog.splice(0, thoughtLog.length - 10);
    }
    if (state.latestRetrieval && state.latestRetrieval !== lastRetrieval) {
        lastRetrieval = state.latestRetrieval;
        retrievalLog.push(state.latestRetrieval);
        if (retrievalLog.length > 10) retrievalLog.splice(0, retrievalLog.length - 10);
    }

    // 4) Render chat messages and logs
    renderMessages(state.messages);
    renderAuxLogs();
});

function renderMessages(messages) {
    const prevCount = lastRenderedChatCount;
    // Chat bubbles (user / assistant / info)
    $chatList.innerHTML = '';
    const chats = messages.filter(m => m.role === 'user' || m.role === 'assistant' || m.role === 'info');
    for (const m of chats) {
        const div = document.createElement('div');
        div.className = `chat-item ${m.role}`;
        div.innerHTML = `
                    <div class="msg-role">${m.role.toUpperCase()}</div>
                    <div class="chat-bubble">${(m.content ?? '').replace(/</g, '&lt;')}</div>
                `;
        $chatList.appendChild(div);
    }
    if (chats.length === 0) {
        $chatList.innerHTML = '<div class="empty-state">No messages yet.</div>';
    }
    const shouldScroll = prevCount !== null && chats.length > prevCount;
    if (shouldScroll) {
        try {
            const oc = document.querySelector('.overlay-content');
            if (oc) oc.scrollTop = oc.scrollHeight;
        } catch (_) { }
    }
    lastRenderedChatCount = chats.length;
}

function renderAuxLogs() {
    // Caption log (render only the newest 10 entries to match data limits)
    $captionLogList.innerHTML = '';
    if (captionLog.length === 0) {
        $captionLogList.innerHTML = '<div class="empty-state">No captions recorded yet.</div>';
    } else {
        for (const item of captionLog.slice(-10)) {
            const div = document.createElement('div');
            div.className = 'content-box caption-box';
            div.textContent = (item.content ?? '').replace(/</g, '&lt;');
            $captionLogList.appendChild(div);
        }
    }
    // Thought list
    $thoughtList.innerHTML = '';
    if (thoughtLog.length === 0) {
        $thoughtList.innerHTML = '<div class="empty-state">No thoughts recorded yet.</div>';
    } else {
        for (const t of thoughtLog.slice(-10)) {
            const div = document.createElement('div');
            div.className = 'content-box thought-box';
            div.textContent = t;
            $thoughtList.appendChild(div);
        }
    }
    // Retrieval list
    $retrievalList.innerHTML = '';
    if (retrievalLog.length === 0) {
        $retrievalList.innerHTML = '<div class="empty-state">No retrieval data recorded yet.</div>';
    } else {
        for (const r of retrievalLog.slice(-10)) {
            const div = document.createElement('div');
            div.className = 'content-box retrieval-box';
            div.textContent = r;
            $retrievalList.appendChild(div);
        }
    }
}

// --- Voice list from backend ---
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
        const resp = await fetch('/api/voices');
        const data = await resp.json();
        availableAudios = data.audios || [];
        $voiceSelect.innerHTML = '<option value="" selected disabled hidden></option>';
        for (const [idx, audio] of availableAudios.entries()) {
            const voiceName = audio.name || audio.path || `voice_${idx}`;
            const opt = document.createElement('option');
            opt.value = voiceName;
            opt.textContent = voiceName;
            opt.dataset.path = audio.path || '';
            $voiceSelect.appendChild(opt);
        }
    } catch (e) {
        console.error('Failed to load reference audios:', e);
        $voiceSelect.innerHTML = '<option value="">Load failed</option>';
    }
}
$voiceSelect.addEventListener('change', (e) => {
    const selectedName = e.target.value;
    const item = availableAudios.find(a => (a.name || a.path) === selectedName);
    if (item) {
        const voiceName = item.name || selectedName;
        convo.changeVoice(voiceName);
        convo.state.currentVoiceName = voiceName;
        convo.state.currentVoicePath = item.path || null;
        syncVoiceSelectValue(voiceName);
    }
});

// Speed selector: call session.changeTTSSpeed (or adjust playback rate if backend lacks support)
$speedSelect?.addEventListener('change', (e) => {
    const val = parseFloat(e.target.value);
    if (!Number.isNaN(val) && val > 0) {
        try { convo.changeTTSSpeed?.(val); } catch (_) { }
    }
});

// Load available TTS models
let availableTTSModels = [];
async function loadTTSModels() {
    try {
        const response = await fetch('/api/available-tts-models');
        const data = await response.json();
        availableTTSModels = data.models || [];

        // Populate the dropdown
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

        // Enable the dropdown
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

        // Populate the dropdown
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
            // Store the extra_body config (JSON string)
            option.dataset.extraBody = JSON.stringify(model.extra_body || null);
            if (index === 0) {
                option.selected = true;
            }
            $llmModelSelect.appendChild(option);
        });

        // Enable the dropdown
        $llmModelSelect.disabled = false;
    } catch (error) {
        console.error('Failed to load LLM models:', error);
        $llmModelSelect.innerHTML = '<option value=\"\">Load failed</option>';
    }
}

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
    const apiKey = selectedOption.dataset.apiKey || 'none';
    // Parse the extra_body config
    let extraBody = null;
    try {
        extraBody = JSON.parse(selectedOption.dataset.extraBody || 'null');
    } catch (_) { }

    if (selectedModelName) {
        convo.changeLLMModel(selectedModelName, baseUrl, apiKey, extraBody);
    }
});

loadReferenceAudios();
loadTTSModels();
loadLLMModels();
