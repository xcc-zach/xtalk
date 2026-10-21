import {createBusinessAPI, readApplicationState} from './business-api.js';

/** Compose a pure audio session with application-owned identity and history. */
export function createBusinessSession(createSession, url, config) {
    const api = createBusinessAPI();
    const snapshotKey = 'xtalk.sample.conversation';
    const saved = readApplicationState(snapshotKey, {});
    let active = null;
    let history = saved.messages ?? [];
    let businessId = saved.sessionId ?? null;
    const listeners = new Set();
    const callbacks = new Map();
    let muted = false;
    const view = {
        connectionState: 'disconnected', streamState: 'idle', messages: history,
        sessionId: businessId, latency: {}, generatedResponses: {}, thought: '',
        caption: '', retrieval: '', tool_call: {name: '', args: {}},
        currentVoiceName: saved.currentVoiceName ?? '', currentVoicePath: saved.currentVoicePath ?? null,
    };
    function save() {
        localStorage.setItem(snapshotKey, JSON.stringify({
            sessionId: businessId, messages: view.messages,
            currentVoiceName: view.currentVoiceName, currentVoicePath: view.currentVoicePath,
        }));
    }
    const state = new Proxy(view, {set(target, key, value) { target[key] = value; save(); return true; }});
    function notify() { save(); for (const callback of listeners) callback(state); }
    async function open() {
        if (active) await active.close();
        await api.login();
        if (!businessId) businessId = (await api.create()).session_id;
        history = (await api.detail(businessId)).messages;
        const target = api.authenticatedURL(url);
        target.searchParams.set('conversation_id', businessId);
        active = createSession(target, config);
        active.muted = muted;
        active.onStateChange(snapshot => {
            Object.assign(view, snapshot, {sessionId: businessId});
            notify();
        });
        for (const [name, callback] of callbacks) active[name](callback);
        if (history.length) await active.restoreHistory(history);
        await active.open();
    }
    return {
        open,
        async close() { if (active) await active.close(); view.connectionState = 'disconnected'; view.streamState = 'idle'; notify(); },
        get state() { return state; },
        get muted() { return active?.muted ?? muted; },
        set muted(value) { muted = value; if (active) active.muted = value; },
        onStateChange(callback) { listeners.add(callback); callback(state); },
        onInputAudioChunk(callback) { callbacks.set('onInputAudioChunk', callback); active?.onInputAudioChunk(callback); },
        onOutputAudioChunk(callback) { callbacks.set('onOutputAudioChunk', callback); active?.onOutputAudioChunk(callback); },
        onFullAudioChunk(callback) { callbacks.set('onFullAudioChunk', callback); active?.onFullAudioChunk(callback); },
        async sendText(text) { if (!active) throw new Error('Open a conversation first'); await active.sendText(text); },
        async changeVoice(voice) { if (!active) throw new Error('Open a conversation first'); await active.changeVoice(voice); },
        async getSessions() { await api.login(); return (await api.list()).sessions; },
        async switchSession(id) {
            if (active) await active.close();
            await api.login();
            businessId = id;
            history = id ? (await api.detail(id)).messages : [];
            active = null;
            Object.assign(view, {sessionId: id, messages: history, generatedResponses: {}, connectionState: 'disconnected', streamState: 'idle'});
            notify();
        },
        uploadFile: file => api.upload(businessId, file),
    };
}
