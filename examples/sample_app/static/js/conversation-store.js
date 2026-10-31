import {createBusinessAPI, readApplicationState} from './business-api.js';

/** Compose a pure audio session with application-owned identity and history. */
export function createBusinessSession(createSession, url, config) {
    const live = config?.mode === "end_to_end";
    const api = createBusinessAPI();
    const snapshotKey = 'xtalk.sample.conversation';
    const saved = live ? {} : readApplicationState(snapshotKey, {});
    let active = null;
    let history = saved.messages ?? [];
    let businessId = saved.sessionId ?? null;
    const listeners = new Set();
    const callbacks = new Map();
    let muted = false;
    let operation = 0;
    const view = {
        connectionState: 'disconnected', streamState: 'idle', messages: history,
        mode: live ? 'end_to_end' : 'cascade', transcriptFragments: [], error: '',
        sessionId: businessId, latency: {}, generatedResponses: {}, thought: '',
        caption: '', retrieval: '', tool_call: {name: '', args: {}},
        currentVoiceName: saved.currentVoiceName ?? '', currentVoicePath: saved.currentVoicePath ?? null,
    };
    function save() {
        if (live) return;
        localStorage.setItem(snapshotKey, JSON.stringify({
            sessionId: businessId, messages: view.messages,
            currentVoiceName: view.currentVoiceName, currentVoicePath: view.currentVoicePath,
        }));
    }
    const state = new Proxy(view, {set(target, key, value) { target[key] = value; save(); return true; }});
    function notify() { save(); for (const callback of listeners) callback(state); }
    async function open() {
        const generation = ++operation;
        if (active) await active.close();
        await api.login();
        if (generation !== operation) throw new Error('Session initialization cancelled');
        if (!live) {
            if (!businessId) businessId = (await api.create()).session_id;
            history = (await api.detail(businessId)).messages;
        } else {
            history = []; businessId = null;
        }
        if (generation !== operation) throw new Error('Session initialization cancelled');
        const target = api.authenticatedURL(url);
        if (!live) target.searchParams.set('conversation_id', businessId);
        const instance = createSession(target, config);
        active = instance;
        instance.muted = muted;
        instance.onStateChange(snapshot => {
            if (active !== instance) return;
            Object.assign(view, snapshot, {sessionId: live ? snapshot.sessionId : businessId});
            notify();
        });
        for (const [name, callback] of callbacks) instance[name](callback);
        if (!live && history.length) await instance.restoreHistory(history);
        await instance.open();
        if (generation !== operation) { await instance.close(); throw new Error('Session initialization cancelled'); }
    }
    return {
        open,
        async close() { operation++; const instance = active; active = null; if (instance) await instance.close(); view.connectionState = 'disconnected'; view.streamState = 'idle'; notify(); },
        get state() { return state; },
        get muted() { return active?.muted ?? muted; },
        set muted(value) { muted = value; if (active) active.muted = value; },
        onStateChange(callback) { listeners.add(callback); callback(state); },
        onInputAudioChunk(callback) { callbacks.set('onInputAudioChunk', callback); active?.onInputAudioChunk(callback); },
        onOutputAudioChunk(callback) { callbacks.set('onOutputAudioChunk', callback); active?.onOutputAudioChunk(callback); },
        onFullAudioChunk(callback) { callbacks.set('onFullAudioChunk', callback); active?.onFullAudioChunk(callback); },
        async sendText(text) { if (!active) throw new Error('Open a conversation first'); await active.sendText(text); },
        async changeVoice(voice) { if (!active) throw new Error('Open a conversation first'); await active.changeVoice(voice); },
        async getSessions() { if (live) return []; await api.login(); return (await api.list()).sessions; },
        async switchSession(id) {
            if (live) throw new Error("History is unavailable in single-call Live");
            if (active) await active.close();
            await api.login();
            businessId = id;
            history = id ? (await api.detail(id)).messages : [];
            active = null;
            Object.assign(view, {sessionId: id, messages: history, generatedResponses: {}, connectionState: 'disconnected', streamState: 'idle'});
            notify();
        },
        uploadFile: file => { if (live) throw new Error("File input is unavailable in Live mode"); return api.upload(businessId, file); },
    };
}
