import { Conversation } from "../conversation";
import { ActionHandler } from "../action-handler";
import { createSessionRuntimeController } from "./runtime";
import type { Session, SessionConfig } from "./types";

/** Creates a pure Realtime conversation client without business HTTP calls. */
export function createSession(websocketURL: string | URL, config: SessionConfig = {}): Session {
    const conversation = new Conversation();
    const actionHandler = new ActionHandler(config.outputAudioEncoding);
    let opened = false;
    let pendingHistory: Parameters<Session["restoreHistory"]>[0] = [];
    async function sendHistory(messages: Parameters<Session["restoreHistory"]>[0]): Promise<void> {
        const active = runtime.requireRuntime();
        for (const message of messages) {
            await actionHandler.restoreItem(message, active.websocket, conversation, active.outputAudioSession);
        }
    }
    const runtime = createSessionRuntimeController({
        actionHandler, conversation,
        beforeAudioStart: async () => {
            await sendHistory(pendingHistory);
            pendingHistory = [];
        },
        getAccessToken: () => config.apiKey ?? null,
        inputConfig: {sampleRate: 16000, ...config.inputConfig},
        outputConfig: {sampleRate: 24000, ...config.outputConfig},
        websocketURL,
        onUnexpectedDisconnect: () => { conversation.state.streamState = "idle"; },
    });
    return {
        async open() { await runtime.initialize(); opened = true; },
        async close() { opened = false; await runtime.close(); },
        onStateChange: callback => conversation.onStateChange(callback),
        get state() { return conversation.state; },
        onInputAudioChunk: runtime.onInputAudioChunk,
        onOutputAudioChunk: runtime.onOutputAudioChunk,
        onFullAudioChunk: callback => conversation.onFullAudioChunk(callback),
        get muted() { return runtime.muted; },
        set muted(value) { runtime.muted = value; },
        async restoreHistory(messages) {
            if (!opened) { pendingHistory.push(...messages); return; }
            await sendHistory(messages);
        },
        async changeVoice(voiceName) {
            const active = runtime.requireRuntime();
            await actionHandler.handleAction("client_change_voice", {voiceName}, active.websocket, conversation, active.outputAudioSession);
        },
        async sendText(text) {
            const normalized = text.trim();
            if (!normalized) throw new Error("Text must not be empty");
            const active = runtime.requireRuntime();
            await actionHandler.submitText(normalized, active.websocket, conversation, active.outputAudioSession);
        },
    };
}
