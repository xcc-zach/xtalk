import { createInputAudioSession, createOutputAudioSession } from "../audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig, BaseOutputAudioSession } from "../bases/audio-session";
import type { BaseWebSocketCloseEvent, BaseWebSocketMessageEvent } from "../bases/websocket";
import { Conversation } from "../conversation";
import { ActionHandler } from "../action-handler";
import { createDeferredTaskScheduler } from "../task-scheduler";
import { createWebSocket } from "../websocket";
import type { AudioChunkCallback } from "./types";

export type { SessionRuntimeController };
export { createSessionRuntimeController };
type SessionRuntimeController = ReturnType<typeof createSessionRuntimeController>;

/** Negotiated server-owned continuous media and capabilities. */
type StreamProfile = {
    stream_id: string; protocol_version: number;
    media: {sample_rate: number; channels: number; encoding: string};
    capabilities: Record<string, boolean>;
};

/** Own one Realtime connection and invalidate media work from older calls. */
function createSessionRuntimeController({actionHandler, beforeAudioStart, conversation, getAccessToken,
    inputConfig, onUnexpectedDisconnect, outputConfig, websocketURL}: {
    actionHandler: ActionHandler; beforeAudioStart: () => Promise<void>; conversation: Conversation;
    getAccessToken: () => string | null; inputConfig: InputAudioSessionConfig;
    onUnexpectedDisconnect: (details: BaseWebSocketCloseEvent) => void;
    outputConfig: OutputAudioSessionConfig; websocketURL: string | URL;
}) {
    type Runtime = {
        websocket: ReturnType<typeof createWebSocket>;
        input: ReturnType<typeof createInputAudioSession> | null;
        output: ReturnType<typeof createOutputAudioSession> | null;
        profile: StreamProfile | null;
        expected: boolean; uploading: boolean; closing: boolean;
        cancel: (reason: unknown) => void; cancelled: Promise<never>;
        ended: Promise<void>; resolveEnded: () => void;
        observers: ReturnType<typeof createDeferredTaskScheduler>; cleanup?: Promise<void>;
    };
    let current: Runtime | null = null;
    let initializing: Promise<void> | null = null;
    let preferredMuted = false;
    let inputCallback: AudioChunkCallback = () => {};
    let outputCallback: AudioChunkCallback = () => {};

    function control(runtime: Runtime, kind: string, fields: Record<string, unknown> = {}): void {
        if (runtime.websocket.ready() && runtime.profile) runtime.websocket.sendJson({
            type: `xtalk.audio_stream.${kind}`, stream_id: runtime.profile.stream_id, ...fields,
        });
    }
    function reportError(error: unknown): void {
        const message = error instanceof Error ? error.message : String(error);
        conversation.state.error = message;
        conversation.appendMessage({role: "info", content: message, final: true});
    }
    function requireRuntime(): {websocket: Runtime["websocket"]; outputAudioSession: BaseOutputAudioSession} {
        if (!current?.output) throw new Error("Session is not open");
        return {websocket: current.websocket, outputAudioSession: current.output};
    }
    async function cleanup(runtime: Runtime): Promise<void> {
        if (runtime.cleanup) return runtime.cleanup;
        runtime.closing = true;
        runtime.uploading = false;
        runtime.cancel(new Error("Session initialization cancelled"));
        runtime.cleanup = (async () => {
            const inputClosed = runtime.input?.close();
            const outputClosed = runtime.output?.close();
            runtime.observers.dispose();
            control(runtime, "close");
            await Promise.allSettled([inputClosed, outputClosed]);
            if (runtime.profile && runtime.websocket.ready()) {
                let timer: ReturnType<typeof setTimeout> | undefined;
                await Promise.race([runtime.ended, new Promise<void>(resolve => { timer = setTimeout(resolve, 17000); })]);
                if (timer !== undefined) clearTimeout(timer);
            }
            runtime.websocket.close();
        })();
        return runtime.cleanup;
    }
    async function initializeOnce(): Promise<void> {
        const token = getAccessToken();
        let cancel: (reason: unknown) => void = () => {};
        let resolveEnded: () => void = () => {};
        const cancelled = new Promise<never>((_, reject) => { cancel = reject; });
        void cancelled.catch(() => {});
        const ended = new Promise<void>(resolve => { resolveEnded = resolve; });
        actionHandler.resetConnection();
        const runtime: Runtime = {
            websocket: createWebSocket(websocketURL, token ? ["realtime", "openai-insecure-api-key." + token] : undefined),
            input: null, output: null, profile: null, expected: false, uploading: false, closing: false,
            cancel, cancelled, ended, resolveEnded, observers: createDeferredTaskScheduler(),
        };
        current = runtime;
        const ws = runtime.websocket;
        conversation.state.connectionState = "connecting";
        conversation.state.error = "";
        const readiness = actionHandler.waitUntilReady();
        void readiness.promise.catch(() => {});
        const opened = new Promise<void>((resolve, reject) => {
            ws.addEventListener("open", resolve);
            ws.addEventListener("error", () => reject(new Error("WebSocket connection failed")));
        });
        void opened.catch(() => {});
        ws.addEventListener("close", event => {
            runtime.resolveEnded();
            runtime.cancel(new Error("Connection closed during initialization"));
            readiness.cancel(new Error("WebSocket closed before session readiness"));
            if (current !== runtime) return;
            actionHandler.cancelPending();
            current = null;
            conversation.state.connectionState = "disconnected";
            conversation.state.streamState = "idle";
            void cleanup(runtime);
            if (!runtime.expected) {
                if (runtime.profile) reportError("Live connection ended. Start a new call to continue.");
                onUnexpectedDisconnect(event);
            }
        });
        let serial = Promise.resolve();
        ws.addEventListener("message", (event: BaseWebSocketMessageEvent) => {
            serial = serial.then(async () => {
                const data = typeof event.data === "string" ? JSON.parse(event.data) : null;
                const kind = data?.type;
                if (kind === "xtalk.audio_stream.ended") {
                    runtime.resolveEnded();
                    if (runtime.closing || current !== runtime) return;
                }
                if (current !== runtime || runtime.closing) return;
                if (kind === "session.created") {
                    const profile: StreamProfile | null = data.session.xtalk?.end_to_end ?? null;
                    if (profile) {
                        if (profile.protocol_version !== 2) throw new Error("Rebuild and refresh the frontend: Live protocol version 2 required");
                        if (![16000, 24000].includes(profile.media.sample_rate) || profile.media.channels !== 1 || profile.media.encoding !== "pcm_s16le") throw new Error("Unsupported Live PCM format");
                        if (inputConfig.mode && inputConfig.mode !== "microphone") throw new Error("Live requires microphone input");
                        runtime.profile = profile;
                        conversation.state.mode = "end_to_end";
                        conversation.switch(data.session.id, []);
                        conversation.state.capabilities = profile.capabilities;
                    }
                    const rate = profile?.media.sample_rate;
                    runtime.input = createInputAudioSession(rate ? {...inputConfig, sampleRate: rate, enableVAD: false, enableEnhancer: false, preserveSilenceWhenMuted: true} : inputConfig);
                    runtime.output = createOutputAudioSession(rate ? {...outputConfig, sampleRate: rate} : outputConfig);
                    if (profile && !runtime.output.supportsContinuousAudio) throw new Error("Platform lacks continuous audio playback");
                }
                if (!runtime.output) {
                    if (kind === "session.queue.updated" || kind === "session.queue.granted") return;
                    throw new Error(data?.error?.message || "Session was not created");
                }
                await actionHandler.handleMessage(event.data, ws, conversation, runtime.output);
                if (kind === "xtalk.audio_stream.error" && data.fatal) {
                    throw new Error(data.message);
                }
                if (kind === "xtalk.audio_stream.ended") {
                    runtime.uploading = false;
                    runtime.expected = true;
                    if (data.outcome !== "completed") reportError(new Error(data.reason || "Live call ended; start a new call to continue"));
                    current = null;
                    conversation.state.connectionState = "disconnected";
                    void cleanup(runtime);
                }
            }).catch(error => {
                readiness.cancel(error);
                runtime.cancel(error);
                if (current !== runtime) return;
                reportError(error);
                actionHandler.cancelPending();
                current = null;
                runtime.expected = true;
                conversation.state.connectionState = "disconnected";
                void cleanup(runtime);
            });
        });
        let timer: ReturnType<typeof setTimeout> | undefined;
        let activeCancel: ((reason: unknown) => void) | undefined;
        try {
            timer = setTimeout(() => runtime.cancel(new Error("Realtime session readiness timed out")), 30000);
            await Promise.race([opened, cancelled]);
            await Promise.race([readiness.promise, cancelled]);
            clearTimeout(timer);
            const input = runtime.input!;
            const output = runtime.output!;
            const profile = runtime.profile;
            const rate = profile?.media.sample_rate;
            input.muted = preferredMuted;
            input.onFrame(chunk => {
                if (current !== runtime || !runtime.uploading || !ws.ready()) return;
                if (profile) {
                    if (ws.bufferedAmount > profile.media.sample_rate * 2 * 5) {
                        reportError("Microphone transport accumulated more than five seconds of audio");
                        void close(); return;
                    }
                    ws.sendAudioChunk(chunk);
                } else {
                    void actionHandler.handleAction("client_audio_frame", {audioChunk: chunk, sampleRate: inputConfig.sampleRate}, ws, conversation, output);
                }
                runtime.observers.schedule(() => inputCallback(chunk, rate ?? inputConfig.sampleRate));
            });
            input.onSpeechStart(() => {
                if (!profile && current === runtime && ws.ready()) return actionHandler.handleAction("client_speech_start", null, ws, conversation, output);
            });
            input.onSpeechEnd(() => {
                if (!profile && current === runtime && ws.ready()) return actionHandler.handleAction("client_speech_end", null, ws, conversation, output);
            });
            output.onAudioStreamProgress(progress => control(runtime, "progress", {played_samples: progress.played_samples}));
            output.onChunkStarted((_responseId, chunk) => {
                if (current !== runtime) return;
                runtime.observers.schedule(() => outputCallback(chunk, rate ?? outputConfig.sampleRate));
                if (!profile && ws.ready()) return actionHandler.handleAction("client_audio_chunk_started", null, ws, conversation, output);
            });
            output.onChunkPlayed(responseId => {
                if (!profile && current === runtime && ws.ready()) return actionHandler.handleAction("client_audio_chunk_played", {response_id: responseId}, ws, conversation, output);
            });
            output.onAllChunksPlayed(responseId => {
                if (!profile && current === runtime && ws.ready()) return actionHandler.handleAction("client_audio_playback_finished", {response_id: responseId}, ws, conversation, output);
            });
            await Promise.race([beforeAudioStart(), cancelled]);
            await Promise.race([output.open(), cancelled]);
            await Promise.race([input.open(), cancelled]);
            if (profile) {
                const active = actionHandler.waitForActionMatching("xtalk.audio_stream.started", data => (data as any)?.stream_id === profile.stream_id && current === runtime);
                activeCancel = active.cancel;
                void active.promise.catch(() => {});
                timer = setTimeout(() => active.cancel(new Error("Live startup timed out")), 60000);
                control(runtime, "ready");
                try { await Promise.race([active.promise, cancelled]); } finally { clearTimeout(timer); }
                if (preferredMuted) control(runtime, "input_muted", {muted: true});
            }
            if (current !== runtime || runtime.closing) throw new Error("Session initialization cancelled");
            runtime.uploading = true;
            conversation.state.connectionState = "connected";
            conversation.state.streamState = "listening";
        } catch (error) {
            if (current === runtime) {
                current = null; runtime.expected = true;
                conversation.state.connectionState = "disconnected";
            }
            await cleanup(runtime);
            throw error;
        } finally {
            if (timer !== undefined) clearTimeout(timer);
            readiness.cancel();
            activeCancel?.(new Error("Initialization finished"));
        }
    }
    async function initialize(): Promise<void> {
        if (initializing) return initializing;
        initializing = initializeOnce();
        try { await initializing; } finally { initializing = null; }
    }
    async function close(): Promise<void> {
        const runtime = current;
        current = null;
        conversation.state.connectionState = "disconnected";
        conversation.state.streamState = "idle";
        actionHandler.cancelPending();
        if (!runtime) return;
        runtime.expected = true;
        await cleanup(runtime);
    }
    return {
        initialize, close, requireRuntime,
        onInputAudioChunk(callback: AudioChunkCallback): void { inputCallback = callback; },
        onOutputAudioChunk(callback: AudioChunkCallback): void { outputCallback = callback; },
        get muted(): boolean { return preferredMuted; },
        set muted(value: boolean) {
            preferredMuted = value;
            if (current?.input) current.input.muted = value;
            if (current?.uploading) control(current, "input_muted", {muted: value});
        },
    };
}
