import { createInputAudioSession, createOutputAudioSession } from "../audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import type { BaseOutputAudioSession } from "../bases/audio-session";
import type { BaseDeferredTaskScheduler } from "../bases/task-scheduler";
import type { BaseWebSocket } from "../bases/websocket";
import type { BaseWebSocketCloseEvent, BaseWebSocketMessageEvent } from "../bases/websocket";
import { Conversation } from "../conversation";
import { buildAuthenticatedWebSocketURL } from "../http";
import { ActionHandler } from "../action-handler";
import { createDeferredTaskScheduler } from "../task-scheduler";
import { createWebSocket } from "../websocket";
import type { AudioChunkCallback } from "./types";

export type { SessionRuntimeController };
export { createSessionRuntimeController };

type SessionRuntimeController = ReturnType<typeof createSessionRuntimeController>;

function createSessionRuntimeController(
    {
        actionHandler,
        conversation,
        getAccessToken,
        inputConfig,
        onUnexpectedDisconnect,
        outputConfig,
        websocketURL,
    }: {
        actionHandler: ActionHandler;
        conversation: Conversation;
        getAccessToken: () => string | null;
        inputConfig: InputAudioSessionConfig;
        onUnexpectedDisconnect: (details: BaseWebSocketCloseEvent) => void;
        outputConfig: OutputAudioSessionConfig;
        websocketURL: string | URL;
    },
) {
    let websocket: BaseWebSocket | null = null;
    let inputAudioSession: ReturnType<typeof createInputAudioSession> | null = null;
    let outputAudioSession: ReturnType<typeof createOutputAudioSession> | null = null;
    let activeCloseState: { expected: boolean } | null = null;
    let preferredMuted = false;
    let inputAudioChunkCallback: AudioChunkCallback = (_chunk, _sr) => { };
    let outputAudioChunkCallback: AudioChunkCallback = (_chunk, _sr) => { };
    let observerTaskScheduler: BaseDeferredTaskScheduler | null = null;

    function requireRuntime(): {
        websocket: BaseWebSocket;
        outputAudioSession: BaseOutputAudioSession;
    } {
        if (!websocket || !outputAudioSession) {
            throw new Error("Session is not open");
        }
        return { websocket, outputAudioSession };
    }

    async function initialize(): Promise<void> {
        const accessToken = getAccessToken();
        if (!accessToken) {
            throw new Error("Access token missing");
        }

        const wsURL = buildAuthenticatedWebSocketURL(websocketURL, accessToken);
        const closeState = { expected: false };
        websocket = createWebSocket(wsURL);
        activeCloseState = closeState;
        inputAudioSession = createInputAudioSession(inputConfig);
        outputAudioSession = createOutputAudioSession(outputConfig);
        observerTaskScheduler?.dispose();
        observerTaskScheduler = createDeferredTaskScheduler();
        const currentWebSocket = websocket;
        const currentInputAudioSession = inputAudioSession;
        const currentOutputAudioSession = outputAudioSession;
        const currentObserverTaskScheduler = observerTaskScheduler;
        currentInputAudioSession.muted = preferredMuted;

        let rejectAttached: ((reason?: unknown) => void) | null = null;
        const actionAttachedPromise = actionHandler.waitForAction("session_attached");
        const attachedPromise = new Promise<void>((resolve, reject) => {
            rejectAttached = reject;
            void actionAttachedPromise.then(resolve, reject);
        });
        const openPromise = new Promise<void>((resolve, reject) => {
            currentWebSocket.addEventListener("open", () => {
                resolve();
            });
            currentWebSocket.addEventListener("error", () => {
                reject(new Error("WebSocket connection failed"));
            });
        });

        currentWebSocket.addEventListener("close", (event: BaseWebSocketCloseEvent) => {
            conversation.state.streamState = "idle";
            rejectAttached?.(new Error("WebSocket closed before session attachment"));
            if (!closeState.expected) {
                onUnexpectedDisconnect(event);
            }
        });
        currentWebSocket.addEventListener(
            "message",
            async (event: BaseWebSocketMessageEvent) => {
                if (typeof event.data === "string") {
                    const message: { action: string; data: unknown } = JSON.parse(event.data);
                    await actionHandler.handleAction(
                        message.action,
                        message.data,
                        currentWebSocket,
                        conversation,
                        currentOutputAudioSession,
                    );
                } else if (event.data instanceof ArrayBuffer) {
                    await currentOutputAudioSession.pushAudioChunk(event.data);
                }
            },
        );

        currentInputAudioSession.onFrame(async (audioChunk) => {
            if (currentWebSocket.ready()) {
                currentWebSocket.sendAudioChunk(audioChunk);
            }
            const callback = inputAudioChunkCallback;
            currentObserverTaskScheduler.schedule(() => {
                callback(audioChunk, inputConfig.sampleRate);
            });
        });
        currentInputAudioSession.onSpeechStart(async () => {
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction(
                    "client_speech_start",
                    null,
                    currentWebSocket,
                    conversation,
                    currentOutputAudioSession,
                );
            }
        });
        currentInputAudioSession.onSpeechEnd(async () => {
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction(
                    "client_speech_end",
                    null,
                    currentWebSocket,
                    conversation,
                    currentOutputAudioSession,
                );
            }
        });

        currentOutputAudioSession.onChunkStarted(async (responseId, audioChunk) => {
            const callback = outputAudioChunkCallback;
            currentObserverTaskScheduler.schedule(() => {
                callback(audioChunk, outputConfig.sampleRate);
            });
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction(
                    "client_audio_chunk_started",
                    null,
                    currentWebSocket,
                    conversation,
                    currentOutputAudioSession,
                );
            }
        });
        currentOutputAudioSession.onChunkPlayed(async (responseId) => {
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction(
                    "client_audio_chunk_played",
                    { response_id: responseId },
                    currentWebSocket,
                    conversation,
                    currentOutputAudioSession,
                );
            }
        });
        currentOutputAudioSession.onAllChunksPlayed(async (responseId) => {
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction(
                    "client_audio_playback_finished",
                    { response_id: responseId },
                    currentWebSocket,
                    conversation,
                    currentOutputAudioSession,
                );
            }
        });

        await openPromise;
        currentWebSocket.sendJson({
            action: "attach_session",
            session_id: conversation.state.sessionId,
        });
        await attachedPromise;
        await currentOutputAudioSession.open();
        await currentInputAudioSession.open();
    }

    async function close(): Promise<void> {
        const currentInput = inputAudioSession;
        const currentOutput = outputAudioSession;
        const currentWebSocket = websocket;

        if (activeCloseState) {
            activeCloseState.expected = true;
            activeCloseState = null;
        }
        observerTaskScheduler?.dispose();
        observerTaskScheduler = null;
        inputAudioSession = null;
        outputAudioSession = null;
        websocket = null;

        if (currentInput) {
            try {
                preferredMuted = currentInput.muted;
                await currentInput.close();
            } catch {
                // Ignore shutdown errors from already-closed audio sessions.
            }
        }
        if (currentOutput) {
            try {
                await currentOutput.close();
            } catch {
                // Ignore shutdown errors from already-closed audio sessions.
            }
        }
        currentWebSocket?.close();
        conversation.state.streamState = "idle";
    }

    return {
        close,
        get muted(): boolean {
            return inputAudioSession ? inputAudioSession.muted : preferredMuted;
        },
        initialize,
        onInputAudioChunk(callback: AudioChunkCallback): void {
            inputAudioChunkCallback = callback;
        },
        onOutputAudioChunk(callback: AudioChunkCallback): void {
            outputAudioChunkCallback = callback;
        },
        requireRuntime,
        set muted(value: boolean) {
            preferredMuted = value;
            if (inputAudioSession) {
                inputAudioSession.muted = value;
            }
        },
    };
}
