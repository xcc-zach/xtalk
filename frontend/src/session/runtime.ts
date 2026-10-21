import { createInputAudioSession, createOutputAudioSession } from "../audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import type { BaseOutputAudioSession } from "../bases/audio-session";
import type { BaseDeferredTaskScheduler } from "../bases/task-scheduler";
import type { BaseWebSocket } from "../bases/websocket";
import type { BaseWebSocketCloseEvent, BaseWebSocketMessageEvent } from "../bases/websocket";
import { Conversation } from "../conversation";

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
        beforeAudioStart,
        conversation,
        getAccessToken,
        inputConfig,
        onUnexpectedDisconnect,
        outputConfig,
        websocketURL,
    }: {
        actionHandler: ActionHandler;
        beforeAudioStart: () => Promise<void>;
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


        const closeState = { expected: false };
        actionHandler.resetConnection();
        websocket = createWebSocket(websocketURL, accessToken ? ["realtime", "openai-insecure-api-key." + accessToken] : undefined);
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

        const readiness = actionHandler.waitUntilReady();
        const attachedPromise = readiness.promise;
        // Consume failures immediately while transport opening is still pending.
        void attachedPromise.catch(() => {});
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
            conversation.state.connectionState = "disconnected";
            readiness.cancel(new Error("WebSocket closed before session readiness"));
            actionHandler.cancelPending();
            if (!closeState.expected) {
                onUnexpectedDisconnect(event);
            }
        });
        currentWebSocket.addEventListener(
            "message",
            async (event: BaseWebSocketMessageEvent) => {
                await actionHandler.handleMessage(event.data, currentWebSocket, conversation, currentOutputAudioSession);
            },
        );

        currentInputAudioSession.onFrame(async (audioChunk) => {
            if (currentWebSocket.ready()) {
                await actionHandler.handleAction("client_audio_frame", {audioChunk, sampleRate: inputConfig.sampleRate}, currentWebSocket, conversation, currentOutputAudioSession);
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

        try {
            await openPromise;
            await attachedPromise;
            await beforeAudioStart();
            await currentOutputAudioSession.open();
            await currentInputAudioSession.open();
        } catch (error) {
            readiness.cancel(error);
            await close();
            throw error;
        }
        conversation.state.connectionState = "connected";
    }

    async function close(): Promise<void> {
        actionHandler.cancelPending();
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
