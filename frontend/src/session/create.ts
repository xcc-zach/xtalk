import { createHTTPClient, delay, resolvePlatformServiceURLs } from "../http";
import { createPersistenceStore } from "../persistence";
import { Conversation } from "../conversation";
import { ActionHandler } from "../action-handler";
import {
    buildPersistedConversationSnapshot,
    clearPersistedConversationSnapshot,
    loadPersistedConversationSnapshot,
    savePersistedConversationSnapshot,
} from "./snapshot";
import { createSessionAuthController } from "./auth";
import { createSessionAPI } from "./api";
import { createSessionRuntimeController } from "./runtime";
import type { Session, SessionConfig } from "./types";

export { createSession };

const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAYS_MS = [0, 1000, 2000, 5000, 5000];
const MAX_TEXT_INPUT_CHARACTERS = 2048;
const TEXT_INPUT_RECEIPT_TIMEOUT_MS = 10000;

type FinishASRData = {
    text: string;
    origin: string;
};

function isMatchingFinishASRData(
    data: unknown,
    normalizedText: string,
): data is FinishASRData {
    return (
        typeof data === "object"
        && data !== null
        && "text" in data
        && typeof data.text === "string"
        && data.text === normalizedText
        && "origin" in data
        && data.origin === "text"
    );
}

/**
 * Creates a session client bound to the provided websocket endpoint.
 *
 * The returned session coordinates authentication, runtime audio streaming,
 * message state synchronization, and persisted conversation restoration.
 *
 * @param websocketURL Websocket endpoint used to establish the realtime session.
 * @param config Optional session configuration overrides.
 * @returns A session controller for opening, closing, and interacting with X-Talk.
 */
function createSession(
    websocketURL: string | URL,
    {
        inputConfig = {},
        outputConfig = {},
        serviceURLs: configuredServiceURLs,
    }: SessionConfig = {},
): Session {
    const resolvedInputConfig = {
        sampleRate: 16000,
        ...inputConfig,
    };
    const resolvedOutputConfig = {
        sampleRate: 48000,
        ...outputConfig,
    };
    const httpClient = createHTTPClient();
    const persistenceStore = createPersistenceStore();
    const serviceURLs = resolvePlatformServiceURLs(websocketURL, configuredServiceURLs);
    const persistenceKey = persistenceStore.resolveKey(websocketURL);
    const conversation = new Conversation();
    const actionHandler = new ActionHandler();
    const restoredSnapshot = loadPersistedConversationSnapshot(
        persistenceStore,
        persistenceKey,
    );

    if (restoredSnapshot) {
        conversation.setUser(restoredSnapshot.user);
        conversation.switch(restoredSnapshot.sessionId, restoredSnapshot.messages);
    }

    function clearPersistedSnapshot(): void {
        clearPersistedConversationSnapshot(persistenceStore, persistenceKey);
    }

    const authController = createSessionAuthController({
        clearPersistedSnapshot,
        conversation,
        httpClient,
        initialAccessToken: restoredSnapshot?.accessToken ?? null,
        initialSupportsSessionRecovery: restoredSnapshot?.supportsSessionRecovery ?? false,
        serviceURLs,
    });
    const runtimeController = createSessionRuntimeController({
        actionHandler,
        conversation,
        getAccessToken: authController.getAccessToken,
        inputConfig: resolvedInputConfig,
        onUnexpectedDisconnect: () => {
            cancelPendingText(
                new Error("Session disconnected before text confirmation"),
            );
            if (manualCloseRequested || pendingOpen || reconnectPromise) {
                return;
            }
            void startReconnect();
        },
        outputConfig: resolvedOutputConfig,
        websocketURL,
    });
    const sessionAPI = createSessionAPI({
        closeRuntime: runtimeController.close,
        conversation,
        ensureLoggedIn: authController.ensureLoggedIn,
        httpClient,
        serviceURLs,
        withAuthorizedToken: authController.withAuthorizedToken,
    });

    conversation.onStateChange((state) => {
        savePersistedConversationSnapshot(
            persistenceStore,
            persistenceKey,
            buildPersistedConversationSnapshot(
                authController.getAccessToken(),
                authController.getSupportsSessionRecovery(),
                state.user,
                state.sessionId,
                state.messages,
            ),
        );
    });

    let pendingOpen: Promise<void> | null = null;
    let reconnectPromise: Promise<void> | null = null;
    let canRetryRuntimeAfterRestoredAuth = restoredSnapshot?.accessToken != null;
    let manualCloseRequested = false;
    let pendingTextCancel: ((reason?: unknown) => void) | null = null;

    function cancelPendingText(reason: unknown): void {
        const cancel = pendingTextCancel;
        pendingTextCancel = null;
        cancel?.(reason);
    }

    function shouldAutoReconnect(): boolean {
        return (
            !manualCloseRequested
            && !!conversation.state.sessionId
            && authController.getSupportsSessionRecovery()
        );
    }

    async function updateSessionRecoverySupport(): Promise<void> {
        const supported = await sessionAPI.probeSessionRecovery(conversation.state.sessionId);
        authController.setSupportsSessionRecovery(supported);
    }

    async function openRuntime(): Promise<void> {
        cancelPendingText(new Error("Session reopened before text confirmation"));
        await authController.ensureLoggedIn();
        await runtimeController.close();
        try {
            await runtimeController.initialize();
            canRetryRuntimeAfterRestoredAuth = false;
        } catch (error) {
            await runtimeController.close();
            if (canRetryRuntimeAfterRestoredAuth) {
                canRetryRuntimeAfterRestoredAuth = false;
                authController.resetAuthState(true);
                await authController.ensureLoggedIn();
                await runtimeController.initialize();
                canRetryRuntimeAfterRestoredAuth = false;
            } else {
                throw error;
            }
        }
        await updateSessionRecoverySupport();
        conversation.state.connectionState = "connected";
    }

    async function startReconnect(): Promise<void> {
        if (reconnectPromise || !shouldAutoReconnect()) {
            if (!manualCloseRequested && !authController.getSupportsSessionRecovery()) {
                conversation.state.connectionState = "disconnected";
            }
            return reconnectPromise ?? Promise.resolve();
        }

        reconnectPromise = (async () => {
            const targetSessionId = conversation.state.sessionId;
            if (!targetSessionId) {
                conversation.state.connectionState = "disconnected";
                return;
            }

            conversation.state.connectionState = "reconnecting";
            conversation.state.streamState = "idle";

            for (let attempt = 0; attempt < MAX_RECONNECT_ATTEMPTS; attempt += 1) {
                if (!shouldAutoReconnect() || conversation.state.sessionId !== targetSessionId) {
                    conversation.state.connectionState = "disconnected";
                    return;
                }
                await delay(RECONNECT_DELAYS_MS[Math.min(attempt, RECONNECT_DELAYS_MS.length - 1)] ?? 5000);
                try {
                    await runtimeController.close();
                    await authController.ensureLoggedIn();
                    await runtimeController.initialize();
                    await sessionAPI.refreshSession(targetSessionId);
                    authController.setSupportsSessionRecovery(true);
                    if (!manualCloseRequested) {
                        conversation.state.connectionState = "connected";
                    }
                    return;
                } catch {
                    await runtimeController.close();
                }
            }

            conversation.state.connectionState = "disconnected";
        })();

        try {
            await reconnectPromise;
        } finally {
            reconnectPromise = null;
        }
    }

    return {
        async open() {
            if (pendingOpen) {
                return pendingOpen;
            }
            if (reconnectPromise) {
                return reconnectPromise;
            }
            pendingOpen = (async () => {
                manualCloseRequested = false;
                await openRuntime();
            })();
            try {
                await pendingOpen;
            } finally {
                pendingOpen = null;
            }
        },
        async close() {
            manualCloseRequested = true;
            cancelPendingText(new Error("Session closed before text confirmation"));
            await runtimeController.close();
            conversation.state.connectionState = "disconnected";
        },
        onStateChange(callback) {
            conversation.onStateChange(callback);
        },
        get state() {
            return conversation.state;
        },
        onInputAudioChunk(callback) {
            runtimeController.onInputAudioChunk(callback);
        },
        onOutputAudioChunk(callback) {
            runtimeController.onOutputAudioChunk(callback);
        },
        onFullAudioChunk(callback) {
            conversation.onFullAudioChunk(callback);
        },
        get muted() {
            return runtimeController.muted;
        },
        async changeVoice(voiceName: string) {
            const runtime = runtimeController.requireRuntime();
            await actionHandler.handleAction(
                "client_change_voice",
                { voiceName },
                runtime.websocket,
                conversation,
                runtime.outputAudioSession,
            );
        },
        async sendText(text: string) {
            const normalizedText = text.trim();
            if (!normalizedText) {
                throw new Error("Text input must not be blank");
            }
            if (normalizedText.length > MAX_TEXT_INPUT_CHARACTERS) {
                throw new Error(
                    `Text input must not exceed ${MAX_TEXT_INPUT_CHARACTERS} characters`,
                );
            }
            if (conversation.state.connectionState !== "connected") {
                throw new Error("Session is not connected");
            }
            if (pendingTextCancel) {
                throw new Error("Another text input is awaiting confirmation");
            }

            const runtime = runtimeController.requireRuntime();
            const receipt = actionHandler.waitForActionMatching<FinishASRData>(
                "finish_asr",
                (data) => isMatchingFinishASRData(data, normalizedText),
            );
            pendingTextCancel = receipt.cancel;
            const timeoutId = globalThis.setTimeout(() => {
                receipt.cancel(new Error("Timed out waiting for text confirmation"));
            }, TEXT_INPUT_RECEIPT_TIMEOUT_MS);

            try {
                await Promise.all([
                    actionHandler.handleAction(
                        "client_submit_text",
                        { text: normalizedText },
                        runtime.websocket,
                        conversation,
                        runtime.outputAudioSession,
                    ),
                    receipt.promise,
                ]);
            } catch (error) {
                receipt.cancel(error);
                throw error;
            } finally {
                globalThis.clearTimeout(timeoutId);
                if (pendingTextCancel === receipt.cancel) {
                    pendingTextCancel = null;
                }
            }
        },
        async uploadFile(file: Blob, endpoint?: string | URL) {
            await sessionAPI.uploadFile(file, endpoint);
        },
        async getSessions() {
            return await sessionAPI.getSessions();
        },
        async switchSession(sessionId: string | null) {
            cancelPendingText(
                new Error("Session switched before text confirmation"),
            );
            await sessionAPI.switchSession(sessionId);
        },
        set muted(value: boolean) {
            runtimeController.muted = value;
        },
    };
}
