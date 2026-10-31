import { BaseInputAudioSession, BaseOutputAudioSession } from "../bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "../bases/audio-session";
import { BaseEncoding } from "../bases/encoding";
import { BaseDeferredTaskScheduler } from "../bases/task-scheduler";
import { BaseWebSocket } from "../bases/websocket";
import {
    WebBridgeInputAudioSession,
    createWebAudioBridge as createWebAudioBridgeInternal,
} from "./web-audio-bridge";
import type { WebAudioBridge } from "./web-audio-bridge";
import {
    WebDeferredTaskScheduler,
    WebEncoding,
    WebInputAudioSession,
    WebOutputAudioSession,
    WebWebSocket,
} from "./web";

export {
    createAudioBridge,
    getPlatformRuntime,
    registerPlatformRuntime,
};
export type { PlatformRuntime };
export type {
    WebAudioBridge,
    WebAudioBridgePublishOptions,
    WebAudioBridgeUserInputConfig,
    WebBridgeInputAudioSessionConfig,
    WebBridgeParticipantId,
} from "./web-audio-bridge";

/**
 * Factories that supply every platform-specific resource used by a session.
 *
 * Register a custom runtime with {@link registerPlatformRuntime} to run the
 * client outside a browser, for example inside automated tests.
 */
type PlatformRuntime = {
    /** Creates the microphone or bridge-backed audio source. */
    createInputAudioSession(config: InputAudioSessionConfig): BaseInputAudioSession;
    /** Creates the speaker-backed playback sink. */
    createOutputAudioSession(config: OutputAudioSessionConfig): BaseOutputAudioSession;
    /** Creates the transport used for the Realtime connection. */
    createWebSocket(url: string | URL, protocols?: string | string[]): BaseWebSocket;
    /** Resolves after the given delay. */
    delay(milliseconds: number): Promise<void>;
    /** Creates the scheduler used to defer observer callbacks. */
    createDeferredTaskScheduler(): BaseDeferredTaskScheduler;
    /** Creates the Base64 and PCM encoder. */
    createEncoding(): BaseEncoding;
};

let registeredRuntime: PlatformRuntime | null = null;

/**
 * Use an explicit platform runtime instead of browser detection.
 *
 * Call this before creating a session. A later call replaces the runtime for
 * sessions created afterwards; sessions already created keep their resources.
 *
 * @param runtime Platform runtime supplying every session resource.
 */
function registerPlatformRuntime(runtime: PlatformRuntime): void {
    registeredRuntime = runtime;
}

enum Platform {
    Web,
}

function detectPlatform(): Platform {
    if (typeof window !== "undefined" && typeof document !== "undefined") {
        return Platform.Web;
    }
    throw new Error("Unknown platform");
}

const WEB_PLATFORM_RUNTIME: PlatformRuntime = {
    createInputAudioSession(config) {
        if (config.mode === "web_bridge") {
            return new WebBridgeInputAudioSession(config as any);
        }
        return new WebInputAudioSession(config);
    },
    createOutputAudioSession(config) {
        return new WebOutputAudioSession(config);
    },
    createWebSocket(url, protocols) {
        return new WebWebSocket(url, protocols);
    },
    delay(milliseconds) {
        return new Promise((resolve) => {
            window.setTimeout(resolve, milliseconds);
        });
    },
    createDeferredTaskScheduler() { return new WebDeferredTaskScheduler(); },
    createEncoding() {
        return new WebEncoding();
    },

};

function getPlatformRuntime(): PlatformRuntime {
    if (registeredRuntime) {
        return registeredRuntime;
    }
    switch (detectPlatform()) {
        case Platform.Web:
            return WEB_PLATFORM_RUNTIME;
        default:
            throw new Error("Unknown platform");
    }
}

/**
 * Create an audio bridge for the detected frontend platform.
 *
 * @returns A new platform audio bridge instance.
 */
function createAudioBridge(): WebAudioBridge {
    switch (detectPlatform()) {
        case Platform.Web:
            return createWebAudioBridgeInternal();
        default:
            throw new Error("Unknown platform");
    }
}
