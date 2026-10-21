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
};
export type {
    WebAudioBridge,
    WebAudioBridgePublishOptions,
    WebAudioBridgeUserInputConfig,
    WebBridgeInputAudioSessionConfig,
    WebBridgeParticipantId,
} from "./web-audio-bridge";

type PlatformRuntime = {
    createInputAudioSession(config: InputAudioSessionConfig): BaseInputAudioSession;
    createOutputAudioSession(config: OutputAudioSessionConfig): BaseOutputAudioSession;
    createWebSocket(url: string | URL, protocols?: string | string[]): BaseWebSocket;
    delay(milliseconds: number): Promise<void>;
    createDeferredTaskScheduler(): BaseDeferredTaskScheduler;
    createEncoding(): BaseEncoding;
};

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
