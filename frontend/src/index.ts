/**
 * @packageDocumentation
 *
 * Public frontend client APIs for creating and controlling X-Talk sessions.
 */

import { createSession } from "./session/create";
import { createAudioBridge, registerPlatformRuntime } from "./platforms/index";
import { BaseInputAudioSession, BaseOutputAudioSession } from "./bases/audio-session";
import { BaseEncoding } from "./bases/encoding";
import { BaseDeferredTaskScheduler } from "./bases/task-scheduler";
import { BaseWebSocket } from "./bases/websocket";

export { createSession };
export { createAudioBridge };
export { registerPlatformRuntime };
export {
    BaseDeferredTaskScheduler,
    BaseEncoding,
    BaseInputAudioSession,
    BaseOutputAudioSession,
    BaseWebSocket,
};
export type {
    Session,
    SessionConfig,
} from "./session/types";
export type { PlatformRuntime } from "./platforms/index";
export type {
    InputAudioSessionConfig,
    OutputAudioSessionConfig,
    OutputAudioStopResult,
} from "./bases/audio-session";
export type {
    BaseWebSocketCloseEvent,
    BaseWebSocketEventType,
    BaseWebSocketMessageEvent,
} from "./bases/websocket";
