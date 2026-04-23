import { getPlatform, Platform } from "./utils";
import { BaseInputAudioSession, BaseOutputAudioSession } from "./bases/audio-session";
import type { InputAudioSessionConfig, OutputAudioSessionConfig } from "./bases/audio-session";
import { WebInputAudioSession, WebOutputAudioSession } from "./platforms/web";
import { WxInputAudioSession, WxOutputAudioSession } from "./platforms/wx";
export { createInputAudioSession, createOutputAudioSession };

function createInputAudioSession(config: InputAudioSessionConfig): BaseInputAudioSession {
    switch (getPlatform()) {
        case Platform.Web:
            return new WebInputAudioSession(config);
        case Platform.MpWeixin:
            return new WxInputAudioSession(config);
        default:
            throw new Error("createInputAudioSession: Unknown platform");
    }
}

function createOutputAudioSession(config: OutputAudioSessionConfig): BaseOutputAudioSession {
    switch (getPlatform()) {
        case Platform.Web:
            return new WebOutputAudioSession(config);
        case Platform.MpWeixin:
            return new WxOutputAudioSession(config);
        default:
            throw new Error("createOutputAudioSession: Unknown platform");
    }
}