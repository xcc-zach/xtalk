import type { ActionToFunctionMap } from "./types";
const outputMap: ActionToFunctionMap = {
    "start_tts": async (data, websocket, conversation, outputAudioSession) => {
        // Leave blank, no use
    },
    "pause_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "stop_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.stop();
    },
    "resume_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
};

export default outputMap;
