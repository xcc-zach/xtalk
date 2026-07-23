import type { ActionToFunctionMap } from "./types";
const outputMap: ActionToFunctionMap = {
    "start_tts": async (data, websocket, conversation, outputAudioSession) => {
        // Leave blank, no use
    },
    "pause_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "stop_tts": async (data, websocket, conversation, outputAudioSession) => {
        const result = await outputAudioSession.stop();
        websocket.sendJson({
            action: "tts_playback_stopped",
            played_audio_ms: result.unconfirmedPlayedMs,
        });
    },
    "resume_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
    "tts_finished": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.notifyTTSFinished();
    },
};

export default outputMap;
