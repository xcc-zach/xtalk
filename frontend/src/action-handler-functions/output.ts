import type { ActionToFunctionMap } from "./types";
const outputMap: ActionToFunctionMap = {
    "start_tts": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id === "string" && data.response_id) {
            outputAudioSession.startTTS(data.response_id);
        }
    },
    "pause_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "stop_tts": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id !== "string" || !data.response_id) {
            return;
        }
        const result = await outputAudioSession.stop(data.response_id);
        websocket.sendJson({
            action: "tts_playback_stopped",
            response_id: data.response_id,
            played_audio_ms: result.unconfirmedPlayedMs,
        });
    },
    "resume_tts": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
    "tts_finished": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id === "string" && data.response_id) {
            await outputAudioSession.notifyTTSFinished(data.response_id);
        }
    },
};

export default outputMap;
