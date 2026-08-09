import type { ActionToFunctionMap } from "./types";
import { onVadSpeechStart, onVadSpeechEnd } from "./utils";
const clientMap: ActionToFunctionMap = {
    "client_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_start" })
    },
    "client_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ action: "vad_speech_end" })
    },
    "client_audio_chunk_started": async (data, websocket, conversation, outputAudioSession) => {
        // Keep listening when user barge-in is detected before queued TTS chunks finish starting.
        if (conversation.state.streamState === "listening") {
            return;
        }
        conversation.state.streamState = 'speaking';
    },
    "client_audio_playback_finished": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data?.response_id !== "string" || !data.response_id) {
            return;
        }
        conversation.state.streamState = 'idle';
        websocket.sendJson({
            action: "tts_playback_finished",
            response_id: data.response_id,
        })
    },
    "client_audio_chunk_played": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data?.response_id !== "string" || !data.response_id) {
            return;
        }
        websocket.sendJson({
            action: "tts_chunk_played",
            response_id: data.response_id,
        })
    }
};

export default clientMap;
