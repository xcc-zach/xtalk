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
        conversation.state.streamState = 'speaking';
    },
    "client_audio_playback_finished": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = 'idle';
        websocket.sendJson({ action: "tts_playback_finished" })
    },
    "client_audio_chunk_played": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({ action: "tts_chunk_played" })
    }
};

export default clientMap;
