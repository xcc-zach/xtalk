import { createEncoding } from "../encoding";
import type { ActionToFunctionMap } from "./handler-types";
import { onVadSpeechStart, onVadSpeechEnd } from "./utils";
const clientMap: ActionToFunctionMap = {
    "client_audio_frame": async (data, websocket) => {
        websocket.sendJson({type: "input_audio_buffer.append", audio: createEncoding().encodePCM(data.audioChunk, data.sampleRate, 24000)});
    },
    "client_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ type: "input_audio_buffer.speech_start" })
    },
    "client_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
        websocket.sendJson({ type: "input_audio_buffer.commit" })
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
            type: "response.output_audio.playback.finished",
            response_id: data.response_id,
        })
    },
    "client_audio_chunk_played": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data?.response_id !== "string" || !data.response_id) {
            return;
        }
        websocket.sendJson({
            type: "response.output_audio.playback.chunk_played",
            response_id: data.response_id,
        })
    }
};

export default clientMap;
