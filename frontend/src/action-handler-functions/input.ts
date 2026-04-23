import type { ActionToFunctionMap } from "./types";
import { onVadSpeechStart, onVadSpeechEnd } from "./utils";

function decodeBase64ToArrayBuffer(base64: string): ArrayBuffer {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
}

const inputMap: ActionToFunctionMap = {
    "vad_speech_start": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
    },
    "vad_speech_end": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
    },
    "full_audio_frame": async (data, websocket, conversation, outputAudioSession) => {
        const audioBase64 = typeof data?.audio_base64 === "string" ? data.audio_base64 : "";
        if (!audioBase64) {
            return;
        }
        const sampleRate = typeof data?.sample_rate === "number" ? data.sample_rate : 48000;
        const pcmChunkInt16 = decodeBase64ToArrayBuffer(audioBase64);
        conversation.emitFullAudioChunk(pcmChunkInt16, sampleRate);
    }
};

export default inputMap;
