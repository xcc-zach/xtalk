import type { ActionToFunctionMap } from "./handler-types";
import { onVadSpeechStart, onVadSpeechEnd } from "./utils";
import type { BaseEncoding } from "../bases/encoding";
import { createEncoding } from "../encoding";

let encoding: BaseEncoding | null = null;

/** Create the platform encoder on first use so importing stays platform-free. */
function getEncoding(): BaseEncoding {
    if (!encoding) {
        encoding = createEncoding();
    }
    return encoding;
}

const inputMap: ActionToFunctionMap = {
    "xtalk.audio.mix.delta": async (data, websocket, conversation) => {
        conversation.emitFullAudioChunk(data.pcm, data.sample_rate);
    },
    "input_audio_buffer.speech_started": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechStart(data, websocket, conversation, outputAudioSession);
    },
    "input_audio_buffer.speech_stopped": async (data, websocket, conversation, outputAudioSession) => {
        onVadSpeechEnd(data, websocket, conversation, outputAudioSession);
    },
    "session.audio.mix.delta": async (data, websocket, conversation, outputAudioSession) => {
        const audioBase64 = typeof data?.delta === "string" ? data.delta : "";
        if (!audioBase64) {
            return;
        }
        const sampleRate = typeof data?.sample_rate === "number" ? data.sample_rate : 48000;
        const pcmChunkInt16 = getEncoding().decodeBase64(audioBase64);
        conversation.emitFullAudioChunk(pcmChunkInt16, sampleRate);
    }
};

export default inputMap;
