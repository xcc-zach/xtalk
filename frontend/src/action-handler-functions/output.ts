import { createEncoding } from "../encoding";
import type { ActionToFunctionMap } from "./handler-types";
const outputMap: ActionToFunctionMap = {
    "xtalk.output_audio.delta": async (data, websocket, conversation, output) => {
        await output.pushAudioChunk(data.pcm);
    },
    "response.output_audio.delta": async (data, websocket, conversation, output) => {
        await output.pushAudioChunk(createEncoding().decodeBase64(data.delta));
    },
    "response.content_part.added": async (data, websocket, conversation, outputAudioSession) => {
        if (data.part?.type === "audio" && typeof data.response_id === "string" && data.response_id) {
            outputAudioSession.startTTS(data.response_id);
        }
    },
    "response.output_audio.playback.pause": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.pause();
    },
    "response.output_audio.playback.stop": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id !== "string" || !data.response_id) {
            return;
        }
        const result = await outputAudioSession.stop(data.response_id);
        websocket.sendJson({
            type: "response.output_audio.playback.stopped",
            response_id: data.response_id,
            played_audio_ms: result.unconfirmedPlayedMs,
        });
    },
    "response.output_audio.playback.resume": async (data, websocket, conversation, outputAudioSession) => {
        await outputAudioSession.resume();
    },
    "response.output_audio.done": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id === "string" && data.response_id) {
            await outputAudioSession.notifyTTSFinished(data.response_id);
        }
    },
};

export default outputMap;
