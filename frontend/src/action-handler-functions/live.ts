import { createEncoding } from "../encoding";
import type { ActionToFunctionMap } from "./handler-types";

/** Continuous Live events use the standard Realtime action dispatcher. */
const liveMap: ActionToFunctionMap = {
    "xtalk.audio_stream.started": (data, _ws, _conversation, output) => output.startAudioStream(data.stream_id),
    "xtalk.audio_stream.delta": async (data, _ws, _conversation, output) => {
        await output.pushStreamAudioChunk(data.pcm ?? createEncoding().decodeBase64(data.delta));
    },
    "xtalk.transcript.delta": (data, _ws, conversation) => conversation.appendTranscript(data),
    "xtalk.audio_stream.ended": async (_data, _ws, conversation, output) => {
        await output.endAudioStream(); conversation.state.streamState = "idle";
    },
    "xtalk.audio_stream.error": (data, _ws, conversation) => {
        conversation.state.error = `${data.code}: ${data.message}`;
        conversation.appendMessage({role: "info", content: conversation.state.error, final: true});
    },
};
export default liveMap;
