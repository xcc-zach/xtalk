import type { ActionToFunctionMap } from "./handler-types";
const latencyMap: ActionToFunctionMap = {
    "session.latency.updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.updateLatency(
            {
                network: Number(data.network_latency_ms) || 0,
                asr: Number(data.asr_latency_ms) || 0,
                llmFirstToken: Number(data.llm_first_token_ms) || 0,
                llmSentence: Number(data.llm_sentence_ms) || 0,
                ttsFirstChunk: Number(data.tts_first_chunk_ms) || 0,
            }
        )
    },
};

export default latencyMap;
