import type { ActionToFunctionMap } from "./types";
const messagesMap: ActionToFunctionMap = {
    "update_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "user",
            content: data.text,
            turnId: data.turn_id
        })
    },
    "finish_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "user",
            content: data.text,
            turnId: data.turn_id
        })
    },
    "update_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            turnId: data.turn_id
        })
    },
    "finish_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            turnId: data.turn_id
        })
    },
};

export default messagesMap;
