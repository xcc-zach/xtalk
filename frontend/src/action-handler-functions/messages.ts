import type { ActionToFunctionMap } from "./types";
const messagesMap: ActionToFunctionMap = {
    "error": async (data, websocket, conversation, outputAudioSession) => {
        const message = typeof data === "string" && data.trim()
            ? data.trim()
            : "The conversation failed without an error message.";
        conversation.finalizePendingMessages();
        conversation.appendMessage({
            role: "info",
            content: message,
            final: true,
        });
        conversation.state.streamState = "idle";
    },
    "update_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.finalizePendingMessages("assistant");
        conversation.appendMessage({
            role: "user",
            content: data.text,
            final: false
        })
    },
    "finish_asr": async (data, websocket, conversation, outputAudioSession) => {
        conversation.finalizePendingMessages("assistant");
        conversation.appendMessage({
            role: "user",
            content: data.text,
            final: true
        })
        conversation.state.streamState = 'processing';
    },
    "update_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            final: false
        })
    },
    "finish_resp": async (data, websocket, conversation, outputAudioSession) => {
        conversation.appendMessage({
            role: "assistant",
            content: data.text,
            final: true
        })
    },
};

export default messagesMap;
