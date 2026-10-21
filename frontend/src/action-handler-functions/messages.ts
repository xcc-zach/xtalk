import type { ActionToFunctionMap } from "./handler-types";
const sessionPreviews = new WeakMap<object, Map<string, {text: string; revision: number; final: boolean}>>();
function previewsFor(conversation: object) {
    let value = sessionPreviews.get(conversation);
    if (!value) { value = new Map(); sessionPreviews.set(conversation, value); }
    return value;
}
const messagesMap: ActionToFunctionMap = {
    "conversation.item.done": async (data, websocket, conversation) => {
        const item = data.item;
        const part = item.content?.find((value: any) => value.type === "text" || value.type === "input_text");
        if (part && ["user", "assistant"].includes(item.role)) conversation.appendMessage({role: item.role, content: part.text, final: true});
    },
    "response.output_text.delta": async (data, websocket, conversation) => {
        const previous = conversation.state.generatedResponses[data.response_id]?.text ?? "";
        conversation.updateGeneratedResponse(data.response_id, previous + data.delta, false);
        conversation.updateAssistantResponse(data.response_id, previous + data.delta, false);
    },
    "response.output_text.done": async (data, websocket, conversation) => {
        conversation.updateGeneratedResponse(data.response_id, data.text, true);
        conversation.updateAssistantResponse(data.response_id, data.text, true);
    },
    "response.output_audio_transcript.delta": async (data, websocket, conversation) => {
        const previous = conversation.state.generatedResponses[data.response_id]?.text ?? "";
        conversation.updateGeneratedResponse(data.response_id, previous + data.delta, false);
    },
    "response.output_audio_transcript.done": async (data, websocket, conversation) => {
        conversation.updateGeneratedResponse(data.response_id, data.transcript, true);
    },
    "error": async (data, websocket, conversation, outputAudioSession) => {
        const message = typeof data.error?.message === "string" && data.error.message.trim()
            ? data.error.message.trim()
            : "The conversation failed without an error message.";
        conversation.finalizePendingMessages();
        conversation.appendMessage({
            role: "info",
            content: message,
            final: true,
        });
        conversation.state.streamState = "idle";
    },
    "conversation.item.input_audio_transcription.delta": async (data, websocket, conversation, outputAudioSession) => {
        const key = data.item_id;
        const previews = previewsFor(conversation);
        const previous = previews.get(key);
        if (previous?.final || (previous && data.xtalk.revision <= previous.revision)) return;
        const text = data.xtalk.mode === "replace" ? data.delta : (previous?.text ?? "") + data.delta;
        previews.set(key, {text, revision: data.xtalk.revision, final: false});
        conversation.updateInputItem(key, text, false);
    },
    "conversation.item.input_audio_transcription.failed": async (data, websocket, conversation) => {
        const previous = previewsFor(conversation).get(data.item_id);
        previewsFor(conversation).set(data.item_id, {text: previous?.text ?? "", revision: Infinity, final: true});
        conversation.updateInputItem(data.item_id, previous?.text ?? "", true);
        conversation.appendMessage({role: "info", content: data.error.message, final: true});
        conversation.state.streamState = "idle";
    },
    "conversation.item.input_audio_transcription.completed": async (data, websocket, conversation) => {
        previewsFor(conversation).set(data.item_id, {text: data.transcript, revision: Infinity, final: true});
        conversation.updateInputItem(data.item_id, data.transcript, true);
        conversation.state.streamState = "processing";
    },
    "response.output_audio.playback.updated": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id !== "string" || !data.response_id) {
            return;
        }
        conversation.updateAssistantResponse(
            data.response_id,
            typeof data.text === "string" ? data.text : "",
            false,
        );
    },
    "response.output_audio.playback.done": async (data, websocket, conversation, outputAudioSession) => {
        if (typeof data.response_id !== "string" || !data.response_id) {
            return;
        }
        conversation.updateAssistantResponse(
            data.response_id,
            typeof data.text === "string" ? data.text : "",
            true,
        );
    },
};

export default messagesMap;
