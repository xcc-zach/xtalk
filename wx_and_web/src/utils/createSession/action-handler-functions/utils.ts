import type { ActionHandlerFunction } from "./types";
export { onVadSpeechStart, onVadSpeechEnd };

const onVadSpeechStart: ActionHandlerFunction = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'listening';
}

const onVadSpeechEnd: ActionHandlerFunction = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'processing';
}