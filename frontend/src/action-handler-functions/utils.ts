import type { ActionHandlerFunction } from "./handler-types";
export { onVadSpeechStart, onVadSpeechEnd };

const onVadSpeechStart: ActionHandlerFunction = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'listening';
}

const onVadSpeechEnd: ActionHandlerFunction = async (data, websocket, conversation, outputAudioSession) => {
    conversation.state.streamState = 'idle';
}