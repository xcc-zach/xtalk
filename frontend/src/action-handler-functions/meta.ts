import type { ActionToFunctionMap } from "./handler-types";
const metaMap: ActionToFunctionMap = {
    "conversation.thought.updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.thought = data.text;
    },
    "conversation.caption.updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.caption = data.text;
    },
    "conversation.retrieval.updated": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.retrieval = data.text;
    },
    "response.tool.called": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.tool_call = {
            name: data.name,
            args: data.args || {}
        };
    }
};

export default metaMap;
