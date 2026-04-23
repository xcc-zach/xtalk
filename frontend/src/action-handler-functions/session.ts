import type { ActionToFunctionMap } from "./types";
const sessionMap: ActionToFunctionMap = {
    "session_info": async (data, websocket, conversation, outputAudioSession) => {
        const sid = data.session_id || null;
        conversation.state.sessionId = sid;
    },
};

export default sessionMap;
