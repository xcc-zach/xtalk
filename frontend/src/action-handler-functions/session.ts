import type { ActionToFunctionMap } from "./handler-types";
const sessionMap: ActionToFunctionMap = {
    "session.created": async (data, websocket, conversation) => {
        conversation.state.sessionId = data.session.id;
    },
    "session.updated": async () => {},
    "session.queue.updated": async () => {},
    "session.queue.granted": async () => {},
};
export default sessionMap;
