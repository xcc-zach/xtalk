import type { ActionToFunctionMap } from "./handler-types";
const clientOperationMap: ActionToFunctionMap = {
    "client_restore_item": async (data, websocket) => {
        websocket.sendJson({type: "conversation.item.create", event_id: data.event_id, item: {id: data.item_id, type: "message", role: data.role, content: [{type: data.role === "assistant" ? "text" : "input_text", text: data.content}]}});
    },
    "client_change_voice": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            type: "session.update",
            session: { audio: { output: { voice: data.voiceName } } },
        })
    },
    "client_submit_text": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            type: "conversation.item.create",
            event_id: data.event_id,
            item: { id: data.item_id, type: "message", role: "user", content: [{type: "input_text", text: data.text}] },
            xtalk: { generate_response: true },
        });
    },
};

export default clientOperationMap;
