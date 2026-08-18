import type { ActionToFunctionMap } from "./types";
const clientOperationMap: ActionToFunctionMap = {
    "client_change_voice": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            action: "change_voice",
            voice_name: data.voiceName,
        })
    },
    "client_submit_text": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            action: "submit_text",
            text: data.text,
        });
    },
};

export default clientOperationMap;
