import type { ActionToFunctionMap } from "./types";
const clientOperationMap: ActionToFunctionMap = {
    "client_change_voice": async (data, websocket, conversation, outputAudioSession) => {
        websocket.sendJson({
            action: "change_voice",
            voice_name: data.voiceName,
        })
    },
    "client_upload_file": async (data, websocket, conversation, outputAudioSession) => {
        conversation.state.streamState = "processing";
        const file = data.file as Blob;
        const endpoint = data.endpoint as string | URL;
        const formData = new FormData();
        formData.append("session_id", conversation.state.sessionId!);
        formData.append("file", file);
        const resp = await fetch(endpoint, {
            method: "POST",
            body: formData,
        });
        if (!resp.ok) {
            conversation.state.streamState = "idle";
        }
    },
};

export default clientOperationMap;
