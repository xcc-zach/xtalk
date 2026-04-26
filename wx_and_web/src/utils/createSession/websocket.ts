import { getPlatform, Platform } from "./utils";
import { BaseWebSocket } from "./bases/websocket";
import { WebWebSocket } from "./platforms/web"
export { createWebSocket };

function createWebSocket(url: string | any, protocols?: string | string[]): BaseWebSocket {
    switch (getPlatform()) {
        case Platform.Web:
        case Platform.MpWeixin:
            return new WebWebSocket(url, protocols);
        default:
            throw new Error("createWebSocket: Unknown platform");
    }
}