export { BaseWebSocket };
export type { BaseWebSocketCloseEvent, BaseWebSocketEventType, BaseWebSocketMessageEvent };

type BaseWebSocketCloseEvent = {
    code?: number;
    reason?: string;
    wasClean?: boolean;
};

type BaseWebSocketMessageEvent = {
    data: string | ArrayBuffer;
};

type BaseWebSocketEventType = 'open' | 'message' | 'close' | 'error';
abstract class BaseWebSocket {
    /** Unsent transport bytes when reported by the platform. */
    get bufferedAmount(): number { return 0; }
    abstract ready(): boolean;

    abstract send(data: string | ArrayBuffer): void;

    abstract close(): void;

    abstract addEventListener(type: 'open' | 'error', listener: () => any): void;

    abstract addEventListener(type: 'message', listener: (evt: BaseWebSocketMessageEvent) => any): void;

    abstract addEventListener(type: 'close', listener: (evt: BaseWebSocketCloseEvent) => any): void;

    sendJson(data: object): void {
        this.send(JSON.stringify(data));
    }

    sendAudioChunk(pcm_chunk_int16: ArrayBuffer): void {
        this.send(pcm_chunk_int16);
    }
}
