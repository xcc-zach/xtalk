export { BaseWebSocket };
type WebSocketEventType = 'open' | 'message' | 'close' | 'error';
abstract class BaseWebSocket {
    abstract ready(): boolean;

    abstract send(data: string | ArrayBuffer): void;

    abstract close(): void;

    abstract addEventListener(type: WebSocketEventType, listener: (evt?: any) => any): void;

    sendJson(data: object): void {
        this.send(JSON.stringify(data));
    }

    sendAudioChunk(pcm_chunk_int16: ArrayBuffer): void {
        this.send(pcm_chunk_int16);
    }
}