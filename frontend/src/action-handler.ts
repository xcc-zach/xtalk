import { BaseWebSocket } from "./bases/websocket";
import { Conversation } from "./conversation";
import { BaseOutputAudioSession } from "./bases/audio-session";
import { ACTION_TO_FUNCTION } from "./action-handler-functions/index";
import type { OutputAudioEncoding, RealtimeEvent } from "./action-handler-functions/protocol";
export { ActionHandler };

type ActionWaitHandle<T> = {
    promise: Promise<T>;
    cancel(reason?: unknown): void;
};

/**
 * Dispatches websocket actions and lets session operations await handled data.
 */
class ActionHandler {
    readonly ACTION_TO_FUNCTION = ACTION_TO_FUNCTION;
    private actionListeners = new Map<string, Set<(data: unknown) => void>>();
    private pendingRequests = new Set<(reason: Error) => void>();
    private nextRequest = 0;
    private outputEncoding: OutputAudioEncoding = "base64";
    private protocolFailed = false;
    private messageQueue: Promise<void> = Promise.resolve();
    private activeAudioResponse: string | null = null;
    private retiredAudioResponses = new Set<string>();

    /** Select the output encoding requested during initial session negotiation. */
    constructor(private readonly requestedEncoding: OutputAudioEncoding = "base64") {}


    /** Reset connection-scoped negotiation and playback identities before opening. */
    resetConnection(): void {
        this.outputEncoding = "base64";
        this.protocolFailed = false;
        this.activeAudioResponse = null;
        this.retiredAudioResponses.clear();
        this.messageQueue = Promise.resolve();
    }

    /** Reject operations awaiting server receipts when their transport closes. */
    cancelPending(reason = new Error("Realtime connection closed")): void {
        for (const cancel of this.pendingRequests) cancel(reason);
    }

    /** Submit real-time user text and wait for its matching accepted Item. */
    async submitText(text: string, websocket: BaseWebSocket, conversation: Conversation, output: BaseOutputAudioSession): Promise<void> {
        await this.requestItem("client_submit_text", {text}, websocket, conversation, output);
    }

    /** Inject one history item and wait for acceptance before opening audio. */
    async restoreItem(message: {role: string; content: string}, websocket: BaseWebSocket, conversation: Conversation, output: BaseOutputAudioSession): Promise<void> {
        await this.requestItem("client_restore_item", message, websocket, conversation, output);
    }

    private async requestItem(action: string, data: object, websocket: BaseWebSocket, conversation: Conversation, output: BaseOutputAudioSession): Promise<void> {
        const id = `client_${Date.now()}_${++this.nextRequest}`;
        const accepted = this.waitForActionMatching("conversation.item.done", data => {
            const event = data as {item?: {id?: string}};
            return event.item?.id === id;
        });
        const failed = this.waitForActionMatching<{error: {message: string}}>("error", data => {
            const event = data as {error?: {event_id?: string}};
            return event.error?.event_id === id;
        });
        const result = Promise.race([
            accepted.promise,
            failed.promise.then(event => { throw new Error(event.error.message); }),
        ]);
        void result.catch(() => {});
        const cancel = (reason: Error) => { accepted.cancel(reason); failed.cancel(reason); };
        this.pendingRequests.add(cancel);
        try {
            await this.handleAction(action, {...data, item_id: id, event_id: id}, websocket, conversation, output);
            await result;
        } finally {
            cancel(new Error("Request settled"));
            this.pendingRequests.delete(cancel);
        }
    }

    /** Wait for protocol readiness while surfacing a server configuration error. */
    waitUntilReady(): ActionWaitHandle<void> {
        const ready = this.waitForActionMatching<void>("session.updated", () => true);
        const failure = this.waitForActionMatching<RealtimeEvent>("error", () => true);
        const promise = Promise.race([
            ready.promise,
            failure.promise.then(() => { throw new Error("Realtime session configuration failed"); }),
        ]).finally(() => { ready.cancel(); failure.cancel(); });
        return {promise, cancel(reason) { ready.cancel(reason); failure.cancel(reason); }};
    }

    /**
     * Resolves after the next matching action finishes its registered handler.
     *
     * @param action Server action name.
     * @returns Promise resolved after the action is applied.
     */
    waitForAction(action: string): Promise<void> {
        return this.waitForActionMatching<void>(action, () => true).promise;
    }

    /**
     * Creates a cancellable waiter for handled action data satisfying a predicate.
     *
     * @param action Server action name.
     * @param predicate Data matcher evaluated after the action handler completes.
     * @returns Cancellable promise handle containing the matching action data.
     */
    waitForActionMatching<T>(
        action: string,
        predicate: (data: unknown) => boolean,
    ): ActionWaitHandle<T> {
        let settled = false;
        let rejectPromise: (reason?: unknown) => void = () => {};
        let listener: (data: unknown) => void;

        const listeners = this.actionListeners.get(action)
            ?? new Set<(data: unknown) => void>();
        const cleanup = (): void => {
            listeners.delete(listener);
            if (listeners.size === 0) {
                this.actionListeners.delete(action);
            }
        };
        const promise = new Promise<T>((resolve, reject) => {
            rejectPromise = reject;
            listener = (data: unknown) => {
                if (settled || !predicate(data)) {
                    return;
                }
                settled = true;
                cleanup();
                resolve(data as T);
            };
            listeners.add(listener);
            this.actionListeners.set(action, listeners);
        });

        return {
            promise,
            cancel(reason: unknown = new Error(`Action wait cancelled: ${action}`)): void {
                if (settled) {
                    return;
                }
                settled = true;
                cleanup();
                rejectPromise(reason);
            },
        };
    }

    private notifyActionHandled(action: string, data: unknown): void {
        const listeners = this.actionListeners.get(action);
        if (!listeners) {
            return;
        }
        for (const callback of [...listeners]) {
            callback(data);
        }
    }

    /** Parses and dispatches one raw network message. */
    async handleMessage(raw: string | ArrayBuffer, websocket: BaseWebSocket, conversation: Conversation, output: BaseOutputAudioSession): Promise<void> {
        const next = this.messageQueue.then(async () => {
            if (this.protocolFailed) return;
            if (typeof raw === "string") {
                const message: RealtimeEvent = JSON.parse(raw);
                if (!message || typeof message !== "object" || typeof message.type !== "string") return;
                await this.handleAction(message.type, message, websocket, conversation, output);
                return;
            }
            // A malformed envelope terminates playback; never guess its encoding.
            try {
                if (this.outputEncoding !== "binary_pcm" || raw.byteLength < 5) throw new Error("Unnegotiated binary audio");
                const length = new DataView(raw).getUint32(0, false);
                if (length < 1 || length > 65536 || 4 + length >= raw.byteLength) throw new Error("Invalid binary header length");
                const header = JSON.parse(new TextDecoder("utf-8", {fatal: true}).decode(new Uint8Array(raw, 4, length)));
                if (!header || Array.isArray(header) || header.version !== 1 || typeof header.event_id !== "string" || !header.event_id || "delta" in header) throw new Error("Invalid binary header");
                const pcm = raw.slice(4 + length);
                if (pcm.byteLength % 2) throw new Error("Unaligned PCM16");
                if (header.type === "xtalk.output_audio.delta") {
                    if (![header.response_id, header.item_id].every(value => typeof value === "string" && value.length > 0)
                        || ![header.output_index, header.content_index].every(value => Number.isInteger(value) && value >= 0)) throw new Error("Missing audio identity");
                    if (header.response_id !== this.activeAudioResponse || this.retiredAudioResponses.has(header.response_id)) return;
                    await this.handleAction(header.type, {...header, pcm}, websocket, conversation, output);
                } else if (header.type === "xtalk.audio_stream.delta") {
                    if (conversation.state.mode !== "end_to_end" || typeof header.stream_id !== "string" || !header.stream_id) throw new Error("Invalid continuous stream identity");
                    await this.handleAction(header.type, {...header, pcm}, websocket, conversation, output);
                } else if (header.type === "xtalk.audio.mix.delta") {
                    if (header.format !== "pcm_s16le" || !Number.isInteger(header.channels) || header.channels <= 0
                        || !Number.isInteger(header.sample_rate) || header.sample_rate <= 0 || pcm.byteLength % (2 * header.channels)) throw new Error("Invalid mixed PCM format");
                    await this.handleAction(header.type, {...header, pcm}, websocket, conversation, output);
                } else throw new Error("Unknown binary audio type");
            } catch (error) {
                this.protocolFailed = true;
                this.cancelPending(new Error("Invalid binary audio protocol"));
                websocket.close();
                await output.stop();
                throw error;
            }
        });
        this.messageQueue = next.catch(() => {});
        return next;
    }

    /** Apply one protocol action and notify receipt waiters after handling it. */
    async handleAction(action: string, data: any, websocket: BaseWebSocket, conversation: Conversation, outputAudioSession: BaseOutputAudioSession) {
        if (action === "session.updated") {
            this.outputEncoding = data.session?.xtalk?.output_audio_encoding ?? "base64";
        }
        if (action === "response.content_part.added" && data.part?.type === "audio") {
            if (this.activeAudioResponse && this.activeAudioResponse !== data.response_id) this.retiredAudioResponses.add(this.activeAudioResponse);
            this.activeAudioResponse = data.response_id;
        }
        if (action === "response.output_audio.playback.stop" || action === "response.output_audio.playback.done") {
            this.retiredAudioResponses.add(data.response_id);
        }
        if (action === "response.output_audio.delta" && (this.retiredAudioResponses.has(data.response_id) || data.response_id !== this.activeAudioResponse)) return;
        const handler = this.ACTION_TO_FUNCTION[action];
        if (handler) {
            await handler(data, websocket, conversation, outputAudioSession);
            if (action === "session.created") {
                websocket.sendJson({type: "session.update", session: {type: "realtime", xtalk: {
                    output_audio_encoding: this.requestedEncoding,
                    ...(data.session?.xtalk?.end_to_end ? {end_to_end_protocol: 2} : {}),
                }}});
            }
            this.notifyActionHandled(action, data);
        } else {
            return;
        }
    }
}
