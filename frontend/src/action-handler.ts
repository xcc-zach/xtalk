import { BaseWebSocket } from "./bases/websocket";
import { Conversation } from "./conversation";
import { BaseOutputAudioSession } from "./bases/audio-session";
import { ACTION_TO_FUNCTION } from "./action-handler-functions/index";
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

    /**
     * Applies one action and then notifies any action-data waiters.
     *
     * @param action Action name to dispatch.
     * @param data Action payload.
     * @param websocket Active websocket transport.
     * @param conversation Conversation state store.
     * @param outputAudioSession Active output audio session.
     */
    async handleAction(action: string, data: any, websocket: BaseWebSocket, conversation: Conversation, outputAudioSession: BaseOutputAudioSession) {
        const handler = this.ACTION_TO_FUNCTION[action];
        if (handler) {
            await handler(data, websocket, conversation, outputAudioSession);
            this.notifyActionHandled(action, data);
        } else {
            throw new Error(`No handler found for action: ${action}`);
        }
    }
}
