/**
 * Helpers that wait for observable session state.
 */

import type { Session } from "../../dist/index.js";
import type { VirtualOutputAudioSession } from "./node-platform.ts";

type SessionState = Session["state"];
type Message = SessionState["messages"][number];

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Resolve once `predicate` holds for the session state. */
export async function waitForState(
    session: Session,
    predicate: (state: SessionState) => boolean,
    timeoutMs = 10_000,
): Promise<SessionState> {
    const deadline = Date.now() + timeoutMs;
    while (!predicate(session.state)) {
        if (Date.now() > deadline) {
            throw new Error(`state condition not met within ${timeoutMs} ms: ${JSON.stringify(session.state.messages)}`);
        }
        await sleep(10);
    }
    return session.state;
}

/** Resolve with the first message satisfying `predicate`. */
export async function waitForMessage(
    session: Session,
    predicate: (message: Message) => boolean,
    timeoutMs = 10_000,
): Promise<Message> {
    const state = await waitForState(session, (current) => current.messages.some(predicate), timeoutMs);
    return state.messages.find(predicate)!;
}

/** Advance the virtual speaker until `ms` of one response were heard. */
export async function hearUntil(
    speaker: VirtualOutputAudioSession,
    responseId: string,
    ms: number,
    timeoutMs = 10_000,
): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (speaker.heardMs(responseId) < ms) {
        if (Date.now() > deadline) {
            throw new Error(`only ${speaker.heardMs(responseId)} ms heard within ${timeoutMs} ms`);
        }
        const consumed = await speaker.advance(Math.min(100, ms - speaker.heardMs(responseId)));
        if (consumed === 0) await sleep(10);
    }
}

/** Record every streamState value the session goes through. */
export function recordStreamStates(session: Session): string[] {
    const states: string[] = [];
    session.onStateChange((state) => {
        if (states[states.length - 1] !== state.streamState) {
            states.push(state.streamState);
        }
    });
    return states;
}
