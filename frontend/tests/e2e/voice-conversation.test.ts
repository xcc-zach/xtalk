import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { createSession, registerPlatformRuntime } from "../../dist/index.js";
import { startMockBackend, type MockBackend } from "../support/backend.ts";
import { createNodeTestPlatform } from "../support/node-platform.ts";
import { recordStreamStates, waitForMessage } from "../support/observe.ts";
import { GREETING_REPLY } from "../support/replies.ts";
import { encode } from "../support/speech.ts";

const MIC_SAMPLE_RATE = 16000;
let backend: MockBackend;
let backendWithoutVad: MockBackend;

before(async () => {
    backend = await startMockBackend();
    backendWithoutVad = await startMockBackend({ noVad: true });
});

after(async () => {
    await backend.stop();
    await backendWithoutVad.stop();
});

function greeting(): ArrayBuffer {
    return encode(0, { voicedMs: 600, trailingSilenceMs: 700, sampleRate: MIC_SAMPLE_RATE });
}

test("spoken speech is transcribed and answered with speech", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    const streamStates = recordStreamStates(session);
    await session.open();

    await platform.microphone!.pushPcm(greeting());
    const user = await waitForMessage(session, (message) => message.role === "user" && message.final === true);
    assert.equal(user.content, "你好");
    const assistant = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content === GREETING_REPLY,
    );
    assert.ok(streamStates.includes("listening"));
    assert.ok(platform.speaker!.heardMs(assistant.responseId!) > 0);
    await session.close();
});

test("muted speech is neither transcribed nor answered", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();

    session.muted = true;
    await platform.microphone!.pushPcm(greeting());
    session.muted = false;
    await platform.microphone!.pushPcm(greeting());
    await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content === GREETING_REPLY,
    );
    assert.deepEqual(
        session.state.messages.filter((message) => message.role === "user").map((message) => message.content),
        ["你好"],
    );
    assert.equal(session.state.messages.filter((message) => message.role === "assistant").length, 1);
    await session.close();
});

test("client speech boundaries complete a turn without server VAD", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backendWithoutVad.url);
    await session.open();

    await platform.microphone!.speechStart();
    await platform.microphone!.pushPcm(greeting(), { realtime: false });
    await platform.microphone!.speechEnd();
    const user = await waitForMessage(session, (message) => message.role === "user" && message.final === true);
    assert.equal(user.content, "你好");
    await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content === GREETING_REPLY,
    );
    await session.close();
});
