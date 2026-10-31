import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { createSession, registerPlatformRuntime } from "../../dist/index.js";
import { startMockBackend, type MockBackend } from "../support/backend.ts";
import { createNodeTestPlatform } from "../support/node-platform.ts";
import { recordStreamStates, waitForMessage, waitForState } from "../support/observe.ts";
import { GREETING_REPLY, summaryReply } from "../support/replies.ts";

let backend: MockBackend;

before(async () => {
    backend = await startMockBackend();
});

after(async () => {
    await backend.stop();
});

test("a typed greeting is answered with speech and shown once played", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    const streamStates = recordStreamStates(session);
    let outputChunks = 0;
    session.onOutputAudioChunk(() => { outputChunks += 1; });
    await session.open();

    await session.sendText("你好");
    const user = await waitForMessage(session, (message) => message.role === "user");
    assert.equal(user.content, "你好");
    const assistant = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content === GREETING_REPLY,
    );
    assert.ok(assistant.responseId);
    assert.ok(platform.speaker!.heardMs(assistant.responseId!) > 0);
    assert.ok(outputChunks > 0);
    await waitForState(session, (state) => state.streamState === "idle");
    assert.ok(streamStates.includes("speaking"));
    assert.equal(streamStates[streamStates.length - 1], "idle");
    assert.ok(session.state.messages.indexOf(user) < session.state.messages.indexOf(assistant));
    await session.close();
});

test("empty text is rejected locally", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    await assert.rejects(session.sendText("   "));
    await session.close();
});

test("the second turn refers to the first reply", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    await session.sendText("你好");
    await waitForMessage(session, (message) => message.role === "assistant" && message.final === true);
    await session.sendText("你刚才说了什么");
    const summary = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content.startsWith("上一句"),
    );
    assert.equal(summary.content, summaryReply(GREETING_REPLY, 2));
    await session.close();
});
