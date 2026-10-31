import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { createSession, registerPlatformRuntime } from "../../dist/index.js";
import { startMockBackend, type MockBackend } from "../support/backend.ts";
import { createNodeTestPlatform } from "../support/node-platform.ts";
import { hearUntil, waitForMessage, waitForState } from "../support/observe.ts";
import { GREETING_REPLY, STORY, summaryReply } from "../support/replies.ts";
import { encode } from "../support/speech.ts";

let backend: MockBackend;

before(async () => {
    backend = await startMockBackend();
});

after(async () => {
    await backend.stop();
});

test("speaking over the story stops it and keeps only the heard part", async () => {
    const platform = createNodeTestPlatform({ autoPlay: false });
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    const speaker = platform.speaker!;

    await session.sendText("请讲一个很长的故事");
    await waitForState(session, () => speaker.activeResponse !== null);
    const storyId = speaker.activeResponse!;
    await hearUntil(speaker, storyId, 1500);

    await platform.microphone!.pushPcm(
        encode(0, { voicedMs: 600, trailingSilenceMs: 700, sampleRate: 16000 }),
    );
    const story = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.responseId === storyId && message.final === true,
    );
    const heardAtStop = speaker.heardMs(storyId);
    // The story is settled; let the new reply play normally.
    speaker.autoPlay = true;
    await speaker.resume();
    assert.ok(story.content.length > 0);
    assert.ok(STORY.startsWith(story.content));
    assert.notEqual(story.content, STORY);

    const greeting = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content === GREETING_REPLY,
    );
    assert.notEqual(greeting.responseId, storyId);
    assert.equal(speaker.heardMs(storyId), heardAtStop);

    await session.sendText("你刚才说了什么");
    const summary = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content.startsWith("上一句"),
    );
    assert.equal(summary.content, summaryReply(GREETING_REPLY, 4));
    await session.close();
});

test("interrupting with a question answers it from the heard prefix", async () => {
    const platform = createNodeTestPlatform({ autoPlay: false });
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    const speaker = platform.speaker!;

    await session.sendText("请讲一个很长的故事");
    await waitForState(session, () => speaker.activeResponse !== null);
    const storyId = speaker.activeResponse!;
    await hearUntil(speaker, storyId, 1500);

    await platform.microphone!.pushPcm(
        encode(4, { voicedMs: 600, trailingSilenceMs: 700, sampleRate: 16000 }),
    );
    const story = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.responseId === storyId && message.final === true,
    );
    assert.ok(STORY.startsWith(story.content));
    assert.notEqual(story.content, STORY);
    speaker.autoPlay = true;
    await speaker.resume();
    const summary = await waitForMessage(
        session,
        (message) => message.role === "assistant" && message.final === true && message.content.startsWith("上一句"),
    );
    assert.equal(summary.content, summaryReply(story.content, 2));
    await session.close();
});
