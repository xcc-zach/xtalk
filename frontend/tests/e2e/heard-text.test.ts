import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { createSession, registerPlatformRuntime } from "../../dist/index.js";
import { startMockBackend, type MockBackend } from "../support/backend.ts";
import { createNodeTestPlatform } from "../support/node-platform.ts";
import { waitForMessage, waitForState } from "../support/observe.ts";
import { STORY } from "../support/replies.ts";

let backend: MockBackend;

before(async () => {
    backend = await startMockBackend();
});

after(async () => {
    await backend.stop();
});

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

test("the shown assistant text follows playback and ends with the full reply", async () => {
    const platform = createNodeTestPlatform({ autoPlay: false });
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    const speaker = platform.speaker!;

    await session.sendText("请讲一个很长的故事");
    await waitForState(session, () => speaker.activeResponse !== null);
    const storyId = speaker.activeResponse!;
    const shown = () => session.state.messages.find((message) => message.responseId === storyId);
    assert.equal(shown(), undefined);

    let previous = "";
    const deadline = Date.now() + 20_000;
    while (!shown()?.final) {
        assert.ok(Date.now() < deadline, "story playback did not finish");
        const consumed = await speaker.advance(300);
        await sleep(consumed > 0 ? 50 : 10);
        const text = shown()?.content ?? "";
        assert.ok(STORY.startsWith(text), `"${text}" is not a prefix of the story`);
        assert.ok(text.length >= previous.length, "shown text went backwards");
        if (!shown()?.final) {
            assert.ok(text.length < STORY.length, "full text shown before playback finished");
        }
        previous = text;
    }
    const final = await waitForMessage(
        session,
        (message) => message.responseId === storyId && message.final === true,
    );
    assert.equal(final.content, STORY);
    await session.close();
});
