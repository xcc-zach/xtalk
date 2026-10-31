import assert from "node:assert/strict";
import { after, before, test } from "node:test";

import { createSession, registerPlatformRuntime } from "../../dist/index.js";
import { startMockBackend, type MockBackend } from "../support/backend.ts";
import { createNodeTestPlatform } from "../support/node-platform.ts";
import { waitForState } from "../support/observe.ts";

const API_KEY = "secret-key";
let backend: MockBackend;
let keyedBackend: MockBackend;

before(async () => {
    backend = await startMockBackend();
    keyedBackend = await startMockBackend({ apiKey: API_KEY });
});

after(async () => {
    await backend.stop();
    await keyedBackend.stop();
});

test("open connects, close disconnects and releases the devices", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const session = createSession(backend.url);
    await session.open();
    assert.equal(session.state.connectionState, "connected");
    assert.equal(platform.microphone?.isOpen, true);
    assert.equal(platform.speaker?.isOpen, true);

    await session.close();
    await waitForState(session, (state) => state.connectionState === "disconnected");
    assert.equal(platform.microphone?.isOpen, false);
    assert.equal(platform.speaker?.isOpen, false);
});

test("a correct API key connects and a wrong one is refused", async () => {
    const platform = createNodeTestPlatform();
    registerPlatformRuntime(platform.runtime);
    const accepted = createSession(keyedBackend.url, { apiKey: API_KEY });
    await accepted.open();
    assert.equal(accepted.state.connectionState, "connected");
    await accepted.close();

    const refused = createSession(keyedBackend.url, { apiKey: "wrong-key" });
    await assert.rejects(refused.open());
    assert.notEqual(refused.state.connectionState, "connected");
});
