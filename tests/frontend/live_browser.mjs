/** Single-call browser acceptance against mock_live_server.py. */
import assert from "node:assert/strict";
const { chromium } = await import(process.env.XTALK_PLAYWRIGHT_MODULE || "playwright");
const demo = process.env.XTALK_DEMO_URL || "http://127.0.0.1:7636";
const fixture = process.env.XTALK_FIXTURE_URL || "http://127.0.0.1:8765";
const browser = await chromium.launch({ headless: true, args: ["--autoplay-policy=no-user-gesture-required"] });
const page = await browser.newPage();
const errors = [];
const requests = [];
page.on("pageerror", error => errors.push(error.message));
page.on("request", req => requests.push(new URL(req.url()).pathname));
await page.addInitScript(() => {
    const NativeContext = window.AudioContext;
    window.testMedia = { contexts: [], tracks: [], sources: new Set(), delay: 0, deny: false, omitEnded: false, suspendNext: false };
    window.AudioContext = class extends NativeContext {
        constructor(options) { super(options); window.testMedia.contexts.push(this); }
        createBufferSource() {
            const source = super.createBufferSource();
            const disconnect = source.disconnect.bind(source);
            source.disconnect = (...args) => { window.testMedia.sources.delete(source); return disconnect(...args); };
            if (this.sampleRate === 24000) window.testMedia.sources.add(source);
            const start = source.start.bind(source);
            source.start = (...args) => {
                start(...args);
                if (window.testMedia.suspendNext && this.sampleRate === 24000) {
                    window.testMedia.suspendNext = false;
                    void this.suspend().then(() => setTimeout(() => { void this.resume(); }, 700));
                }
            };
            if (window.testMedia.omitEnded && this.sampleRate === 24000) Object.defineProperty(source, "onended", { get: () => null, set: () => {} });
            return source;
        }
    };
    navigator.mediaDevices.getUserMedia = async () => {
        if (window.testMedia.delay) await new Promise(resolve => setTimeout(resolve, window.testMedia.delay));
        if (window.testMedia.deny) throw new DOMException("Microphone permission denied", "NotAllowedError");
        const context = new AudioContext({ sampleRate: 24000 });
        const oscillator = context.createOscillator();
        const output = context.createMediaStreamDestination();
        oscillator.connect(output);
        oscillator.start();
        await context.resume();
        for (const track of output.stream.getTracks()) {
            window.testMedia.tracks.push(track);
            const stop = track.stop.bind(track);
            track.stop = () => { stop(); try { oscillator.stop(); } catch {} if (context.state !== "closed") void context.close(); };
        }
        return output.stream;
    };
});
async function configure(fields) {
    const response = await fetch(`${fixture}/fixture`, {method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(fields)});
    assert.equal(response.ok, true);
}
async function metrics() { return (await (await fetch(`${fixture}/fixture`)).json()).calls; }
async function waitConnected() { await page.waitForFunction(() => document.querySelector("#connection-state").textContent === "connected"); }
async function waitStopped() { await page.waitForFunction(() => document.querySelector("#connection-state").textContent === "disconnected"); }
try {
    await configure({startup_delay_s: 0, scenario: "burst"});
    await page.goto(demo);
    await page.evaluate(() => { window.testMedia.suspendNext = true; });
    await page.locator("#btn-call").click();
    await waitConnected();
    await page.waitForFunction(() => document.querySelector("#messages").textContent.includes("本地协议测试成功"));
    assert.equal(await page.locator("#call-status").textContent(), "可以讲话");
    await page.waitForFunction(() => window.testMedia.sources.size === 0);
    assert.equal(await page.locator("#btn-upload-file").isVisible(), false);
    assert.equal(await page.locator("#btn-toggle-sessions").isVisible(), false);
    assert.equal(await page.getByRole("button", {name: "暂停声音", exact: true}).count(), 0);
    assert.equal(requests.some(path => path.includes("/api/sessions") || path === "/api/voices"), false);
    await page.locator("#btn-mute").click();
    await page.waitForTimeout(300);
    const mutedMetrics = (await metrics()).at(-1);
    assert(mutedMetrics.zero_frames > 0);
    await page.locator("#btn-call").click();
    await waitStopped();
    assert((await page.locator("#messages").textContent()).includes("本地协议测试成功"));
    // Ten fresh calls, preserving the microphone preference and clearing captions.
    for (let i = 0; i < 10; i++) {
        await page.locator("#btn-call").click();
        await waitConnected();
        assert.equal(await page.locator("#btn-mute").getAttribute("aria-pressed"), "true");
        assert.equal((await page.locator("#messages").textContent()).includes("本地协议测试成功"), false);
        await page.locator("#btn-mute").click();
        await page.waitForFunction(() => document.querySelector("#messages").textContent.includes("本地协议测试成功"));
        await page.locator("#btn-mute").click();
        await page.locator("#btn-call").click();
        await waitStopped();
        await page.waitForFunction(() => window.testMedia.tracks.every(track => track.readyState === "ended") && window.testMedia.sources.size === 0 && window.testMedia.contexts.filter(context => context.sampleRate === 24000).every(context => context.state === "closed"));
    }
    await page.reload();
    assert.equal((await page.locator("#messages").textContent()).includes("本地协议测试成功"), false);
    // The fallback must acknowledge every source when onended is omitted.
    await page.evaluate(() => { window.testMedia.omitEnded = true; });
    await page.locator("#btn-call").click();
    await waitConnected();
    await page.waitForFunction(() => document.querySelector("#messages").textContent.includes("本地协议测试成功"));
    await page.waitForTimeout(900);
    assert.equal(await page.evaluate(() => window.testMedia.sources.size), 0);
    await page.locator("#btn-call").click();
    await waitStopped();
    // Cancelling while getUserMedia is pending must clean up its late result.
    await page.evaluate(() => { window.testMedia.delay = 700; window.testMedia.omitEnded = false; });
    await page.locator("#btn-call").click();
    await page.waitForTimeout(150);
    await page.locator("#btn-call").click();
    await waitStopped();
    await page.waitForTimeout(900);
    assert.equal(await page.evaluate(() => window.testMedia.tracks.every(track => track.readyState === "ended")), true);
    await page.evaluate(() => { window.testMedia.delay = 0; window.testMedia.deny = true; });
    await page.locator("#btn-call").click();
    await waitStopped();
    await page.waitForFunction(() => document.querySelector("#toast-region").textContent.includes("permission denied"));
    await page.evaluate(() => { window.testMedia.deny = false; });
    // Slow upstream stays connecting, then sends the first ready speech without loss.
    await configure({startup_delay_s: 0.8, scenario: "command_rejection"});
    await page.locator("#btn-call").click();
    await page.waitForTimeout(300);
    assert.notEqual(await page.locator("#connection-state").textContent(), "connected");
    await waitConnected();
    await page.waitForFunction(() => document.querySelector("#messages").textContent.includes("本地协议测试成功"));
    await page.locator("#btn-mute").click();
    await page.waitForFunction(() => document.querySelector("#toast-region").textContent.includes("command_rejected"));
    assert.equal(await page.locator("#connection-state").textContent(), "connected");
    await page.locator("#btn-call").click();
    await waitStopped();
    await configure({startup_delay_s: 0, scenario: "disconnect"});
    await page.locator("#btn-call").click();
    await waitConnected();
    await page.locator("#btn-mute").click();
    await page.waitForFunction(() => document.querySelector("#connection-state").textContent === "disconnected");
    await page.waitForFunction(() => document.querySelector("#toast-region").textContent.includes("Live"));
    const count = (await metrics()).length;
    await page.waitForTimeout(700);
    assert.equal((await metrics()).length, count, "Live must not auto-reconnect");
    assert.deepEqual(errors, []);
    console.log("PASS: burst playback, AudioContext recovery, captions, mute, ten restarts, missing onended, startup cancellation, permission denial, slow startup, command rejection, disconnection, no history requests");
} catch (error) {
    console.error({errors, state: await page.locator("#connection-state").textContent(), notices: await page.locator("#toast-region").textContent()});
    throw error;
} finally { await browser.close(); }
