# GPTLive single-call demo

This implementation uses latest `dev` and its `/v1/realtime` endpoint. The pure SDK owns realtime transport and media; the sample owns login. Live calls skip history storage and restoration.

## Start

Use the existing developer environment and build the browser library:

```bash
conda activate xtalk
python -m pip install -e ".[example]"
cd frontend
npm install
npm run build
cd ..
```

Copy the template to a local config:

```bash
cp examples/sample_app/configs/openai_gpt_live.example.json \
  examples/sample_app/configs/openai_gpt_live.json
```

Set `end_to_end.params.api_key` in the local config to your API key. GPTLive reads this field exclusively. The local config is Git-ignored; the `.example.json` template contains a placeholder. The key stays on the server.

```bash
PYTHONPATH=src python examples/sample_app/configurable_server.py \
  --port 7636 \
  --config examples/sample_app/configs/openai_gpt_live.json
```

Open <http://localhost:7636>, refresh after rebuilding, and press Start. Wait for the connecting state to finish before speaking. The same button cancels startup or stops the call. Mute keeps output playing; its preference survives reopening. Captions use the existing chat view, remain after stop, and clear at the next start or page reload. Errors also appear as Toast notifications in the orb view.

The sample uses `gpt-live-1`, `marin`, and explicit Responses delegation (`gpt-5.6-luna`, empty tools, `tool_choice: "none"`). Keep the intended backend configuration and verify its model access. Every call starts with empty context. Live history, automatic reconnect, output suppression, uploads, dynamic voice changes, recording, and custom tools are deferred.

## Media and architecture

Model `end_to_end.params.sample_rate` supports 16000 or 24000, default 24000, for both directions. Browser AEC remains enabled; additional VAD and enhancement are disabled. Muted capture sends equal-duration zero PCM from the input layer.

The adapter has one paced upstream writer and one independent reader. The Manager consumes provider events independently of browser playback, with a bounded pending PCM queue. Playback accounting lives inside `end_to_end_manager.py`, following the original TTS approach: 100 ms chunks, 300 ms in flight, and waiting when the window is full. `TTSManager` retains its original implementation. Default pending audio is 5 seconds, and outstanding audio without progress for 15 seconds produces a visible error. TTS and Live use one browser PCM scheduling core. Details: [protocol 2 specification](../technical_reference/end_to_end_live_spec.md).

Live defaults to persistence disabled. Enabling a deferred capability produces a startup error. Existing cascade configuration and stored database contents remain available.

## Local mock

Run two terminals:

```bash
python examples/sample_app/mock_live_server.py --port 8765
```

```bash
PYTHONPATH=src python examples/sample_app/configurable_server.py \
  --port 7636 \
  --config examples/sample_app/configs/mock_gpt_live.json
```

The fixture replies to the first nonzero input with captions and six consecutive 100 ms tones. Its loopback-only `/fixture` endpoint exposes test metrics and accepts startup delay/failure scenarios. Use it locally without a project key.

## Automated checks

```bash
PYTHONPATH=src python -m pytest tests/test_gpt_live.py tests/test_audio_delivery_failures.py
cd frontend
npx tsc --noEmit
npm test
npm run build
cd ..
```

With the two mock servers running and Playwright/Chromium installed in a test environment:

```bash
XTALK_DEMO_URL=http://127.0.0.1:7636 \
XTALK_FIXTURE_URL=http://127.0.0.1:8765 \
node tests/frontend/live_browser.mjs
```

Set `XTALK_PLAYWRIGHT_MODULE` to the installed Playwright ESM entrypoint when it is outside Node's normal module search path. The browser test uses synthetic capture, and covers burst output, AudioContext suspension/recovery, missing `onended`, captions, mute, ten fresh calls, permission denial, startup cancellation, slow upstream startup, command rejection, disconnection, cleanup, and absence of history requests. Backend tests cover queue bounds, stall cancellation, 16/24 kHz media, finalization, protocol validation and database preservation.

## Latest-dev adaptation validation (2026-10-31)

- Based on `dev` at `461df8f`, carrying the working Live version at `ca6b7c9`. The backend retains the `gpt-live-1` adapter; both sides use the new Realtime session, ordered transport queue, and event extension APIs.
- Backend: 269 tests and 24 subtests pass. All 10 frontend protocol/lifecycle tests, type checking, production build, and mock browser workflow pass. Coverage includes Base64/binary envelopes, 16/24 kHz, no history writes, burst playback, mute, ten restarts, and stale callback isolation.
- Two existing speaker-adapter cancellation tests fail in the full Python run. Both reproduce with unmodified `dev` sources: their `Task.cancelling()` calls are unavailable in this Python 3.10 environment. They remain outside this change.
- Real upstream validation passes after correcting the local credential: the config value contained 19 extra characters compared with the user's successful GPT test. The adapter receives speech, transcripts and output audio, acknowledges mute/unmute, and closes completely. Two fresh browser calls with synthetic speech pass question answering, playback progress, zero PCM while muted, preserved mute preference, restart and media cleanup; no provider/page errors or history requests occur.

Automated browser tests use synthetic microphone input. Human listening, natural backchannels, room echo, and native interruption quality remain manual acceptance items. Check ordinary questions, short acknowledgements, speaking over the assistant, mute, stop, and restart. Live decides interruptions; the client does not promise word-level rollback.
