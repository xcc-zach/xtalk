/** Exercise the actual SDK and action handlers through platform abstractions. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

function fixture(profile = null) {
    const sockets = [], inputs = [], outputs = [], operations = [], modules = new Map();
    const stubs = {
        encoding: {createEncoding: () => ({encodePCM: () => 'AQACAA==', decodeBase64: text => Uint8Array.from(Buffer.from(text, 'base64')).buffer})},
        'task-scheduler': {createDeferredTaskScheduler: () => ({schedule: fn => fn(), dispose() {}})},
        websocket: {createWebSocket: (url, protocols) => {
            const listeners = new Map();
            const socket = {url, protocols, sent: [], alive: true, bufferedAmount: 0,
                addEventListener(name, fn) { if (!listeners.has(name)) listeners.set(name, []); listeners.get(name).push(fn); },
                emit(name, data) { for (const fn of listeners.get(name) ?? []) fn(data); },
                message(data) { this.emit('message', {data: JSON.stringify(data)}); },
                ready() { return this.alive; }, sendAudioChunk(pcm) { this.sent.push(pcm); },
                sendJson(message) {
                    this.sent.push(message); operations.push(message.type);
                    queueMicrotask(() => {
                        if (message.type === 'session.update') this.message({type: 'session.updated', session: {xtalk: message.session.xtalk}});
                        if (message.type === 'conversation.item.create') this.message({type: 'conversation.item.done', item: message.item});
                        if (message.type === 'xtalk.audio_stream.ready') this.message({type: 'xtalk.audio_stream.started', stream_id: 'stream'});
                        if (message.type === 'xtalk.audio_stream.close') this.message({type: 'xtalk.audio_stream.ended', outcome: 'completed'});
                    });
                },
                close() { if (this.alive) { this.alive = false; this.emit('close', {code: 1000}); } },
            };
            sockets.push(socket);
            queueMicrotask(() => { socket.emit('open', {}); socket.message({type: 'session.created', session: {id: 'call', xtalk: profile ? {end_to_end: profile} : {}}}); });
            return socket;
        }},
        'audio-session': {
            createInputAudioSession(config) {
                const input = {config, muted: false, closed: false,
                    onFrame(fn) { this.frame = fn; }, onSpeechStart(fn) { this.speechStart = fn; }, onSpeechEnd(fn) { this.speechEnd = fn; },
                    async open() { operations.push('input.open'); }, async close() { this.closed = true; },
                }; inputs.push(input); return input;
            },
            createOutputAudioSession(config) {
                const output = {config, supportsContinuousAudio: true, closed: false, played: [],
                    onAudioStreamProgress(fn) { this.progress = fn; }, onChunkStarted(fn) { this.started = fn; },
                    onChunkPlayed(fn) { this.chunk = fn; }, onAllChunksPlayed(fn) { this.finished = fn; },
                    async open() { operations.push('output.open'); }, async close() { this.closed = true; },
                    startTTS(id) { this.response = id; }, async pushAudioChunk(pcm) { this.played.push(pcm); },
                    startAudioStream(id) { this.stream = id; }, async pushStreamAudioChunk(pcm) { this.played.push(pcm); },
                    async endAudioStream() { this.stream = null; },
                }; outputs.push(output); return output;
            },
        },
    };
    function load(name) {
        if (stubs[name]) return stubs[name];
        if (modules.has(name)) return modules.get(name);
        const filename = path.resolve(__dirname, '../src', name + '.ts');
        const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020}}).outputText;
        const module = {exports: {}}; modules.set(name, module.exports);
        vm.runInNewContext('(function(require,module,exports){' + code + '\n})', {console, setTimeout, clearTimeout, Date, TextDecoder, DataView, Uint8Array})(
            dependency => load(path.posix.normalize(path.posix.join(path.posix.dirname(name), dependency))), module, module.exports);
        return module.exports;
    }
    return {createSession: load('session/create').createSession, sockets, inputs, outputs, operations,
        tick: () => new Promise(resolve => setImmediate(resolve))};
}

test('cascade retains history before media, input, TTS receipts, mute and stale callback isolation', async () => {
    const f = fixture();
    const session = f.createSession('wss://local/v1/realtime', {apiKey: 'host-key'});
    await session.restoreHistory([{role: 'user', content: 'earlier'}]); await session.open();
    assert.equal(session.state.connectionState, 'connected');
    assert.equal(f.sockets[0].protocols[1], 'openai-insecure-api-key.host-key');
    assert(f.operations.indexOf('conversation.item.create') < f.operations.indexOf('input.open'));
    await f.inputs[0].frame(new ArrayBuffer(4)); await f.inputs[0].speechStart(); await f.inputs[0].speechEnd();
    assert(f.sockets[0].sent.some(e => e.type === 'input_audio_buffer.append'));
    assert(f.sockets[0].sent.some(e => e.type === 'input_audio_buffer.commit'));
    f.sockets[0].message({type: 'response.content_part.added', response_id: 'r', part: {type: 'audio'}});
    f.sockets[0].message({type: 'response.output_audio.delta', response_id: 'r', delta: 'AQACAA=='});
    await f.tick(); assert.equal(f.outputs[0].response, 'r'); assert.equal(f.outputs[0].played.length, 1);
    await f.outputs[0].chunk('r'); await f.outputs[0].finished('r');
    assert(f.sockets[0].sent.some(e => e.type === 'response.output_audio.playback.chunk_played'));
    assert(f.sockets[0].sent.some(e => e.type === 'response.output_audio.playback.finished'));
    session.muted = true; await session.close(); await session.close();
    assert(f.inputs[0].closed && f.outputs[0].closed);
    await session.open(); assert.equal(f.inputs[1].muted, true);
    f.sockets[0].message({type: 'error', error: {message: 'late old error'}}); f.sockets[0].emit('close', {code: 1006});
    await f.tick(); assert.equal(session.state.connectionState, 'connected'); await session.close();
});

test('Live negotiates media, disables extra VAD and sends raw PCM with cumulative receipts', async () => {
    const profile = {protocol_version: 2, stream_id: 'stream', media: {sample_rate: 16000, channels: 1, encoding: 'pcm_s16le'}, capabilities: {input_mute: true}};
    const f = fixture(profile), session = f.createSession('wss://local/v1/realtime', {mode: 'end_to_end'});
    await session.open(); assert.equal(session.state.connectionState, 'connected');
    assert.equal(f.inputs.length, 1); assert.equal(f.outputs.length, 1);
    assert.equal(f.inputs[0].config.sampleRate, 16000); assert.equal(f.outputs[0].config.sampleRate, 16000);
    assert.equal(f.inputs[0].config.enableVAD, false); assert.equal(f.inputs[0].config.preserveSilenceWhenMuted, true);
    const pcm = new ArrayBuffer(4); await f.inputs[0].frame(pcm); assert(f.sockets[0].sent.includes(pcm));
    f.outputs[0].progress({stream_id: 'stream', played_samples: 1600});
    assert(f.sockets[0].sent.some(e => e.type === 'xtalk.audio_stream.progress' && e.played_samples === 1600));
    await assert.rejects(session.restoreHistory([]), /unavailable/); await assert.rejects(session.sendText('hello'), /unavailable/);
    session.muted = true; assert(f.sockets[0].sent.some(e => e.type === 'xtalk.audio_stream.input_muted' && e.muted));
    await session.close(); assert(f.inputs[0].closed && f.outputs[0].closed);
});
