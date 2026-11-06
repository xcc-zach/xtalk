/** Exercise actual protocol handlers with deterministic transports and no soundcard. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');
const source = path.resolve(__dirname, '../src');
const modules = new Map();
function load(relative) {
    if (modules.has(relative)) return modules.get(relative);
    const filename = path.join(source, relative + '.ts');
    const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
        compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020},
    }).outputText;
    const module = {exports: {}};
    const requireSource = name => {
        if (name === './action-handler-functions/index') return {ACTION_TO_FUNCTION: {
            ...load('action-handler-functions/messages').default,
            ...load('action-handler-functions/client-operations').default,
            ...load('action-handler-functions/session').default,
            ...load('action-handler-functions/output').default,
            ...load('action-handler-functions/input').default,
            ...load('action-handler-functions/live').default,
        }};
        if (name === './utils') return {onVadSpeechStart() {}, onVadSpeechEnd() {}};
        if (name === '../encoding') return {createEncoding: () => ({decodeBase64: text => {
            const bytes = Buffer.from(text, 'base64');
            return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
        }})};
        throw new Error('Unexpected runtime dependency: ' + name);
    };
    vm.runInNewContext('(function(require,module,exports){' + code + '\n})', {console, TextDecoder, DataView, Uint8Array})(requireSource, module, module.exports);
    modules.set(relative, module.exports);
    return module.exports;
}
const {Conversation} = load('conversation');
const {ActionHandler} = load('action-handler');

test('Live uses Realtime negotiation, binary PCM and independent transcript fragments', async () => {
    const handler = new ActionHandler('binary_pcm');
    const conversation = new Conversation();
    conversation.state.mode = 'end_to_end';
    const sent = [], played = [];
    const socket = {sendJson: message => sent.push(message)};
    const output = {
        startAudioStream(id) { this.id = id; },
        async pushStreamAudioChunk(pcm) { played.push(Buffer.from(pcm)); },
        async endAudioStream() { this.id = null; },
    };
    const receive = event => handler.handleMessage(JSON.stringify(event), socket, conversation, output);
    await receive({type: 'session.created', session: {id: 'call', xtalk: {end_to_end: {protocol_version: 2}}}});
    assert.equal(sent[0].session.xtalk.end_to_end_protocol, 2);
    await receive({type: 'session.updated', session: {xtalk: {output_audio_encoding: 'binary_pcm'}}});
    await receive({type: 'xtalk.audio_stream.started', stream_id: 'stream'});
    await handler.handleMessage(envelope({version: 1, type: 'xtalk.audio_stream.delta', event_id: 'pcm', stream_id: 'stream'}), socket, conversation, output);
    assert.deepEqual(played[0], Buffer.from([1, 0, 2, 0]));
    const fragment = {type: 'xtalk.transcript.delta', stream_id: 'stream', arrival_sequence: 1,
        provider_event_id: 'caption', delta: ' exact text ', speaker: 'assistant', start_ms: 0, end_ms: 20};
    await receive(fragment);
    await receive(fragment);
    assert.equal(conversation.state.transcriptFragments.length, 1);
    assert.equal(conversation.state.transcriptFragments[0].delta, ' exact text ');
    assert.equal(conversation.state.messages.length, 0);
    await receive({type: 'xtalk.audio_stream.error', code: 'command_rejected', message: 'rejected', fatal: false});
    assert.equal(output.id, 'stream');
    assert.match(conversation.state.error, /command_rejected/);
    await receive({type: 'xtalk.audio_stream.ended'});
    assert.equal(output.id, null);
    assert.equal(conversation.state.transcriptFragments.length, 1);
});

function fixture() {
    const handler = new ActionHandler();
    const conversation = new Conversation();
    const sent = [];
    const socket = {sendJson: message => sent.push(message)};
    const output = {};
    const receive = event => handler.handleMessage(JSON.stringify(event), socket, conversation, output);
    return {handler, conversation, sent, socket, output, receive};
}

test('ASR revisions replace text, ignore stale previews, and stay scoped to each item', async () => {
    const {conversation, receive} = fixture();
    const preview = (id, delta, mode, revision) => receive({type: 'conversation.item.input_audio_transcription.delta', item_id: id, delta, xtalk: {mode, revision}});
    await preview('first', 'hel', 'append', 1);
    await preview('first', 'lo', 'append', 2);
    await preview('first', 'help', 'replace', 3);
    await preview('second', 'other', 'append', 1);
    await preview('first', 'stale', 'replace', 2);
    await receive({type: 'conversation.item.input_audio_transcription.completed', item_id: 'first', transcript: 'final'});
    await preview('first', 'late', 'replace', 4);
    assert.deepEqual(Array.from(conversation.state.messages, m => m.content), ['final', 'other']);
});

test('generated transcript and playback-confirmed transcript remain independent', async () => {
    const {conversation, receive} = fixture();
    await receive({type: 'response.output_audio_transcript.delta', response_id: 'r', delta: 'complete generated text'});
    assert.equal(conversation.state.generatedResponses.r.text, 'complete generated text');
    assert.equal(conversation.state.messages.length, 0);
    await receive({type: 'response.output_audio.playback.done', response_id: 'r', text: 'heard prefix'});
    assert.equal(conversation.state.messages[0].content, 'heard prefix');
    assert.equal(conversation.state.generatedResponses.r.text, 'complete generated text');
});

test('real-time text waits for its own accepted Item and rejects correlated errors', async () => {
    const {handler, conversation, sent, socket, output, receive} = fixture();
    let settled = false;
    const request = handler.submitText('hello', socket, conversation, output).then(() => {settled = true;});
    await Promise.resolve();
    const event = sent[0];
    assert.equal(event.xtalk.generate_response, true);
    await receive({type: 'conversation.item.done', item: {...event.item, id: 'unrelated'}});
    assert.equal(settled, false);
    await receive({type: 'conversation.item.done', item: event.item});
    await request;
    assert.equal(settled, true);
    const failed = handler.submitText('reject', socket, conversation, output);
    await receive({type: 'error', error: {event_id: sent[1].event_id, message: 'invalid text'}});
    await assert.rejects(failed, /invalid text/);
});

test('context-only injection omits generation and pending requests reject on close', async () => {
    const {handler, conversation, sent, socket, output, receive} = fixture();
    const history = handler.restoreItem({role: 'assistant', content: 'history'}, socket, conversation, output);
    assert.equal(sent[0].xtalk, undefined);
    await receive({type: 'conversation.item.done', item: sent[0].item});
    await history;
    const pending = handler.submitText('hello', socket, conversation, output);
    handler.cancelPending();
    await assert.rejects(pending, /connection closed/);
});


function envelope(header, pcm = Buffer.from([1, 0, 2, 0])) {
    const json = Buffer.from(JSON.stringify({version: 1, event_id: 'evt_binary', ...header}));
    const prefix = Buffer.alloc(4);
    prefix.writeUInt32BE(json.length);
    const bytes = Buffer.concat([prefix, json, pcm]);
    return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
}
const audioIdentity = {type: 'xtalk.output_audio.delta', response_id: 'r', item_id: 'item_r', output_index: 0, content_index: 0};

test('binary negotiation, raw playback, mixed audio and retired response filtering', async () => {
    const handler = new ActionHandler('binary_pcm');
    const conversation = new Conversation();
    const sent = [], played = [], mixed = [];
    conversation.onFullAudioChunk((pcm, rate) => mixed.push([Array.from(new Uint8Array(pcm)), rate]));
    const socket = {sendJson: message => sent.push(message), close() {throw new Error('Unexpected close');}};
    const output = {startTTS() {}, pushAudioChunk: async pcm => played.push(Array.from(new Uint8Array(pcm))), stop: async () => ({unconfirmedPlayedMs: 0})};
    const receive = event => handler.handleMessage(JSON.stringify(event), socket, conversation, output);
    await receive({type: 'session.created', session: {id: 's'}});
    assert.equal(sent[0].session.xtalk.output_audio_encoding, 'binary_pcm');
    await receive({type: 'session.updated', session: {xtalk: {output_audio_encoding: 'binary_pcm'}}});
    await receive({type: 'response.content_part.added', response_id: 'r', part: {type: 'audio'}});
    await handler.handleMessage(envelope(audioIdentity), socket, conversation, output);
    await handler.handleMessage(envelope({type: 'xtalk.audio.mix.delta', sample_rate: 48000, channels: 2, format: 'pcm_s16le'}), socket, conversation, output);
    assert.deepEqual(played, [[1, 0, 2, 0]]);
    assert.deepEqual(mixed, [[[1, 0, 2, 0], 48000]]);
    await receive({type: 'response.output_audio.playback.stop', response_id: 'r'});
    await handler.handleMessage(envelope(audioIdentity), socket, conversation, output);
    assert.equal(played.length, 1);
});

test('malformed or unnegotiated binary audio stops playback and closes once', async () => {
    const malformed = [
        envelope({...audioIdentity, version: 2}),
        envelope({...audioIdentity, type: 'unknown'}),
        envelope({...audioIdentity, item_id: ''}),
        envelope(audioIdentity, Buffer.from([1])),
        envelope({...audioIdentity, delta: 'forbidden'}),
        new ArrayBuffer(4),
        envelope({type: 'xtalk.audio.mix.delta', channels: 2, sample_rate: 48000, format: 'wav'}),
    ];
    for (const [index, raw] of [envelope(audioIdentity), ...malformed].entries()) {
        const handler = new ActionHandler();
        let stops = 0, closes = 0;
        const socket = {close() {closes++;}};
        const output = {stop: async () => {stops++;}};
        if (index > 0) await handler.handleMessage(JSON.stringify({type: 'session.updated', session: {xtalk: {output_audio_encoding: 'binary_pcm'}}}), socket, {}, output);
        await assert.rejects(handler.handleMessage(raw, socket, {}, output));
        await handler.handleMessage(raw, socket, {}, output);
        assert.equal(closes, 1);
        assert.equal(stops, 1);
    }
});
