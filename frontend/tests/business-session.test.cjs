/** Verify the example owns credentials/history while composing the pure SDK. */
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const ts = require('typescript');

test('sample application authenticates, restores before opening, and reconnects to its stable business ID', async () => {
    const storage = new Map();
    const calls = [];
    const history = [{role: 'user', content: 'earlier turn'}];
    const localStorage = {getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key)};
    const context = {
        URL, Date, console, localStorage,
        window: {location: {href: 'https://example.test/'}},
        atob: value => Buffer.from(value, 'base64').toString(),
        fetch: async (url, options) => {
            calls.push([url.pathname, options]);
            const result = url.pathname === '/api/auth/login'
                ? {access_token: 'header.' + Buffer.from(JSON.stringify({exp: Math.floor(Date.now() / 1000) + 3600})).toString('base64url') + '.signature', user: {id: 'owner'}}
                : url.pathname === '/api/sessions' ? {session_id: 'business'} : {messages: history};
            return {ok: true, json: async () => result};
        },
    };
    const modules = new Map();
    function load(name) {
        if (modules.has(name)) return modules.get(name);
        const file = path.resolve(__dirname, '../../examples/sample_app/static/js', name + '.js');
        const code = ts.transpileModule(fs.readFileSync(file, 'utf8'), {compilerOptions: {allowJs: true, module: ts.ModuleKind.CommonJS}}).outputText;
        const module = {exports: {}};
        vm.runInNewContext('(function(require,module,exports){' + code + '\n})', context)(name => load(name.replace('./', '').replace('.js', '')), module, module.exports);
        modules.set(name, module.exports);
        return module.exports;
    }
    const {createBusinessSession} = load('conversation-store');
    const sessions = [];
    const createSession = (url, config) => {
        const operations = [];
        const session = {
            muted: false,
            onStateChange(callback) { this.notify = callback; },
            async restoreHistory(messages) { operations.push(['restore', messages]); },
            async open() { operations.push(['open']); this.notify({sessionId: 'temporary-' + sessions.length, messages: history, connectionState: 'connected'}); },
            async close() { operations.push(['close']); },
        };
        sessions.push({session, url, config, operations});
        return session;
    };
    const config = {inputConfig: {}};
    const session = createBusinessSession(createSession, 'wss://example.test/v1/realtime?model=xtalk', config);
    await session.open();
    assert.equal(session.state.sessionId, 'business');
    assert.equal(sessions[0].url.searchParams.get('conversation_id'), 'business');
    assert.ok(sessions[0].url.searchParams.get('access_token'));
    assert.equal(sessions[0].url.searchParams.get('model'), 'xtalk');
    assert.equal(sessions[0].config, config);
    assert.deepEqual(sessions[0].operations.map(op => op[0]), ['restore', 'open']);
    await session.close();
    await session.open();
    assert.equal(calls.filter(([p]) => p === '/api/auth/login').length, 1);
    assert.equal(calls.filter(([p]) => p === '/api/sessions').length, 1);
    assert.equal(calls.filter(([p]) => p === '/api/sessions/business').length, 2);
    assert.equal(session.state.sessionId, 'business');
    assert.equal(JSON.parse(storage.get('xtalk.sample.conversation')).sessionId, 'business');
});
