X-Talk keeps most models and execution on the server side. The client is mainly responsible for microphone access, audio streaming, WebSocket messaging, and lightweight session logic.

The client API is published as [xtalk-client](https://www.npmjs.com/package/xtalk-client), and its public interface is exported from [`frontend/src/index.ts`](https://github.com/xcc-zach/xtalk/blob/main/frontend/src/index.ts).

## Minimal client setup

`createSession()` takes a WebSocket URL. When you call `session.open()`, the client will:

1. call the login endpoint,
2. obtain an access token,
3. connect to the WebSocket endpoint with that token,
4. keep the current session state in sync.

If you use a bundler, install the package first:

```bash
npm install xtalk-client
```

> For developers, client code is under [`frontend`](https://github.com/xcc-zach/xtalk/tree/main/frontend). In that directory, you may run `npm run watch` for a dev build.

Then import it in your frontend code:

```ts
import { createSession } from "xtalk-client";

const wsUrl =
    location.protocol === "https:"
        ? `wss://${location.host}/ws`
        : `ws://${location.host}/ws`;

const session = createSession(wsUrl);

session.onStateChange((state) => {
    console.log("connection:", state.connectionState);
    console.log("session:", state.sessionId);
    console.log("messages:", state.messages);
});

await session.open();
```

If you do not want to bundle the client package yourself, you can load it directly from a CDN:

```html
<script type="module">
    const { createSession } = await import("https://unpkg.com/xtalk-client@0.2.6/dist/index.js");

    const wsUrl =
        location.protocol === "https:"
            ? `wss://${location.host}/ws`
            : `ws://${location.host}/ws`;

    const session = createSession(wsUrl);
    await session.open();
</script>
```

## Minimal server setup

For the public `xtalk-client`, the recommended server setup is to create an `Xtalk` instance from a config file and mount its built-in routes on a FastAPI app:

```python
from fastapi import FastAPI
from xtalk import Xtalk

app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config("path/to/config.json")
xtalk_instance.mount_routes(app)
```

For `config.json`, see [Config the Service](config_the_service.md).

This mounts the default endpoints used by the frontend:

- `POST /api/auth/login`
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `POST /api/upload`
- `WS /ws`

With those default routes, the frontend can simply call `createSession(wsUrl)` and then `session.open()`.

## Custom route setup

If your backend does not use the default built-in route paths, pass explicit service URLs when creating the session:

```ts
const session = createSession(wsUrl, {
    serviceURLs: {
        login: "/custom/login",
        sessions: "/custom/sessions",
        sessionDetail: (sessionId) => `/custom/sessions/${sessionId}`,
        upload: "/custom/upload",
    },
});
```

## Full example

For a fuller example including static files, templates, upload UI, session switching, and voice selection, see:

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)
