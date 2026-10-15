X-Talk keeps most models and execution on the server side. The client is mainly responsible for microphone access, audio streaming, WebSocket messaging, and lightweight session logic.

## Client setup

### npm

When using a bundler, install the client through npm:

```bash
npm install xtalk-client
```

Create `index.html`:

```html
<!doctype html>
<html lang="en">
  <body>
    <button id="start">Start conversation</button>

    <script type="module">
      import { createSession } from "xtalk-client";

      const wsUrl =
        location.protocol === "https:"
          ? `wss://${location.host}/v1/realtime?model=xtalk`
          : `ws://${location.host}/v1/realtime?model=xtalk`;

      const session = createSession(wsUrl);

      document.querySelector("#start").addEventListener("click", async () => {
        await session.open();
      });
    </script>
  </body>
</html>
```

### CDN

Using a CDN, create `index.html`:

```html
<!doctype html>
<html lang="en">
  <body>
    <button id="start">Start conversation</button>

    <script type="module">
      const { createSession } = await import(
        "https://unpkg.com/xtalk-client@latest/dist/index.js"
      );

      const wsUrl =
        location.protocol === "https:"
          ? `wss://${location.host}/v1/realtime?model=xtalk`
          : `ws://${location.host}/v1/realtime?model=xtalk`;

      const session = createSession(wsUrl);

      document.querySelector("#start").addEventListener("click", async () => {
        await session.open();
      });
    </script>
  </body>
</html>
```

## Server setup

Follow the [Quick Start](../quickstart.md) to install X-Talk and create a JSON configuration file.

Create `server.py`:

```python
from pathlib import Path

from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from fastapi.responses import FileResponse
from xtalk import Xtalk

BASE_DIR = Path(__file__).resolve().parent
xtalk_instance = Xtalk.from_config(str(BASE_DIR / "config.json"))
realtime_app = xtalk_instance.create_asgi_app()

@asynccontextmanager
async def lifespan(app):
    async with LifespanManager(realtime_app):
        yield

app = FastAPI(title="Xtalk Server", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Return the frontend page."""
    return FileResponse(BASE_DIR / "index.html")

app.mount("/", realtime_app)
```

Place `server.py`, `config.json`, and `index.html` in the same directory.

Start the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 11995
```

After startup, open `http://localhost:11995`.

After clicking **Start conversation** for the first time, the browser needs to load ONNX Runtime and VAD from public CDNs. Wait for the browser's microphone permission prompt; the conversation starts successfully after the prompt appears and permission is granted.

## Full example

For a fuller example including static files, templates, upload UI, session switching, and voice selection, see:

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/css/index.css`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/css/index.css)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)


See [Realtime API](realtime_api.md) for lifecycle composition and host-owned authentication, knowledge indexing and recording storage. Install FastAPI and asgi-lifespan when using the framework example.
