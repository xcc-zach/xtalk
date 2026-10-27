# Start the Service

This tutorial ends with a browser page that holds a spoken conversation with X-Talk. You will write a one-button client page and a `server.py` that serves it together with the X-Talk Realtime API.

Before starting, complete the [Quickstart](../quickstart.md) so that X-Talk is installed and a `config.json` with working ASR, LLM and TTS models exists.

X-Talk keeps most models and execution on the server side. The client is responsible for microphone access, audio streaming, WebSocket messaging, and lightweight session logic.

## Client setup

Pick one of the two ways to load the client library. Both produce the same `index.html`: a page with a **Start conversation** button that opens a session to `/v1/realtime?model=xtalk` on the current host.

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

`createSession(wsUrl)` uses default input and output settings. The full option set is listed in [Client Session Config](../reference/client_session_config.md).

## Server setup

The server has two jobs: return `index.html` at `/`, and expose the X-Talk Realtime API. `Xtalk.from_config()` loads the models from `config.json`, and `create_asgi_app()` turns the instance into an ASGI application that FastAPI mounts. `LifespanManager` forwards FastAPI's startup and shutdown to that mounted application so models are loaded and released with the host.

Install the host dependencies:

```bash
pip install fastapi uvicorn asgi-lifespan
```

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

Uvicorn prints `Uvicorn running on http://0.0.0.0:11995` once the models are loaded. Open `http://localhost:11995` in the browser; the page shows the **Start conversation** button.

Click **Start conversation**. On the first click the browser loads ONNX Runtime and the VAD model from public CDNs, so a few seconds pass before anything happens. Then the browser asks for microphone permission. Grant it; the conversation is running once the prompt is gone. Say something and the reply plays back through the speakers.

## Full example

For a fuller example including static files, templates, upload UI, session switching, and voice selection, see:

- [`examples/sample_app/configurable_server.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/configurable_server.py)
- [`examples/sample_app/static/css/index.css`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/css/index.css)
- [`examples/sample_app/static/js/index.js`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/static/js/index.js)
- [`examples/sample_app/templates/index.html`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/templates/index.html)

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Configure the Service →](config_the_service.md)** · Tutorial

-   **[Mount in FastAPI under a path prefix →](../how_to/mount_in_fastapi.md)** · How-to

-   **[Client Session Config →](../reference/client_session_config.md)** · Reference

</div>
