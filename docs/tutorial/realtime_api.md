# Realtime API

Xtalk exposes an OpenAI Realtime style API with a standard subset and Xtalk extensions.

## Run an ASGI application

Choose and install your ASGI server, for example `pip install uvicorn`:

```python
import uvicorn
from xtalk import Xtalk

runtime = Xtalk.from_config("path/to/config.json")
uvicorn.run(runtime.create_asgi_app(), host="127.0.0.1", port=8000)
```

Set the API model alias and API keys in the configuration file:

```json
{"service_config": {"realtime_api": {"model": "xtalk", "api_keys": []}}}
```

The defaults are model `xtalk` and API key authentication disabled. Configure `api_keys` to require `Authorization: Bearer <key>`. Browsers may use the `openai-insecure-api-key.<key>` WebSocket subprotocol.

The public endpoints are `GET /v1/models` and `WS /v1/realtime?model=xtalk`.

## Mount in FastAPI

The host owns framework installation and lifespan composition. Mounted applications require startup and shutdown forwarding; install `fastapi`, `uvicorn` and `asgi-lifespan` for this example.

```python
from contextlib import asynccontextmanager
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from xtalk import Xtalk

realtime_app = Xtalk.from_config("config.json").create_asgi_app()

@asynccontextmanager
async def lifespan(app):
    async with LifespanManager(realtime_app):
        yield

app = FastAPI(lifespan=lifespan)
app.mount("/speech", realtime_app)
```

Connect to `/speech/v1/realtime?model=xtalk`. Each WebSocket owns a temporary conversation; disconnect releases its resources. Login, stored history, uploads and recording files belong to the host application. [`examples/sample_app/host`](https://github.com/xcc-zach/xtalk/tree/main/examples/sample_app/host) illustrates that composition.

## Minimal audio client

```sh
pip install websockets numpy soxr soundfile sounddevice
python examples/sample_client.py --base-url http://127.0.0.1:8000 --model xtalk
```

For a soundcard-free run, add `--input-wav input.wav --no-speaker --rounds 2`. Add `--api-key <key>` when required. The sample handles WebSocket communication and audio input/output; VAD and enhancement run on the server.

The sample keeps one output stream open, fills consecutive protocol blocks continuously, and acknowledges each block after its final sample reaches the output device. Interruptions flush the stopped response and report only unconfirmed played audio.

## Exchange events

For event directions, fields, audio formats, playback acknowledgements, and compatibility, see the [Realtime API protocol reference](../technical_reference/realtime_api.md).
