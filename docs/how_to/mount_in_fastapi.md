# Mount in FastAPI

Mount the X-Talk Realtime API under a path prefix of an existing FastAPI application; this assumes a working `config.json` from [Configure the Service](../tutorial/config_the_service.md).

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

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Realtime API Protocol →](../reference/realtime_api.md)** · Reference

-   **[Service Configuration: model alias and API keys →](../reference/service_config.md)** · Reference

-   **[Start the Service →](../tutorial/start_the_service.md)** · Tutorial

</div>
