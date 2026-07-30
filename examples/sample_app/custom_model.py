import argparse
import asyncio
import logging
import mimetypes
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from xtalk import Xtalk, model
from xtalk.model_types import Agent
from xtalk.models.agents import AgentContext, AgentOutput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

parser = argparse.ArgumentParser(
    description="Configurable Xtalk Server with custom model"
)
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Xtalk Server")


@model
class EchoAgent(Agent):
    """A simple agent that echoes finalized ASR text."""

    def accept(self, context: AgentContext) -> Iterable[AgentOutput]:
        """Synchronously bridge ``async_accept()`` for compatibility."""

        yield from self._sync_iter_from_async(self.async_accept(context))

    async def async_accept(
        self,
        context: AgentContext,
    ) -> AsyncIterator[AgentOutput]:
        """Emit the finalized ASR text for ``asr_final`` contexts."""

        if str(context.get("type", "") or "") != "asr_final":
            return
        payload = context.get("data") or {}
        if not isinstance(payload, dict):
            return
        text = str(payload.get("text", ""))
        if text:
            yield text

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Ignore persisted history for the stateless echo agent."""

        del messages
        return None

    def clone(self) -> "EchoAgent":
        """Create a fresh stateless echo agent."""

        return EchoAgent()

    def _sync_iter_from_async(
        self,
        async_iter: AsyncIterator[AgentOutput],
    ) -> Iterable[AgentOutput]:
        """Convert an async iterator into a synchronous generator."""

        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    item = loop.run_until_complete(async_iter.__anext__())
                except StopAsyncIteration:
                    break
                yield item
        finally:
            aclose = getattr(async_iter, "aclose", None)
            if callable(aclose):
                try:
                    loop.run_until_complete(aclose())
                except Exception:
                    pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()


xtalk_instance = Xtalk.configure(args.config).set_model(EchoAgent).build()
xtalk_instance.mount_routes(app)


# Serve static files
example_server_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(example_server_path / "templates"))
static_root = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
try:
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except:
    print("No local X-Talk frontend library found. You may use the library from CDN.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/modern", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index_modern.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
