import argparse
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
from xtalk import Xtalk
from xtalk.log_utils import mute_other_logging

mute_other_logging()

parser = argparse.ArgumentParser(
    description="Configurable Xtalk Server with custom model"
)
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Xtalk Server")

# Register an Agent implemented in an external Python file
Xtalk.register_model_search_spec(
    slot="llm_agent",
    spec=Path(__file__).parent / "echo_agent.py",
)
# Instantiate Xtalk from config
# config can be passed as a path to json file or a dict
xtalk_instance = Xtalk.from_config(args.config)


# Mount WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)


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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/modern", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index_modern.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
