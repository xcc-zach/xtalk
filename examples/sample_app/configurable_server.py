import argparse
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import json
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")

from xtalk import Xtalk
from xtalk.log_utils import mute_other_logging

mute_other_logging()

parser = argparse.ArgumentParser(description="Xtalk Dev Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
parser.add_argument("--ssl-keyfile", type=str, help="Path to the SSL key file for HTTPS")
parser.add_argument("--ssl-certfile", type=str, help="Path to the SSL certificate file for HTTPS")
args = parser.parse_args()

app = FastAPI(title="Xtalk Dev Server")

xtalk_instance = Xtalk.from_config(args.config)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)


logs_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(logs_path / "templates"))
app.mount("/static", StaticFiles(directory=str(logs_path / "static")), name="static")
try:
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except Exception:
    print("No local Xtalk frontend library found.")


@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    content_type = (file.content_type or "").lower()
    is_text = content_type.startswith("text/") if content_type else False
    if content_type and not is_text:
        raise HTTPException(status_code=400, detail="Only text files are supported.")
    text = (await file.read()).decode("utf-8", errors="ignore")
    await xtalk_instance.embed_text(session_id=session_id, text=text)
    return {"status": "ok"}


@app.get("/api/voices")
async def get_reference_audios():
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
        try:
            voices = config["tts"]["params"]["voices"]
        except:
            voices = []
    return JSONResponse(content={"audios": voices})


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":# python configurable_server.py --port 7635 --config config.json --ssl-keyfile key.pem --ssl-certfile cert.pem
    import uvicorn

    uvicorn_kwargs = {
        "host": "0.0.0.0",
        "port": args.port or 11995,
    }
    if args.ssl_keyfile and args.ssl_certfile:
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = args.ssl_certfile

    uvicorn.run(app, **uvicorn_kwargs)
