import argparse
import json
import logging
import mimetypes
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from xtalk import Xtalk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")

FRONTEND_UTILITIES_ROUTE = "/xtalk/frontend-utilities"
FRONTEND_UTILITIES_DIR = Path(__file__).parent / "dist"
DOWNLOAD_TIMEOUT_SECONDS = 60
ORT_VERSIONS = ("1.22.0", "1.17.0")
VAD_VERSION = "0.0.27"
FASTENHANCER_URL = (
    "https://github.com/aask1357/fastenhancer/releases/download/"
    "onnx-48khz-v1/fastenhancer_s.onnx"
)


def download_frontend_utilities() -> None:
    """Download browser-side runtime and model assets for the sample app.

    The assets are stored under ``examples/sample_app/dist`` and are mounted by
    this server at ``/xtalk/frontend-utilities``.

    Raises
    ------
    RuntimeError
        Raised when any missing asset cannot be downloaded.
    """

    def download_file(url: str, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_name(f"{target.name}.tmp")
        try:
            with urllib.request.urlopen(
                url, timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                with tmp_target.open("wb") as file:
                    shutil.copyfileobj(response, file)
            tmp_target.replace(target)
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as exc:
            if tmp_target.exists():
                tmp_target.unlink()
            raise RuntimeError(
                "Failed to download frontend utilities. "
                "Please download the missing file manually to "
                f"{target}, or run this server once on a machine with internet "
                f"access. Source URL: {url}. Error: {exc}"
            ) from exc

    def extract_npm_dist_files(
        *,
        package_name: str,
        tarball_url: str,
        target_dir: Path,
        required_names: set[str],
        include_wasm: bool = False,
        include_mjs: bool = False,
        require_mjs: bool = False,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        has_required_files = all((target_dir / name).exists() for name in required_names)
        has_wasm_files = any(target_dir.glob("*.wasm"))
        has_mjs_files = any(target_dir.glob("*.mjs"))
        if has_required_files:
            if (not include_wasm or has_wasm_files) and (not include_mjs or has_mjs_files):
                return

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "package.tgz"
            download_file(tarball_url, archive_path)
            try:
                with tarfile.open(archive_path, "r:gz") as archive:
                    for member in archive.getmembers():
                        member_path = Path(member.name)
                        if (
                            not member.isfile()
                            or len(member_path.parts) < 3
                            or member_path.parts[0] != "package"
                            or member_path.parts[1] != "dist"
                        ):
                            continue
                        file_name = member_path.name
                        should_extract_wasm = include_wasm and file_name.endswith(".wasm")
                        should_extract_mjs = include_mjs and file_name.endswith(".mjs")
                        if (
                            file_name not in required_names
                            and not should_extract_wasm
                            and not should_extract_mjs
                        ):
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            continue
                        with source:
                            with (target_dir / file_name).open("wb") as target:
                                shutil.copyfileobj(source, target)
            except (OSError, tarfile.TarError) as exc:
                raise RuntimeError(
                    "Failed to extract frontend utilities. "
                    "Please download the missing files manually, or run this "
                    "server once on a machine with internet access. "
                    f"Package: {package_name}. Error: {exc}"
                ) from exc

        missing = [name for name in required_names if not (target_dir / name).exists()]
        if include_wasm and not any(target_dir.glob("*.wasm")):
            missing.append("*.wasm")
        if require_mjs and not any(target_dir.glob("*.mjs")):
            missing.append("*.mjs")
        if missing:
            raise RuntimeError(
                "Failed to prepare frontend utilities. "
                "Please download the missing files manually, or run this server "
                "once on a machine with internet access. "
                f"Package: {package_name}. Missing files: {', '.join(missing)}"
            )

    for version in ORT_VERSIONS:
        extract_npm_dist_files(
            package_name=f"onnxruntime-web@{version}",
            tarball_url=(
                "https://registry.npmjs.org/onnxruntime-web/-/"
                f"onnxruntime-web-{version}.tgz"
            ),
            target_dir=(
                FRONTEND_UTILITIES_DIR
                / "onnxruntime-web"
                / version
                / "dist"
            ),
            required_names={"ort.js"},
            include_wasm=True,
            include_mjs=version == "1.22.0",
            require_mjs=version == "1.22.0",
        )

    extract_npm_dist_files(
        package_name=f"@ricky0123/vad-web@{VAD_VERSION}",
        tarball_url=(
            "https://registry.npmjs.org/@ricky0123/vad-web/-/"
            f"vad-web-{VAD_VERSION}.tgz"
        ),
        target_dir=FRONTEND_UTILITIES_DIR / "vad-web" / VAD_VERSION / "dist",
        required_names={"bundle.min.js", "silero_vad_v5.onnx"},
    )

    fastenhancer_target = (
        FRONTEND_UTILITIES_DIR / "xtalk" / "models" / "fastenhancer_s.onnx"
    )
    if not fastenhancer_target.exists():
        download_file(FASTENHANCER_URL, fastenhancer_target)


parser = argparse.ArgumentParser(description="Xtalk Dev Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Xtalk Dev Server")

xtalk_instance = Xtalk.from_config(args.config)
xtalk_instance.mount_routes(app)


logs_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(logs_path / "templates"))
app.mount("/static", StaticFiles(directory=str(logs_path / "static")), name="static")
download_frontend_utilities()
app.mount(
    FRONTEND_UTILITIES_ROUTE,
    StaticFiles(directory=str(FRONTEND_UTILITIES_DIR)),
    name="frontend_utilities",
)
try:
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except Exception:
    print("No local Xtalk frontend library found. Will load frontend library from CDN.")


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
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
