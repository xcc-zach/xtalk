from dataclasses import dataclass, field
from typing import Optional, Any
import argparse
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
from xtalk import (
    Xtalk,
    DefaultPipeline,
    DefaultService,
    create_event_class,
    Manager,
    EventBus,
    Pipeline,
)
from xtalk.events import *
from xtalk.model_types import *
from xtalk.serving.module_types import *
from xtalk.log_utils import mute_other_logging

mute_other_logging()
parser = argparse.ArgumentParser(description="Custom Xtalk Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()


# Define a custom model
class LLMOutputRefactorModel:
    def refactor(self, llm_output: str) -> str:
        # Custom logic to refactor LLM output
        return "Assistant response: " + llm_output

    # If custom model has internal state, implement clone method with concrete state
    def clone(self):
        return LLMOutputRefactorModel()


# Define a custom pipeline that includes the custom model
@dataclass(init=False)
class CustomPipeline(DefaultPipeline):
    llm_output_refactor_model: Optional["LLMOutputRefactorModel"] = field(
        default=None,
        metadata={"init_key": "llm_output_refactor_model", "clone": True},
    )

    def __init__(
        self,
        asr: ASR,
        llm_agent: Agent,
        tts: TTS,
        captioner: Optional[Captioner] = None,
        punt_restorer_model: Optional[PuntRestorer] = None,
        caption_rewriter: Optional[Rewriter | BaseChatModel] = None,
        thought_rewriter: Optional[Rewriter | BaseChatModel] = None,
        vad: Optional[VAD] = None,
        speech_enhancer: Optional[SpeechEnhancer] = None,
        speaker_encoder: Optional[SpeakerEncoder] = None,
        speech_speed_controller: Optional[SpeechSpeedController] = None,
        embeddings: Optional[Embeddings] = None,
        llm_output_refactor_model: Optional["LLMOutputRefactorModel"] = None,
        **kwargs,
    ):
        super().__init__(
            asr=asr,
            llm_agent=llm_agent,
            tts=tts,
            captioner=captioner,
            punt_restorer_model=punt_restorer_model,
            caption_rewriter=caption_rewriter,
            thought_rewriter=thought_rewriter,
            vad=vad,
            speech_enhancer=speech_enhancer,
            speaker_encoder=speaker_encoder,
            speech_speed_controller=speech_speed_controller,
            embeddings=embeddings,
            **kwargs,
        )
        self.llm_output_refactor_model = llm_output_refactor_model

    def get_llm_output_refactor_model(
        self,
    ) -> Optional["LLMOutputRefactorModel"]:
        return self.llm_output_refactor_model


# Instantiate the Pipeline
pipeline = Xtalk.create_pipeline_from_config(
    pipeline_cls=CustomPipeline,
    config_path_or_dict=args.config,
    additional_model_registry={
        "llm_output_refactor_model": LLMOutputRefactorModel(),
    },
)

# Define custom events and manager
LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal", fields={"text": "", "turn_id": 0}
)


class LLMOutputRefactorManager(Manager):
    # Signature of __init__ must match below
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any],
    ):
        self.event_bus = event_bus
        self.pipeline = pipeline

    @Manager.event_handler(LLMAgentResponseFinish)
    async def handle_llm_response_finish(self, event: LLMAgentResponseFinish):
        refactor_model = self.pipeline.get_llm_output_refactor_model()
        if refactor_model:
            refactored_output = refactor_model.refactor(event.text)
            new_event = LLMOutputRefactoredFinal(
                session_id=event.session_id,
                text=refactored_output,
                turn_id=event.turn_id,
            )
            await self.event_bus.publish(new_event)

    # If you have cleanup logic on service shutdown, put something concrete here
    async def shutdown(self):
        pass


# Create a Service and register the custom manager
custom_service = DefaultService(pipeline=pipeline, service_config={"sim_gen": False})
custom_service.register_manager(LLMOutputRefactorManager)

# Rewire event listeners of existing managers if needed
# Here we replace the OutputGateway's handler for LLMAgentResponseFinish
# to handle LLMOutputRefactoredFinal instead.
custom_service.unsubscribe_event(
    event_listener_cls=OutputGateway, event_type=LLMAgentResponseFinish
)


async def output_gateway_llm_output_refactored_final_handler(
    self: OutputGateway,
    event,
):
    await self.send_signal(
        {
            "action": "finish_resp",
            "data": {"text": event.text, "turn_id": event.turn_id},
        }
    )


custom_service.subscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMOutputRefactoredFinal,
    method_or_handler=output_gateway_llm_output_refactored_final_handler,
)

# Create Xtalk instance with the custom service and start the app
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, Form, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

xtalk_instance = Xtalk(service_prototype=custom_service, max_sessions=10)


app = FastAPI(title="Xtalk Server")


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


@app.post("/api/upload")
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    # Check file type
    content_type = (file.content_type or "").lower()
    filename = (file.filename or "").lower()
    is_text = content_type.startswith("text/") if content_type else False
    if content_type and not is_text:
        raise HTTPException(status_code=400, detail="Only text files are supported.")
    # Read file content and embed
    text = (await file.read()).decode("utf-8", errors="ignore")
    await xtalk_instance.embed_text(session_id=session_id, text=text)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
