from typing import Any
import argparse
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
from xtalk import (
    Xtalk,
    DefaultService,
    create_event_class,
    Manager,
    EventBus,
    Models,
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


# Instantiate the model container
models = Xtalk.create_models_from_config(config_path_or_dict=args.config)
models.set(LLMOutputRefactorModel, LLMOutputRefactorModel())

# Define custom events and manager
LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal", fields={"text": ""}
)


class LLMOutputRefactorManager(Manager):
    # Signature of __init__ must match below
    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any],
    ):
        self.event_bus = event_bus
        self.models = models

    @Manager.event_handler(LLMAgentResponseFinish)
    async def handle_llm_response_finish(self, event: LLMAgentResponseFinish):
        refactor_model = self.models.get(LLMOutputRefactorModel)
        if refactor_model:
            refactored_output = refactor_model.refactor(event.text)
            new_event = LLMOutputRefactoredFinal(
                session_id=event.session_id,
                text=refactored_output,
            )
            await self.event_bus.publish(new_event)

    # If you have cleanup logic on service shutdown, put something concrete here
    async def shutdown(self):
        pass


# Create a Service and register the custom manager
custom_service = DefaultService(models=models)
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
            "data": {"text": event.text},
        }
    )


custom_service.subscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMOutputRefactoredFinal,
    method_or_handler=output_gateway_llm_output_refactored_final_handler,
)

# Create Xtalk instance with the custom service and start the app
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse

xtalk_instance = Xtalk(service_prototype=custom_service, max_sessions=10)


app = FastAPI(title="Xtalk Server")
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
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
