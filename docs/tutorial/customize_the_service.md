*Experimental API*

> **Note**
> See [`examples/sample_app/custom_service.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_service.py) for the complete example. It adds a dummy `LLMOutputRefactorModel` to X-Talk and prepends `Assistant response: ` before the final LLM response text is sent to the frontend.

You may want to introduce a service-local custom model and connect its logic to the rest of the service flow.

## Introduce a New Model Type

There are two common ways to introduce a new model:

1. Define the model directly in service code and manually put it into `Models`. [`examples/sample_app/custom_service.py`](https://github.com/xcc-zach/xtalk/blob/main/examples/sample_app/custom_service.py) uses this approach.
2. Define a new model type first, then register a concrete model implementation so the model can be created from config.

### Approach 1: Introduce a Model Directly

This approach fits service-local helper models: the model only serves the current custom service and does not need to be exposed as a general model type in config.

#### Define the Custom Model

The example `LLMOutputRefactorModel` prepends a prefix to the final LLM response:

```python
class LLMOutputRefactorModel:
    def refactor(self, llm_output: str) -> str:
        return "Assistant response: " + llm_output

    def clone(self):
        return LLMOutputRefactorModel()
```

If the model has per-session state, implement `clone` so each session gets an isolated model instance.

#### Create Models and Inject the Custom Model

Built-in models are still created from the config file:

```python
models = Xtalk.create_models_from_config(config_path_or_dict=args.config)
```

Then put the custom model into the same `Models` container:

```python
models.set(LLMOutputRefactorModel, LLMOutputRefactorModel())
```

The model class itself is used as the key. Later, the `Manager` uses the same `LLMOutputRefactorModel` class to retrieve the model.

### Approach 2: Introduce a New Model Type, Then a New Model

If you want the model to be created from config, first define a model type interface and register it with `@model_type`; then define the concrete model implementation and register it with `@model`.

The modules defining the model type and model implementation must be imported before `Xtalk.create_models_from_config(...)` is called. If they are written in the same service file, define them before creating `Models`.

```python
from abc import ABC, abstractmethod

from xtalk import model, model_type


@model_type(aliases=["llm_output_refactor_model"])
class LLMOutputRefactor(ABC):
    @abstractmethod
    def refactor(self, llm_output: str) -> str:
        pass

    @abstractmethod
    def clone(self) -> "LLMOutputRefactor":
        pass


@model
class PrefixLLMOutputRefactor(LLMOutputRefactor):
    def __init__(self, prefix: str = "Assistant response: "):
        self.prefix = prefix

    def refactor(self, llm_output: str) -> str:
        return self.prefix + llm_output

    def clone(self):
        return PrefixLLMOutputRefactor(prefix=self.prefix)
```

`clone()` is not required by `@model_type` registration itself, but when the service clones `Models` for each session, it automatically calls `clone()` on model instances. If implementations of this model type should not share state across sessions, put `clone()` in the interface contract.

Then add the model to the config file:

```json
{
  "llm_output_refactor_model": {
    "type": "PrefixLLMOutputRefactor",
    "params": {
      "prefix": "Assistant response: "
    }
  }
}
```

When using this approach, you no longer need to manually `set` this model when creating `Models`:

```python
models = Xtalk.create_models_from_config(config_path_or_dict=args.config)
```

The later `Manager` should retrieve the model through the model type interface:

```python
refactor_model = self.models.get(LLMOutputRefactor)
```

If you only want to temporarily override the instance created from config, continue using the same model type interface as the key:

```python
models.set(LLMOutputRefactor, PrefixLLMOutputRefactor())
```

## Create a Custom Event

The refactored text needs to be passed to the frontend output component, so the example creates a new event type:

```python
LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal",
    fields={"text": ""},
)
```

This event only carries a `text` field for the refactored final response.

## Define a Custom Manager

`LLMOutputRefactorManager` listens for `LLMAgentResponseFinish`, retrieves the custom model, refactors the text, and publishes `LLMOutputRefactoredFinal`.

The code below continues using `LLMOutputRefactorModel` from approach 1. If you use approach 2, replace `self.models.get(LLMOutputRefactorModel)` with `self.models.get(LLMOutputRefactor)`.

```python
class LLMOutputRefactorManager(Manager):
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

    async def shutdown(self):
        pass
```

The service passes `models` to the manager when creating a session. The example uses `models.get(...)`, so it skips refactoring if the model is absent. If your custom logic requires the model, require it directly so configuration errors surface early.

## Register the Manager

Create the service with `models`, then register the custom `Manager`:

```python
custom_service = DefaultService(models=models)
custom_service.register_manager(LLMOutputRefactorManager)
```

Each session will create an `LLMOutputRefactorManager` and share the model container managed by the service.

## Adjust Existing Event Subscriptions

By default, `OutputGateway` handles the original `LLMAgentResponseFinish`. The example wants the frontend to receive the refactored text, so it first unsubscribes the old handler:

```python
custom_service.unsubscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMAgentResponseFinish,
)
```

Then it registers a handler for the new `LLMOutputRefactoredFinal` event:

```python
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
```

## Start the Service

Finally, pass the custom service to `Xtalk` and mount it on FastAPI:

```python
xtalk_instance = Xtalk(service_prototype=custom_service)

app = FastAPI(title="Xtalk Server")
xtalk_instance.mount_routes(app)
```
