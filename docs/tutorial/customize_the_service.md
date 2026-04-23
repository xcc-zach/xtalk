> [!NOTE]
> See `examples/sample_app/custom_service.py` for details. A dummy `LLMOutputRefactorModel` is added to X-Talk to prepend `Assistant response: ` before model response text.
    
If you want to add new functionality, you can follow the procesures below:
    
First, you may want to define a new model. Here is a model that prepend some text before input:
```python
# Define a custom model
class LLMOutputRefactorModel:
    def refactor(self, llm_output: str) -> str:
        # Custom logic to refactor LLM output
        return "Assistant response: " + llm_output

    # If custom model has internal state, implement clone method with concrete state
    def clone(self):
        return LLMOutputRefactorModel()
```
Note that `clone` is neccesary when your model has internal state that should be distinct across user sessions, like the recognition cache of a streaming speech recognition model.
    
If you define a new model, or want to add some new function to `Pipeline`, the second step is to define a custom `Pipeline`:
```python
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
```
Note that `**kwargs` is necessary in `__init__` to swallow shadowed parameters from `DefaultPipeline`. And if you add a new arg to `__init__`, you will need to register it as a `field`, specifying its `clone` behavior (`True/False`)

Based on X-Talk’s event-bus mechanism, then you can add a new `Manager` to subscribe to an existing `Event` and implement the custom functionality you need. Meanwhile, you can create a new `Event` if needed.
    For Example:
```python
LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal", fields={"text": "", "turn_id": 0} # key: default_value
)

class LLMOutputRefactorManager(Manager):
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

    async def shutdown(self):
        pass
    
custom_service = DefaultService(pipeline=pipeline)
custom_service.register_manager(LLMOutputRefactorManager)
```
    
Then you can optionally use `unsubscribe_event` and `subscribe_event` to switch other components (such as `OutputGateway`) from subscribing the old event to the new event. Meanwhile, for the new event, you need to implement the handling method.

```python
custom_service.unsubscribe_event(
    event_listener_cls=OutputGateway, event_type=LLMAgentResponseFinish
)

async def output_gateway_llm_output_refactored_final_handler(
    self: OutputGateway,
    event,
):
    await self.send_signal(
        {
            "action": "finish_resp", # you can find "finish_resp" in frontend/src/js/index.js
            "data": {"text": event.text, "turn_id": event.turn_id},
        }
    )

custom_service.subscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMOutputRefactoredFinal,
    method_or_handler=output_gateway_llm_output_refactored_final_handler,
)
```