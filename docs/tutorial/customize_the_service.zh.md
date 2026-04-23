> [!NOTE]
> 详情请参阅 `examples/sample_app/custom_service.py`。其中向 X-Talk 添加了一个假的 `LLMOutputRefactorModel`，用于在模型响应文本前附加 `Assistant response: `。

如果您想添加新的功能，可以按照下面的流程进行：

首先，您可能需要定义一个新模型。下面这个模型会在输入前添加一段文本：
```python
# 定义一个自定义模型
class LLMOutputRefactorModel:
    def refactor(self, llm_output: str) -> str:
        # 自定义逻辑：重写 LLM 输出
        return "Assistant response: " + llm_output

    # 如果自定义模型有内部状态，请实现 clone 方法以复制具体状态
    def clone(self):
        return LLMOutputRefactorModel()
```
请注意，如果您的模型具有需要在不同用户会话之间隔离的内部状态，例如流式语音识别模型中的识别缓存，那么就必须实现 `clone`。

如果您定义了一个新模型，或者希望给 `Pipeline` 添加新功能，第二步就是定义一个自定义 `Pipeline`：
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
请注意，`__init__` 中的 `**kwargs` 是必须的，用来接收从 `DefaultPipeline` 继承但被遮蔽的参数。如果您在 `__init__` 中新增了参数，就需要把它注册为一个 `field`，并指定其 `clone` 行为（`True/False`）。

基于 X-Talk 的事件总线机制，接下来您可以新增一个 `Manager`，订阅已有的 `Event`，并实现所需的自定义功能。必要时，您也可以创建新的 `Event`。
例如：
```python
LLMOutputRefactoredFinal = create_event_class(
    name="LLMOutputRefactoredFinal", fields={"text": "", "turn_id": 0} # 键: 默认值
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

然后，您还可以通过 `unsubscribe_event` 和 `subscribe_event` 将其他组件（例如 `OutputGateway`）从订阅旧事件切换为订阅新事件。同时，对于新事件，您也需要实现相应的处理方法。

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
            "action": "finish_resp", # 你可以在 frontend/src/js/index.js 中找到 "finish_resp"
            "data": {"text": event.text, "turn_id": event.turn_id},
        }
    )

custom_service.subscribe_event(
    event_listener_cls=OutputGateway,
    event_type=LLMOutputRefactoredFinal,
    method_or_handler=output_gateway_llm_output_refactored_final_handler,
)
```
