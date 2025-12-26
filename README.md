# X-Talk
<img width="460" height="249" alt="xtalk-logo-new" src="https://github.com/user-attachments/assets/4e252ce8-7450-4335-b86a-4b9b26200792" />

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge)](https://xtalk.sjtuxlance.com/)
[![arXiv](https://img.shields.io/badge/arXiv-Tech_Report-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2512.18706)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue?style=for-the-badge&labelColor=555555)](https://opensource.org/licenses/Apache-2.0)


<!-- <img src="PENDING" alt="Watermark" style="width: 40px; height: auto"> -->
> ⚠️ X-Talk is in active prototyping. Interfaces and functions are subject to change. We will try to keep interfaces stable.

X-Talk is an open-source full-duplex cascaded spoken dialogue system framework featuring:
- ⚡ **Low-Latency, Interruptible, Human-Like Speech Interaction**
    - Speech flow is optimized to support **impressive low latency**
    - Enables **natural user interruption** during interaction
    - **Paralinguistic information** (e.g. environment noise, emotion) is encoded in parallel to support in-depth understanding and empathy
- 🧪 **Researcher Friendly**
    - **New models and relevant logic** can be added [within one Python script](#introduce-a-new-model), and seamlessly integrated with the default pipeline.
- 🧩 **Super Lightweight**
    - The framework backend is **pure Python**; nothing to build and install beyond `pip install`.
- 🏭 **Production Ready**
    - **Concurrency** is ensured through asynchronous backend
    - Websocket-based implementation empowers deployment **from web browsers to edge devices**.
## 📚 Contents

- [Demo](#demo)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Tutorial](#tutorial)
    - [Start the Server](#start-the-server)
        - [Text Embedding](#text-embedding)
        - [Tool Use](#tool-use)
    - [Config the Server](#config-the-server)
    - [Introduce a New Model](#introduce-a-new-model)
        - [Recipe](#recipe)
    - [Customize the Service](#customize-the-service)
- [Design Philosophy](#design-philosophy)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

<a id="demo"></a>
## 🎬 Demo

### Online Demo
[Demo Link](https://xtalk.sjtuxlance.com/)

This demo runs on 4090 cluster with 8-bit quantized *SenseVoice* as speech recognizer, *IndexTTS 1.5* as speech generator, and 4-bit quantized *Qwen3-30B-A3B* as language model. Though at the cost of intelligence due to a relatively small language model, it demonstrates low latency.

### Demo Videos
<table class="center">
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/e7946357-cd83-493c-8967-354cf87b2acb" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/ca45c463-6738-4b5c-8305-71fce4ab490e" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/8c0f489a-6af6-4711-a28c-7a48740f666c" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/d8fc4d15-edfb-4476-a9d3-983a1ce9be0e" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/7ea4dc44-d43c-45ca-8788-2032b3a387d8" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/9f296d5e-a752-435e-91a2-a9f1a71f9fac" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/2b44f2f1-93c4-47b8-99e0-830338cdba02" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/c4cd4c1b-c4fd-493b-8cb2-347c48ac5809" muted="false"></video>
    </td>
</tr>
<tr>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/d33ca5ef-c722-45a6-93df-2fdb7ffcc729" muted="false"></video>
    </td>
    <td width=50% style="border: none">
        <video controls autoplay loop src="https://github.com/user-attachments/assets/09370641-7a26-4f93-9c98-dee887612fda" muted="false"></video>
    </td>
</tr>
</table>

The tour guiding demos are conducted with *Qwen3-Next-80B-A3B-Instruct* as language model, and the other eight demos are aligned with the online demo setting. Larger language models are more intelligent at the cost of latency.

<a id="installation"></a>
## 🛠️ Installation

```bash
pip install git+https://github.com/xcc-zach/xtalk.git@main
```

<a id="quickstart"></a>
## 🚀 Quickstart

We will use APIs from AliCloud to demonstrate the basic capability of **X-Talk**.

First, install dependencies for AliCloud and server script:
```bash
pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"
pip install jinja2 'uvicorn[standard]'
```

Then, obtain an API key from [AliCloud Bailian Platform](https://bailian.console.aliyun.com/?tab=model#/api-key). We will be using free-tier service from AliCloud.

> Online service may be unstable and of high latency. We recommend using locally deployed models for better user experience. See [server config tutorial](#config-the-server) and [supported models](#supported-models) for details.

After that, create a JSON config specifying the models to use, and **fill in <API_KEY>** with the key you obtained:
<details>
<summary>Basic Config</summary>
    
```json
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": {
        "type": "DefaultAgent",
        "params": {
            "model": {
                "api_key": "<API_KEY>",
                "model": "qwen-plus-2025-12-01",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"
            }
        }
    },
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```
    
</details>

> If you find *Qwen3ASRFlashRealtime* not working properly, you can use `"asr": "SenseVoiceSmallLocal",` instead which is a ~1GB local model. Also, you can try to use local speech generation model *IndexTTS* ([setup tutorial](https://github.com/Ksuriuri/index-tts-vllm)):
> ```json
> "tts": {
>     "type": "IndexTTS",
>     "params": {
>         "port": 6006
>     }
> },
> ```

The next step is to compose the startup script. Since we also need to link frontend webpage and scripts to get the demo working, the startup script is ready at `examples/sample_app/configurable_server.py`. We simply need to start the server with the config file (**fill in <PATH_TO_CONFIG>.json** with the path to the config file we just created) and a custom port:
```bash
git clone https://github.com/xcc-zach/xtalk.git
cd xtalk
python examples/sample_app/configurable_server.py  --port 7635 --config <PATH_TO_CONFIG>.json
```

Finally, our demo is ready at `http://localhost:7635`. View it in the browser!



<a id="tutorial"></a>
## 📖 Tutorial
    
### Start the Server
    
> [!NOTE]
> See `examples/sample_app/configurable_server.py`, `frontend/src` and `examples/sample_app/templates` for details.
   
X-Talk has most models and execution on server side, and the client is responsible for interacting with microphone, transmitting audio and Websocket messages, and handle lightweight operations like Voice-Actitvty-Detection.
    
For client side, you can start with snippet in `examples/sample_app/templates/index.html` and track where `convo` is used to see how to use frontend API:
```html
<script type="module">
        import { createConversation } from "/static/js/index.js";

        const convo = createConversation();
    ...
</script>
```
    
The client-side API mainly comes from `frontend/src/js/index.js`, and if interested, you can check the core code to see how different Websocket messages are handled:
```javascript
switch (json.action) {
            case 'queue_status': {...}
            case 'queue_granted': {...}
    ...
}    
```

We plan to improve the client-side API in the near future.

For the server side, the core logic is to connect a X-Talk instance to Websocket of FastAPI instance:
```python
from fastapi import FastAPI, WebSocket
from xtalk import Xtalk
app = FastAPI(title="Xtalk Server")
xtalk_instance = Xtalk.from_config("path/to/config.json")
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)
```
    
Then you can check `examples/sample_app/configurable_server.py` for how to mount client-side scripts and pages.
    
#### Text Embedding

> [!NOTE]
> See `examples/sample_app/configurable_server.py` and `frontend/src/js/index.js` for details.
    
X-Talk can understand documents uploaded through embedding search. To enable embedding, you need `langchain_openai.OpenAIEmbeddings` in the config:
```json
"embeddings": {
    "type": "OpenAIEmbeddings",
    "params": {
      "api_key": "<API_KEY>",
      "base_url": "<URL LIKE http://127.0.0.1:8002/v1>",
      "model": "<MODEL LIKE Qwen/Qwen3-Embedding-0.6B>"
    }
  },
```

Then you can fetch `text` and `session_id` from client side and notify X-Talk instance through `embed_text`:
```python
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
```
    
Note that client side should save `session_id` and send it in the request. Search `'session_info'` and `uploadFile` in `frontend/src/js/index.js` for how `session_id` is saved and used.

#### Tool Use
    
> [!NOTE]
> See `examples/sample_app/mental_consultant_server.py` for details.
    
X-Talk supports textual tool customization through `add_agent_tools`:
```python
xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])
```
    
Here tool should be a [Langchain tool](https://docs.langchain.com/oss/python/langchain/tools):
```python
from langchain.tools import tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to look for
        limit: Maximum number of results to return
    """
    return f"Found {limit} results for '{query}'"
```

In order to maintain seperate states for a tool in echo agent, you can also use a tool factory to maintain internal states (see `build_mental_questionnaire_tool` `examples/sample_app/mental_consultant_server.py`)
    
##### Built-in Tools
    
> [!NOTE]
> See source code under `src/xtalk/llm_agent/tools` for all built-in tools.
    
Built-in tools include agent-scope ones like `web_search` and `get_time`, and pipeline control ones like emotion, timbre and speed of speech. `DefaultAgent` has built-in tools registered by default.
    
> [!NOTE]
> In order to enable `web_search` tool, `SERPER_API_KEY` needs to be set. See [SerpAPI](https://serpapi.com/).

### Config the Server
    
As mentioned [before](#start-the-server), X-Talk instance can be created from a JSON config, which customizes models used and controls concurrency behavior.
    
For model config, config should match model Python class name and init args. For example, the definition of `DefaultAgent` lies in `src/xtalk/llm_agent/default.py`:
```python
class DefaultAgent(Agent):
    def __init__(
            self,
            model: BaseChatModel | dict,
            system_prompt: str = _BASE_PROMPT,
            voice_names: Optional[List[str]] = None,
            emotions: Optional[List[str]] = None,
            tools: Optional[List[Union[BaseTool, Callable[[], BaseTool]]]] = None,
        ):
    ...
```
    
In order to match with the init args, the config item should look like:
```
"llm_agent": {
    "type": "DefaultAgent",
    "params": {
      "model": {
        "api_key": "none",
        "base_url": "http://127.0.0.1:8000/v1",
        "model": "cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
      },
      "voice_names": [
        "Man",
        "Woman",
        "Child"
      ],
      "emotions": [
        "happy",
        "angry",
        "sad",
        "fear",
        "disgust",
        "depressed",
        "surprised",
        "calm",
        "normal"
      ]
    }
  },
```

Optional keys like `voice_names`, `emotions` and `tools`(not supported in config yet) can be ignored.
    
See [below](##Supported-Models) for full list of model types (`slot`), their optional dependencies, and their adapting location in source code.
> [!NOTE]
> Most model implementations are client-side adaptors. You may need to start the model instance following coresponding instructions.
    
Also, you can restrict concurrency through:
```json
    "max_connections": 1
```

### Introduce a New Model
    
> [!NOTE]
> See `examples/sample_app/custom_model.py` and `examples/sample_app/echo_agent.py` for details.
    
> [!NOTE]
> See [Recipe](#recipe) for adding a model of existing types.
    
You may want to introduce a new model of an existing type (e.g. text-to-speech), or add a model of new type (e.g. a model that handles backchannel). This can be achieved by `register_model_search_spec` before a `xtalk_instance` is created from config:
    
```python
from xtalk import Xtalk
Xtalk.register_model_search_spec(
    slot="llm_agent",
    spec=Path(__file__).parent / "echo_agent.py",
)
xtalk_instance = Xtalk.from_config(args.config)
```
    
Here `slot` matches the name of corresponding init arg in `Pipeline`. You can check `Xtalk.MODEL_REGISTRY` for existing slots, or use a new slot to represent a new type of models (see `examples\sample_app\custom_service.py` and there `llm_output_refactor_model` can be the new slot).
    
`spec` is the path to model implementation, an example implementation in `echo_agent.py` looks like this:

```python
from xtalk.model_types import Agent

class EchoAgent(Agent):
    """A simple agent that echoes user input."""

    def generate(self, input) -> str:
        if isinstance(input, dict):
            return input["content"]
        return input

    def clone(self) -> "EchoAgent":
        return EchoAgent()

```
    
Then you can use the custom model in config file:
```python
{
    "asr": {
        "type": "Qwen3ASRFlashRealtime",
        "params": {
            "api_key": "<API_KEY>"
        }
    },
    "llm_agent": "EchoAgent",
    "tts": {
        "type": "CosyVoice",
        "params": {
            "api_key": "<API_KEY>"
        }
    }
}
```

#### Recipe

Recipes for major model customization are listed below. You can read source code for interfaces of other model types. We will update these interfaces from time to time.
    
> [!NOTE]
> See `src/xtalk/model_types.py` for all available model types.
    
> [!IMPORTANT]
> X-Talk has asynchronous default implementations for sync versions, which usually with `run_in_executor`, like `async_recognize` for `recognize` w.r.t. ASR. However, in order to achieve best concurrency for production, we recommend to implement these async versions by your self.

##### New ASR (auto-speech-recognition) Model
    
Your ASR class must inherit from `xtalk.speech.interfaces.ASR` and implement the following methods:

* **`recognize(audio: bytes) -> str`**
    * Recognize audio in a single pass. 
* **`reset() -> None`**
    * Reset internal recognition state.
* **`clone() -> ASR`**
    * Return a new instance for use in new or concurrent sessions.
    * Sharing weights/connections (e.g., `_shared_model`) is allowed, but you can't share states. 
    
Methods below are optional:
* **`recognize_stream(audio: bytes, *, is_final: bool = False) -> str`**
    * Interface for streaming incremental recognition.
    * Returns the "current cumulative recognition result up to this point".
* **`async_recognize(audio: bytes)`**
* **`async def async_recognize_stream(
        self, audio: bytes, *, is_final: bool = False
    )`**

    
> [!IMPORTANT]
> Input for `recognize` and `recognize_stream` is PCM 16-bit mono 16 kHz raw bytes. You may need to do conversion by yourself.
    
> [!NOTE]
> X-Talk have default implementation for `recognize_stream` with a `MockStreamRecognizer`. Therefore, no worry for your non-streaming ASR models.

> [!NOTE]
> You can refer to existing implementations (e.g., `src/xtalk/speech/asr/zipformer_local.py`) when building your own ASR class. We recommend deploying ASR as a separate service and invoking it via API calls within the ASR class, referencing the implementation of `src/xtalk/speech/asr/sherpa_onnx_asr.py`.
    
##### New TTS (text-to-speech) Model
    
Your new TTS class must inherit from `xtalk.speech.interfaces.TTS` and implement the following methods:


- **`synthesize(self, text: str) -> bytes`**

  - Input: The text to synthesize.
  - Output: Raw audio bytes in PCM 16-bit, mono, 48000 Hz.

- **`clone(self) -> TTS`**

  - Return a new TTS instance:
    - It should have isolated runtime state to avoid cross-session interference and it may share read-only resources if your backend supports that.
    
**Optional methods**

- **`synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]`**
  - If your backend supports streaming synthesis, you can override this method.
- **`set_voice(self, voice_names: list[str])`**

  - This method works with the `TTSVoiceChange` event in `TTSManager` to switch voices via language model tool calls.
  - Usually there is only one element in `voice_names`, and this is the current behavior for tool call result. However, some TTS models may support mixing multiple voices for reference. Therefore, `voice_names` is list type.

- **`set_emotion(self, emotion: str | list[float])`**

  - This method works with the `TTSEmotionChange` event in `TTSManager` to switch emotions via language model tool calls.
  - Current tool call result only carries `emotion` as `str`. However, you may also want `list[float]` as emotion vector for future use.
    
- **`async def async_synthesize(self, text: str, **kwargs: Any)`**
- **`async def async_synthesize_stream(
        self, text: str, **kwargs: Any
    )`**    
    
### Customize the Service

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

<a id="design-philosophy"></a>
## 🧩 Design Philosophy

![XTalk Data Flow](https://github.com/user-attachments/assets/db52398a-80b1-46e0-9e09-63cf301eed59)
*Prospective Data Flow of X-Talk*

![XTalk Architecture](https://github.com/user-attachments/assets/5187cd7b-ce8d-4273-8c60-6b552e139c3d)
*Architecture of X-Talk*

X-Talk follows a **modular, stage-wise functional flow**, progressing from noisy speech input, through frontend speech interaction, speech understanding, and an LLM-driven conversational agent, to speech generation. This logical pipeline is realized through a **layered, event-driven, and loosely-coupled architecture**, which forms the core of the system.

This design systematically addresses the key challenges of real-time speech-to-speech dialogue systems:

- Controlling **sub-second end-to-end latency**
- Orchestrating **multiple heterogeneous components**
- Enabling **flexible integration and swapping of backend models and services**

The entire system is built around a **centralized event bus**. All layers communicate asynchronously through event publishing and subscribing, enabling efficient management of complex conversational state and data flow.


### Frontend Layer

The Frontend Layer serves as the user-facing interface and directly handles browser-based interaction. It is responsible for:

- Rendering the conversational user interface  
- Performing client-side Voice Activity Detection (VAD)  
- Applying audio denoising and enhancement  
- Displaying real-time latency metrics to the user  

This layer packages audio streams, VAD markers, and contextual information for transmission to the backend.


### Event Center Layer

The Event Center Layer acts as the system’s **communication hub and network boundary**, unifying event routing and protocol translation. It consists of two tightly integrated components:

- **Gateways**  
  - The *Input Gateway* converts frontend streams into typed internal events  
  - The *Output Gateway* delivers processed events back to the frontend  

- **Event Bus**  
  - Provides the asynchronous messaging fabric  
  - Routes events between all components in the system  

Together, these components decouple all other layers by handling protocol adaptation, event distribution, and lifecycle isolation, forming the **extensible backbone** of the architecture.


### Managers Layer

The Managers Layer orchestrates the core conversational workflow through **specialized, capability-specific managers**. Each manager:

- Subscribes to relevant events  
- Executes its dedicated logic (e.g., ASR, LLM inference, TTS)  
- Publishes new events to drive the dialogue forward  

This event-driven orchestration enables fine-grained control over execution order, concurrency, and latency.


### Agents Layer

The Agents Layer functions as the system’s **task-planning and execution engine**. It integrates structured inputs from upstream models—such as ASR outputs, voice captions, and contextual signals—into a coherent speech understanding.

Based on this understanding, the agent orchestrates tool usage, including:

- Web search  
- Local retrieval  
- Audio control  
- External API calls  

Finally, it synthesizes retrieved or processed information into a **context-aware natural language response**.


### Models Layer

The Models Layer provides a **unified, interface-driven abstraction** for core speech-to-speech dialogue capabilities, including:

- Speech understanding  
- LLM-based conversational agents  
- Speech generation  

By defining stable and modular contracts for each capability, this layer allows compliant implementations to be seamlessly integrated, swapped, or scaled without impacting other system components.

<a id="supported-models"></a>
## ✅ Supported Models
    
### Speech Recognition
    
**Slot**: `asr`

[SherpaOnnx](https://github.com/k2-fsa/sherpa-onnx) is recommended for its wide support of models and optimized inference performance.
    
<details>
<summary>SherpaOnnx</summary>
    
**Dependency:** `pip install "xtalk[sherpa-onnx-asr] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/sherpa_onnx_asr.py`
    
A high-performance speech recognition framework and beyond.

[Repo](https://github.com/k2-fsa/sherpa-onnx)
    
[Models](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models)
    
[Tutorial to start speech recognition server](https://k2-fsa.github.io/sherpa/onnx/sense-voice/python-api.html#websocket-server-and-client-example)
    
</details>

<details>
<summary>Qwen3ASRFlashRealtime</summary>
    
**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/qwen3_asr_flash_realtime.py`
    
[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/qwen3-asr-flash)
    
</details>
    
<details>
<summary>Zipformer</summary>
    
**Dependency:** `pip install "xtalk[zipformer-local] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/zipformer_local.py`
    
[Details](https://www.modelscope.cn/models/yhdai666/xtalk_zipformer_onnx/summary)
    
</details>
    
<details>
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/asr/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/speech-to-text/convert)
    
</details>
    
### Text to Speech
    
**Slot**: `tts`
    
<details>
<summary>IndexTTS</summary>

**Dependency:** `pip install "xtalk[index-tts] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** 
- `src/xtalk/speech/tts/index_tts.py`
- `src/xtalk/speech/tts/index_tts2.py`
    
[Repo](https://github.com/index-tts/index-tts)
    
[Installation (vllm boost)](https://github.com/Ksuriuri/index-tts-vllm)

    
</details>
    
<details>
<summary>GPT-SoVITS</summary>

> Experimental. Feel free to open an issue for any problem.

**Dependency:** `pip install "xtalk[gpt-sovits] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/gpt_sovits.py`

    
[Repo](https://github.com/RVC-Boss/GPT-SoVITS)
    

    
</details>
    
<details>
<summary>CosyVoice</summary>

**Dependency:** `pip install "xtalk[ali] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/cosyvoice.py`

[Details](https://bailian.console.aliyun.com/?tab=model#/model-market/detail/cosyvoice-v3-flash)
    
</details>
    
<details>
<summary>ElevenLabs</summary>
    
**Dependency:** `pip install "xtalk[elevenlabs] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/tts/elevenlabs.py`
    
[API Reference](https://elevenlabs.io/docs/api-reference/text-to-speech/convert)
    
</details>

### Voice Activity Detection
    
**Slot**: `vad`
    
X-Talk has VAD on client side, so you may not need one.

<details>
<summary>Silero VAD</summary>

**Dependency:** `pip install "xtalk[silero-vad] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/vad/silero_vad.py`
    
[Model Details](https://github.com/snakers4/silero-vad)
[VAD-Web](https://github.com/ricky0123/vad)
    
</details>
    
### Speech Enhancement
    
**Slot**: `speech_enhancer`

<details>
<summary>FastEnhancer</summary>
    
**Dependency:** `pip install onnxruntime`
**Path:** `src/xtalk/speech/speech_enhancer/speech_enhancer.py`
    
[Model Details](https://github.com/aask1357/fastenhancer)
    
</details>
    
### Speaker Recognition
    
**Slot**: `speaker_encoder`    
    
<details>
<summary>Wespeaker-Voxceleb-Resnet34-LM</summary>
  
**Dependency:** `pip install "xtalk[pyannote] @ git+https://github.com/xcc-zach/xtalk.git@main"`
**Path:** `src/xtalk/speech/speaker_encoder/pyannote_embedding.py`
    
[Wespeaker](https://github.com/wenet-e2e/wespeaker)
[Model Details](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)
    
</details>
    
### Captioner
    
**Slot**: `captioner`
    
Captioners give you description of audio clip.
    
<details>
<summary>Qwen3-Omni-30B-A3B-Captioner</summary>
   
**Dependency:** None
**Path:** `src/xtalk/speech/captioner/qwen3_omni_captioner.py`
    
[HuggingFace](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Captioner)
[ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Omni-30B-A3B-Captioner)
    
</details>
    

## Contributing
    
See [Contribution Guide](CONTRIBUTING.md)

## Acknowledgements

We express sincere gratitude for:

- [Langchain](https://www.langchain.com/) as backbone of LLM agents
- [vllm](https://github.com/vllm-project/vllm) for deployment of most models
- All model providers mentioned in [Supported Models](#supported-models)

All of you provide the solid foundation of X-Talk!

## License
    
This project is licensed under the Apache License 2.0, if you do not install optional dependencies. Some optional dependencies may be under incompatible licenses.
