<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.model_types

## Embeddings

_定义于 [`xtalk.models.embeddings.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/embeddings/interfaces.py)。_

```python
@model_type(aliases=['embeddings'])
class Embeddings(_LangChainEmbeddings)
```

Interface marker for embedding models.

## ForcedAligner

_定义于 [`xtalk.models.forced_aligner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/interfaces.py)。_

```python
@model_type
class ForcedAligner(ABC)
```

Abstract interface for forced alignment models.

### 方法

#### align

_定义于 [`xtalk.models.forced_aligner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/interfaces.py)。_

```python
def align(self, *, audio: bytes, text: str, language: str | None = None) -> list[ForcedAlignmentUnit]
```

Align text units against 48 kHz PCM audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 48 kHz.
- `text` (`str`)
  Original text that the audio speaks.
- `language` (`str | None, optional`)
  Optional model-specific language hint.

#### async_align

_定义于 [`xtalk.models.forced_aligner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/interfaces.py)。_

```python
async def async_align(self, *, audio: bytes, text: str, language: str | None = None) -> list[ForcedAlignmentUnit]
```

Asynchronously align text units against 48 kHz PCM audio.

#### clone

_定义于 [`xtalk.models.forced_aligner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/forced_aligner/interfaces.py)。_

```python
def clone(self) -> 'ForcedAligner'
```

Clone the aligner for a new service session.

## BaseChatModel

_定义于 `langchain_core.language_models.chat_models`。_

```python
from langchain_core.language_models.chat_models import BaseChatModel
```

External dependency re-exported by this module.

## Agent

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
@model_type(aliases=['llm_agent'])
class Agent(ABC)
```

Abstract interface for conversational agents used by Xtalk.

### 方法

#### content_to_text

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def content_to_text(content: Any) -> str
```

Normalize model content blocks into plain text.

##### 参数

- `content:`
  Content emitted by a LangChain model chunk or message.

##### 返回

- `str`
  Plain-text content extracted from the input.

#### accept

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def accept(self, context: AgentContext) -> Iterable[AgentOutput]
```

Accept an incremental context update.

##### 参数

- `context` (`AgentContext`)
  Context payload forwarded from serving-layer events.

##### 生成

- `AgentStreamItem`
  Zero or more streamed response items triggered by the context
  update.

#### async_accept

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
async def async_accept(self, context: AgentContext) -> AsyncIterator[AgentOutput]
```

Asynchronously accept an incremental context update.

##### 参数

- `context` (`AgentContext`)
  Context payload forwarded from serving-layer events.

##### 生成

- `AgentStreamItem`
  Streamed response items triggered by the context update.

#### sync_iter_from_async

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def sync_iter_from_async(self, async_iter: AsyncIterator[T]) -> Iterable[T]
```

Convert an async iterator into a synchronous generator.

##### 参数

- `async_iter` (`AsyncIterator[T]`)
  Async iterator to bridge into synchronous iteration.

##### 生成

- `T`
  Items produced by ``async_iter``.

#### clone

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def clone(self) -> 'Agent'
```

Clone the agent for a new session.

##### 返回

- `Agent`
  Session-safe agent instance.

#### restore_history

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def restore_history(self, messages: list[dict[str, Any]]) -> None
```

Restore persisted conversation messages into the agent state.

##### 参数

- `messages` (`list[dict[str, Any]]`)
  Persisted chat messages ordered by session history.

#### get_chat_history

_定义于 [`xtalk.models.agents.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/agents/interfaces.py)。_

```python
def get_chat_history(self, with_system: bool = False) -> str | None
```

Return the serialized conversation history when available.

##### 参数

- `with_system` (`bool, optional`)
  Whether to include the system prompt message when supported by the
  concrete implementation.

##### 返回

- `str | None`
  Conversation history or ``None``.

## Rewriter

_定义于 [`xtalk.models.rewriters.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/rewriters/interfaces.py)。_

```python
@model_type(aliases=['caption_rewriter'])
class Rewriter(ABC)
```

Abstract interface for text rewriting helpers.

### 方法

#### rewrite

_定义于 [`xtalk.models.rewriters.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/rewriters/interfaces.py)。_

```python
def rewrite(self, input: str) -> str
```

Rewrite input text.

##### 参数

- `input` (`str`)
  Source text to rewrite.

##### 返回

- `str`
  Rewritten text.

#### async_rewrite

_定义于 [`xtalk.models.rewriters.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/rewriters/interfaces.py)。_

```python
async def async_rewrite(self, input: str) -> str
```

Asynchronously rewrite input text.

##### 参数

- `input` (`str`)
  Source text to rewrite.

##### 返回

- `str`
  Rewritten text.

## ASR

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
@model_type(aliases=['asr'])
class ASR(ABC)
```

Abstract interface for automatic speech recognition.

### 方法

#### recognize

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
def recognize(self, audio: bytes) -> str
```

Recognize a full audio buffer.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `str`
  Recognized text.

#### recognize_stream

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
def recognize_stream(self, audio: bytes, *, is_final: bool = False, chat_history: str | None = None) -> str
```

Recognize audio incrementally in streaming mode.

##### 参数

- `audio` (`bytes`)
  Incremental PCM 16-bit mono audio bytes.
- `is_final` (`bool, optional`)
  Whether the caller is asking the ASR to treat the current point as
  a temporary boundary and optionally flush any tail audio that would
  otherwise remain buffered. This is only a decoding hint. It does
  not mean the streaming state must be reset, and previously
  recognized text for the session must be preserved so later audio
  can continue from the accumulated result.
- `chat_history` (`str | None, optional`)
  Serialized chat history for the current session, excluding the
  in-progress turn when unavailable.

##### 返回

- `str`
  Current recognition result.

#### stream_chunk_bytes_hint

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
def stream_chunk_bytes_hint(self) -> int | None
```

Return the preferred streaming chunk size.

##### 返回

- `int | None`
  Recommended byte count for each chunk passed to
  ``recognize_stream``, or ``None`` when no preference is provided.

#### reset

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
def reset(self) -> None
```

Reset internal recognition state.

#### clone

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
def clone(self) -> 'ASR'
```

Clone the ASR instance for a new session.

##### 返回

- `ASR`
  Clone with shared weights and independent runtime state.

#### async_recognize

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
async def async_recognize(self, audio: bytes) -> str
```

Asynchronously recognize a full audio buffer.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `str`
  Recognized text.

#### async_recognize_stream

_定义于 [`xtalk.models.asr.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/asr/interfaces.py)。_

```python
async def async_recognize_stream(self, audio: bytes, *, is_final: bool = False, chat_history: str | None = None) -> str
```

Asynchronously recognize incremental audio input.

##### 参数

- `audio` (`bytes`)
  Incremental PCM 16-bit mono audio bytes.
- `is_final` (`bool, optional`)
  Whether the caller is asking the ASR to treat the current point as
  a temporary boundary and optionally flush any tail audio that would
  otherwise remain buffered. This is only a decoding hint. It does
  not mean the streaming state must be reset, and previously
  recognized text for the session must be preserved so later audio
  can continue from the accumulated result.
- `chat_history` (`str | None, optional`)
  Serialized chat history for the current session, excluding the
  in-progress turn when unavailable.

##### 返回

- `str`
  Current recognition result.

## TTS

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
@model_type(aliases=['tts'])
class TTS(ABC)
```

Abstract base class for text-to-speech engines.

### 说明

``synthesize`` is the required baseline API for every implementation.
Streaming-capable engines should additionally override
``synthesize_stream``; non-streaming engines should inherit the default
compatibility wrapper. The inherited streaming helpers do not by
themselves declare native streaming capability.

### 方法

#### synthesize

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
def synthesize(self, text: str) -> bytes
```

Synthesize audio for a full text input.

##### 参数

- `text` (`str`)
  Text to synthesize.

##### 返回

- `bytes`
  PCM 16-bit mono audio bytes at 48 kHz.

##### 说明

Every TTS implementation, including streaming backends, must provide
this method.

#### synthesize_stream

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
def synthesize_stream(self, text: str, **kwargs) -> Iterable[bytes]
```

Stream synthesized audio chunks for a text input.

##### 参数

- `text` (`str`)
  Text to synthesize.
- `**kwargs`
  Model-specific streaming options.

##### 生成

- `bytes`
  PCM 16-bit mono audio bytes at 48 kHz.

##### 说明

Override this method only when the backend supports native streaming
synthesis. The default implementation yields a single chunk produced
by ``synthesize`` for compatibility and should not be treated as a
declaration of streaming support.

#### async_synthesize

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
async def async_synthesize(self, text: str, **kwargs: Any) -> bytes
```

Asynchronously synthesize audio for text.

##### 参数

- `text` (`str`)
  Text to synthesize.
- `**kwargs`
  Model-specific synthesis options.

##### 返回

- `bytes`
  Synthesized PCM audio bytes.

##### 说明

This method is an optional async optimization. Implementations may
inherit the default executor-based wrapper.

#### async_synthesize_stream

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
async def async_synthesize_stream(self, text: str, **kwargs: Any) -> AsyncIterator[bytes]
```

Asynchronously stream synthesized audio chunks.

##### 参数

- `text` (`str`)
  Text to synthesize.
- `**kwargs`
  Model-specific synthesis options.

##### 生成

- `bytes`
  Streamed PCM audio chunks.

##### 说明

This method is an optional async optimization for streaming-capable
backends. When not overridden, it asynchronously iterates over
``synthesize_stream``.

#### clone

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
def clone(self) -> 'TTS'
```

Clone the TTS engine for a new session.

##### 返回

- `TTS`
  Session-safe clone.

#### set_voice

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
def set_voice(self, voice_names: list[str]) -> None
```

Update the active voice selection.

##### 参数

- `voice_names` (`list[str]`)
  One or more voice names understood by the implementation.

#### set_emotion

_定义于 [`xtalk.models.tts.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/tts/interfaces.py)。_

```python
def set_emotion(self, emotion: str | list[float]) -> None
```

Update the active synthesis emotion.

##### 参数

- `emotion` (`str | list[float]`)
  Emotion label or model-specific emotion vector.

## Captioner

_定义于 [`xtalk.models.captioner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/interfaces.py)。_

```python
@model_type(aliases=['captioner'])
class Captioner(ABC)
```

Abstract base class for audio captioning models.

### 方法

#### caption

_定义于 [`xtalk.models.captioner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/interfaces.py)。_

```python
def caption(self, audio: bytes) -> str
```

Generate a caption for audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `str`
  Generated caption text.

#### caption_stream

_定义于 [`xtalk.models.captioner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/interfaces.py)。_

```python
def caption_stream(self, audio: bytes) -> Iterable[str]
```

Stream caption text for audio input.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 生成

- `str`
  Streamed caption text.

#### async_caption

_定义于 [`xtalk.models.captioner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/interfaces.py)。_

```python
async def async_caption(self, audio: bytes) -> str
```

Asynchronously caption audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `str`
  Generated caption text.

#### async_caption_stream

_定义于 [`xtalk.models.captioner.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/captioner/interfaces.py)。_

```python
async def async_caption_stream(self, audio: bytes) -> AsyncIterator[str]
```

Asynchronously stream caption text.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 生成

- `str`
  Streamed caption text.

## PuntRestorer

_定义于 [`xtalk.models.punt_restorer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/punt_restorer/interfaces.py)。_

```python
@model_type(aliases=['punt_restorer_model'])
class PuntRestorer(ABC)
```

Abstract base class for punctuation restoration models.

### 方法

#### restore

_定义于 [`xtalk.models.punt_restorer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/punt_restorer/interfaces.py)。_

```python
def restore(self, text: str) -> str
```

Restore punctuation in text.

##### 参数

- `text` (`str`)
  Text without reliable punctuation.

##### 返回

- `str`
  Text with restored punctuation.

#### async_restore

_定义于 [`xtalk.models.punt_restorer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/punt_restorer/interfaces.py)。_

```python
async def async_restore(self, text: str) -> str
```

Asynchronously restore punctuation in text.

##### 参数

- `text` (`str`)
  Text without reliable punctuation.

##### 返回

- `str`
  Restored text.

## VAD

_定义于 [`xtalk.models.vad.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/interfaces.py)。_

```python
@model_type(aliases=['vad'])
class VAD(ABC)
```

Abstract base class for voice activity detection engines.

### 方法

#### is_speech

_定义于 [`xtalk.models.vad.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/interfaces.py)。_

```python
def is_speech(self, frame: bytes) -> bool
```

Determine whether an audio frame contains speech.

##### 参数

- `frame` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bool`
  ``True`` if speech is detected, otherwise ``False``.

#### async_is_speech

_定义于 [`xtalk.models.vad.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/interfaces.py)。_

```python
async def async_is_speech(self, frame: bytes) -> bool
```

Asynchronously determine whether an audio frame contains speech.

##### 参数

- `frame` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.

##### 返回

- `bool`
  ``True`` if speech is detected, otherwise ``False``.

#### reset

_定义于 [`xtalk.models.vad.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/interfaces.py)。_

```python
def reset(self) -> None
```

Reset session-local VAD state and release external resources.

#### clone

_定义于 [`xtalk.models.vad.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/vad/interfaces.py)。_

```python
def clone(self) -> 'VAD'
```

Clone the VAD instance for a new session.

##### 返回

- `VAD`
  Clone with shared weights and independent runtime state.

## SpeechEnhancer

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
@model_type(aliases=['speech_enhancer'])
class SpeechEnhancer(ABC)
```

Abstract base class for speech enhancement engines.

### 说明

Inputs and outputs use PCM 16-bit mono audio bytes at 16 kHz.

### 方法

#### enhance

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
def enhance(self, audio: bytes, far: bytes) -> bytes
```

Enhance an audio frame.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.
- `far` (`bytes`)
  Far-end reference PCM 16-bit mono audio bytes at 16 kHz. The
  upstream audio pipeline guarantees that it has the same byte length
  as ``audio``.

##### 返回

- `bytes`
  Enhanced PCM audio bytes.

#### flush

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
def flush(self) -> bytes
```

Flush any internally buffered audio.

##### 返回

- `bytes`
  Remaining enhanced PCM audio bytes.

#### async_enhance

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
async def async_enhance(self, audio: bytes, far: bytes) -> bytes
```

Asynchronously enhance audio.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes at 16 kHz.
- `far` (`bytes`)
  Far-end reference PCM 16-bit mono audio bytes at 16 kHz. The
  upstream audio pipeline guarantees that it has the same byte length
  as ``audio``.

##### 返回

- `bytes`
  Enhanced PCM audio bytes.

#### async_flush

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
async def async_flush(self) -> bytes
```

Asynchronously flush buffered audio.

##### 返回

- `bytes`
  Remaining enhanced PCM audio bytes.

#### reset

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
def reset(self) -> None
```

Reset internal buffers and caches.

#### clone

_定义于 [`xtalk.models.speech_enhancer.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_enhancer/interfaces.py)。_

```python
def clone(self) -> 'SpeechEnhancer'
```

Clone the speech enhancer for a new session.

##### 返回

- `SpeechEnhancer`
  Clone with shared weights and isolated runtime state.

## SpeakerEncoder

_定义于 [`xtalk.models.speaker_encoder.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_encoder/interfaces.py)。_

```python
@model_type(aliases=['speaker_encoder'])
class SpeakerEncoder(ABC)
```

Abstract base class for speaker embedding models.

### 方法

#### extract

_定义于 [`xtalk.models.speaker_encoder.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_encoder/interfaces.py)。_

```python
def extract(self, audio: bytes) -> np.ndarray
```

Generate a speaker embedding vector.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `np.ndarray`
  Speaker embedding vector.

#### async_extract

_定义于 [`xtalk.models.speaker_encoder.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_encoder/interfaces.py)。_

```python
async def async_extract(self, audio: bytes) -> np.ndarray
```

Asynchronously extract a speaker embedding.

##### 参数

- `audio` (`bytes`)
  PCM 16-bit mono audio bytes.

##### 返回

- `np.ndarray`
  Speaker embedding vector.

#### similarity

_定义于 [`xtalk.models.speaker_encoder.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speaker_encoder/interfaces.py)。_

```python
def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float
```

Compute similarity between two speaker embeddings.

##### 参数

- `embedding1` (`np.ndarray`)
  First speaker embedding.
- `embedding2` (`np.ndarray`)
  Second speaker embedding.

##### 返回

- `float`
  Cosine similarity score.

## SpeechSpeedController

_定义于 [`xtalk.models.speech_speed_controller.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_speed_controller/interfaces.py)。_

```python
@model_type(aliases=['speech_speed_controller'])
class SpeechSpeedController(ABC)
```

Interface for TTS speed controllers.

### 方法

#### process

_定义于 [`xtalk.models.speech_speed_controller.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_speed_controller/interfaces.py)。_

```python
def process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes
```

Apply a speed adjustment to synthesized audio.

##### 参数

- `audio_bytes` (`bytes`)
  Synthesized audio bytes.
- `speed` (`float, optional`)
  Speed multiplier.

##### 返回

- `bytes`
  Processed audio bytes.

#### async_process

_定义于 [`xtalk.models.speech_speed_controller.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/speech_speed_controller/interfaces.py)。_

```python
async def async_process(self, audio_bytes: bytes, speed: float = 1.0) -> bytes
```

Asynchronously apply a speed adjustment to audio.

##### 参数

- `audio_bytes` (`bytes`)
  Synthesized audio bytes.
- `speed` (`float, optional`)
  Speed multiplier.

##### 返回

- `bytes`
  Processed audio bytes.

## TurnDetector

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
@model_type(aliases=['turn_detector'])
class TurnDetector(ABC)
```

Abstract interface for turn-taking detectors.

### 方法

#### __init__

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def __init__(self) -> None
```

#### listening

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def listening(self) -> bool
```

Return whether the detector is currently listening for user turns.

##### 返回

- `bool`
  Current listening state.

#### listening

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def listening(self, value: bool) -> None
```

Update the listening state.

##### 参数

- `value` (`bool`)
  New listening state.

#### listening_lock

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def listening_lock(self, is_async: bool = True)
```

Return the lock guarding listening state changes.

##### 参数

- `is_async` (`bool, optional`)
  Whether to return the async lock instead of the threading lock.

##### 返回

- `asyncio.Lock | threading.Lock`
  Lock object matching the requested concurrency model.

#### detect

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def detect(self, audio: Optional[bytes] = None, text: Optional[str] = None, assistant_text: Optional[str] = None, speech_start: bool = False, speech_pause: Optional[bool] = None) -> TurnDetectionResult
```

Detect conversational turn state from audio and/or text context.

##### 参数

- `audio` (`bytes | None, optional`)
  Current PCM 16-bit mono audio frame at 16 kHz.
- `text` (`str | None, optional`)
  ASR text for the current turn.
- `assistant_text` (`str | None, optional`)
  Cumulative AI response text confirmed as played to the user.
  ``None`` means that this call carries no assistant response update.
- `speech_start` (`bool, optional`)
  Whether VAD has just detected the start of speech. This may be
  provided without ``audio``, ``text``, or ``assistant_text``.
- `speech_pause` (`bool | None, optional`)
  Whether the user appears to have paused speaking. This is typically
  provided together with ``text``.

##### 返回

- `TurnDetectionResult`
  Turn-detection decision for the current input.

#### async_detect

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
async def async_detect(self, audio: Optional[bytes] = None, text: Optional[str] = None, assistant_text: Optional[str] = None, speech_start: bool = False, speech_pause: Optional[bool] = None) -> TurnDetectionResult
```

Asynchronously detect conversational turn state.

##### 参数

- `audio` (`bytes | None, optional`)
  Current PCM 16-bit mono audio frame at 16 kHz.
- `text` (`str | None, optional`)
  ASR text for the current turn.
- `assistant_text` (`str | None, optional`)
  Cumulative AI response text confirmed as played to the user.
  ``None`` means that this call carries no assistant response update.
- `speech_start` (`bool, optional`)
  Whether VAD has just detected the start of speech. This may be
  provided without ``audio``, ``text``, or ``assistant_text``.
- `speech_pause` (`bool | None, optional`)
  Whether the user appears to have paused speaking.

##### 返回

- `TurnDetectionResult`
  Turn-detection decision for the current input.

#### clone

_定义于 [`xtalk.models.turn_detector.interfaces`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/turn_detector/interfaces.py)。_

```python
def clone(self) -> 'TurnDetector'
```

Clone the turn detector for a new session.

##### 返回

- `TurnDetector`
  Session-safe clone.
