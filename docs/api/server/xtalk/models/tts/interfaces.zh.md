<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.tts.interfaces

## TTS

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

```python
def clone(self) -> 'TTS'
```

Clone the TTS engine for a new session.

##### 返回

- `TTS`
  Session-safe clone.

#### set_voice

```python
def set_voice(self, voice_names: list[str]) -> None
```

Update the active voice selection.

##### 参数

- `voice_names` (`list[str]`)
  One or more voice names understood by the implementation.

#### set_emotion

```python
def set_emotion(self, emotion: str | list[float]) -> None
```

Update the active synthesis emotion.

##### 参数

- `emotion` (`str | list[float]`)
  Emotion label or model-specific emotion vector.
