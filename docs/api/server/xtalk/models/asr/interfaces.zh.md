<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.asr.interfaces

## ASR

```python
@model_type(aliases=['asr'])
class ASR(ABC)
```

Abstract interface for automatic speech recognition.

### 方法

#### recognize

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

```python
def stream_chunk_bytes_hint(self) -> int | None
```

Return the preferred streaming chunk size.

##### 返回

- `int | None`
  Recommended byte count for each chunk passed to
  ``recognize_stream``, or ``None`` when no preference is provided.

#### reset

```python
def reset(self) -> None
```

Reset internal recognition state.

#### clone

```python
def clone(self) -> 'ASR'
```

Clone the ASR instance for a new session.

##### 返回

- `ASR`
  Clone with shared weights and independent runtime state.

#### async_recognize

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
