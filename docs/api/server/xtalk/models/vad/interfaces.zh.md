<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.vad.interfaces

## VAD

```python
@model_type(aliases=['vad'])
class VAD(ABC)
```

Abstract base class for voice activity detection engines.

### 方法

#### set_threshold

```python
def set_threshold(self, *, threshold: float) -> None
```

Configure a model-supported speech probability threshold.

##### 参数

- `threshold` (`float`)
  Probability threshold between zero and one.

##### 抛出

- `NotImplementedError`
  This boolean-only model cannot change its default threshold.

#### is_speech

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

```python
def reset(self) -> None
```

Reset session-local VAD state and release external resources.

#### detect_frames

```python
def detect_frames(self, frames: list[bytes]) -> list[bool]
```

Detect consecutive frames while preserving stream state and order.

##### 参数

- `frames` (`list[bytes]`)
  Ordered PCM16 mono frames using the single-frame audio contract.

##### 返回

- `list[bool]`
  One speech decision per input frame, in input order. Empty input
  returns an empty list without calling the model.

##### 说明

The fallback calls ``is_speech`` sequentially. Native batch clients
may override this method without changing the single-frame interface.

#### async_detect_frames

```python
async def async_detect_frames(self, frames: list[bytes]) -> list[bool]
```

Asynchronously detect ordered frames using sequential fallback calls.

##### 参数

- `frames` (`list[bytes]`)
  Ordered PCM16 mono frames using the single-frame audio contract.

##### 返回

- `list[bool]`
  One speech decision per frame, in input order; empty input
  returns an empty list without calling the model.

##### 说明

Calls ``async_is_speech`` sequentially to retain existing async
implementations and recurrent state. Native async batch clients may
override this method. Overriding only the synchronous batch method
does not change this asynchronous fallback.

#### clone

```python
def clone(self) -> 'VAD'
```

Clone the VAD instance for a new session.

##### 返回

- `VAD`
  Clone with shared weights and independent runtime state.
