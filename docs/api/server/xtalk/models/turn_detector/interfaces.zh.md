<!-- 此文件由 generate_server_docs.py 自动生成。 -->
# xtalk.models.turn_detector.interfaces

## TurnDetectionAction

```python
class TurnDetectionAction(Enum)
```

Immediate action selected by a turn detector.

### 类字段

- `DO_NOTHING` = `1`
- `STOP_SPEAKING` = `2`
- `START_GENERATION` = `3`

## TurnDetectionSemantic

```python
class TurnDetectionSemantic(Enum)
```

Semantic state selected by a turn detector.

### 类字段

- `IDLE` = `'idle'`
- `INCOMPLETE` = `'incomplete'`
- `COMPLETE` = `'complete'`
- `WAIT` = `'wait'`
- `BACKCHANNEL` = `'backchannel'`
- `SHOULD_BACKCHANNEL` = `'should_backchannel'`

## TurnVADResult

```python
class TurnVADResult(Enum)
```

Optional VAD state produced by a turn detector.

### 类字段

- `SPEECH` = `1`
- `SILENCE` = `2`

## TurnDetectionResult

```python
@dataclass(frozen=True)
class TurnDetectionResult
```

Decision emitted by a turn detector.

### 属性

- `action` (`TurnDetectionAction`)
  Immediate action the service should take.
- `semantic` (`TurnDetectionSemantic`)
  Semantic interpretation of the current conversational state.
- `vad_result` (`TurnVADResult | None`)
  Optional VAD result; only used when VAD is absent

### 类字段

- `action: TurnDetectionAction`
- `semantic: TurnDetectionSemantic`
- `vad_result: TurnVADResult | None` = `None`

## TurnDetector

```python
@model_type(aliases=['turn_detector'])
class TurnDetector(ABC)
```

Abstract interface for turn-taking detectors.

### 方法

#### __init__

```python
def __init__(self) -> None
```

#### listening

```python
def listening(self) -> bool
```

Return whether the detector is currently listening for user turns.

##### 返回

- `bool`
  Current listening state.

#### listening

```python
def listening(self, value: bool) -> None
```

Update the listening state.

##### 参数

- `value` (`bool`)
  New listening state.

#### listening_lock

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

```python
def clone(self) -> 'TurnDetector'
```

Clone the turn detector for a new session.

##### 返回

- `TurnDetector`
  Session-safe clone.
