```python
class TurnDetectionAction(Enum):
    DO_NOTHING = 1
    STOP_SPEAKING = 2
    START_GENERATION = 3


class TurnDetectionSemantic(Enum):
    IDLE = "idle"
    INCOMPLETE = "incomplete"
    COMPLETE = "complete"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    SHOULD_BACKCHANNEL = "should_backchannel"


class TurnVADResult(Enum):
    SPEECH = 1
    SILENCE = 2


@dataclass(frozen=True)
class TurnDetectionResult:
    action: TurnDetectionAction
    semantic: TurnDetectionSemantic
    vad_result: TurnVADResult | None = None


class TurnDetector(ABC):
    """Abstract interface for turn-taking detectors."""

    @property
    def listening(self) -> bool:
        ...

    @listening.setter
    def listening(self, value: bool) -> None:
        ...

    def listening_lock(self, is_async: bool = True):
        ...

    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        ...

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        ...

    @abstractmethod
    def clone(self) -> "TurnDetector":
        ...
```

## Best Practice for Implementing `detect`

The framework actually calls `async_detect`. Therefore, the best practice when implementing a new turn detector is to implement `async_detect` first, and then wrap it synchronously in `detect`.

```python
import asyncio

def detect(
    self,
    audio: Optional[bytes] = None,
    text: Optional[str] = None,
    speech_start: bool = False,
    speech_pause: Optional[bool] = None,
) -> TurnDetectionResult:
    return asyncio.run(
        self.async_detect(audio, text, speech_start, speech_pause)
    )
```

If the underlying implementation is already synchronous, you may also implement `detect` directly and reuse the default `async_detect` wrapper from the base class.

## `async_detect`

`TurnDetector` can consume audio signals, ASR text, and VAD side signals at the same time. Each call should return a `TurnDetectionResult` representing the turn-taking judgment at the current moment.

### Input Parameters

- `audio`: The current audio frame, in PCM 16-bit, mono, 16 kHz bytes.
- `text`: The ASR text accumulated so far for the current turn.
- `speech_start`: A signal passed in when VAD has just detected speech start.
- `speech_pause`: A signal passed in when the user may currently be pausing, usually used together with `text`.

Typical combinations of these inputs are:

- Only `audio`: pure audio-based detection path
- Only `text` and `speech_pause`: text-semantic detection path
- Only `speech_start=True`: notify the detector that the current speaking turn has started

The two representative implementations in the current repository correspond to two different paths:

- `SoulxDuplug`: primarily audio-based, with fallback on text pause signals
- `LLMTurnDetector`: primarily text-semantic, mainly relying on `text` and `speech_pause`

### Return Value

The return value is `TurnDetectionResult`, consisting of three parts:

- `action`: the action the service layer should execute immediately
- `semantic`: the semantic interpretation of the current conversation state
- `vad_result`: an optional VAD result, used only when the detector also proxies VAD state

### Meaning of `TurnDetectionAction`

- `DO_NOTHING`: do not trigger any extra action at the moment
- `STOP_SPEAKING`: the system should interrupt its ongoing spoken output
- `START_GENERATION`: the system should start generating a response

In practice:

- `STOP_SPEAKING` is generally used when the user interrupts the system while it is speaking
- `START_GENERATION` is generally used when the user is determined to have finished speaking and the system can start answering

### Meaning of `TurnDetectionSemantic`

- `IDLE`: there is no clear turn-advancing signal at the moment
- `INCOMPLETE`: the user is still continuing the current turn and has not finished
- `COMPLETE`: the user's current input is semantically complete
- `WAIT`: the user explicitly expresses a waiting intent
- `BACKCHANNEL`: the input is a short acknowledgment and should not count as turn completion
- `SHOULD_BACKCHANNEL`: the current state suggests that the system may produce a backchannel

`semantic` is mainly used to express the detector's semantic judgment, while `action` determines the immediate service-layer behavior. They are related, but not equivalent.

### Meaning of `vad_result`

`vad_result` is optional and is only used when the turn detector also serves as a VAD proxy.

- `TurnVADResult.SPEECH`: the current state is speaking
- `TurnVADResult.SILENCE`: the current state is silence

When the pipeline does not have an independent VAD configured, the system uses this field to trigger VAD behavior.
If a frontend or backend VAD is configured, this field is ignored.

## Meaning of `listening`

The `TurnDetector` base class includes a built-in `listening` state and its locks. This state is used to distinguish whether the detector is currently "listening for the user to finish input" or "listening for whether the user interrupts system output".

Common conventions are:

- `listening = True`: the system is waiting for the user to finish, and the detector should decide when to emit `START_GENERATION`
- `listening = False`: the system is playing output, and the detector should decide whether the user's input should trigger `STOP_SPEAKING`

In the service layer, `TurnDetectorManager` sets `listening` to `False` when TTS starts playback, and restores it to `True` when playback finishes or is interrupted.

## Implementation Suggestions

- Whenever returning a result, ensure that `action` and `semantic` are semantically consistent.
- If the detector maintains session state internally, that state must belong only to the current instance and must not be shared across sessions.
- If the implementation uses `speech_pause`, it should treat it as a hint that "a pause is occurring now", not as a reset signal after the turn ends.

## `clone`

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).
