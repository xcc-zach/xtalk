# Turn Detector Interface

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
    def listening(self) -> bool: ...

    @listening.setter
    def listening(self, value: bool) -> None: ...

    def listening_lock(self, is_async: bool = True): ...

    @abstractmethod
    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult: ...

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        assistant_text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
        assistant_interrupted: bool = False,
    ) -> TurnDetectionResult: ...

    @abstractmethod
    def clone(self) -> "TurnDetector": ...
```

`TurnDetector` judges turn-taking from audio, ASR text, played assistant text, and VAD side signals. The service layer calls `async_detect`; the default implementation runs `detect` in a thread pool.

## `async_detect`

Return the turn-taking judgment at the current moment.

### Parameters

- `audio`: the current audio frame, PCM 16-bit, mono, 16 kHz bytes.
- `text`: the ASR text accumulated so far for the current turn.
- `assistant_text`: the cumulative assistant response text confirmed as played to the user. `None` means the call carries no assistant response update. The value is a cumulative prefix; a detector that stores it replaces the stored value, and repeated identical values leave the state unchanged.
- `speech_start`: `True` when VAD has just detected speech start.
- `speech_pause`: `True` when the user may currently be pausing; usually passed together with `text`. It is a hint that a pause is occurring now, and carries no reset semantics.
- `assistant_interrupted`: `True` when assistant speech has just been interrupted before normal playback completion.

### Returns

`TurnDetectionResult` with three fields:

- `action`: the action the service layer executes immediately.
    - `DO_NOTHING`: trigger no extra action.
    - `STOP_SPEAKING`: interrupt the ongoing spoken output; used when the user interrupts the system while it is speaking.
    - `START_GENERATION`: start generating a response; used when the user is determined to have finished speaking.
- `semantic`: the detector's semantic interpretation of the conversation state. `action` and `semantic` are related and must be consistent; `action` alone determines the immediate service-layer behavior.
    - `IDLE`: no clear turn-advancing signal.
    - `INCOMPLETE`: the user is still continuing the current turn.
    - `COMPLETE`: the user's current input is semantically complete.
    - `WAIT`: the user explicitly expresses a waiting intent.
    - `BACKCHANNEL`: the input is a short acknowledgment and does not count as turn completion.
    - `SHOULD_BACKCHANNEL`: the system may produce a backchannel.
- `vad_result`: optional; used only when the detector also proxies VAD state. `SPEECH` means speaking, `SILENCE` means silence. The service layer uses it to trigger VAD behavior when no independent VAD is configured and ignores it when a frontend or backend VAD exists.

## `listening`

A built-in state with its locks, distinguishing whether the detector is listening for the user to finish input or listening for an interruption of system output.

- `listening = True`: the system waits for the user to finish; the detector decides when to emit `START_GENERATION`.
- `listening = False`: the system is playing output; the detector decides whether user input triggers `STOP_SPEAKING`.

`TurnDetectorManager` sets `listening` to `False` when TTS playback starts and restores it to `True` when playback finishes or is interrupted.

## `clone`

Session state held by a detector belongs to one instance only. See [Model `clone()` and `reset()`](../model_clone_reset.md).

## Next Steps

<div class="grid cards next-step-cards" markdown>

-   **[Input combinations and detection paths →](../../explanation/turn_detector.md)** · Explanation

-   **[Register a new model →](../../tutorial/introduce_a_new_model.md)** · Tutorial

-   **[Wrap an async detector in `detect` →](../../how_to/adapt_async_models.md)** · How-to

</div>
