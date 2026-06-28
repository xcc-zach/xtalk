```python
class ASR(ABC):
    """Abstract interface for automatic speech recognition."""

    def _get_mock_recognizer(self) -> MockStreamRecognizer:
        """Create the fallback streaming recognizer on first use.

        Returns
        -------
        MockStreamRecognizer
            Helper that adapts one-shot ASR implementations to streaming calls.
        """
        recognizer = getattr(self, "_mock_recognizer", None)
        if recognizer is None:
            recognizer = MockStreamRecognizer(
                self.async_recognize,
                window_size=10,
            )
            setattr(self, "_mock_recognizer", recognizer)
        return recognizer

    @abstractmethod
    def recognize(self, audio: bytes) -> str:
        """Recognize a full audio buffer.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        str
            Recognized text.
        """
        pass

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Recognize audio incrementally in streaming mode.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the caller is asking the ASR to treat the current point as
            a temporary boundary and optionally flush any tail audio that would
            otherwise remain buffered. This is only a decoding hint. It does
            not mean the streaming state must be reset, and previously
            recognized text for the session must be preserved so later audio
            can continue from the accumulated result.
        chat_history : str | None, optional
            Serialized chat history for the current session, excluding the
            in-progress turn when unavailable.

        Returns
        -------
        str
            Current recognition result.
        """
        del chat_history
        recognizer = self._get_mock_recognizer()

        if not audio:
            return recognizer.recognized_text

        return recognizer.recognize(audio, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        """Return the preferred streaming chunk size.

        Returns
        -------
        int | None
            Recommended byte count for each chunk passed to
            ``recognize_stream``, or ``None`` when no preference is provided.
        """
        return None

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recognition state."""
        pass

    @abstractmethod
    def clone(self) -> "ASR":
        """Clone the ASR instance for a new session.

        Returns
        -------
        ASR
            Clone with shared weights and independent runtime state.
        """
        pass

    async def async_recognize(self, audio: bytes) -> str:
        """Asynchronously recognize a full audio buffer.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono audio bytes.

        Returns
        -------
        str
            Recognized text.
        """
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(None, self.recognize, audio)
        return result

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Asynchronously recognize incremental audio input.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the caller is asking the ASR to treat the current point as
            a temporary boundary and optionally flush any tail audio that would
            otherwise remain buffered. This is only a decoding hint. It does
            not mean the streaming state must be reset, and previously
            recognized text for the session must be preserved so later audio
            can continue from the accumulated result.
        chat_history : str | None, optional
            Serialized chat history for the current session, excluding the
            in-progress turn when unavailable.

        Returns
        -------
        str
            Current recognition result.
        """
        loop = asyncio.get_running_loop()
        result: str = await loop.run_in_executor(
            None,
            partial(
                self.recognize_stream,
                audio,
                is_final=is_final,
                chat_history=chat_history,
            ),
        )
        return result
```

## Best Practice for Implementing `recognize`

The framework actually calls `async_recognize_stream`. Therefore, the best practice is to implement `async_recognize_stream` first, and then implement `recognize` like this:

```python
import asyncio

def recognize(self, audio: bytes) -> str:
    return self._run_coro(
        self.async_recognize_stream(audio, is_final=True, chat_history=None)
    )

def _run_coro(self, coro: "asyncio.Future[str]") -> str:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
```

## Parameters and Return Value of `async_recognize_stream`

Each call should return the complete recognized text accumulated from the last `reset` up to the current moment.

`audio` is the newly arrived audio for this call. `is_final` means the current recognition point has reached an acoustic pause, such as the end of one sentence. Developers may use it for special handling, but it does not mean the accumulated recognized text should be cleared. Do not confuse it with `reset`. `chat_history` is an experimental field containing serialized chat history.

## Pseudo-streaming Integration for Non-streaming / Offline ASR Models

You can use `MockStreamRecognizer` from [`src/xtalk/models/audio_utils.py`](https://github.com/xcc-zach/xtalk/blob/main/src/xtalk/models/audio_utils.py) and build pseudo-streaming behavior on top of `async_recognize`.

## Meaning of `stream_chunk_bytes_hint`

It indicates how many bytes of audio should trigger one streaming recognition step. By default, recognition is triggered frequently.

## `clone` and `reset`

See [Semantics of `clone()` and `reset()` on Model Objects](model_clone_reset.md).
