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

## `recognize`实现的最佳实践

框架中实际调用`async_recognize_stream`，因此实现时最佳实践是首先实现`async_recognize_stream`，然后采用如下方式实现`recognize`：
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

## `async_recognize_stream`入参与返回值说明

每次调用应返回从上次`reset`后到目前为止的完整识别文本。

`audio`每轮输入为新到来的音频；`is_final`意味着当前识别出现声学停顿，例如一句话结束，开发者可据此做特殊处理，并不意味着要清空积累的识别的文本，注意区分`reset`；`chat_history`为实验性字段，为聊天历史字符串。

## 非流式/离线语音识别模型的伪流式接入

可使用`src/xtalk/speech/utils.py`中的`MockStreamRecognizer`，调用`async_recognize`实现伪流式效果。

## `stream_chunk_bytes_hint`语义

指出每隔多少字节音频要触发一次流式识别；默认频繁触发。

## `clone`与`reset`

请参阅[模型对象的 `clone()` 与 `reset()` 语义](model_clone_reset.zh.md)。
