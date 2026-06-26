from .interfaces import (
    TurnDetector,
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
)
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Optional
import asyncio
import time
from ..registry import model


@model
class LLMTurnDetector(TurnDetector):
    INCOMPLETE_MAX_LAST_SECS = 7
    STOP_SPEAKING_PROMPT = """Classify the user's input. Return exactly one JSON object with keys `reason` and `judgement`.

`reason` must be a short string explaining why you reached the judgement.
`judgement` must be exactly `backchannel` or `interrupt`.

Apply the following rules in order:

If the user's input is a backchannel expression, such as “mm-hmm” or “okay,” or "嗯嗯" or "对对" then output `backchannel`. Backchannel can only be very short, and semantically meaningless.

If the user's input is semantically complete, or the intent is to interrupt someone else's response, then output `interrupt`.

"""
    START_GENERATION_PROMPT = """Classify the user's input. Return exactly one JSON object with keys `reason` and `judgement`.

`reason` must be a short string explaining why you reached the judgement.
`judgement` must be exactly `finished` or `incomplete` or `wait`.

Apply the following rules in order:

As long as the user's input is semantically complete or is a question, output `finished`, even if there is hesitation words in it. Most inputs should fall in this category.

If the user's input explicitly indicates "wait" or "等一下", then output `wait`.

If the user's input is semantically incomplete, as if they stopped halfway through speaking, for example ending with hesitation words or fillers, then output `incomplete`.

If you find the input abnormal, for example containing ASR misrecognized characters, also output finished.

"""

    def __init__(self, model: dict | BaseChatModel) -> None:
        super().__init__()
        if isinstance(model, dict):
            model = ChatOpenAI(**model)
        self._model = model
        # If AI is listening, try to determine whether to start generation and toggle state;
        # else determine whether to stop speaking and toggle state
        self._listening = True
        # FIXED: lock listening
        self._listening_lock = asyncio.Lock()
        self._text_prefix = ""
        self._incomplete_started_at = None

    def clone(self) -> "TurnDetector":
        return LLMTurnDetector(self._model)

    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        return asyncio.run(self.async_detect(audio, text, speech_start, speech_pause))

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        del audio
        if speech_start:
            self._reset_incomplete_state()
        if text == None:
            # Avoid infinite incomplete first
            if self._incomplete_started_at != None:
                elapsed = time.monotonic() - self._incomplete_started_at
                if elapsed > self.INCOMPLETE_MAX_LAST_SECS:
                    self._reset_incomplete_state()
                    return TurnDetectionResult(
                        action=TurnDetectionAction.START_GENERATION,
                        semantic=TurnDetectionSemantic.COMPLETE,
                    )
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.IDLE,
            )
        async with self.listening_lock():
            text_without_prefix = self._remove_prefix(text)
            if self.listening:
                if speech_pause:
                    # Use full text for wait detection, and use text without prefix for generation completion detection, to avoid misclassification caused by redundant info
                    response, no_prefix_response = await self._gather_two_responses(
                        self.START_GENERATION_PROMPT, text, text_without_prefix
                    )
                    if "finished" in no_prefix_response.lower():
                        self._record_prefix("")
                        self._reset_incomplete_state()
                        return TurnDetectionResult(
                            action=TurnDetectionAction.START_GENERATION,
                            semantic=TurnDetectionSemantic.COMPLETE,
                        )
                    if "wait" in response.lower():
                        self._record_prefix("")
                        self._reset_incomplete_state()
                        return TurnDetectionResult(
                            action=TurnDetectionAction.START_GENERATION,
                            semantic=TurnDetectionSemantic.WAIT,
                        )
                    self._set_incomplete_state(time.monotonic())
                    return TurnDetectionResult(
                        action=TurnDetectionAction.DO_NOTHING,
                        semantic=TurnDetectionSemantic.INCOMPLETE,
                    )
            else:
                response, no_prefix_response = await self._gather_two_responses(
                    self.STOP_SPEAKING_PROMPT, text, text_without_prefix
                )
                if "backchannel" in response.lower():
                    self._record_prefix(text)
                    return TurnDetectionResult(
                        action=TurnDetectionAction.DO_NOTHING,
                        semantic=TurnDetectionSemantic.BACKCHANNEL,
                    )
                if "interrupt" in no_prefix_response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.STOP_SPEAKING,
                        semantic=TurnDetectionSemantic.INCOMPLETE,
                    )
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.IDLE,
            )

    # Remove prefix to avoid misclassfication
    def _remove_prefix(self, text: str) -> str:
        prefix = self._text_prefix
        common_prefix_len = 0
        for curr_char, prefix_char in zip(text, prefix):
            if curr_char != prefix_char:
                break
            common_prefix_len += 1
        return text[common_prefix_len:]

    def _record_prefix(self, prefix: str):
        self._text_prefix = prefix

    async def _gather_two_responses(
        self, system_message: str, user_text_1: str, user_text_2: str
    ) -> tuple[str, str]:
        messages_1 = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_text_1),
        ]
        messages_2 = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_text_2),
        ]
        response_1, response_2 = await asyncio.gather(
            self._model.ainvoke(messages_1),
            self._model.ainvoke(messages_2),
        )
        return response_1.content, response_2.content

    def _reset_incomplete_state(self):
        self._incomplete_started_at = None

    def _set_incomplete_state(self, start_time: float):
        self._incomplete_started_at = start_time
