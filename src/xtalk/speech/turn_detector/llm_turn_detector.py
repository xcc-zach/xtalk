from ..interfaces import (
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


# TODO: re-implement
class LLMTurnDetector(TurnDetector):
    STOP_SPEAKING_PROMPT = """You are a classifier. Given a user utterance that is a *text transcription* from a live speech conversation (ASR output), decide whether it is:

* **backchannel**: short acknowledgements / continuer signals that do **not** take the floor or change the topic.
* **wait**: explicit request for the assistant to pause/hold or “one moment”.
* **interrupt**: user is cutting in to take the floor, correct/redirect, ask a question, or stop/modify what the assistant is saying.

**Output exactly one label from:** `["backchannel", "wait", "interrupt"]`

## Decision rules (use the first matching rule)

### 1) **wait**

Label as **wait** if the utterance explicitly asks to pause or hold, e.g.:

* “wait”, “hold on”, “one sec/second”, “a moment”, “give me a second”
* “stop for a bit”, “pause”, “hang on”, “let me think”
* “等一下/等等/稍等/先别说/先停一下/给我一分钟”

### 2) **backchannel**

Label as **backchannel** if it's primarily a continuer/acknowledgement and **does not** introduce new content or a request, e.g.:

* “uh-huh”, “mm-hmm”, “yeah”, “yep”, “ok”, “right”, “I see”, “got it”
* “嗯/啊哈/对/好/行/懂了/明白/是的”
* brief laughter tokens like “haha” when used as acknowledgement

**Heuristic:** typically ≤ 3-4 words, no question, no directive verbs (stop/pause/repeat), no new task/topic.

### 3) **interrupt**

Label as **interrupt** otherwise, including:

* asking a question mid-stream: “but why...?”, “what about...?”
* correcting/redirecting: “no, I meant...”, “actually...”, “not that...”
* stopping/modifying assistant speech without asking to “wait”: “stop”, “don't say that”, “let's switch”
* adding substantial info: names, numbers, constraints, new topic, instructions

## Tie-breakers / ASR noise handling

* If it contains both acknowledgement and a new request/topic (e.g., “yeah but...”, “ok so do X”), choose **interrupt**.
* If it contains “wait/hold on/等一下” anywhere and it's a genuine pause request, choose **wait** even if there's also an acknowledgement.
* If it's unclear but longer than a simple acknowledgement or includes content words beyond agreement, choose **interrupt**.

## Output format

Return **only** the label string.
"""
    START_GENERATION_PROMPT = """You are a real-time speech-UI classifier. Given an ASR transcript of what the user just said (may be partial, noisy, or cut off), classify whether the user's utterance is:

* **incomplete**: the user is still speaking / the thought is unfinished, or ASR looks truncated.
* **complete**: the user finished a coherent utterance (question/command/statement) and is yielding the floor.
* **wait**: the user explicitly asks the assistant to pause/hold while they think or do something.

**Output exactly one label from:** `["incomplete", "complete", "wait"]`

## Decision rules (apply in order)

### 1) **wait**

Choose **wait** only if the utterance is a **pure pause/hold request** — i.e., the transcript contains a pause/hold phrase and **does NOT contain any additional meaningful/semantic content after it**.

**Pause/hold keywords examples:**
* English: “wait”, “hold on”, “hang on”, “one sec/second”, “a moment”, “give me a second”, “pause”, “let me think”
* Chinese: “等一下/等等/稍等/先别说/先停一下/给我一分钟/我想一下”

#### Semantic continuation override (CRITICAL)

If a pause/hold phrase appears but the user continues with **any concrete semantic information afterwards**, **do NOT output `wait`**. Instead, proceed to rules **2) incomplete** and **3) complete** to classify the utterance as a whole.

**What counts as “concrete semantic information” (any of these):**
* A **question** or interrogative cue: “怎么/为什么/能不能/what/how/why…”
* A **command/request**: “帮我…/给我…/打开…/explain…/do…”
* A **statement with content words** (not just fillers), e.g. mentions objects/actions/names/numbers:
  * “等一下，我要找一下文件”
  * “等一下，刚才那个接口的报错是 403”
  * “等一下，我的意思是你先改这段逻辑”
* Any continuation markers with real content: “然后…/其实…/我的意思是…/就是说…/I mean…/actually…”

**What does NOT count as semantic continuation (still `wait`-eligible):**
Only short fillers/politeness/acknowledgements after the pause phrase, such as:
* Chinese: “嗯/啊/哦/好/行/可以/谢谢/让我想想(仅此一句且没有别的内容)”
* English: “uh/um/okay/yeah/sure/thanks/let me think (and nothing else)”

**Tie-break inside Rule 1:**
* If pause/hold phrase is present **and** there is **no semantic continuation** → output **`wait`**.
* If pause/hold phrase is present **but** there **is semantic continuation** → **NOT `wait`**, go to rules 2–3.

### 2) **incomplete**

Choose **incomplete** if **any** of these are true:

**A. Truncation / cut-off indicators**
* Transcript ends with unfinished connectors: “and”, “so”, “but”, “because”, “if”, “then”, “like”, “which”, “that...”
* Ends with filler or restart signals: “uh”, “um”, “er”, “I mean”, “well...”, “你知道”, “就是”, “然后...”
* Ends with a dangling preposition/article: “to”, “with”, “for”, “a/an/the”, “的/了/在/把” (when clearly hanging)

**B. Mid-thought structure**
* Starts a clause but doesn't complete it: “I want to...”, “Can you...”, “Let's...”, “Could we...”, “我想.../能不能.../我们...”
* Contains self-correction or continuation cues without finishing: “no—”, “actually—”, “wait— I mean—”, “不是...我是说...”

**C. ASR partialness signs**
* Very short fragment that looks like a partial start (1-3 content words) and not a full intent, e.g. “so the...”, “about the...”, “那个...”, “就是...”
* Strong repetition/restarts: “I I I...”, “we we...”, “我我我...”

### 3) **complete**

Choose **complete** if neither of the above applies and the utterance forms a complete communicative unit, e.g.:

* A full question: ends naturally or with “?”, has an interrogative (“what/why/how/能不能/怎么”)
* A full command/request: “Open X”, “Explain Y”, “帮我把...”
* A complete statement: “I'm done”, “That's fine”, “We should do A first”
* Even if short, it clearly conveys a finished intent: “yes”, “no”, “okay”, “got it”, “不用了”, “可以”

## Tie-breakers

* If the transcript ends with a period-like finality (or clear completion) vs. a dangling connector, prefer **complete**.
* If it includes both acknowledgement and a new clause starter (e.g., “yeah but...”, “好的然后...”), choose **incomplete** unless it clearly finishes.
* When uncertain, prefer **incomplete** (safer for turn-taking) unless it's an explicit pause request (**wait**).

## Output format

Return only the label string with no extra text.
"""
    CHECK_COMPLETION_PROMPT = """You are a real-time speech-UI classifier. Given an ASR transcript of what the user just said (may be partial, noisy, or cut off), classify whether the user's utterance is:

* **incomplete**: the user is still speaking / the thought is unfinished, or ASR looks truncated.
* **complete**: the user finished a coherent utterance (question/command/statement) and is yielding the floor.

### 1) **incomplete**

Choose **incomplete** if **any** of these are true:

**A. Truncation / cut-off indicators**

* Transcript ends with unfinished connectors: “and”, “so”, “but”, “because”, “if”, “then”, “like”, “which”, “that...”
* Ends with filler or restart signals: “uh”, “um”, “er”, “I mean”, “well...”, “你知道”, “就是”, “然后...”
* Ends with a dangling preposition/article: “to”, “with”, “for”, “a/an/the”, “的/了/在/把” (when clearly hanging)

**B. Mid-thought structure**

* Starts a clause but doesn't complete it: “I want to...”, “Can you...”, “Let's...”, “Could we...”, “我想.../能不能.../我们...”
* Contains self-correction or continuation cues without finishing: “no—”, “actually—”, “wait— I mean—”, “不是...我是说...”

**C. ASR partialness signs**

* Very short fragment that looks like a partial start (1-3 content words) and not a full intent, e.g. “so the...”, “about the...”, “那个...”, “就是...”
* Strong repetition/restarts: “I I I...”, “we we...”, “我我我...”

### 2) **complete**

Choose **complete** if neither of the above applies and the utterance forms a complete communicative unit, e.g.:

* A full question: ends naturally or with “?”, has an interrogative (“what/why/how/能不能/怎么”)
* A full command/request: “Open X”, “Explain Y”, “帮我把...”
* A complete statement: “I'm done”, “That's fine”, “We should do A first”
* Even if short, it clearly conveys a finished intent: “yes”, “no”, “okay”, “got it”, “不用了”, “可以”

## Tie-breakers

* If the transcript ends with a period-like finality (or clear completion) vs. a dangling connector, prefer **complete**.
* If it includes both acknowledgement and a new clause starter (e.g., “yeah but...”, “好的然后...”), choose **incomplete** unless it clearly finishes.

## Output format

Return only the label string with no extra text.
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

    def clone(self) -> "TurnDetector":
        return LLMTurnDetector(self._model)

    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult | list[TurnDetectionResult]:
        return asyncio.run(self.async_detect(audio, text, speech_pause))

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult | list[TurnDetectionResult]:
        if text == None:
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.IDLE,
            )
        async with self.listening_lock():
            if self.listening:
                messages = [
                    SystemMessage(content=self.START_GENERATION_PROMPT),
                    HumanMessage(content=text),
                ]
                response = (await self._model.ainvoke(messages)).content
                if speech_pause and "complete" in response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.START_GENERATION,
                        semantic=TurnDetectionSemantic.COMPLETE,
                    )
                if "incomplete" in response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.DO_NOTHING,
                        semantic=TurnDetectionSemantic.INCOMPLETE,
                    )
                if "wait" in response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.DO_NOTHING,
                        semantic=TurnDetectionSemantic.WAIT,
                    )
            else:
                messages = [
                    SystemMessage(content=self.STOP_SPEAKING_PROMPT),
                    HumanMessage(content=text),
                ]
                response = (await self._model.ainvoke(messages)).content
                if "backchannel" in response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.DO_NOTHING,
                        semantic=TurnDetectionSemantic.BACKCHANNEL,
                    )
                if "wait" in response.lower():
                    return TurnDetectionResult(
                        action=TurnDetectionAction.STOP_SPEAKING,
                        semantic=TurnDetectionSemantic.WAIT,
                    )
                if "interrupt" in response.lower():
                    result = TurnDetectionResult(
                        action=TurnDetectionAction.STOP_SPEAKING,
                        semantic=TurnDetectionSemantic.INCOMPLETE,
                    )

                    # Need to additional check for start generation if meet speech paused (indicating a potential end of speech)
                    if speech_pause:
                        messages = [
                            SystemMessage(content=self.CHECK_COMPLETION_PROMPT),
                            HumanMessage(content=text),
                        ]
                        response = (await self._model.ainvoke(messages)).content
                        if "complete" in response.lower():
                            result = [
                                TurnDetectionResult(
                                    action=TurnDetectionAction.STOP_SPEAKING,
                                    semantic=TurnDetectionSemantic.COMPLETE,
                                ),
                                TurnDetectionResult(
                                    action=TurnDetectionAction.START_GENERATION,
                                    semantic=TurnDetectionSemantic.COMPLETE,
                                ),
                            ]
                    return result
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.IDLE,
            )
