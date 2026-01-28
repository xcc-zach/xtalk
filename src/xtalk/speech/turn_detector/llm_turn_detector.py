from ..interfaces import TurnDetector, TurnDetectionAction
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage


# TODO: re-implement
class LLMTurnDetector(TurnDetector):
    SYSTEM_PROMPT = """Classify user input as ["complete", "incomplete", "backchannel"].

You are a classifier in a spoken dialogue system. Your task is to label each single user message based on whether it requires a model response.

### Labeling Rules
- **complete**: The message contains a clear question, instruction, or statement that should receive a meaningful model response immediately.
- **incomplete**: The message is truncated, ambiguous, missing critical information, or is part of ongoing input where the model should wait instead of responding now.
- **backchannel**: The message is a short feedback signal used only to maintain conversational flow (e.g., "yeah", "uh-huh", "right"), carries no primary intent, and does not require a model response.

### Constraints
- Classify the message itself only, not the surrounding context.
- Output **one label only**.
- Do not generate any explanation, only return the label string.
- Be robust to casual speech, fillers, and interruptions.
- Treat acknowledgements without actionable intent as backchannel.
- Treat actionable or reply-worthy content as complete, even if it does not finish a multi-step task.

### Examples
User: "How do I rename a git branch?" → complete  
User: "I was thinking the architecture could maybe use…" → incomplete  
User: "uh-huh" → backchannel  
User: "Deploy it to Windows, Linux, and macOS." → complete  
User: "vLLM needs to load the model from…" *(stops mid-sentence)* → incomplete  
User: "yes, exactly" *(no further action implied)* → backchannel
"""

    def __init__(self, model: dict | BaseChatModel) -> None:
        if isinstance(model, dict):
            model = ChatOpenAI(**model)
        self._model = model

    def clone(self) -> "TurnDetector":
        return LLMTurnDetector(self._model)

    def detect(self, audio=None, text=None) -> TurnDetectionAction:
        if not text:
            raise RuntimeError("Text for LLMTurnDetector should not be empty")
        messages: List = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(content=text),
        ]
        response = self._model.invoke(messages).content
        if "backchannel" in response.lower():
            return TurnDetectionAction.BACKCHANNEL
        elif "incomplete" in response.lower():
            return TurnDetectionAction.INCOMPLETE
        else:
            return TurnDetectionAction.COMPLETE
