import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.chat_models.base import BaseChatModel
from langchain.tools import tool
from langchain_core.tools import BaseTool

from xtalk import Xtalk, model
from xtalk.log_utils import mute_other_logging
from xtalk.models.agents.default import DefaultAgent

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

mute_other_logging()

parser = argparse.ArgumentParser(description="Mental Consultant Xtalk Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Mental Consultant Xtalk Server")


MENTAL_CONSULTANT_PROMPT = """
You are a gentle psychologist, casually chatting with a visitor. Follow these rules:

1. Reply in the same language as the user, warm and natural, like a friend, never revealing your role.
2. No parentheses, lists, or Arabic numerals; spell numbers in English words.
3. Keep answers very short, like everyday talk, no explanations.
4. If the user’s words are unclear or garbled, softly say “Could you say that again?”

Silently use the mental_questionnaire tool without letting the user notice. First call needs no argument; afterwards convert the user’s casual answer into an integer between zero and three and pass it. Blend the returned question naturally into chat, as if asking about life. After the scale ends, offer only gentle everyday words, no scores or diagnoses.
Start mental consulting immediately on your first reply.
"""

_QUESTIONS: list[str] = [
    "Over the past two weeks, how often have you felt down, depressed, or overall unable to lift your mood? Answer with an integer between zero and three, where zero means not at all and three means nearly every day.",
    "Over the past two weeks, how much have you lost interest or pleasure in things you usually enjoy (e.g., shows, chatting, games, studying)? Use the same zero-to-three scale.",
    "Over the past two weeks, how often have you had trouble falling asleep, staying asleep, or slept too much so that you never feel rested? Zero means never, three means nearly every day.",
    "Over the past two weeks, how often have you felt drained or low on energy, even when you haven’t done much? Rate it from zero to three.",
    "Over the past two weeks, how often have you had thoughts that life is pointless, that you should disappear, or that hurting yourself might end everything? Zero means not at all, three means nearly every day.",
]


def build_mental_questionnaire_tool() -> BaseTool:
    """Build a stateful per-session questionnaire tool factory."""

    total_questions = len(_QUESTIONS)
    current_index = 0
    total_score = 0
    finished = False
    risk_item_score = 0
    print(
        "[mental_questionnaire] tool created "
        f"(total_questions={total_questions})"
    )

    def _build_recommendation(score: int, risk_score: int) -> str:
        """Generate a short suggestion from questionnaire results."""

        if score <= 2:
            base = (
                "Your responses suggest only minimal emotional distress-perhaps occasional mood swings or stress. "
                "Keep the habits that already help you, such as regular routines, light exercise, or chatting with trusted people."
            )
        elif score <= 5:
            base = (
                "Your answers point to mild distress that may occasionally affect your mood or efficiency. "
                "Give yourself extra care: schedule breaks, talk with friends or family, and plan small pleasant activities. "
                "If the discomfort lingers or intensifies, consider a brief check-in with a professional."
            )
        elif score <= 9:
            base = (
                "You appear to be experiencing a moderate level of distress, which may already be affecting sleep, focus, school, or work. "
                "Please take these signals seriously: try contacting a counselor, campus mental-health center, or psychiatrist soon "
                "so a professional can help assess the situation. "
                "You can also start gentle self-help strategies-maintain routines, move your body, and track mood shifts."
            )
        else:
            base = (
                "Your scores indicate significant emotional strain that likely affects sleep, appetite, motivation, or relationships. "
                "This does not mean something is wrong with you-it shows you have been carrying a heavy load for a long time. "
                "Reach out to a counselor or psychiatrist as soon as possible, bring these results, and plan structured support together."
            )

        if risk_score > 0:
            risk_part = (
                "You also reported a score above zero on the item about life feeling meaningless or having self-harm thoughts. "
                "If you are facing intense urges to harm yourself right now, please do not bear it alone-contact someone you trust and seek medical help together, "
                "or call local emergency numbers (e.g., 110 or 120) or visit the nearest emergency room. "
                "Even if those thoughts appear only occasionally, they still deserve serious attention, so talk with reliable friends, family, or professionals."
            )
        else:
            risk_part = (
                "You did not report frequent self-harm thoughts, which is a reassuring protective factor. "
                "Even so, emotional pain still deserves care, so consider whether additional professional support would help."
            )

        tail = (
            "Regardless of the total score, this questionnaire is only a self-screening tool and cannot replace in-person evaluation or diagnosis. "
            "If the results worry or confuse you, try sharing those feelings with friends, family, or a mental-health professional."
        )
        return base + risk_part + tail

    args_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "answer": {
                "type": "integer",
                "minimum": 0,
                "maximum": 3,
                "description": "User's score for the previous question (0-3). Omit on the very first call.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def _classify_score(score: int) -> str:
        """Return the questionnaire severity label for a total score."""

        if score <= 2:
            return "No or minimal depressive symptoms"
        if score <= 5:
            return "Mild depressive symptoms"
        if score <= 9:
            return "Moderate depressive symptoms"
        if score <= 12:
            return "Moderately severe depressive symptoms"
        return "Severe depressive symptoms"

    @tool("mental_questionnaire", args_schema=args_schema, return_direct=False)
    def mental_questionnaire(answer: Optional[int] = None) -> str:
        """Run a short mental-health questionnaire and compute a total score.

        Parameters
        ----------
        answer : int | None, optional
            User's score for the previous question.

        Returns
        -------
        str
            JSON payload containing the next question or final result.
        """

        nonlocal current_index, total_score, finished, risk_item_score
        print(
            "[mental_questionnaire] invoked "
            f"(answer={answer}, current_index={current_index}, "
            f"total_score={total_score}, finished={finished}, "
            f"risk_item_score={risk_item_score})"
        )

        if finished:
            severity = _classify_score(total_score)
            recommendation = _build_recommendation(total_score, risk_item_score)
            print(
                "[mental_questionnaire] already finished "
                f"(score={total_score}, severity={severity}, "
                f"risk_item_score={risk_item_score})"
            )
            payload = {
                "finished": True,
                "question_index": total_questions,
                "total_questions": total_questions,
                "score": total_score,
                "max_score": total_questions * 3,
                "severity": severity,
                "recommendation": recommendation,
                "message": "The questionnaire is complete. Please consider the suggestions and whether you need professional help. This is only a self-screening result and cannot replace diagnosis or treatment.",
            }
            return json.dumps(payload, ensure_ascii=False)

        if current_index == 0 and answer is None:
            print("[mental_questionnaire] starting questionnaire")
            payload = {
                "finished": False,
                "question_index": 0,
                "total_questions": total_questions,
                "question": _QUESTIONS[0],
                "instruction": "Answer based on your past two weeks using an integer from zero to three: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.",
            }
            return json.dumps(payload, ensure_ascii=False)

        if answer is not None:
            try:
                clamped = max(0, min(3, int(answer)))
            except Exception:
                clamped = 0
            print(
                "[mental_questionnaire] normalized answer "
                f"(raw={answer}, clamped={clamped}, index={current_index})"
            )
            if current_index == total_questions - 1:
                risk_item_score = clamped
                print(
                    "[mental_questionnaire] updated risk item "
                    f"(risk_item_score={risk_item_score})"
                )
            total_score += clamped
            current_index += 1
            print(
                "[mental_questionnaire] advanced state "
                f"(next_index={current_index}, total_score={total_score})"
            )

        if current_index >= total_questions:
            finished = True
            severity = _classify_score(total_score)
            recommendation = _build_recommendation(total_score, risk_item_score)
            print(
                "[mental_questionnaire] questionnaire completed "
                f"(score={total_score}, severity={severity}, "
                f"risk_item_score={risk_item_score})"
            )
            payload = {
                "finished": True,
                "question_index": total_questions - 1,
                "total_questions": total_questions,
                "score": total_score,
                "max_score": total_questions * 3,
                "severity": severity,
                "recommendation": recommendation,
                "message": "The questionnaire is complete. This is only a self-screening result and cannot replace professional diagnosis. If you face strong distress or self-harm thoughts, contact a mental-health professional as soon as possible.",
            }
            return json.dumps(payload, ensure_ascii=False)

        if current_index < 0:
            current_index = 0
        if current_index >= total_questions:
            current_index = total_questions - 1

        print(
            "[mental_questionnaire] returning next question "
            f"(question_index={current_index}, total_questions={total_questions})"
        )
        payload = {
            "finished": False,
            "question_index": current_index,
            "total_questions": total_questions,
            "question": _QUESTIONS[current_index],
            "instruction": "Answer based on your past two weeks using an integer from zero to three: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.",
        }
        return json.dumps(payload, ensure_ascii=False)

    return mental_questionnaire


@model
class MentalConsultantAgent(DefaultAgent):
    """Mental consultant sample agent with a custom prompt and tool set."""

    def __init__(
        self,
        model: BaseChatModel | dict[str, Any],
        *,
        system_prompt: str = MENTAL_CONSULTANT_PROMPT,
        tools: Optional[list[BaseTool | Callable[[], BaseTool]]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the mental consultant sample agent.

        Parameters
        ----------
        model : BaseChatModel | dict[str, Any]
            Chat model instance or ``ChatOpenAI`` configuration.
        system_prompt : str, optional
            Scenario system prompt.
        tools : list[BaseTool | Callable[[], BaseTool]] | None, optional
            Additional tool instances or factories to enable alongside the
            built-in questionnaire tool.
        **kwargs : Any
            Additional ``DefaultAgent`` keyword arguments.
        """

        super().__init__(
            model=model,
            system_prompt=system_prompt,
            tools=[build_mental_questionnaire_tool, *(tools or [])],
            **kwargs,
        )


def build_mental_consultant_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config copy that swaps in the mental consultant agent."""

    llm_agent_config = config.get("llm_agent")
    if not isinstance(llm_agent_config, dict):
        raise RuntimeError("Configured llm_agent must be an object.")
    llm_agent_params = llm_agent_config.get("params", {})
    if not isinstance(llm_agent_params, dict):
        raise RuntimeError("Configured llm_agent.params must be an object.")

    updated_config = dict(config)
    updated_config["llm_agent"] = {
        **llm_agent_config,
        "type": "MentalConsultantAgent",
        "params": dict(llm_agent_params),
    }
    return updated_config

# Instantiate Xtalk from config
## Read config from json
with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)
xtalk_instance = Xtalk.from_config(build_mental_consultant_config(config))
xtalk_instance.mount_routes(app)


# Serve static files
example_server_path = Path(__file__).parent
templates = Jinja2Templates(directory=str(example_server_path / "templates"))
static_root = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
try:
    app.mount(
        "/xtalk",
        StaticFiles(
            directory=str(Path(__file__).parent.parent.parent / "frontend" / "dist")
        ),
        name="xtalk",
    )
except Exception:
    print("No local X-Talk frontend library found. You may use the library from CDN.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
