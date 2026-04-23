import argparse
import json
from typing import Optional, Any
from pathlib import Path
from langchain.tools import tool
from fastapi import FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import mimetypes

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")
from xtalk import Xtalk
from xtalk.log_utils import mute_other_logging

mute_other_logging()

parser = argparse.ArgumentParser(description="Mental Consultant Xtalk Server")
parser.add_argument("--config", type=str, help="Path to the server configuration file")
parser.add_argument("--port", type=int, help="Port number for the server to listen on")
args = parser.parse_args()

app = FastAPI(title="Mental Consultant Xtalk Server")

# Instantiate Xtalk from config
## Read config from json
with open(args.config, "r", encoding="utf-8") as f:
    config = json.load(f)
## Change system prompt to mental consultant
config["llm_agent"]["params"][
    "system_prompt"
] = """
You are a gentle psychologist, casually chatting with a visitor. Follow these rules:

1. Reply in the same language as the user, warm and natural, like a friend, never revealing your role.
2. No parentheses, lists, or Arabic numerals; spell numbers in English words.
3. Keep answers very short, like everyday talk, no explanations.
4. If the user’s words are unclear or garbled, softly say “Could you say that again?”

Silently use the mental_questionnaire tool without letting the user notice. First call needs no argument; afterwards convert the user’s casual answer into an integer between zero and three and pass it. Blend the returned question naturally into chat, as if asking about life. After the scale ends, offer only gentle everyday words, no scores or diagnoses.
"""
xtalk_instance = Xtalk.from_config(config)


# Register mental questionnaire tool
_QUESTIONS: list[str] = [
    "Over the past two weeks, how often have you felt down, depressed, or overall unable to lift your mood? Answer with an integer between zero and three, where zero means not at all and three means nearly every day.",
    "Over the past two weeks, how much have you lost interest or pleasure in things you usually enjoy (e.g., shows, chatting, games, studying)? Use the same zero-to-three scale.",
    "Over the past two weeks, how often have you had trouble falling asleep, staying asleep, or slept too much so that you never feel rested? Zero means never, three means nearly every day.",
    "Over the past two weeks, how often have you felt drained or low on energy, even when you haven’t done much? Rate it from zero to three.",
    "Over the past two weeks, how often have you noticed a much poorer appetite or eaten far more than usual? Rate it from zero to three.",
    "Over the past two weeks, how often have you felt bad about yourself—felt you’re not good enough, a failure, or that you’ve let others down? Rate it from zero to three.",
    "Over the past two weeks, how often have you had trouble concentrating on daily tasks such as reading text or watching videos? Rate it from zero to three.",
    "Over the past two weeks, how often have you moved or spoken noticeably slower than usual—or the opposite, felt restless, on edge, and unable to sit still? Rate it from zero to three.",
    "Over the past two weeks, how often have you had thoughts that life is pointless, that you should disappear, or that hurting yourself might end everything? Zero means not at all, three means nearly every day.",
]


def build_mental_questionnaire_tool():

    total_questions = len(_QUESTIONS)
    current_index = 0
    total_score = 0
    finished = False
    # Track the score of question 9 (risk item) for personalized suggestions
    risk_item_score = 0

    def _build_recommendation(score: int, risk_score: int) -> str:
        """
        Generate a short, readable suggestion based on the total score and the risk-item score.
        This is not a medical diagnosis; it only helps the user understand their mood and next steps.
        """
        # Follow common PHQ-9 tiers: 0-4 none/minimal, 5-9 mild, 10-14 moderate, 15-27 moderately severe
        if score <= 4:
            base = (
                "Your responses suggest only minimal emotional distress—perhaps occasional mood swings or stress. "
                "Keep the habits that already help you, such as regular routines, light exercise, or chatting with trusted people."
            )
        elif score <= 9:
            base = (
                "Your answers point to mild distress that may occasionally affect your mood or efficiency. "
                "Give yourself extra care: schedule breaks, talk with friends or family, and plan small pleasant activities. "
                "If the discomfort lingers or intensifies, consider a brief check-in with a professional."
            )
        elif score <= 14:
            base = (
                "You appear to be experiencing a moderate level of distress, which may already be affecting sleep, focus, school, or work. "
                "Please take these signals seriously: try contacting a counselor, campus mental-health center, or psychiatrist soon "
                "so a professional can help assess the situation. "
                "You can also start gentle self-help strategies—maintain routines, move your body, and track mood shifts."
            )
        else:
            base = (
                "Your scores indicate significant emotional strain that likely affects sleep, appetite, motivation, or relationships. "
                "This does not mean something is wrong with you—it shows you have been carrying a heavy load for a long time. "
                "Reach out to a counselor or psychiatrist as soon as possible, bring these results, and plan structured support together."
            )

        if risk_score > 0:
            risk_part = (
                "You also reported a score above zero on the item about life feeling meaningless or having self-harm thoughts. "
                "If you are facing intense urges to harm yourself right now, please do not bear it alone—contact someone you trust and seek medical help together, "
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
        if score <= 4:
            return "No or minimal depressive symptoms"
        if score <= 9:
            return "Mild depressive symptoms"
        if score <= 14:
            return "Moderate depressive symptoms"
        if score <= 19:
            return "Moderately severe depressive symptoms"
        return "Severe depressive symptoms"

    MENTAL_QUESTIONNAIRE_TOOL = "mental_questionnaire"

    @tool(MENTAL_QUESTIONNAIRE_TOOL, args_schema=args_schema, return_direct=False)
    def mental_questionnaire(answer: Optional[int] = None) -> str:
        """Run a short mental-health questionnaire and compute a total score.

        Usage (the tool maintains its own per-session state):
        1. On the very first call, pass no parameters. The tool returns the first question plus:
           - `finished`: false
           - `question_index`: current question index (zero-based)
           - `total_questions`: total number of questions
        2. After the user answers a question, infer an integer between zero and three and call the tool again:
           - The parameter is called `answer` and must be 0, 1, 2, or 3.
           - The tool tracks progress and the cumulative score internally.
        3. Return payload:
           - If more questions remain:
             * `finished`: false
             * `question`: the next prompt
             * `question_index`, `total_questions`: returned for reference in replies
           - After the final question:
             * `finished`: true
             * `score`: accumulated score
             * `max_score`: maximum possible score
             * `severity`: rough severity label (from `_classify_score`)
             * `recommendation`: personalized suggestion based on `score` and the ninth question
             * `message`: fixed reminder that this is only a self-screening tool
        4. Calling the tool after completion returns the same final result. Recreate the session or reset the agent to restart.

        When you receive `finished: true`:
        - Respond in a natural, empathetic tone; acknowledge the user’s feelings before discussing scores.
        - Briefly explain what the `score` and `severity` imply about emotional distress.
        - Follow the `recommendation` text to outline next steps, for example:
          * Mild distress: encourage self-care, routine adjustments, and emotional awareness.
          * Moderate or higher distress: gently suggest consulting a professional soon.
          * If `recommendation` indicates self-harm risk: focus on safety and urge the user to seek immediate offline help,
            such as trusted people nearby, campus counseling centers, hospitals, or crisis hotlines.
        - Do not provide formal diagnoses or suggest medications/dangerous actions.

        This tool only aids self-awareness of emotional states; it is not medical advice or diagnosis.
        """

        nonlocal current_index, total_score, finished, risk_item_score

        if finished:
            severity = _classify_score(total_score)
            recommendation = _build_recommendation(total_score, risk_item_score)
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

        # First call: no answer parameter
        if current_index == 0 and answer is None:
            question = _QUESTIONS[0]
            payload = {
                "finished": False,
                "question_index": 0,
                "total_questions": total_questions,
                "question": question,
                "instruction": "Answer based on your past two weeks using an integer from zero to three: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.",
            }
            return json.dumps(payload, ensure_ascii=False)

        if answer is not None:
            try:
                clamped = max(0, min(3, int(answer)))
            except Exception:
                clamped = 0
            # Record the score of question 9 (risk item, last index)
            if current_index == total_questions - 1:
                risk_item_score = clamped
            total_score += clamped
            current_index += 1

        if current_index >= total_questions:
            finished = True
            severity = _classify_score(total_score)
            recommendation = _build_recommendation(total_score, risk_item_score)
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

        question = _QUESTIONS[current_index]
        payload = {
            "finished": False,
            "question_index": current_index,
            "total_questions": total_questions,
            "question": question,
            "instruction": "Answer based on your past two weeks using an integer from zero to three: 0 = not at all, 1 = several days, 2 = more than half the days, 3 = nearly every day.",
        }
        return json.dumps(payload, ensure_ascii=False)

    return mental_questionnaire


xtalk_instance.add_agent_tools([build_mental_questionnaire_tool])


# Mount WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await xtalk_instance.connect(websocket)


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
except:
    print("No local X-Talk frontend library found. You may use the library from CDN.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=args.port or 11995)
