"""
Minimal LangChain ChatModel-based rewriter.

Features:
- Initialize with `BaseChatModel` and `system_prompt`.
- `rewrite(text)` sends [System, Human] messages to the model.
- Returns the generated text content.

Notes:
- Implements just the essentials for quick pipeline integration.
- Extend this class if you need streaming or advanced control.
"""

from __future__ import annotations

import asyncio
from typing import List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .interfaces import Rewriter
from ..log_utils import logger


class SimpleRewriter(Rewriter):
    """Simple text rewriter powered by a system prompt."""

    def __init__(self, model: BaseChatModel | dict, system_prompt: str) -> None:
        if isinstance(model, dict):
            model = ChatOpenAI(**model)
        self._model = model
        self._system_prompt = system_prompt

    def rewrite(self, input: str) -> str:
        """Rewrite the input text."""
        # Build message sequence (System defines style, Human provides text)
        messages: List = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=input),
        ]

        # Use invoke API (recommended for LangChain 0.3+)
        response = self._model.invoke(messages)

        # Usually AIMessage; just grab content
        output = getattr(response, "content", "")

        if not isinstance(output, str):
            # Fallback to string to keep upstream typing expectations
            logger.warning("Model response content is not str; coercing to str")
            output = str(output)

        return output

    async def async_rewrite(self, input: str) -> str:
        """Async rewrite that prefers model.ainvoke when available."""

        messages: List = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=input),
        ]

        if hasattr(self._model, "ainvoke"):
            response = await self._model.ainvoke(messages)
        else:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, self._model.invoke, messages)

        output = getattr(response, "content", "")
        if not isinstance(output, str):
            logger.warning("Model response content is not str; coercing to str")
            output = str(output)
        return output


class DefaultThoughtRewriter(SimpleRewriter):
    """Simple rewriter tailored for internal thought summaries."""

    DEFAULT_THOUGHT_REWRITE_PROMPT = """
You are an internal Thought refiner for a conversational agent.

You will receive a single text block with two XML-like sections:
<thought>...</thought>  – the previous internal Thought summary (a rough draft).
<chat>...</chat>        – the full flattened chat history, with the most recent turns at the end.

Your job is to produce an UPDATED Thought that is more precise, structured, and insightful than the previous one.

Behavioral goals:
- Refinement, not reset:
  - Treat the previous Thought as a draft.
  - Preserve information that is still correct and relevant.
  - Remove parts that are outdated, redundant, or contradicted by the chat history.
- Deeper reasoning:
  - Make implicit information explicit when it is clearly implied by the dialogue.
  - Clarify the user's underlying goals, constraints, concerns, and emotional tendencies.
  - If helpful, summarize a short plan or strategy for how the agent should respond next.
- Incremental improvement:
  - Every time you should slightly improve the Thought:
    - more concise,
    - more insights
    - more useful abstraction for future decisions.
  - However, keep the semantics stable; do NOT wildly change the meaning each time.
- Concise and focused:
  - Output 2–4 short sentences.
  - Focus on: current user goals, key context, constraints/preferences, emotional tone (if obvious), and near-term plan.
  - Do NOT restate long dialogue content or quote messages verbatim.

Style and formatting rules:
- Write as a neutral third-person internal note, NOT as if speaking to the user.
- Do NOT mention that this is a "Thought" or that you are an AI.
- Do NOT include XML tags, labels, bullet points, or explanations.
- Use the same language as the most recent user message.
- If there is very little information, output a short best-effort guess of the current intent.

Input format (example):
<thought>[Previous Thought]</thought>
<chat>[Chat History]</chat>

Output:
ONLY the updated Thought text, nothing else.
    """

    def __init__(self, model: BaseChatModel | dict) -> None:
        super().__init__(model, self.DEFAULT_THOUGHT_REWRITE_PROMPT)


class DefaultCaptionRewriter(SimpleRewriter):
    DEFAULT_CAPTION_REWRITE_PROMPT = """
    You are a helper that cleans up and standardizes short audio captions.

The input is a noisy caption generated from the last few seconds of audio.
It may contain:
- Redundant details, repetitions, or filler words.
- Meta-information about the audio clip (e.g., timestamps, "in this recording", "audio length").
- Partial transcriptions of speech or rough environment descriptions.

Your task:
- Rewrite the caption into a single concise scene description.
- Focus on what is happening in the environment and what the main speaker is doing or intending.
- Use "Speaker" as the subject when describing the human voice (for example: "Speaker is asking about today's schedule." / "Speaker is typing on a keyboard while thinking about a coding problem.").
- Remove any references to the fact that this is an audio clip or recording (no "in this audio", "in the clip", etc.).
- Do NOT add information that is not clearly implied by the input.
- Use the same language as the input caption (if the caption is in Chinese, answer in Chinese; if in English, answer in English).
- Be brief but cover the key points (typically 1–2 short sentences).
- If the input is empty or contains no meaningful content, output an empty string.

Output:
- Only the rewritten caption text, without quotes, bullet points, labels, or explanations.
    """

    def __init__(self, model: BaseChatModel | dict) -> None:
        super().__init__(model, self.DEFAULT_CAPTION_REWRITE_PROMPT)
