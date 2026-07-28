"""ASR wrapper that applies final-turn LLM correction using chat history."""

from __future__ import annotations

import copy
import inspect
import json
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from ...model_loader import init_registered_model
from ..registry import model
from ..rewriters.simple import SimpleRewriter
from .interfaces import ASR

logger = logging.getLogger(__name__)


@model
class HistoryCorrectingASR(ASR):
    """Wrap an ASR and apply LLM-based correction to final streaming results.

    Parameters
    ----------
    base_asr : ASR | dict[str, Any]
        Concrete ASR instance or nested config used to instantiate one.
    corrector_model : BaseChatModel | dict[str, Any]
        Chat model instance or ``ChatOpenAI`` configuration dict used for
        final-turn correction.
    system_prompt : str, optional
        System prompt guiding the LLM correction behavior.
    """

    DEFAULT_SYSTEM_PROMPT = """
You correct the final ASR transcript for the current user turn.

Use the previous chat history only to resolve likely ASR mistakes.
Keep the user's original language.
Make only local changes; do not rewrite the whole sentence.
Do not answer the user.
Do not add new information.
Return exactly one JSON object and nothing else.
The JSON object must contain keys "changes" and "result".
"changes" must be a JSON object whose keys are original words or phrases and
whose values are the corrected words or phrases.
"result" must be the corrected full sentence.
""".strip()

    def __init__(
        self,
        *,
        base_asr: ASR | dict[str, Any],
        corrector_model: BaseChatModel | dict[str, Any],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._base_asr = self._coerce_base_asr(base_asr)

        self._corrector_model_spec = (
            copy.deepcopy(corrector_model)
            if isinstance(corrector_model, dict)
            else None
        )
        self._corrector_model = corrector_model
        self._system_prompt = system_prompt
        self._rewriter = SimpleRewriter(corrector_model, system_prompt)

    def recognize(self, audio: bytes) -> str:
        """Recognize an entire audio buffer without history-aware correction."""

        return self._base_asr.recognize(audio)

    async def async_recognize(self, audio: bytes) -> str:
        """Asynchronously recognize an entire audio buffer without correction."""

        return await self._base_asr.async_recognize(audio)

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Recognize incremental audio and correct only the final result.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the current chunk is the final chunk for the turn.
        chat_history : str | None, optional
            Serialized chat history before the current user turn.

        Returns
        -------
        str
            Raw partial transcript or corrected final transcript.
        """

        raw_text = self._base_asr.recognize_stream(
            audio,
            **self._build_stream_kwargs(
                self._base_asr.recognize_stream,
                is_final=is_final,
                chat_history=chat_history,
            ),
        )
        if not is_final:
            return raw_text
        return self._correct_final_text(
            raw_text,
            chat_history=chat_history,
        )

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Asynchronously recognize incremental audio and correct final text.

        Parameters
        ----------
        audio : bytes
            Incremental PCM 16-bit mono audio bytes.
        is_final : bool, optional
            Whether the current chunk is the final chunk for the turn.
        chat_history : str | None, optional
            Serialized chat history before the current user turn.

        Returns
        -------
        str
            Raw partial transcript or corrected final transcript.
        """

        raw_text = await self._base_asr.async_recognize_stream(
            audio,
            **self._build_stream_kwargs(
                self._base_asr.async_recognize_stream,
                is_final=is_final,
                chat_history=chat_history,
            ),
        )
        if not is_final:
            return raw_text
        return await self._async_correct_final_text(
            raw_text,
            chat_history=chat_history,
        )

    def stream_chunk_bytes_hint(self) -> int | None:
        """Delegate streaming chunk hints to the wrapped ASR."""

        return self._base_asr.stream_chunk_bytes_hint()

    def reset(self) -> None:
        """Reset the wrapped ASR state."""

        self._base_asr.reset()

    def clone(self) -> "HistoryCorrectingASR":
        """Clone the wrapper with an isolated wrapped ASR instance.

        Returns
        -------
        HistoryCorrectingASR
            Wrapper clone with a cloned wrapped ASR.
        """

        corrector_model = (
            copy.deepcopy(self._corrector_model_spec)
            if self._corrector_model_spec is not None
            else self._corrector_model
        )
        return HistoryCorrectingASR(
            base_asr=self._base_asr.clone(),
            corrector_model=corrector_model,
            system_prompt=self._system_prompt,
        )

    @staticmethod
    def _coerce_base_asr(base_asr: ASR | dict[str, Any]) -> ASR:
        """Normalize nested base ASR config into an ``ASR`` instance."""

        if isinstance(base_asr, ASR):
            return base_asr
        resolved = init_registered_model(slot="asr", model_config=base_asr)
        if not isinstance(resolved, ASR):
            raise TypeError(f"base_asr must resolve to ASR, got {type(resolved)}")
        return resolved

    @staticmethod
    def _build_stream_kwargs(
        method: Any,
        *,
        is_final: bool,
        chat_history: str | None,
    ) -> dict[str, Any]:
        """Build keyword arguments accepted by a wrapped stream method."""

        kwargs: dict[str, Any] = {"is_final": is_final}
        try:
            params = inspect.signature(method).parameters
        except (TypeError, ValueError):
            params = {}

        accepts_kwargs = any(
            param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

        if accepts_kwargs or "chat_history" in params:
            kwargs["chat_history"] = chat_history
        return kwargs

    def _correct_final_text(
        self,
        raw_text: str,
        *,
        chat_history: str | None,
    ) -> str:
        """Return a corrected final transcript or the raw ASR result."""

        payload = self._build_correction_payload(
            raw_text,
            chat_history=chat_history,
        )
        if payload is None:
            return raw_text
        try:
            model_output = self._rewriter.rewrite(payload).strip()
            corrected = self._extract_corrected_result(model_output)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "HistoryCorrectingASR correction returned invalid JSON, "
                "falling back to raw text: %s",
                exc,
            )
            return raw_text
        except Exception as exc:
            logger.warning("HistoryCorrectingASR correction failed: %s", exc)
            return raw_text
        return corrected or raw_text

    async def _async_correct_final_text(
        self,
        raw_text: str,
        *,
        chat_history: str | None,
    ) -> str:
        """Asynchronously return a corrected final transcript or raw result."""

        payload = self._build_correction_payload(
            raw_text,
            chat_history=chat_history,
        )
        if payload is None:
            return raw_text
        try:
            model_output = (await self._rewriter.async_rewrite(payload)).strip()
            corrected = self._extract_corrected_result(model_output)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "HistoryCorrectingASR async correction returned invalid JSON, "
                "falling back to raw text: %s",
                exc,
            )
            return raw_text
        except Exception as exc:
            logger.warning("HistoryCorrectingASR async correction failed: %s", exc)
            return raw_text
        return corrected or raw_text

    @staticmethod
    def _extract_corrected_result(model_output: str) -> str:
        """Extract the corrected transcript from a structured JSON response.

        Parameters
        ----------
        model_output : str
            Raw LLM output expected to contain a single JSON object.

        Returns
        -------
        str
            Corrected full sentence from the ``result`` field.

        Raises
        ------
        ValueError
            Raised when the model output cannot be parsed as the expected JSON
            object.
        """

        stripped = model_output.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        payload = json.loads(stripped)
        if not isinstance(payload, dict):
            raise ValueError("Correction output must be a JSON object.")

        changes = payload.get("changes")
        result = payload.get("result", "")
        if not isinstance(changes, dict):
            raise ValueError("Correction output field 'changes' must be an object.")
        if not isinstance(result, str):
            raise ValueError("Correction output field 'result' must be a string.")
        return result.strip()

    @staticmethod
    def _build_correction_payload(
        raw_text: str,
        *,
        chat_history: str | None,
    ) -> str | None:
        """Build the LLM correction payload for a final ASR transcript."""

        normalized_raw = raw_text.strip()
        if not normalized_raw:
            return None

        normalized_history = (chat_history or "").strip()
        if not normalized_history:
            return None

        return (
            f"<chat_history>\n{normalized_history}\n</chat_history>\n\n"
            f"<raw_final_asr>\n{normalized_raw}\n</raw_final_asr>\n"
        )
