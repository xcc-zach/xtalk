"""AgenticASR: sherpa-onnx WebSocket ASR with AgenticASR windowed refinement.

Composes the sherpa-onnx WebSocket ASR (``SherpaOnnxASR``) with the AgenticASR
refinement pipeline:

- ``streaming`` mode: the online sherpa-onnx WebSocket server provides the raw
  cumulative transcript; AgenticASR chunking plus the K=3 sliding-window
  refinement rewrites stable chunks with the remote Refiner
  (OpenAI-compatible chat-completions service).
- ``offline`` mode: the one-shot offline sherpa-onnx WebSocket server is
  wrapped in a ``MockStreamRecognizer`` to simulate incremental hypotheses
  (the same approach as ``sherpa_onnx_asr.py``), then fed through the same
  AgenticASR chunking/refinement pipeline.

``is_final=True`` maps to the AgenticASR VAD boundary: the pending raw
hypothesis is flushed through chunking/refinement and the underlying ASR
segment is closed. The refinement session persists across segments, matching
the AgenticASR streaming loop where each VAD segment owns one ASR stream.
"""

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from ..registry import model
from .interfaces import ASR
from .sherpa_onnx_asr import SherpaOnnxASR

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "你是 ASR 文本纠错助手。保留原意，最小修改：去口癖/重复，修错字，补必要标点，"
    "规范数字、日期、术语和代码符号，处理自我修正。不要总结、扩写或解释。"
    "重要易错实体在末尾追加 <KEY>[词1、词2]；没有则不加。"
)

# Model name sent to the OpenAI-compatible Refiner service. The class exposes
# only ``asr_base_url`` / ``refiner_base_url`` / ``asr_mode``, so this is the
# single knob for servers that validate the ``model`` field (for example vLLM
# ``--served-model-name refiner``).
_REFINER_MODEL = "refiner"
_MAX_REFINER_TOKENS = 512
_REFINER_TIMEOUT_SECONDS = 60.0

_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f]\s*")
_ANY_PUNCTUATION = re.compile(
    r"[,;:!?\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001]\s*"
)


def _resolve_chat_completions_url(base_url: str) -> str:
    """Resolve a Refiner base URL to a ``/chat/completions`` endpoint."""

    normalized = base_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("refiner_base_url must not be empty")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("refiner_base_url must use http:// or https://")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _run_coro(coro):
    """Run a coroutine to completion in a fresh event loop."""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _clean_response(response: str) -> str:
    """Strip chat markers that a Refiner may echo around its answer."""

    value = response.strip()
    value = re.sub(r"^<\|im_start\|>assistant\s*", "", value)
    value = re.sub(r"<\|im_end\|>\s*$", "", value)
    value = re.sub(r"^assistant\s*[:\uff1a]?\s*", "", value, flags=re.IGNORECASE)
    return re.sub(r"</s>\s*$", "", value).strip()


def _extract_message_content(data) -> str:
    """Extract the assistant message content from a chat-completions body."""

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"refiner returned an unexpected response: {data!r}") from exc
    return "" if content is None else str(content)


class _OpenAICompatibleRefiner:
    """Minimal OpenAI-compatible chat-completions client for the Refiner."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = _REFINER_TIMEOUT_SECONDS,
    ) -> None:
        self.chat_completions_url = _resolve_chat_completions_url(base_url)
        self.timeout = timeout

    def refine(self, text: str) -> str:
        """Refine one raw transcript window (synchronous)."""

        return _run_coro(self.async_refine(text))

    async def async_refine(self, text: str) -> str:
        """Refine one raw transcript window."""

        payload = {
            "model": _REFINER_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            "temperature": 0.0,
            "max_tokens": _MAX_REFINER_TOKENS,
        }
        try:
            import aiohttp
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("AgenticASR refiner requires `pip install aiohttp`") from exc

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.chat_completions_url, json=payload) as response:
                    if response.status != 200:
                        body = await response.text()
                        raise RuntimeError(
                            f"refiner returned HTTP {response.status}: {body[:300]}"
                        )
                    data = await response.json()
        except aiohttp.ClientError as exc:
            raise RuntimeError(f"refiner request failed: {exc}") from exc
        return _extract_message_content(data)


class _ChunkManager:
    """Convert evolving ASR hypotheses into bounded, stable raw chunks.

    Ported from the AgenticASR streaming pipeline. Hypotheses are cumulative
    within one VAD segment; sentence-final punctuation closes a chunk, text
    longer than ``max_chars`` is cut at the nearest punctuation (or exactly at
    the limit), and a VAD boundary (``flush``) emits the remaining text.
    """

    def __init__(self, max_chars: int = 80) -> None:
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        self.max_chars = max_chars
        self._hypothesis = ""
        self._committed_source = ""

    def reset(self) -> None:
        """Clear all chunking state for a fresh segment."""

        self._hypothesis = ""
        self._committed_source = ""

    def update(self, hypothesis: str, *, vad_boundary: bool = False) -> list[str]:
        """Accept a cumulative partial hypothesis and emit newly stable chunks."""

        normalized = hypothesis.strip()
        if self._committed_source and not normalized.startswith(self._committed_source):
            raise ValueError(
                "ASR revised text that has already been committed; start a new "
                "VAD segment or delay chunk emission"
            )
        self._hypothesis = normalized
        chunks = self._emit_available(flush=vad_boundary)
        if vad_boundary:
            self._hypothesis = ""
            self._committed_source = ""
        return chunks

    def _emit_available(self, *, flush: bool) -> list[str]:
        emitted: list[str] = []
        while pending := self._pending:
            split = self._split_point(pending, flush=flush)
            if split is None:
                break
            source = pending[:split]
            self._committed_source += source
            text = source.strip()
            if text:
                emitted.append(text)
        return emitted

    @property
    def _pending(self) -> str:
        return self._hypothesis[len(self._committed_source) :]

    def _split_point(self, text: str, *, flush: bool) -> int | None:
        sentence_end = _SENTENCE_END.search(text)
        if sentence_end is not None and sentence_end.end() <= self.max_chars:
            return sentence_end.end()

        if len(text) > self.max_chars:
            candidates = list(_ANY_PUNCTUATION.finditer(text[: self.max_chars]))
            return candidates[-1].end() if candidates else self.max_chars

        return len(text) if flush else None


class _RefinementSession:
    """K-window sliding refinement over raw ASR chunks.

    Ported from the AgenticASR streaming pipeline: each new chunk recomputes
    the refinement of the most recent ``window_size`` chunks, chunks that slide
    out of the window are committed one by one, and the returned transcript is
    the committed text followed by the active window.
    """

    def __init__(self, refiner, *, window_size: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.refiner = refiner
        self.window_size = window_size
        self.raw_chunks: list[str] = []
        self._committed_text: list[str] = []
        self._active_text = ""

    @property
    def transcript(self) -> str:
        """Return committed text followed by the active refined window."""

        return self._join([*self._committed_text, self._active_text])

    def reset(self) -> None:
        """Clear all refinement state."""

        self.raw_chunks = []
        self._committed_text = []
        self._active_text = ""

    def add(self, chunk: str) -> str:
        """Accept one raw chunk and return the refined transcript."""

        text = chunk.strip()
        if not text:
            raise ValueError("refiner chunk text must not be empty")
        self.raw_chunks.append(text)
        return self._process()

    async def aadd(self, chunk: str) -> str:
        """Accept one raw chunk asynchronously and return the refined transcript."""

        text = chunk.strip()
        if not text:
            raise ValueError("refiner chunk text must not be empty")
        self.raw_chunks.append(text)
        return await self._aprocess()

    def _window_start(self) -> int:
        return max(0, len(self.raw_chunks) - self.window_size)

    def _process(self) -> str:
        start = self._window_start()
        while len(self._committed_text) < start:
            index = len(self._committed_text)
            refined = self.refiner.refine(self.raw_chunks[index])
            self._committed_text.append(_clean_response(refined))
        active = self.refiner.refine(self._join(self.raw_chunks[start:]))
        self._active_text = _clean_response(active)
        return self.transcript

    async def _aprocess(self) -> str:
        start = self._window_start()
        while len(self._committed_text) < start:
            index = len(self._committed_text)
            refined = await self.refiner.async_refine(self.raw_chunks[index])
            self._committed_text.append(_clean_response(refined))
        active = await self.refiner.async_refine(self._join(self.raw_chunks[start:]))
        self._active_text = _clean_response(active)
        return self.transcript

    @staticmethod
    def _join(chunks: list[str]) -> str:
        output = ""
        for chunk in chunks:
            value = chunk.strip()
            if not value:
                continue
            if output and output[-1].isascii() and value[0].isascii():
                output += " "
            output += value
        return output


@model
class AgenticASR(ASR):
    """Sherpa-ONNX WebSocket ASR combined with AgenticASR windowed refinement.

    Parameters
    ----------
    asr_base_url : str
        sherpa-onnx WebSocket server URL (``ws://`` or ``wss://``). Offline
        mode expects the offline WebSocket server; streaming mode expects the
        online WebSocket server.
    refiner_base_url : str
        Base URL of the OpenAI-compatible Refiner service, for example
        ``http://127.0.0.1:8000/v1``. ``/chat/completions`` is appended when
        the URL does not already end with it.
    asr_mode : str, optional
        ``"streaming"`` or ``"offline"``. Defaults to ``"offline"``.
    """

    def __init__(
        self,
        *,
        asr_base_url: str,
        refiner_base_url: str,
        asr_mode: str = "offline",
    ) -> None:
        self.asr_base_url = asr_base_url
        self.refiner_base_url = refiner_base_url
        self.asr_mode = asr_mode

        # Offline mode uses SherpaOnnxASR's MockStreamRecognizer-wrapped
        # one-shot WebSocket decode to simulate streaming hypotheses; streaming
        # mode uses the online WebSocket server's cumulative transcript.
        self._asr = SherpaOnnxASR(base_url=asr_base_url, mode=asr_mode)
        self._refiner = _OpenAICompatibleRefiner(refiner_base_url)
        self._chunk_manager = _ChunkManager()
        self._session = _RefinementSession(self._refiner)

    def recognize(self, audio: bytes) -> str:
        """Recognize a full audio buffer and refine it once."""

        if not audio:
            return ""
        raw = self._asr.recognize(audio)
        return self._refine_full(raw)

    async def async_recognize(self, audio: bytes) -> str:
        """Asynchronously recognize a full audio buffer and refine it once."""

        if not audio:
            return ""
        raw = await self._asr.async_recognize(audio)
        return await self._async_refine_full(raw)

    def recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Recognize incremental audio and return the refined transcript."""

        del chat_history
        raw = self._asr.recognize_stream(audio, is_final=is_final)
        return self._process_stream(raw, is_final=is_final)

    async def async_recognize_stream(
        self,
        audio: bytes,
        *,
        is_final: bool = False,
        chat_history: str | None = None,
    ) -> str:
        """Asynchronously recognize incremental audio and refine it."""

        del chat_history
        raw = await self._asr.async_recognize_stream(audio, is_final=is_final)
        return await self._aprocess_stream(raw, is_final=is_final)

    def stream_chunk_bytes_hint(self) -> int | None:
        """Delegate the preferred streaming chunk size to the wrapped ASR."""

        return self._asr.stream_chunk_bytes_hint()

    def reset(self) -> None:
        """Reset the ASR segment and the AgenticASR refinement pipeline."""

        self._asr.reset()
        self._chunk_manager.reset()
        self._session.reset()

    def clone(self) -> "AgenticASR":
        """Create a clone that reuses the remote config but keeps separate state."""

        return AgenticASR(
            asr_base_url=self.asr_base_url,
            refiner_base_url=self.refiner_base_url,
            asr_mode=self._asr.mode,
        )

    # ------------------------------------------------------------------
    # AgenticASR pipeline
    # ------------------------------------------------------------------
    def _process_stream(self, raw: str, *, is_final: bool) -> str:
        """Run raw hypotheses through chunking and K-window refinement."""

        try:
            chunks = self._chunk_manager.update(raw, vad_boundary=is_final)
        except ValueError as exc:
            logger.warning(
                "AgenticASR raw hypothesis revised committed text; "
                "restarting the ASR segment: %s",
                exc,
            )
            self._asr.reset()
            self._chunk_manager.reset()
            chunks = self._chunk_manager.update(raw, vad_boundary=is_final)

        for chunk in chunks:
            self._session.add(chunk)

        if is_final:
            # Close the ASR segment at the VAD boundary; the refinement
            # session persists so the transcript accumulates across segments.
            self._asr.reset()
        return self._session.transcript

    async def _aprocess_stream(self, raw: str, *, is_final: bool) -> str:
        """Async variant of :meth:`_process_stream`."""

        try:
            chunks = self._chunk_manager.update(raw, vad_boundary=is_final)
        except ValueError as exc:
            logger.warning(
                "AgenticASR raw hypothesis revised committed text; "
                "restarting the ASR segment: %s",
                exc,
            )
            self._asr.reset()
            self._chunk_manager.reset()
            chunks = self._chunk_manager.update(raw, vad_boundary=is_final)

        for chunk in chunks:
            await self._session.aadd(chunk)

        if is_final:
            self._asr.reset()
        return self._session.transcript

    def _refine_full(self, raw: str) -> str:
        text = raw.strip()
        if not text:
            return ""
        return _clean_response(self._refiner.refine(text))

    async def _async_refine_full(self, raw: str) -> str:
        text = raw.strip()
        if not text:
            return ""
        return _clean_response(await self._refiner.async_refine(text))
