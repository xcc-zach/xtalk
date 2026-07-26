"""Headless MTD decode API backed by the official vLLM Python interface."""

from __future__ import annotations

import argparse
import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from aiohttp import web


_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"(?:\[(?P<speaker>S\d+)\])?"
    r"(?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.IGNORECASE | re.DOTALL,
)
_LEADING_NESTED_SPEAKER_RE = re.compile(r"^\[*\[(S\d+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class _Segment:
    """One timestamped raw-model segment."""

    start_s: float
    end_s: float
    speaker_id: str
    text: str


def _parse_segments(text: str, *, max_time_s: float) -> list[_Segment]:
    """Parse timestamped MTD output with concrete speaker fallbacks."""

    result: list[_Segment] = []
    last_speaker = "S01"
    for match in _SEGMENT_RE.finditer(text or ""):
        start_s = float(match.group("start"))
        end_s = min(float(match.group("end")), max_time_s)
        if end_s <= start_s or start_s > max_time_s + 1.0:
            continue
        segment_text = match.group("text")
        nested = _LEADING_NESTED_SPEAKER_RE.match(segment_text)
        explicit_speaker = nested.group(1) if nested else match.group("speaker")
        if explicit_speaker:
            last_speaker = explicit_speaker.upper()
        if nested:
            segment_text = segment_text[nested.end() :]
        result.append(
            _Segment(
                start_s=max(0.0, start_s),
                end_s=end_s,
                speaker_id=last_speaker,
                text=re.sub(r"\s+", " ", segment_text).strip(),
            )
        )
    return result


def _crop_to_current(
    segments: list[_Segment],
    *,
    context_seconds: float,
    current_audio_seconds: float,
) -> list[dict[str, object]]:
    """Clip request-global timestamps to current-audio-local coordinates."""

    right_s = context_seconds + current_audio_seconds
    result: list[dict[str, object]] = []
    for item in segments:
        start_s = max(context_seconds, item.start_s)
        end_s = min(right_s, item.end_s)
        if end_s <= start_s:
            continue
        result.append(
            {
                "start_s": start_s - context_seconds,
                "end_s": end_s - context_seconds,
                "speaker_id": item.speaker_id,
                "text": item.text,
            }
        )
    return result


class OfficialMtdRuntime:
    """Own one official vLLM engine and serve immutable snapshot decodes."""

    def __init__(self, args: argparse.Namespace) -> None:
        from vllm import LLM
        from vllm.model_executor.model_loader import get_model_architecture

        self.sample_rate = int(args.sample_rate)
        self.default_instruction = str(args.instruction)
        self.llm = LLM(
            model=args.model,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enforce_eager=args.enforce_eager,
            trust_remote_code=True,
            limit_mm_per_prompt={"audio": 1},
            enable_prefix_caching=True,
            mm_processor_cache_gb=args.mm_processor_cache_gb,
            disable_log_stats=False,
        )
        self.model_config = self.llm.llm_engine.model_config
        self.model_cls, self.model_arch = get_model_architecture(self.model_config)
        self.stt_config = self.model_cls.get_speech_to_text_config(
            self.model_config,
            "transcribe",
        )
        self._decode_lock = asyncio.Lock()
        self._cancelled: set[str] = set()

    def _generation_prompt(self, audio: np.ndarray, instruction: str) -> Any:
        """Build the official multimodal generation prompt."""

        from vllm.config.speech_to_text import SpeechToTextParams

        params = SpeechToTextParams(
            audio=np.asarray(audio, dtype=np.float32),
            stt_config=self.stt_config,
            model_config=self.model_config,
            request_prompt=instruction,
        )
        return self.model_cls.get_generation_prompt(params)

    def _decode_sync(
        self,
        *,
        audio: np.ndarray,
        instruction: str,
        decoder_prefix: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """Run the blocking official vLLM decode on the worker thread."""

        from vllm import SamplingParams
        from vllm.inputs import TextPrompt

        prompt = self._generation_prompt(audio, instruction)
        if not isinstance(prompt, dict):
            raise TypeError(
                "MTD get_generation_prompt() must return a TextPrompt mapping, "
                f"got {type(prompt)!r}"
            )
        prompt_text = str(prompt["prompt"]) + decoder_prefix
        request = TextPrompt(
            prompt=prompt_text,
            multi_modal_data=prompt["multi_modal_data"],
        )
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        started = time.perf_counter()
        outputs = self.llm.generate([request], sampling_params=sampling, use_tqdm=False)
        wall_ms = (time.perf_counter() - started) * 1000.0
        request_output = outputs[0]
        completion_output = request_output.outputs[0]
        metrics = getattr(request_output, "metrics", None)
        return {
            "raw_text": decoder_prefix + str(completion_output.text or ""),
            "latency_ms": wall_ms,
            "metrics": {
                "prompt_tokens": len(request_output.prompt_token_ids or []),
                "completion_tokens": len(completion_output.token_ids or []),
                "cached_tokens": getattr(request_output, "num_cached_tokens", None),
                "first_token_latency_ms": (
                    float(metrics.first_token_latency) * 1000.0
                    if metrics is not None
                    and getattr(metrics, "first_token_latency", None) is not None
                    else None
                ),
            },
        }

    async def decode(self, fields: dict[str, Any]) -> dict[str, Any]:
        """Decode one parsed multipart request and crop exemplar timestamps."""

        request_id = str(fields["request_id"])
        sample_rate = int(fields["sample_rate"])
        if sample_rate != self.sample_rate:
            raise web.HTTPBadRequest(
                text=f"sample_rate must be {self.sample_rate}, got {sample_rate}"
            )
        pcm16 = bytes(fields["audio"])
        if len(pcm16) % 2:
            raise web.HTTPBadRequest(text="PCM16 payload must contain complete samples")
        audio = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        decoder_prefix = str(fields.get("decoder_prefix") or "")
        context_seconds = float(fields.get("context_seconds") or 0.0)
        current_audio_seconds = float(fields.get("current_audio_seconds") or 0.0)
        instruction = str(fields.get("instruction") or self.default_instruction)
        temperature = float(fields.get("temperature") or 0.0)
        max_tokens = int(fields.get("max_tokens") or 2048)
        async with self._decode_lock:
            if request_id in self._cancelled:
                self._cancelled.discard(request_id)
                raise web.HTTPConflict(text="request cancelled")
            output = await asyncio.to_thread(
                self._decode_sync,
                audio=audio,
                instruction=instruction,
                decoder_prefix=decoder_prefix,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        if request_id in self._cancelled:
            self._cancelled.discard(request_id)
            raise web.HTTPConflict(text="request cancelled")
        parsed = _parse_segments(
            output["raw_text"],
            max_time_s=len(audio) / self.sample_rate,
        )
        output["request_id"] = request_id
        output["current_segments"] = _crop_to_current(
            parsed,
            context_seconds=context_seconds,
            current_audio_seconds=current_audio_seconds,
        )
        return output

    def cancel(self, request_id: str) -> None:
        """Mark a queued or running request so its result is discarded."""

        self._cancelled.add(request_id)


async def _read_multipart(request: web.Request) -> dict[str, Any]:
    """Read text fields and the binary PCM field from one multipart request."""

    if not request.content_type.startswith("multipart/"):
        raise web.HTTPBadRequest(text="multipart/form-data is required")
    reader = await request.multipart()
    fields: dict[str, Any] = {}
    while True:
        part = await reader.next()
        if part is None:
            break
        if not part.name:
            continue
        if part.name == "audio":
            fields[part.name] = await part.read(decode=False)
        else:
            fields[part.name] = await part.text()
    missing = [
        name
        for name in ("request_id", "sample_rate", "audio")
        if name not in fields
    ]
    if missing:
        raise web.HTTPBadRequest(text=f"missing fields: {', '.join(missing)}")
    return fields


def create_app(runtime: OfficialMtdRuntime) -> web.Application:
    """Create the minimal aiohttp runtime application."""

    app = web.Application(client_max_size=256 * 1024**2)

    async def health(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "model_arch": runtime.model_arch})

    async def decode(request: web.Request) -> web.Response:
        fields = await _read_multipart(request)
        result = await runtime.decode(fields)
        return web.json_response(result)

    async def cancel(request: web.Request) -> web.Response:
        runtime.cancel(request.match_info["request_id"])
        return web.json_response({"status": "accepted"}, status=202)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/mtd/decode", decode)
    app.router.add_delete("/v1/mtd/requests/{request_id}", cancel)
    return app


def parse_args() -> argparse.Namespace:
    """Parse runtime configuration from command-line flags."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18604)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.72)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--mm-processor-cache-gb", type=float, default=4.0)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument(
        "--instruction",
        default=(
            "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
            "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
            "并在段末标注结束时间戳，以清晰标明该段语音范围。"
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Initialize the official engine once and serve the headless API."""

    args = parse_args()
    runtime = OfficialMtdRuntime(args)
    web.run_app(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
