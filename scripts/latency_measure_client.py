# latency_measure_client.py
#
# Strict ticker version:
# - Use ONE shared monotonic start time (start_mono) for:
#     1) ticker-paced chunk schedule
#     2) last_frame_end + vad_end_delay schedule
# - Each tick (every chunk_sec) spawns an async task to send the chunk (ws.send)
# - Keeps original behavior for non-realtime (send as fast as possible)
#
# Usage:
#   python latency_measure_client.py --audio /path/to/clip.wav --ws ws://localhost:7635/ws
#
# Dependencies:
#   pip install websockets numpy

import argparse
import asyncio
import contextlib
import json
import os
import time
import wave
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import websockets

FRAME_SAMPLES = 512
TARGET_SR = 16000


def now_ms() -> int:
    return int(time.time() * 1000)


def _read_wav_via_wave(path: str) -> Tuple[np.ndarray, int]:
    """
    Read WAV using stdlib wave.
    Supports PCM16/PCM32/PCM8.
    Returns mono float32 in [-1, 1] and sample_rate.
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        # unsigned 8-bit PCM
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        # could be int32 PCM; wave doesn't distinguish float32
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        x = x.reshape(-1, n_channels).mean(axis=1)

    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x, sr


def linear_resample(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Stateless linear resampler (mirrors the simple frontend style)."""
    if from_sr == to_sr:
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return x.astype(np.float32, copy=False)

    out_len = max(1, int(round(len(x) * (to_sr / from_sr))))
    ratio = from_sr / to_sr  # input samples per output sample

    out = np.empty(out_len, dtype=np.float32)
    for j in range(out_len):
        pos = j * ratio
        i = int(pos)
        frac = pos - i
        s0 = x[i] if i < len(x) else 0.0
        i1 = i + 1 if (i + 1) < len(x) else i
        s1 = x[i1] if i1 < len(x) else s0
        out[j] = s0 + (s1 - s0) * frac

    return np.clip(out, -1.0, 1.0).astype(np.float32)


def float32_to_int16(x: np.ndarray) -> np.ndarray:
    """Match frontend mapping: negative uses 0x8000, positive uses 0x7fff."""
    x = np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)
    out = np.empty_like(x, dtype=np.int16)
    neg = x < 0
    out[neg] = (x[neg] * 32768.0).astype(np.int16)
    out[~neg] = (x[~neg] * 32767.0).astype(np.int16)
    return out


def pad_to_full_frames(x: np.ndarray, frame_samples: int = FRAME_SAMPLES) -> np.ndarray:
    """Pad the tail with zeros so we always send full frames."""
    if x.size == 0:
        return x
    r = x.size % frame_samples
    if r:
        x = np.concatenate([x, np.zeros(frame_samples - r, dtype=np.float32)], axis=0)
    return x


@dataclass
class LatencyMetrics:
    network_latency_ms: float = 0.0
    asr_latency_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    llm_sentence_ms: float = 0.0
    tts_first_chunk_ms: float = 0.0

    @staticmethod
    def from_payload(d: dict) -> "LatencyMetrics":
        return LatencyMetrics(
            network_latency_ms=float(d.get("network_latency_ms", 0) or 0),
            asr_latency_ms=float(d.get("asr_latency_ms", 0) or 0),
            llm_first_token_ms=float(d.get("llm_first_token_ms", 0) or 0),
            llm_sentence_ms=float(d.get("llm_sentence_ms", 0) or 0),
            tts_first_chunk_ms=float(d.get("tts_first_chunk_ms", 0) or 0),
        )

    def pretty(self) -> str:
        return (
            "Latency metrics (ms):\n"
            f"  network:       {self.network_latency_ms:.1f}\n"
            f"  asr:           {self.asr_latency_ms:.1f}\n"
            f"  llmFirstToken: {self.llm_first_token_ms:.1f}\n"
            f"  llmSentence:   {self.llm_sentence_ms:.1f}\n"
            f"  ttsFirstChunk: {self.tts_first_chunk_ms:.1f}\n"
        )


async def send_json(ws, obj: dict):
    await ws.send(json.dumps(obj))


def prepare_frames(
    audio_float32_16k: np.ndarray, sr: int = TARGET_SR
) -> Tuple[List[bytes], float]:
    """
    Prepare padded PCM frames for sending.
    Returns (frames_bytes_list, frame_sec).
    """
    x = pad_to_full_frames(audio_float32_16k, FRAME_SAMPLES)
    int16 = float32_to_int16(x)
    frame_sec = FRAME_SAMPLES / float(sr)

    frames = [
        int16[i : i + FRAME_SAMPLES].tobytes(order="C")
        for i in range(0, len(int16), FRAME_SAMPLES)
    ]
    return frames, frame_sec


async def send_pcm_frames_ticker_strict(
    ws,
    frames: List[bytes],
    frame_sec: float,
    start_mono: float,
    realtime: bool = True,
) -> None:
    """
    Strict ticker-loop sender sharing EXACT same start_mono with other schedules.

    realtime=True:
      - Ticker wakes at start_mono + k*frame_sec (k=0..n-1)
      - On each tick, spawn an async send task (ws.send(frame))
      - After scheduling all ticks, await all send tasks

    realtime=False:
      - Send sequentially as fast as possible
    """
    if not frames:
        return

    if not realtime:
        for b in frames:
            await ws.send(b)
        return

    loop = asyncio.get_running_loop()
    send_tasks: List[asyncio.Task] = []

    async def send_one(frame_bytes: bytes):
        await ws.send(frame_bytes)

    for k, frame_bytes in enumerate(frames):
        target = start_mono + k * frame_sec
        remain = target - loop.time()
        if remain > 0:
            await asyncio.sleep(remain)

        # spawn send task so pacing isn't blocked by ws.send latency
        send_tasks.append(asyncio.create_task(send_one(frame_bytes)))

    await asyncio.gather(*send_tasks)


async def latency_measure(
    ws_url: str,
    audio_path: str,
    turn_id: int = 1,
    realtime: bool = True,
    wait_timeout_s: float = 60.0,
    vad_end_delay_ms: int = 500,
):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    # Load + resample to 16kHz mono float32
    audio, sr = _read_wav_via_wave(audio_path)
    audio_16k = linear_resample(audio, sr, TARGET_SR)

    # Prepare frames once
    frames, frame_sec = prepare_frames(audio_16k, sr=TARGET_SR)
    n_frames = len(frames)

    queue_granted = asyncio.Event()
    got_latency = asyncio.Event()
    last_latency: Optional[LatencyMetrics] = None

    async with websockets.connect(ws_url, ping_interval=None) as ws:
        print(f"[WS] Connected: {ws_url}")

        async def recv_loop():
            nonlocal last_latency
            try:
                async for msg in ws:
                    if isinstance(msg, (bytes, bytearray)):
                        # ignore binary TTS audio
                        continue
                    try:
                        j = json.loads(msg)
                    except Exception:
                        continue

                    action = j.get("action")
                    if action == "queue_granted":
                        queue_granted.set()
                        print("[Queue] granted")
                    elif action == "queue_status":
                        pos = j.get("position", None)
                        print(f"[Queue] status position={pos}")
                    elif action == "session_info":
                        sid = (j.get("data") or {}).get("session_id")
                        print(f"[Session] session_id={sid}")
                    elif action == "latency_metrics":
                        last_latency = LatencyMetrics.from_payload(j.get("data") or {})
                        got_latency.set()
                    elif action == "error":
                        print(f"[Server Error] {j.get('data')}")
            except websockets.ConnectionClosed:
                pass
            except asyncio.CancelledError:
                raise

        recv_task = asyncio.create_task(recv_loop())

        # Some deployments may queue you; wait briefly.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(queue_granted.wait(), timeout=5.0)

        # Speech start (wall-clock timestamp for server)
        await send_json(
            ws,
            {"action": "vad_speech_start", "timestamp": now_ms(), "turn_id": turn_id},
        )

        # STRICT shared monotonic start point for all schedules below
        loop = asyncio.get_running_loop()
        start_mono = loop.time()

        # VAD end should happen at: last_frame_end + vad_end_delay_ms
        last_frame_end_mono = start_mono + (n_frames * frame_sec)
        target_end_mono = last_frame_end_mono + (vad_end_delay_ms / 1000.0)

        async def timer_send_vad_end():
            remain = target_end_mono - loop.time()
            if remain > 0:
                await asyncio.sleep(remain)
            await send_json(
                ws,
                {"action": "vad_speech_end", "timestamp": now_ms(), "turn_id": turn_id},
            )

        audio_task = asyncio.create_task(
            send_pcm_frames_ticker_strict(
                ws,
                frames=frames,
                frame_sec=frame_sec,
                start_mono=start_mono,
                realtime=realtime,
            )
        )
        end_task = asyncio.create_task(timer_send_vad_end())

        # Ensure both complete (audio + VAD end)
        await asyncio.gather(audio_task, end_task)

        # Wait for latency metrics
        try:
            await asyncio.wait_for(got_latency.wait(), timeout=wait_timeout_s)
        except asyncio.TimeoutError:
            print(f"[Timeout] No latency_metrics within {wait_timeout_s}s")
        else:
            print(
                last_latency.pretty()
                if last_latency
                else "[Warn] latency_metrics received but parse failed"
            )

        # Clean shutdown of recv loop
        recv_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await recv_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Path to a WAV audio clip")
    p.add_argument(
        "--ws",
        default="ws://127.0.0.1:8000/ws",
        help="WebSocket URL, e.g. ws://localhost:8000/ws",
    )
    p.add_argument(
        "--turn-id", type=int, default=1, help="turn_id to attach to vad messages"
    )
    p.add_argument(
        "--realtime",
        dest="realtime",
        action="store_true",
        default=True,
        help="Send frames in (approx) real-time using strict ticker pacing (default: enabled)",
    )
    p.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Disable realtime sending (send audio as fast as possible)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for latency_metrics",
    )
    p.add_argument(
        "--vad-end-delay-ms",
        type=int,
        default=500,
        help="Simulated VAD endpointing delay in ms (added after last frame end before sending vad_speech_end)",
    )
    args = p.parse_args()

    asyncio.run(
        latency_measure(
            ws_url=args.ws,
            audio_path=args.audio,
            turn_id=args.turn_id,
            realtime=args.realtime,
            wait_timeout_s=args.timeout,
            vad_end_delay_ms=args.vad_end_delay_ms,
        )
    )


if __name__ == "__main__":
    main()
