# scripts/offline_client.py
"""
Audio exchange test client for xtalk server.

Usage:
    # With client-side VAD signals (only send audio during speech)
    python scripts/audio_exchange_client.py --with-vad \
        --audio greeting.wav:0 \
        --audio question.wav:on_response_finish

    # Without VAD signals (continuous silence + audio stream, server detects speech)
    python scripts/audio_exchange_client.py \
        --audio greeting.wav:0 \
        --audio question.wav:5.0 \
        --audio followup.wav:on_response_finish
"""

import argparse
import asyncio
import json
import time
import wave
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import websockets

FRAME_SAMPLES = 512
TARGET_SR = 16000
TTS_SR = 48000  # TTS output sample rate
BYTES_PER_SAMPLE = 2  # int16
SILENCE_FRAME = b"\x00" * (FRAME_SAMPLES * BYTES_PER_SAMPLE)
VAD_LATENCY_SEC = 0.3  # extra wait after audio end, before vad_speech_end


@dataclass
class AudioTask:
    """An audio file with its scheduled send time."""

    path: str
    timing: Union[float, str]  # float = absolute seconds, "on_response_finish" = wait


@dataclass
class ClientState:
    """Track conversation state."""

    start_time: float = 0.0  # monotonic start time
    tts_bytes_received: int = 0
    tts_finished: bool = False
    response_finish_event: Optional[asyncio.Event] = None


class AudioExchangeClient:
    def __init__(
        self,
        ws_url: str,
        audio_tasks: List[AudioTask],
        with_vad: bool = False,
    ):
        self.ws_url = ws_url
        self.audio_tasks = audio_tasks
        self.with_vad = with_vad
        self.state = ClientState()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = True
        self._silence_task: Optional[asyncio.Task] = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=None)
        print(f"[Connected] {self.ws_url}")

    async def send_json(self, obj: dict):
        await self.ws.send(json.dumps(obj))

    async def _send_silence_loop(self):
        """Continuously send silence frames when not in VAD mode."""
        frame_sec = FRAME_SAMPLES / TARGET_SR
        while self._running:
            await self.ws.send(SILENCE_FRAME)
            await asyncio.sleep(frame_sec)

    async def send_audio_file(self, path: str):
        """Load and send a WAV file as PCM frames."""
        frames = self._load_wav_as_frames(path)
        frame_sec = FRAME_SAMPLES / TARGET_SR

        # Reset state for this turn
        self.state.tts_bytes_received = 0
        self.state.tts_finished = False
        self.state.response_finish_event = asyncio.Event()

        if self.with_vad:
            # Send VAD start signal
            await self.send_json(
                {"action": "vad_speech_start", "timestamp": int(time.time() * 1000)}
            )

        # Send audio frames in real-time pacing
        start = asyncio.get_event_loop().time()
        for i, frame in enumerate(frames):
            target_time = start + i * frame_sec
            now = asyncio.get_event_loop().time()
            if target_time > now:
                await asyncio.sleep(target_time - now)
            await self.ws.send(frame)

        if self.with_vad:
            # Small delay then send VAD end signal
            await asyncio.sleep(VAD_LATENCY_SEC)
            await self.send_json(
                {"action": "vad_speech_end", "timestamp": int(time.time() * 1000)}
            )

        print(f"[Sent] {path}")

    async def wait_for_response_playback(self):
        """Wait for TTS to finish (playback is simulated per-chunk in receive_loop)."""
        # Wait for tts_finished signal
        while not self.state.tts_finished:
            await asyncio.sleep(0.05)

        # Calculate total duration for logging
        duration_s = self.state.tts_bytes_received / (TTS_SR * BYTES_PER_SAMPLE)
        print(f"[Playback] Total TTS duration: {duration_s:.2f}s")

        # Notify server playback finished
        await self.send_json({"action": "tts_playback_finished"})

    async def receive_loop(self):
        """Handle incoming messages from server."""
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    self.state.tts_bytes_received += len(msg)
                    # Simulate playback duration for this chunk
                    chunk_duration = len(msg) / (TTS_SR * BYTES_PER_SAMPLE)
                    await asyncio.sleep(chunk_duration)
                    # Notify server this chunk has been played
                    await self.send_json({"action": "tts_chunk_played"})
                else:
                    data = json.loads(msg)
                    action = data.get("action", "")

                    if action == "tts_finished":
                        self.state.tts_finished = True
                        print("[Server] TTS generation finished")
                    elif action == "finish_asr":
                        text = data.get("data", {}).get("text", "")
                        print(f"[ASR] {text}")
                    elif action == "finish_resp":
                        text = data.get("data", {}).get("text", "")
                        print(f"[LLM] {text[:100]}...")
                    elif action == "start_tts":
                        print("[Server] TTS started")
                    elif action == "error":
                        print(f"[Error] {data.get('data')}")
        except websockets.ConnectionClosed:
            print("[Disconnected]")

    async def run(self):
        """Main execution loop."""
        await self.connect()

        # Start receive loop
        recv_task = asyncio.create_task(self.receive_loop())

        # Start silence loop if not using client-side VAD
        if not self.with_vad:
            self._silence_task = asyncio.create_task(self._send_silence_loop())

        self.state.start_time = asyncio.get_event_loop().time()

        try:
            for i, task in enumerate(self.audio_tasks):
                print(f"\n=== Audio {i + 1}/{len(self.audio_tasks)}: {task.path} ===")

                if task.timing == "on_response_finish":
                    # Wait for previous response to finish playing
                    if i > 0:  # Skip wait for first audio
                        await self.wait_for_response_playback()
                else:
                    # Wait until absolute timestamp
                    target = self.state.start_time + task.timing
                    now = asyncio.get_event_loop().time()
                    if target > now:
                        wait_time = target - now
                        print(f"[Wait] {wait_time:.2f}s until timestamp {task.timing}")
                        await asyncio.sleep(wait_time)

                # Pause silence sending while sending real audio (if applicable)
                if self._silence_task and not self.with_vad:
                    self._silence_task.cancel()
                    try:
                        await self._silence_task
                    except asyncio.CancelledError:
                        pass

                await self.send_audio_file(task.path)

                # Resume silence sending after audio
                if not self.with_vad:
                    self._silence_task = asyncio.create_task(self._send_silence_loop())

            # Wait for final response
            print("\n[Wait] Waiting for final response...")
            await self.wait_for_response_playback()

        finally:
            self._running = False
            if self._silence_task:
                self._silence_task.cancel()
            recv_task.cancel()
            await self.ws.close()
            print("[Done]")

    def _load_wav_as_frames(self, path: str) -> List[bytes]:
        """Load WAV and convert to PCM16 frames at 16kHz."""
        with wave.open(path, "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw = wf.readframes(wf.getnframes())

        # Convert to float32
        if sampwidth == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            audio = (
                np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128
            ) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        # Mono
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        # Resample to 16kHz
        if sr != TARGET_SR:
            ratio = TARGET_SR / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len), np.arange(len(audio)), audio
            )

        # Convert to int16 frames
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)

        frames = []
        for i in range(0, len(audio_int16), FRAME_SAMPLES):
            chunk = audio_int16[i : i + FRAME_SAMPLES]
            if len(chunk) < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - len(chunk)))
            frames.append(chunk.tobytes())
        return frames


def parse_audio_arg(arg: str) -> AudioTask:
    """Parse 'file.wav:timing' format."""
    if ":" not in arg:
        raise ValueError(f"Invalid format '{arg}', expected 'file.wav:timing'")

    path, timing_str = arg.rsplit(":", 1)

    if timing_str == "on_response_finish":
        timing = "on_response_finish"
    else:
        try:
            timing = float(timing_str)
        except ValueError:
            raise ValueError(
                f"Invalid timing '{timing_str}', expected number or 'on_response_finish'"
            )

    return AudioTask(path=path, timing=timing)


def main():
    parser = argparse.ArgumentParser(
        description="Audio exchange test client for xtalk server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With client-side VAD (send vad_speech_start/end signals)
  python %(prog)s --with-vad --audio greeting.wav:0 --audio question.wav:on_response_finish

  # Without VAD (continuous audio stream, server detects speech)
  python %(prog)s --audio greeting.wav:0 --audio question.wav:5.0

  # Mixed timing
  python %(prog)s --audio intro.wav:0 --audio q1.wav:on_response_finish --audio q2.wav:30.0
        """,
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:8000/ws", help="WebSocket URL")
    parser.add_argument(
        "--audio",
        nargs="+",
        required=True,
        help="Audio files with timing: 'file.wav:seconds' or 'file.wav:on_response_finish'",
    )
    parser.add_argument(
        "--with-vad",
        action="store_true",
        help="Send vad_speech_start/end signals (client-side VAD mode)",
    )
    args = parser.parse_args()

    audio_tasks = [parse_audio_arg(a) for a in args.audio]

    client = AudioExchangeClient(
        ws_url=args.ws,
        audio_tasks=audio_tasks,
        with_vad=args.with_vad,
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
