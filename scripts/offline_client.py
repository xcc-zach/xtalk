# scripts/offline_client.py
"""
Audio exchange test client for xtalk server.

Dependencies:
    pip install websockets soundfile numpy soxr

Usage:
    python scripts/offline_client.py --ws ws://127.0.0.1:8000/ws --input /path/to/audio_dir
    python scripts/offline_client.py --ws ws://127.0.0.1:8000/ws --input /path/to/audio_dir --with-vad
    python scripts/offline_client.py --ws ws://127.0.0.1:8000/ws --input /path/to/audio_dir --output /path/to/recording.wav

The input directory should contain audio files and a timestamp.txt file.
Each line in timestamp.txt has the format: <timestamp>:<audio_file_name.suffix>

Timestamp formats:
    - Float: absolute seconds from start (e.g., 0, 5.0, 10.5)
    - ai_start: when first AI audio chunk starts playing
    - ai_end: when AI response finishes playing
    - user_start: when first user audio chunk is about to be sent
    - user_end: when last user audio chunk is sent (after chunk duration)
    - <label>+<offset>: offset in seconds after the event (e.g., ai_end+2.5)

Example timestamp.txt:
    0:greeting.wav
    ai_end:question.wav
    ai_end+2.5:followup.wav
    user_end+1.0:interrupt.wav
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import soxr
import websockets

FRAME_SAMPLES = 512
TARGET_SR = 16000
TTS_SR = 48000  # TTS output sample rate
BYTES_PER_SAMPLE = 2  # int16
SILENCE_FRAME = b"\x00" * (FRAME_SAMPLES * BYTES_PER_SAMPLE)
VAD_LATENCY_SEC = 0.5  # extra wait after audio end, before vad_speech_end

# Valid timestamp labels
TIMESTAMP_LABELS = {"ai_start", "ai_end", "user_start", "user_end"}


@dataclass
class TimingSpec:
    """Timing specification for an audio task."""

    label: Optional[str] = None  # None means absolute time
    offset: float = 0.0  # offset in seconds (absolute time or relative to label)


@dataclass
class AudioTask:
    """An audio file with its scheduled send time."""

    path: str
    timing: TimingSpec


@dataclass
class ClientState:
    """Track conversation state.

    Event semantics:
    - Events persist across tasks to support cross-turn timing (e.g., ai_end:next.wav)
    - Each event is recorded with a monotonic timestamp when it fires
    - Event counters track how many times each event has fired (for multi-turn waits)
    - reset_for_new_audio() clears per-audio state but preserves event history
    """

    start_time: float = 0.0  # monotonic start time
    tts_bytes_received: int = 0
    tts_finished: bool = False
    ai_started: bool = False  # first audio chunk received
    ai_end_fired: bool = False  # ai_end fired for current response

    # Event timestamps (monotonic time) - persisted across tasks
    event_times: Dict[str, float] = field(default_factory=dict)

    # Event counters - how many times each event has fired
    event_counts: Dict[str, int] = field(
        default_factory=lambda: {l: 0 for l in TIMESTAMP_LABELS}
    )

    # Async events for waiting - recreated per wait
    events: Dict[str, asyncio.Event] = field(default_factory=dict)

    # Snapshot of event counts at task start (to detect new events)
    _wait_snapshot: Dict[str, int] = field(default_factory=dict)

    def init_events(self):
        """Initialize async events (call once at start)."""
        self.events = {label: asyncio.Event() for label in TIMESTAMP_LABELS}

    def snapshot_for_wait(self):
        """Take snapshot of current event counts before waiting.

        Events persist across tasks - if already set, wait proceeds immediately.
        This enables cross-task timing like ai_end:next.wav to work correctly.
        """
        self._wait_snapshot = self.event_counts.copy()

    def reset_for_new_audio(self):
        """Reset per-audio state but preserve cross-task event history."""
        self.tts_bytes_received = 0
        self.tts_finished = False
        self.ai_started = False
        self.ai_end_fired = False


class AudioExchangeClient:
    def __init__(
        self,
        ws_url: str,
        audio_tasks: List[AudioTask],
        with_vad: bool = False,
        output_path: Optional[str] = None,
    ):
        self.ws_url = ws_url
        self.audio_tasks = audio_tasks
        self.with_vad = with_vad
        self.output_path = output_path
        self.state = ClientState()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = True
        self._silence_task: Optional[asyncio.Task] = None

    async def connect(self):
        self.ws = await websockets.connect(self.ws_url, ping_interval=None)
        print(f"[Connected] {self.ws_url}")

        # Send session config if output path is specified
        if self.output_path:
            await self.send_json(
                {
                    "action": "session_config",
                    "recording_path": self.output_path,
                }
            )
            print(f"[Config] Recording path: {self.output_path}")

    async def send_json(self, obj: dict):
        try:
            await self.ws.send(json.dumps(obj))
        except websockets.ConnectionClosed:
            print(
                f"[Send] Connection closed, failed to send: {obj.get('action', 'unknown')}"
            )

    async def _send_silence_loop(self):
        """Continuously send silence frames.

        Hardened against connection closure - exits gracefully if connection drops.
        """
        frame_sec = FRAME_SAMPLES / TARGET_SR
        while self._running:
            try:
                await self.ws.send(SILENCE_FRAME)
            except websockets.ConnectionClosed:
                print("[Silence] Connection closed, stopping silence loop")
                break
            except Exception as e:
                print(f"[Silence] Error sending silence: {e}")
                break
            await asyncio.sleep(frame_sec)

    def _record_event(self, label: str):
        """Record an event timestamp, increment counter, and signal waiters."""
        now = asyncio.get_event_loop().time()
        self.state.event_times[label] = now
        self.state.event_counts[label] = self.state.event_counts.get(label, 0) + 1
        if label in self.state.events:
            self.state.events[label].set()
        print(
            f"[Event] {label} at {now - self.state.start_time:.3f}s (count={self.state.event_counts[label]})"
        )

    async def send_audio_file(self, path: str):
        """Load and send a WAV file as PCM frames."""
        frames = self._load_wav_as_frames(path)
        frame_sec = FRAME_SAMPLES / TARGET_SR

        if self.with_vad:
            # Send VAD start signal
            await self.send_json(
                {"action": "vad_speech_start", "timestamp": int(time.time() * 1000)}
            )

        # Record user_start before sending first frame
        self._record_event("user_start")

        # Send audio frames in real-time pacing
        start = asyncio.get_event_loop().time()
        for i, frame in enumerate(frames):
            target_time = start + i * frame_sec
            now = asyncio.get_event_loop().time()
            if target_time > now:
                await asyncio.sleep(target_time - now)
            try:
                await self.ws.send(frame)
            except websockets.ConnectionClosed:
                print(f"[Send] Connection closed while sending audio")
                return

        # Record user_end after last frame + frame duration
        await asyncio.sleep(frame_sec)
        self._record_event("user_end")

        if self.with_vad:
            # Small delay then send VAD end signal
            await asyncio.sleep(VAD_LATENCY_SEC)
            await self.send_json(
                {"action": "vad_speech_end", "timestamp": int(time.time() * 1000)}
            )

        print(f"[Sent] {path}")

    async def wait_for_response_playback(self):
        """Wait for actual playback completion (ai_end event), not just tts_finished.

        The ai_end event fires after the last audio chunk has been "played"
        (simulated via sleep in receive_loop), which is the true playback end.
        """
        # If no audio was received, just check tts_finished
        if not self.state.ai_started:
            # Wait for tts_finished in case server sends it with no audio
            timeout = 30.0  # reasonable timeout
            start = asyncio.get_event_loop().time()
            while not self.state.tts_finished:
                if asyncio.get_event_loop().time() - start > timeout:
                    print("[Playback] Timeout waiting for response (no audio received)")
                    break
                await asyncio.sleep(0.05)
            return

        # Wait for ai_end event (set after last chunk is played)
        ai_end_event = self.state.events.get("ai_end")
        if ai_end_event:
            await ai_end_event.wait()

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
                    # Record ai_start on first audio chunk
                    if not self.state.ai_started:
                        self.state.ai_started = True
                        self._record_event("ai_start")

                    self.state.tts_bytes_received += len(msg)
                    # Simulate playback duration for this chunk
                    chunk_duration = len(msg) / (TTS_SR * BYTES_PER_SAMPLE)
                    await asyncio.sleep(chunk_duration)
                    # Notify server this chunk has been played
                    await self.send_json({"action": "tts_chunk_played"})

                    # Record ai_end when this is the last chunk (tts_finished already received)
                    # Use ai_end_fired flag to allow ai_end to fire once per response
                    if self.state.tts_finished and not self.state.ai_end_fired:
                        self.state.ai_end_fired = True
                        self._record_event("ai_end")
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

    async def wait_for_timing(self, timing: TimingSpec):
        """Wait until the specified timing condition is met.

        For event-based timing (ai_end, user_end, etc.):
        - Uses event counts to detect NEW events since snapshot
        - If a new event fired after snapshot, uses its timestamp
        - If no new event, waits for the next one
        - Offset is calculated from the actual event timestamp

        This enables reliable cross-task scheduling like:
            ai_end:question.wav      -> send after previous response ends
            ai_end+2.5:followup.wav  -> send 2.5s after previous response ends
        """
        if timing.label is None:
            # Absolute time from start
            target = self.state.start_time + timing.offset
            now = asyncio.get_event_loop().time()
            if target > now:
                wait_time = target - now
                print(f"[Wait] {wait_time:.2f}s until timestamp {timing.offset}")
                await asyncio.sleep(wait_time)
        else:
            label = timing.label
            snapshot_count = self.state._wait_snapshot.get(label, 0)
            current_count = self.state.event_counts.get(label, 0)

            if current_count <= snapshot_count:
                # No new event since snapshot, need to wait
                event = self.state.events.get(label)
                if event:
                    # Clear event so we can wait for the next firing
                    event.clear()
                    print(f"[Wait] Waiting for {label}...")
                    await event.wait()
            else:
                print(
                    f"[Wait] {label} already fired (count {current_count} > snapshot {snapshot_count}), proceeding"
                )

            # Apply offset relative to the actual event timestamp
            event_time = self.state.event_times.get(label)
            if timing.offset > 0 and event_time:
                target = event_time + timing.offset
                now = asyncio.get_event_loop().time()
                if target > now:
                    remaining = target - now
                    print(
                        f"[Wait] {remaining:.2f}s remaining (offset {timing.offset}s after {label})"
                    )
                    await asyncio.sleep(remaining)
                else:
                    print(f"[Wait] Offset already elapsed, proceeding")

    async def run(self):
        """Main execution loop.

        Event lifecycle:
        1. Initialize events once at start
        2. For each task:
           a. Snapshot event counts (to detect new events)
           b. Wait for timing (may reference events from previous tasks)
           c. Reset per-audio state (tts_bytes_received, etc.)
           d. Send audio and process response
        3. Events persist across tasks to support cross-turn timing
        """
        await self.connect()

        # Start receive loop
        recv_task = asyncio.create_task(self.receive_loop())

        # Start silence loop
        self._silence_task = asyncio.create_task(self._send_silence_loop())

        self.state.start_time = asyncio.get_event_loop().time()
        self.state.init_events()

        try:
            for i, task in enumerate(self.audio_tasks):
                print(f"\n=== Audio {i + 1}/{len(self.audio_tasks)}: {task.path} ===")

                # Snapshot event counts before waiting (to track new events)
                self.state.snapshot_for_wait()

                # Wait for timing condition (may reference events from previous tasks)
                await self.wait_for_timing(task.timing)

                # Reset per-audio state AFTER waiting, BEFORE sending
                # This preserves event history for cross-task timing
                self.state.reset_for_new_audio()

                # Pause silence sending while sending real audio
                if self._silence_task:
                    self._silence_task.cancel()
                    try:
                        await self._silence_task
                    except asyncio.CancelledError:
                        pass

                await self.send_audio_file(task.path)

                # Resume silence sending after audio
                self._silence_task = asyncio.create_task(self._send_silence_loop())

            # Wait for final response
            print("\n[Wait] Waiting for final response...")
            # Snapshot for final wait
            self.state.snapshot_for_wait()
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
        # Read audio file using soundfile
        audio, sr = sf.read(path, dtype="float32")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Resample to 16kHz using soxr
        if sr != TARGET_SR:
            audio = soxr.resample(audio, sr, TARGET_SR)

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


def parse_timing(timing_str: str) -> TimingSpec:
    """Parse timing string into TimingSpec.

    Formats:
        - "5.0" -> absolute time 5.0s
        - "ai_end" -> wait for ai_end event
        - "ai_end+2.5" -> 2.5s after ai_end
    """
    timing_str = timing_str.strip()

    # Check for label with offset pattern: label+offset
    match = re.match(r"^([a-z_]+)\+(\d+(?:\.\d+)?)$", timing_str)
    if match:
        label, offset_str = match.groups()
        if label not in TIMESTAMP_LABELS:
            raise ValueError(
                f"Invalid label '{label}', expected one of: {', '.join(sorted(TIMESTAMP_LABELS))}"
            )
        return TimingSpec(label=label, offset=float(offset_str))

    # Check for plain label
    if timing_str in TIMESTAMP_LABELS:
        return TimingSpec(label=timing_str, offset=0.0)

    # Try parsing as absolute time
    try:
        return TimingSpec(label=None, offset=float(timing_str))
    except ValueError:
        raise ValueError(
            f"Invalid timing '{timing_str}', expected number, label ({', '.join(sorted(TIMESTAMP_LABELS))}), or label+offset"
        )


def parse_audio_arg(arg: str, base_dir: Optional[str] = None) -> AudioTask:
    """Parse 'timing:file.wav' format."""
    if ":" not in arg:
        raise ValueError(f"Invalid format '{arg}', expected 'timing:file.wav'")

    timing_str, path = arg.split(":", 1)

    # If base_dir is provided, resolve path relative to it
    if base_dir:
        path = os.path.join(base_dir, path)

    timing = parse_timing(timing_str)
    return AudioTask(path=path, timing=timing)


def load_tasks_from_directory(input_dir: str) -> List[AudioTask]:
    """Load audio tasks from a directory containing audio files and timestamp.txt."""
    input_path = Path(input_dir)

    if not input_path.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    timestamp_file = input_path / "timestamp.txt"
    if not timestamp_file.exists():
        raise ValueError(f"timestamp.txt not found in {input_dir}")

    tasks = []
    with open(timestamp_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                task = parse_audio_arg(line, base_dir=input_dir)
                # Verify the audio file exists
                if not os.path.exists(task.path):
                    raise ValueError(f"Audio file not found: {task.path}")
                tasks.append(task)
            except ValueError as e:
                raise ValueError(f"Error on line {line_num} of timestamp.txt: {e}")

    if not tasks:
        raise ValueError(f"No valid audio tasks found in {timestamp_file}")

    return tasks


def main():
    parser = argparse.ArgumentParser(
        description="Audio exchange test client for xtalk server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Timestamp formats:
  - Float: absolute seconds from start (e.g., 0, 5.0)
  - ai_start: when first AI audio chunk starts playing
  - ai_end: when AI response finishes playing
  - user_start: when first user audio chunk is about to be sent
  - user_end: when last user audio chunk is sent
  - <label>+<offset>: seconds after the event (e.g., ai_end+2.5)

Examples:
  # Using input directory with timestamp.txt
  python %(prog)s --input /path/to/audio_dir

  # With client-side VAD
  python %(prog)s --input /path/to/audio_dir --with-vad

  # With custom server-side recording path
  python %(prog)s --input /path/to/audio_dir --output /path/to/recording.wav
        """,
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:8000/ws", help="WebSocket URL")

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing audio files and timestamp.txt",
    )
    parser.add_argument(
        "--with-vad",
        action="store_true",
        help="Send vad_speech_start/end signals (client-side VAD mode)",
    )
    parser.add_argument(
        "--output",
        help="Server-side recording output path (sent via session_config)",
    )
    args = parser.parse_args()

    audio_tasks = load_tasks_from_directory(args.input)

    client = AudioExchangeClient(
        ws_url=args.ws,
        audio_tasks=audio_tasks,
        with_vad=args.with_vad,
        output_path=args.output,
    )
    asyncio.run(client.run())


if __name__ == "__main__":
    main()
