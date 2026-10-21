"""Single-session Realtime audio client.

Install websockets, numpy, sounddevice and soundfile. Use --input-wav with
--no-speaker for synthesized-audio acceptance without a physical sound card.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urlencode, urlsplit, urlunsplit


def build_url(base_url: str, model: str) -> str:
    """Construct a mounted Realtime URL.

    Parameters
    ----------
    base_url : str
        HTTP or WebSocket service root, including an optional mount prefix.
    model : str
        Public model alias.

    Returns
    -------
    str
        WebSocket endpoint with an encoded model query.
    """
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.netloc:
        raise ValueError("base_url must be an HTTP or WebSocket URL")
    scheme = "wss" if parts.scheme in {"https", "wss"} else "ws"
    return urlunsplit(
        (
            scheme,
            parts.netloc,
            parts.path.rstrip("/") + "/v1/realtime",
            urlencode({"model": model}),
            "",
        )
    )


@dataclass(eq=False)
class _PlaybackBlock:
    """Track one original protocol block until its DAC playback is confirmed."""

    response_id: str
    pcm: bytes
    done: asyncio.Future
    stopped: bool = False
    generation: int = 0
    offset: int = 0
    spans: list[tuple[float, float]] = field(default_factory=list)


class _AudioPlayback:
    """Feed a persistent output stream and settle original blocks after playback.

    Parameters
    ----------
    no_speaker : bool
        Consume PCM at audio speed without opening an output device.
    """

    def __init__(self, *, no_speaker: bool):
        self._loop = asyncio.get_running_loop()
        self._lock = threading.Lock()
        self._queue = deque()
        self._pending = []
        self._stopped = set()
        self._paused = False
        self._closed = False
        self._stream = None
        self._simulation = None
        if no_speaker:
            self._simulation = asyncio.create_task(self._simulate())
        else:
            import sounddevice as sd

            self._stream = sd.RawOutputStream(
                samplerate=24000,
                channels=1,
                dtype="int16",
                blocksize=480,
                callback=self._callback,
            )
            self._stream.start()

    def _clock(self):
        """Read the clock used for output-buffer DAC timestamps."""
        return self._stream.time if self._stream is not None else self._loop.time()

    def enqueue(self, *, response_id: str, pcm: bytes):
        """Queue an original audio block without waiting for the device.

        Parameters
        ----------
        response_id : str
            Owning response identifier.
        pcm : bytes
            Mono 24 kHz PCM16 data from one audio delta.

        Returns
        -------
        _PlaybackBlock | None
            Completion receipt, or None for a stopped response.
        """
        if response_id in self._stopped:
            return None
        block = _PlaybackBlock(response_id, pcm, self._loop.create_future())
        with self._lock:
            self._pending.append(block)
            if pcm:
                self._queue.append(block)
            else:
                block.done.set_result(None)
        return block

    def _callback(self, outdata, frames, timing, status):
        """Fill device output without blocking the audio callback thread."""
        self._render(outdata, frames, timing.outputBufferDacTime)

    def _render(self, outdata, frames, dac_time):
        """Fill one callback across block boundaries and record actual DAC spans."""
        target = memoryview(outdata).cast("B")
        target[:] = bytes(len(target))
        position = 0
        with self._lock:
            if self._paused or self._closed:
                return
            while self._queue and position < frames * 2:
                block = self._queue[0]
                size = min(frames * 2 - position, len(block.pcm) - block.offset)
                target[position : position + size] = block.pcm[
                    block.offset : block.offset + size
                ]
                end = dac_time + (position + size) / 48000
                block.spans.append((dac_time + position / 48000, size / 48000))
                block.offset += size
                position += size
                if block.offset == len(block.pcm):
                    self._queue.popleft()
                    self._loop.call_soon_threadsafe(
                        self._schedule_completion, block, end, block.generation
                    )

    def _schedule_completion(self, block, end, generation):
        """Resolve a block only once its last sample has reached the DAC."""

        def complete():
            if (
                not block.done.done()
                and generation == block.generation
                and not block.stopped
            ):
                block.done.set_result(None)

        if not block.done.done():
            self._loop.call_later(max(0, end - self._clock()), complete)

    def acknowledge(self, block):
        """Retire a block after its protocol acknowledgement is sent.

        Parameters
        ----------
        block : _PlaybackBlock
            Fully played original protocol block.
        """
        with self._lock:
            if block in self._pending:
                self._pending.remove(block)

    def has_pending(self, response_id):
        """Report whether a response still has unacknowledged output.

        Parameters
        ----------
        response_id : str
            Response to inspect.

        Returns
        -------
        bool
            Whether blocks still await playback acknowledgement.
        """
        return any(block.response_id == response_id for block in self._pending)

    def pause(self, paused):
        """Pause or resume callback consumption without reopening the device.

        Parameters
        ----------
        paused : bool
            Whether to hold queued audio.
        """
        with self._lock:
            self._paused = paused

    async def stop_response(self, response_id):
        """Discard a stopped response and return unconfirmed played milliseconds.

        Parameters
        ----------
        response_id : str
            Response whose queued and partially played blocks must stop.

        Returns
        -------
        float
            Actual played duration since the last block acknowledgement.
        """
        now = self._clock()
        with self._lock:
            self._stopped.add(response_id)
            blocks = [
                block for block in self._pending if block.response_id == response_id
            ]
            elapsed = sum(
                min(duration, max(0, now - start))
                for block in blocks
                for start, duration in block.spans
            )
            self._queue = deque(
                block for block in self._queue if block.response_id != response_id
            )
            self._pending = [
                block for block in self._pending if block.response_id != response_id
            ]
            for block in blocks:
                block.stopped = True
                if not block.done.done():
                    block.done.cancel()
            # Flushing a hardware buffer is needed only for current audio;
            # a late stop must not interrupt a subsequent response.
            flushing = self._stream is not None and any(
                start + duration > now
                for block in blocks
                for start, duration in block.spans
            )
            self._paused = flushing
            if flushing:
                for block in self._pending:
                    if any(start + duration > now for start, duration in block.spans):
                        played = [
                            (start, min(duration, max(0, now - start)))
                            for start, duration in block.spans
                        ]
                        block.offset = sum(
                            round(duration * 24000) * 2 for _, duration in played
                        )
                        block.spans = [
                            (start, duration)
                            for start, duration in played
                            if duration > 0
                        ]
                        block.generation += 1
                self._queue = deque(
                    block for block in self._pending if block.offset < len(block.pcm)
                )
        if flushing:
            await asyncio.to_thread(self._stream.abort)
            self._stream.start()
        self.pause(False)
        return elapsed * 1000

    async def _simulate(self):
        """Use the same block accounting with a real-time synthetic DAC."""
        while True:
            self._render(bytearray(960), 480, self._loop.time())
            await asyncio.sleep(0.02)

    async def close(self):
        """Release the single output stream and pending completion receipts."""
        with self._lock:
            self._closed = True
            for block in self._pending:
                if not block.done.done():
                    block.done.cancel()
            self._queue.clear()
            self._pending.clear()
        if self._simulation is not None:
            self._simulation.cancel()
            await asyncio.gather(self._simulation, return_exceptions=True)
        if self._stream is not None:
            self._stream.abort()
            self._stream.close()


async def run_client(args) -> None:
    """Run one audio session using a microphone or synthetic WAV input.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line options.
    """
    import numpy as np
    import websockets

    queue = asyncio.Queue()
    finished = set()
    active = None
    playing_response = None
    audio_done: dict[str, asyncio.Event] = {}
    playback_settled: dict[str, asyncio.Event] = {}
    response_started = asyncio.Queue()

    def completion(events, response_id):
        return events.setdefault(response_id, asyncio.Event())

    first_output = asyncio.Event()
    interruption = asyncio.Event()
    headers = {"Authorization": "Bearer " + args.api_key} if args.api_key else {}
    async with websockets.connect(
        build_url(args.base_url, args.model), additional_headers=headers, max_size=2**22
    ) as ws:

        async def send(kind, **data):
            await ws.send(json.dumps({"type": kind, **data}))

        async def finish_playback(response_id):
            if (
                response_id in finished
                and response_id not in playback._stopped
                and not playback.has_pending(response_id)
                and not completion(audio_done, response_id).is_set()
            ):
                await send(
                    "response.output_audio.playback.finished", response_id=response_id
                )
                completion(audio_done, response_id).set()

        async def play():
            nonlocal playing_response
            while True:
                block = await queue.get()
                playing_response = block.response_id
                try:
                    try:
                        await block.done
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                        continue
                    if block.stopped:
                        continue
                    await send(
                        "response.output_audio.playback.chunk_played",
                        response_id=block.response_id,
                    )
                    playback.acknowledge(block)
                    await finish_playback(block.response_id)
                finally:
                    playing_response = None
                    queue.task_done()

        async def capture():
            if args.input_wav:
                import soundfile as sf
                import soxr

                for round_index in range(args.rounds):
                    wav_path = (
                        args.interrupt_wav
                        if round_index > 0 and args.interrupt_wav
                        else args.input_wav
                    )
                    pcm, sr = sf.read(wav_path, dtype="int16", always_2d=True)
                    mono = pcm.mean(axis=1).astype("int16")
                    if sr != 24000:
                        mono = soxr.resample(mono, sr, 24000).astype("int16")
                    samples = np.concatenate([mono, np.zeros(24000 * 2, dtype="int16")])
                    for start in range(0, len(samples), 480):
                        await send(
                            "input_audio_buffer.append",
                            audio=base64.b64encode(
                                samples[start : start + 480].astype("<i2").tobytes()
                            ).decode(),
                        )
                        await asyncio.sleep(0.02)
                    response_id = await asyncio.wait_for(
                        response_started.get(), args.timeout
                    )
                    if round_index == 0 and args.interrupt_after is not None:
                        await asyncio.wait_for(first_output.wait(), args.timeout)
                        await asyncio.sleep(args.interrupt_after)
                        continue
                    await asyncio.wait_for(
                        completion(audio_done, response_id).wait(), args.timeout
                    )
                    await asyncio.wait_for(
                        completion(playback_settled, response_id).wait(), args.timeout
                    )
                    print("Completed audio round")
                if args.interrupt_after is not None:
                    await asyncio.wait_for(interruption.wait(), args.timeout)
                    print("Verified playback interruption")
                return
            import sounddevice as sd

            with sd.RawInputStream(
                samplerate=24000, channels=1, dtype="int16", blocksize=480
            ) as stream:
                while True:
                    pcm, _ = await asyncio.to_thread(stream.read, 480)
                    await send(
                        "input_audio_buffer.append",
                        audio=base64.b64encode(bytes(pcm)).decode(),
                    )

        async def receive():
            nonlocal active
            async for raw in ws:
                message = json.loads(raw)
                kind = message["type"]
                if args.trace_events:
                    print(
                        kind,
                        message.get("response_id", message.get("item_id", "")),
                        flush=True,
                    )
                if kind == "response.created":
                    response_started.put_nowait(message["response"]["id"])
                elif kind == "response.output_audio.delta":
                    first_output.set()
                    active = message["response_id"]
                    block = playback.enqueue(
                        response_id=active, pcm=base64.b64decode(message["delta"])
                    )
                    if block is not None:
                        await queue.put(block)
                elif kind == "response.output_audio.done":
                    response_id = message["response_id"]
                    finished.add(response_id)
                    await finish_playback(response_id)
                elif kind == "response.output_audio.playback.stop":
                    interruption.set()
                    elapsed = await playback.stop_response(message["response_id"])
                    await send(
                        "response.output_audio.playback.stopped",
                        response_id=message["response_id"],
                        played_audio_ms=elapsed,
                    )
                elif kind == "response.output_audio.playback.pause":
                    if message["response_id"] in {active, playing_response}:
                        playback.pause(True)
                elif kind == "response.output_audio.playback.resume":
                    if message["response_id"] in {active, playing_response}:
                        playback.pause(False)
                elif (
                    kind == "error"
                    or kind == "conversation.item.input_audio_transcription.failed"
                ):
                    raise RuntimeError(message["error"])
                elif kind.endswith(("transcription.completed", "playback.done")):
                    if kind.endswith("playback.done"):
                        completion(playback_settled, message["response_id"]).set()
                    print(kind, message.get("transcript", message.get("text", "")))

        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == "session.created":
                break
        await send(
            "session.update",
            session={
                "type": "realtime",
                "audio": {"input": {"turn_detection": {"type": "server_vad"}}},
            },
        )
        async for raw in ws:
            message = json.loads(raw)
            if message["type"] == "error":
                raise RuntimeError(message["error"])
            if message["type"] == "session.updated":
                break
        playback = _AudioPlayback(no_speaker=args.no_speaker)
        tasks = [
            asyncio.create_task(play()),
            asyncio.create_task(receive()),
            asyncio.create_task(capture()),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await playback.close()


def main() -> None:
    """Parse CLI settings and start the single-session client."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key")
    parser.add_argument("--input-wav")
    parser.add_argument(
        "--interrupt-wav",
        help="Optional different speech fixture for later audio rounds",
    )
    parser.add_argument("--no-speaker", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--trace-events", action="store_true")
    parser.add_argument(
        "--interrupt-after",
        type=float,
        help="Feed the next WAV turn during first-response playback",
    )
    asyncio.run(run_client(parser.parse_args()))


if __name__ == "__main__":
    main()
