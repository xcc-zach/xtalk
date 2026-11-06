"""Single-call orchestration with Live-local playback accounting."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import OrderedDict, deque
from typing import Any

from ...models import EndToEnd, Models
from ...models.end_to_end import EndToEndCloseResult
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    AudioFrameReceived,
    AudioStreamControlRequested,
    AudioStreamReady,
    EndToEndAudioChunkReady,
    EndToEndSignal,
    EndToEndTranscriptDelta,
    ServiceRuntimeStarted,
)
from ..interfaces import Manager


class _AudioPlaybackBudget:
    """Bound Live PCM delivery using the original TTS budget approach.

    Parameters
    ----------
    window_ms : float
        Maximum audio duration in flight.
    chunk_ms : float
        PCM delivery chunk duration.
    stall_timeout_s : float or None
        Maximum time without an acknowledgement while waiting.
    """

    def __init__(
        self,
        window_ms: float = 300,
        chunk_ms: float = 100,
        stall_timeout_s: float | None = None,
    ) -> None:
        self.window_ms, self.chunk_ms = window_ms, chunk_ms
        self.stall_timeout_s = stall_timeout_s
        self._condition = asyncio.Condition()
        self._chunks: deque[tuple[int, float]] = deque()
        self._generation = 0
        self._next_token = 0
        self._stopped = False
        self._identity = ""
        self._sent_ms = self._played_ms = 0.0
        self._progress_at = time.monotonic()

    @property
    def outstanding_ms(self) -> float:
        """Return unconfirmed audio duration."""
        return sum(duration for _, duration in self._chunks)

    @staticmethod
    def duration_ms(pcm: bytes, sample_rate: int) -> float:
        """Return mono PCM16 duration in milliseconds."""
        return len(pcm) * 500.0 / sample_rate if sample_rate > 0 else 0.0

    def split(self, pcm: bytes, sample_rate: int) -> list[bytes]:
        """Split PCM at sample boundaries using the configured chunk duration."""
        if not pcm or sample_rate <= 0:
            return []
        size = max(1, round(sample_rate * self.chunk_ms / 1000)) * 2
        return [pcm[i : i + size] for i in range(0, len(pcm), size)]

    async def reset(self, identity: str = "") -> None:
        """Invalidate old reservations and wake blocked delivery."""
        async with self._condition:
            self._generation += 1
            self._identity = identity
            self._stopped = False
            self._chunks.clear()
            self._sent_ms = self._played_ms = 0.0
            self._progress_at = time.monotonic()
            self._condition.notify_all()

    async def stop(self) -> None:
        """Cancel blocked reservations immediately."""
        async with self._condition:
            self._stopped = True
            self._generation += 1
            self._chunks.clear()
            self._condition.notify_all()

    async def reserve(self, duration_ms: float) -> tuple[int, int]:
        """Wait for capacity and account before the caller publishes audio."""
        generation = self._generation
        async with self._condition:
            while self.outstanding_ms + duration_ms > self.window_ms + 0.001:
                if self._stopped or generation != self._generation:
                    raise asyncio.CancelledError
                timeout = None
                if self.stall_timeout_s is not None:
                    timeout = self.stall_timeout_s - (
                        time.monotonic() - self._progress_at
                    )
                    if timeout <= 0:
                        raise asyncio.TimeoutError("Audio playback stalled without progress")
                try:
                    await asyncio.wait_for(self._condition.wait(), timeout)
                except asyncio.TimeoutError:
                    raise asyncio.TimeoutError(
                        "Audio playback stalled without progress"
                    ) from None
            if self._stopped or generation != self._generation:
                raise asyncio.CancelledError
            if not self._chunks:
                self._progress_at = time.monotonic()
            self._next_token += 1
            self._chunks.append((self._next_token, duration_ms))
            self._sent_ms += duration_ms
            return generation, self._next_token

    async def rollback(self, reservation: tuple[int, int]) -> None:
        """Undo a failed publication without touching a newer delivery."""
        async with self._condition:
            if reservation[0] != self._generation:
                return
            for token, duration in self._chunks:
                if token == reservation[1]:
                    self._chunks.remove((token, duration))
                    self._sent_ms -= duration
                    break
            self._condition.notify_all()

    async def acknowledge_samples(
        self, identity: str, played_samples: int, sample_rate: int
    ) -> bool:
        """Release cumulative Live progress, rejecting stale or invalid reports."""
        async with self._condition:
            played_ms = played_samples * 1000 / sample_rate
            if (
                identity != self._identity
                or type(played_samples) is not int
                or played_ms < self._played_ms
                or played_ms > self._sent_ms + 0.001
            ):
                return False
            remaining = played_ms - self._played_ms
            if remaining > 0:
                self._progress_at = time.monotonic()
            while remaining > 0.001 and self._chunks:
                token, duration = self._chunks.popleft()
                consumed = min(duration, remaining)
                remaining -= consumed
                if duration > consumed + 0.001:
                    self._chunks.appendleft((token, duration - consumed))
            self._played_ms = played_ms
            self._condition.notify_all()
            return True


def create_stream_profile(model: EndToEnd, config: dict[str, Any]) -> dict[str, Any]:
    """Validate first-release capabilities and describe this call's media.

    Parameters
    ----------
    model : EndToEnd
        Unopened per-call adapter.
    config : dict
        Service policy.

    Returns
    -------
    dict
        Protocol version, stream identity, media and supported capabilities.
    """
    for option in (
        "recording",
        "send_full_audio_to_client",
        "multi_speaker",
        "forced_alignment",
        "enable_persistence",
    ):
        if config.get(option):
            raise ValueError(f"{option} is not implemented for single-call Live")
    media = model.audio_profile.to_dict()
    if (
        media["encoding"] != "pcm_s16le"
        or media["channels"] != 1
        or media["sample_rate"] not in {16000, 24000}
    ):
        raise ValueError("Live requires mono PCM16LE at 16 or 24 kHz")
    if config.get("input_sample_rate", media["sample_rate"]) != media["sample_rate"]:
        raise ValueError("Configure sample_rate on the Live model for both directions")
    policy = {
        "client_ready_timeout_s": 30,
        "pending_audio_ms": 5000,
        "playback_stall_timeout_s": 15,
    }
    supplied = config.get("end_to_end", {})
    if not isinstance(supplied, dict) or supplied.keys() - policy.keys():
        raise ValueError("Unknown end_to_end service policy")
    policy.update(supplied)
    for value in policy.values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("Live limits must be finite and positive")
    return {
        "mode": "end_to_end",
        "protocol_version": 2,
        "stream_id": str(uuid.uuid4()),
        "media": media,
        "policy": policy,
        "metadata": model.metadata,
        "capabilities": {
            "continuous_audio": True,
            "transcript_fragments": True,
            "input_mute": True,
            "history": False,
            "text_input": False,
            "change_voice": False,
            "recording": False,
            "full_audio_export": False,
        },
    }


class EndToEndManager(Manager):
    """Own one call, separating provider events from playback delivery.

    Parameters
    ----------
    event_bus : EventBus
        Existing session dispatcher.
    session_id : str
        Authorized session identity.
    models : Models
        Cloned model container.
    config : dict
        Service configuration containing the negotiated profile.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any],
    ) -> None:
        self.event_bus, self.session_id = event_bus, session_id
        self.model = models.require(EndToEnd)
        self.profile = config["_end_to_end_profile"]
        self.stream_id = self.profile["stream_id"]
        self.sample_rate = self.profile["media"]["sample_rate"]
        self._budget = _AudioPlaybackBudget(
            stall_timeout_s=self.profile["policy"]["playback_stall_timeout_s"]
        )
        self._ready, self._stop = asyncio.Event(), asyncio.Event()
        self._task: asyncio.Task | None = None
        self._active = False
        self._pending: asyncio.Queue[bytes] = asyncio.Queue()
        self._pending_samples = 0
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._arrival = 0
        self._failed = False
        self._reason = ""
        self._outcome = "stopped"
        self._progress_at = time.monotonic()

    async def _signal(self, action: str, **data: Any) -> None:
        succeeded = await self.event_bus.publish(
            EndToEndSignal(
                session_id=self.session_id,
                action=action,
                data={"stream_id": self.stream_id, **data},
            ),
            mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
            require_success=True,
        )
        if not succeeded and not self._stop.is_set():
            self._failed = True
            self._reason = "Browser control delivery failed"
            self._stop.set()
            await self._budget.stop()

    @Manager.event_handler(ServiceRuntimeStarted)
    async def _on_start(self, event: ServiceRuntimeStarted) -> None:
        if event.mode == "end_to_end" and self._task is None:
            self._task = asyncio.create_task(
                self._run(), name=f"live-call-{self.session_id}"
            )

    @Manager.event_handler(AudioStreamReady)
    async def _on_ready(self, event: AudioStreamReady) -> None:
        if event.stream_id == self.stream_id:
            self._ready.set()

    @Manager.event_handler(AudioFrameReceived)
    async def _on_audio(self, event: AudioFrameReceived) -> None:
        if not self._active or self._stop.is_set():
            return
        if (
            event.sample_rate != self.sample_rate
            or len(event.audio_data) % 2
            or len(event.audio_data) > self.sample_rate * 2 / 5
        ):
            await self._fail("media", "Live requires PCM16 frames of at most 200 ms")
            return
        try:
            await self.model.append_audio(event.audio_data)
        except (BufferError, RuntimeError, ValueError) as error:
            await self._fail("input_failed", str(error))

    @Manager.event_handler(AudioStreamControlRequested)
    async def _on_control(self, event: AudioStreamControlRequested) -> None:
        if event.stream_id != self.stream_id:
            return
        if event.command == "close_audio_stream":
            self._stop.set()
            await self._budget.stop()
        elif event.command == "_transport_failure":
            await self._fail("transport", "Browser transport failed")
        elif event.command == "set_input_muted":
            if self._active and isinstance(event.data.get("muted"), bool):
                await self.model.set_input_muted(event.data["muted"])
        elif event.command == "audio_stream_progress":
            played = event.data.get("played_samples")
            if type(played) is int and played >= 0:
                before = self._budget.outstanding_ms
                if await self._budget.acknowledge_samples(
                    self.stream_id, played, self.sample_rate
                ):
                    if self._budget.outstanding_ms < before:
                        self._progress_at = time.monotonic()
                    return
            await self._signal(
                "end_to_end_error",
                code="invalid_progress",
                message="Invalid playback progress",
                fatal=False,
            )
        else:
            await self._signal(
                "end_to_end_error",
                code="unsupported_capability",
                message="Control is unavailable in single-call Live",
                fatal=False,
            )

    async def _fail(self, code: str, message: str) -> None:
        if self._failed:
            return
        self._failed = True
        self._reason = message
        self._stop.set()
        await self._budget.stop()
        await self._signal("end_to_end_error", code=code, message=message, fatal=True)

    async def _startup(self) -> None:
        await asyncio.wait_for(
            self._ready.wait(), self.profile["policy"]["client_ready_timeout_s"]
        )
        await self.model.open()

    async def _run(self) -> None:
        workers: list[asyncio.Task] = []
        try:
            await self._budget.reset(self.stream_id)
            startup = asyncio.create_task(self._startup())
            stopped = asyncio.create_task(self._stop.wait())
            workers = [startup, stopped]
            done, _ = await asyncio.wait(workers, return_when=asyncio.FIRST_COMPLETED)
            if stopped in done:
                return
            await startup
            if self._stop.is_set():
                return
            self._active = True
            await self._signal("audio_stream_start", media=self.profile["media"])
            workers.extend(
                [
                    asyncio.create_task(self._consume(), name="live-events"),
                    asyncio.create_task(self._deliver(), name="live-playback"),
                    asyncio.create_task(
                        self._watch_playback(), name="live-playback-watch"
                    ),
                ]
            )
            done, _ = await asyncio.wait(
                workers[1:], return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                await task
        except asyncio.CancelledError:
            self._reason = "Call cancelled"
        except asyncio.TimeoutError as error:
            await self._fail(
                "playback_stalled" if self._active else "startup_timeout",
                str(error) or "Live client/provider startup timed out",
            )
        except Exception as error:  # noqa: BLE001 - owned session task boundary
            await self._fail("session_failed", str(error)[:200])
        finally:
            self._active = False
            self._stop.set()
            await self._budget.stop()
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            while not self._pending.empty():
                self._pending.get_nowait()
            self._pending_samples = 0
            try:
                result = await self.model.close()
            except Exception:  # noqa: BLE001 - provider cleanup must not strand the browser
                result = EndToEndCloseResult("incomplete", "Provider cleanup failed")
            await self._signal(
                "audio_stream_end",
                finalization=result.finalization,
                reason=self._reason or result.reason,
                outcome="failed" if self._failed else self._outcome,
            )

    async def _consume(self) -> None:
        async for event in self.model.events():
            if event.kind == "audio":
                samples = len(event.audio) // 2
                if (
                    self._pending_samples + samples
                    > self.sample_rate
                    * self.profile["policy"]["pending_audio_ms"]
                    / 1000
                ):
                    await self._fail(
                        "playback_congestion",
                        "Pending playback exceeded its configured audio limit",
                    )
                    return
                self._pending_samples += samples
                self._pending.put_nowait(event.audio)
            elif event.kind == "transcript":
                identity = event.data.get("provider_event_id")
                if identity and identity in self._seen:
                    continue
                if identity:
                    self._seen[identity] = None
                    if len(self._seen) > 4096:
                        self._seen.popitem(last=False)
                self._arrival += 1
                await self.event_bus.publish(
                    EndToEndTranscriptDelta(
                        session_id=self.session_id,
                        stream_id=self.stream_id,
                        arrival_sequence=self._arrival,
                        **event.data,
                    ),
                    mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                )
            elif event.kind == "error":
                await self._signal("end_to_end_error", **event.data)
                if event.data.get("fatal"):
                    self._failed = True
                    self._reason = event.data.get("message", "Provider error")
                    return
            elif event.kind == "closed":
                self._outcome = "completed"
                reason = event.data.get("reason", "")
                if reason in {"content", "connection_lost", "expired"}:
                    self._failed = True
                    self._reason = f"Live session ended: {reason}"
                return

    async def _deliver(self) -> None:
        while True:
            pcm = await self._pending.get()
            # Keep dequeued PCM in pending accounting until each piece is admitted.
            for chunk in self._budget.split(pcm, self.sample_rate):
                reservation = await self._budget.reserve(
                    self._budget.duration_ms(chunk, self.sample_rate)
                )
                self._pending_samples -= len(chunk) // 2
                if (
                    self._budget.outstanding_ms
                    <= self._budget.duration_ms(chunk, self.sample_rate) + 0.001
                ):
                    self._progress_at = time.monotonic()
                try:
                    succeeded = await self.event_bus.publish(
                        EndToEndAudioChunkReady(
                            session_id=self.session_id,
                            stream_id=self.stream_id,
                            audio_data=chunk,
                        ),
                        mode=EventDispatchMode.WAIT_UNTIL_COMPLETE,
                        require_success=True,
                    )
                    if not succeeded:
                        raise RuntimeError("Browser audio delivery failed")
                except BaseException:
                    await self._budget.rollback(reservation)
                    raise

    async def _watch_playback(self) -> None:
        timeout = self.profile["policy"]["playback_stall_timeout_s"]
        while True:
            await asyncio.sleep(min(1, timeout / 2))
            if (
                self._budget.outstanding_ms
                and time.monotonic() - self._progress_at >= timeout
            ):
                await self._fail(
                    "playback_stalled",
                    f"No browser playback progress for {timeout:g} seconds",
                )
                return

    async def shutdown(self) -> None:
        """Stop delivery immediately and await idempotent provider finalization."""
        self._stop.set()
        await self._budget.stop()
        if self._task is not None and self._task is not asyncio.current_task():
            await self._task
        else:
            await self.model.close()
