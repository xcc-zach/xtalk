"""
VADManager

Backend VAD manager used when the frontend cannot run VAD. It mimics the logic in
`frontend/src/index.js` using a configured VAD model:

- Process frames at 16 kHz with 512 samples (~32 ms/frame)
- Speech start triggers when speech frames accumulate beyond `min_speech_ms`
- Speech end triggers when silence frames exceed `redemption_ms`
- Emits `VADSpeechStart`/`VADSpeechEnd`; speech end is published with
  `mode=EventDispatchMode.WAIT_UNTIL_COMPLETE` so higher-priority handlers can flush state first.

Notes:
- Relies on `VAD` interface (`is_speech(frame: bytes) -> bool`) from
  `src/xtalk/models/*/interfaces.py`.
- Uses duration smoothing on boolean outputs instead of probability thresholds.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any, TypeVar

from ...models import VAD, Models
from ..event_bus import EventBus, EventDispatchMode
from ..events import (
    EnhancedAudioFrameReceived,
    TurnInputAbortRequested,
    VADSpeechEnd,
    VADSpeechStart,
    VADThresholdUpdateRequested,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")


@dataclass
class _VadState:
    """Internal state container for VAD smoothing."""

    speech_run_started_at: float | None = None
    in_speech: bool = False
    speech_run_frames: int = 0
    non_speech_run_frames: int = 0


class VADManager(Manager):
    """Smooth single-frame or ordered batch VAD decisions into speech events.

    Parameters
    ----------
    event_bus : EventBus
        Session event bus.
    session_id : str
        Owning session identifier.
    models : Models
        Model container providing an optional VAD implementation.
    config : dict[str, Any] | None, optional
        Service settings. Batch frames default to 1, preserving synchronous
        event handling. Values above 1 enable a background worker; batch wait
        defaults to 128 ms and queued capacity to 256 frames.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize frame sizing, smoothing, and optional batch scheduling."""
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        # Enable only when a backend VAD model is configured
        self.vad = models.get(VAD)

        # Sampling parameters (aligned with frontend defaults: 16 kHz / 512 samples)
        self.sample_rate: int = int(self.config.get("vad_sample_rate", 16000))
        self.frame_samples: int = int(self.config.get("vad_frame_samples", 512))
        self.bytes_per_sample: int = 2  # PCM16
        self.ms_per_frame: float = (self.frame_samples * 1000.0) / self.sample_rate

        # Thresholds for smoothing (milliseconds)
        self.min_speech_ms: int = int(self.config.get("vad_min_speech_ms", 250))
        self.redemption_ms: int = int(self.config.get("vad_redemption_ms", 500))

        # Convert to frame counts
        self._min_speech_frames: int = max(
            1, round(self.min_speech_ms / self.ms_per_frame)
        )
        self._redemption_frames: int = max(
            1, round(self.redemption_ms / self.ms_per_frame)
        )

        # Buffer until enough data for a full frame
        self._buf = bytearray()
        self._st = _VadState()
        self._state_lock = asyncio.Lock()
        self._text_turn_active = False

        self._batch_frames = self.config.get("vad_batch_frames", 1)
        self._max_queue_frames = self.config.get("vad_max_queue_frames", 256)
        wait_ms = self.config.get("vad_batch_wait_ms", 128)
        if type(self._batch_frames) is not int or not 1 <= self._batch_frames <= 32:
            raise ValueError("vad_batch_frames must be an integer from 1 to 32")
        if (
            type(self._max_queue_frames) is not int
            or self._max_queue_frames < self._batch_frames
        ):
            raise ValueError("vad_max_queue_frames must be an integer covering a batch")
        if (
            isinstance(wait_ms, bool)
            or not isinstance(wait_ms, (int, float))
            or not math.isfinite(wait_ms)
            or wait_ms <= 0
        ):
            raise ValueError("vad_batch_wait_ms must be finite and positive")
        self._batch_wait = wait_ms / 1000
        self._frame_bytes = self.frame_samples * self.bytes_per_sample
        self._frame_times: deque[float] = deque()
        self._ready = asyncio.Event()
        self._request_lock = asyncio.Lock()
        self._worker: asyncio.Task[None] | None = None
        self._stopped = False
        self._epoch = 0

    # ----------------------------
    # Event handling
    # ----------------------------
    @Manager.event_handler(VADThresholdUpdateRequested)
    async def _update_threshold(self, event: VADThresholdUpdateRequested) -> None:
        """Check model availability or apply a validated threshold."""
        if self.vad is None:
            raise ValueError("No backend VAD model is configured")
        if event.check_only:
            return
        async with self._request_lock, self._state_lock:
            self.vad.set_threshold(threshold=event.threshold)

    @Manager.event_handler(EnhancedAudioFrameReceived, priority=100)
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Process audio frames and run backend VAD when enabled."""
        if (
            self._stopped
            or self.vad is None
            or self._text_turn_active
            or self.config.get("realtime_turn_detection", True) is None
        ):
            return
        if self._batch_frames > 1:
            self._enqueue_audio(event)
            return
        try:
            async with self._state_lock:
                if (
                    self._stopped
                    or self.vad is None
                    or self._text_turn_active
                    or self.config.get("realtime_turn_detection", True) is None
                ):
                    return

                if not event.audio_data:
                    return

                # Accumulate and slice into fixed-length frames
                self._buf.extend(event.audio_data)
                frame_bytes = self.frame_samples * self.bytes_per_sample

                while len(self._buf) >= frame_bytes:
                    frame = bytes(self._buf[:frame_bytes])
                    del self._buf[:frame_bytes]

                    epoch = self._epoch
                    try:
                        is_speech = bool(
                            await self._await_operation(self.vad.async_is_speech(frame))
                        )
                    except Exception:
                        logger.exception("[VADManager] VAD error")
                        is_speech = False

                    if epoch != self._epoch or self._stopped:
                        return
                    await self._advance_state(is_speech)

        except Exception:
            logger.exception("[VADManager] handle frame failed")

    @Manager.event_handler(TurnInputAbortRequested, priority=110)
    async def _on_input_abort(self, event: TurnInputAbortRequested) -> None:
        """Suppress and clear acoustic VAD state for a synthetic text turn."""
        self._text_turn_active = event.origin == "text"
        self._epoch += 1
        async with self._state_lock:
            self._clear_detection_state()
            self._ready.set()
            await self._reset_vad()

    @Manager.event_handler(VADSpeechEnd, priority=110)
    async def _on_text_vad_end(self, event: VADSpeechEnd) -> None:
        """Resume acoustic VAD after a synthetic text boundary completes."""
        if event.origin != "text" or self._stopped:
            return
        async with self._state_lock:
            self._clear_detection_state()
            self._text_turn_active = False

    async def _advance_state(
        self, is_speech: bool, *, frame_time: float | None = None
    ) -> None:
        """Advance the state machine and emit start/end events based on durations."""
        st = self._st
        detection = self.config.get("realtime_turn_detection")
        redemption_frames = self._redemption_frames
        if isinstance(detection, dict):
            redemption_frames = max(
                1, round(detection["silence_duration_ms"] / self.ms_per_frame)
            )
        if is_speech:
            if st.speech_run_frames == 0:
                st.speech_run_started_at = (
                    frame_time
                    if frame_time is not None
                    else asyncio.get_running_loop().time()
                )
            st.speech_run_frames += 1
            st.non_speech_run_frames = 0

            # Enter speech when enough speech frames accumulate
            if not st.in_speech and st.speech_run_frames >= self._min_speech_frames:
                st.in_speech = True
                await self._emit_vad_start()
        else:
            st.non_speech_run_frames += 1
            st.speech_run_frames = 0
            st.speech_run_started_at = None

            # Exit speech when enough silence frames accumulate
            if st.in_speech and st.non_speech_run_frames >= redemption_frames:
                st.in_speech = False
                await self._emit_vad_end()

    async def _emit_vad_start(self) -> None:
        """Publish VADSpeechStart event."""
        evt = VADSpeechStart(
            session_id=self.session_id,
            origin="server",
            audio_lookback_ms=(
                (asyncio.get_running_loop().time() - self._st.speech_run_started_at)
                * 1000
                + self.config.get("realtime_turn_detection", {}).get(
                    "prefix_padding_ms", 300
                )
            )
            if self._st.speech_run_started_at is not None
            else None,
        )
        await self.event_bus.publish(evt)

    async def _emit_vad_end(self) -> None:
        """Publish VADSpeechEnd and wait for ordered completion."""
        evt = VADSpeechEnd(
            session_id=self.session_id,
            origin="server",
        )
        await self.event_bus.publish(evt, mode=EventDispatchMode.WAIT_UNTIL_COMPLETE)

    def _clear_detection_state(self) -> None:
        """Clear buffered audio and duration-smoothing state."""
        self._buf.clear()
        self._frame_times.clear()
        self._st = _VadState()

    def _enqueue_audio(self, event: EnhancedAudioFrameReceived) -> None:
        """Buffer an entire audio block and timestamp newly completed frames."""
        if not event.audio_data:
            return
        if event.sample_rate != self.sample_rate:
            raise ValueError("VAD audio sample rate does not match configuration")
        if (
            len(self._buf) + len(event.audio_data)
            > self._max_queue_frames * self._frame_bytes
        ):
            raise BufferError("VAD audio queue is full")
        previous = len(self._buf) // self._frame_bytes
        self._buf.extend(event.audio_data)
        completed = len(self._buf) // self._frame_bytes - previous
        now = asyncio.get_running_loop().time()
        self._frame_times.extend([now] * completed)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_batches())
        if self._frame_times:
            self._ready.set()

    async def _wait_for_batch(self) -> None:
        """Wait only until the oldest complete frame's original deadline."""
        await self._ready.wait()
        self._ready.clear()
        while (
            not self._stopped
            and not self._text_turn_active
            and 0 < len(self._frame_times) < self._batch_frames
        ):
            remaining = (
                self._frame_times[0]
                + self._batch_wait
                - asyncio.get_running_loop().time()
            )
            if remaining <= 0:
                return
            try:
                await asyncio.wait_for(self._ready.wait(), remaining)
            except asyncio.TimeoutError:
                return
            self._ready.clear()

    async def _run_batches(self) -> None:
        """Run one ordered batch at a time and discard invalidated results."""
        while not self._stopped:
            await self._wait_for_batch()
            if self._stopped:
                return
            if self._text_turn_active or not self._frame_times:
                continue
            count = min(self._batch_frames, len(self._frame_times))
            frames = [
                bytes(self._buf[i * self._frame_bytes : (i + 1) * self._frame_bytes])
                for i in range(count)
            ]
            del self._buf[: count * self._frame_bytes]
            frame_times = [self._frame_times.popleft() for _ in range(count)]
            epoch = self._epoch
            async with self._request_lock:
                if epoch != self._epoch:
                    continue
                results = await self._detect_batch(frames)
            async with self._state_lock:
                for is_speech, frame_time in zip(results, frame_times, strict=True):
                    if epoch != self._epoch or self._stopped:
                        break
                    await self._advance_state(is_speech, frame_time=frame_time)
            if self._frame_times:
                self._ready.set()

    async def _detect_batch(self, frames: list[bytes]) -> list[bool]:
        """Validate complete results or apply the existing non-speech error policy."""
        assert self.vad is not None
        try:
            results = await self._await_operation(self.vad.async_detect_frames(frames))
            if (
                not isinstance(results, list)
                or len(results) != len(frames)
                or any(type(value) is not bool for value in results)
            ):
                raise TypeError("VAD batch results must contain one bool per frame")
            return results
        except Exception:
            logger.exception("[VADManager] VAD batch error")
            return [False] * len(frames)

    @staticmethod
    async def _await_operation(operation: Coroutine[Any, Any, _Result]) -> _Result:
        """Finish model work before propagating cancellation to its caller."""
        task = asyncio.create_task(operation)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                if cancelled:
                    raise asyncio.CancelledError from None
                raise
        if cancelled:
            # Retrieve any model exception before propagating the caller's cancellation.
            if not task.cancelled():
                task.exception()
            raise asyncio.CancelledError
        return task.result()

    async def _reset_vad(self) -> None:
        """Serialize reset with inference and finish cleanup despite cancellation."""
        if self.vad is None:
            return
        async with self._request_lock:
            try:
                self.vad.reset()
            except Exception:
                logger.exception("[VADManager] Reset VAD failed")

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """Reset VAD state and release any remote session resources."""
        self._stopped = True
        self._buf.clear()
        self._frame_times.clear()
        self._epoch += 1
        self._ready.set()
        try:
            if self._worker is not None:
                await self._worker
        except asyncio.CancelledError:
            pass  # Protected inference completes before the worker exits.
        finally:
            async with self._state_lock:
                self._clear_detection_state()
                await self._reset_vad()
