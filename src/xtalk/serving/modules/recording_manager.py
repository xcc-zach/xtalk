# -*- coding: utf-8 -*-
"""
RecordingManager

Session-level audio recorder that produces a stereo WAV file:
- Left channel: scheduled user audio
- Right channel: scheduled TTS audio

Recording file is initialized lazily on first audio frame or when a
SessionConfigReceived event is received. If the client sends a session_config
message with a recording_path, that path is used; otherwise the default path
logs/session_audio/<timestamp>.wav is used.

Client usage:
    ws.send(JSON.stringify({
        action: "session_config",
        recording_path: "custom/path/recording.wav"
    }));
"""

import asyncio
import logging
import os
import time
import wave
from collections import deque
from typing import Any, Optional

import numpy as np

from ..event_bus import EventBus
from ..events import (
    AudioFrameReceived,
    FullAudioFrameReady,
    SessionConfigReceived,
    TTSChunkReady,
    TTSPaused,
    TTSResumed,
    TTSStarted,
    TTSStopped,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class _MonoBuffer:
    """Sparse mono PCM buffer that keeps silence as sample counts."""

    def __init__(self) -> None:
        self._segments: deque[np.ndarray | int] = deque()
        self.total_samples: int = 0

    def clear(self) -> None:
        """Remove all buffered samples."""
        self._segments.clear()
        self.total_samples = 0

    def append_pcm(self, data_i16: np.ndarray) -> None:
        """Append PCM samples to the tail."""
        if data_i16.size <= 0:
            return
        self._segments.append(data_i16.copy())
        self.total_samples += int(data_i16.size)

    def append_silence(self, n_samples: int) -> None:
        """Append silence to the tail."""
        if n_samples <= 0:
            return
        if self._segments and isinstance(self._segments[-1], int):
            self._segments[-1] += n_samples
        else:
            self._segments.append(n_samples)
        self.total_samples += n_samples

    def truncate(self, keep_samples: int) -> None:
        """Keep only the first ``keep_samples`` samples."""
        keep = max(0, min(keep_samples, self.total_samples))
        while self.total_samples > keep and self._segments:
            tail = self._segments[-1]
            tail_samples = tail if isinstance(tail, int) else int(tail.size)
            overflow = self.total_samples - keep
            if tail_samples <= overflow:
                self._segments.pop()
                self.total_samples -= tail_samples
                continue

            remain = tail_samples - overflow
            if isinstance(tail, int):
                self._segments[-1] = remain
            else:
                self._segments[-1] = tail[:remain].copy()
            self.total_samples = keep

    def consume_prefix(self, n_samples: int) -> bytes:
        """Remove and materialize a prefix as PCM int16 bytes."""
        remaining = min(max(0, n_samples), self.total_samples)
        parts: list[bytes] = []
        while remaining > 0 and self._segments:
            head = self._segments[0]
            head_samples = head if isinstance(head, int) else int(head.size)
            take = min(remaining, head_samples)
            if isinstance(head, int):
                parts.append(b"\x00" * (take * 2))
                if take == head_samples:
                    self._segments.popleft()
                else:
                    self._segments[0] = head_samples - take
            else:
                parts.append(head[:take].tobytes())
                if take == head_samples:
                    self._segments.popleft()
                else:
                    self._segments[0] = head[take:].copy()
            self.total_samples -= take
            remaining -= take
        return b"".join(parts)

    def materialize(self) -> np.ndarray:
        """Materialize the entire buffer into a contiguous PCM array."""
        if self.total_samples <= 0:
            return np.zeros((0,), dtype=np.int16)

        out = np.zeros((self.total_samples,), dtype=np.int16)
        cursor = 0
        for segment in self._segments:
            if isinstance(segment, int):
                cursor += segment
                continue
            next_cursor = cursor + int(segment.size)
            out[cursor:next_cursor] = segment
            cursor = next_cursor
        return out


class RecordingManager(Manager):
    """Record user and TTS audio streams for each session."""

    TARGET_SR: int = 48000

    def __init__(
        self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        self._recording_enabled: bool = self.config.get("recording") is True
        self._send_full_audio_enabled: bool = (
            self.config.get("send_full_audio_to_client") is True
        )
        self._enabled: bool = self._recording_enabled or self._send_full_audio_enabled
        if not self._enabled:
            return

        self._file_initialized = False
        self._out_dir: str = ""
        self._out_path: str = ""
        self._wf: Optional[wave.Wave_write] = None

        self._user_buffer = _MonoBuffer()
        self._tts_buffer = _MonoBuffer()
        self._emitted_user_samples = 0
        self._emitted_tts_samples = 0

        self._origin_mono: Optional[float] = None

        self._user_queue: deque[np.ndarray] = deque()
        self._user_frame_samples = 0
        self._user_next_tick_mono: Optional[float] = None
        self._user_wakeup = asyncio.Event()

        self._tts_queue: deque[np.ndarray] = deque()
        self._tts_wakeup = asyncio.Event()
        self._tts_paused = False
        self._tts_stop_allowed = False
        self._tts_record_point_mono: Optional[float] = None
        self._tts_playing_chunk: Optional[np.ndarray] = None
        self._tts_playing_start_sample: Optional[int] = None
        self._tts_playing_ends_at: Optional[float] = None

        self._lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._user_task: Optional[asyncio.Task] = asyncio.create_task(
            self._user_scheduler_loop()
        )
        self._tts_task: Optional[asyncio.Task] = asyncio.create_task(
            self._tts_scheduler_loop()
        )

    def _init_file(self, custom_path: str | None = None) -> None:
        """Initialize the WAV file lazily."""
        if not self._recording_enabled or self._file_initialized:
            return

        if custom_path:
            self._out_path = custom_path
            self._out_dir = os.path.dirname(custom_path) or "."
        else:
            self._out_dir = os.path.join("logs", "session_audio")
            now_wall = time.time()
            now_str = (
                time.strftime("%Y%m%d_%H%M%S", time.localtime(now_wall))
                + f"_{int((now_wall - int(now_wall)) * 1000):03d}"
            )
            self._out_path = os.path.join(self._out_dir, f"{now_str}.wav")

        os.makedirs(self._out_dir, exist_ok=True)

        self._wf = wave.open(self._out_path, "wb")
        self._wf.setnchannels(2)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.TARGET_SR)
        self._file_initialized = True

    def _ensure_timeline_started(self, now_mono: float | None = None) -> None:
        """Start the session monotonic clock on first audio activity."""
        if self._origin_mono is None:
            self._origin_mono = time.monotonic() if now_mono is None else now_mono

    def _clock_samples_locked(self, now_mono: float | None = None) -> int:
        """Return elapsed session samples on the monotonic clock."""
        if self._origin_mono is None:
            return 0
        current = time.monotonic() if now_mono is None else now_mono
        elapsed_sec = max(0.0, current - self._origin_mono)
        return int(elapsed_sec * self.TARGET_SR)

    def _total_tts_samples_locked(self) -> int:
        """Return the absolute right-channel length in samples."""
        return self._emitted_tts_samples + self._tts_buffer.total_samples

    def _clear_tts_playback_locked(self) -> None:
        """Reset the active TTS wait state."""
        self._tts_playing_chunk = None
        self._tts_playing_start_sample = None
        self._tts_playing_ends_at = None

    def _emit_user_tick_locked(self) -> bool:
        """Emit one user cadence tick into the left channel."""
        if self._user_frame_samples <= 0:
            return False
        if self._user_queue:
            self._user_buffer.append_pcm(self._user_queue.popleft())
        else:
            self._user_buffer.append_silence(self._user_frame_samples)
        return True

    def _advance_user_until_locked(self, now_mono: float) -> None:
        """Emit every user tick whose scheduled time is due."""
        while self._user_next_tick_mono is not None and self._user_frame_samples > 0:
            if self._user_next_tick_mono > now_mono + 1e-6:
                break
            if not self._emit_user_tick_locked():
                break
            interval_sec = self._user_frame_samples / self.TARGET_SR
            self._user_next_tick_mono += interval_sec

    def _truncate_tts_buffer_locked(self, target_total_samples: int) -> None:
        """Trim the unflushed right channel to an absolute sample position."""
        keep_samples = max(0, target_total_samples - self._emitted_tts_samples)
        self._tts_buffer.truncate(keep_samples)

    def _append_tts_gap_to_now_locked(self, now_mono: float) -> None:
        """Fill right-channel silence from the last record point to now."""
        record_point_mono = self._tts_record_point_mono
        if record_point_mono is None:
            record_point_mono = self._origin_mono
        if record_point_mono is None:
            self._tts_record_point_mono = now_mono
            return

        gap_samples = max(
            0,
            self._clock_samples_locked(now_mono)
            - self._clock_samples_locked(record_point_mono),
        )
        if gap_samples > 0:
            self._tts_buffer.append_silence(gap_samples)
        self._tts_record_point_mono = now_mono

    def _start_next_tts_chunk_locked(self, now_mono: float) -> None:
        """Append the next queued TTS chunk to the right channel."""
        if not self._tts_queue:
            return
        self._append_tts_gap_to_now_locked(now_mono)
        chunk = self._tts_queue.popleft()
        start_sample = self._total_tts_samples_locked()
        self._tts_buffer.append_pcm(chunk)
        self._tts_playing_chunk = chunk
        self._tts_playing_start_sample = start_sample
        self._tts_playing_ends_at = now_mono + (chunk.size / self.TARGET_SR)

    def _pause_or_stop_current_tts_locked(
        self, now_mono: float, *, keep_remainder: bool
    ) -> np.ndarray | None:
        """Cut the active TTS chunk at the current clock position."""
        target_total = min(
            self._total_tts_samples_locked(), self._clock_samples_locked(now_mono)
        )
        remainder: np.ndarray | None = None

        if (
            self._tts_playing_chunk is not None
            and self._tts_playing_start_sample is not None
        ):
            played_samples = max(0, target_total - self._tts_playing_start_sample)
            played_samples = min(played_samples, int(self._tts_playing_chunk.size))
            target_total = self._tts_playing_start_sample + played_samples
            if keep_remainder and played_samples < self._tts_playing_chunk.size:
                remainder = self._tts_playing_chunk[played_samples:].copy()

        self._truncate_tts_buffer_locked(target_total)
        self._clear_tts_playback_locked()
        return remainder

    def _finalize_tts_to_boundary_locked(self, now_mono: float) -> None:
        """Cut pending TTS playback to the current boundary and drop future queue items."""
        self._pause_or_stop_current_tts_locked(now_mono, keep_remainder=False)
        self._tts_queue.clear()
        self._tts_paused = False
        self._tts_stop_allowed = False
        self._tts_record_point_mono = now_mono

    def _prepare_shutdown_locked(self, now_mono: float) -> None:
        """Fold queued scheduler state into channel buffers before finalization."""
        if self._user_next_tick_mono is not None:
            self._advance_user_until_locked(now_mono)
        while self._user_queue:
            self._user_buffer.append_pcm(self._user_queue.popleft())

        self._finalize_tts_to_boundary_locked(now_mono)

    def _resample_to_int16(
        self, pcm_bytes: bytes, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        """Resample PCM int16 bytes to the target sample rate."""
        if not pcm_bytes:
            return np.zeros((0,), dtype=np.int16)
        data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if src_sr == dst_sr or data.size == 0:
            return np.clip(data, -32768, 32767).astype(np.int16)

        old_n = data.size
        new_n = int(round(old_n * (dst_sr / float(src_sr))))
        if new_n <= 0:
            return np.zeros((0,), dtype=np.int16)

        x_old = np.linspace(0.0, 1.0, num=old_n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        resampled = np.interp(x_new, x_old, data)
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    async def _user_scheduler_loop(self) -> None:
        """Consume queued user audio at the latest frame cadence."""
        try:
            while True:
                timeout: float | None = None

                async with self._lock:
                    now_mono = time.monotonic()
                    self._advance_user_until_locked(now_mono)
                    if self._user_next_tick_mono is None or self._user_frame_samples <= 0:
                        self._user_wakeup.clear()
                    else:
                        timeout = max(0.0, self._user_next_tick_mono - now_mono)

                if timeout is None:
                    await self._user_wakeup.wait()
                    continue

                try:
                    await asyncio.wait_for(self._user_wakeup.wait(), timeout=timeout)
                    self._user_wakeup.clear()
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    async def _tts_scheduler_loop(self) -> None:
        """Consume queued TTS audio according to the monotonic clock."""
        try:
            while True:
                timeout: float | None = None
                should_flush = False

                async with self._lock:
                    while True:
                        now_mono = time.monotonic()

                        if self._tts_paused:
                            self._tts_wakeup.clear()
                            break

                        if (
                            self._tts_playing_ends_at is not None
                            and now_mono + 1e-6 >= self._tts_playing_ends_at
                        ):
                            self._clear_tts_playback_locked()
                            self._tts_record_point_mono = now_mono
                            if not self._tts_queue:
                                should_flush = True
                                self._tts_wakeup.clear()
                                break
                            continue

                        if self._tts_playing_ends_at is not None:
                            timeout = max(0.0, self._tts_playing_ends_at - now_mono)
                            break

                        if self._tts_queue:
                            self._ensure_timeline_started(now_mono)
                            self._start_next_tts_chunk_locked(now_mono)
                            timeout = max(0.0, self._tts_playing_ends_at - now_mono)
                            break

                        self._tts_wakeup.clear()
                        break

                if should_flush:
                    await self._flush_all_outputs()
                    continue

                if timeout is None:
                    await self._tts_wakeup.wait()
                    continue

                try:
                    await asyncio.wait_for(self._tts_wakeup.wait(), timeout=timeout)
                    self._tts_wakeup.clear()
                except asyncio.TimeoutError:
                    continue
        except asyncio.CancelledError:
            pass

    def _interleave_stereo(
        self, user_bytes: bytes, tts_bytes: bytes, n_samples: int
    ) -> bytes:
        """Interleave user/TTS mono PCM bytes into stereo int16 PCM."""
        if n_samples <= 0:
            return b""
        user = np.frombuffer(user_bytes, dtype=np.int16)
        tts = np.frombuffer(tts_bytes, dtype=np.int16)
        interleaved = np.empty((n_samples * 2,), dtype=np.int16)
        interleaved[0::2] = user
        interleaved[1::2] = tts
        return interleaved.tobytes()

    @staticmethod
    def _pad_pcm_bytes(pcm_bytes: bytes, n_samples: int) -> bytes:
        """Pad mono PCM bytes with trailing silence to the requested length."""
        target_len = n_samples * 2
        if len(pcm_bytes) >= target_len:
            return pcm_bytes
        return pcm_bytes + (b"\x00" * (target_len - len(pcm_bytes)))

    async def _drain_aligned_stereo_chunk(self) -> bytes:
        """Drain buffered audio and pad the shorter channel only in the output."""
        async with self._lock:
            user_samples = self._user_buffer.total_samples
            tts_samples = self._tts_buffer.total_samples
            n_write = max(user_samples, tts_samples)
            if n_write <= 0:
                return b""

            user_bytes = self._user_buffer.consume_prefix(n_write)
            tts_bytes = self._tts_buffer.consume_prefix(n_write)
            self._emitted_user_samples += user_samples
            self._emitted_tts_samples += tts_samples
            user_bytes = self._pad_pcm_bytes(user_bytes, n_write)
            tts_bytes = self._pad_pcm_bytes(tts_bytes, n_write)
        return self._interleave_stereo(user_bytes, tts_bytes, n_write)

    async def _flush_outputs(self) -> bool:
        """Flush one aligned stereo chunk to every enabled sink."""
        stereo_chunk = await self._drain_aligned_stereo_chunk()
        if not stereo_chunk:
            return False
        if self._recording_enabled:
            await self._write_stereo_chunk(stereo_chunk)
        if self._send_full_audio_enabled:
            await self._publish_full_audio_chunk(stereo_chunk)
        return True

    async def _flush_all_outputs(self) -> None:
        """Flush every aligned stereo chunk currently available."""
        while await self._flush_outputs():
            pass

    async def _write_stereo_chunk(self, stereo_chunk: bytes) -> None:
        """Write a stereo PCM chunk into the WAV file."""
        if not stereo_chunk:
            return
        if self._wf is None:
            self._init_file(None)
        if self._wf is None:
            return
        loop = asyncio.get_running_loop()
        async with self._io_lock:
            await loop.run_in_executor(None, self._wf.writeframes, stereo_chunk)

    async def _publish_full_audio_chunk(self, stereo_chunk: bytes) -> None:
        """Publish a full-conversation stereo PCM chunk for downstream transport."""
        if not stereo_chunk:
            return
        await self.event_bus.publish(
            FullAudioFrameReady(
                session_id=self.session_id,
                audio_chunk=stereo_chunk,
                sample_rate=self.TARGET_SR,
                channels=2,
                format="pcm_s16le",
            ),
            wait_for_completion=True,
        )

    def _reset_buffers_locked(self) -> None:
        """Reset in-memory buffers after the final chunk is built."""
        self._user_buffer.clear()
        self._tts_buffer.clear()
        self._emitted_user_samples = 0
        self._emitted_tts_samples = 0
        self._user_queue.clear()
        self._user_frame_samples = 0
        self._user_next_tick_mono = None
        self._tts_queue.clear()
        self._tts_paused = False
        self._tts_stop_allowed = False
        self._tts_record_point_mono = None
        self._origin_mono = None
        self._clear_tts_playback_locked()

    async def _build_final_stereo_chunk(self) -> bytes:
        """Build the final padded stereo chunk from any remaining buffered audio."""
        async with self._lock:
            n_samples = max(self._user_buffer.total_samples, self._tts_buffer.total_samples)
            if n_samples <= 0:
                self._reset_buffers_locked()
                return b""

            user = self._user_buffer.materialize()
            tts = self._tts_buffer.materialize()
            if user.size < n_samples:
                user = np.concatenate(
                    [user, np.zeros((n_samples - user.size,), dtype=np.int16)]
                )
            if tts.size < n_samples:
                tts = np.concatenate(
                    [tts, np.zeros((n_samples - tts.size,), dtype=np.int16)]
                )

            stereo_chunk = self._interleave_stereo(
                user.tobytes(), tts.tobytes(), n_samples
            )
            self._reset_buffers_locked()
            return stereo_chunk

    @Manager.event_handler(SessionConfigReceived, priority=100)
    async def _on_session_config(self, event: SessionConfigReceived) -> None:
        """Initialize recording with a client-provided path."""
        if not self._recording_enabled or self._file_initialized:
            return
        self._init_file(event.recording_path)

    @Manager.event_handler(AudioFrameReceived, priority=50)
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        """Queue user audio for left-channel scheduling."""
        if not self._enabled:
            return
        try:
            pcm = event.audio_data or b""
            if not pcm:
                return

            src_sr = event.sample_rate or 16000
            data_i16 = self._resample_to_int16(pcm, src_sr, self.TARGET_SR)
            if data_i16.size <= 0:
                return

            async with self._lock:
                now_mono = time.monotonic()
                self._ensure_timeline_started(now_mono)
                self._advance_user_until_locked(now_mono)
                self._user_frame_samples = int(data_i16.size)
                self._user_queue.append(data_i16)
                if self._user_next_tick_mono is None:
                    self._user_next_tick_mono = (
                        now_mono + (self._user_frame_samples / self.TARGET_SR)
                    )
                self._user_wakeup.set()
        except Exception as e:
            logger.warning("RecordingManager: failed to handle audio frame: %s", e)

    @Manager.event_handler(TTSStarted, priority=50)
    async def _on_tts_started(self, event: TTSStarted) -> None:
        """Wake the TTS scheduler when a new generation starts."""
        if not self._enabled:
            return
        self._tts_wakeup.set()

    @Manager.event_handler(TTSChunkReady, priority=50)
    async def _on_tts_chunk_generated(self, event: TTSChunkReady) -> None:
        """Queue generated TTS chunks for right-channel scheduling."""
        if not self._enabled:
            return
        try:
            pcm = event.audio_chunk or b""
            if not pcm:
                return

            src_sr = event.sample_rate or 48000
            data_i16 = self._resample_to_int16(pcm, src_sr, self.TARGET_SR)
            if data_i16.size <= 0:
                return

            async with self._lock:
                now_mono = time.monotonic()
                self._ensure_timeline_started(now_mono)
                self._tts_queue.append(data_i16)
                self._tts_stop_allowed = True
                self._tts_wakeup.set()
        except Exception as e:
            logger.warning("RecordingManager: failed to queue TTS chunk: %s", e)

    @Manager.event_handler(TTSPaused, priority=50)
    async def _on_tts_paused(self, event: TTSPaused) -> None:
        """Pause TTS scheduling and cut the active chunk at the current clock."""
        if not self._enabled:
            return

        async with self._lock:
            if self._tts_paused or not self._tts_stop_allowed:
                return
            if self._tts_playing_ends_at is None and not self._tts_queue:
                return

            now_mono = time.monotonic()
            remainder = self._pause_or_stop_current_tts_locked(
                now_mono, keep_remainder=True
            )
            if remainder is not None and remainder.size > 0:
                self._tts_queue.appendleft(remainder)
            self._tts_paused = True
            self._tts_record_point_mono = now_mono
            self._tts_wakeup.set()

    @Manager.event_handler(TTSResumed, priority=50)
    async def _on_tts_resumed(self, event: TTSResumed) -> None:
        """Resume TTS scheduling from the paused point."""
        if not self._enabled:
            return

        should_flush = False
        async with self._lock:
            if not self._tts_paused:
                return

            now_mono = time.monotonic()
            self._ensure_timeline_started(now_mono)
            self._append_tts_gap_to_now_locked(now_mono)
            self._tts_paused = False

            if self._tts_playing_ends_at is None and not self._tts_queue:
                should_flush = True

            self._tts_wakeup.set()

        if should_flush:
            await self._flush_all_outputs()

    @Manager.event_handler(TTSStopped, priority=50)
    async def _on_tts_stopped(self, event: TTSStopped) -> None:
        """Stop TTS scheduling, truncate the active chunk, and flush outputs."""
        if not self._enabled:
            return
        try:
            async with self._lock:
                if not self._tts_stop_allowed:
                    return

                now_mono = time.monotonic()
                self._finalize_tts_to_boundary_locked(now_mono)
                self._tts_wakeup.set()
            await self._flush_all_outputs()
        except Exception as e:
            logger.warning("RecordingManager: failed to handle TTS stopped: %s", e)

    async def shutdown(self) -> None:
        """Finalize recording by flushing remaining buffers and closing the file."""
        if not self._enabled:
            return

        for task in (self._user_task, self._tts_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._user_task = None
        self._tts_task = None

        async with self._lock:
            self._prepare_shutdown_locked(time.monotonic())

        try:
            await self._flush_all_outputs()
        except Exception:
            pass

        final_chunk = await self._build_final_stereo_chunk()
        if final_chunk:
            if self._recording_enabled:
                await self._write_stereo_chunk(final_chunk)
            if self._send_full_audio_enabled:
                await self._publish_full_audio_chunk(final_chunk)

        try:
            if self._recording_enabled and self._wf is not None:
                self._wf.close()
        finally:
            self._wf = None
