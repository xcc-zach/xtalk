# -*- coding: utf-8 -*-
"""
RecordingManager

Session-level audio recorder that produces a stereo WAV file:
- Left channel: raw user audio
- Right channel: TTS output

Recording starts on WebSocket connection (__init__) and stops on shutdown().
Audio is saved to logs/session_audio/<timestamp>.wav.
"""

import os
import time
import wave
import asyncio
from typing import Optional, Any

import numpy as np

from ...log_utils import logger
from ..event_bus import EventBus
from ..interfaces import Manager
from ..events import (
    AudioFrameReceived,
    TTSChunkGenerated,
    TTSChunkPlayed,
)


class RecordingManager(Manager):
    """Record user and TTS audio streams for each session."""

    TARGET_SR: int = 48000  # Unified output sample rate
    FLUSH_INTERVAL_SEC: float = 10.0  # Periodic flush interval

    def __init__(
        self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        # Stereo buffers (int16 PCM bytes)
        self._ch_user = bytearray()  # Left channel: raw user input
        self._ch_tts = bytearray()  # Right channel: TTS output
        self._samples_user = 0
        self._samples_tts = 0

        # FIFO queue of pending TTS chunks until playback confirmed; each item=(pcm_bytes, sample_rate)
        self._pending_tts_chunks: list[tuple[bytes, int]] = []

        # Time-based padding: track when each channel ends (in seconds, using time.time())
        _now = time.time()
        self._timer_user: float = _now  # User channel end time
        self._timer_tts: float = _now  # TTS channel end time

        # Output directory and file path
        self._out_dir = os.path.join("logs", "session_audio")
        os.makedirs(self._out_dir, exist_ok=True)

        # Timestamp-based filename
        _ts = time.time()
        _ts_str = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(_ts))
            + f"_{int((_ts - int(_ts)) * 1000):03d}"
        )
        self._out_path = os.path.join(self._out_dir, f"{_ts_str}.wav")

        # Concurrency primitives
        self._lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None

        # Open WAV file for the session
        self._wf: Optional[wave.Wave_write] = wave.open(self._out_path, "wb")
        self._wf.setnchannels(2)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.TARGET_SR)

        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    # ==================== Event handlers ====================

    @Manager.event_handler(AudioFrameReceived, priority=50)
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        """Append raw user audio to the left channel."""
        try:
            pcm = event.audio_data or b""
            if not pcm:
                return
            src_sr = getattr(event, "sample_rate", 16000) or 16000
            data_i16 = self._resample_to_int16(pcm, src_sr, self.TARGET_SR)
            await self._append_user_audio(data_i16)
        except Exception as e:
            logger.warning("RecordingManager: failed to handle audio frame: %s", e)

    @Manager.event_handler(TTSChunkGenerated, priority=50)
    async def _on_tts_chunk_generated(self, event: TTSChunkGenerated) -> None:
        """Queue generated TTS chunks until playback is confirmed."""
        try:
            pcm = getattr(event, "audio_chunk", b"") or b""
            if not pcm:
                return
            src_sr = getattr(event, "sample_rate", 48000) or 48000
            async with self._lock:
                self._pending_tts_chunks.append((pcm, src_sr))
        except Exception as e:
            logger.warning("RecordingManager: failed to queue TTS chunk: %s", e)

    @Manager.event_handler(TTSChunkPlayed, priority=50)
    async def _on_tts_chunk_played(self, event: TTSChunkPlayed) -> None:
        """Pop one TTS chunk from queue and append to the right channel."""
        try:
            async with self._lock:
                if not self._pending_tts_chunks:
                    return
                pcm, src_sr = self._pending_tts_chunks.pop(0)
            data_i16 = self._resample_to_int16(pcm, src_sr, self.TARGET_SR)
            await self._append_tts_audio(data_i16)
        except Exception as e:
            logger.warning("RecordingManager: failed to handle TTS chunk played: %s", e)

    # ==================== Resampling ====================

    def _resample_to_int16(
        self, pcm_bytes: bytes, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        """Resample PCM int16 bytes to target sample rate."""
        if not pcm_bytes:
            return np.zeros((0,), dtype=np.int16)
        data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if src_sr == dst_sr or data.size == 0:
            return np.clip(data, -32768, 32767).astype(np.int16)
        # Linear interpolation resampling
        old_n = data.size
        new_n = int(round(old_n * (dst_sr / float(src_sr))))
        if new_n <= 0:
            return np.zeros((0,), dtype=np.int16)
        x_old = np.linspace(0.0, 1.0, num=old_n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        resampled = np.interp(x_new, x_old, data)
        return np.clip(resampled, -32768, 32767).astype(np.int16)

    # ==================== Channel append with time-based silence padding ====================

    async def _append_user_audio(self, data_i16: np.ndarray) -> None:
        """Append user audio to left channel with time-based silence padding."""
        n = data_i16.size
        if n <= 0:
            return
        async with self._lock:
            now = time.time()
            # Pad silence for elapsed time since last audio
            silence_duration = max(0.0, now - self._timer_user)
            silence_samples = int(silence_duration * self.TARGET_SR)
            if silence_samples > 0:
                self._ch_user.extend(b"\x00" * (silence_samples * 2))
                self._samples_user += silence_samples

            # Append audio chunk
            self._ch_user.extend(data_i16.tobytes())
            self._samples_user += n

            # Update timer: current time + audio duration
            audio_duration = n / self.TARGET_SR
            self._timer_user = now + audio_duration

    async def _append_tts_audio(self, data_i16: np.ndarray) -> None:
        """Append TTS audio to right channel with time-based silence padding."""
        n = data_i16.size
        if n <= 0:
            return
        async with self._lock:
            now = time.time()
            # Pad silence for elapsed time since last audio
            silence_duration = max(0.0, now - self._timer_tts)
            silence_samples = int(silence_duration * self.TARGET_SR)
            if silence_samples > 0:
                self._ch_tts.extend(b"\x00" * (silence_samples * 2))
                self._samples_tts += silence_samples

            # Append audio chunk
            self._ch_tts.extend(data_i16.tobytes())
            self._samples_tts += n

            # Update timer: current time + audio duration
            audio_duration = n / self.TARGET_SR
            self._timer_tts = now + audio_duration

    # ==================== Periodic flushing ====================

    async def _periodic_flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.FLUSH_INTERVAL_SEC)
                try:
                    await self._flush_to_file()
                except Exception as e:
                    logger.warning("RecordingManager: periodic flush error: %s", e)
        except asyncio.CancelledError:
            pass

    async def _flush_to_file(self) -> bool:
        """Flush aligned buffers to disk."""
        async with self._lock:
            n_write = min(self._samples_user, self._samples_tts)
            if n_write <= 0:
                return False

            bytes_len = n_write * 2
            user_bytes = bytes(self._ch_user[:bytes_len])
            tts_bytes = bytes(self._ch_tts[:bytes_len])

            del self._ch_user[:bytes_len]
            del self._ch_tts[:bytes_len]

            self._samples_user -= n_write
            self._samples_tts -= n_write

        loop = asyncio.get_running_loop()

        def _interleave_and_write(u_b: bytes, t_b: bytes, n_samples: int) -> None:
            u = np.frombuffer(u_b, dtype=np.int16)
            t = np.frombuffer(t_b, dtype=np.int16)
            inter = np.empty((n_samples * 2,), dtype=np.int16)
            inter[0::2] = u
            inter[1::2] = t
            self._wf.writeframes(inter.tobytes())

        async with self._io_lock:
            await loop.run_in_executor(
                None, _interleave_and_write, user_bytes, tts_bytes, n_write
            )
        return True

    # ==================== Lifecycle ====================

    async def shutdown(self) -> None:
        """Finalize recording by flushing remaining buffers and closing the file."""
        # Stop periodic flush
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Final flush
        try:
            await self._flush_to_file()
        except Exception:
            pass

        # Handle any leftover samples
        async with self._lock:
            ns = max(self._samples_user, self._samples_tts)
            if ns > 0:
                u = np.frombuffer(bytes(self._ch_user), dtype=np.int16)
                t = np.frombuffer(bytes(self._ch_tts), dtype=np.int16)
                # Pad to match lengths
                if u.size < ns:
                    u = np.concatenate([u, np.zeros((ns - u.size,), dtype=np.int16)])
                if t.size < ns:
                    t = np.concatenate([t, np.zeros((ns - t.size,), dtype=np.int16)])
                inter = np.empty((ns * 2,), dtype=np.int16)
                inter[0::2] = u
                inter[1::2] = t

                loop = asyncio.get_running_loop()
                async with self._io_lock:
                    await loop.run_in_executor(
                        None, self._wf.writeframes, inter.tobytes()
                    )

            # Clear buffers
            self._ch_user.clear()
            self._ch_tts.clear()
            self._samples_user = 0
            self._samples_tts = 0
            self._pending_tts_chunks.clear()
            self._timer_user = 0.0
            self._timer_tts = 0.0

        # Close file handle
        try:
            if self._wf is not None:
                self._wf.close()
        finally:
            self._wf = None
