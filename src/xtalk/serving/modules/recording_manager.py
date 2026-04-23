# -*- coding: utf-8 -*-
"""
RecordingManager

Session-level audio recorder that produces a stereo WAV file:
- Left channel: raw user audio
- Right channel: TTS output

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
    FullAudioFrameReady,
    TTSChunkGenerated,
    TTSChunkPlayed,
    TTSStarted,
    SessionConfigReceived,
)


# TODO: refined time control by adding timestamps to user audio frames and TTS chunks; current implementation does not consider network latency and code execution time and may drift.
# TODO: when interrupting during an audio chunk on frontend, that chunk's tts_chunk_played event will not be sent, and the played part of that chunk will not appear in the final recording; this will result in TTS early stops before user interruption in the recording
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

        self._recording_enabled: bool = self.config.get("recording") is True
        self._send_full_audio_enabled: bool = (
            self.config.get("send_full_audio_to_client") is True
        )
        self._enabled: bool = self._recording_enabled or self._send_full_audio_enabled
        if not self._enabled:
            return

        # Defer file creation until SessionConfigReceived or first audio
        self._file_initialized = False
        self._out_dir: str = ""
        self._out_path: str = ""

        # Stereo buffers (int16 PCM bytes)
        self._ch_user = bytearray()  # Left channel: raw user input
        self._ch_tts = bytearray()  # Right channel: TTS output
        self._samples_user = 0
        self._samples_tts = 0

        # FIFO queue of pending TTS chunks until playback confirmed; each item=(pcm_bytes, sample_rate)
        self._pending_tts_chunks: list[tuple[bytes, int]] = []

        # Time-based padding: track when each channel ends (in seconds, using time.time())
        # Initialized lazily on first audio to avoid initial silence gap
        self._timer_user: Optional[float] = None  # User channel end time
        self._timer_tts: Optional[float] = None  # TTS channel end time

        # Concurrency primitives
        self._lock = asyncio.Lock()
        self._io_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = asyncio.create_task(
            self._periodic_flush_loop()
        )
        self._wf: Optional[wave.Wave_write] = None

    def _init_file(self, custom_path: str | None = None) -> None:
        """Initialize WAV file. Called lazily on first audio or config message."""
        if not self._recording_enabled or self._file_initialized:
            return

        if custom_path:
            self._out_path = custom_path
            self._out_dir = os.path.dirname(custom_path) or "."
        else:
            self._out_dir = os.path.join("logs", "session_audio")
            _ts = time.time()
            _ts_str = (
                time.strftime("%Y%m%d_%H%M%S", time.localtime(_ts))
                + f"_{int((_ts - int(_ts)) * 1000):03d}"
            )
            self._out_path = os.path.join(self._out_dir, f"{_ts_str}.wav")

        os.makedirs(self._out_dir, exist_ok=True)

        # Open WAV file for the session
        self._wf = wave.open(self._out_path, "wb")
        self._wf.setnchannels(2)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.TARGET_SR)
        self._file_initialized = True

    def _init_audio_timer(self):
        """Initialize audio timers on first audio frame (user/tts)."""
        if self._timer_user is not None and self._timer_tts is not None:
            return
        now = time.time()
        self._timer_user = now
        self._timer_tts = now

    def _init_on_audio(self):
        """Initialize recording on first audio frame if not yet initialized."""
        if self._recording_enabled and not self._file_initialized:
            self._init_file(None)
        self._init_audio_timer()

    # ==================== Event handlers ====================

    @Manager.event_handler(SessionConfigReceived, priority=100)
    async def _on_session_config(self, event: SessionConfigReceived) -> None:
        """Initialize recording with client-provided path."""
        if not self._recording_enabled or self._file_initialized:
            return
        self._init_file(event.recording_path)

    @Manager.event_handler(AudioFrameReceived, priority=50)
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        """Append raw user audio to the left channel."""
        if not self._enabled:
            return
        self._init_on_audio()
        try:
            pcm = event.audio_data or b""
            if not pcm:
                return
            src_sr = getattr(event, "sample_rate", 16000) or 16000
            data_i16 = self._resample_to_int16(pcm, src_sr, self.TARGET_SR)
            await self._append_user_audio(data_i16)
        except Exception as e:
            logger.warning("RecordingManager: failed to handle audio frame: %s", e)

    @Manager.event_handler(TTSStarted, priority=50)
    async def _on_tts_started(self, event: TTSStarted) -> None:
        """Clear pending TTS chunks when a new TTS generation starts. Because chunks played now will be from the new generation"""
        if not self._enabled:
            return
        async with self._lock:
            self._pending_tts_chunks.clear()

    @Manager.event_handler(TTSChunkGenerated, priority=50)
    async def _on_tts_chunk_generated(self, event: TTSChunkGenerated) -> None:
        """Queue generated TTS chunks until playback is confirmed."""
        if not self._enabled:
            return
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
        if not self._enabled:
            return
        self._init_on_audio()
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
        audio_duration = n / self.TARGET_SR
        async with self._lock:
            now = time.time()
            # Pad silence when elapsed time since last audio is larger than audio_duration
            silence_duration = max(0.0, now - audio_duration - self._timer_user)
            silence_samples = int(silence_duration * self.TARGET_SR)
            if silence_samples > 0:
                self._ch_user.extend(b"\x00" * (silence_samples * 2))
                self._samples_user += silence_samples

            # Append audio chunk
            self._ch_user.extend(data_i16.tobytes())
            self._samples_user += n

            # Update timer
            self._timer_user = self._timer_user + silence_duration + audio_duration

    async def _append_tts_audio(self, data_i16: np.ndarray) -> None:
        """Append TTS audio to right channel with time-based silence padding."""
        n = data_i16.size
        if n <= 0:
            return
        audio_duration = n / self.TARGET_SR
        async with self._lock:
            now = time.time()
            # Pad silence when elapsed time since last audio is larger than audio_duration
            silence_duration = max(0.0, now - audio_duration - self._timer_tts)
            silence_samples = int(silence_duration * self.TARGET_SR)
            if silence_samples > 0:
                self._ch_tts.extend(b"\x00" * (silence_samples * 2))
                self._samples_tts += silence_samples

            # Append audio chunk
            self._ch_tts.extend(data_i16.tobytes())
            self._samples_tts += n

            # Update timer
            self._timer_tts = self._timer_tts + silence_duration + audio_duration

    # ==================== Periodic flushing ====================

    async def _periodic_flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.FLUSH_INTERVAL_SEC)
                try:
                    await self._flush_outputs()
                except Exception as e:
                    logger.warning("RecordingManager: periodic flush error: %s", e)
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

    async def _drain_aligned_stereo_chunk(self) -> bytes:
        """Drain the aligned portion shared by both channels as stereo PCM."""
        async with self._lock:
            n_write = min(self._samples_user, self._samples_tts)
            if n_write <= 0:
                return b""

            bytes_len = n_write * 2
            user_bytes = bytes(self._ch_user[:bytes_len])
            tts_bytes = bytes(self._ch_tts[:bytes_len])

            del self._ch_user[:bytes_len]
            del self._ch_tts[:bytes_len]

            self._samples_user -= n_write
            self._samples_tts -= n_write
        return self._interleave_stereo(user_bytes, tts_bytes, n_write)

    async def _flush_outputs(self) -> bool:
        """Flush aligned stereo audio to every enabled sink."""
        stereo_chunk = await self._drain_aligned_stereo_chunk()
        if not stereo_chunk:
            return False
        if self._recording_enabled:
            await self._write_stereo_chunk(stereo_chunk)
        if self._send_full_audio_enabled:
            await self._publish_full_audio_chunk(stereo_chunk)
        return True

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

    def _reset_buffers(self) -> None:
        """Reset in-memory audio assembly state."""
        self._ch_user.clear()
        self._ch_tts.clear()
        self._samples_user = 0
        self._samples_tts = 0
        self._pending_tts_chunks.clear()
        self._timer_user = None
        self._timer_tts = None

    async def _build_final_stereo_chunk(self) -> bytes:
        """Build the final padded stereo chunk from any remaining buffered audio."""
        async with self._lock:
            n_samples = max(self._samples_user, self._samples_tts)
            if n_samples <= 0:
                self._reset_buffers()
                return b""
            user = np.frombuffer(bytes(self._ch_user), dtype=np.int16)
            tts = np.frombuffer(bytes(self._ch_tts), dtype=np.int16)
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
            self._reset_buffers()
            return stereo_chunk

    # ==================== Lifecycle ====================

    async def shutdown(self) -> None:
        """Finalize recording by flushing remaining buffers and closing the file."""
        if not self._enabled:
            return
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
            await self._flush_outputs()
        except Exception:
            pass

        final_chunk = await self._build_final_stereo_chunk()
        if final_chunk:
            if self._recording_enabled:
                await self._write_stereo_chunk(final_chunk)

        # Close file handle
        try:
            if self._recording_enabled and self._wf is not None:
                self._wf.close()
        finally:
            self._wf = None
