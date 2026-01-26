# -*- coding: utf-8 -*-
"""
RecordingManager

Session-level audio recorder that keeps aligned stereo WAV files for both
enhanced and raw user audio against TTS output. Two files are produced:
- `logs/session_audio/<ts>.wav` (L=enhanced user, R=TTS)
- `logs/session_audio/<ts>_raw.wav` (L=raw user, R=TTS)

Highlights:
- Purely event-driven: listens to EnhancedAudioFrameReceived, AudioFrameReceived,
  TTSChunkGenerated, TTSChunkPlayedConfirm, and ConversationEnded.
- Maintains a shared timeline by padding both channels with silence before
  appending data, ensuring left/right sample counts always match.
- Buffers TTS chunks until the frontend confirms playback, dropping unconfirmed
  chunks at session end.
- Resamples everything to TARGET_SR (default 48 kHz) and periodically flushes.
- On shutdown writes any remaining data, pads silence, and closes files cleanly.
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
    EnhancedAudioFrameReceived,
    TTSChunkGenerated,
    TTSChunkPlayed,
    ConversationEnded,
)


class RecordingManager(Manager):
    """Record aligned user/TTS audio streams for each session."""

    TARGET_SR: int = 48000  # Unified output sample rate
    FLUSH_INTERVAL_SEC: float = 1.0  # Periodic flush interval

    def __init__(
        self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        # Session-level configuration
        self.config: dict[str, Any] = config or {}

        # Buffers for enhanced-aligned stereo file (int16 PCM bytes)
        self._ch_user_enh = bytearray()  # Left channel: enhanced user input
        self._ch_tts_enh = bytearray()  # Right channel: TTS output
        self._samples_user_enh = 0
        self._samples_tts_enh = 0

        # Buffers for raw-aligned stereo file
        self._ch_user_raw = bytearray()  # Left channel: raw user input
        self._ch_tts_raw = bytearray()  # Right channel: TTS output copy
        self._samples_user_raw = 0
        self._samples_tts_raw = 0

        # FIFO queue of pending TTS chunks until playback is confirmed; each item=(bytes, sr)
        self._pending_tts_chunks: list[tuple[bytes, int]] = []

        # Output directories
        self._out_dir = os.path.join("logs", "session_audio")
        os.makedirs(self._out_dir, exist_ok=True)
        # Timestamp-based prefix (milliseconds, filesystem-friendly)
        _ts = time.time()
        _ts_prefix = (
            time.strftime("%Y%m%d_%H%M%S", time.localtime(_ts))
            + f"_{int((_ts - int(_ts)) * 1000):03d}"
        )
        # Enhanced alignment output
        self._out_path = os.path.join(self._out_dir, f"{_ts_prefix}.wav")
        # Raw alignment output
        self._out_path_raw = os.path.join(self._out_dir, f"{_ts_prefix}_raw.wav")

        # Concurrency primitives
        self._lock = asyncio.Lock()  # Protect channel buffers
        self._io_lock = asyncio.Lock()  # Serialize file writes
        self._flush_task: Optional[asyncio.Task] = None
        self._wf: Optional[wave.Wave_write] = None  # Enhanced WAV handle
        self._wf_raw: Optional[wave.Wave_write] = None  # Raw WAV handle

        # Open WAV files for the entire session
        self._wf = wave.open(self._out_path, "wb")
        self._wf.setnchannels(2)
        self._wf.setsampwidth(2)
        self._wf.setframerate(self.TARGET_SR)

        self._wf_raw = wave.open(self._out_path_raw, "wb")
        self._wf_raw.setnchannels(2)
        self._wf_raw.setsampwidth(2)
        self._wf_raw.setframerate(self.TARGET_SR)

        # Start periodic flush task
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    # ==================== Event handlers ====================

    @Manager.event_handler(EnhancedAudioFrameReceived, priority=50)
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Append enhanced user audio frames to the aligned left channel."""
        try:
            pcm = event.audio_data or b""
            if not pcm:
                return
            src_sr = getattr(event, "sample_rate", 16000) or 16000
            await self._append_user_audio_enh(pcm, src_sr)
        except Exception as e:
            logger.warning("RecordingManager: failed to handle audio frame: %s", e)

    @Manager.event_handler(AudioFrameReceived, priority=40)
    async def _on_raw_audio_frame(self, event: AudioFrameReceived) -> None:
        """Append raw user audio frames to the raw-aligned left channel."""
        try:
            pcm = event.audio_data or b""
            if not pcm:
                return
            src_sr = getattr(event, "sample_rate", 16000) or 16000
            await self._append_user_audio_raw(pcm, src_sr)
        except Exception as e:
            logger.warning("RecordingManager: failed to handle raw audio frame: %s", e)

    @Manager.event_handler(TTSChunkGenerated, priority=50)
    async def _on_tts_chunk_generated(self, event: TTSChunkGenerated) -> None:
        """Cache generated TTS chunks until playback is confirmed (FIFO order)."""
        try:
            pcm = getattr(event, "audio_chunk", b"") or b""
            if not pcm:
                return
            # Default to 48 kHz; event may override in the future
            src_sr = getattr(event, "sample_rate", 48000) or 48000
            async with self._lock:
                self._pending_tts_chunks.append((pcm, src_sr))
        except Exception as e:
            logger.warning(
                "RecordingManager: failed to cache generated tts chunk: %s", e
            )

    @Manager.event_handler(TTSChunkPlayed, priority=50)
    async def _on_tts_chunk_played_confirm(self, event: TTSChunkPlayed) -> None:
        """
        Commit confirmed TTS chunks to the right channel (FIFO order).

        Assumptions:
        - This event fires after the chunk finishes playing on the client.
        - Chunks are played in the same order they were generated.
        """
        try:
            async with self._lock:
                if not self._pending_tts_chunks:
                    return
                pcm, src_sr = self._pending_tts_chunks.pop(0)
            data_i16 = self._to_int16_and_resample(pcm, src_sr, self.TARGET_SR)
            await self._append_tts_audio(data_i16)
        except Exception as e:
            logger.warning("RecordingManager: failed to commit played tts chunk: %s", e)

    @Manager.event_handler(ConversationEnded, priority=40)
    async def _on_conversation_end(self, event: ConversationEnded) -> None:
        """Drop any pending TTS chunks when the conversation ends."""
        try:
            async with self._lock:
                self._pending_tts_chunks.clear()
        except Exception as e:
            logger.warning(
                "RecordingManager: failed to clear pending tts chunks on end: %s", e
            )

    # ==================== Helpers: resampling & zero padding ====================

    def _to_int16_and_resample(
        self, pcm_bytes: bytes, src_sr: int, dst_sr: int
    ) -> np.ndarray:
        """Resample PCM int16 bytes to target sample rate and return np.int16 array."""
        if not pcm_bytes:
            return np.zeros((0,), dtype=np.int16)
        data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
        if src_sr == dst_sr or data.size == 0:
            out = np.clip(data, -32768, 32767).astype(np.int16)
            return out
        # Lightweight linear interpolation resampling
        old_n = data.size
        new_n = int(round(old_n * (dst_sr / float(src_sr))))
        if new_n <= 0:
            return np.zeros((0,), dtype=np.int16)
        x_old = np.linspace(0.0, 1.0, num=old_n, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=new_n, endpoint=False)
        resampled = np.interp(x_new, x_old, data).astype(np.float32)
        out = np.clip(resampled, -32768, 32767).astype(np.int16)
        return out

    def _append_with_other_silent(
        self,
        ch_nonzero: bytearray,
        ch_zero: bytearray,
        samples_nonzero: int,
        samples_zero: int,
        data: np.ndarray,
    ) -> tuple[int, int]:
        """
        Append real audio to one channel while padding the other with silence so both
        channels remain aligned. Returns updated sample counts.
        """
        n = int(data.size)
        if n <= 0:
            return samples_nonzero, samples_zero

        # Step 1: align to global timeline
        global_len = max(samples_nonzero, samples_zero)
        if samples_nonzero < global_len:
            diff = global_len - samples_nonzero
            ch_nonzero.extend(b"\x00" * (diff * 2))
            samples_nonzero += diff
        if samples_zero < global_len:
            diff = global_len - samples_zero
            ch_zero.extend(b"\x00" * (diff * 2))
            samples_zero += diff

        # Step 2: append real audio to one channel and silence to the other
        ch_nonzero.extend(data.tobytes())
        samples_nonzero += n

        ch_zero.extend(b"\x00" * (n * 2))
        samples_zero += n

        return samples_nonzero, samples_zero

    # ==================== Appending user & TTS audio ====================

    async def _append_user_audio_enh(self, pcm: bytes, src_sr: int) -> None:
        """Append enhanced user audio (left channel)."""
        data_i16 = self._to_int16_and_resample(pcm, src_sr, self.TARGET_SR)
        async with self._lock:
            self._samples_user_enh, self._samples_tts_enh = (
                self._append_with_other_silent(
                    self._ch_user_enh,
                    self._ch_tts_enh,
                    self._samples_user_enh,
                    self._samples_tts_enh,
                    data_i16,
                )
            )

    async def _append_user_audio_raw(self, pcm: bytes, src_sr: int) -> None:
        """Append raw user audio (left channel)."""
        data_i16 = self._to_int16_and_resample(pcm, src_sr, self.TARGET_SR)
        async with self._lock:
            self._samples_user_raw, self._samples_tts_raw = (
                self._append_with_other_silent(
                    self._ch_user_raw,
                    self._ch_tts_raw,
                    self._samples_user_raw,
                    self._samples_tts_raw,
                    data_i16,
                )
            )

    async def _append_tts_audio(self, data_i16: np.ndarray) -> None:
        """
        Append confirmed TTS audio to the right channel of both files while padding
        the opposite channel with silence so lengths stay aligned.
        """
        async with self._lock:
            # Enhanced file: TTS active, user padded
            self._samples_tts_enh, self._samples_user_enh = (
                self._append_with_other_silent(
                    self._ch_tts_enh,
                    self._ch_user_enh,
                    self._samples_tts_enh,
                    self._samples_user_enh,
                    data_i16,
                )
            )

            # Raw file: same behavior
            self._samples_tts_raw, self._samples_user_raw = (
                self._append_with_other_silent(
                    self._ch_tts_raw,
                    self._ch_user_raw,
                    self._samples_tts_raw,
                    self._samples_user_raw,
                    data_i16,
                )
            )

    # ==================== Periodic flushing & lifecycle ====================

    async def _periodic_flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.FLUSH_INTERVAL_SEC)
                try:
                    # Flush both files
                    await self._flush_to_file_enh()
                    await self._flush_to_file_raw()
                except Exception as e:
                    logger.warning("RecordingManager: periodic flush error: %s", e)
        except asyncio.CancelledError:
            pass

    async def _flush_to_file_enh(self) -> bool:
        """Flush enhanced alignment buffers to disk."""
        async with self._lock:
            ns_user = self._samples_user_enh
            ns_tts = self._samples_tts_enh
            n_write = min(ns_user, ns_tts)
            if n_write <= 0:
                return False

            bytes_len = n_write * 2
            user_bytes = bytes(self._ch_user_enh[:bytes_len])
            tts_bytes = bytes(self._ch_tts_enh[:bytes_len])

            del self._ch_user_enh[:bytes_len]
            del self._ch_tts_enh[:bytes_len]

            self._samples_user_enh -= n_write
            self._samples_tts_enh -= n_write

        loop = asyncio.get_running_loop()

        def _interleave_and_write(u_b: bytes, t_b: bytes, n_samples: int) -> None:
            u = np.frombuffer(u_b, dtype=np.int16)
            t = np.frombuffer(t_b, dtype=np.int16)
            # In theory u.size == t.size == n_samples
            inter = np.empty((n_samples * 2,), dtype=np.int16)
            inter[0::2] = u
            inter[1::2] = t
            self._wf.writeframes(inter.tobytes())

        async with self._io_lock:
            await loop.run_in_executor(
                None, _interleave_and_write, user_bytes, tts_bytes, n_write
            )
        return True

    async def _flush_to_file_raw(self) -> bool:
        """Flush raw alignment buffers to disk (mirrors enhanced logic)."""
        async with self._lock:
            ns_user = self._samples_user_raw
            ns_tts = self._samples_tts_raw
            n_write = min(ns_user, ns_tts)
            if n_write <= 0:
                return False

            bytes_len = n_write * 2
            user_bytes = bytes(self._ch_user_raw[:bytes_len])
            tts_bytes = bytes(self._ch_tts_raw[:bytes_len])

            del self._ch_user_raw[:bytes_len]
            del self._ch_tts_raw[:bytes_len]

            self._samples_user_raw -= n_write
            self._samples_tts_raw -= n_write

        loop = asyncio.get_running_loop()

        def _interleave_and_write_raw(u_b: bytes, t_b: bytes, n_samples: int) -> None:
            u = np.frombuffer(u_b, dtype=np.int16)
            t = np.frombuffer(t_b, dtype=np.int16)
            inter = np.empty((n_samples * 2,), dtype=np.int16)
            inter[0::2] = u
            inter[1::2] = t
            self._wf_raw.writeframes(inter.tobytes())

        async with self._io_lock:
            await loop.run_in_executor(
                None, _interleave_and_write_raw, user_bytes, tts_bytes, n_write
            )
        return True

    async def shutdown(self) -> None:
        """Finalize recordings by flushing remaining buffers and closing files."""
        # Stop periodic flush
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush aligned chunks first
        try:
            await self._flush_to_file_enh()
            await self._flush_to_file_raw()
        except Exception:
            pass

        # Handle any leftover samples (defensive, channels should match)
        async with self._lock:
            loop = asyncio.get_running_loop()

            # ---- Enhanced file ----
            ns_user = self._samples_user_enh
            ns_tts = self._samples_tts_enh
            ns = max(ns_user, ns_tts)
            if ns > 0:
                u = np.frombuffer(bytes(self._ch_user_enh), dtype=np.int16)
                t = np.frombuffer(bytes(self._ch_tts_enh), dtype=np.int16)
                if u.size < ns:
                    u = np.concatenate([u, np.zeros((ns - u.size,), dtype=np.int16)])
                if t.size < ns:
                    t = np.concatenate([t, np.zeros((ns - t.size,), dtype=np.int16)])
                inter = np.empty((ns * 2,), dtype=np.int16)
                inter[0::2] = u
                inter[1::2] = t

                async with self._io_lock:
                    await loop.run_in_executor(
                        None, self._wf.writeframes, inter.tobytes()
                    )

            # Clear enhanced buffers
            self._ch_user_enh.clear()
            self._ch_tts_enh.clear()
            self._samples_user_enh = 0
            self._samples_tts_enh = 0

            # ---- Raw file ----
            ns_user = self._samples_user_raw
            ns_tts = self._samples_tts_raw
            ns = max(ns_user, ns_tts)
            if ns > 0:
                u = np.frombuffer(bytes(self._ch_user_raw), dtype=np.int16)
                t = np.frombuffer(bytes(self._ch_tts_raw), dtype=np.int16)
                if u.size < ns:
                    u = np.concatenate([u, np.zeros((ns - u.size,), dtype=np.int16)])
                if t.size < ns:
                    t = np.concatenate([t, np.zeros((ns - t.size,), dtype=np.int16)])
                inter = np.empty((ns * 2,), dtype=np.int16)
                inter[0::2] = u
                inter[1::2] = t

                async with self._io_lock:
                    await loop.run_in_executor(
                        None, self._wf_raw.writeframes, inter.tobytes()
                    )

            # Clear raw buffers
            self._ch_user_raw.clear()
            self._ch_tts_raw.clear()
            self._samples_user_raw = 0
            self._samples_tts_raw = 0

        # Close file handles
        try:
            if self._wf is not None:
                self._wf.close()
            if self._wf_raw is not None:
                self._wf_raw.close()
        finally:
            self._wf = None
            self._wf_raw = None
