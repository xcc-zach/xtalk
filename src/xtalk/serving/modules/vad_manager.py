# -*- coding: utf-8 -*-
"""
VADManager

Backend VAD manager used when the frontend cannot run VAD. It mimics the logic in
`frontend/src/index.js` using a configured VAD model:

- Process frames at 16 kHz with 512 samples (~32 ms/frame)
- Speech start triggers when speech frames accumulate beyond `min_speech_ms`
- Speech end triggers when silence frames exceed `redemption_ms`
- Emits `VADSpeechStart`/`VADSpeechEnd`; speech end is published with
  `wait_for_completion=True` so higher-priority handlers can flush state first.

Notes:
- Relies on `VAD` interface (`is_speech(frame: bytes) -> bool`) from
  `src/xtalk/models/*/interfaces.py`.
- Uses duration smoothing on boolean outputs instead of probability thresholds.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from ...models import VAD, Models
from ..event_bus import EventBus
from ..events import (
    EnhancedAudioFrameReceived,
    TurnInputAbortRequested,
    VADSpeechEnd,
    VADSpeechStart,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


@dataclass
class _VadState:
    """Internal state container for VAD smoothing."""

    in_speech: bool = False
    speech_run_frames: int = 0
    non_speech_run_frames: int = 0


class VADManager(Manager):
    """Backend VAD manager."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
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
            1, int(round(self.min_speech_ms / self.ms_per_frame))
        )
        self._redemption_frames: int = max(
            1, int(round(self.redemption_ms / self.ms_per_frame))
        )

        # Buffer until enough data for a full frame
        self._buf = bytearray()
        self._st = _VadState()
        self._state_lock = asyncio.Lock()
        self._text_turn_active = False

    # ----------------------------
    # Event handling
    # ----------------------------
    @Manager.event_handler(EnhancedAudioFrameReceived, priority=100)
    async def _on_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Process audio frames and run backend VAD when enabled."""
        try:
            async with self._state_lock:
                if self.vad is None or self._text_turn_active:
                    return

                if not event.audio_data:
                    return

                # Accumulate and slice into fixed-length frames
                self._buf.extend(event.audio_data)
                frame_bytes = self.frame_samples * self.bytes_per_sample

                while len(self._buf) >= frame_bytes:
                    frame = bytes(self._buf[:frame_bytes])
                    del self._buf[:frame_bytes]

                    try:
                        is_speech = bool(await self.vad.async_is_speech(frame))
                    except Exception as e:
                        logger.error("[VADManager] VAD error: %s", e)
                        is_speech = False

                    await self._advance_state(is_speech)

        except Exception as e:
            logger.error("[VADManager] handle frame failed: %s", e)

    @Manager.event_handler(TurnInputAbortRequested, priority=110)
    async def _on_input_abort(self, event: TurnInputAbortRequested) -> None:
        """Suppress and clear acoustic VAD state for a synthetic text turn."""
        if event.origin != "text":
            return
        async with self._state_lock:
            self._text_turn_active = True
            self._clear_detection_state()
            if self.vad is not None:
                self.vad.reset()

    @Manager.event_handler(VADSpeechEnd, priority=110)
    async def _on_text_vad_end(self, event: VADSpeechEnd) -> None:
        """Resume acoustic VAD after a synthetic text boundary completes."""
        if event.origin != "text":
            return
        async with self._state_lock:
            self._clear_detection_state()
            self._text_turn_active = False

    async def _advance_state(self, is_speech: bool) -> None:
        """Advance the state machine and emit start/end events based on durations."""
        st = self._st
        if is_speech:
            st.speech_run_frames += 1
            st.non_speech_run_frames = 0

            # Enter speech when enough speech frames accumulate
            if not st.in_speech and st.speech_run_frames >= self._min_speech_frames:
                st.in_speech = True
                await self._emit_vad_start()
        else:
            st.non_speech_run_frames += 1
            st.speech_run_frames = 0

            # Exit speech when enough silence frames accumulate
            if st.in_speech and st.non_speech_run_frames >= self._redemption_frames:
                st.in_speech = False
                await self._emit_vad_end()

    async def _emit_vad_start(self) -> None:
        """Publish VADSpeechStart event."""
        evt = VADSpeechStart(
            session_id=self.session_id,
            origin="server",
        )
        await self.event_bus.publish(evt)

    async def _emit_vad_end(self) -> None:
        """Publish VADSpeechEnd and wait for ordered completion."""
        evt = VADSpeechEnd(
            session_id=self.session_id,
            origin="server",
        )
        await self.event_bus.publish(evt, wait_for_completion=True)

    def _clear_detection_state(self) -> None:
        """Clear buffered audio and duration-smoothing state."""
        self._buf.clear()
        self._st = _VadState()

    # ----------------------------
    # Lifecycle
    # ----------------------------
    async def shutdown(self) -> None:  # type: ignore[override]
        """Reset VAD state and release any remote session resources."""
        self._text_turn_active = False
        self._clear_detection_state()
        if self.vad is not None:
            try:
                self.vad.reset()
            except Exception as e:
                logger.error("[VADManager] Reset VAD failed: %s", e)
        return None
