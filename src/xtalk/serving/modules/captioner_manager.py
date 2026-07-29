# -*- coding: utf-8 -*-
"""
CaptionerManager (audio caption manager).

Key responsibilities:
- Maintain a rolling audio buffer throughout the session and periodically invoke a
  Captioner to generate scene-level descriptions from cached audio.
- Cooperate with VAD/ASR: during speech we retain more context; after speech ends we
  trim back to the target window.

Inputs/assumptions:
- Frontend streams 16 kHz s16le mono PCM (AudioFrameReceived events).
- TurnTakingManager interprets VAD and dispatches turn events; this manager simply
  consumes audio frames.

Caching strategy:
- Keep a timestamped deque of audio bytes; typically store ~15 seconds but allow
  temporary growth while speech is active, then trim afterward.
- Refresh caption at a fixed interval (default 1.5s) using the entire cache, publish
  the latest CaptionUpdated event, and only keep the latest result.

Concurrency notes:
- Refresh loop runs in the background via `_refresh_loop`.
- If no captioner is configured, the manager remains idle while still buffering audio.

Events:
- Subscribe: AudioFrameReceived.
- Publish: CaptionUpdated (refresh only).
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Deque, Optional, Tuple

from ...models import Captioner, Models, Rewriter
from ..event_bus import EventBus
from ..events import (
    AudioFrameReceived,
    CaptionUpdated,
)
from ..interfaces import Manager

logger = logging.getLogger(__name__)


class CaptionerManager(Manager):
    # Audio / timing constants
    SAMPLE_RATE_HZ: int = 16000
    BYTES_PER_SAMPLE: int = 2  # s16le mono

    CACHE_SECONDS: int = 15
    REFRESH_INTERVAL_SEC: float = 1.5
    MIN_WINDOW_SEC: float = 2.5

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        models: Models,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        # Per-session config
        self.config: dict[str, Any] = config or {}

        # Check captioner/rewriter availability
        self.captioner = models.get(Captioner)
        self.caption_rewriter: Optional[Rewriter] = models.get(Rewriter)

        # Audio cache: deque of (timestamp, bytes). During speech_active, we let it grow beyond the 15s cap;
        # after speech ends we prune down to the configured limit.
        self.cache: Deque[Tuple[float, bytes]] = deque()
        self.cache_bytes_limit = (
            self.CACHE_SECONDS * self.SAMPLE_RATE_HZ * self.BYTES_PER_SAMPLE
        )
        self.cache_total_bytes = 0

        # Start refresh loop when captioner exists
        self._refresh_task: Optional[asyncio.Task] = None
        if self.captioner is not None:
            self._refresh_task = asyncio.create_task(self._refresh_loop())

    @Manager.event_handler(AudioFrameReceived, priority=80)
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        if self.captioner is None:
            return
        b = event.audio_data or b""
        if not b:
            return
        ts = time.time()
        self.cache.append((ts, b))
        self.cache_total_bytes += len(b)
        # Prune cache to target size
        while self.cache and self.cache_total_bytes > self.cache_bytes_limit:
            _, old = self.cache.popleft()
            self.cache_total_bytes -= len(old)

    async def _refresh_loop(self) -> None:
        try:
            while True:
                # Global refresh loop even without explicit VAD triggers
                await self._try_refresh()
                await asyncio.sleep(self.REFRESH_INTERVAL_SEC)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("CaptionerManager refresh loop error: %s", e)

    def _collect_all_bytes(self) -> bytes:
        """Collect entire cached audio in order (oldest -> newest)."""
        if not self.cache:
            return b""
        return b"".join(chunk for _, chunk in self.cache)

    async def _try_refresh(self) -> None:
        # Continuous refresh; no ASR end suspension

        # Use entire cache but require minimum duration
        available_sec = self.cache_total_bytes / (
            self.SAMPLE_RATE_HZ * self.BYTES_PER_SAMPLE
        )
        if available_sec < self.MIN_WINDOW_SEC:
            return

        audio_bytes = self._collect_all_bytes()
        if not audio_bytes:
            return

        try:
            text = await self.captioner.async_caption(audio_bytes)
        except Exception as e:
            logger.warning("Captioner error (refresh): %s", e)
            return
        if text:
            # Apply rewriter if configured.
            rewritten = await self._rewrite_caption(text)
            await self.event_bus.publish(
                CaptionUpdated(
                    session_id=self.session_id,
                    text=rewritten,
                    is_final=False,
                    reason="refresh",
                ),
                wait_for_completion=True,
            )

    async def _rewrite_caption(self, caption: Optional[str]) -> Optional[str]:
        """Rewrite caption if a rewriter is configured."""
        if caption is None:
            return caption
        c = caption.strip()
        if not c:
            return caption
        if not self.caption_rewriter:
            return caption
        try:
            return await self.caption_rewriter.async_rewrite(c)
        except Exception as e:
            logger.warning("Caption rewrite failed: %s", e)
            return caption

    async def shutdown(self) -> None:
        if self._refresh_task and not self._refresh_task.done():
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        self._refresh_task = None
        self.cache.clear()
        self.cache_total_bytes = 0
        self.speech_active = False
        self.utterance_start_ts = None
