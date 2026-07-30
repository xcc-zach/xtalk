# -*- coding: utf-8 -*-
"""
Session-level latency tracker with clock synchronization support.

Captured metrics:
1. network_latency: frontend VAD start → backend VAD start (clock-synced)
2. asr_latency: backend VAD end → ASR final result
3. llm_first_token: ASR final → first LLM chunk/tool call
4. llm_sentence: ASR final → first synthesizable LLM sentence
5. tts_first_chunk: sentence ready → first TTS audio chunk
6. e2e_latency: VAD end → first TTS chunk

Clock sync (NTP-style):
- Ping/pong heartbeats to estimate RTT and offset
- Keep last ~5 samples using median for stability
- Correct network latency with measured offset

Publishes `LatencyMetricsUpdated` for the OutputGateway to forward to clients.
"""

from __future__ import annotations

from typing import Any, Optional

from ..event_bus import EventBus
from ..events import (
    VADSpeechStart,
    VADSpeechEnd,
    ASRResultFinal,
    LLMFirstChunk,
    LLMFirstSentence,
    TTSChunkReady,
    LatencyMetricsUpdated,
    WebSocketMessageReceived,
    ClockSyncReceived,
)
from ..interfaces import EventListenerMixin, Manager


class LatencyManager(EventListenerMixin):
    """Per-session latency tracker that listens to VAD/ASR/LLM/TTS events."""

    def __init__(
        self, event_bus: EventBus, session_id: str, config: dict[str, Any] | None = None
    ) -> None:
        # Session metadata
        self.event_bus = event_bus
        self.session_id = session_id
        self.config: dict[str, Any] = config or {}

        # Timestamp bookkeeping (seconds)
        self._frontend_vad_start_ts: Optional[float] = (
            None  # t0_front: frontend VAD start timestamp
        )
        self._backend_vad_start_ts: Optional[float] = (
            None  # t0_back: backend VAD start timestamp
        )

        self._backend_vad_end_recv_ts: Optional[float] = None

        self._vad_end_ts: Optional[float] = None  # t1
        self._asr_final_ts: Optional[float] = None  # t2
        self._llm_first_token_ts: Optional[float] = None  # t3
        self._llm_sentence_ts: Optional[float] = None  # t4
        self._tts_first_chunk_ts: Optional[float] = None  # t5

        # Clock sync state
        self._clock_offset: Optional[float] = (
            None  # Offset in seconds: client_time - server_time
        )
        self._clock_sync_samples: list[float] = []  # Recent offset samples
        self._max_sync_samples: int = 5  # Keep last 5 samples

        # Prevent duplicate reports
        self._reported_for_turn: bool = False

    def update_clock_offset(
        self, client_send_ts: float, server_recv_ts: float, client_recv_ts: float
    ) -> None:
        """
        Update the clock offset estimate using an NTP-style ping/pong exchange.

        client_send_ts = T1, server_recv_ts = T2, client_recv_ts = T4.
        Offset = T2 - (T1 + T4)/2 and we track a rolling median for stability.
        """
        # Round-trip time
        rtt = client_recv_ts - client_send_ts

        # NTP offset (server time minus client midpoint)
        offset = server_recv_ts - (client_send_ts + client_recv_ts) / 2.0

        # Store sample
        self._clock_sync_samples.append(offset)

        # Keep recent samples only
        if len(self._clock_sync_samples) > self._max_sync_samples:
            self._clock_sync_samples.pop(0)

        # Use median for robustness when enough samples exist
        if len(self._clock_sync_samples) >= 3:
            sorted_samples = sorted(self._clock_sync_samples)
            self._clock_offset = sorted_samples[len(sorted_samples) // 2]
        else:
            self._clock_offset = offset

    @Manager.event_handler(ClockSyncReceived, priority=50)
    async def _on_clock_sync_received(self, event: ClockSyncReceived) -> None:
        """Handle clock-sync events and update offset estimates."""
        self.update_clock_offset(
            event.client_send_ts, event.server_recv_ts, event.client_recv_ts
        )

    @Manager.event_handler(
        WebSocketMessageReceived,
        priority=60,
    )
    async def _on_websocket_message_received(
        self, event: WebSocketMessageReceived
    ) -> None:
        """Capture frontend-provided VAD start timestamps from WebSocket payloads."""
        try:
            import json

            message_data = json.loads(event.message)
            action = message_data.get("action") or message_data.get("type", "")

            # Capture frontend-provided VAD start timestamp
            if action == "vad_speech_start" and "timestamp" in message_data:
                # Convert milliseconds to seconds
                self._frontend_vad_start_ts = message_data["timestamp"] / 1000.0
                # Record backend receive time
                self._backend_vad_start_ts = event.timestamp
            if action == "vad_speech_end":
                self._backend_vad_end_recv_ts = event.timestamp
        except (json.JSONDecodeError, KeyError, TypeError):
            # Ignore malformed payloads
            pass

    @Manager.event_handler(VADSpeechStart, priority=50)
    async def _on_vad_start(self, event: VADSpeechStart) -> None:
        """Reset per-turn timing when VAD starts."""
        if event.origin != "client":
            self._frontend_vad_start_ts = None
            self._backend_vad_start_ts = None
        self._vad_end_ts = None
        self._backend_vad_end_recv_ts = None
        self._asr_final_ts = None
        self._llm_first_token_ts = None
        self._llm_sentence_ts = None
        self._tts_first_chunk_ts = None
        self._reported_for_turn = False

    @Manager.event_handler(VADSpeechEnd, priority=50)
    async def _on_vad_end(self, event: VADSpeechEnd) -> None:
        """Record backend VAD end timestamp (t1)."""
        self._vad_end_ts = event.timestamp

    @Manager.event_handler(ASRResultFinal, priority=50)
    async def _on_asr_final(self, event: ASRResultFinal) -> None:
        """Record ASR completion timestamp (t2)."""
        self._asr_final_ts = event.timestamp

    @Manager.event_handler(LLMFirstChunk, priority=50)
    async def _on_llm_first_token(self, event: LLMFirstChunk) -> None:
        """Record LLM first token timestamp (t3)."""
        if self._llm_first_token_ts is None:
            self._llm_first_token_ts = event.timestamp

    @Manager.event_handler(LLMFirstSentence, priority=50)
    async def _on_llm_sentence_ready(self, event: LLMFirstSentence) -> None:
        """Record LLM sentence-ready timestamp (t4)."""
        if self._llm_sentence_ts is None:
            self._llm_sentence_ts = event.timestamp

    @Manager.event_handler(TTSChunkReady, priority=50)
    async def _on_tts_chunk_generated(self, event: TTSChunkReady) -> None:
        """Record TTS first chunk timestamp (t5) and compute latency metrics."""
        if self._reported_for_turn:
            return

        # Record TTS first chunk timestamp
        if self._tts_first_chunk_ts is None:
            self._tts_first_chunk_ts = event.timestamp

        # Compute latencies (milliseconds)
        network_latency = 0
        asr_latency = 0
        llm_first_token = 0
        llm_sentence = 0
        tts_first_chunk = 0
        e2e_latency = 0

        # 1. Network latency: backend VAD start - frontend VAD start (offset corrected)
        if self._frontend_vad_start_ts and self._backend_vad_start_ts:
            # Raw measurement (contains offset)
            raw_latency = (
                self._backend_vad_start_ts - self._frontend_vad_start_ts
            ) * 1000.0

            # Adjust when offset is known
            if self._clock_offset is not None:
                # Corrected = measured latency - offset
                corrected_latency = raw_latency - (self._clock_offset * 1000.0)
                network_latency = int(max(0.0, corrected_latency))
            else:
                network_latency = int(max(0.0, raw_latency))

        # 2. ASR latency: t2 - t1
        vad_end_ts = self._backend_vad_end_recv_ts or self._vad_end_ts
        if vad_end_ts and self._asr_final_ts:
            asr_latency = int(max(0.0, (self._asr_final_ts - vad_end_ts) * 1000.0))

        # 3. LLM first-token latency: t3 - t2
        if self._asr_final_ts and self._llm_first_token_ts:
            llm_first_token = int(
                max(0.0, (self._llm_first_token_ts - self._asr_final_ts) * 1000.0)
            )

        # 4. LLM sentence latency: t4 - t2
        if self._asr_final_ts and self._llm_sentence_ts:
            llm_sentence = int(
                max(0.0, (self._llm_sentence_ts - self._asr_final_ts) * 1000.0)
            )

        # 5. TTS first-chunk latency: t5 - t4
        if self._llm_sentence_ts and self._tts_first_chunk_ts:
            tts_first_chunk = int(
                max(0.0, (self._tts_first_chunk_ts - self._llm_sentence_ts) * 1000.0)
            )

        # Mark as reported for this turn
        self._reported_for_turn = True

        # Publish metrics event
        metrics_evt = LatencyMetricsUpdated(
            session_id=self.session_id,
            network_latency_ms=network_latency,
            asr_latency_ms=asr_latency,
            llm_first_token_ms=llm_first_token,
            llm_sentence_ms=llm_sentence,
            tts_first_chunk_ms=tts_first_chunk,
        )
        await self.event_bus.publish(metrics_evt)

    async def shutdown(self):
        pass
