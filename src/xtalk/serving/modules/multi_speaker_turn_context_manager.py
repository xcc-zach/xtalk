"""Join ASR hard-turn finals with terminal MTD speaker timelines."""

from __future__ import annotations

import asyncio
from typing import Any

from ..event_bus import EventBus
from ..events import (
    ASRResultFinal,
    MultiSpeakerTurnReady,
    SpeakerDiarizationTurnFinal,
)
from ..interfaces import Manager


class MultiSpeakerTurnContextManager(Manager):
    """Publish one combined LLM input after ASR and MTD independently finish."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config = config or {}
        multi_config = dict(self.config.get("multi_speaker") or {})
        self.enabled = bool(multi_config.get("enabled", False))
        self.response_policy = str(multi_config.get("response_policy", "all"))
        self.focus_speaker_ids = {
            str(item) for item in multi_config.get("focus_speaker_ids", [])
        }
        self.suppress_when_speaker_missing = bool(
            multi_config.get("suppress_when_speaker_missing", False)
        )
        self.join_timeout_s = float(multi_config.get("join_timeout_s", 5.0))
        self.fallback_on_timeout = bool(multi_config.get("fallback_on_timeout", True))
        self._asr_finals: dict[int, ASRResultFinal] = {}
        self._mtd_finals: dict[int, SpeakerDiarizationTurnFinal] = {}
        self._published: set[int] = set()
        self._timeout_tasks: dict[int, asyncio.Task[None]] = {}

    @Manager.event_handler(
        ASRResultFinal,
        priority=30,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_asr_final(self, event: ASRResultFinal) -> None:
        """Cache ASR text and start a bounded MTD join wait."""

        if not self.enabled:
            return
        self._asr_finals[event.turn_id] = event
        if event.turn_id not in self._timeout_tasks:
            self._timeout_tasks[event.turn_id] = asyncio.create_task(
                self._wait_for_timeout(event.turn_id)
            )
        await self._try_publish(event.turn_id)

    @Manager.event_handler(
        SpeakerDiarizationTurnFinal,
        priority=30,
        enabled_if=lambda manager: manager.enabled,
    )
    async def _on_mtd_final(self, event: SpeakerDiarizationTurnFinal) -> None:
        """Cache the complete speaker timeline and attempt the join."""

        if not self.enabled:
            return
        self._mtd_finals[event.turn_id] = event
        await self._try_publish(event.turn_id)

    async def _try_publish(self, turn_id: int) -> None:
        """Publish a combined event when both independent results exist."""

        if turn_id in self._published:
            return
        asr_event = self._asr_finals.get(turn_id)
        mtd_event = self._mtd_finals.get(turn_id)
        if asr_event is None or mtd_event is None:
            return
        await self._publish_ready(asr_event, mtd_event)

    async def _wait_for_timeout(self, turn_id: int) -> None:
        """Apply the configured ASR-only fallback if MTD misses the join bound."""

        try:
            await asyncio.sleep(self.join_timeout_s)
            if turn_id in self._published:
                return
            asr_event = self._asr_finals.get(turn_id)
            if asr_event is None or not self.fallback_on_timeout:
                return
            await self._publish_ready(asr_event, None, timeout=True)
        except asyncio.CancelledError:
            raise

    async def _publish_ready(
        self,
        asr_event: ASRResultFinal,
        mtd_event: SpeakerDiarizationTurnFinal | None,
        *,
        timeout: bool = False,
    ) -> None:
        """Resolve response policy and emit one idempotent joined event."""

        turn_id = asr_event.turn_id
        if turn_id in self._published:
            return
        active_speaker_id = (
            mtd_event.active_speaker_id if mtd_event is not None else None
        )
        should_respond = self._should_respond(active_speaker_id)
        degraded_reasons: list[str] = []
        if timeout:
            degraded_reasons.append("join_timeout")
        if mtd_event is not None and mtd_event.degraded:
            degraded_reasons.append(mtd_event.degraded_reason)
        self._published.add(turn_id)
        timeout_task = self._timeout_tasks.pop(turn_id, None)
        if timeout_task is not None and timeout_task is not asyncio.current_task():
            timeout_task.cancel()
        await self.event_bus.publish(
            MultiSpeakerTurnReady(
                session_id=self.session_id,
                turn_id=turn_id,
                asr_text=asr_event.text,
                diarization_text=(mtd_event.diarization_text if mtd_event else ""),
                diarization_segments=(mtd_event.segments if mtd_event else []),
                active_speaker_id=active_speaker_id,
                should_respond=should_respond,
                degraded=bool(degraded_reasons),
                degraded_reason=",".join(
                    reason for reason in dict.fromkeys(degraded_reasons) if reason
                ),
            )
        )
        self._asr_finals.pop(turn_id, None)
        self._mtd_finals.pop(turn_id, None)

    def _should_respond(self, active_speaker_id: str | None) -> bool:
        """Return whether response generation is allowed for the active speaker."""

        if self.response_policy != "focus_only":
            return True
        if active_speaker_id is None:
            return not self.suppress_when_speaker_missing
        return active_speaker_id in self.focus_speaker_ids

    async def shutdown(self) -> None:
        """Cancel outstanding join timers."""

        tasks = list(self._timeout_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._timeout_tasks.clear()
