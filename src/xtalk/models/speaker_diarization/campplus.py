"""HTTP CAM++ embedding client with session-local online clustering."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from ..registry import model
from .interfaces import DiarizationResult, DiarizationSegment, SpeakerDiarization


_SAMPLE_RATE = 16000
_EMBEDDING_DIMENSIONS = 192
_UNKNOWN_SPEAKER_ID = "S00"


class CampPlusRequestCancelled(RuntimeError):
    """Signal that an obsolete CAM++ snapshot request was cancelled."""


@dataclass
class _SpeakerProfile:
    """One committed session-local speaker centroid."""

    speaker_id: str
    embedding: np.ndarray
    sample_count: int = 1


@dataclass
class _PendingNewSpeaker:
    """Track consecutive partial embeddings for one uncommitted speaker."""

    embedding: np.ndarray
    confirmations: int = 1


@dataclass(frozen=True)
class _SpeakerDecision:
    """Describe one clustering decision rendered into a public result."""

    speaker_id: str | None
    action: str
    best_similarity: float | None = None
    confirmations: int = 0
    centroid_updated: bool = False


@model
class CampPlusDiarization(SpeakerDiarization):
    """Classify full VAD snapshots through a remote CAM++ embedding service.

    The remote service is stateless and returns one normalized-compatible
    speaker embedding for each complete PCM snapshot. This client owns the
    session-local speaker centroids. Partial snapshots may classify speakers
    but never mutate committed centroids; final snapshots register or update
    them.

    Parameters
    ----------
    base_url : str
        CAM++ embedding service root URL.
    request_timeout_s : float, optional
        Total HTTP timeout for one embedding request.
    similarity_threshold : float, optional
        Minimum cosine similarity required to reuse a committed speaker ID.
    min_audio_duration_s : float, optional
        Minimum snapshot duration accepted for embedding extraction.
    new_speaker_confirmations : int, optional
        Consecutive partial embeddings required before publishing a provisional
        new speaker ID. Final snapshots always make a terminal decision.
    centroid_update_alpha : float, optional
        Exponential moving-average weight assigned to a matched final
        embedding.
    max_speakers : int, optional
        Maximum number of committed speakers retained by one session clone.
    """

    def __init__(
        self,
        base_url: str,
        request_timeout_s: float = 5.0,
        similarity_threshold: float = 0.65,
        min_audio_duration_s: float = 0.5,
        new_speaker_confirmations: int = 2,
        centroid_update_alpha: float = 0.1,
        max_speakers: int = 16,
    ) -> None:
        normalized_url = str(base_url).rstrip("/")
        if not normalized_url:
            raise ValueError("base_url must be non-empty")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive")
        if not -1.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if min_audio_duration_s <= 0:
            raise ValueError("min_audio_duration_s must be positive")
        if new_speaker_confirmations < 1:
            raise ValueError("new_speaker_confirmations must be at least 1")
        if not 0.0 < centroid_update_alpha <= 1.0:
            raise ValueError("centroid_update_alpha must be in (0, 1]")
        if max_speakers < 1:
            raise ValueError("max_speakers must be at least 1")

        self.base_url = normalized_url
        self.request_timeout_s = float(request_timeout_s)
        self.similarity_threshold = float(similarity_threshold)
        self.min_audio_duration_s = float(min_audio_duration_s)
        self.new_speaker_confirmations = int(new_speaker_confirmations)
        self.centroid_update_alpha = float(centroid_update_alpha)
        self.max_speakers = int(max_speakers)

        self._session: aiohttp.ClientSession | None = None
        self._requests: dict[
            str, asyncio.Task[tuple[dict[str, Any], float, str]]
        ] = {}
        self._profiles: list[_SpeakerProfile] = []
        self._pending_new: dict[str, _PendingNewSpeaker] = {}

    def clone(self) -> CampPlusDiarization:
        """Create a client with identical configuration and fresh speaker state."""

        return CampPlusDiarization(
            base_url=self.base_url,
            request_timeout_s=self.request_timeout_s,
            similarity_threshold=self.similarity_threshold,
            min_audio_duration_s=self.min_audio_duration_s,
            new_speaker_confirmations=self.new_speaker_confirmations,
            centroid_update_alpha=self.centroid_update_alpha,
            max_speakers=self.max_speakers,
        )

    async def decode_snapshot(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        is_final: bool,
    ) -> DiarizationResult:
        """Extract one embedding and classify the current complete snapshot."""

        if not request_id:
            raise ValueError("request_id must be non-empty")
        if sample_rate != _SAMPLE_RATE:
            raise ValueError(
                f"CAM++ requires {_SAMPLE_RATE} Hz PCM, got {sample_rate} Hz"
            )
        if len(pcm16) % 2:
            raise ValueError("PCM16 payload must contain complete samples")

        duration_s = len(pcm16) / (sample_rate * 2)
        if duration_s < self.min_audio_duration_s:
            return self._render_result(
                duration_s=duration_s,
                latency_ms=0.0,
                decision=_SpeakerDecision(
                    speaker_id=_UNKNOWN_SPEAKER_ID,
                    action="too_short",
                ),
                metrics={"request_id": request_id, "is_final": is_final},
            )

        request_task = asyncio.create_task(
            self._post_embedding(
                request_id=request_id,
                pcm16=pcm16,
                sample_rate=sample_rate,
                is_final=is_final,
            )
        )
        previous = self._requests.setdefault(request_id, request_task)
        if previous is not request_task:
            request_task.cancel()
            raise ValueError(f"duplicate CAM++ request_id: {request_id}")
        try:
            payload, wall_latency_ms, remote_request_id = await request_task
        except asyncio.CancelledError as exc:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise
            raise CampPlusRequestCancelled(
                f"CAM++ request was cancelled: {request_id}"
            ) from exc
        finally:
            if self._requests.get(request_id) is request_task:
                self._requests.pop(request_id, None)

        server_metrics = _response_metrics(payload)
        latency_ms = _response_latency_ms(payload, wall_latency_ms)
        if not _speech_accepted(payload, server_metrics):
            return self._render_result(
                duration_s=duration_s,
                latency_ms=latency_ms,
                decision=_SpeakerDecision(
                    speaker_id=_UNKNOWN_SPEAKER_ID,
                    action="embedding_rejected",
                ),
                metrics={
                    **server_metrics,
                    "request_id": request_id,
                    "remote_request_id": remote_request_id,
                    "is_final": is_final,
                },
            )

        embedding = _normalized_embedding(payload.get("embedding"))
        request_group = _request_group(request_id)
        decision = self._classify(
            embedding,
            request_group=request_group,
            is_final=is_final,
        )
        return self._render_result(
            duration_s=duration_s,
            latency_ms=latency_ms,
            decision=decision,
            metrics={
                **server_metrics,
                "request_id": request_id,
                "remote_request_id": remote_request_id,
                "is_final": is_final,
                "model": payload.get("model"),
            },
        )

    async def cancel(self, request_id: str) -> None:
        """Cancel a locally in-flight embedding request when it is obsolete."""

        request_task = self._requests.get(request_id)
        if request_task is not None and not request_task.done():
            request_task.cancel()

    async def close(self) -> None:
        """Cancel requests, close HTTP resources, and discard session state."""

        for request_task in tuple(self._requests.values()):
            request_task.cancel()
        if self._requests:
            await asyncio.gather(*self._requests.values(), return_exceptions=True)
        self._requests.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._profiles.clear()
        self._pending_new.clear()

    def _classify(
        self,
        embedding: np.ndarray,
        *,
        request_group: str,
        is_final: bool,
    ) -> _SpeakerDecision:
        """Match, provisionally classify, or commit one normalized embedding."""

        best_profile, best_similarity = self._best_match(embedding)
        matched = (
            best_profile is not None
            and best_similarity is not None
            and best_similarity >= self.similarity_threshold
        )

        if is_final:
            self._pending_new.pop(request_group, None)
            if matched and best_profile is not None:
                self._update_profile(best_profile, embedding)
                return _SpeakerDecision(
                    speaker_id=best_profile.speaker_id,
                    action="matched_final",
                    best_similarity=best_similarity,
                    centroid_updated=True,
                )
            if len(self._profiles) >= self.max_speakers:
                return _SpeakerDecision(
                    speaker_id=_UNKNOWN_SPEAKER_ID,
                    action="speaker_limit_reached",
                    best_similarity=best_similarity,
                )
            profile = _SpeakerProfile(
                speaker_id=self._next_speaker_id(),
                embedding=embedding.copy(),
            )
            self._profiles.append(profile)
            return _SpeakerDecision(
                speaker_id=profile.speaker_id,
                action="registered_final",
                best_similarity=best_similarity,
                centroid_updated=True,
            )

        if matched and best_profile is not None:
            self._pending_new.pop(request_group, None)
            return _SpeakerDecision(
                speaker_id=best_profile.speaker_id,
                action="matched_partial",
                best_similarity=best_similarity,
            )

        if not self._profiles:
            return _SpeakerDecision(
                speaker_id="S01",
                action="provisional_first_speaker",
                confirmations=1,
            )
        if len(self._profiles) >= self.max_speakers:
            return _SpeakerDecision(
                speaker_id=_UNKNOWN_SPEAKER_ID,
                action="speaker_limit_reached",
                best_similarity=best_similarity,
            )

        pending = self._pending_new.get(request_group)
        if pending is None or _cosine_similarity(pending.embedding, embedding) < (
            self.similarity_threshold
        ):
            pending = _PendingNewSpeaker(embedding=embedding.copy())
            self._pending_new[request_group] = pending
        else:
            pending.confirmations += 1
            combined = pending.embedding + embedding
            pending.embedding = _normalize_vector(combined)

        if pending.confirmations < self.new_speaker_confirmations:
            return _SpeakerDecision(
                speaker_id=None,
                action="new_speaker_pending",
                best_similarity=best_similarity,
                confirmations=pending.confirmations,
            )
        return _SpeakerDecision(
            speaker_id=self._next_speaker_id(),
            action="provisional_new_speaker",
            best_similarity=best_similarity,
            confirmations=pending.confirmations,
        )

    def _best_match(
        self,
        embedding: np.ndarray,
    ) -> tuple[_SpeakerProfile | None, float | None]:
        """Return the highest-scoring committed profile without a margin rule."""

        best_profile: _SpeakerProfile | None = None
        best_similarity: float | None = None
        for profile in self._profiles:
            similarity = _cosine_similarity(profile.embedding, embedding)
            if best_similarity is None or similarity > best_similarity:
                best_profile = profile
                best_similarity = similarity
        return best_profile, best_similarity

    def _update_profile(
        self,
        profile: _SpeakerProfile,
        embedding: np.ndarray,
    ) -> None:
        """Update one committed centroid from a matched final embedding."""

        alpha = self.centroid_update_alpha
        profile.embedding = _normalize_vector(
            (1.0 - alpha) * profile.embedding + alpha * embedding
        )
        profile.sample_count += 1

    def _next_speaker_id(self) -> str:
        """Return the next compact committed speaker identifier."""

        return f"S{len(self._profiles) + 1:02d}"

    def _render_result(
        self,
        *,
        duration_s: float,
        latency_ms: float,
        decision: _SpeakerDecision,
        metrics: dict[str, Any],
    ) -> DiarizationResult:
        """Render one clustering decision into the shared diarization contract."""

        speaker_id = decision.speaker_id
        segments: list[DiarizationSegment] = []
        raw_text = ""
        if speaker_id is not None and duration_s > 0:
            segments.append(
                {
                    "start_s": 0.0,
                    "end_s": duration_s,
                    "speaker_id": speaker_id,
                    "text": "",
                }
            )
            raw_text = f"[0.00][{speaker_id}][{duration_s:.2f}]"
        return DiarizationResult(
            raw_text=raw_text,
            segments=segments,
            latency_ms=latency_ms,
            metrics={
                **metrics,
                "clustering_action": decision.action,
                "speaker_id": speaker_id,
                "best_similarity": decision.best_similarity,
                "similarity_threshold": self.similarity_threshold,
                "confirmations": decision.confirmations,
                "centroid_updated": decision.centroid_updated,
                "committed_speakers": len(self._profiles),
            },
        )

    async def _post_embedding(
        self,
        *,
        request_id: str,
        pcm16: bytes,
        sample_rate: int,
        is_final: bool,
    ) -> tuple[dict[str, Any], float, str]:
        """Submit one PCM snapshot to the stateless embedding endpoint."""

        session = await self._get_session()
        form = aiohttp.FormData()
        form.add_field("request_id", request_id)
        form.add_field("sample_rate", str(sample_rate))
        form.add_field("is_final", "true" if is_final else "false")
        form.add_field(
            "audio",
            pcm16,
            filename="snapshot.pcm",
            content_type="audio/pcm",
        )
        started = time.perf_counter()
        async with session.post(
            f"{self.base_url}/v1/speaker/embeddings",
            data=form,
        ) as response:
            response.raise_for_status()
            payload = await response.json()
            remote_request_id = response.headers.get("X-Request-Id", "")
        if not isinstance(payload, dict):
            raise TypeError("CAM++ embedding response must be a JSON object")
        return payload, (time.perf_counter() - started) * 1000.0, remote_request_id

    async def _get_session(self) -> aiohttp.ClientSession:
        """Return a live HTTP session for the current event loop."""

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session


def _normalized_embedding(value: object) -> np.ndarray:
    """Validate and normalize one CAM++ embedding returned by the service."""

    embedding = np.asarray(value, dtype=np.float32)
    if embedding.shape != (_EMBEDDING_DIMENSIONS,):
        raise ValueError(
            "CAM++ embedding must have shape "
            f"({_EMBEDDING_DIMENSIONS},), got {embedding.shape}"
        )
    if not np.all(np.isfinite(embedding)):
        raise ValueError("CAM++ embedding contains non-finite values")
    return _normalize_vector(embedding)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    """Return a finite float32 vector with unit L2 norm."""

    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-8:
        raise ValueError("CAM++ embedding must have a non-zero finite norm")
    return vector / norm


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity for two normalized-compatible vectors."""

    return float(np.dot(left, right))


def _request_group(request_id: str) -> str:
    """Remove manager revision and kind suffixes from a snapshot request ID."""

    parts = request_id.rsplit("/", 2)
    if len(parts) == 3 and parts[1].isdigit() and parts[2] in {"partial", "final"}:
        return parts[0]
    return request_id


def _response_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Return service metrics when they use the expected object shape."""

    value = payload.get("metrics")
    return dict(value) if isinstance(value, dict) else {}


def _speech_accepted(
    payload: dict[str, Any],
    metrics: dict[str, Any],
) -> bool:
    """Return whether the embedding service accepted the supplied speech."""

    value = payload.get("speech_accepted", metrics.get("speech_accepted", True))
    return value is not False


def _response_latency_ms(payload: dict[str, Any], fallback: float) -> float:
    """Return a positive service latency or the measured client fallback."""

    value = payload.get("latency_ms")
    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
        return float(value)
    return fallback
