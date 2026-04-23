# -*- coding: utf-8 -*-
"""
Speaker Identification Manager

Maintains per-session speaker embeddings to recognize or register speakers based on
voiceprints provided by a SpeakerEncoder.
"""
import asyncio
import numpy as np
import json
import os
from datetime import datetime
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
from collections import deque

from ...log_utils import logger
from ..event_bus import EventBus
from ..events import (
    EnhancedAudioFrameReceived,
    TurnASRStartRequested,
    TurnASREndRequested,
    SpeakerRecognized,
)
from ..interfaces import Manager
from ...pipelines import Pipeline


@dataclass
class SpeakerProfile:
    """Speaker profile entry stored for each session."""

    speaker_id: str  # e.g., "Speaker 1", "Speaker 2"
    embedding: np.ndarray  # Voiceprint embedding
    sample_count: int = 1  # Number of samples used for averaging


class SpeakerManager(Manager):
    """Session-scoped speaker identification manager.

    Responsibilities:
    - Collect enhanced audio frames per turn and extract embeddings.
    - Compare against previously registered speakers.
    - Recognize an existing speaker or register a new one.
    - Write `speaker_id` to PipelineContext and emit SpeakerRecognized events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        pipeline: Pipeline,
        config: dict[str, Any] | None = None,
    ):
        """Initialize the speaker manager.

        Args:
            event_bus: shared event bus
            session_id: unique session identifier
            pipeline: pipeline providing a speaker encoder
            config: optional parameters
                - similarity_threshold: cosine threshold (default 0.4)
                - min_audio_length_sec: minimum audio length (default 0.5s)
                - embedding_update_alpha: EMA rate for embeddings (default 0.05)
        """
        self.event_bus = event_bus
        self.session_id = session_id
        self.pipeline = pipeline
        self.config = config or {}

        # Obtain speaker encoder
        self.speaker_encoder = self.pipeline.get_speaker_encoder()

        self.similarity_threshold = self.config.get("similarity_threshold", 0.4)
        self.min_audio_length_sec = float(self.config.get("min_audio_length_sec", 0.5))
        self.embedding_update_alpha = float(
            self.config.get("embedding_update_alpha", 0.05)
        )

        # Debug logging
        self.debug_log_dir = self.config.get("debug_log_dir", "logs/speaker_debug")

        # Detailed identification logs
        self.identification_history: List[Dict[str, Any]] = []

        # Speaker profiles (per session)
        self.speaker_profiles: List[SpeakerProfile] = []
        self.current_speaker_id: Optional[str] = None

        # Audio buffer for a full turn
        self.audio_buffer = deque(maxlen=500)
        self.buffer_lock = asyncio.Lock()
        self.speech_active = False

    @Manager.event_handler(
        EnhancedAudioFrameReceived,
        priority=95,
    )
    async def _handle_audio_frame(self, event: EnhancedAudioFrameReceived) -> None:
        """Collect enhanced audio frames while speech is active."""
        if not self.speaker_encoder:
            return

        if not self.speech_active:
            return

        # Cache audio frames
        async with self.buffer_lock:
            self.audio_buffer.append(
                {
                    "data": event.audio_data,
                    "sample_rate": getattr(event, "sample_rate", 16000),
                }
            )

    @Manager.event_handler(TurnASRStartRequested, priority=95)
    async def _handle_turn_start(self, event: TurnASRStartRequested) -> None:
        """Prepare buffer for a new speech turn."""
        self.speech_active = True
        # Clear buffer for the next utterance
        async with self.buffer_lock:
            self.audio_buffer.clear()

        # Reset previous speaker id
        self.current_speaker_id = None

    @Manager.event_handler(TurnASREndRequested, priority=93)
    async def _handle_turn_end(self, event: TurnASREndRequested) -> None:
        """
        Handle end-of-turn events by running speaker identification on the buffered
        audio chunk. Writes `speaker_id` into PipelineContext and emits status events.
        """
        self.speech_active = False

        if not self.speaker_encoder:
            return

        try:
            # Retrieve buffered audio
            async with self.buffer_lock:
                if not self.audio_buffer:
                    # No audio: clear speaker_id and notify frontend
                    ctx = self.pipeline.context
                    ctx["speaker_id"] = None
                    self.pipeline.context = ctx
                    await self.event_bus.publish(
                        SpeakerRecognized(
                            session_id=self.session_id,
                            speaker_id=None,
                            reason="no_audio",
                        )
                    )
                    return

                # Merge all frames
                audio_chunks = [chunk["data"] for chunk in self.audio_buffer]
                sample_rate = self.audio_buffer[0]["sample_rate"]

            # Concatenate into a single buffer
            audio_data = b"".join(audio_chunks)

            # Ensure we have enough audio (16-bit PCM => 2 bytes/sample)
            audio_length_sec = len(audio_data) / (sample_rate * 2)
            if audio_length_sec < self.min_audio_length_sec:
                # Too short: clear speaker_id and notify frontend
                ctx = self.pipeline.context
                ctx["speaker_id"] = None
                self.pipeline.context = ctx
                await self.event_bus.publish(
                    SpeakerRecognized(
                        session_id=self.session_id,
                        speaker_id=None,
                        reason="too_short",
                    )
                )
                return

            # Extract embedding
            embedding_vec = await self.speaker_encoder.async_extract(audio_data)
            current_embedding = embedding_vec

            # Identify or register speaker
            speaker_id = await self._identify_or_register_speaker(current_embedding)

            # Update context and notify frontend
            self.current_speaker_id = speaker_id
            ctx = self.pipeline.context
            ctx["speaker_id"] = speaker_id
            self.pipeline.context = ctx
            await self.event_bus.publish(
                SpeakerRecognized(
                    session_id=self.session_id,
                    speaker_id=speaker_id,
                    reason="recognized",
                )
            )

        except Exception as e:
            logger.error(
                "Speaker identification failed - session: %s, error: %s",
                self.session_id,
                e,
            )
            # On error, clear speaker_id and notify frontend
            ctx = self.pipeline.context
            ctx["speaker_id"] = None
            self.pipeline.context = ctx
            try:
                await self.event_bus.publish(
                    SpeakerRecognized(
                        session_id=self.session_id,
                        speaker_id=None,
                        reason="error",
                    )
                )
            except Exception:
                pass

    async def _identify_or_register_speaker(self, embedding: np.ndarray) -> str:
        """
        Identify the speaker if similarity exceeds the threshold; otherwise register
        a new speaker profile.
        """
        timestamp = datetime.now().isoformat()

        # Register first speaker if none exist
        if not self.speaker_profiles:
            speaker_id = "Speaker 1"
            # Normalize initial embedding
            norm = np.linalg.norm(embedding)
            normalized_embedding = embedding / norm if norm > 1e-8 else embedding
            self.speaker_profiles.append(
                SpeakerProfile(
                    speaker_id=speaker_id,
                    embedding=normalized_embedding,
                    sample_count=1,
                )
            )

            # Persist debug info
            self._save_debug_info(
                timestamp=timestamp,
                speaker_id=speaker_id,
                embedding=embedding,
                similarities=[],
                action="register_first",
                best_similarity=None,
            )

            return speaker_id

        # Compare against existing profiles
        best_match_id = None
        best_similarity = -1.0

        similarities = []

        for profile in self.speaker_profiles:
            similarity = self.speaker_encoder.similarity(embedding, profile.embedding)
            similarities.append((profile.speaker_id, similarity))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = profile.speaker_id

        # Check match
        if best_similarity >= self.similarity_threshold and best_match_id is not None:
            matched_profile = next(
                (
                    profile
                    for profile in self.speaker_profiles
                    if profile.speaker_id == best_match_id
                ),
                None,
            )
            if matched_profile is not None:
                # Update embedding via EMA
                old_norm = np.linalg.norm(matched_profile.embedding)
                new_norm = np.linalg.norm(embedding)
                if old_norm > 1e-8 and new_norm > 1e-8:
                    old_embedding_normalized = matched_profile.embedding / old_norm
                    new_embedding_normalized = embedding / new_norm
                    alpha = self.embedding_update_alpha
                    updated_embedding = (
                        1 - alpha
                    ) * old_embedding_normalized + alpha * new_embedding_normalized
                    updated_norm = np.linalg.norm(updated_embedding)
                    if updated_norm > 1e-8:
                        matched_profile.embedding = updated_embedding / updated_norm
                matched_profile.sample_count += 1

            # Persist debug info
            self._save_debug_info(
                timestamp=timestamp,
                speaker_id=best_match_id,
                embedding=embedding,
                similarities=similarities,
                action="matched",
                best_similarity=best_similarity,
            )

            return best_match_id

        # Register a new speaker
        new_speaker_id = f"Speaker {len(self.speaker_profiles) + 1}"
        # Normalize before storing
        norm = np.linalg.norm(embedding)
        normalized_embedding = embedding / norm if norm > 1e-8 else embedding
        self.speaker_profiles.append(
            SpeakerProfile(
                speaker_id=new_speaker_id,
                embedding=normalized_embedding,
                sample_count=1,
            )
        )

        # Persist debug info
        self._save_debug_info(
            timestamp=timestamp,
            speaker_id=new_speaker_id,
            embedding=embedding,
            similarities=similarities,
            action="register_new",
            best_similarity=best_similarity,
        )

        return new_speaker_id

    def _save_debug_info(
        self,
        timestamp: str,
        speaker_id: str,
        embedding: np.ndarray,
        similarities: List[tuple],
        action: str,
        best_similarity: Optional[float],
    ) -> None:
        return  # disable debug info saving
        """Persist debug info (JSON + embedding) for later inspection."""
        os.makedirs(self.debug_log_dir, exist_ok=True)
        try:
            # Build debug record
            debug_record = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "speaker_id": speaker_id,
                "action": action,
                "embedding_shape": embedding.shape,
                "embedding_norm": float(np.linalg.norm(embedding)),
                "best_similarity": (
                    float(best_similarity) if best_similarity is not None else None
                ),
                "threshold": self.similarity_threshold,
                "similarities": [
                    {"speaker_id": sid, "score": float(score)}
                    for sid, score in similarities
                ],
                "total_speakers": len(self.speaker_profiles),
            }

            # Append to in-memory history
            self.identification_history.append(debug_record)

            # Save embedding to .npy
            vector_filename = (
                f"{timestamp.replace(':', '-')}_{speaker_id.replace(' ', '_')}.npy"
            )
            vector_path = os.path.join(self.debug_log_dir, vector_filename)
            np.save(vector_path, embedding)

            # Save JSON record
            record_filename = (
                f"{timestamp.replace(':', '-')}_{speaker_id.replace(' ', '_')}.json"
            )
            record_path = os.path.join(self.debug_log_dir, record_filename)
            with open(record_path, "w", encoding="utf-8") as f:
                json.dump(debug_record, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Failed to save speaker debug info: %s", e)

    def _save_session_summary(self) -> None:
        return  # disable debug info saving
        """Write session-level summary JSON for debugging."""
        os.makedirs(self.debug_log_dir, exist_ok=True)
        if not self.identification_history:
            return

        try:
            summary = {
                "session_id": self.session_id,
                "total_identifications": len(self.identification_history),
                "total_speakers": len(self.speaker_profiles),
                "threshold": self.similarity_threshold,
                "min_audio_length_sec": self.min_audio_length_sec,
                "speaker_profiles": [
                    {
                        "speaker_id": profile.speaker_id,
                        "sample_count": profile.sample_count,
                        "embedding_shape": profile.embedding.shape,
                        "embedding_norm": float(np.linalg.norm(profile.embedding)),
                    }
                    for profile in self.speaker_profiles
                ],
                "identification_history": self.identification_history,
            }

            summary_filename = f"session_{self.session_id}_summary.json"
            summary_path = os.path.join(self.debug_log_dir, summary_filename)

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Failed to save session summary: %s", e)

    async def shutdown(self) -> None:
        """Stop buffering audio and persist debug summaries."""
        self.speech_active = False
        async with self.buffer_lock:
            self.audio_buffer.clear()

        # Always persist a session summary
        self._save_session_summary()
