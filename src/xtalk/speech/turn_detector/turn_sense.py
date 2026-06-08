import asyncio
from dataclasses import dataclass
from typing import Literal, Optional

import aiohttp

from ..interfaces import (
    TurnDetectionAction,
    TurnDetectionResult,
    TurnDetectionSemantic,
    TurnDetector,
)

_IDLE_RESULT = TurnDetectionResult(
    action=TurnDetectionAction.DO_NOTHING,
    semantic=TurnDetectionSemantic.IDLE,
)


@dataclass(frozen=True)
class TurnSenseProbabilities:
    """Class probabilities returned by the TurnSense HTTP service."""

    complete: float
    incomplete: float
    invalid: float


@dataclass(frozen=True)
class TurnSenseInferenceResult:
    """Structured inference result returned by the TurnSense HTTP service."""

    prediction: Literal["complete", "incomplete", "invalid"]
    probabilities: TurnSenseProbabilities


class TurnSense(TurnDetector):
    """Turn detector client for the external TurnSense HTTP service."""

    AUDIO_CHANNELS = 1
    AUDIO_FORMAT = "pcm_s16le"
    PCM_BYTES_PER_SECOND = 16000 * 2
    SAMPLE_RATE = 16000

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 10.0,
        inference_interval_ms: int = 200,
    ) -> None:
        """Initialize the TurnSense HTTP client.

        Parameters
        ----------
        base_url : str, optional
            Base URL of the TurnSense HTTP server.
        timeout : float, optional
            Request timeout in seconds for inference calls.
        inference_interval_ms : int, optional
            Buffered-audio interval in milliseconds between inference requests.
        """
        super().__init__()
        if inference_interval_ms <= 0:
            raise ValueError("inference_interval_ms must be positive")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._inference_interval_ms = inference_interval_ms
        self._infer_bytes_url = f"{self._base_url}/infer/bytes"
        self._inference_interval_bytes = (
            self.PCM_BYTES_PER_SECOND * self._inference_interval_ms // 1000
        )
        self._state_lock = asyncio.Lock()
        self._audio_buffer = bytearray()
        self._speech_started = False
        self._epoch = 0
        self._latest_inference_seq = 0
        self._latest_result_seq = 0
        self._latest_result: Optional[TurnSenseInferenceResult] = None
        self._latest_inference_task: Optional[asyncio.Task] = None
        self._last_requested_num_bytes = 0
        self._pending_complete_after_interrupt = False

    def _reset_detection_state(self) -> None:
        """Reset buffered audio and invalidate stale in-flight inference results."""
        self._epoch += 1
        self._audio_buffer.clear()
        self._speech_started = False
        self._latest_inference_seq = 0
        self._latest_result_seq = 0
        self._latest_result = None
        self._latest_inference_task = None
        self._last_requested_num_bytes = 0

    def _begin_speech(self) -> None:
        """Start a new buffered speech segment if one is not already active."""
        if self._speech_started:
            return
        self._reset_detection_state()
        self._speech_started = True

    async def _run_inference(
        self, epoch: int, sequence: int, audio: bytes
    ) -> tuple[int, int, TurnSenseInferenceResult]:
        """Run one HTTP inference request for a buffered audio snapshot."""
        result = await self._infer_audio_bytes(audio)
        return epoch, sequence, result

    def _store_inference_result(self, task: asyncio.Task) -> None:
        """Keep only the newest completed inference result for the active epoch."""
        try:
            epoch, sequence, result = task.result()
        except Exception:
            return
        if epoch != self._epoch or sequence < self._latest_result_seq:
            return
        self._latest_result_seq = sequence
        self._latest_result = result

    def _launch_inference_locked(self) -> Optional[asyncio.Task]:
        """Launch inference for the latest buffered audio snapshot."""
        if not self._audio_buffer:
            return None
        snapshot = bytes(self._audio_buffer)
        self._latest_inference_seq += 1
        task = asyncio.create_task(
            self._run_inference(
                self._epoch,
                self._latest_inference_seq,
                snapshot,
            )
        )
        task.add_done_callback(self._store_inference_result)
        self._latest_inference_task = task
        self._last_requested_num_bytes = len(snapshot)
        return task

    def _maybe_launch_periodic_inference_locked(self) -> None:
        """Launch another inference every configured interval of new audio."""
        if not self._speech_started:
            return
        if len(self._audio_buffer) - self._last_requested_num_bytes < (
            self._inference_interval_bytes
        ):
            return
        self._launch_inference_locked()

    async def _await_latest_inference(self) -> Optional[TurnSenseInferenceResult]:
        """Wait until the newest scheduled inference result is available."""
        while True:
            async with self._state_lock:
                task = self._latest_inference_task
                if task is None:
                    task = self._launch_inference_locked()
                    if task is None:
                        return self._latest_result
                current_task = task
            try:
                await current_task
            except Exception:
                async with self._state_lock:
                    if self._latest_inference_task is current_task:
                        self._latest_inference_task = None
                raise
            async with self._state_lock:
                if self._latest_inference_task is current_task:
                    self._latest_inference_task = None
                    return self._latest_result

    def _result_for_pause(
        self, inference_result: Optional[TurnSenseInferenceResult]
    ) -> TurnDetectionResult:
        """Map the newest pause-time inference result to a listening decision."""
        if inference_result is None:
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.INCOMPLETE,
            )
        if inference_result.prediction == TurnDetectionSemantic.COMPLETE.value:
            self._reset_detection_state()
            return TurnDetectionResult(
                action=TurnDetectionAction.START_GENERATION,
                semantic=TurnDetectionSemantic.COMPLETE,
            )
        return TurnDetectionResult(
            action=TurnDetectionAction.DO_NOTHING,
            semantic=TurnDetectionSemantic.INCOMPLETE,
        )

    def _result_for_interrupt(
        self, inference_result: Optional[TurnSenseInferenceResult]
    ) -> TurnDetectionResult:
        """Map the newest streaming inference result to an interruption decision."""
        if inference_result is None:
            return _IDLE_RESULT
        if inference_result.prediction == "invalid":
            return _IDLE_RESULT
        if inference_result.prediction == TurnDetectionSemantic.COMPLETE.value:
            self._pending_complete_after_interrupt = True
            self._reset_detection_state()
            return TurnDetectionResult(
                action=TurnDetectionAction.STOP_SPEAKING,
                semantic=TurnDetectionSemantic.COMPLETE,
            )
        return TurnDetectionResult(
            action=TurnDetectionAction.STOP_SPEAKING,
            semantic=TurnDetectionSemantic.INCOMPLETE,
        )

    async def _infer_audio_bytes(
        self, audio: bytes, source: Optional[str] = None
    ) -> TurnSenseInferenceResult:
        """Send raw PCM audio bytes to the TurnSense HTTP inference endpoint.

        Parameters
        ----------
        audio : bytes
            PCM 16-bit mono 16 kHz audio bytes to post to the TurnSense service.
        source : str | None, optional
            Optional source name sent via the ``X-Audio-Source`` header.

        Returns
        -------
        TurnSenseInferenceResult
            Parsed response returned by the TurnSense service.
        """
        headers = {
            "Content-Type": "application/octet-stream",
            "X-Audio-Source": source or "turn_sense_audio.pcm",
            "X-Audio-Format": self.AUDIO_FORMAT,
            "X-Audio-Sample-Rate": str(self.SAMPLE_RATE),
            "X-Audio-Channels": str(self.AUDIO_CHANNELS),
        }

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self._infer_bytes_url,
                data=audio,
                headers=headers,
            ) as response:
                response.raise_for_status()
                payload = await response.json()

        probabilities = payload["probabilities"]
        return TurnSenseInferenceResult(
            prediction=payload["prediction"],
            probabilities=TurnSenseProbabilities(
                complete=probabilities["complete"],
                incomplete=probabilities["incomplete"],
                invalid=probabilities["invalid"],
            ),
        )

    def detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Synchronously run turn detection via the async implementation.

        Parameters
        ----------
        audio : bytes | None, optional
            Current PCM 16-bit mono audio frame at 16 kHz.
        text : str | None, optional
            ASR text for the current turn.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech.
        speech_pause : bool | None, optional
            Whether the user appears to have paused speaking.

        Returns
        -------
        TurnDetectionResult
            Synchronous wrapper around ``async_detect``.
        """
        return asyncio.run(
            self.async_detect(
                audio=audio,
                text=text,
                speech_start=speech_start,
                speech_pause=speech_pause,
            )
        )

    async def async_detect(
        self,
        audio: Optional[bytes] = None,
        text: Optional[str] = None,
        speech_start: bool = False,
        speech_pause: Optional[bool] = None,
    ) -> TurnDetectionResult:
        """Asynchronously detect turn state using the TurnSense HTTP service.

        Parameters
        ----------
        audio : bytes | None, optional
            Current PCM 16-bit mono audio frame at 16 kHz.
        text : str | None, optional
            ASR text for the current turn. The current implementation only uses
            this path as a carrier for ``speech_pause`` events.
        speech_start : bool, optional
            Whether VAD has just detected the start of speech.
        speech_pause : bool | None, optional
            Whether the user appears to have paused speaking.

        Returns
        -------
        TurnDetectionResult
            Turn-detection decision derived from the latest available TurnSense
            prediction.

        Notes
        -----
        The detector keeps one shared audio buffer for both listening states.
        When ``listening`` is ``True``, speech starts buffering on
        ``speech_start=True`` and a new inference is launched every configured
        interval of newly buffered audio. On ``speech_pause=True``, the
        detector waits for the newest inference result and starts generation
        only when that newest result is ``complete``. Otherwise it keeps the
        buffered audio and returns ``DO_NOTHING`` with ``INCOMPLETE``.

        When ``listening`` is ``False``, the detector still buffers audio from
        ``speech_start=True`` onward and launches the same periodic inferences.
        During this state, the newest non-``invalid`` inference result triggers
        ``STOP_SPEAKING``. If that newest result is ``complete``, the detector
        stores a one-shot pending completion so that the next call observed in
        the listening state returns ``START_GENERATION`` with ``COMPLETE``.
        """
        del text
        async with self.listening_lock():
            listening = self.listening

        async with self._state_lock:
            if listening and self._pending_complete_after_interrupt:
                self._pending_complete_after_interrupt = False
                self._reset_detection_state()
                return TurnDetectionResult(
                    action=TurnDetectionAction.START_GENERATION,
                    semantic=TurnDetectionSemantic.COMPLETE,
                )

            if speech_start:
                self._begin_speech()

            if audio is not None and self._speech_started:
                self._audio_buffer.extend(audio)
                self._maybe_launch_periodic_inference_locked()

            speech_started = self._speech_started
            latest_result = self._latest_result

        if listening and speech_pause:
            if not speech_started:
                return _IDLE_RESULT
            return self._result_for_pause(await self._await_latest_inference())

        if not listening and speech_started:
            return self._result_for_interrupt(latest_result)

        if speech_started:
            return TurnDetectionResult(
                action=TurnDetectionAction.DO_NOTHING,
                semantic=TurnDetectionSemantic.INCOMPLETE,
            )
        return _IDLE_RESULT

    def clone(self) -> TurnDetector:
        """Clone the detector for a new session.

        Returns
        -------
        TurnDetector
            A new ``TurnSense`` instance.
        """
        return TurnSense(
            base_url=self._base_url,
            timeout=self._timeout,
            inference_interval_ms=self._inference_interval_ms,
        )
