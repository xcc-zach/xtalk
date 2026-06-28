import edge_tts
import asyncio
import numpy as np
import soxr
import soundfile as sf
from io import BytesIO
from .interfaces import TTS
from ..registry import model
from typing import AsyncIterator, Iterable, List


def _bytes_per_ms(sample_rate: int) -> int:
    # s16le mono -> 2 bytes per sample
    return int(sample_rate * 2 / 1000)


def _trim_tail_silence_pcm16(
    pcm_bytes: bytes,
    sample_rate: int,
    threshold_db: float = -40.0,
    min_silence_ms: int = 60,
    frame_ms: int = 10,
) -> bytes:
    if not pcm_bytes:
        return pcm_bytes
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    total = len(audio)
    num_frames = (total + frame_size - 1) // frame_size
    padded = np.zeros(num_frames * frame_size, dtype=np.int16)
    padded[:total] = audio
    frames = padded.reshape(num_frames, frame_size).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1)) + 1e-8
    db = 20 * np.log10(rms / 32768.0 + 1e-12)
    is_sil = db < threshold_db
    min_sil_frames = max(1, int(min_silence_ms / frame_ms))
    end_frame = num_frames
    run = 0
    for i in range(num_frames - 1, -1, -1):
        if is_sil[i]:
            run += 1
            if run >= min_sil_frames:
                end_frame = i
        else:
            break
    end_sample = min(total, end_frame * frame_size)
    return audio[:end_sample].astype(np.int16).tobytes()


async def _stream_with_tail_trim_async(
    text: str,
    voice: str,
    sample_rate: int,
    window_ms: int = 300,
    threshold_db: float = -40.0,
    min_silence_ms: int = 60,
    frame_ms: int = 10,
) -> AsyncIterator[bytes]:
    """Yield PCM s16le mono chunks with tail-window silence trimming. If PCM streaming
    is not supported by the installed edge-tts, fall back to single-shot synthesis
    and yield once after trimming the whole sentence.
    """
    # Try PCM streaming first
    try:
        output_format = f"raw-{sample_rate // 1000}khz-16bit-mono-pcm"
        communicate = edge_tts.Communicate(text, voice, output_format=output_format)
        want_pcm = True
    except TypeError:
        want_pcm = False

    if not want_pcm:
        # Fallback: use existing path, then trim and yield once
        # This uses the synchronous synthesize implemented in the class below
        # to keep behavior consistent.
        from typing import cast  # local import to avoid top pollution

        engine = cast(EdgeTTS, EdgeTTS(voice=voice, sample_rate=sample_rate))
        audio_bytes = await engine._synthesize_async(text)
        trimmed = _trim_tail_silence_pcm16(
            audio_bytes,
            sample_rate,
            threshold_db=threshold_db,
            min_silence_ms=min_silence_ms,
            frame_ms=frame_ms,
        )
        if trimmed:
            yield trimmed
        return

    tail_buf = bytearray()
    win_bytes = _bytes_per_ms(sample_rate) * max(0, int(window_ms))

    async for chunk in communicate.stream():
        if chunk["type"] != "audio":
            continue
        data = chunk["data"]  # already PCM s16le mono
        if not data:
            continue
        tail_buf.extend(data)
        trimmed_tail = _trim_tail_silence_pcm16(
            bytes(tail_buf),
            sample_rate,
            threshold_db=threshold_db,
            min_silence_ms=min_silence_ms,
            frame_ms=frame_ms,
        )
        if len(trimmed_tail) > win_bytes:
            prefix = trimmed_tail[:-win_bytes]
            if prefix:
                yield prefix
            tail_buf = bytearray(trimmed_tail[-win_bytes:])
        else:
            tail_buf = bytearray(trimmed_tail)

    # stream finished: flush the remaining tail with a final trim
    if tail_buf:
        final_tail = _trim_tail_silence_pcm16(
            bytes(tail_buf),
            sample_rate,
            threshold_db=threshold_db,
            min_silence_ms=min_silence_ms,
            frame_ms=frame_ms,
        )
        if final_tail:
            yield final_tail


@model
class EdgeTTS(TTS):
    """
    Edge-TTS implementation aligned with the TTS interface.
    Supports Chinese and multiple voice options.
    """

    def __init__(
        self,
        voice: str = "zh-CN-XiaoyiNeural",
        sample_rate: int = 48000,
        trim_tail: bool = False,
        threshold_db: float = -40.0,
        min_silence_ms: int = 100,
        frame_ms: int = 20,
        pad_ms: int = 100,
    ):
        self._sample_rate = sample_rate
        self.voice = voice
        # Tail-silence trimming options
        self._trim_tail = trim_tail
        self._trim_threshold_db = threshold_db
        self._trim_min_silence_ms = min_silence_ms
        self._trim_frame_ms = frame_ms
        self._trim_pad_ms = pad_ms

    def clone(self):
        return EdgeTTS(
            voice=self.voice,
            sample_rate=self._sample_rate,
            trim_tail=self._trim_tail,
            threshold_db=self._trim_threshold_db,
            min_silence_ms=self._trim_min_silence_ms,
            frame_ms=self._trim_frame_ms,
            pad_ms=self._trim_pad_ms,
        )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int):
        self._sample_rate = value

    @staticmethod
    def _float32_to_pcm_bytes(audio_float: np.ndarray) -> bytes:
        audio_int16 = np.clip(audio_float * 32768.0, -32768, 32767).astype(np.int16)
        return audio_int16.tobytes()

    @staticmethod
    def _trim_tail_silence_pcm16(
        pcm_bytes: bytes,
        sample_rate: int,
        threshold_db: float = -40.0,
        min_silence_ms: int = 60,
        frame_ms: int = 10,
        pad_ms: int = 10,
    ) -> bytes:
        if not pcm_bytes:
            return pcm_bytes
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio.size == 0:
            return pcm_bytes
        frame_size = max(1, int(sample_rate * frame_ms / 1000))
        total = len(audio)
        num_frames = (total + frame_size - 1) // frame_size
        padded = np.zeros(num_frames * frame_size, dtype=np.int16)
        padded[:total] = audio
        frames = padded.reshape(num_frames, frame_size).astype(np.float32)
        rms = np.sqrt(np.mean(frames * frames, axis=1)) + 1e-8
        db = 20 * np.log10(rms / 32768.0 + 1e-12)
        is_sil = db < threshold_db
        min_sil_frames = max(1, int(min_silence_ms / frame_ms))
        end_frame = num_frames
        run = 0
        for i in range(num_frames - 1, -1, -1):
            if is_sil[i]:
                run += 1
                if run >= min_sil_frames:
                    end_frame = i
            else:
                break
        pad_frames = max(0, int(pad_ms / frame_ms))
        end_frame = min(num_frames, end_frame + pad_frames)
        end_sample = min(total, end_frame * frame_size)
        return audio[:end_sample].astype(np.int16).tobytes()

    async def _synthesize_async(self, text: str) -> bytes:
        """Asynchronous synthesis using Edge-TTS"""
        communicate = edge_tts.Communicate(text, self.voice)

        # Collect audio data in memory
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])

        # Process audio in memory without temporary files
        with BytesIO(audio_data) as audio_buffer:
            audio, src_sr = sf.read(audio_buffer, dtype="float32")

            # Resample if necessary
            if src_sr != self.sample_rate:
                audio = soxr.resample(audio, src_sr, self.sample_rate)

            return self._float32_to_pcm_bytes(audio)

    def synthesize(self, text: str) -> bytes:
        """Synchronous synthesis"""
        pcm = asyncio.run(self._synthesize_async(text))
        if self._trim_tail:
            pcm = self._trim_tail_silence_pcm16(
                pcm,
                sample_rate=self.sample_rate,
                threshold_db=self._trim_threshold_db,
                min_silence_ms=self._trim_min_silence_ms,
                frame_ms=self._trim_frame_ms,
                pad_ms=self._trim_pad_ms,
            )
        return pcm

    def synthesize_stream(
        self,
        text: str,
        window_ms: int = 300,
        threshold_db: float = -40.0,
        min_silence_ms: int = 60,
        frame_ms: int = 10,
    ) -> Iterable[bytes]:
        """Streaming TTS with tail-window silence trimming.
        Note: If the installed edge-tts doesn't support PCM streaming, this
        will fall back to single-shot output (yield once after trimming).
        """
        chunks: List[bytes] = []

        async def _run() -> None:
            async for part in _stream_with_tail_trim_async(
                text=text,
                voice=self.voice,
                sample_rate=self.sample_rate,
                window_ms=window_ms,
                threshold_db=threshold_db,
                min_silence_ms=min_silence_ms,
                frame_ms=frame_ms,
            ):
                chunks.append(part)

        asyncio.run(_run())
        for c in chunks:
            yield c

    def get_available_voices(self) -> list:
        """Get list of available Chinese voices"""

        async def _get_voices():
            return await edge_tts.list_voices()

        voices = asyncio.run(_get_voices())
        return [v for v in voices if "zh-CN" in v["ShortName"]]

    def set_voice(self, voice: str):
        """Set the voice to use"""
        self.voice = voice


# Common Chinese voice options
CHINESE_VOICES = {
    "xiaoyi": "zh-CN-XiaoyiNeural",  # Young female voice
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # Default female voice
    "yunxi": "zh-CN-YunxiNeural",  # Young male voice
    "yunjian": "zh-CN-YunJianNeural",  # Mature male voice
    "xiaochen": "zh-CN-XiaochenNeural",  # Cute female voice
    "xiaomo": "zh-CN-XiaomoNeural",  # Gentle female voice
}
