from ..interfaces import ASR
from funasr import AutoModel
from typing import Optional, Dict, Any
import numpy as np


# TODO: align sample rates
class ParaformerLocal(ASR):
    TARGET_SAMPLE_RATE = 16000

    def __init__(
        self,
        model: str = "paraformer-zh-streaming",
        vad_model: Optional[str] = "fsmn-vad",
        punc_model: Optional[str] = "ct-punc",
        spk_model: Optional[str] = None,
        device: str = "cpu",
        ncpu: int = 4,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
        hub: str = "hf",
        batch_size_s: int = 300,
        hotword: Optional[str] = None,
        disable_update: bool = True,
        chunk_size: Optional[list] = [0, 10, 5],  # [0, 10, 5] 600ms
        encoder_chunk_look_back: int = 4,  # number of chunks to lookback for encoder self-attention
        decoder_chunk_look_back: int = 1,  # number of encoder chunks to lookback for decoder cross-attention
        _shared_model: Optional[Any] = None,  # shared model instance for clone()
        **kwargs: Dict[str, Any],
    ):
        """
        Local Paraformer ASR implementation.

        Args:
            model (str): Model name or local path, defaults to "paraformer-zh-streaming".
            vad_model (str, optional): VAD model name, defaults to "fsmn-vad"; set to None for streaming inference.
            punc_model (str, optional): Punctuation model name, defaults to "ct-punc"; set to None for streaming inference.
            spk_model (str, optional): Speaker diarization model name, defaults to None; set to None for streaming inference.
            device (str): Device identifier such as cuda:0 / cpu / mps / xpu.
            ncpu (int): Number of CPU threads, defaults to 4.
            output_dir (str, optional): Directory for dumping recognition outputs.
            batch_size (int): Batch size during decoding.
            hub (str): Model download source, ms (ModelScope) or hf (Hugging Face).
            batch_size_s (int): Batch size measured in seconds, defaults to 300.
            hotword (str, optional): Hotword string.
            disable_update (bool): Whether to disable the model update check, defaults to True.
            chunk_size (list, optional): Chunk configuration for streaming inference, default [0, 10, 5] (600 ms).
            encoder_chunk_look_back (int): Encoder self-attention history size, defaults to 4.
            decoder_chunk_look_back (int): Decoder cross-attention look-back size, defaults to 1.
            **kwargs: Extra config parameters, e.g., values from config.yaml such as max_single_segment_time=6000.
        """
        self.model_name = model
        self.vad_model = vad_model
        self.punc_model = punc_model
        self.spk_model = spk_model
        self.device = device
        self.ncpu = ncpu
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.hub = hub
        self.batch_size_s = batch_size_s
        self.hotword = hotword
        self.disable_update = disable_update
        self.kwargs = kwargs

        # Streaming recognition parameters
        self.chunk_size = chunk_size
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.chunk_secs = self.chunk_size[1] * 0.06

        # Streaming caches and buffers
        # funasr streaming mode needs cache; non-streaming models emulate streaming via buffered input
        self._stream_cache: dict = {}
        self._stream_text: str = ""
        self._stream_audio = bytearray()

        # Initialize the model unless a shared instance is provided
        if _shared_model is not None:
            self.model = _shared_model
        else:
            self._init_model()

    def _init_model(self):
        """Initialize the funasr model."""
        try:
            # Build base kwargs
            model_kwargs = {
                "model": self.model_name,
                "device": self.device,
                "ncpu": self.ncpu,
                "batch_size": self.batch_size,
                "hub": self.hub,
                "disable_update": self.disable_update,
            }

            # Attach optional sub-models when available
            if self.vad_model and not self.model_name.endswith("streaming"):
                model_kwargs["vad_model"] = self.vad_model
            if self.punc_model and not self.model_name.endswith("streaming"):
                model_kwargs["punc_model"] = self.punc_model
            if self.spk_model and not self.model_name.endswith("streaming"):
                model_kwargs["spk_model"] = self.spk_model

            # Output directory
            if self.output_dir:
                model_kwargs["output_dir"] = self.output_dir

            # Extra configuration parameters
            if self.kwargs:
                model_kwargs.update(self.kwargs)

            self.model = AutoModel(**model_kwargs)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Paraformer model: {e}")

    def recognize(self, audio: bytes) -> str:
        """Convert PCM 16 kHz mono 16-bit raw bytes to text (one-shot)."""
        try:
            # Convert PCM bytes to float32 samples in [-1, 1]
            speech = self._pcm_to_float(audio)

            if self.model_name.endswith("streaming"):
                # Streaming models still support batch decoding with a temporary cache
                temp_cache: dict = {}
                result = self.model.generate(
                    input=speech,
                    cache=temp_cache,
                    is_final=True,
                    chunk_size=self.chunk_size,
                    encoder_chunk_look_back=self.encoder_chunk_look_back,
                    decoder_chunk_look_back=self.decoder_chunk_look_back,
                )
            else:
                # Non-streaming models perform standard batch decoding
                result = self.model.generate(
                    input=speech, batch_size_s=self.batch_size_s, hotword=self.hotword
                )

            return self._parse_result(result)
        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {e}")

    def _set_hotword(self, hotword: str):
        """Set the hotword (internal API)."""
        self.hotword = hotword

    def _set_device(self, device: str):
        """Set inference device (internal API)."""
        self.device = device
        # Device change requires reinitialization
        self._init_model()

    def _get_model_info(self) -> dict:
        """Return model information (internal API)."""
        return {
            "model": self.model_name,
            "vad_model": self.vad_model,
            "punc_model": self.punc_model,
            "spk_model": self.spk_model,
            "device": self.device,
            "ncpu": self.ncpu,
            "output_dir": self.output_dir,
            "batch_size": self.batch_size,
            "hub": self.hub,
            "batch_size_s": self.batch_size_s,
            "hotword": self.hotword,
            "extra_kwargs": self.kwargs,
        }

    def _single_recognize_stream(self, audio: np.array, is_final: bool) -> str:
        result = self.model.generate(
            input=audio,
            cache=self._stream_cache,
            is_final=is_final,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
        )
        return self._parse_result(result)

    def recognize_stream(self, audio: bytes, *, is_final: bool) -> str:
        """Streaming recognition that manages funasr cache or buffered chunks."""
        if not self.model_name.endswith("streaming"):
            if audio:
                self._stream_audio.extend(audio)
            if self._stream_audio:
                text = self.recognize(bytes(self._stream_audio))
                if text:
                    self._stream_text = text
            return self._stream_text
        audio_float = self._pcm_to_float(audio)
        delta = ""
        chunks = self._get_chunks(audio_float)
        for idx, chunk in enumerate(chunks):
            chunk_is_final = bool(is_final and (idx == len(chunks) - 1))
            chunk_text = self._single_recognize_stream(chunk, chunk_is_final)
            if chunk_text:
                delta += chunk_text
        if delta:
            self._stream_text += delta
        return self._stream_text

    def reset(self) -> None:
        """Reset streaming buffers."""
        self._stream_cache = {}
        self._stream_text = ""
        self._stream_audio = bytearray()

    def clone(self) -> "ParaformerLocal":
        """Create a clone that shares the model while keeping independent state."""
        return ParaformerLocal(
            model=self.model_name,
            vad_model=self.vad_model,
            punc_model=self.punc_model,
            spk_model=self.spk_model,
            device=self.device,
            ncpu=self.ncpu,
            output_dir=self.output_dir,
            batch_size=self.batch_size,
            hub=self.hub,
            batch_size_s=self.batch_size_s,
            hotword=self.hotword,
            disable_update=self.disable_update,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
            _shared_model=self.model,  # shared model instance
            **self.kwargs,
        )

    def stream_chunk_bytes_hint(self) -> int | None:
        """Return the suggested streaming trigger threshold (in bytes).

        Translate the internal chunk duration (chunk_secs) to a byte hint:
        bytes = seconds * sample_rate * 2 (16-bit). Return None if the model
        does not align with this behavior, though Paraformer streaming does.
        """
        secs = float(self.chunk_secs)
        sr = self.TARGET_SAMPLE_RATE
        return max(0, int(secs * sr * 2)) or None

    def _get_chunks(self, audio: np.ndarray) -> list[np.ndarray]:
        chunk_stride = int(self.chunk_secs * self.TARGET_SAMPLE_RATE)
        total_chunk_num = int((len(audio) - 1) / chunk_stride + 1)
        chunks = []
        for i in range(total_chunk_num):
            chunks.append(audio[i * chunk_stride : (i + 1) * chunk_stride])
        return chunks

    def _pcm_to_float(self, pcm: bytes) -> np.ndarray:
        """Convert PCM int16 bytes into a float32 array in [-1, 1]."""
        if not pcm:
            return np.zeros((0,), dtype=np.float32)
        x = np.frombuffer(pcm, dtype=np.int16)
        return x.astype(np.float32) / 32768.0

    def _parse_result(self, result: Any) -> str:
        """Normalize funasr return values into a string."""
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "text" in result[0]:
                return "".join([item.get("text", "") for item in result])
            elif isinstance(result[0], str):
                return "".join(result)
            else:
                return str(result[0])
        elif isinstance(result, dict) and "text" in result:
            return result["text"]
        else:
            return str(result)
