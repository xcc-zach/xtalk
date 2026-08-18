import torch
import numpy as np
from pyannote.audio import Inference
from pyannote.audio import Model
from .interfaces import SpeakerEncoder
from ..registry import model


@model
class PyannoteSpeakerEncoder(SpeakerEncoder):
    """pyannote.audio-based speaker encoder (wespeaker resnet34 LM)."""

    def __init__(
        self,
        model_name: str = "pyannote/wespeaker-voxceleb-resnet34-LM",
        device: str = None,
        min_duration_sec: float = 3,
    ):
        """Initialize the encoder."""
        # Keep device as string for Inference class
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.min_duration_sec = float(min_duration_sec)
        self._expected_sr = 16000  # Unified sample rate

        model = Model.from_pretrained(model_name)
        self.model = Inference(model, window="whole", device=torch.device(self.device))

    def extract(self, audio: bytes) -> np.ndarray:
        """Extract an embedding from PCM bytes."""
        # Validate expected PCM format (16 kHz mono 16-bit)
        if not isinstance(audio, (bytes, bytearray, memoryview)):
            raise TypeError(
                f"[SpeakerEncoder] Expected PCM bytes input, got {type(audio)}"
            )

        np_int16 = np.frombuffer(audio, dtype=np.int16)
        if np_int16.size == 0:
            raise ValueError("[SpeakerEncoder] Empty PCM bytes for embedding.")

        # Normalize to [-1, 1]
        waveform_np = np_int16.astype(np.float32) / 32768.0

        # Repeat audio if shorter than min_duration_sec (16 kHz)
        min_samples = int(self._expected_sr * self.min_duration_sec)
        if waveform_np.shape[0] < min_samples:
            reps = int(np.ceil(min_samples / waveform_np.shape[0]))
            # Tile and truncate to meet minimum duration
            waveform_np = np.tile(waveform_np, reps)[:min_samples]

        waveform = torch.from_numpy(waveform_np)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)

        embedding = self.model({"waveform": waveform, "sample_rate": self._expected_sr})

        if isinstance(embedding, np.ndarray):
            return embedding.flatten()
        elif torch.is_tensor(embedding):
            return embedding.cpu().numpy().flatten()
        else:
            raise TypeError(
                f"[SpeakerEncoder] Unexpected embedding type: {type(embedding)}"
            )

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Override to clarify cosine similarity is used."""
        e1 = embedding1.astype(np.float32, copy=False).ravel()
        e2 = embedding2.astype(np.float32, copy=False).ravel()
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        return float(np.dot(e1, e2) / (n1 * n2))
