"""
Silero VAD wrapper that mimics webrtcvad.Vad API (`is_speech`).
It keeps an internal VADIterator so it can be called chunk-by-chunk.
"""


from pathlib import Path
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np
import torch
from ..interfaces import VAD


SILERO_VAD_ZIP_URL = (
    "https://codeload.github.com/snakers4/silero-vad/legacy.zip/refs/heads/master"
)
SILERO_VAD_REPO_DIRNAME = "snakers4_silero-vad_master"


def _ensure_local_silero_vad_repo() -> Path:
    hub_dir = Path(torch.hub.get_dir())
    repo_dir = hub_dir / SILERO_VAD_REPO_DIRNAME
    if repo_dir.exists():
        return repo_dir

    hub_dir.mkdir(parents=True, exist_ok=True)
    zip_path = hub_dir / "snakers4_silero-vad_master.zip"
    if not zip_path.exists():
        with urlopen(SILERO_VAD_ZIP_URL) as response, zip_path.open("wb") as file_obj:
            shutil.copyfileobj(response, file_obj)

    extract_dir = Path(tempfile.mkdtemp(prefix="silero-vad-", dir=hub_dir))
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)

        extracted_dirs = [path for path in extract_dir.iterdir() if path.is_dir()]
        if len(extracted_dirs) != 1:
            raise RuntimeError(
                f"Expected one extracted directory from Silero VAD archive, got {len(extracted_dirs)}"
            )

        extracted_dirs[0].rename(repo_dir)
    except Exception:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        raise
    finally:
        shutil.rmtree(extract_dir, ignore_errors=True)

    return repo_dir


class SileroVAD(VAD):
    def __init__(self, threshold: float = 0.5) -> None:
        repo_dir = _ensure_local_silero_vad_repo()
        model, _ = torch.hub.load(
            repo_or_dir=str(repo_dir),
            model="silero_vad",
            source="local",
        )
        self._model = model

        self.threshold = threshold
        self.window_samples = 512

    def is_speech(self, frame: bytes) -> bool:
        # int16 PCM ➜ float32 tensor
        pcm = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        wav = torch.from_numpy(pcm).unsqueeze(0)  # [1, T]

        # Feed window_samples-sized chunks to VADIterator
        num_samples = self.window_samples
        prob: float = 0.0  # probability of the last processed chunk

        # Iterate over the waveform using fixed windows
        for start in range(0, wav.shape[1], num_samples):
            chunk = wav[:, start : start + num_samples]
            if chunk.shape[1] < num_samples:
                break
            # VADIterator returns speech probability for this chunk
            prob = float(self._model(chunk.squeeze(0), 16000).item())

        # Use the probability of the last full chunk as the speech decision
        return prob >= self.threshold
