import asyncio
import unicodedata
from typing import Callable, Awaitable


def is_punctuation(ch: str) -> bool:
    return unicodedata.category(ch).startswith("P")


class MockStreamRecognizer:
    SAMPLE_RATE = 16000

    def __init__(
        self,
        recognize_fn: Callable[[bytes], Awaitable[str]],
        *,
        window_size: int = 5,
        trigger_interval_sec: float = 2,
    ):
        self.recognize_fn = recognize_fn
        self.window_size = window_size
        self._bytes_per_second = self.SAMPLE_RATE * 2 * 1
        self._trigger_bytes = int(trigger_interval_sec * self._bytes_per_second)
        self._reset_epoch = 0
        self.reset()

    def reset(self):
        self._reset_epoch = (self._reset_epoch + 1) % 1000
        self._cache: list[tuple[str, bytes]] = []
        self._stable_text = ""
        self.recognized_text = ""
        self._audio_to_recognize = bytearray()
        self._delta_audio = bytearray()

    def _get_prefix_change_index(self):
        len_cache = len(self._cache)
        if len_cache == 1:
            return 0
        for i in range(len_cache - 1):
            prev_elem = self._cache[i][0]
            target_elem = self._cache[i + 1][0]
            if not target_elem.startswith(prev_elem):
                return i + 1
        return len_cache - 1

    def recognize(self, audio: bytes, *, is_final: bool = False) -> str:
        return asyncio.run(self.async_recognize(audio, is_final=is_final))

    async def async_recognize(self, audio: bytes, *, is_final: bool = False) -> str:
        start_epoch = self._reset_epoch
        self._audio_to_recognize.extend(audio)
        self._delta_audio.extend(audio)

        if not is_final and len(self._audio_to_recognize) < self._trigger_bytes:
            return self.recognized_text

        old_recognized_text = self.recognized_text
        new_text = await self.recognize_fn(bytes(self._audio_to_recognize))
        if self._reset_epoch != start_epoch:
            return self.recognized_text
        self.recognized_text = self._stable_text + (new_text or "")
        # Remove trailing punctuation
        if self.recognized_text and is_punctuation(self.recognized_text[-1]):
            self.recognized_text = self.recognized_text[:-1]
        if self.recognized_text != old_recognized_text:
            self._cache.append((self.recognized_text, bytes(self._delta_audio)))
            self._delta_audio.clear()
            if len(self._cache) >= self.window_size:
                prefix_change_index = self._get_prefix_change_index()
                self._stable_text = self._cache[prefix_change_index][0]
                self._audio_to_recognize = bytearray()
                for _, audio_chunk in self._cache[prefix_change_index + 1 :]:
                    self._audio_to_recognize.extend(audio_chunk)
                self._cache = []
        return self.recognized_text
