# -*- coding: utf-8 -*-
"""
FunASR ct-punc based punctuation restorer.

Usage:

    from xtalk.models.punt_restorer.ct_punt import CTPunctRestorer

    restorer = CTPunctRestorer()
    text = "Meeting ends here, happy new year, see you next year"
    print(restorer.restore(text))

Notes:
- Depends on funasr.AutoModel (default model "ct-punc").
- Provides `restore` and `restore_batch`; errors are logged and original text returned.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

from ..registry import model
from .interfaces import PuntRestorer

logger = logging.getLogger(__name__)

try:
    from funasr import AutoModel  # type: ignore
except Exception as e:  # pragma: no cover - import guard for missing env
    AutoModel = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


@model
class CTPunctRestorer(PuntRestorer):
    """ct-punc punctuation restoration helper."""

    def __init__(
        self,
        model_name: str = "ct-punc",
        device: str = "cpu",
        hub: str = "ms",
        disable_update: bool = True,
        **kwargs,
    ) -> None:
        if AutoModel is None:  # pragma: no cover
            logger.error(f"funasr is not available: {_IMPORT_ERROR}")
            raise RuntimeError(
                "funasr is required for CTPunctRestorer but failed to import."
            )

        # Initialize FunASR model
        try:
            self._model = AutoModel(
                model=model_name,
                device=device,
                hub=hub,
                disable_update=disable_update,
                **kwargs,
            )
        except Exception as e:  # pragma: no cover - dependency init failure
            logger.error(f"Failed to init ct-punc model: {e}")
            raise

    def restore(self, text: str) -> str:
        """Restore punctuation for a single text; fall back to original on error."""
        if not isinstance(text, str) or not text:
            return text
        try:
            res = self._model.generate(input=text)
            # Expect shape [{'key': '...', 'text': '...'}]
            if isinstance(res, (list, tuple)) and res:
                item = res[0]
                restored = item.get("text") if isinstance(item, dict) else None  # type: ignore[assignment]
                return restored if isinstance(restored, str) and restored else text
            return text
        except Exception as e:  # pragma: no cover
            logger.error(f"Punctuation restoration failed: {e}")
            return text

    def restore_batch(self, texts: Iterable[str]) -> List[str]:
        """Restore punctuation for iterable of strings, preserving order."""
        out: List[str] = []
        for t in texts:
            out.append(self.restore(t))
        return out
