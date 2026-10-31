"""Test harness for the conversation experience tests.

Importing this package puts the repository's ``src`` directory first on
``sys.path`` so the tests and the mock server always exercise the checkout
they live in, even when another xtalk installation is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
    sys.modules.pop("xtalk", None)
