"""Contract checks for the private model snapshot helper."""

from __future__ import annotations

import ast
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_helper_only_reads_token_from_environment() -> None:
    """Ensure the helper has no token command-line option."""

    source_path = APP_ROOT / "scripts" / "fetch_model_snapshot.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    option_strings = {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
    }
    assert "--token" not in option_strings
    assert "--hf-token" not in option_strings
