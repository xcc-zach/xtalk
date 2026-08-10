"""Enforce the desktop application's source and public-API boundaries."""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = APP_ROOT.parent
CORE_IMPORT = re.compile(r"""from\s+["']xtalk-client/|import\s+["']xtalk-client/""")
APPROVED_XTALK_MODULES = frozenset(
    {
        "xtalk.models.agents.default",
        "xtalk.models.agents.interfaces",
        "xtalk.models.agents.tools",
        "xtalk.models.agents.tools.utils",
        "xtalk.models.asr.sherpa_onnx_asr",
        "xtalk.models.tts.moss_tts_nano",
        "xtalk.models.tts.sherpa_onnx_tts",
    }
)
FORBIDDEN_PATHS = (
    "frontend/src/platforms/tauri.ts",
    "frontend/src/bases/local-capabilities.ts",
    "src/xtalk/local_tools/",
    "apps/desktop/",
)


def iter_python_sources() -> list[Path]:
    """Return Python sources that form the desktop implementation.

    Returns
    -------
    list[pathlib.Path]
        Sorted Python paths.
    """

    return sorted(
        path
        for path in APP_ROOT.rglob("*.py")
        if not any(
            part in {".venv", ".build", "build"} for part in path.parts
        )
    )


def verify_python_imports() -> None:
    """Reject imports outside XTalk's documented Python integration APIs."""

    violations: list[str] = []
    for path in iter_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if (
                        alias.name == "xtalk"
                        or alias.name in APPROVED_XTALK_MODULES
                        or not alias.name.startswith("xtalk.")
                    ):
                        continue
                    violations.append(f"{path.relative_to(APP_ROOT)}:{node.lineno}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (
                    module.startswith("xtalk.")
                    and module not in APPROVED_XTALK_MODULES
                ):
                    violations.append(f"{path.relative_to(APP_ROOT)}:{node.lineno}: from {module}")
    if violations:
        raise ValueError("private/core submodule imports are forbidden:\n" + "\n".join(violations))


def verify_typescript_imports() -> None:
    """Reject deep imports from the frontend client package."""

    violations: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.ts")):
        if any(part in {"node_modules", "dist"} for part in path.parts):
            continue
        source = path.read_text(encoding="utf-8")
        if CORE_IMPORT.search(source):
            violations.append(path.relative_to(APP_ROOT).as_posix())
    if violations:
        raise ValueError("xtalk-client deep imports are forbidden: " + ", ".join(violations))


def changed_paths(base_ref: str) -> list[str]:
    """List committed paths changed from a base revision.

    Parameters
    ----------
    base_ref : str
        Git revision used as comparison base.

    Returns
    -------
    list[str]
        Changed repository-relative paths.
    """

    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def verify_change_scope(base_ref: str) -> None:
    """Require committed desktop changes to stay under ``app/``.

    Parameters
    ----------
    base_ref : str
        Git revision used as comparison base.
    """

    invalid = [
        path
        for path in changed_paths(base_ref)
        if not path.startswith("app/")
    ]
    if invalid:
        raise ValueError("desktop change escapes app/:\n" + "\n".join(invalid))


def verify_forbidden_paths() -> None:
    """Reject desktop implementation files in explicitly forbidden locations."""

    found = [
        path
        for value in FORBIDDEN_PATHS
        if (path := REPOSITORY_ROOT / value).exists()
    ]
    if found:
        rendered = "\n".join(str(path.relative_to(REPOSITORY_ROOT)) for path in found)
        raise ValueError("forbidden desktop paths exist:\n" + rendered)


def parse_args() -> argparse.Namespace:
    """Parse optional Git scope verification arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-ref",
        help="also verify committed paths changed from this Git revision",
    )
    return parser.parse_args()


def main() -> int:
    """Run source boundary checks.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    verify_python_imports()
    verify_typescript_imports()
    verify_forbidden_paths()
    if args.base_ref:
        verify_change_scope(args.base_ref)
    print("desktop boundary verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
