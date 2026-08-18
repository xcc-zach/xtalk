"""Executable entrypoint for the desktop Python sidecar."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path
from typing import TextIO

if __package__ in {None, ""}:
    # PyInstaller may analyze and execute this file as a direct script. Add only
    # the app package root, then continue through the same public entrypoint.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend.config import read_startup_config
else:
    from .config import read_startup_config


def main(
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
) -> int:
    """Run the sidecar from one newline-delimited stdin launch message.

    Normal application output is redirected to stderr for the full process
    lifetime. The original stdout is reserved so its first line is always the
    readiness protocol message.

    Parameters
    ----------
    stdin : TextIO | None, optional
        Launch protocol input. Defaults to process stdin.
    stdout : TextIO | None, optional
        Launch protocol output. Defaults to process stdout.
    stderr : TextIO | None, optional
        Diagnostic output. Defaults to process stderr.

    Returns
    -------
    int
        Process exit status.
    """

    input_stream = stdin if stdin is not None else sys.stdin
    protocol_output = stdout if stdout is not None else sys.stdout
    diagnostic_output = stderr if stderr is not None else sys.stderr

    try:
        startup = read_startup_config(input_stream)
    except (ValueError, OSError) as exc:
        diagnostic_output.write(f"Invalid sidecar startup configuration: {exc}\n")
        diagnostic_output.flush()
        return 2

    try:
        # Importing and building model providers may produce ordinary stdout.
        # Redirect it before importing the runtime so protocol stdout stays clean.
        with contextlib.redirect_stdout(diagnostic_output):
            if __package__ in {None, ""}:
                from backend.server import serve_sidecar
            else:
                from .server import serve_sidecar

            asyncio.run(
                serve_sidecar(
                    startup,
                    protocol_output=protocol_output,
                )
            )
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        diagnostic_output.write(
            f"Sidecar failed ({type(exc).__name__}): {exc}\n"
        )
        diagnostic_output.flush()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
