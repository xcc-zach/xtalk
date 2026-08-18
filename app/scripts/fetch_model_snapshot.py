"""Fetch an immutable private model snapshot using an environment token."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse the repository, revision, and destination.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    """Download a model snapshot without exposing its access token.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    destination = args.local_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "HF_HOME",
        str(destination / ".cache" / "huggingface"),
    )
    os.environ.setdefault(
        "HF_XET_CACHE",
        str(destination / ".cache" / "xet"),
    )
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=destination,
        token=token,
    )
    print(Path(resolved).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
