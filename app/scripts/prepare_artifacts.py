"""Copy immutable core artifacts into the isolated desktop build context."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = APP_ROOT / "resources" / "artifacts"
MANIFEST_PATH = APP_ROOT / "resources" / "manifests" / "core-artifacts.lock.json"


@dataclass(frozen=True)
class ArtifactInput:
    """Describe one immutable artifact supplied to the desktop build.

    Parameters
    ----------
    kind : str
        Stable artifact category.
    path : pathlib.Path
        Existing source artifact.
    version : str
        Declared immutable version.
    expected_suffixes : tuple[str, ...]
        Accepted filename endings.
    """

    kind: str
    path: Path
    version: str
    expected_suffixes: tuple[str, ...]


def sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest of a file.

    Parameters
    ----------
    path : pathlib.Path
        File to hash.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_artifact(artifact: ArtifactInput) -> dict[str, str]:
    """Validate and copy one artifact into the app resource directory.

    Parameters
    ----------
    artifact : ArtifactInput
        Artifact metadata and source path.

    Returns
    -------
    dict[str, str]
        Serializable locked artifact record.

    Raises
    ------
    FileNotFoundError
        Raised when the source does not exist.
    ValueError
        Raised when version or filename is invalid.
    """

    source = artifact.path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not artifact.version.strip():
        raise ValueError(f"{artifact.kind} version must not be empty")
    if not source.name.endswith(artifact.expected_suffixes):
        endings = ", ".join(artifact.expected_suffixes)
        raise ValueError(f"{artifact.kind} must end with one of: {endings}")

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    destination = ARTIFACT_DIR / source.name
    shutil.copyfile(source, destination)
    return {
        "kind": artifact.kind,
        "version": artifact.version.strip(),
        "filename": destination.relative_to(APP_ROOT).as_posix(),
        "sha256": sha256_file(destination),
        "source": "explicit-build-input",
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line artifact locations and versions.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtalk-wheel", required=True, type=Path)
    parser.add_argument("--xtalk-version", required=True)
    parser.add_argument("--client-package", required=True, type=Path)
    parser.add_argument("--client-version", required=True)
    return parser.parse_args()


def main() -> int:
    """Prepare both core artifacts and write their lock manifest.

    Returns
    -------
    int
        Process exit status.
    """

    args = parse_args()
    records = [
        prepare_artifact(
            ArtifactInput(
                kind="xtalk-wheel",
                path=args.xtalk_wheel,
                version=args.xtalk_version,
                expected_suffixes=(".whl",),
            )
        ),
        prepare_artifact(
            ArtifactInput(
                kind="xtalk-client",
                path=args.client_package,
                version=args.client_version,
                expected_suffixes=(".tgz", ".tar.gz"),
            )
        ),
    ]
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {"schema_version": 1, "artifacts": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(MANIFEST_PATH.relative_to(APP_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
