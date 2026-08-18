"""Build and publish one validated macOS application and disk image."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
APP_OUTPUT = (
    APP_ROOT
    / "src-tauri"
    / "target"
    / "release"
    / "bundle"
    / "macos"
    / "XTalk.app"
)
TAURI_CONFIG = APP_ROOT / "src-tauri" / "tauri.conf.json"


def parse_args() -> argparse.Namespace:
    """Parse release orchestration options.

    Returns
    -------
    argparse.Namespace
        Selected local or distributable packaging mode.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="create an ad-hoc signed installer for this Mac only",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def default_output_path() -> Path:
    """Derive the versioned disk-image path before the build starts.

    Returns
    -------
    pathlib.Path
        Conventional Tauri disk-image output path.
    """

    config = json.loads(TAURI_CONFIG.read_text(encoding="utf-8"))
    version = config["version"]
    architecture = {"arm64": "aarch64"}.get(
        platform.machine(),
        platform.machine(),
    )
    return APP_OUTPUT.parent.parent / "dmg" / (
        f"XTalk_{version}_{architecture}.dmg"
    )


def distribution_credentials() -> tuple[str, str]:
    """Read mandatory release credentials from the process environment.

    Returns
    -------
    tuple[str, str]
        Developer ID identity and notarytool Keychain profile.

    Raises
    ------
    ValueError
        If either required external credential is absent.
    """

    identity = os.environ.get("APPLE_SIGNING_IDENTITY", "").strip()
    profile = os.environ.get("APPLE_NOTARY_KEYCHAIN_PROFILE", "").strip()
    if not identity or identity == "-":
        raise ValueError(
            "APPLE_SIGNING_IDENTITY must name a Developer ID Application "
            "certificate for distributable packaging"
        )
    if not profile:
        raise ValueError(
            "APPLE_NOTARY_KEYCHAIN_PROFILE must name a notarytool "
            "Keychain profile for distributable packaging"
        )
    return identity, profile


def remove_output(path: Path) -> None:
    """Remove one known packaging output without following broad patterns.

    Parameters
    ----------
    path : pathlib.Path
        Exact App bundle or disk-image output path.
    """

    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def prepare_native_runtime() -> None:
    """Download, verify, build, and stage the current host runtime."""

    subprocess.run(
        [
            sys.executable,
            str(APP_ROOT / "scripts" / "download_managed_runtime.py"),
        ],
        cwd=APP_ROOT,
        check=True,
    )


def run_release(
    *,
    local: bool,
    output: Path,
    identity: str,
    notary_profile: str | None,
) -> None:
    """Build, sign, verify, and publish without leaving partial outputs.

    Parameters
    ----------
    local : bool
        Whether to use local ad-hoc signing instead of notarization.
    output : pathlib.Path
        Exact final DMG output path.
    identity : str
        Code-signing identity or ``-`` for local packaging.
    notary_profile : str or None
        Keychain profile used by ``notarytool`` in distribution mode.
    """

    remove_output(APP_OUTPUT)
    remove_output(output)
    completed = False
    try:
        prepare_native_runtime()
        subprocess.run(
            [
                "npm",
                "run",
                "tauri",
                "--",
                "build",
                "--bundles",
                "app",
            ],
            cwd=APP_ROOT,
            check=True,
        )
        package_command = [
            sys.executable,
            str(APP_ROOT / "scripts" / "package_macos_dmg.py"),
            "--app",
            str(APP_OUTPUT),
            "--output",
            str(output),
            "--identity",
            identity,
        ]
        if not local:
            if notary_profile is None:
                raise ValueError("distribution packaging requires a notary profile")
            package_command.extend(
                ["--notarize", "--notary-profile", notary_profile]
            )
        subprocess.run(package_command, cwd=APP_ROOT, check=True)
        if not APP_OUTPUT.is_dir() or not output.is_file():
            raise RuntimeError("packaging completed without both App and DMG outputs")
        completed = True
    finally:
        if not completed:
            remove_output(APP_OUTPUT)
            remove_output(output)


def main() -> int:
    """Run the supported macOS packaging entrypoint.

    Returns
    -------
    int
        Process exit status.
    """

    if sys.platform != "darwin":
        raise RuntimeError("macOS release packaging must run on macOS")
    args = parse_args()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path()
    )
    if output.suffix.lower() != ".dmg":
        raise ValueError("--output must use the .dmg extension")

    if args.local:
        identity = "-"
        notary_profile = None
    else:
        identity, notary_profile = distribution_credentials()
    run_release(
        local=args.local,
        output=output,
        identity=identity,
        notary_profile=notary_profile,
    )
    print(f"validated App: {APP_OUTPUT}")
    print(f"validated installer: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
