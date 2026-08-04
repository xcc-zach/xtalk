"""Sign the macOS application bundle and create a locally installable DMG."""

from __future__ import annotations

import argparse
import os
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = (
    APP_ROOT
    / "src-tauri"
    / "target"
    / "release"
    / "bundle"
    / "macos"
    / "XTalk.app"
)
ENTITLEMENTS = APP_ROOT / "src-tauri" / "Entitlements.plist"
CODEX_HOST_ENTITLEMENTS = (
    APP_ROOT / "src-tauri" / "CodexHostEntitlements.plist"
)
METADATA_SUFFIXES = (".dist-info", ".egg-info")


def parse_args() -> argparse.Namespace:
    """Parse macOS package inputs.

    Returns
    -------
    argparse.Namespace
        Parsed application, output, and signing settings.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--identity",
        default=os.environ.get("APPLE_SIGNING_IDENTITY", "-"),
        help="codesign identity; defaults to an ad-hoc local signature",
    )
    return parser.parse_args()


def run(
    command: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one packaging command without a shell.

    Parameters
    ----------
    command : list[str]
        Executable and argument vector.
    capture_output : bool, optional
        Capture text output for programmatic validation.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Successful process result.
    """

    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def link_python_metadata_to_resources(app: Path) -> list[Path]:
    """Replace framework metadata bundles with links to resource copies.

    macOS code signing interprets top-level ``*.dist-info`` directories inside
    ``Contents/Frameworks`` as nested bundles. PyInstaller still needs those
    directories at runtime, so the signed framework layout links them to the
    identical copies already bundled under ``Contents/Resources``.

    Parameters
    ----------
    app : pathlib.Path
        Tauri-generated ``.app`` bundle.

    Returns
    -------
    list[pathlib.Path]
        Metadata paths converted to relative symbolic links.
    """

    contents = app / "Contents"
    frameworks = contents / "Frameworks"
    resource_runtime = contents / "Resources" / "app-backend-runtime"
    if not frameworks.is_dir() or not resource_runtime.is_dir():
        raise ValueError("app bundle is missing the PyInstaller runtime layout")

    linked: list[Path] = []
    for path in sorted(frameworks.iterdir()):
        if not path.name.endswith(METADATA_SUFFIXES):
            continue
        resource_copy = resource_runtime / path.name
        if not resource_copy.is_dir():
            raise ValueError(f"resource runtime is missing {path.name}")
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise ValueError(f"unexpected framework metadata entry: {path}")
        relative_target = os.path.relpath(resource_copy, path.parent)
        path.symlink_to(relative_target, target_is_directory=True)
        linked.append(path)
    return linked


def verify_internal_bundle_links(app: Path) -> list[Path]:
    """Reject symbolic links that make the App depend on the build machine.

    Parameters
    ----------
    app : pathlib.Path
        Prepared application bundle.

    Returns
    -------
    list[pathlib.Path]
        Resolved internal symbolic-link targets.
    """

    bundle_root = app.resolve()
    resolved_targets: list[Path] = []
    for path in bundle_root.rglob("*"):
        if not path.is_symlink():
            continue
        link_target = Path(os.readlink(path))
        if not link_target.is_absolute():
            link_target = path.parent / link_target
        resolved_target = link_target.resolve()
        if (
            not resolved_target.is_relative_to(bundle_root)
            or not resolved_target.exists()
        ):
            raise ValueError(
                f"app bundle contains an external or broken link: {path}"
            )
        resolved_targets.append(resolved_target)
    return resolved_targets


def sign_app(app: Path, identity: str) -> None:
    """Sign nested code, the Python sidecar, and the outer app in loadable order.

    Parameters
    ----------
    app : pathlib.Path
        Prepared application bundle.
    identity : str
        Apple signing identity or ``-`` for an ad-hoc signature.
    """

    common = [
        "codesign",
        "--force",
        "--sign",
        identity,
        "--options",
        "runtime",
        "--entitlements",
        str(ENTITLEMENTS),
    ]
    run([*common, "--deep", str(app)])

    # --deep signs nested executables but does not propagate entitlements to
    # them. The Codex code-mode host embeds V8 and requires executable-memory
    # entitlements even in jitless mode; without them hardened-runtime builds
    # crash while creating the first isolate. Sign both packaged runtime copies
    # because Tauri may resolve either layout. The PyInstaller bootloader must
    # also be allowed to load libpython, then the outer seal is refreshed.
    codex_host_relative = (
        Path("codex_cli_bin") / "bin" / "codex-code-mode-host"
    )
    codex_common = [
        "codesign",
        "--force",
        "--sign",
        identity,
        "--options",
        "runtime",
        "--entitlements",
        str(CODEX_HOST_ENTITLEMENTS),
    ]
    for runtime_root in (
        app / "Contents" / "Frameworks",
        app / "Contents" / "Resources" / "app-backend-runtime",
    ):
        codex_host = runtime_root / codex_host_relative
        if not codex_host.is_file():
            raise ValueError(
                "app bundle is missing Codex code-mode host: "
                f"{codex_host}"
            )
        run([*codex_common, str(codex_host)])
    backend = app / "Contents" / "MacOS" / "app-backend"
    run([*common, str(backend)])
    run([*common, str(app)])
    run(["codesign", "--verify", "--deep", "--strict", str(app)])


def smoke_test_backend(app: Path) -> None:
    """Confirm the signed PyInstaller process can load its Python runtime.

    Parameters
    ----------
    app : pathlib.Path
        Signed application bundle.
    """

    backend = app / "Contents" / "MacOS" / "app-backend"
    result = subprocess.run(
        [str(backend)],
        input="",
        capture_output=True,
        text=True,
        check=False,
    )
    expected = "startup configuration was not provided on stdin"
    if result.returncode != 2 or expected not in result.stderr:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            f"signed app-backend smoke test failed ({result.returncode}): {detail}"
        )


def smoke_test_matcha_runtime(app: Path) -> None:
    """Confirm the signed Matcha sidecar resolves only packaged libraries.

    Parameters
    ----------
    app : pathlib.Path
        Signed application bundle.
    """

    runtime = app / "Contents" / "MacOS" / "matcha-model-runtime"
    if not runtime.is_file():
        raise ValueError(f"app bundle is missing Matcha runtime: {runtime}")
    result = subprocess.run(
        [str(runtime), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(
            "signed Matcha runtime smoke test failed "
            f"({result.returncode}): {detail}"
        )


def default_output_path(app: Path) -> Path:
    """Derive the conventional Tauri DMG output path.

    Parameters
    ----------
    app : pathlib.Path
        Application bundle containing ``Info.plist``.

    Returns
    -------
    pathlib.Path
        Versioned architecture-specific disk image path.
    """

    with (app / "Contents" / "Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    version = info["CFBundleShortVersionString"]
    architecture = {"arm64": "aarch64"}.get(platform.machine(), platform.machine())
    return app.parent.parent / "dmg" / f"XTalk_{version}_{architecture}.dmg"


def create_dmg(app: Path, output: Path) -> None:
    """Create and verify a compressed DMG containing the signed app.

    Parameters
    ----------
    app : pathlib.Path
        Signed application bundle.
    output : pathlib.Path
        Destination disk image.
    """

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not output.is_file():
            raise ValueError("DMG output path exists and is not a file")
        output.unlink()
    with tempfile.TemporaryDirectory(prefix="xtalk-dmg-") as temporary:
        staging = Path(temporary)
        run(["ditto", str(app), str(staging / app.name)])
        (staging / "Applications").symlink_to("/Applications", target_is_directory=True)
        run(
            [
                "hdiutil",
                "create",
                "-volname",
                "XTalk",
                "-srcfolder",
                str(staging),
                "-ov",
                "-format",
                "UDZO",
                str(output),
            ]
        )
    run(["hdiutil", "verify", str(output)])


def main() -> int:
    """Prepare, sign, smoke-test, and package the macOS application.

    Returns
    -------
    int
        Process exit status.
    """

    if sys.platform != "darwin":
        raise RuntimeError("macOS DMG packaging must run on macOS")
    args = parse_args()
    app = args.app.expanduser().resolve()
    if not app.is_dir() or app.suffix != ".app":
        raise ValueError("--app must point to an existing .app bundle")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(app)
    )
    if output.suffix.lower() != ".dmg":
        raise ValueError("--output must use the .dmg extension")

    linked = link_python_metadata_to_resources(app)
    internal_links = verify_internal_bundle_links(app)
    sign_app(app, args.identity)
    smoke_test_backend(app)
    smoke_test_matcha_runtime(app)
    create_dmg(app, output)
    print(
        f"linked {len(linked)} Python metadata directories and verified "
        f"{len(internal_links)} internal links"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
