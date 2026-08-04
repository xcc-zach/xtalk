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
    parser.add_argument(
        "--notarize",
        action="store_true",
        help="submit the signed DMG to Apple's notary service and staple it",
    )
    parser.add_argument(
        "--notary-profile",
        default=os.environ.get("APPLE_NOTARY_KEYCHAIN_PROFILE"),
        help="notarytool Keychain profile used when --notarize is set",
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


def runtime_layout(root: Path) -> dict[Path, str]:
    """Describe non-metadata entries in a packaged Python runtime.

    Parameters
    ----------
    root : pathlib.Path
        Complete Python runtime directory.

    Returns
    -------
    dict[pathlib.Path, str]
        Relative paths mapped to ``directory``, ``file``, or ``symlink``.
    """

    layout: dict[Path, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts[0].endswith(METADATA_SUFFIXES):
            continue
        if path.is_symlink():
            kind = "symlink"
        elif path.is_dir():
            kind = "directory"
        elif path.is_file():
            kind = "file"
        else:
            kind = "other"
        layout[relative] = kind
    return layout


def path_logical_size(path: Path) -> int:
    """Measure regular-file bytes allocated below one runtime entry.

    Parameters
    ----------
    path : pathlib.Path
        File, link, or directory about to be removed.

    Returns
    -------
    int
        Sum of regular-file logical sizes in bytes.
    """

    if path.is_symlink():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(
        entry.stat().st_size
        for entry in path.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )


def prune_duplicate_resource_runtime(app: Path) -> tuple[int, int]:
    """Keep one complete macOS Python runtime under ``Frameworks``.

    Tauri initially places the PyInstaller onedir tree in both ``Frameworks``
    and ``Resources``. The bootloader loads native code from ``Frameworks``;
    only package metadata must remain in ``Resources`` so it can be referenced
    without being interpreted as nested code-signing bundles.

    Parameters
    ----------
    app : pathlib.Path
        Tauri-generated application bundle.

    Returns
    -------
    tuple[int, int]
        Number of removed top-level entries and their logical byte size.

    Raises
    ------
    ValueError
        If either runtime is missing or their non-metadata layouts differ.
    """

    contents = app / "Contents"
    frameworks = contents / "Frameworks"
    resource_runtime = contents / "Resources" / "app-backend-runtime"
    if not frameworks.is_dir() or not resource_runtime.is_dir():
        raise ValueError("app bundle is missing the PyInstaller runtime layout")

    framework_layout = runtime_layout(frameworks)
    resource_layout = runtime_layout(resource_runtime)
    if framework_layout != resource_layout:
        framework_only = sorted(framework_layout.keys() - resource_layout.keys())
        resource_only = sorted(resource_layout.keys() - framework_layout.keys())
        kind_mismatches = sorted(
            path
            for path in framework_layout.keys() & resource_layout.keys()
            if framework_layout[path] != resource_layout[path]
        )
        detail = ", ".join(
            str(path)
            for path in [*framework_only, *resource_only, *kind_mismatches][:5]
        )
        raise ValueError(
            "Frameworks and Resources Python runtime layouts differ"
            + (f": {detail}" if detail else "")
        )

    removed_count = 0
    removed_bytes = 0
    for path in sorted(resource_runtime.iterdir()):
        if path.name.endswith(METADATA_SUFFIXES):
            continue
        removed_bytes += path_logical_size(path)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise ValueError(f"unexpected Python runtime entry: {path}")
        removed_count += 1
    return removed_count, removed_bytes


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


def signing_options(identity: str) -> list[str]:
    """Return hardened-runtime options for one signing identity.

    Parameters
    ----------
    identity : str
        Apple signing identity or ``-`` for an ad-hoc signature.

    Returns
    -------
    list[str]
        Common ``codesign`` options, including a secure timestamp for a
        Developer ID identity.
    """

    options = ["--options", "runtime"]
    if identity != "-":
        options.append("--timestamp")
    return options


def sign_app(app: Path, identity: str) -> None:
    """Sign nested code, the Python sidecar, and the outer app in loadable order.

    Parameters
    ----------
    app : pathlib.Path
        Prepared application bundle.
    identity : str
        Apple signing identity or ``-`` for an ad-hoc signature.
    """

    managed_runtime = app / "Contents" / "Resources" / "managed-runtime" / "ort"
    if managed_runtime.is_dir():
        library_signing = [
            "codesign",
            "--force",
            "--sign",
            identity,
            *signing_options(identity),
        ]
        for library in sorted(managed_runtime.glob("*.dylib")):
            run([*library_signing, str(library)])
            run(["codesign", "--verify", "--strict", str(library)])

    common = [
        "codesign",
        "--force",
        "--sign",
        identity,
        *signing_options(identity),
        "--entitlements",
        str(ENTITLEMENTS),
    ]
    run([*common, "--deep", str(app)])

    # The PyInstaller bootloader must be allowed to load libpython, then the
    # outer seal is refreshed. Codex is supplied by the user and is not nested
    # code owned or signed by this App bundle.
    backend = app / "Contents" / "MacOS" / "app-backend"
    run([*common, str(backend)])
    run([*common, str(app)])
    run(["codesign", "--verify", "--deep", "--strict", str(app)])


def sign_dmg(output: Path, identity: str) -> None:
    """Sign and verify one disk image.

    Parameters
    ----------
    output : pathlib.Path
        Created disk image.
    identity : str
        Developer ID identity or ``-`` for an ad-hoc local signature.
    """

    command = ["codesign", "--force", "--sign", identity]
    if identity != "-":
        command.append("--timestamp")
    command.append(str(output))
    run(command)
    run(["codesign", "--verify", "--strict", str(output)])


def sign_and_notarize_dmg(
    output: Path,
    identity: str,
    notary_profile: str,
) -> None:
    """Sign, notarize, staple, and assess a distributable disk image.

    Parameters
    ----------
    output : pathlib.Path
        Created disk image.
    identity : str
        Developer ID Application signing identity.
    notary_profile : str
        Keychain profile previously created for ``notarytool``.
    """

    if identity == "-":
        raise ValueError("notarization requires a Developer ID identity")
    if not notary_profile.strip():
        raise ValueError("notarization requires a notarytool Keychain profile")
    sign_dmg(output, identity)
    run(
        [
            "xcrun",
            "notarytool",
            "submit",
            str(output),
            "--keychain-profile",
            notary_profile,
            "--wait",
        ]
    )
    run(["xcrun", "stapler", "staple", str(output)])
    run(["xcrun", "stapler", "validate", str(output)])
    run(
        [
            "spctl",
            "--assess",
            "--type",
            "open",
            "--context",
            "context:primary-signature",
            "--verbose=2",
            str(output),
        ]
    )


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

    removed_count, removed_bytes = prune_duplicate_resource_runtime(app)
    linked = link_python_metadata_to_resources(app)
    internal_links = verify_internal_bundle_links(app)
    sign_app(app, args.identity)
    smoke_test_backend(app)
    smoke_test_matcha_runtime(app)
    create_dmg(app, output)
    if args.notarize:
        sign_and_notarize_dmg(
            output,
            args.identity,
            args.notary_profile or "",
        )
    else:
        sign_dmg(output, args.identity)
    print(
        f"removed {removed_count} duplicate Python runtime entries "
        f"({removed_bytes / (1024 * 1024):.1f} MiB), linked {len(linked)} "
        "metadata directories, and verified "
        f"{len(internal_links)} internal links"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
