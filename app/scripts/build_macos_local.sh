#!/usr/bin/env bash
#
# Build the XTalk macOS desktop app from a clean checkout.
#
# This script never installs toolchain components. It verifies that every
# required tool is present and aborts with remediation hints when any is
# missing. On success it installs the App's locked Node dependencies and
# builds the local verification installer.
#
# Usage:
#   scripts/build_macos_local.sh [--check-only]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="$(cd "${APP_ROOT}/.." && pwd)"

CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" ]]; then
  CHECK_ONLY=1
fi

failures=()

record_failure() {
  failures+=("$1")
}

report_failures() {
  if [[ "${#failures[@]}" -gt 0 ]]; then
    printf '\nMissing build prerequisites:\n' >&2
    for failure in "${failures[@]}"; do
      printf '  ✗ %s\n' "${failure}" >&2
    done
    printf 'Toolchain components are never installed automatically. Fix the items above and re-run.\n' >&2
    exit 1
  fi
}

note() {
  printf '  ✓ %s\n' "$1"
}

# Platform: the local packaging entrypoint is macOS-only.
if [[ "$(uname -s)" != "Darwin" ]]; then
  printf '✗ XTalk local packaging currently supports macOS only (found %s).\n' "$(uname -s)" >&2
  exit 1
fi

# Repository layout: app/, src/, and frontend/ must be siblings.
for path in "${APP_ROOT}" "${REPOSITORY_ROOT}/src" "${REPOSITORY_ROOT}/frontend"; do
  if [[ ! -d "${path}" ]]; then
    record_failure "repository checkout is incomplete: missing ${path}"
  fi
done
if [[ "${#failures[@]}" -gt 0 ]]; then
  report_failures
fi
note "repository layout (app/, src/, frontend/)"

# Node.js: package engines declare ^20.19.0 || >=22.12.0.
if command -v node >/dev/null 2>&1; then
  node_version="$(node --version | sed 's/^v//')"
  node_major="$(awk -F. '{print $1}' <<<"${node_version}")"
  node_minor="$(awk -F. '{print $2}' <<<"${node_version}")"
  if [[ "${node_major}" -eq 20 && "${node_minor}" -lt 19 ]]; then
    record_failure "Node.js 20.19+ required (found ${node_version})."
  elif [[ "${node_major}" -eq 22 && "${node_minor}" -lt 12 ]]; then
    record_failure "Node.js 22.12+ required (found ${node_version})."
  elif [[ "${node_major}" -lt 20 || "${node_major}" -eq 21 ]]; then
    record_failure "Node.js 20.19+ or 22.12+ required (found ${node_version})."
  else
    note "node ${node_version}"
  fi
else
  record_failure "'node' was not found. Install Node.js 20.19+ or 22.12+ (e.g. https://nodejs.org)."
fi

# npm: the lockfile needs a modern npm; the repository pins npm@11.12.1.
if command -v npm >/dev/null 2>&1; then
  npm_version="$(npm --version)"
  npm_major="${npm_version%%.*}"
  if [[ "${npm_major}" -lt 9 ]]; then
    record_failure "npm 9+ required (found ${npm_version}). Upgrade with: npm install --global npm@11.12.1"
  else
    note "npm ${npm_version}"
  fi
else
  record_failure "'npm' was not found. Install it with: npm install --global npm@11.12.1"
fi

# Python: the source build invokes `python3` directly and requires 3.10+.
if command -v python3 >/dev/null 2>&1; then
  python_version="$(python3 --version 2>&1 | sed 's/^Python //')"
  python_major="$(awk -F. '{print $1}' <<<"${python_version}")"
  python_minor="$(awk -F. '{print $2}' <<<"${python_version}")"
  if [[ "${python_major}" -lt 3 || ( "${python_major}" -eq 3 && "${python_minor}" -lt 10 ) ]]; then
    record_failure "Python 3.10+ required but 'python3' is ${python_version}. Install Python 3.10+ (3.12 preferred) and put it first on PATH."
  else
    note "python3 ${python_version}"
  fi
else
  for candidate in python3.12 python3.13 python3.11 python3.10; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      record_failure "'python3' was not found but '${candidate}' is available. Make 'python3' resolve to a 3.10+ interpreter (adjust PATH) and re-run."
      break
    fi
  done
  if [[ "${#failures[@]}" -eq 0 ]]; then
    record_failure "'python3' was not found. Install Python 3.10+ and ensure it is on PATH."
  fi
fi

# Rust: required by Tauri bundling and the native model runtime builds.
for tool in cargo rustc; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    record_failure "'${tool}' was not found. Install the Rust toolchain (https://rustup.rs)."
  fi
done
if command -v rustc >/dev/null 2>&1; then
  note "rust $(rustc --version)"
fi

# Xcode Command Line Tools.
if ! xcode-select -p >/dev/null 2>&1; then
  record_failure "Xcode Command Line Tools are missing. Run: xcode-select --install"
else
  note "xcode command line tools"
fi

# Metal toolchain is required for the Apple Silicon MLX runtime build.
if [[ "$(uname -m)" == "arm64" ]]; then
  if ! xcrun --find metal >/dev/null 2>&1; then
    record_failure "The Metal toolchain is required on Apple Silicon. Install it with: xcodebuild -downloadComponent MetalToolchain"
  else
    note "metal toolchain"
  fi
fi

report_failures

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  printf '\nAll build prerequisites are present. Toolchain components are intentionally not installed by this script.\n'
  exit 0
fi

cd "${APP_ROOT}"
printf '\nInstalling locked App dependencies (npm ci)...\n'
npm ci
printf '\nBuilding the local verification installer...\n'
npm run package:macos:local

printf '\nBuild complete. The App is at %s\n' \
  "${APP_ROOT}/src-tauri/target/release/bundle/macos/XTalk.app"
