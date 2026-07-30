"""Developer-installable local file browser tool."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool


DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_CHARACTERS = 20_000
MAX_ENTRIES = 1_000
MAX_CHARACTERS = 100_000


def _bounded_limit(value: int, maximum: int) -> int:
    """Clamp one requested output limit to a usable range."""

    return max(1, min(value, maximum))


def _entry_kind(path: Path) -> str:
    """Return a compact display kind for one directory entry."""

    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _browse_directory(path: Path, max_entries: int) -> str:
    """Return a sorted listing for one directory."""

    entries = sorted(
        path.iterdir(),
        key=lambda entry: (
            not entry.is_dir(),
            entry.name.casefold(),
        ),
    )
    visible_entries = entries[:max_entries]
    lines = [f"Directory: {path}"]
    lines.extend(
        f"{_entry_kind(entry)}\t{entry.name}" for entry in visible_entries
    )
    if not visible_entries:
        lines.append("(empty directory)")
    if len(entries) > len(visible_entries):
        lines.append(
            f"... {len(entries) - len(visible_entries)} additional entries omitted"
        )
    return "\n".join(lines)


def _browse_file(path: Path, max_characters: int) -> str:
    """Return a UTF-8 text preview for one local file."""

    content = path.read_text(encoding="utf-8", errors="replace")
    visible_content = content[:max_characters]
    lines = [f"File: {path}", visible_content]
    if len(content) > len(visible_content):
        lines.append(
            f"... {len(content) - len(visible_content)} additional characters omitted"
        )
    return "\n".join(lines)


@tool
def browse_local_files(
    path: str,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
) -> str:
    """Browse an unrestricted local path.

    For a directory, return a sorted list of its immediate children. For a
    regular file, return a UTF-8 text preview. Use an absolute path when the
    user supplies one; otherwise the path is resolved from the sidecar working
    directory. This tool does not recursively traverse directories.
    """

    resolved_path = Path(path).expanduser().resolve()
    if resolved_path.is_dir():
        return _browse_directory(
            resolved_path,
            _bounded_limit(max_entries, MAX_ENTRIES),
        )
    if resolved_path.is_file():
        return _browse_file(
            resolved_path,
            _bounded_limit(max_characters, MAX_CHARACTERS),
        )
    raise FileNotFoundError(f"local path does not exist: {resolved_path}")


def create_tools() -> list[BaseTool]:
    """Create the tools exported by this directory.

    Returns
    -------
    list[langchain_core.tools.BaseTool]
        LangChain tools registered with the configured Agent.
    """

    return [browse_local_files]
